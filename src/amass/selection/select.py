"""topb_select — HOT Triton static top-b page selection (kv-union group-max).

The r8 score is already group-maxed across the GQA group in ``r8_score`` (the
kv-union), so selection is a plain per-(request, kv-head) top-b over the
SELECTABLE region [sink_pages, n_sel_hi), plus always-keeping the sink pages,
the recent window and (subsumed by the window) the partial tail.  STATIC budget:
``b`` is the same for all heads and all layers of a step (derived once from the
per-step page count + config budget); there is NO adaptive per-head coverage.

Reuses the rank-threshold / union-pack structure of
``vllm/kernels/select_pages.py`` (``_rank_threshold_kernel_b`` MODE 0 +
``_union_pack_kernel_b``), collapsed into ONE kernel per (req, kv) because the
group union is already folded into the score.  Grid (n_req, n_kv); fixed launch
shapes, no host sync, no allocation -> full-CUDA-graph safe.
"""
from __future__ import annotations

import triton
import triton.language as tl

from .build import r8_build_refresh  # noqa: F401  (re-exported convenience)
from .score import r8_score
from .state import R8State


# --------------------------------------------------------------------------- #
# derive_page_params: per-request (n_pages, n_sel_hi, static b) once per step. #
# ceil(budget * n_selectable) in fp64 for host-parity; clamped to [1, S].      #
# --------------------------------------------------------------------------- #
@triton.jit
def _derive_params_kernel(sl_ptr, budget_ptr, npg_ptr, nsh_ptr, b_ptr,
                          n_reqs, PAGE: tl.constexpr, WIN: tl.constexpr,
                          SINK: tl.constexpr, R_PAD: tl.constexpr):
    offs = tl.arange(0, R_PAD)
    mask = offs < n_reqs
    sl = tl.load(sl_ptr + offs, mask=mask, other=0)
    npg = (sl + (PAGE - 1)) // PAGE
    nsh = npg - WIN
    s = nsh - SINK                               # n_selectable
    budget = tl.load(budget_ptr)                 # fp64
    bf = tl.math.ceil(s.to(tl.float64) * budget)
    b = tl.minimum(tl.maximum(bf.to(tl.int32), 1), tl.maximum(s, 1))
    tl.store(npg_ptr + offs, npg, mask=mask)
    tl.store(nsh_ptr + offs, nsh, mask=mask)
    tl.store(b_ptr + offs, b, mask=mask)


def derive_page_params(st: R8State, seq_lens, n_req: int) -> None:
    """Refresh st.{n_pages, n_sel_hi, b_fix} for this decode step (one launch)."""
    assert n_req <= st.max_reqs
    _derive_params_kernel[(1,)](
        seq_lens, st.budget64, st.n_pages, st.n_sel_hi, st.b_fix, n_req,
        PAGE=st.page, WIN=st.window_pages, SINK=st.sink_pages,
        R_PAD=triton.next_power_of_2(max(st.max_reqs, 2)), num_warps=1)


# --------------------------------------------------------------------------- #
# topb_select: static top-b on the group-maxed score, kv-union already folded. #
# --------------------------------------------------------------------------- #
@triton.jit
def _topb_select_kernel(score_ptr, npg_ptr, nsh_ptr, b_ptr,
                        tab_ptr, cnt_ptr, n_sink,
                        stride_sr, stride_sh, mp,
                        stride_tabr, stride_tabh, stride_cntr,
                        P_PAD: tl.constexpr, S_PAD: tl.constexpr):
    r = tl.program_id(0)
    kh = tl.program_id(1)
    n_pages = tl.load(npg_ptr + r)               # 0 for padded request rows
    n_sel_hi = tl.load(nsh_ptr + r)
    b = tl.load(b_ptr + r)
    base = r * stride_sr + kh * stride_sh
    offs_p = tl.arange(0, P_PAD)
    pmask = offs_p < n_pages
    score = tl.load(score_ptr + base + offs_p, mask=pmask, other=float("-inf"))

    selmask = (offs_p >= n_sink) & (offs_p < n_sel_hi)
    keep_all = (n_sel_hi - n_sink) <= 1          # too short: attend all (exact)

    # rank threshold over the SELECTABLE slice only.
    offs_s = tl.arange(0, S_PAD)
    smask = offs_s < (n_sel_hi - n_sink)
    sv = tl.load(score_ptr + base + n_sink + offs_s, mask=smask,
                 other=float("-inf"))
    srt = tl.sort(sv, descending=True)
    keepvec = offs_s < b                         # static b (per request)
    tau = tl.min(tl.where(keepvec, srt, float("inf")), axis=0)

    always = (offs_p < n_sink) | (offs_p >= n_sel_hi)   # sinks + window + tail
    keep = ((selmask & (score >= tau)) | always | keep_all) & pmask

    # in-kernel cumsum compaction -> ascending page ids, -1 padded.
    ki = keep.to(tl.int32)
    pos = tl.cumsum(ki, axis=0)                  # 1-based rank at kept pages
    cnt = tl.sum(ki, axis=0)
    tl.store(cnt_ptr + r * stride_cntr + kh, cnt)
    tabbase = tab_ptr + r * stride_tabr + kh * stride_tabh
    for c0 in range(0, mp, P_PAD):               # -1 pads: [cnt, mp)
        offs_c = c0 + offs_p
        tl.store(tabbase + offs_c, tl.full([P_PAD], -1, tl.int32),
                 mask=(offs_c >= cnt) & (offs_c < mp))
    tl.store(tabbase + (pos - 1), offs_p.to(tl.int32), mask=keep)


def topb_select(st: R8State, n_req: int) -> None:
    """HOT Stage-A.2: write st.page_table (R, n_kv, MP) / st.page_cnt (R, n_kv)
    from st.score.  Static budget from st.b_fix (derive_page_params)."""
    n_kv, MP = st.n_kv, st.max_pages
    P_PAD = triton.next_power_of_2(MP)
    S_PAD = triton.next_power_of_2(
        max(MP - st.window_pages - st.sink_pages, 2))
    _topb_select_kernel[(n_req, n_kv)](
        st.score, st.n_pages, st.n_sel_hi, st.b_fix,
        st.page_table, st.page_cnt, st.sink_pages,
        st.score.stride(0), st.score.stride(1), MP,
        st.page_table.stride(0), st.page_table.stride(1), st.page_cnt.stride(0),
        P_PAD=P_PAD, S_PAD=S_PAD, num_warps=4)


def select_pages_r8(st: R8State, layer: int, q, block_table, seq_lens,
                    n_req: int, scale: float, *, derive: bool = True) -> None:
    """One-call Stage-A entry: (derive) + r8_score + topb_select.

    ``derive`` refreshes the per-request page params from seq_lens; set False
    when the metadata builder already called ``derive_page_params`` this step
    (it is per-step, not per-layer).  Writes st.page_table / st.page_cnt.
    """
    if derive:
        derive_page_params(st, seq_lens, n_req)
    r8_score(st, layer, q, block_table, seq_lens, n_req, scale)
    topb_select(st, n_req)
