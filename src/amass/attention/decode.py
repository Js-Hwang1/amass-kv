"""Stage B, launch 1 of 2: the split-K sparse paged decode.

Flash-decode-style online-softmax attention (fp32 m/l/acc) over ONLY the pages
Stage A selected, reading the vLLM paged cache directly (no gather):

  * KV-HEAD-GROUPED: grid (n_req, n_kv, SPLIT). Each program reads a
    (kv-head, page) K/V tile ONCE and processes the whole GQA group's G query
    heads as a register tile via ``tl.dot`` -- one K/V read per group, never per
    query head.
  * The P.V product is an fp16 ``tl.dot`` with fp32 accumulate (tensor cores;
    p in [0,1] is exact-range in fp16, v bf16->fp16 is exact below 65504) -- the
    fp32 IEEE dot on CUDA cores was the 3x-off-roofline bottleneck.
  * Split-K over the selected pages in CONTIGUOUS chunks:
    pps = max(cdiv(cnt, SPLIT), PPS_MIN) is derived on device from page_cnt, so
    small selections use fewer, fatter programs. Inactive splits exit WITHOUT
    storing; the merge kernel (attention/merge.py) recomputes the active count.
    Launch shapes stay static (graph-capturable); every dynamic quantity is a
    device-side value.
  * seq_len masks the trailing partial page (logical page index from page_table
    gives the token positions; block_table gives the physical page).

The V SOURCE is injected (AMASS_DESIGN.md 8.1). K always comes from the resident
paged cache; V is loaded through a compile-time-specialized path selected by the
``V_SRC`` constexpr the ``VSource`` carries:

  * fast  -> ResidentVSource: V_SRC=0, the engine V half of kv_cache.
  * mem-v -> TierVSource (later agent): V_SRC=1, hot buffer | staged | pinned.

So this kernel takes ``(v_ptr, v strides, V_SRC)`` and nothing else about
offload, and the call site (backend/attn.py) only swaps which VSource it builds
-- the tier slots in without touching this file's call sites.
"""
from __future__ import annotations

from typing import Protocol, Tuple

import torch
import triton
import triton.language as tl

from .merge import merge_splits_kernel


# --------------------------------------------------------------------------- #
# V-source seam (AMASS_DESIGN.md 8.1)                                          #
# --------------------------------------------------------------------------- #
class VSource(Protocol):
    """Injected V provider. ``SRC_ID`` is a compile-time source id; ``args()``
    returns the (base pointer tensor, block stride, token stride, head stride)
    the decode kernel loads V through. One decode kernel, a ``V_SRC`` constexpr
    branch per source -- never a python branch per tile."""

    SRC_ID: int

    def args(self) -> Tuple[torch.Tensor, int, int, int]:
        ...


def _split_kv(kv_cache: torch.Tensor):
    """(K, V) views (num_blocks, page, n_kv, d) for either cache layout.

    K-only caches (mem-v/mem-kv: shape (nb, 1, page, n_kv, d), the V half lives
    off-engine) return V=None -- Stage B must then be given a tier VSource."""
    if kv_cache.shape[0] == 2:                       # 0.11: (2, nb, page, kv, d)
        return kv_cache[0], kv_cache[1]
    if kv_cache.shape[1] == 1:                        # K-only engine cache
        return kv_cache.select(1, 0), None
    return kv_cache.select(1, 0), kv_cache.select(1, 1)   # 0.24: (nb, 2, ...)


class ResidentVSource:
    """AMASS-fast V source: the resident engine V half of ``kv_cache``.
    Same paged layout as K, addressed by physical block id (V_SRC=0)."""

    SRC_ID = 0

    def __init__(self, kv_cache: torch.Tensor):
        _, v = _split_kv(kv_cache)
        if v is None:
            raise ValueError(
                "ResidentVSource requires a resident V half; this cache is "
                "K-only (mem variant) -- use a tier VSource instead.")
        self._v = v

    def args(self) -> Tuple[torch.Tensor, int, int, int]:
        v = self._v
        return v, v.stride(0), v.stride(1), v.stride(2)

    def tier_ptrs(self):
        """Placeholder tier-pointer pack for V_SRC=0 (never dereferenced: the
        tiered V load is dead code under the ``V_SRC==0`` constexpr compile).
        Returns valid-memory tensors so the launch signature is satisfied."""
        v = self._v
        # hot, page2slot, vbo, pool -> reuse v (bf16) / a dummy int tensor.
        return v, v, v, v, 0, 0, 0


def ensure_stage_b_buffers(st, device, split: int) -> None:
    """Allocate the split-K partials on the (shared) decode state if absent.

    Stage-B buffer ownership lives in attention/ even though the buffers ride on
    the state object Stage A also uses; call ONCE at builder init (outside any
    graph). Shapes: (R, n_kv, split, G[, D]); ``m_part.stride(0) == n_kv*split*G``
    is the per-request stride the kernels index by, and acc_part shares it (times
    D), so all three must be contiguous with this layout."""
    if getattr(st, "m_part", None) is not None:
        st.split = getattr(st, "split", split)
        return
    R, n_kv, G, D = st.max_reqs, st.n_kv, st.G, st.D
    f32 = torch.float32
    st.split = split
    st.m_part = torch.empty(R, n_kv, split, G, device=device, dtype=f32)
    st.l_part = torch.empty(R, n_kv, split, G, device=device, dtype=f32)
    st.acc_part = torch.empty(R, n_kv, split, G, D, device=device, dtype=f32)


def _pps_min(n_req: int) -> int:
    """Min pages per split program: amortizes the q-load / state-init /
    partial-store fixed cost. Swept on H200: 16 wins when the grid has >= 2
    requests' worth of programs, but starves occupancy at bs=1 (fewer than ~2
    programs/SM), where 4 wins. Graph-safe: n_req is a capture-bucket constant."""
    return 4 if n_req == 1 else 16


# --------------------------------------------------------------------------- #
# Split-K decode kernel                                                        #
# --------------------------------------------------------------------------- #
@triton.jit
def sparse_decode_split_kernel(q_ptr, k_ptr, v_ptr, bt_ptr, tab_ptr,
                               cnt_ptr, sl_ptr, m_ptr, l_ptr, acc_ptr,
                               sm_scale,
                               stride_qt, stride_qh,
                               stride_kb, stride_kt, stride_kh,
                               stride_vb, stride_vt, stride_vh,
                               hotv_ptr, p2s_ptr, vbo_ptr, poolv_ptr,
                               hotk_ptr, poolk_ptr,
                               stride_btr, stride_tabr, stride_tabh,
                               stride_cntr, stride_pr,
                               t_lidx, t_NB, t_S,
                               G: tl.constexpr, G_PAD: tl.constexpr,
                               PAGE: tl.constexpr, D: tl.constexpr,
                               SPLIT: tl.constexpr, PPS_MIN: tl.constexpr,
                               V_SRC: tl.constexpr, K_SRC: tl.constexpr,
                               NKV: tl.constexpr):
    r = tl.program_id(0)
    kh = tl.program_id(1)
    sp = tl.program_id(2)
    cnt = tl.load(cnt_ptr + r * stride_cntr + kh)   # 0 for padded requests
    pps = tl.maximum(tl.cdiv(cnt, SPLIT), PPS_MIN)  # contiguous chunk
    j0 = sp * pps
    if j0 >= cnt:                                    # inactive split: no store
        return
    j1 = tl.minimum(j0 + pps, cnt)
    offs_g = tl.arange(0, G_PAD)
    offs_d = tl.arange(0, D)
    offs_t = tl.arange(0, PAGE)
    gmask = offs_g < G

    seq_len = tl.load(sl_ptr + r)
    q = tl.load(q_ptr + r * stride_qt
                + (kh * G + offs_g)[:, None] * stride_qh + offs_d[None, :],
                mask=gmask[:, None], other=0.0)                   # (G_PAD, D)

    m_i = tl.full([G_PAD], float("-inf"), tl.float32)
    l_i = tl.zeros([G_PAD], tl.float32)
    acc = tl.zeros([G_PAD, D], tl.float32)

    for j in tl.range(j0, j1, num_stages=3):
        pt = tl.load(tab_ptr + r * stride_tabr + kh * stride_tabh + j
                     ).to(tl.int64)                 # logical page index
        blk = tl.load(bt_ptr + r * stride_btr + pt).to(tl.int64)  # physical page
        tok = pt * PAGE + offs_t
        tmask = tok < seq_len
        # ---- KV-SOURCE seam: 3-way page-source select, SHARED by K and V --- #
        # mem-kv (K_SRC=V_SRC=1): both K and V for a selected page live in the
        # tier (hot cache | staged pool | pinned host pool over UVA); the page is
        # in the SAME slot/staging-block for both, so the select is computed ONCE
        # and drives both loads. mem-v (K_SRC=0): K resident, only V from tier.
        if (V_SRC == 1) or (K_SRC == 1):
            complete = ((pt + 1) * PAGE) <= seq_len
            slot = tl.load(p2s_ptr + kh * t_NB + blk)             # page2slot
            vb = tl.load(vbo_ptr + blk)                           # staging blk
            use_hot = complete & (slot >= 0)
            use_stg = (use_hot == 0) & (vb >= 0)
            use_pin = (use_hot == 0) & (vb < 0)
            slot_c = tl.maximum(slot, 0).to(tl.int64)
            vb_c = tl.maximum(vb, 0).to(tl.int64)
            slot_off = (kh * t_S + slot_c) * (PAGE * D)
            pool_off = (((t_lidx * t_NB + blk) * NKV + kh).to(tl.int64)
                        * (PAGE * D))
        # ---- K load ------------------------------------------------------- #
        if K_SRC == 0:                              # resident engine K half
            k = tl.load(k_ptr + blk * stride_kb + kh * stride_kh
                        + offs_t[:, None] * stride_kt + offs_d[None, :],
                        mask=tmask[:, None], other=0.0)           # (PAGE, D)
        else:                                       # TIER K (k_ptr = staged k_pool)
            k_h = tl.load(hotk_ptr + slot_off + offs_t[:, None] * D
                          + offs_d[None, :], mask=tmask[:, None] & use_hot,
                          other=0.0)
            k_s = tl.load(k_ptr + vb_c * stride_kb + kh * stride_kh
                          + offs_t[:, None] * stride_kt + offs_d[None, :],
                          mask=tmask[:, None] & use_stg, other=0.0)
            k_p16 = tl.load(poolk_ptr + pool_off + offs_t[:, None] * D
                            + offs_d[None, :], mask=tmask[:, None] & use_pin,
                            other=0)
            k = tl.where(use_hot, k_h,
                         tl.where(use_stg, k_s, k_p16.to(tl.bfloat16,
                                                         bitcast=True)))
        # ---- V load ------------------------------------------------------- #
        if V_SRC == 0:                              # resident engine V half
            v = tl.load(v_ptr + blk * stride_vb + kh * stride_vh
                        + offs_t[:, None] * stride_vt + offs_d[None, :],
                        mask=tmask[:, None], other=0.0)           # (PAGE, D)
        else:                                       # TIER V (hot | staged | pinned)
            v_h = tl.load(hotv_ptr + slot_off + offs_t[:, None] * D
                          + offs_d[None, :], mask=tmask[:, None] & use_hot,
                          other=0.0)
            v_s = tl.load(v_ptr + vb_c * stride_vb + kh * stride_vh
                          + offs_t[:, None] * stride_vt + offs_d[None, :],
                          mask=tmask[:, None] & use_stg, other=0.0)
            v_p16 = tl.load(poolv_ptr + pool_off + offs_t[:, None] * D
                            + offs_d[None, :], mask=tmask[:, None] & use_pin,
                            other=0)
            v = tl.where(use_hot, v_h,
                         tl.where(use_stg, v_s, v_p16.to(tl.bfloat16,
                                                         bitcast=True)))
        s = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale
        s = tl.where(tmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = tl.dot(p.to(tl.float16), v.to(tl.float16),
                     acc=acc * alpha[:, None])      # fp16 dot, fp32 acc
        m_i = m_new

    base = r * stride_pr + (kh * SPLIT + sp) * G
    tl.store(m_ptr + base + offs_g, m_i, mask=gmask)
    tl.store(l_ptr + base + offs_g, l_i, mask=gmask)
    tl.store(acc_ptr + (base + offs_g)[:, None] * D + offs_d[None, :], acc,
             mask=gmask[:, None])


# --------------------------------------------------------------------------- #
# Wrapper: 2 fixed-shape launches for the whole decode batch                   #
# --------------------------------------------------------------------------- #
def sparse_paged_decode_batched(q: torch.Tensor, kv_cache: torch.Tensor,
                                block_table: torch.Tensor,
                                seq_lens: torch.Tensor, st, out: torch.Tensor,
                                vsource: VSource, *, scale: float = None) -> None:
    """Batched Stage B: 2 fixed-shape launches for the WHOLE decode batch.

    Consumes ``st.page_table`` / ``st.page_cnt`` (from Stage A) and the split-K
    partial buffers ``st.{m_part, l_part, acc_part}``; writes rows [0, n_req) of
    ``out`` (T, H, d) IN PLACE (the engine's persistent output buffer). K is read
    from the resident ``kv_cache``; V through ``vsource`` (the fast/mem seam). No
    allocation, no host sync -> safe inside FULL CUDA graphs.

    ``scale`` defaults to ``st.scale`` (the softmax scale is a model constant the
    builder stamps once). ``n_req`` is ``seq_lens.shape[0]`` (num_reqs_padded).
    """
    if scale is None:
        scale = st.scale
    n_req = seq_lens.shape[0]
    G = st.G
    split = st.split
    pps_min = _pps_min(n_req)

    v_ptr, svb, svt, svh = vsource.args()
    V_SRC = vsource.SRC_ID
    # V-tier pointer pack (dead-code placeholders for V_SRC==0). ``v_ptr`` is the
    # STAGED v_pool base for V_SRC==1 (same paged layout as resident V).
    hot_ptr, p2s_ptr, vbo_ptr, pool_ptr, t_lidx, t_NB, t_S = vsource.tier_ptrs()

    # ---- K source seam: resident kv_cache (K_SRC=0) or the tier (K_SRC=1) --- #
    K_SRC = getattr(vsource, "K_SRC", 0)
    if K_SRC == 1:                                  # mem-kv: K is offloaded too
        k_ptr, skb, skt, skh = vsource.k_args()     # staged k_pool base
        hotk_ptr, poolk_ptr = vsource.k_tier_ptrs()
    else:                                           # resident K half
        k_ptr, _ = _split_kv(kv_cache)
        skb, skt, skh = k_ptr.stride(0), k_ptr.stride(1), k_ptr.stride(2)
        hotk_ptr, poolk_ptr = v_ptr, v_ptr          # dead-code placeholders
    _, page, n_kv, d = k_ptr.shape

    sparse_decode_split_kernel[(n_req, n_kv, split)](
        q, k_ptr, v_ptr, block_table, st.page_table, st.page_cnt, seq_lens,
        st.m_part, st.l_part, st.acc_part, scale,
        q.stride(0), q.stride(1),
        skb, skt, skh,
        svb, svt, svh,
        hot_ptr, p2s_ptr, vbo_ptr, pool_ptr, hotk_ptr, poolk_ptr,
        block_table.stride(0), st.page_table.stride(0), st.page_table.stride(1),
        st.page_cnt.stride(0), st.m_part.stride(0),
        t_lidx, t_NB, t_S,
        G=G, G_PAD=max(16, triton.next_power_of_2(G)), PAGE=page, D=d,
        SPLIT=split, PPS_MIN=pps_min, V_SRC=V_SRC, K_SRC=K_SRC, NKV=n_kv,
        num_warps=4)

    merge_splits_kernel[(n_req, n_kv * G)](
        st.m_part, st.l_part, st.acc_part, st.page_cnt, out,
        out.stride(0), out.stride(1), st.m_part.stride(0), st.page_cnt.stride(0),
        G=G, D=d, SPLIT=split, SPLIT_PAD=triton.next_power_of_2(split),
        PPS_MIN=pps_min)
