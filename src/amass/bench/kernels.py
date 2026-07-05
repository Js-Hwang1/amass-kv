"""Per-kernel latency benchmarks for the AMASS hot path (standalone, no engine).

Three hot kernels, each hand-CUDA (Hopper sm_90a) vs its golden Triton reference,
built from synthetic inputs on a real :class:`~amass.selection.state.R8State` (no
vLLM engine required):

  * ``r8_score``  -- CUDA vs Triton, plus the HBM byte roofline (it is
    bandwidth-bound: ~1.25 KB int8 codes / page / kv-head). Reproduces the
    documented ~7x (59 us CUDA vs ~410 us Triton) at bs4/16K/G8.
  * ``topb``      -- CUDA vs Triton static top-b page selection.
  * ``decode``    -- CUDA vs Triton split-K sparse decode, AND vs a *dense*
    FlashAttention baseline (``F.scaled_dot_product_attention`` over the FULL
    context, GQA-expanded) so the sparsity win is quantified against what a dense
    attention step would cost at the same shape.

Every kernel is timed with :mod:`amass.bench.timing` (CUDA events, warmup,
medians) in BOTH eager and CUDA-graph-replay modes, and a numerical
CUDA-vs-Triton agreement check is reported alongside the speedup so a "fast"
result that is silently wrong is caught.

Public entry: :func:`run_kernels` sweeps batch x context x budget and returns a
list of result dicts (also consumable as a table via ``amass.bench.cli``).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from ..selection import (R8State, derive_page_params, r8_build_refresh,
                         r8_score, topb_select)
from ..selection.score_cuda import r8_score_cuda
from ..selection.select_cuda import topb_select_cuda
from ..attention import (ResidentVSource, ensure_stage_b_buffers,
                        sparse_paged_decode_batched)
from ..attention.decode_cuda import sparse_paged_decode_batched_cuda
from . import timing

_DEV = "cuda"
# Stage-B split-K width (matches backend/builder.py:_SPLIT).
_SPLIT = 128


# --------------------------------------------------------------------------- #
# Shared synthetic-input construction                                         #
# --------------------------------------------------------------------------- #
def _r8_state(bs: int, ctx: int, n_kv: int, G: int, *, d: int = 128,
              page: int = 16, r: int = 8, budget: float = 0.1, seed: int = 0):
    """Build an R8State + synthetic paged K / q / block_table / seq_lens, run the
    eigh build + per-step param derive so the score/topb kernels are ready.

    Mirrors ``scratch_r8d/bench_cuda.build_state`` so the numbers reproduce."""
    torch.manual_seed(seed)
    npg = ctx // page
    NB = bs * npg + 8
    MP = npg + 2
    st = R8State(_DEV, num_layers=1, num_blocks=NB, n_kv=n_kv, G=G, head_dim=d,
                 page=page, max_reqs=bs, max_pages=MP, rank=r, budget=budget,
                 sink_pages=1, window_pages=1)
    bt = torch.zeros(bs, MP, dtype=torch.int32, device=_DEV)
    perm = torch.randperm(NB, device=_DEV)[:bs * npg].int()
    for req in range(bs):
        bt[req, :npg] = perm[req * npg:(req + 1) * npg]
    K = torch.randn(NB, page, n_kv, d, device=_DEV, dtype=torch.bfloat16)
    sl = torch.full((bs,), ctx, dtype=torch.int32, device=_DEV)
    q = torch.randn(bs, n_kv * G, d, device=_DEV, dtype=torch.bfloat16)
    r8_build_refresh(st, 0, K, bt, sl, bs)
    derive_page_params(st, sl, bs)
    return st, q, bt, sl, npg, MP


def _r8_score_roofline_bytes(bs: int, npg: int, n_kv: int, G: int, *,
                             d: int = 128, page: int = 16, r: int = 8,
                             win: int = 1) -> float:
    """HBM bytes r8_score must move (== scratch_r8d/bench_cuda.roofline_us)."""
    nsh = npg - win
    tiles = bs * n_kv * nsh
    codes = d * r + d + page * r          # Vk + mu + c (int8)
    scales = (r + page + 1) * 2           # Vk_s + c_s + mu_s (fp16)
    per_tile = codes + scales + 4 + 4     # + block_table read + score write
    qbytes = bs * n_kv * G * d * 2        # q read once per (req, kv)
    return tiles * per_tile + qbytes


class _DecodeInputs:
    """Standalone Stage-B inputs: a paged KV cache + an evenly-spaced selected
    page_table at the target budget (isolates decode from selection), plus a
    dense contiguous K/V for the dense-FA baseline. Mirrors
    ``scratch_r8f/bench_cuda.build``."""

    def __init__(self, bs: int, ctx: int, n_kv: int, G: int, budget: float,
                 *, d: int = 128, page: int = 16, seed: int = 0):
        torch.manual_seed(seed)
        H = n_kv * G
        MP = (ctx + page - 1) // page
        nb = bs * MP + 16
        self.kv_cache = torch.randn(nb, 2, page, n_kv, d, device=_DEV,
                                    dtype=torch.bfloat16) * 0.5
        self.q = torch.randn(bs, H, d, device=_DEV, dtype=torch.bfloat16) * 0.5
        perm = torch.randperm(nb, device=_DEV)
        self.block_table = torch.zeros(bs, MP, device=_DEV, dtype=torch.int32)
        for rr in range(bs):
            self.block_table[rr, :MP] = perm[rr * MP:(rr + 1) * MP].to(torch.int32)
        self.seq_lens = torch.full((bs,), ctx, device=_DEV, dtype=torch.int32)

        # lightweight state object with exactly the fields Stage B reads
        class _St:
            pass
        st = _St()
        st.max_reqs, st.n_kv, st.G, st.D, st.max_pages = bs, n_kv, G, d, MP
        st.page_table = torch.full((bs, n_kv, MP), -1, dtype=torch.int32,
                                   device=_DEV)
        st.page_cnt = torch.zeros((bs, n_kv), dtype=torch.int32, device=_DEV)
        st.scale = d ** -0.5
        ensure_stage_b_buffers(st, _DEV, _SPLIT)

        sink, win = 1, 1
        for rr in range(bs):
            npg = (ctx + page - 1) // page
            S = npg - win - sink
            b = max(1, int(budget * S))
            step = max(1, S // b)
            for kh in range(n_kv):
                keep = set(range(sink)) | set(range(npg - win, npg))
                keep |= set(range(sink, npg - win, step))
                idx = sorted(keep)
                st.page_table[rr, kh, :len(idx)] = torch.tensor(
                    idx, device=_DEV, dtype=torch.int32)
                st.page_cnt[rr, kh] = len(idx)
        self.st = st
        self.vsrc = ResidentVSource(self.kv_cache)
        self.out_t = torch.zeros(bs, H, d, device=_DEV, dtype=torch.bfloat16)
        self.out_c = torch.zeros(bs, H, d, device=_DEV, dtype=torch.bfloat16)
        self.sel_pages = float(st.page_cnt.float().mean().item())
        self.npg = MP
        self.d = d
        # dense-FA baseline tensors: full contiguous K/V (bs, n_kv, ctx, d),
        # GQA-expanded to H at call time by SDPA (enable_gqa) -- the "what a dense
        # attention step costs" reference. Independent tensors (latency ref only).
        self.k_dense = torch.randn(bs, n_kv, ctx, d, device=_DEV,
                                   dtype=torch.bfloat16) * 0.5
        self.v_dense = torch.randn(bs, n_kv, ctx, d, device=_DEV,
                                   dtype=torch.bfloat16) * 0.5
        self.q_dense = self.q.view(bs, H, 1, d)


def _dense_sdpa_call(inp: _DecodeInputs):
    """One dense flash-decode step over the FULL context (the baseline)."""
    scale = inp.d ** -0.5
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION,
                          SDPBackend.EFFICIENT_ATTENTION]):
            return F.scaled_dot_product_attention(
                inp.q_dense, inp.k_dense, inp.v_dense, enable_gqa=True,
                scale=scale)
    except Exception:  # noqa: BLE001 (older torch / no enable_gqa)
        G = inp.q_dense.shape[1] // inp.k_dense.shape[1]
        k = inp.k_dense.repeat_interleave(G, dim=1)
        v = inp.v_dense.repeat_interleave(G, dim=1)
        return F.scaled_dot_product_attention(inp.q_dense, k, v, scale=scale)


# --------------------------------------------------------------------------- #
# Per-kernel benchmarks                                                        #
# --------------------------------------------------------------------------- #
def bench_r8_score(bs: int, ctx: int, *, n_kv: int = 8, G: int = 8,
                   graph: bool = True, seed: int = 0, **kw) -> Dict:
    """r8_score: CUDA vs Triton + HBM roofline + CUDA-vs-Triton agreement."""
    st, q, bt, sl, npg, MP = _r8_state(bs, ctx, n_kv, G, seed=seed)
    scale = 1.0 / math.sqrt(st.d)

    def run_tri():
        r8_score(st, 0, q, bt, sl, bs, scale)

    def run_cuda():
        r8_score_cuda(st, 0, q, bt, sl, bs, scale)

    # numerical agreement over the selectable region
    run_tri()
    ref = st.score.clone()
    st.score.fill_(float("nan"))
    run_cuda()
    nsh = int(st.n_sel_hi[0])
    max_abs = float((st.score[:, :, :nsh] - ref[:, :, :nsh]).abs().max())

    t_tri = timing.bench_median(run_tri, **kw)
    t_cuda = timing.bench_median(run_cuda, **kw)
    g_tri = timing.bench_graph(run_tri, **kw) if graph else None
    g_cuda = timing.bench_graph(run_cuda, **kw) if graph else None

    bytes_moved = _r8_score_roofline_bytes(bs, npg, n_kv, G)
    roof = timing.hbm_roofline_us(bytes_moved)
    return {
        "kernel": "r8_score", "bs": bs, "ctx": ctx, "n_kv": n_kv, "G": G,
        "npg": npg, "triton_us": t_tri.us, "cuda_us": t_cuda.us,
        "speedup": t_tri.us / t_cuda.us,
        "graph_triton_us": g_tri.us if g_tri and g_tri.ok else None,
        "graph_cuda_us": g_cuda.us if g_cuda and g_cuda.ok else None,
        "graph_speedup": (g_tri.us / g_cuda.us
                          if g_tri and g_cuda and g_tri.ok and g_cuda.ok
                          else None),
        "roofline_us": roof, "bytes_mb": bytes_moved / 1e6,
        "cuda_pct_bw": timing.pct_of_peak_bw(t_cuda.us, bytes_moved),
        "cuda_x_roofline": t_cuda.us / roof,
        "max_abs_cuda_vs_triton": max_abs,
    }


def bench_topb(bs: int, ctx: int, *, budget: float = 0.1, n_kv: int = 8,
               G: int = 8, graph: bool = True, seed: int = 0, **kw) -> Dict:
    """topb_select: CUDA vs Triton + bytewise agreement of the selected sets."""
    st, q, bt, sl, npg, MP = _r8_state(bs, ctx, n_kv, G, budget=budget, seed=seed)
    scale = 1.0 / math.sqrt(st.d)
    r8_score(st, 0, q, bt, sl, bs, scale)  # fill st.score once (fixed input)

    def run_tri():
        topb_select(st, bs)

    def run_cuda():
        topb_select_cuda(st, bs)

    # bytewise agreement: Triton table vs CUDA table on the same score
    run_tri()
    tab_t = st.page_table.clone()
    cnt_t = st.page_cnt.clone()
    st.page_table.fill_(-999)
    st.page_cnt.fill_(-999)
    run_cuda()
    tab_eq = bool(torch.equal(st.page_table[:bs, :n_kv], tab_t[:bs, :n_kv]))
    cnt_eq = bool(torch.equal(st.page_cnt[:bs, :n_kv], cnt_t[:bs, :n_kv]))

    t_tri = timing.bench_median(run_tri, **kw)
    t_cuda = timing.bench_median(run_cuda, **kw)
    g_tri = timing.bench_graph(run_tri, **kw) if graph else None
    g_cuda = timing.bench_graph(run_cuda, **kw) if graph else None
    return {
        "kernel": "topb", "bs": bs, "ctx": ctx, "budget": budget, "n_kv": n_kv,
        "G": G, "b_fix": int(st.b_fix[0]),
        "triton_us": t_tri.us, "cuda_us": t_cuda.us,
        "speedup": t_tri.us / t_cuda.us,
        "graph_triton_us": g_tri.us if g_tri and g_tri.ok else None,
        "graph_cuda_us": g_cuda.us if g_cuda and g_cuda.ok else None,
        "graph_speedup": (g_tri.us / g_cuda.us
                          if g_tri and g_cuda and g_tri.ok and g_cuda.ok
                          else None),
        "page_table_byte_eq": tab_eq, "page_cnt_byte_eq": cnt_eq,
    }


def bench_decode(bs: int, ctx: int, *, budget: float = 0.1, n_kv: int = 8,
                 G: int = 8, graph: bool = True, seed: int = 0,
                 dense: bool = True, **kw) -> Dict:
    """sparse decode: CUDA vs Triton (+ agreement) + a dense-FA baseline."""
    inp = _DecodeInputs(bs, ctx, n_kv, G, budget, seed=seed)
    st = inp.st

    def run_tri():
        sparse_paged_decode_batched(inp.q, inp.kv_cache, inp.block_table,
                                    inp.seq_lens, st, inp.out_t, inp.vsrc,
                                    scale=st.scale)

    def run_cuda():
        sparse_paged_decode_batched_cuda(inp.q, inp.kv_cache, inp.block_table,
                                         inp.seq_lens, st, inp.out_c, inp.vsrc,
                                         scale=st.scale)

    run_tri()
    run_cuda()
    max_abs = float((inp.out_c.float() - inp.out_t.float()).abs().max())

    t_tri = timing.bench_median(run_tri, **kw)
    t_cuda = timing.bench_median(run_cuda, **kw)
    g_tri = timing.bench_graph(run_tri, **kw) if graph else None
    g_cuda = timing.bench_graph(run_cuda, **kw) if graph else None

    res = {
        "kernel": "decode", "bs": bs, "ctx": ctx, "budget": budget,
        "n_kv": n_kv, "G": G,
        "sel_pages": inp.sel_pages, "n_pages": inp.npg,
        "sel_pct": 100.0 * inp.sel_pages / inp.npg,
        "triton_us": t_tri.us, "cuda_us": t_cuda.us,
        "speedup": t_tri.us / t_cuda.us,
        "graph_triton_us": g_tri.us if g_tri and g_tri.ok else None,
        "graph_cuda_us": g_cuda.us if g_cuda and g_cuda.ok else None,
        "graph_speedup": (g_tri.us / g_cuda.us
                          if g_tri and g_cuda and g_tri.ok and g_cuda.ok
                          else None),
        "max_abs_cuda_vs_triton": max_abs,
        "dense_us": None, "cuda_vs_dense": None, "dense_backend": None,
    }
    if dense:
        try:
            _dense_sdpa_call(inp)  # warm / trigger backend selection
            t_dense = timing.bench_median(lambda: _dense_sdpa_call(inp), **kw)
            res["dense_us"] = t_dense.us
            res["cuda_vs_dense"] = t_dense.us / t_cuda.us
        except Exception as e:  # noqa: BLE001
            res["dense_err"] = f"{type(e).__name__}: {e}"
    return res


# --------------------------------------------------------------------------- #
# Sweep driver                                                                 #
# --------------------------------------------------------------------------- #
_DEFAULT_BATCHES = (1, 2, 4, 8)
_DEFAULT_CONTEXTS = (4096, 16384, 32768)
_DEFAULT_BUDGETS = (0.02, 0.05, 0.10, 1.0)


def run_kernels(*, which: Sequence[str] = ("r8_score", "topb", "decode"),
                batches: Sequence[int] = _DEFAULT_BATCHES,
                contexts: Sequence[int] = _DEFAULT_CONTEXTS,
                budgets: Sequence[float] = _DEFAULT_BUDGETS,
                n_kv: int = 8, G: int = 8, graph: bool = True,
                dense: bool = True, warmup: int = 25, iters: int = 50,
                reps: int = 7, verbose: bool = True) -> List[Dict]:
    """Sweep the per-kernel benchmarks and return a list of result dicts.

    ``r8_score`` is budget-independent (it scores every selectable page) so it is
    run once per (batch, context); ``topb`` / ``decode`` are run per budget. All
    timings are CUDA-event medians (see :mod:`amass.bench.timing`); ``graph``
    adds a captured-replay column. Public API: ``amass.bench.run_kernels``.
    """
    tk = dict(graph=graph, warmup=warmup, iters=iters, reps=reps)
    results: List[Dict] = []

    def _safe(kernel, fn):
        # One infeasible config (OOM / cusolver batched-eigh limit at extreme
        # synthetic scale) must not abort the sweep: record + continue.
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001
            r = {"kernel": kernel, "error": f"{type(e).__name__}: {e}"}
            if verbose:
                print(f"[{kernel:8s}] SKIPPED: {r['error']}", flush=True)
            _free()
            return
        results.append(r)
        if verbose:
            _print_row(r)

    for ctx in contexts:
        for bs in batches:
            if "r8_score" in which:
                _safe("r8_score",
                      lambda: bench_r8_score(bs, ctx, n_kv=n_kv, G=G, **tk))
            for budget in budgets:
                if "topb" in which:
                    _safe("topb", lambda b=budget: bench_topb(
                        bs, ctx, budget=b, n_kv=n_kv, G=G, **tk))
                if "decode" in which:
                    _safe("decode", lambda b=budget: bench_decode(
                        bs, ctx, budget=b, n_kv=n_kv, G=G, dense=dense, **tk))
            _free()
    return results


def _free() -> None:
    """Release the per-config synthetic tensors between sweep points so a big
    (bs, ctx) does not stack allocations into the next one."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def _print_row(r: Dict) -> None:
    """One-line human summary of a result dict."""
    k = r["kernel"]
    head = f"[{k:8s}] bs={r['bs']} ctx={r['ctx']:>5d}"
    if "budget" in r:
        head += f" budget={r['budget']:<5g}"
    g = (f" | graph {r['graph_speedup']:.2f}x"
         if r.get("graph_speedup") else "")
    if k == "r8_score":
        print(f"{head} G={r['G']} | Triton {r['triton_us']:7.1f}us  "
              f"CUDA {r['cuda_us']:6.1f}us  {r['speedup']:.2f}x  "
              f"(roof {r['roofline_us']:.1f}us, {r['cuda_pct_bw']:.0f}% BW, "
              f"{r['cuda_x_roofline']:.1f}x roof){g}  "
              f"|d|<={r['max_abs_cuda_vs_triton']:.1e}", flush=True)
    elif k == "topb":
        print(f"{head} b={r['b_fix']} | Triton {r['triton_us']:6.1f}us  "
              f"CUDA {r['cuda_us']:6.1f}us  {r['speedup']:.2f}x{g}  "
              f"byte-eq={r['page_table_byte_eq'] and r['page_cnt_byte_eq']}",
              flush=True)
    elif k == "decode":
        dn = (f"  dense {r['dense_us']:6.1f}us ({r['cuda_vs_dense']:.2f}x)"
              if r.get("dense_us") else "")
        print(f"{head} sel={r['sel_pct']:.1f}% | Triton {r['triton_us']:6.1f}us  "
              f"CUDA {r['cuda_us']:6.1f}us  {r['speedup']:.2f}x{g}{dn}  "
              f"|d|<={r['max_abs_cuda_vs_triton']:.1e}", flush=True)
