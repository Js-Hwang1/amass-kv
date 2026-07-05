"""topb_select_cuda — hand-CUDA (Hopper sm_90a) static top-b page selector.

Bitwise-identical drop-in for the golden Triton ``topb_select`` in
``select.py``: one CTA per (request, kv-head), STATIC per-request budget ``b``
(from ``st.b_fix``, identical across heads/layers), select the top-b pages by
``st.score (R, n_kv, MP) fp32`` over the SELECTABLE region
``[sink_pages, n_sel_hi)``, ALWAYS-keep sinks + recent window + partial tail,
and write ``st.page_table (R, n_kv, MP) int32`` (ascending-compacted, -1 padded)
+ ``st.page_cnt (R, n_kv) int32``.  The GQA group union is already folded into
the score by ``r8_score`` (group-max), so this is one top-b per (req, kv).

WHY radix-select (not sort).  Selection only needs ONE order statistic --
``tau`` = the b-th LARGEST selectable score -- yet the previous kernel ran a full
O(n log^2 n) in-smem BITONIC SORT of all selectable pages to extract it (72us at
bs1/64K = a third of the decode step; ~6% SM occupancy at bs1).  A sort is
wasted work.  This kernel finds ``tau`` in O(n) with a 4-pass MSD radix-select
over the fp32 scores reinterpreted as order-preserving uint32 (``f2u``): each
pass histograms one 8-bit digit (256 shared bins) of the elements whose already
fixed high bits match, scans the buckets high->low to locate the one holding
rank ``b``, fixes that digit, and narrows.  After 4 digits every bit is fixed and
the 32-bit prefix IS the order-uint of ``tau`` (``u2f`` back to fp32).  The final
``score >= tau`` keep test is a PURE FLOAT compare, byte-identical to Triton's
``score >= tl.sort(sv,desc)[b-1]``: radix returns the exact b-th-largest VALUE
(ties at ``tau`` all kept, order-statistic exact under duplicates), so the kept
SET and its ascending compaction are bytewise the reference's.

Semantics matched to the reference kernel (``select.py:_topb_select_kernel``),
EXACTLY:
  * tau = the b-th LARGEST score over the selectable slice.  Radix-select finds
    the value ``v`` with ``count(score>v) < b <= count(score>=v)`` == srt(desc)
    [b-1], the same value ``tl.sort`` yields.  ``b`` is clamped to ``[1,n_sel]``;
    the -inf pad Triton sorts to the tail is never the b-th largest (b<=n_sel) so
    it is simply not scanned here.
  * keep(page) = ( selectable & score >= tau ) | always | keep_all, with
    always = (p < sink) | (p >= n_sel_hi)  and  keep_all = (n_sel_hi-sink) <= 1.
    A THRESHOLD rule: exact ties AT tau are all kept (kept count can exceed b),
    identical to Triton -- there is NO page-index tie-break in the *selection*.
  * compaction: ascending page-id order via an in-smem inclusive prefix sum,
    ids in [0, cnt), -1 in [cnt, mp) -- bytewise identical to the Triton
    ``cumsum`` scatter.

Static-``b`` specialisation: ``b`` is a single per-request scalar (block-uniform,
also uniform across the kv-head grid dim) -> zero warp divergence in the
histogram/keep loops; the radix passes are ``b``-independent, ``b`` only steers
the bucket scan.  Loop bounds / smem sizes come from ``P_PAD`` (a stable launch
constant for a given state), so the launch config is fixed.

Fixed launch shapes (grid = (n_req, n_kv), block = 512 by default, override with
AMASS_TOPB_BLK), host-sync-free, allocation-free -> FULL CUDA-graph safe (verified
by a capture/replay test in ``scratch_topb/``).  Self-contained ``load_inline``
build (sm_90a, hash-cached).
"""
from __future__ import annotations

import os

import triton  # only for next_power_of_2 (identical P_PAD to select.py)
import torch

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <math_constants.h>

// fp32 -> order-preserving uint32 and back (radix key).  Monotone in float
// order for all finite values and +-inf; +0/-0 map to distinct keys but the
// downstream keep test is a pure float compare so the kept set is unaffected.
__device__ __forceinline__ unsigned f2u(float f) {
    unsigned u = __float_as_uint(f);
    unsigned mask = (unsigned)(-(int)(u >> 31)) | 0x80000000u;
    return u ^ mask;
}
__device__ __forceinline__ float u2f(unsigned x) {
    unsigned mask = ((x >> 31) - 1u) | 0x80000000u;
    return __uint_as_float(x ^ mask);
}

// One CTA per (request, kv-head).  blockIdx.x = request, blockIdx.y = kv-head.
// Dynamic smem carves: sc[P_PAD] fp32 (page-score cache: ONE global read shared
// by radix + keep) | hist[256] i32 (radix bins) | kbuf[P_PAD] i32 (keep flags,
// scanned in-place into the compaction rank).  All loop bounds strided over
// blockDim so any (P_PAD, BLK) is correct.  Static smem holds the small scan
// scratch (thread totals + per-warp offsets + the radix bucket/rank broadcast).
#define MAXBLK 1024
__global__ void topb_select_kernel(
    const float* __restrict__ score,     // (R, n_kv, MP)  fp32
    const int*   __restrict__ npg,       // (R,)  n_pages
    const int*   __restrict__ nsh,       // (R,)  n_sel_hi
    const int*   __restrict__ bfix,      // (R,)  static b
    int*         __restrict__ page_table,// (R, n_kv, MP)  int32
    int*         __restrict__ page_cnt,  // (R, n_kv)      int32
    const int  n_sink,
    const long s_sr, const long s_sh,    // score strides (row, head)
    const int  mp,
    const long t_sr, const long t_sh,    // page_table strides (row, head)
    const long c_sr,                     // page_cnt row stride
    const int  P_PAD, const int prof_mode)
{
    const int r   = blockIdx.x;
    const int kh  = blockIdx.y;
    const int tid = threadIdx.x;
    const int BLK = blockDim.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    const int nwarp = (BLK + 31) >> 5;

    const int n_pages  = npg[r];
    const int n_sel_hi = nsh[r];
    int       b        = bfix[r];
    const int n_sel    = n_sel_hi - n_sink;      // selectable page count
    const bool keep_all = (n_sel <= 1);          // too short -> attend all

    const long sbase = (long)r * s_sr + (long)kh * s_sh;
    const long tbase = (long)r * t_sr + (long)kh * t_sh;

    extern __shared__ unsigned char smem[];
    float* sc   = reinterpret_cast<float*>(smem);            // P_PAD
    int*   hist = reinterpret_cast<int*>(sc + P_PAD);        // 256
    int*   kbuf = hist + 256;                                // P_PAD
    __shared__ int   thtot[MAXBLK];                          // per-thread chunk sum
    __shared__ int   warp_ex[32];                            // per-warp offsets
    __shared__ int   s_sel, s_k;                             // radix bucket + rank

    // ---- cache all page scores into smem (one coalesced global read) ----
    for (int i = tid; i < P_PAD; i += BLK)
        sc[i] = (i < n_pages) ? score[sbase + i] : -CUDART_INF_F;
    __syncthreads();

    // ---- tau = b-th largest selectable score via 4-pass MSD radix-select ----
    // Order-uint keys; ``prefix`` accumulates fixed high bits, each pass
    // histograms the next 8-bit digit of the keys still matching ``prefix`` and
    // one warp scans buckets HIGH->low (via shuffle) to the one holding rank k.
    float tau = -CUDART_INF_F;
    if (!keep_all && prof_mode != 2) {   // prof_mode 2 = compaction-only (tau=-inf)
        if (b < 1)     b = 1;
        if (b > n_sel) b = n_sel;
        unsigned prefix = 0u;    // fixed high bits of tau's key
        unsigned kmask  = 0u;    // which high bits are fixed
        int      k      = b;     // rank sought within the matching set
        // NOTE (deep-opt audit): a warp-aggregated histogram (__match_any +
        // leader-only atomicAdd) was tried against digit clustering and
        // MEASURED SLOWER (radix 7.0->8.5us bs1/16K, 8.6->12.3 bs16/64K):
        // Hopper's smem atomics absorb the hot-bin bursts cheaper than the
        // ballot/match/ffs/popc overhead per element.  Plain atomics stay.
        #pragma unroll
        for (int digit = 0; digit < 4; ++digit) {
            const int shift = 24 - 8 * digit;
            for (int i = tid; i < 256; i += BLK) hist[i] = 0;
            __syncthreads();
            for (int i = tid; i < n_sel; i += BLK) {
                unsigned u = f2u(sc[n_sink + i]);
                if ((u & kmask) == prefix)
                    atomicAdd(&hist[(u >> shift) & 0xFF], 1);
            }
            __syncthreads();
            // warp 0 locates the crossing bucket: each lane owns 8 bins, a
            // shuffle suffix-scan finds the segment holding rank k, then the
            // crossing lane linearly scans its 8 bins (all warp-synchronous).
            if (wid == 0) {
                int seg = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) seg += hist[lane * 8 + j];
                int suf = seg;                       // inclusive suffix over lanes
                #pragma unroll
                for (int d = 1; d < 32; d <<= 1) {
                    int up = __shfl_down_sync(0xffffffffu, suf, d);
                    if (lane + d < 32) suf += up;
                }
                int above = suf - seg;               // elements in higher lanes
                bool cross = (above < k) && (k <= suf);
                unsigned bal = __ballot_sync(0xffffffffu, cross);
                int cl = __ffs(bal) - 1;             // crossing lane
                if (lane == cl) {
                    int acc = above, sel = cl * 8;
                    #pragma unroll
                    for (int d = cl * 8 + 7; d >= cl * 8; --d) {
                        int c = hist[d];
                        if (acc + c >= k) { sel = d; break; }
                        acc += c;
                    }
                    s_sel = sel;
                    s_k   = k - acc;                 // residual rank in the bucket
                }
            }
            __syncthreads();
            prefix |= ((unsigned)s_sel) << shift;
            kmask  |= 0xFFu << shift;
            k       = s_k;
            __syncthreads();
        }
        tau = u2f(prefix);                                // b-th largest value
    }
    if (prof_mode == 1) {                // prof: radix-only, skip keep/compact
        if (tid == 0) page_cnt[(long)r * c_sr + kh] = (int)(tau > -CUDART_INF_F);
        return;
    }

    // ---- keep flag per page into kbuf (reads the cached score) ----
    for (int i = tid; i < P_PAD; i += BLK) {
        int keep = 0;
        if (i < n_pages) {
            bool always = (i < n_sink) || (i >= n_sel_hi);   // sinks+window+tail
            keep = (always || keep_all || sc[i] >= tau) ? 1 : 0;
        }
        kbuf[i] = keep;
    }
    __syncthreads();

    // ---- inclusive prefix sum of keep flags -> compaction rank, in place ----
    // Blocked 2-phase scan (each thread owns a contiguous ELS chunk): O(P_PAD)
    // work in ~2 passes + a single BLK-wide block scan, vs the log(P_PAD)-pass
    // Hillis-Steele.  Integer adds -> the incl[] (hence page_table) is bytewise
    // the reference's regardless of scan order.
    const int ELS  = (P_PAD + BLK - 1) / BLK;
    const int base = tid * ELS;
    int ttot = 0;
    #pragma unroll 4
    for (int j = 0; j < ELS; ++j) { int idx = base + j; if (idx < P_PAD) ttot += kbuf[idx]; }
    // block exclusive scan of the per-thread totals (warp scan + warp offsets)
    int incl_w = ttot;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        int up = __shfl_up_sync(0xffffffffu, incl_w, d);
        if (lane >= d) incl_w += up;
    }
    if (lane == 31) warp_ex[wid] = incl_w;   // per-warp total
    __syncthreads();
    if (wid == 0) {
        int w  = (tid < nwarp) ? warp_ex[tid] : 0;
        int wi = w;
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            int up = __shfl_up_sync(0xffffffffu, wi, d);
            if (lane >= d) wi += up;
        }
        if (tid < nwarp) warp_ex[tid] = wi - w;   // exclusive per-warp offset
    }
    __syncthreads();
    int off = warp_ex[wid] + (incl_w - ttot);     // exclusive prefix for this thread
    #pragma unroll 4
    for (int j = 0; j < ELS; ++j) {
        int idx = base + j;
        if (idx < P_PAD) { off += kbuf[idx]; kbuf[idx] = off; }  // inclusive rank
    }
    __syncthreads();
    const int cnt = kbuf[P_PAD - 1];             // total kept pages

    if (tid == 0) page_cnt[(long)r * c_sr + kh] = cnt;

    // ---- -1 pad [cnt, mp) then ascending scatter of kept ids into [0, cnt) ----
    for (int i = cnt + tid; i < mp; i += BLK)
        page_table[tbase + i] = -1;
    for (int i = tid; i < n_pages; i += BLK) {
        int prev = (i > 0) ? kbuf[i - 1] : 0;
        if (kbuf[i] - prev == 1)                 // this page is kept
            page_table[tbase + (kbuf[i] - 1)] = i;
    }
}

void topb_select_cuda(torch::Tensor score, torch::Tensor npg,
                      torch::Tensor nsh, torch::Tensor bfix,
                      torch::Tensor page_table, torch::Tensor page_cnt,
                      int64_t n_req, int64_t n_kv, int64_t n_sink,
                      int64_t mp, int64_t P_PAD, int64_t blk,
                      int64_t prof_mode) {
    TORCH_CHECK(score.scalar_type() == at::kFloat, "score must be fp32");
    TORCH_CHECK(page_table.scalar_type() == at::kInt, "page_table int32");
    TORCH_CHECK(page_cnt.scalar_type() == at::kInt, "page_cnt int32");
    TORCH_CHECK(npg.scalar_type() == at::kInt && nsh.scalar_type() == at::kInt
                && bfix.scalar_type() == at::kInt, "params int32");
    const int BLK = (int)blk;
    // sc[P_PAD] fp32 + hist[256] i32 + kbuf[P_PAD] i32  (thtot/warp_ex are static)
    const size_t smem = 256ul * sizeof(int)
                        + (size_t)P_PAD * sizeof(float)
                        + (size_t)P_PAD * sizeof(int);
    // opt in to >48KB dynamic smem once (host-side attr set; NOT a stream op,
    // so it is never captured into a CUDA graph -- safe under capture).
    static int recorded_max = 0;
    if ((int)smem > recorded_max) {
        cudaFuncSetAttribute(topb_select_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        recorded_max = (int)smem;
    }
    dim3 grid((unsigned)n_req, (unsigned)n_kv);
    auto stream = at::cuda::getCurrentCUDAStream();
    topb_select_kernel<<<grid, BLK, smem, stream>>>(
        score.data_ptr<float>(), npg.data_ptr<int>(), nsh.data_ptr<int>(),
        bfix.data_ptr<int>(), page_table.data_ptr<int>(),
        page_cnt.data_ptr<int>(), (int)n_sink,
        score.stride(0), score.stride(1), (int)mp,
        page_table.stride(0), page_table.stride(1), page_cnt.stride(0),
        (int)P_PAD, (int)prof_mode);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topb_select_cuda", &topb_select_cuda,
          "static top-b page selection radix-select (Hopper sm_90a)");
}
"""

_MOD = None
_TRIED = False


def _get():
    """Build (once, hash-cached) and return the extension."""
    global _MOD, _TRIED
    if _TRIED:
        return _MOD
    _TRIED = True
    from torch.utils.cpp_extension import load_inline
    _MOD = load_inline(
        name="amass_topb_select_cuda",
        cpp_sources="",
        cuda_sources=_CUDA_SRC,
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        verbose=False)
    print("[topb_select_cuda] hand-CUDA radix-select top-b ACTIVE", flush=True)
    return _MOD


def topb_select_cuda(st, n_req: int) -> None:
    """HOT Stage-A.2 (hand-CUDA): write st.page_table / st.page_cnt from
    st.score using the static per-request budget st.b_fix.  Bitwise drop-in for
    ``amass.selection.topb_select`` (identical selected sets, compaction order,
    page_cnt).  Requires ``derive_page_params`` already run this step.
    """
    mod = _get()
    n_kv, MP = st.n_kv, st.max_pages
    P_PAD = triton.next_power_of_2(MP)
    # Adaptive block: long context (big P_PAD) wins with more threads (shorter
    # per-thread compaction chunk); short context prefers 512 (less scan
    # overhead).  Env override always wins.  Benched on H200: 512@16K, 1024@64K.
    env = os.environ.get("AMASS_TOPB_BLK")
    blk = int(env) if env else (1024 if P_PAD > 1024 else 512)
    blk = max(32, min(1024, (blk // 32) * 32))           # 32..1024, multiple of 32
    prof = int(os.environ.get("AMASS_TOPB_PROF", "0"))   # 0 full,1 radix,2 compact
    mod.topb_select_cuda(
        st.score, st.n_pages, st.n_sel_hi, st.b_fix,
        st.page_table, st.page_cnt,
        int(n_req), int(n_kv), int(st.sink_pages), int(MP),
        int(P_PAD), blk, prof)
