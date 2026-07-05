"""r8_score_cuda -- hand-CUDA Hopper (sm_90a) kernel for the r8 page score.

Drop-in for the golden Triton ``r8_score`` (``amass.selection.score``), same
signature/semantics: writes ``st.score`` (R, n_kv, MP) fp32 over the selectable
region [0, n_sel_hi) for the STATIC r8-ranked selector.  Reference math per
(req, kv-head, page), dequantizing the int8 summary in-kernel:

    q~[g,r]  = Vkᵀ q_g                     (reduce over d=128)
    tok[g,t] = (c q~[g]) * scale           (reduce over r=8)
    S[g]     = mu·q_g * scale + logsumexp_t tok[g,t]
    score    = max_g S[g]                  (kv-union group max)

TWO-PASS DESIGN (2026-07-05).  The single fused kernel was register-bound at
25% occupancy (125 regs, forced by the hoisted int8 mma A-fragments) where the
load(17µs)+mma(12µs)+fp32-tail(30µs) pipes ran FULLY SERIAL -> ~60µs, 6.5x off
the 9µs HBM roofline.  The 30µs "tail" is only 128 logsumexps of 16 elements --
trivial compute that was 100% serialization at that occupancy.  The fix
DECOUPLES the two register regimes:

  * **Pass 1 (r8_mma_kernel)** -- register-heavy, does the parts that NEED q:
    (a) ``q~ = Vkᵀq`` on the int8 tensor core (2-limb int8 q -> bf16 precision,
        hoisted A-fragments), and (b) ``mu·q`` (the fp32 d-reduction), both of
        which need the query.  cp.async-streams Vk (1 KB) + mu (128 B)/page.
        Stores the EXACT int q~ (``128*hi+lo``, GMX·r int32/page) + the per-group
        ``qmu`` (mu·q, fp32) + the per-(req,kv) ``qscale`` to scratch.  mu·q
        overlaps the mma (it was measured free in the fused kernel).
  * **Pass 2 (r8_tail_kernel)** -- register-LIGHT (48 regs -> ~62% occupancy),
    reads q~ + qscale + c + mu + the fp16 scales (q staged fp32 with NO amax,
    qscale comes from Pass 1) and does ``c·q~`` + ``mu·q`` + logsumexp +
    ``S = qmu + lse`` + the kv-union group-max.

MEASURED (H200, RULER-16K shapes, nsys per-pass + ptxas occupancy):
  * Correct: gate 28/28 green, rel|Δ| <= 1.25e-4 vs Triton, top-b identical.
    Scratch ``st._r8_qtilde/_qsc`` alloc'd ONCE (lazy, outside any graph) ->
    stable address, full CUDA-graph capture/replay safe.
  * The 2-pass MOVES THE NEEDLE but does NOT reach the HBM roofline.  Pass 1
    (mma) ~25us is register-bound at 25% occupancy AND the int8 mma is
    TC-throughput-bound near its ~12us floor (Vk load ~7us does not fully
    overlap).  Pass 2 (tail) ~33us does NOT fully hide the per-page logsumexp
    dependency chain even at 62% occupancy -- the tail is chain-latency +
    cp.async-ring-prologue bound, not occupancy bound (measured: 25%->62%
    occupancy did not speed it, and it stays ~7x off its load roofline at all
    batches bs4..bs32).  Net r8_score ~60us == the fused kernel, BUT the
    decoupling wins where the fused was most occupancy-starved: bs1 (r8_score
    ~38->25us) and high batch, pushing the whole AMASS attention from 4/45 to
    14/45 grid cells >= 1.0x FA3 (best 1.22x).  Still 0/45 >= 2x at 10%.
  * PATH TO ROOFLINE (not built): fold ``mu·q`` onto the tensor core via a
    16x16 augmented mma ``[Vk|mu]`` (col 8 = mu) -- this also HALVES the A-frag
    registers (16x16 vs 32x8), freeing registers to software-PIPELINE the tail
    behind the next page's mma in ONE fused kernel (the pipeline that failed
    before purely for lack of registers).  Target ~15-17us r8_score.  Note: 2x
    FA3 also needs the decode kernel near roofline (it is ~5x off), so r8_score
    alone is necessary but not sufficient.

Tunables: R8_CUDA_{WARPS,ZSPLIT,NSTAGE,HOIST,MINBLK} tune Pass 1; R8_TAIL_{WARPS,
ZSPLIT,NSTAGE} tune Pass 2 (it wants MORE warps/CTAs -- occupancy- not register-
bound).  R8_PTXAS_V=1 prints ptxas register/occupancy for both kernels.
"""
from __future__ import annotations

import os

import torch

from .state import R8State

# ---- tunables (compile-time) ----------------------------------------------- #
_WARPS = int(os.environ.get("R8_CUDA_WARPS", "4"))       # Pass-1 warps / CTA
_ZSPLIT = int(os.environ.get("R8_CUDA_ZSPLIT", "16"))    # Pass-1 page-splits (grid.z)
_NSTAGE = int(os.environ.get("R8_CUDA_NSTAGE", "3"))     # Pass-1 cp.async ring depth
_MINBLK = int(os.environ.get("R8_CUDA_MINBLK", "0"))     # __launch_bounds__ minblocks
_HOIST = int(os.environ.get("R8_CUDA_HOIST", "2"))       # hoist A-frags: 0/1/2
_TWARPS = int(os.environ.get("R8_TAIL_WARPS", "8"))      # Pass-2 warps / CTA
_TZSPLIT = int(os.environ.get("R8_TAIL_ZSPLIT", "32"))   # Pass-2 page-splits (grid.z)
_TNSTAGE = int(os.environ.get("R8_TAIL_NSTAGE", "4"))    # Pass-2 ring depth

# ---- fused single-kernel path (Piece A: 16x16 augmented [Vk|mu] int8 MMA) --- #
# Fused single-kernel (Piece A).  MEASURED under CUDA-graph replay it does NOT
# beat the tuned 2-pass in int8 (wmma is SYNCHRONOUS -> no per-warp mma/tail
# overlap; the fused footprint caps occupancy ~50% vs the 2-pass tail's ~62%, and
# graph replay already removes the launch overhead fusion would save in eager
# mode).  So the int8 default is the 2-pass; the fused kernel is auto-selected for
# int4 (it is the natural unpack vehicle and halves the Vk HBM load).  R8_FUSE=1
# forces the fused kernel for int8 too (for A/B benchmarking).
_FUSE = int(os.environ.get("R8_FUSE", "0"))              # 0 = 2-pass int8 (fastest)
_FWARPS = int(os.environ.get("R8_FUSE_WARPS", "8"))      # fused warps / CTA
_FZSPLIT = int(os.environ.get("R8_FUSE_ZSPLIT", "32"))   # fused page-splits (grid.z)
_FNSTAGE = int(os.environ.get("R8_FUSE_NSTAGE", "3"))    # fused cp.async ring depth
_FHOIST = int(os.environ.get("R8_FUSE_HOIST", "2"))      # hoist A-frags: 0/1/2
_FMINBLK = int(os.environ.get("R8_FUSE_MINBLK", "0"))    # __launch_bounds__ minblocks
_FPIPE = int(os.environ.get("R8_FUSE_PIPE", "0"))        # 1 = double-buffer MMA/tail
                                                          # (no gain: wmma is sync)

GMX = 8  # max GQA group supported (mirrors the CUDA constant)

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mma.h>
#include <ATen/cuda/CUDAContext.h>

using namespace nvcuda;

#ifndef R8_WARPS
#define R8_WARPS 4
#endif
#ifndef R8_NSTAGE
#define R8_NSTAGE 3
#endif
#ifndef R8_TWARPS
#define R8_TWARPS 8
#endif
#ifndef R8_TNSTAGE
#define R8_TNSTAGE 4
#endif
#ifndef R8_MINBLK
#define R8_MINBLK 0
#endif
#ifndef R8_HOIST
#define R8_HOIST 2
#endif
#if R8_MINBLK > 0
#define R8_LB __launch_bounds__(R8_WARPS * 32, R8_MINBLK)
#else
#define R8_LB __launch_bounds__(R8_WARPS * 32)
#endif

#define DMAX 128
#define PAGE 16
#define RANK 8
#define QLIM 16383       // 2-limb int8 q: |qi| <= 127*128+127
#define MM_M 32          // wmma int8 tile 32x8x16 (N=8 == RANK, natural Vk)
#define MM_N 8
#define MM_K 16
#define KT   (DMAX / MM_K)   // 8 k-tiles over d=128
#define GMX  8           // max GQA group size supported
#define GR   (GMX * RANK)    // 64 int32 q~ stored per page (padded to 8 groups)
#define QFPAD 4          // qf smem row pad -> conflict-free mu·q reads
#define COMBW 12         // comb smem row stride (pad from 8) -> conflict-free

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
__device__ __forceinline__ int _sext4(int nib) {    // 4-bit two's-complement
    return (nib & 0x8) ? (nib - 16) : nib;
}

// ======================================================================== //
// PASS 1: q~ = Vkᵀq (2-limb int8 TC) + qmu = mu·q -> scratch. Register-heavy //
// (hoisted A-frags); cp.async ring streams Vk (1 KB) + mu (128 B)/page.      //
// ======================================================================== //
__global__ void R8_LB
r8_mma_kernel(
    const __nv_bfloat16* __restrict__ q,     // (T,H,d)
    const int8_t*  __restrict__ vk,          // (NB,n_kv,d,r)
    const int*     __restrict__ bt,          // (n_req, bt_stride)
    const int*     __restrict__ nsh,         // (R,)
    int*           __restrict__ qtilde,      // (R,n_kv,MP,GR) int32
    float*         __restrict__ qsc_o,       // (R,n_kv,GMX) fp32
    const int n_kv, const int G,
    const long q_st, const long q_hs,
    const long vk_sb, const long vk_ks,
    const int bt_stride,
    const long qt_sr, const long qt_sh, const long qt_sp,
    const long qs_sr, const long qs_sh, const int MP)
{
    const int req  = blockIdx.x;
    const int kh   = blockIdx.y;
    const int zsp  = blockIdx.z;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nselhi = nsh[req];

    // ---- q -> 2-limb int8 (hi,lo) + per-group amax scale ------------------ //
    __shared__ signed char qhi_sh[MM_M][DMAX];
    __shared__ signed char qlo_sh[MM_M][DMAX];
    __shared__ float       qscale_sh[GMX];
    const __nv_bfloat16* qb = q + (long)req * q_st + (long)(kh * G) * q_hs;
    if (warp == 0) {
        for (int g = 0; g < G; ++g) {
            float mx = 0.f;
            for (int d = lane; d < DMAX; d += 32)
                mx = fmaxf(mx, fabsf(__bfloat162float(qb[(long)g * q_hs + d])));
            #pragma unroll
            for (int o = 16; o >= 1; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
            if (lane == 0) qscale_sh[g] = fmaxf(mx, 1e-12f) / (float)QLIM;
        }
    }
    __syncthreads();
    // publish qscale once (zsp 0) so Pass 2 needs NO amax
    if (zsp == 0 && warp == 0 && lane < G)
        qsc_o[(long)req * qs_sr + (long)kh * qs_sh + lane] = qscale_sh[lane];
    for (int i = threadIdx.x; i < MM_M * DMAX; i += R8_WARPS * 32) {
        int g = i >> 7, d = i & 127;
        signed char hi = 0, lo = 0;
        if (g < G) {
            float qv = __bfloat162float(qb[(long)g * q_hs + d]);
            int qi = __float2int_rn(qv / qscale_sh[g]);
            qi = max(-QLIM, min(QLIM, qi));
            int h = __float2int_rn(qi * (1.f / 128.f));
            h = max(-127, min(127, h));
            int l = max(-127, min(127, qi - h * 128));
            hi = (signed char)h; lo = (signed char)l;
        }
        qhi_sh[g][d] = hi; qlo_sh[g][d] = lo;
    }
    __syncthreads();

    // ---- per-warp cp.async ring: Vk (1 KB) only -------------------------- //
    __shared__ int8_t vk_ring[R8_WARPS][R8_NSTAGE][DMAX * RANK];
    const int nworker = gridDim.z * R8_WARPS;
    const int wid = zsp * R8_WARPS + warp;
    auto load_page = [&](int p, int slot) {
        const int blk = bt[(long)req * bt_stride + p];
        const int8_t* vg = vk + (long)blk * vk_sb + (long)kh * vk_ks;
        int8_t* vd = vk_ring[warp][slot];
        cp_async16(vd + lane * 16, vg + lane * 16);
        cp_async16(vd + 512 + lane * 16, vg + 512 + lane * 16);
    };

    int npages = 0;
    for (int p = wid; p < nselhi && p < MP; p += nworker) npages++;
    int prol = min(R8_NSTAGE - 1, npages);
    for (int s = 0; s < prol; ++s) { load_page(wid + s * nworker, s); cp_commit(); }
    for (int s = prol; s < R8_NSTAGE - 1; ++s) cp_commit();

    __shared__ int comb_sh[R8_WARPS][MM_M][COMBW];

    typedef wmma::fragment<wmma::matrix_a, MM_M, MM_N, MM_K, signed char, wmma::row_major> AFrag;
#if R8_HOIST >= 1
    AFrag fh[KT];
    #pragma unroll
    for (int t = 0; t < KT; ++t) wmma::load_matrix_sync(fh[t], &qhi_sh[0][t * MM_K], DMAX);
#endif
#if R8_HOIST >= 2
    AFrag fl[KT];
    #pragma unroll
    for (int t = 0; t < KT; ++t) wmma::load_matrix_sync(fl[t], &qlo_sh[0][t * MM_K], DMAX);
#endif

    for (int k = 0; k < npages; ++k) {
        const int p = wid + k * nworker;
        const int slot = k % R8_NSTAGE;
        cp_wait<R8_NSTAGE - 2>();
        __syncwarp();
        const int8_t* Vs = vk_ring[warp][slot];

        wmma::fragment<wmma::accumulator, MM_M, MM_N, MM_K, int> ah, al;
        wmma::fill_fragment(ah, 0); wmma::fill_fragment(al, 0);
        #pragma unroll
        for (int t = 0; t < KT; ++t) {
            wmma::fragment<wmma::matrix_b, MM_M, MM_N, MM_K, signed char, wmma::row_major> fv;
            wmma::load_matrix_sync(fv, Vs + t * MM_K * RANK, RANK);
#if R8_HOIST >= 1
            wmma::mma_sync(ah, fh[t], fv, ah);
#else
            AFrag fh_l; wmma::load_matrix_sync(fh_l, &qhi_sh[0][t * MM_K], DMAX);
            wmma::mma_sync(ah, fh_l, fv, ah);
#endif
#if R8_HOIST >= 2
            wmma::mma_sync(al, fl[t], fv, al);
#else
            AFrag fl_l; wmma::load_matrix_sync(fl_l, &qlo_sh[0][t * MM_K], DMAX);
            wmma::mma_sync(al, fl_l, fv, al);
#endif
        }
        #pragma unroll
        for (int i = 0; i < ah.num_elements; ++i) ah.x[i] = 128 * ah.x[i] + al.x[i];
        wmma::store_matrix_sync(&comb_sh[warp][0][0], ah, COMBW, wmma::mem_row_major);

        int kn = k + (R8_NSTAGE - 1);
        if (kn < npages) load_page(wid + kn * nworker, kn % R8_NSTAGE);
        cp_commit();
        __syncwarp();

        // ---- store q~ (GR int32) for this page --------------------------- //
        int* qtp = qtilde + (long)req * qt_sr + (long)kh * qt_sh + (long)p * qt_sp;
        for (int i = lane; i < GR; i += 32) {
            int g = i / RANK, r = i % RANK;
            qtp[i] = comb_sh[warp][g][r];
        }
    }
}

// ======================================================================== //
// PASS 2: read q~ + qmu + qscale + c + scales -> c·q~ + logsumexp -> score.  //
// Register-LIGHT (no A-frags, no q staging) -> HIGH occupancy hides the      //
// per-page logsumexp latency behind the many resident warps.                //
// ======================================================================== //
__global__ __launch_bounds__(R8_TWARPS * 32) void
r8_tail_kernel(
    const __nv_bfloat16* __restrict__ q,     // (T,H,d)
    const int8_t*  __restrict__ mu,          // (NB,n_kv,d)
    const __half*  __restrict__ mus,         // (NB,n_kv)
    const __half*  __restrict__ vks,         // (NB,n_kv,r)
    const int8_t*  __restrict__ cc,          // (NB,n_kv,page,r)
    const __half*  __restrict__ ccs,         // (NB,n_kv,page)
    const int*     __restrict__ bt,          // (n_req, bt_stride)
    const int*     __restrict__ sl,          // (n_req,)
    const int*     __restrict__ nsh,         // (R,)
    const int*     __restrict__ qtilde,      // (R,n_kv,MP,GR) int32
    const float*   __restrict__ qsc_i,       // (R,n_kv,GMX) fp32
    float*         __restrict__ score,       // (R,n_kv,MP)
    const float scale, const int n_kv, const int G,
    const long q_st, const long q_hs,
    const long mu_sb, const long mu_ks, const long mus_sb,
    const long c_sb,  const long c_ks,
    const long vks_sb, const long ccs_sb,
    const int bt_stride,
    const long sc_sr, const long sc_sh,
    const long qt_sr, const long qt_sh, const long qt_sp,
    const long qs_sr, const long qs_sh, const int MP)
{
    const int req  = blockIdx.x;
    const int kh   = blockIdx.y;
    const int zsp  = blockIdx.z;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nselhi = nsh[req];
    const int seqlen = sl[req];
    const int gl  = lane >> 2;                 // group 0..7
    const int sub = lane & 3;                  // sublane 0..3

    // ---- fp32 q + per-group qscale (from Pass 1; NO amax here) ------------ //
    __shared__ float qf_sh[GMX][DMAX + QFPAD];
    __shared__ float qsc_sh[GMX];
    const __nv_bfloat16* qb = q + (long)req * q_st + (long)(kh * G) * q_hs;
    if (warp == 0 && lane < GMX)
        qsc_sh[lane] = (lane < G)
            ? qsc_i[(long)req * qs_sr + (long)kh * qs_sh + lane] : 0.f;
    for (int i = threadIdx.x; i < GMX * DMAX; i += R8_TWARPS * 32) {
        int g = i >> 7, d = i & 127;
        qf_sh[g][d] = (g < G) ? __bfloat162float(qb[(long)g * q_hs + d]) : 0.f;
    }
    __syncthreads();
    const float qsc = qsc_sh[gl];

    // ---- per-warp cp.async ring: q~ (256 B) + mu (128 B) + c (128 B) ------ //
    __shared__ int    qt_ring[R8_TWARPS][R8_TNSTAGE][GR];
    __shared__ int8_t mu_ring[R8_TWARPS][R8_TNSTAGE][DMAX];
    __shared__ int8_t c_ring [R8_TWARPS][R8_TNSTAGE][PAGE * RANK];
    const int nworker = gridDim.z * R8_TWARPS;
    const int wid = zsp * R8_TWARPS + warp;
    auto load_page = [&](int p, int slot) {
        const int blk = bt[(long)req * bt_stride + p];
        const int*   qg = qtilde + (long)req * qt_sr + (long)kh * qt_sh + (long)p * qt_sp;
        const int8_t* mg = mu    + (long)blk * mu_sb + (long)kh * mu_ks;
        const int8_t* cg = cc    + (long)blk * c_sb  + (long)kh * c_ks;
        int*    qd = qt_ring[warp][slot];
        int8_t* md = mu_ring[warp][slot];
        int8_t* cd = c_ring[warp][slot];
        if (lane < 16) cp_async16((char*)qd + lane * 16, (const char*)qg + lane * 16);
        if (lane < 8)  cp_async16(md + lane * 16, mg + lane * 16);
        if (lane < 8)  cp_async16(cd + lane * 16, cg + lane * 16);
    };

    int npages = 0;
    for (int p = wid; p < nselhi && p < MP; p += nworker) npages++;
    int prol = min(R8_TNSTAGE - 1, npages);
    for (int s = 0; s < prol; ++s) { load_page(wid + s * nworker, s); cp_commit(); }
    for (int s = prol; s < R8_TNSTAGE - 1; ++s) cp_commit();

    for (int k = 0; k < npages; ++k) {
        const int p = wid + k * nworker;
        const int slot = k % R8_TNSTAGE;
        cp_wait<R8_TNSTAGE - 2>();
        __syncwarp();
        const int*    Qt = qt_ring[warp][slot];
        const int8_t* Ms = mu_ring[warp][slot];
        const int8_t* Cs = c_ring[warp][slot];
        const int blk = bt[(long)req * bt_stride + p];

        int kn = k + (R8_TNSTAGE - 1);
        if (kn < npages) load_page(wid + kn * nworker, kn % R8_TNSTAGE);
        cp_commit();

        const bool act = (gl < G);
        float myvks = (lane < RANK)
            ? __half2float(vks[(long)blk * vks_sb + (long)kh * RANK + lane]) : 0.f;
        float myccs = (lane < PAGE)
            ? __half2float(ccs[(long)blk * ccs_sb + (long)kh * PAGE + lane]) : 0.f;
        float musf = __shfl_sync(~0u, (lane == 0)
            ? __half2float(mus[(long)blk * mus_sb + kh]) : 0.f, 0);
        float qtil[RANK];
        #pragma unroll
        for (int r = 0; r < RANK; ++r)
            qtil[r] = qsc * __shfl_sync(~0u, myvks, r) * (float)Qt[gl * RANK + r];
        // mu·q (fp32) -- at Pass-2's high occupancy the latency is hidden
        float qm0 = 0.f, qm1 = 0.f, qm2 = 0.f, qm3 = 0.f;
        #pragma unroll
        for (int i = 0; i < DMAX / 16; ++i) {
            int d0 = sub + 4 * (4 * i + 0), d1 = sub + 4 * (4 * i + 1);
            int d2 = sub + 4 * (4 * i + 2), d3 = sub + 4 * (4 * i + 3);
            qm0 += qf_sh[gl][d0] * (float)Ms[d0];
            qm1 += qf_sh[gl][d1] * (float)Ms[d1];
            qm2 += qf_sh[gl][d2] * (float)Ms[d2];
            qm3 += qf_sh[gl][d3] * (float)Ms[d3];
        }
        float qmu = (qm0 + qm1) + (qm2 + qm3);
        qmu += __shfl_xor_sync(~0u, qmu, 1);
        qmu += __shfl_xor_sync(~0u, qmu, 2);
        qmu *= scale * musf;
        // c·q~ + logsumexp: sublane handles tokens t = sub*4 .. sub*4+3
        float m = -CUDART_INF_F, tok[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int t = sub * 4 + j;
            float csc = __shfl_sync(~0u, myccs, t);
            if (p * PAGE + t < seqlen) {
                float a = 0.f;
#if R8_C_BITS == 4
                const int8_t* crow = Cs + t * (RANK / 2);
                #pragma unroll
                for (int jj = 0; jj < RANK / 2; ++jj) {
                    int pb = crow[jj];
                    a += (float)_sext4(pb & 0xF) * qtil[2 * jj];
                    a += (float)_sext4((pb >> 4) & 0xF) * qtil[2 * jj + 1];
                }
#else
                #pragma unroll
                for (int r = 0; r < RANK; ++r) a += (float)Cs[t * RANK + r] * qtil[r];
#endif
                tok[j] = a * scale * csc; m = fmaxf(m, tok[j]);
            } else tok[j] = -CUDART_INF_F;
        }
        float gm = fmaxf(m, __shfl_xor_sync(~0u, m, 1));
        gm = fmaxf(gm, __shfl_xor_sync(~0u, gm, 2));
        float se = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) se += __expf(tok[j] - gm);
        se += __shfl_xor_sync(~0u, se, 1);
        se += __shfl_xor_sync(~0u, se, 2);
        float S = act ? (qmu + (gm + __logf(se))) : -CUDART_INF_F;
        S = fmaxf(S, __shfl_xor_sync(~0u, S, 4));
        S = fmaxf(S, __shfl_xor_sync(~0u, S, 8));
        S = fmaxf(S, __shfl_xor_sync(~0u, S, 16));
        if (lane == 0)
            score[(long)req * sc_sr + (long)kh * sc_sh + p] = S;
    }
}

// ======================================================================== //
// FUSED single kernel (Piece A).  ONE 16x16x16 int8 MMA over the augmented    //
// B = [Vk | mu | 0]  (col 8 = mu) folds mu.q onto the tensor core, and the    //
// 16x16 A-fragment HALVES the register count vs the 32x8 Pass-1 tile.  That    //
// frees registers to run the logsumexp tail IN THE SAME kernel (optionally     //
// software-pipelined behind the next page's MMA), killing the q~ global        //
// round-trip + the serial second launch of the 2-pass design.                  //
//                                                                              //
//   comb[g,0..7] = 128*(qhi.Vk) + (qlo.Vk)   (int q~ = Vkᵀq, 2-limb int8 q)     //
//   comb[g,8]    = 128*(qhi.mu) + (qlo.mu)   (int mu.q,  augmented col 8)       //
//   qtil[g,r]    = qscale[g] * Vk_scale[r] * comb[g,r]                          //
//   qmu[g]       = comb[g,8] * qscale[g] * mu_scale * scale                     //
//   tok[g,t]     = (c[t,:].qtil[g,:]) * scale * c_scale[t]                      //
//   S[g]         = qmu[g] + logsumexp_t tok[g,t] ;  score = max_g S[g]          //
// ======================================================================== //
#ifndef R8_FWARPS
#define R8_FWARPS 8
#endif
#ifndef R8_FNSTAGE
#define R8_FNSTAGE 3
#endif
#ifndef R8_FHOIST
#define R8_FHOIST 0
#endif
#ifndef R8_FMINBLK
#define R8_FMINBLK 0
#endif
#ifndef R8_FPIPE
#define R8_FPIPE 1
#endif
#ifndef R8_VK_BITS
#define R8_VK_BITS 8     // Piece B: 8 = int8 codes, 4 = int4 packed (2/byte)
#endif
#ifndef R8_C_BITS
#define R8_C_BITS 8
#endif
#if R8_FMINBLK > 0
#define R8_FLB __launch_bounds__(R8_FWARPS * 32, R8_FMINBLK)
#else
#define R8_FLB __launch_bounds__(R8_FWARPS * 32)
#endif

#define NAUG 16          // augmented B width (Vk 8 + mu 1 + pad 7)
#define MUCOL 8          // mu occupies column 8 of the augmented B
#define CBW 16           // comb smem row stride
// int4 packs 2 signed nibbles / byte along r: per-(d/token) code bytes halve.
#define VKPB (R8_VK_BITS == 4 ? RANK / 2 : RANK)   // Vk code bytes per d-row
#define CPB  (R8_C_BITS  == 4 ? RANK / 2 : RANK)   // c  code bytes per token
#define VK_BYTES (DMAX * VKPB)                      // Vk code bytes per page
#define C_BYTES  (PAGE * CPB)                       // c  code bytes per page

__global__ void R8_FLB
r8_fused_kernel(
    const __nv_bfloat16* __restrict__ q,     // (T,H,d)
    const int8_t*  __restrict__ vk,          // (NB,n_kv,d,r)
    const __half*  __restrict__ vks,         // (NB,n_kv,r)
    const int8_t*  __restrict__ mu,          // (NB,n_kv,d)
    const __half*  __restrict__ mus,         // (NB,n_kv)
    const int8_t*  __restrict__ cc,          // (NB,n_kv,page,r)
    const __half*  __restrict__ ccs,         // (NB,n_kv,page)
    const int*     __restrict__ bt,          // (n_req, bt_stride)
    const int*     __restrict__ sl,          // (n_req,)
    const int*     __restrict__ nsh,         // (R,)
    float*         __restrict__ score,       // (R,n_kv,MP)
    const float scale, const int n_kv, const int G,
    const long q_st, const long q_hs,
    const long vk_sb, const long vk_ks, const long vks_sb,
    const long mu_sb, const long mu_ks, const long mus_sb,
    const long c_sb,  const long c_ks,  const long ccs_sb,
    const int bt_stride,
    const long sc_sr, const long sc_sh, const int MP)
{
    const int req  = blockIdx.x;
    const int kh   = blockIdx.y;
    const int zsp  = blockIdx.z;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nselhi = nsh[req];
    const int seqlen = sl[req];
    const int gl  = lane >> 2;                 // group 0..7
    const int sub = lane & 3;                  // sublane 0..3

    // ---- q -> 2-limb int8 (hi,lo) + per-group amax scale (16 rows: 8 grp+pad) //
    __shared__ signed char qhi_sh[16][DMAX];     // 16x16 MMA -> only 16 A-rows
    __shared__ signed char qlo_sh[16][DMAX];
    __shared__ float       qscale_sh[GMX];
    const __nv_bfloat16* qb = q + (long)req * q_st + (long)(kh * G) * q_hs;
    if (warp == 0) {
        for (int g = 0; g < G; ++g) {
            float mx = 0.f;
            for (int d = lane; d < DMAX; d += 32)
                mx = fmaxf(mx, fabsf(__bfloat162float(qb[(long)g * q_hs + d])));
            #pragma unroll
            for (int o = 16; o >= 1; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
            if (lane == 0) qscale_sh[g] = fmaxf(mx, 1e-12f) / (float)QLIM;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 16 * DMAX; i += R8_FWARPS * 32) {
        int g = i >> 7, d = i & 127;
        signed char hi = 0, lo = 0;
        if (g < G) {
            float qv = __bfloat162float(qb[(long)g * q_hs + d]);
            int qi = __float2int_rn(qv / qscale_sh[g]);
            qi = max(-QLIM, min(QLIM, qi));
            int h = __float2int_rn(qi * (1.f / 128.f));
            h = max(-127, min(127, h));
            int l = max(-127, min(127, qi - h * 128));
            hi = (signed char)h; lo = (signed char)l;
        }
        qhi_sh[g][d] = hi; qlo_sh[g][d] = lo;
    }
    __syncthreads();

    // ---- per-warp rings: Vk (1 KB) + mu (128 B) + c (128 B) --------------- //
    __shared__ int8_t vk_ring[R8_FWARPS][R8_FNSTAGE][DMAX * RANK];
    __shared__ int8_t mu_ring[R8_FWARPS][R8_FNSTAGE][DMAX];
    __shared__ int8_t c_ring [R8_FWARPS][R8_FNSTAGE][PAGE * RANK];
    // augmented B tile [d][NAUG] is TRANSIENT (built + consumed inside do_mma),
    // so it is single-buffered; only comb + staged c are double-buffered so the
    // deferred tail is ring-slot-independent under pipelining.
    __shared__ int8_t baug_sh[R8_FWARPS][DMAX * NAUG];
    __shared__ int    comb_sh[R8_FWARPS][R8_FPIPE ? 2 : 1][16 * CBW];
    __shared__ int8_t c_buf  [R8_FWARPS][R8_FPIPE ? 2 : 1][PAGE * RANK];

    const int nworker = gridDim.z * R8_FWARPS;
    const int wid = zsp * R8_FWARPS + warp;
    auto load_page = [&](int p, int slot) {
        const int blk = bt[(long)req * bt_stride + p];
        const int8_t* vg = vk + (long)blk * vk_sb + (long)kh * vk_ks;
        const int8_t* mg = mu + (long)blk * mu_sb + (long)kh * mu_ks;
        const int8_t* cg = cc + (long)blk * c_sb  + (long)kh * c_ks;
        int8_t* vd = vk_ring[warp][slot];
        int8_t* md = mu_ring[warp][slot];
        int8_t* cd = c_ring[warp][slot];
        // Vk: int8 = 1024 B (two 16 B/lane); int4 = 512 B (one 16 B/lane).
        cp_async16(vd + lane * 16, vg + lane * 16);
#if R8_VK_BITS != 4
        cp_async16(vd + 512 + lane * 16, vg + 512 + lane * 16);
#endif
        if (lane < 8) cp_async16(md + lane * 16, mg + lane * 16);
        if (lane * 16 < C_BYTES) cp_async16(cd + lane * 16, cg + lane * 16);
    };

    int npages = 0;
    for (int p = wid; p < nselhi && p < MP; p += nworker) npages++;
    int prol = min(R8_FNSTAGE - 1, npages);
    for (int s = 0; s < prol; ++s) { load_page(wid + s * nworker, s); cp_commit(); }
    for (int s = prol; s < R8_FNSTAGE - 1; ++s) cp_commit();

    typedef wmma::fragment<wmma::matrix_a, 16, NAUG, MM_K, signed char, wmma::row_major> AFrag;
    typedef wmma::fragment<wmma::matrix_b, 16, NAUG, MM_K, signed char, wmma::row_major> BFrag;
    typedef wmma::fragment<wmma::accumulator, 16, NAUG, MM_K, int> CFrag;
#if R8_FHOIST >= 1
    AFrag fh[KT];
    #pragma unroll
    for (int t = 0; t < KT; ++t) wmma::load_matrix_sync(fh[t], &qhi_sh[0][t * MM_K], DMAX);
#endif
#if R8_FHOIST >= 2
    AFrag fl[KT];
    #pragma unroll
    for (int t = 0; t < KT; ++t) wmma::load_matrix_sync(fl[t], &qlo_sh[0][t * MM_K], DMAX);
#endif

    // ---- repack ring (Vk,mu) -> augmented B, then MMA -> comb_sh[buf] ------ //
    auto do_mma = [&](int k, int buf) {
        const int slot = k % R8_FNSTAGE;
        const int8_t* Vs = vk_ring[warp][slot];
        const int8_t* Ms = mu_ring[warp][slot];
        int8_t* Baug = baug_sh[warp];
        // cols 0..7 = Vk, col 8 = mu; cols 9..15 left as-is (acc there is dropped)
        for (int d = lane; d < DMAX; d += 32) {
            const int8_t* vrow = Vs + d * VKPB;
            int8_t* brow = Baug + d * NAUG;
#if R8_VK_BITS == 4
            #pragma unroll
            for (int j = 0; j < RANK / 2; ++j) {          // 2 int4 nibbles / byte
                int pb = vrow[j];
                brow[2 * j]     = (signed char)_sext4(pb & 0xF);
                brow[2 * j + 1] = (signed char)_sext4((pb >> 4) & 0xF);
            }
#else
            #pragma unroll
            for (int r = 0; r < RANK; ++r) brow[r] = vrow[r];
#endif
            brow[MUCOL] = Ms[d];
        }
        __syncwarp();
        CFrag ah, al;
        wmma::fill_fragment(ah, 0); wmma::fill_fragment(al, 0);
        #pragma unroll
        for (int t = 0; t < KT; ++t) {
            BFrag fv;
            wmma::load_matrix_sync(fv, Baug + t * MM_K * NAUG, NAUG);
#if R8_FHOIST >= 1
            wmma::mma_sync(ah, fh[t], fv, ah);
#else
            AFrag fh_l; wmma::load_matrix_sync(fh_l, &qhi_sh[0][t * MM_K], DMAX);
            wmma::mma_sync(ah, fh_l, fv, ah);
#endif
#if R8_FHOIST >= 2
            wmma::mma_sync(al, fl[t], fv, al);
#else
            AFrag fl_l; wmma::load_matrix_sync(fl_l, &qlo_sh[0][t * MM_K], DMAX);
            wmma::mma_sync(al, fl_l, fv, al);
#endif
        }
        #pragma unroll
        for (int i = 0; i < ah.num_elements; ++i) ah.x[i] = 128 * ah.x[i] + al.x[i];
        wmma::store_matrix_sync(comb_sh[warp][buf], ah, CBW, wmma::mem_row_major);
        // stage c into the per-buffer store so the tail no longer reads the ring
        // (lets the ring slot recycle immediately when pipelining).
        const int8_t* Cr = c_ring[warp][slot];
        int8_t* Cd = c_buf[warp][buf];
        if (lane < C_BYTES / 4) {
            ((int*)Cd)[lane] = ((const int*)Cr)[lane];
        }
        __syncwarp();   // publish comb_sh[buf] + c_buf[buf] to the whole warp
    };

    // ---- logsumexp tail on comb_sh[buf] -> score[p] ----------------------- //
    auto do_tail = [&](int k, int buf) {
        const int p = wid + k * nworker;
        const int blk = bt[(long)req * bt_stride + p];
        const int* Cmb = comb_sh[warp][buf];
        const int8_t* Cs = c_buf[warp][buf];
        const float qsc = qscale_sh[gl];
        const bool act = (gl < G);
        float myvks = (lane < RANK)
            ? __half2float(vks[(long)blk * vks_sb + (long)kh * RANK + lane]) : 0.f;
        float myccs = (lane < PAGE)
            ? __half2float(ccs[(long)blk * ccs_sb + (long)kh * PAGE + lane]) : 0.f;
        float musf = __shfl_sync(~0u, (lane == 0)
            ? __half2float(mus[(long)blk * mus_sb + kh]) : 0.f, 0);
        float qtil[RANK];
        #pragma unroll
        for (int r = 0; r < RANK; ++r)
            qtil[r] = qsc * __shfl_sync(~0u, myvks, r) * (float)Cmb[gl * CBW + r];
        float qmu = (float)Cmb[gl * CBW + MUCOL] * qsc * scale * musf;
        // c.q~ + logsumexp: sublane handles tokens t = sub*4 .. sub*4+3
        float m = -CUDART_INF_F, tok[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int t = sub * 4 + j;
            float csc = __shfl_sync(~0u, myccs, t);
            if (p * PAGE + t < seqlen) {
                float a = 0.f;
#if R8_C_BITS == 4
                const int8_t* crow = Cs + t * (RANK / 2);
                #pragma unroll
                for (int jj = 0; jj < RANK / 2; ++jj) {
                    int pb = crow[jj];
                    a += (float)_sext4(pb & 0xF) * qtil[2 * jj];
                    a += (float)_sext4((pb >> 4) & 0xF) * qtil[2 * jj + 1];
                }
#else
                #pragma unroll
                for (int r = 0; r < RANK; ++r) a += (float)Cs[t * RANK + r] * qtil[r];
#endif
                tok[j] = a * scale * csc; m = fmaxf(m, tok[j]);
            } else tok[j] = -CUDART_INF_F;
        }
        float gm = fmaxf(m, __shfl_xor_sync(~0u, m, 1));
        gm = fmaxf(gm, __shfl_xor_sync(~0u, gm, 2));
        float se = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) se += __expf(tok[j] - gm);
        se += __shfl_xor_sync(~0u, se, 1);
        se += __shfl_xor_sync(~0u, se, 2);
        float S = act ? (qmu + (gm + __logf(se))) : -CUDART_INF_F;
        S = fmaxf(S, __shfl_xor_sync(~0u, S, 4));
        S = fmaxf(S, __shfl_xor_sync(~0u, S, 8));
        S = fmaxf(S, __shfl_xor_sync(~0u, S, 16));
        if (lane == 0)
            score[(long)req * sc_sr + (long)kh * sc_sh + p] = S;
    };

#if R8_FPIPE
    // Software pipeline: the MMA cursor leads the tail cursor by one page, so
    // MMA(k+1) (tensor core) overlaps tail(k) (fp32) across distinct smem
    // double-buffers.  c is staged into c_buf by do_mma, so a ring slot recycles
    // as soon as its page is MMA'd (the tail no longer pins it).
    if (npages > 0) {
        cp_wait<R8_FNSTAGE - 2>();
        __syncwarp();
        do_mma(0, 0);
        if (R8_FNSTAGE - 1 < npages)
            load_page(wid + (R8_FNSTAGE - 1) * nworker, (R8_FNSTAGE - 1) % R8_FNSTAGE);
        cp_commit();
        for (int k = 0; k < npages; ++k) {
            if (k + 1 < npages) {
                cp_wait<R8_FNSTAGE - 2>();
                __syncwarp();
                do_mma(k + 1, (k + 1) & 1);
                int kn = k + 1 + (R8_FNSTAGE - 1);
                if (kn < npages) load_page(wid + kn * nworker, kn % R8_FNSTAGE);
                cp_commit();
            }
            do_tail(k, k & 1);
        }
    }
#else
    for (int k = 0; k < npages; ++k) {
        cp_wait<R8_FNSTAGE - 2>();
        __syncwarp();
        int kn = k + (R8_FNSTAGE - 1);
        if (kn < npages) load_page(wid + kn * nworker, kn % R8_FNSTAGE);
        cp_commit();
        do_mma(k, 0);
        do_tail(k, 0);
    }
#endif
}

void r8_score_launch(
    torch::Tensor q, torch::Tensor mu, torch::Tensor mus,
    torch::Tensor vk, torch::Tensor vks, torch::Tensor cc, torch::Tensor ccs,
    torch::Tensor bt, torch::Tensor sl, torch::Tensor nsh, torch::Tensor score,
    torch::Tensor qtilde, torch::Tensor qsc,
    double scale, int64_t n_req, int64_t n_kv, int64_t G, int64_t MP,
    int64_t zsplit, int64_t tzsplit)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid1((unsigned)n_req, (unsigned)n_kv, (unsigned)zsplit);
    dim3 block1((unsigned)(R8_WARPS * 32));
    r8_mma_kernel<<<grid1, block1, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        vk.data_ptr<int8_t>(),
        bt.data_ptr<int>(), nsh.data_ptr<int>(),
        qtilde.data_ptr<int>(), qsc.data_ptr<float>(),
        (int)n_kv, (int)G,
        (long)q.stride(0), (long)q.stride(1),
        (long)vk.stride(0), (long)vk.stride(1),
        (int)bt.stride(0),
        (long)qtilde.stride(0), (long)qtilde.stride(1), (long)qtilde.stride(2),
        (long)qsc.stride(0), (long)qsc.stride(1), (int)MP);
    dim3 grid2((unsigned)n_req, (unsigned)n_kv, (unsigned)tzsplit);
    dim3 block2((unsigned)(R8_TWARPS * 32));
    r8_tail_kernel<<<grid2, block2, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        mu.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(mus.data_ptr()),
        reinterpret_cast<const __half*>(vks.data_ptr()),
        cc.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(ccs.data_ptr()),
        bt.data_ptr<int>(), sl.data_ptr<int>(), nsh.data_ptr<int>(),
        qtilde.data_ptr<int>(), qsc.data_ptr<float>(),
        score.data_ptr<float>(),
        (float)scale, (int)n_kv, (int)G,
        (long)q.stride(0), (long)q.stride(1),
        (long)mu.stride(0), (long)mu.stride(1), (long)mus.stride(0),
        (long)cc.stride(0), (long)cc.stride(1),
        (long)vks.stride(0), (long)ccs.stride(0),
        (int)bt.stride(0),
        (long)score.stride(0), (long)score.stride(1),
        (long)qtilde.stride(0), (long)qtilde.stride(1), (long)qtilde.stride(2),
        (long)qsc.stride(0), (long)qsc.stride(1), (int)MP);
}

void r8_fused_launch(
    torch::Tensor q, torch::Tensor mu, torch::Tensor mus,
    torch::Tensor vk, torch::Tensor vks, torch::Tensor cc, torch::Tensor ccs,
    torch::Tensor bt, torch::Tensor sl, torch::Tensor nsh, torch::Tensor score,
    double scale, int64_t n_req, int64_t n_kv, int64_t G, int64_t MP,
    int64_t zsplit)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)n_req, (unsigned)n_kv, (unsigned)zsplit);
    dim3 block((unsigned)(R8_FWARPS * 32));
    r8_fused_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        vk.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(vks.data_ptr()),
        mu.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(mus.data_ptr()),
        cc.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(ccs.data_ptr()),
        bt.data_ptr<int>(), sl.data_ptr<int>(), nsh.data_ptr<int>(),
        score.data_ptr<float>(),
        (float)scale, (int)n_kv, (int)G,
        (long)q.stride(0), (long)q.stride(1),
        (long)vk.stride(0), (long)vk.stride(1), (long)vks.stride(0),
        (long)mu.stride(0), (long)mu.stride(1), (long)mus.stride(0),
        (long)cc.stride(0), (long)cc.stride(1), (long)ccs.stride(0),
        (int)bt.stride(0),
        (long)score.stride(0), (long)score.stride(1), (int)MP);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("r8_score_launch", &r8_score_launch, "AMASS r8 page score (Hopper 2-pass)");
    m.def("r8_fused_launch", &r8_fused_launch, "AMASS r8 page score (Hopper fused)");
}
"""

_MODS = {}   # (vk_bits, c_bits) -> loaded extension (int4 needs a distinct build)


def _get(vk_bits: int = 8, c_bits: int = 8):
    key = (int(vk_bits), int(c_bits))
    if key in _MODS:
        return _MODS[key]
    from torch.utils.cpp_extension import load_inline
    flags = ["-O3", "--use_fast_math",
             f"-DR8_WARPS={_WARPS}", f"-DR8_NSTAGE={_NSTAGE}",
             f"-DR8_MINBLK={_MINBLK}", f"-DR8_HOIST={_HOIST}",
             f"-DR8_TWARPS={_TWARPS}", f"-DR8_TNSTAGE={_TNSTAGE}",
             f"-DR8_FWARPS={_FWARPS}", f"-DR8_FNSTAGE={_FNSTAGE}",
             f"-DR8_FHOIST={_FHOIST}", f"-DR8_FMINBLK={_FMINBLK}",
             f"-DR8_FPIPE={_FPIPE}",
             f"-DR8_VK_BITS={key[0]}", f"-DR8_C_BITS={key[1]}",
             "-gencode=arch=compute_90a,code=sm_90a"]
    suffix = (f"i8_w{_WARPS}s{_NSTAGE}b{_MINBLK}h{_HOIST}_tw{_TWARPS}s{_TNSTAGE}"
              f"_fw{_FWARPS}s{_FNSTAGE}h{_FHOIST}b{_FMINBLK}p{_FPIPE}"
              f"_vk{key[0]}c{key[1]}")
    _verbose = os.environ.get("R8_PTXAS_V", "0") != "0"
    if _verbose:
        flags.append("-Xptxas=-v")
        suffix += "_v"
    mod = load_inline(
        name=f"amass_r8_score_{suffix}",
        cpp_sources="", cuda_sources=_CUDA_SRC,
        extra_cuda_cflags=flags, verbose=_verbose)
    _MODS[key] = mod
    return mod


def _scratch(st: R8State):
    """Lazily allocate the Pass-1 -> Pass-2 scratch (once, outside any graph ->
    stable address, graph-capture safe)."""
    qt = getattr(st, "_r8_qtilde", None)
    if qt is None:
        R, n_kv, MP, r = st.max_reqs, st.n_kv, st.max_pages, st.r
        qt = torch.empty(R, n_kv, MP, GMX * r, device=st.device, dtype=torch.int32)
        qs = torch.empty(R, n_kv, GMX, device=st.device, dtype=torch.float32)
        st._r8_qtilde, st._r8_qsc = qt, qs
    return st._r8_qtilde, st._r8_qsc


def r8_score_cuda(st: R8State, layer: int, q, block_table, seq_lens,
                  n_req: int, scale: float) -> None:
    """CUDA drop-in for ``amass.selection.score.r8_score`` (same contract).

    ``R8_FUSE=1`` (default): the fused single kernel (16x16 augmented [Vk|mu]
    int8 MMA + in-kernel logsumexp tail).  ``R8_FUSE=0``: the legacy 2-pass
    (Pass 1 int8 mma -> q~ scratch; Pass 2 tail at high occupancy)."""
    mu, mu_s, Vk, Vk_s, c, c_s, _ = st.layer_state(layer)
    vk_bits = getattr(st, "vk_bits", 8)
    c_bits = getattr(st, "c_bits", 8)
    mod = _get(vk_bits, c_bits)
    # int4 requires the fused kernel (it unpacks int4->int8 in smem); int8
    # defaults to the faster 2-pass unless R8_FUSE forces the fused kernel.
    if _FUSE or vk_bits == 4 or c_bits == 4:
        mod.r8_fused_launch(
            q, mu, mu_s, Vk, Vk_s, c, c_s,
            block_table, seq_lens, st.n_sel_hi, st.score,
            float(scale), int(n_req), int(st.n_kv), int(st.G),
            int(st.max_pages), int(_FZSPLIT))
        return
    qt, qs = _scratch(st)
    mod.r8_score_launch(
        q, mu, mu_s, Vk, Vk_s, c, c_s,
        block_table, seq_lens, st.n_sel_hi, st.score, qt, qs,
        float(scale), int(n_req), int(st.n_kv), int(st.G), int(st.max_pages),
        int(_ZSPLIT), int(_TZSPLIT))
