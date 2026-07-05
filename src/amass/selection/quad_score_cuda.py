"""quad_score_cuda -- hand-CUDA Hopper (sm_90a) kernels for the AMASS *quad*
page score (the Gaussian-MGF quadratic form; see ``ours_doc/QUAD_SUMMARY_METHOD
.md`` and ``QUAD_IMPL_SPEC.md``).  Drop-in for the Triton ``quad_score`` (same
contract): writes ``st.score`` (R, n_kv, MP) fp32 over the selectable region
[0, n_sel_hi), group-maxed over the GQA group (kv-union).

Reference math per (req, kv-head, page), dequantizing the int8 summary in-kernel
(s = 1/sqrt(d) = ``scale``, P = page, r' = quad rank, default 2):

    qh[g,k] = V_k . q_g                             (r' dots over d)
    quad[g] = (s^2 / (2P)) * sum_{k<r'} sig2[k] * qh[g,k]^2
    qmu[g]  = (mu . q_g) * s
    S[g]    = qmu[g] + quad[g]
    score   = max_g S[g]                            (kv-union group max)

Dequant: mu = mu_code * mu_scale (per (block,head)); V = V_code * V_scale[k]
(per-column r'); sig2 = quad_sig2 (fp16 eigenvalues).

TWO implementations behind one wrapper (``QUAD_CUDA_IMPL`` = ``i8`` default |
``f32`` escape):

* **I8 (default, r'=2 int8/int4 V)** -- the whole (3 x 128 x 8-groups) page dot
  runs on the int8 TENSOR CORE as mma.m16n8k32.s8: A(16x32) = the query in
  2-limb int8 (rows 0-7 hi limbs of the 8 GQA groups, rows 8-15 lo limbs; the
  r8-validated precision recipe qi=round(q/qs), |qi|<=16383, hi=round(qi/128),
  lo=qi-128*hi -> rel|d| ~2e-4, top-b sets identical on real summaries), and
  B(32x8) packs TWO pages per mma: cols [V0 V1 mu 0 | V0' V1' mu' 0].  The
  int32 accumulator is EXACT; comb = 128*hi + lo needs one IMAD (in-lane: C
  rows 0-7/8-15 land in the same thread).  Four mma per page PAIR replace the
  fp32 kernel's ~270 issue slots/page (96 FMA + 128 LDS + converts) -> ~8x
  fewer instructions, 1.4-2.9x measured end-to-end (H200, the win grows with
  batch/ctx as the kernel leaves the latency floor).  V int4 (packed nibbles,
  ``QuadState.v_bits=4``) is a compile variant: same mma, B built by nibble
  sign-extend (__vsub4), V bytes halve.
* **F32 (fallback, any r')** -- the previous fp32 CUDA-core kernel (convert-
  once smem shadow); kept for r' != 2 and as a bitwise escape hatch.

Layout (QuadState, per QUAD_IMPL_SPEC), physical-block keyed, L-major:
    quad_mu    (L, NB, n_kv, d)      int8   + mu_scale (L, NB, n_kv)     fp16
    quad_V     (L, NB, n_kv, d, r')  int8   + V_scale  (L, NB, n_kv, r') fp16
    quad_sig2  (L, NB, n_kv, r')     fp16   (int4 V packs 2 nibbles/byte: d,r'/2)

Fixed launch shapes, no host sync, no allocation -> full-CUDA-graph safe.
Tunables: QUAD_CUDA_{WARPS,ZSPLIT,NSTAGE,RP} (f32), QI8_{WARPS,NSTAGE,ZSPLIT}
(i8); QUAD_PTXAS_V=1 prints ptxas register/occupancy.
"""
from __future__ import annotations

import os

import torch

_WARPS = int(os.environ.get("QUAD_CUDA_WARPS", "8"))      # warps / CTA
_ZSPLIT = int(os.environ.get("QUAD_CUDA_ZSPLIT", "0"))    # page-splits (grid.z);
#   0 = ADAPTIVE (target ~TARGET_BLK CTAs so low batch still fills the SMs).
_NSTAGE = int(os.environ.get("QUAD_CUDA_NSTAGE", "4"))    # cp.async ring depth
_TARGET_BLK = int(os.environ.get("QUAD_CUDA_TARGET_BLK", "4096"))

_I8_WARPS = int(os.environ.get("QI8_WARPS", "4"))
_I8_NSTAGE = int(os.environ.get("QI8_NSTAGE", "4"))
_I8_ZSPLIT = int(os.environ.get("QI8_ZSPLIT", "0"))


def _zsplit(n_req: int, n_kv: int) -> int:
    """grid.z page-split.  The quad score is a tiny per-page op, so at low batch
    (n_req*n_kv small) the grid is too shallow to fill the 132 SMs -> latency-,
    not bandwidth-bound.  Split pages across grid.z to reach ~TARGET_BLK CTAs
    (measured sweet spot; bs1/16K 13.6->11.9us, bs1/64K 41->36us), capped so
    high batch (already block-rich) is not over-split (bs16 prefers ~z32)."""
    if _ZSPLIT > 0:
        return _ZSPLIT
    per = max(1, n_req * n_kv)
    return max(8, min(128, (_TARGET_BLK + per - 1) // per))


def _zsplit_i8(n_req: int, n_kv: int, max_pages: int) -> int:
    """grid.z for the I8 kernel (PAIR-based work units, warps=4).  Swept on
    H200 (scratch_deepopt/score_i8_matrix.py): the knee is ~1024 total CTAs at
    low batch, z<=MP/16 so every worker keeps >=2 pairs, z>=8 floor; every
    grid cell lands within ~8% of its measured best."""
    if _I8_ZSPLIT > 0:
        return _I8_ZSPLIT
    per = max(1, n_req * n_kv)
    z = min(1024 // per if per <= 1024 else 1, max(1, max_pages // 16))
    return max(8, min(128, z))

GMX = 8  # max GQA group supported (mirrors the CUDA constant)

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef QUAD_WARPS
#define QUAD_WARPS 8
#endif
#ifndef QUAD_NSTAGE
#define QUAD_NSTAGE 4
#endif
#ifndef QUAD_RP
#define QUAD_RP 2          // quad rank r' (2 default; supports 1,2,4)
#endif

#define DMAX 128
#define PAGE 16
#define GMX  8
#define QFPAD 4            // qf smem row pad -> conflict-free reads
#define VBYTES (DMAX * QUAD_RP)   // int8 V code bytes / page
#define VW16   (VBYTES / 16)      // 16 B cp.async chunks for V

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// ======================================================================== //
// quad page score.  One CTA per (req, kv-head); warps stream pages round-    //
// robin via a per-warp cp.async ring of V (d*r' int8) + mu (d int8).  q is   //
// staged fp32 to smem once.  Per page: r' fp32 dots (V_k.q) + mu.q, a squared //
// scaled add, and the group-max -- NO logsumexp tail.                        //
// ======================================================================== //
__global__ __launch_bounds__(QUAD_WARPS * 32) void
quad_score_kernel(
    const __nv_bfloat16* __restrict__ q,     // (T,H,d)
    const int8_t*  __restrict__ mu,          // (NB,n_kv,d)
    const __half*  __restrict__ mus,         // (NB,n_kv)
    const int8_t*  __restrict__ vv,          // (NB,n_kv,d,r')
    const __half*  __restrict__ vvs,         // (NB,n_kv,r')
    const __half*  __restrict__ sig2,        // (NB,n_kv,r')
    const int*     __restrict__ bt,          // (n_req, bt_stride)
    const int*     __restrict__ nsh,         // (R,)
    float*         __restrict__ score,       // (R,n_kv,MP)
    const float scale, const int n_kv, const int G,
    const long q_st, const long q_hs,
    const long mu_sb, const long mu_ks, const long mus_sb,
    const long v_sb,  const long v_ks,  const long vs_sb, const long sg_sb,
    const int bt_stride,
    const long sc_sr, const long sc_sh, const int MP)
{
    const int req  = blockIdx.x;
    const int kh   = blockIdx.y;
    const int zsp  = blockIdx.z;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nselhi = nsh[req];
    const int gl  = lane >> 2;                 // group 0..7
    const int sub = lane & 3;                  // sublane 0..3 (d partition)

    // ---- stage q -> fp32 smem once (padded heads -> 0) -------------------- //
    __shared__ float qf_sh[GMX][DMAX + QFPAD];
    const __nv_bfloat16* qb = q + (long)req * q_st + (long)(kh * G) * q_hs;
    for (int i = threadIdx.x; i < GMX * DMAX; i += QUAD_WARPS * 32) {
        int g = i >> 7, d = i & 127;
        qf_sh[g][d] = (g < G) ? __bfloat162float(qb[(long)g * q_hs + d]) : 0.f;
    }
    __syncthreads();

    const float qcoef = scale * scale / (2.f * (float)PAGE);   // s^2/(2P)

    // ---- per-warp cp.async ring: V (d*r' int8) + mu (d int8) -------------- //
    __shared__ int8_t v_ring [QUAD_WARPS][QUAD_NSTAGE][VBYTES];
    __shared__ int8_t mu_ring[QUAD_WARPS][QUAD_NSTAGE][DMAX];
    // Per-page fp32 shadow of V/mu: the int8 codes are per (block,kv-head) --
    // SHARED across the 8 GQA groups the warp scores -- so converting them once
    // here (cooperatively) instead of per group-lane in the dot kills an 8x
    // redundant int8->float convert (the compute bottleneck: the dot is
    // instruction-issue bound on the converts, NOT the smem loads or HBM).
    __shared__ float Vf_sh[QUAD_WARPS][VBYTES];   // d*r' fp32
    __shared__ float Mf_sh[QUAD_WARPS][DMAX];     // d fp32
    const int nworker = gridDim.z * QUAD_WARPS;
    const int wid = zsp * QUAD_WARPS + warp;
    auto load_page = [&](int p, int slot) {
        const int blk = bt[(long)req * bt_stride + p];
        const int8_t* vg = vv + (long)blk * v_sb  + (long)kh * v_ks;
        const int8_t* mg = mu + (long)blk * mu_sb + (long)kh * mu_ks;
        int8_t* vd = v_ring[warp][slot];
        int8_t* md = mu_ring[warp][slot];
        if (lane < VW16)      cp_async16(vd + lane * 16, vg + lane * 16);
        if (lane < DMAX / 16) cp_async16(md + lane * 16, mg + lane * 16);
    };

    int npages = 0;
    for (int p = wid; p < nselhi && p < MP; p += nworker) npages++;
    int prol = min(QUAD_NSTAGE - 1, npages);
    for (int s = 0; s < prol; ++s) { load_page(wid + s * nworker, s); cp_commit(); }
    for (int s = prol; s < QUAD_NSTAGE - 1; ++s) cp_commit();

    for (int k = 0; k < npages; ++k) {
        const int p = wid + k * nworker;
        const int slot = k % QUAD_NSTAGE;
        cp_wait<QUAD_NSTAGE - 2>();
        __syncwarp();
        const int8_t* Vs = v_ring[warp][slot];
        const int8_t* Ms = mu_ring[warp][slot];
        const int blk = bt[(long)req * bt_stride + p];

        // prefetch page k + (NSTAGE-1)
        int kn = k + (QUAD_NSTAGE - 1);
        if (kn < npages) load_page(wid + kn * nworker, kn % QUAD_NSTAGE);
        cp_commit();

        // per-page small scales (direct __half reads, coalesced over pages)
        float myvvs = (lane < QUAD_RP)
            ? __half2float(vvs[(long)blk * vs_sb + (long)kh * QUAD_RP + lane]) : 0.f;
        float mysig = (lane < QUAD_RP)
            ? __half2float(sig2[(long)blk * sg_sb + (long)kh * QUAD_RP + lane]) : 0.f;
        float musf = __shfl_sync(~0u, (lane == 0)
            ? __half2float(mus[(long)blk * mus_sb + kh]) : 0.f, 0);

        // ---- convert V+mu int8 -> fp32 ONCE per page (cooperative, shared
        // across the 8 groups), then the dot reads fp32 (no per-lane convert).
        #pragma unroll
        for (int i = lane; i < VBYTES; i += 32) Vf_sh[warp][i] = (float)Vs[i];
        #pragma unroll
        for (int i = lane; i < DMAX; i += 32) Mf_sh[warp][i] = (float)Ms[i];
        __syncwarp();
        const float* Vf = Vf_sh[warp];
        const float* Mf = Mf_sh[warp];

        // ---- r' dots (V_k.q) + mu.q over this lane's d-partition ---------- //
        float qh[QUAD_RP];
        #pragma unroll
        for (int k2 = 0; k2 < QUAD_RP; ++k2) qh[k2] = 0.f;
        float qm = 0.f;
        #pragma unroll
        for (int i = 0; i < DMAX / 4; ++i) {
            int d = sub + 4 * i;
            float qv = qf_sh[gl][d];
            const float* vr = Vf + d * QUAD_RP;
            #pragma unroll
            for (int k2 = 0; k2 < QUAD_RP; ++k2) qh[k2] += qv * vr[k2];
            qm += qv * Mf[d];
        }
        // reduce across the 4 sublanes of group gl
        #pragma unroll
        for (int k2 = 0; k2 < QUAD_RP; ++k2) {
            qh[k2] += __shfl_xor_sync(~0u, qh[k2], 1);
            qh[k2] += __shfl_xor_sync(~0u, qh[k2], 2);
        }
        qm += __shfl_xor_sync(~0u, qm, 1);
        qm += __shfl_xor_sync(~0u, qm, 2);

        // ---- quad + qmu -> S[g] ------------------------------------------- //
        float quad = 0.f;
        #pragma unroll
        for (int k2 = 0; k2 < QUAD_RP; ++k2) {
            float vsc = __shfl_sync(~0u, myvvs, k2);   // V_scale[k2]
            float sg  = __shfl_sync(~0u, mysig, k2);   // sig2[k2]
            float qhk = qh[k2] * vsc;
            quad += sg * qhk * qhk;
        }
        quad *= qcoef;
        float qmu = qm * scale * musf;
        float S = (gl < G) ? (qmu + quad) : -CUDART_INF_F;

        // ---- kv-union group max over the 8 groups (lane bits 4,8,16) ------ //
        S = fmaxf(S, __shfl_xor_sync(~0u, S, 4));
        S = fmaxf(S, __shfl_xor_sync(~0u, S, 8));
        S = fmaxf(S, __shfl_xor_sync(~0u, S, 16));
        if (lane == 0)
            score[(long)req * sc_sr + (long)kh * sc_sh + p] = S;
    }
}

void quad_score_launch(
    torch::Tensor q, torch::Tensor mu, torch::Tensor mus,
    torch::Tensor vv, torch::Tensor vvs, torch::Tensor sig2,
    torch::Tensor bt, torch::Tensor nsh, torch::Tensor score,
    double scale, int64_t n_req, int64_t n_kv, int64_t G, int64_t MP,
    int64_t zsplit)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)n_req, (unsigned)n_kv, (unsigned)zsplit);
    dim3 block((unsigned)(QUAD_WARPS * 32));
    quad_score_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        mu.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(mus.data_ptr()),
        vv.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(vvs.data_ptr()),
        reinterpret_cast<const __half*>(sig2.data_ptr()),
        bt.data_ptr<int>(), nsh.data_ptr<int>(),
        score.data_ptr<float>(),
        (float)scale, (int)n_kv, (int)G,
        (long)q.stride(0), (long)q.stride(1),
        (long)mu.stride(0), (long)mu.stride(1), (long)mus.stride(0),
        (long)vv.stride(0), (long)vv.stride(1), (long)vvs.stride(0),
        (long)sig2.stride(0),
        (int)bt.stride(0),
        (long)score.stride(0), (long)score.stride(1), (int)MP);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quad_score_launch", &quad_score_launch, "AMASS quad page score (Hopper)");
}
"""

# =========================================================================== #
# I8 kernel: the r'=2 quad score on the int8 tensor core (see module docstring)
# =========================================================================== #
_CUDA_SRC_I8 = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef QI8_WARPS
#define QI8_WARPS 4
#endif
#ifndef QI8_NSTAGE
#define QI8_NSTAGE 4
#endif
#ifndef QI8_V4
#define QI8_V4 0           // 1 = int4 V (packed nibbles along r')
#endif

#define DMAX 128
#define PAGE 16
#define GMX  8
#define RP   2
#define QLIM 16383
// ring slot layout (bytes): VA | pad | VB | pad | muA | pad | muB | pad |
// scalesA 8 (V_scale half2, sig2 half2) | scalesB 8.  The 32 B pads keep the
// page-B smem streams 8 banks away from page-A (conflict-free B-frag loads).
#if QI8_V4
#define VSLOT   DMAX               // packed 2 nibbles/byte along r'=2
#else
#define VSLOT   (DMAX * RP)
#endif
#define VPAD    32
#define MUPAD   32
#define OFF_VA  0
#define OFF_VB  (VSLOT + VPAD)
#define OFF_MUA (2 * (VSLOT + VPAD))
#define OFF_MUB (2 * (VSLOT + VPAD) + DMAX + MUPAD)
#define OFF_SCA (2 * (VSLOT + VPAD) + 2 * (DMAX + MUPAD))
#define OFF_SCB (OFF_SCA + 8)
#define SLOT_BYTES (OFF_SCA + 16)

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async4(void* smem, const void* gmem) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
// mma.m16n8k32 row.col s8*s8 -> s32 (exact integer accumulate)
__device__ __forceinline__ void mma_s8(int d[4], const unsigned a[4],
                                       const unsigned b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ __launch_bounds__(QI8_WARPS * 32) void
qi8_score_kernel(
    const __nv_bfloat16* __restrict__ q,     // (T,H,d)
    const int8_t*  __restrict__ mu,          // (NB,n_kv,d)
    const __half*  __restrict__ mus,         // (NB,n_kv)
    const int8_t*  __restrict__ vv,          // (NB,n_kv,d,r') or packed int4
    const __half*  __restrict__ vvs,         // (NB,n_kv,r')
    const __half*  __restrict__ sig2,        // (NB,n_kv,r')
    const int*     __restrict__ bt,          // (n_req, bt_stride)
    const int*     __restrict__ nsh,         // (R,)
    float*         __restrict__ score,       // (R,n_kv,MP)
    const float scale, const int n_kv, const int G,
    const long q_st, const long q_hs,
    const long mu_sb, const long mu_ks, const long mus_sb,
    const long v_sb,  const long v_ks,  const long vs_sb, const long sg_sb,
    const int bt_stride,
    const long sc_sr, const long sc_sh, const int MP)
{
    const int req  = blockIdx.x;
    const int kh   = blockIdx.y;
    const int zsp  = blockIdx.z;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nselhi = nsh[req];
    const int gid  = lane >> 2;                // 0..7
    const int t4   = lane & 3;                 // 0..3

    // ---- q -> 2-limb int8 (hi,lo words) + per-group scale ------------------ //
    // warp w quantizes groups {w, w+QI8_WARPS, ...}. Each lane owns d=lane*4..+3
    // -> packs 4 hi / 4 lo bytes into one word each.
    __shared__ unsigned qhi_w[GMX][DMAX / 4];
    __shared__ unsigned qlo_w[GMX][DMAX / 4];
    __shared__ float    qsc_sh[GMX];
    for (int g = warp; g < GMX; g += QI8_WARPS) {
        const __nv_bfloat16* qg =
            q + (long)req * q_st + (long)(kh * G + g) * q_hs;
        float v4[4] = {0.f, 0.f, 0.f, 0.f};
        if (g < G) {
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                v4[j] = __bfloat162float(qg[lane * 4 + j]);
        }
        float mx = fmaxf(fmaxf(fabsf(v4[0]), fabsf(v4[1])),
                         fmaxf(fabsf(v4[2]), fabsf(v4[3])));
        #pragma unroll
        for (int o = 16; o >= 1; o >>= 1)
            mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
        const float qs = fmaxf(mx, 1e-12f) / (float)QLIM;
        unsigned hw = 0, lw = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int qi = __float2int_rn(v4[j] / qs);
            qi = max(-QLIM, min(QLIM, qi));
            int h = __float2int_rn(qi * (1.f / 128.f));
            h = max(-127, min(127, h));
            int l = max(-127, min(127, qi - h * 128));
            hw |= (unsigned)((unsigned char)(signed char)h) << (8 * j);
            lw |= (unsigned)((unsigned char)(signed char)l) << (8 * j);
        }
        qhi_w[g][lane] = (g < G) ? hw : 0u;
        qlo_w[g][lane] = (g < G) ? lw : 0u;
        if (lane == 0) qsc_sh[g] = qs;
    }
    __syncthreads();

    // ---- hoisted A fragments (constant across pages) ----------------------- //
    // A[16x32] per k-chunk kk: rows 0-7 = qhi groups, rows 8-15 = qlo groups.
    unsigned Af[4][4];
    #pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        Af[kk][0] = qhi_w[gid][kk * 8 + t4];
        Af[kk][1] = qlo_w[gid][kk * 8 + t4];
        Af[kk][2] = qhi_w[gid][kk * 8 + t4 + 4];
        Af[kk][3] = qlo_w[gid][kk * 8 + t4 + 4];
    }
    const float qcoef = scale * scale / (2.f * (float)PAGE);
    const float qs = qsc_sh[gid];              // constant across pages
    const int half = t4 >> 1;                  // this lane's page-half of C

    // ---- per-warp cp.async ring of page PAIRS ------------------------------ //
    extern __shared__ int8_t ring[];           // [QI8_WARPS][QI8_NSTAGE][SLOT]
    int8_t* wring = ring + (long)warp * QI8_NSTAGE * SLOT_BYTES;
    const int nworker = gridDim.z * QI8_WARPS;
    const int wid = zsp * QI8_WARPS + warp;
    const int npair_hi = (nselhi + 1) >> 1;    // pairs cover pages [0, nselhi)

    auto load_pair = [&](int pp, int slot) {
        int8_t* sl8 = wring + (long)slot * SLOT_BYTES;
        const int p0 = 2 * pp, p1 = 2 * pp + 1;
        {
            const int blk = bt[(long)req * bt_stride + p0];
            const int8_t* vg = vv + (long)blk * v_sb  + (long)kh * v_ks;
            const int8_t* mg = mu + (long)blk * mu_sb + (long)kh * mu_ks;
            if (lane < VSLOT / 16) cp_async16(sl8 + OFF_VA + lane * 16, vg + lane * 16);
            else if (lane < VSLOT / 16 + DMAX / 16)
                cp_async16(sl8 + OFF_MUA + (lane - VSLOT / 16) * 16,
                           mg + (lane - VSLOT / 16) * 16);
            else if (lane == 30)
                cp_async4(sl8 + OFF_SCA, (const int8_t*)(vvs + (long)blk * vs_sb + (long)kh * RP));
            else if (lane == 31)
                cp_async4(sl8 + OFF_SCA + 4, (const int8_t*)(sig2 + (long)blk * sg_sb + (long)kh * RP));
        }
        if (p1 < nselhi) {
            const int blk = bt[(long)req * bt_stride + p1];
            const int8_t* vg = vv + (long)blk * v_sb  + (long)kh * v_ks;
            const int8_t* mg = mu + (long)blk * mu_sb + (long)kh * mu_ks;
            if (lane < VSLOT / 16) cp_async16(sl8 + OFF_VB + lane * 16, vg + lane * 16);
            else if (lane < VSLOT / 16 + DMAX / 16)
                cp_async16(sl8 + OFF_MUB + (lane - VSLOT / 16) * 16,
                           mg + (lane - VSLOT / 16) * 16);
            else if (lane == 30)
                cp_async4(sl8 + OFF_SCB, (const int8_t*)(vvs + (long)blk * vs_sb + (long)kh * RP));
            else if (lane == 31)
                cp_async4(sl8 + OFF_SCB + 4, (const int8_t*)(sig2 + (long)blk * sg_sb + (long)kh * RP));
        }
    };

    int npairs = 0;
    for (int pp = wid; pp < npair_hi; pp += nworker) npairs++;
    int prol = min(QI8_NSTAGE - 1, npairs);
    for (int s = 0; s < prol; ++s) { load_pair(wid + s * nworker, s); cp_commit(); }
    for (int s = prol; s < QI8_NSTAGE - 1; ++s) cp_commit();

    for (int k = 0; k < npairs; ++k) {
        const int pp = wid + k * nworker;
        const int slot = k % QI8_NSTAGE;
        cp_wait<QI8_NSTAGE - 2>();
        __syncwarp();
        const int8_t* sl8 = wring + (long)slot * SLOT_BYTES;
        const int p0 = 2 * pp, p1 = 2 * pp + 1;

        // mus (fp16, 1/page) direct gmem read on the odd lanes, issued EARLY
        const int pmy = half ? p1 : p0;
        float musf = 0.f;
        if (pmy < nselhi && t4 == (half ? 3 : 1)) {
            const int blkm = bt[(long)req * bt_stride + pmy];
            musf = __half2float(mus[(long)blkm * mus_sb + kh]);
        }

        // prefetch pair k + (NSTAGE-1)
        int kn = k + (QI8_NSTAGE - 1);
        if (kn < npairs) load_pair(wid + kn * nworker, kn % QI8_NSTAGE);
        cp_commit();

        // ---- 4x mma over K=128: B cols [V0 V1 mu 0 | V0' V1' mu' 0] -------- //
        // B frag (col-major): b0 = B[k=t4*4+0..3][n=gid], b1 = B[k+16][n=gid].
        int  acc[4] = {0, 0, 0, 0};
        const int col = gid & 3;               // 0,1 = V ; 2 = mu ; 3 = zero
        const int pb  = gid >> 2;              // page half of the B column
        const int8_t* vbase  = sl8 + (pb ? OFF_VB  : OFF_VA);
        const int8_t* mubase = sl8 + (pb ? OFF_MUB : OFF_MUA);
#if QI8_V4
        const unsigned nsh4 = (col == 1) ? 4u : 0u;   // nibble shift for V col
#else
        const unsigned psel = (col == 1) ? 0x7531u : 0x6420u;
#endif
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            unsigned b[2];
            if (col < 2) {
#if QI8_V4
                // packed nibbles: byte d holds cols {0 lo, 1 hi}; sign-extend
                // via (x ^ 8) - 8 per nibble-byte (__vsub4, borrow-free).
                const unsigned* vw = reinterpret_cast<const unsigned*>(
                    vbase + kk * 32 + t4 * 4);
                unsigned w0 = vw[0], w1 = vw[4];
                b[0] = __vsub4(((w0 >> nsh4) & 0x0F0F0F0Fu) ^ 0x08080808u,
                               0x08080808u);
                b[1] = __vsub4(((w1 >> nsh4) & 0x0F0F0F0Fu) ^ 0x08080808u,
                               0x08080808u);
#else
                // V bytes at d*2+col for d = kk*32 + t4*4 (+16): 8 consecutive
                // bytes -> two words + PRMT even/odd.
                const unsigned* vw = reinterpret_cast<const unsigned*>(
                    vbase + (kk * 32 + t4 * 4) * 2);
                unsigned w0 = vw[0], w1 = vw[1], w2 = vw[8], w3 = vw[9];
                b[0] = __byte_perm(w0, w1, psel);
                b[1] = __byte_perm(w2, w3, psel);
#endif
            } else if (col == 2) {
                const unsigned* mw = reinterpret_cast<const unsigned*>(
                    mubase + kk * 32 + t4 * 4);
                b[0] = mw[0]; b[1] = mw[4];
            } else {
                b[0] = 0u; b[1] = 0u;
            }
            mma_s8(acc, Af[kk], b);
        }

        // ---- epilogue: comb = 128*hi + lo (exact int32), dequant, S -------- //
        // lane (gid,t4) holds C cols {t4*2, t4*2+1}: acc0/1 = hi rows, 2/3 = lo.
        const int comb0 = 128 * acc[0] + acc[2];
        const int comb1 = 128 * acc[1] + acc[3];
        const __half2 vvs2 = *reinterpret_cast<const __half2*>(
            sl8 + (half ? OFF_SCB : OFF_SCA));
        const __half2 sg2  = *reinterpret_cast<const __half2*>(
            sl8 + (half ? OFF_SCB : OFF_SCA) + 4);
        float S = -CUDART_INF_F;
        // t4 odd lanes own the mu col -> qmu, shuffled to the even lane.
        float qmu_own = (float)comb0 * qs * musf * scale;
        float qmu = __shfl_sync(~0u, qmu_own, lane | 1);
        if ((t4 & 1) == 0 && (half ? p1 : p0) < nselhi && gid < G) {
            const float2 vs = __half22float2(vvs2);
            const float2 sg = __half22float2(sg2);
            const float qh0 = (float)comb0 * qs * vs.x;
            const float qh1 = (float)comb1 * qs * vs.y;
            S = qmu + qcoef * (sg.x * qh0 * qh0 + sg.y * qh1 * qh1);
        }
        // kv-union group max over gid (lane bits 4,8,16), t4 preserved
        S = fmaxf(S, __shfl_xor_sync(~0u, S, 4));
        S = fmaxf(S, __shfl_xor_sync(~0u, S, 8));
        S = fmaxf(S, __shfl_xor_sync(~0u, S, 16));
        if (lane == 0)
            score[(long)req * sc_sr + (long)kh * sc_sh + p0] = S;
        if (lane == 2 && p1 < nselhi)
            score[(long)req * sc_sr + (long)kh * sc_sh + p1] = S;
    }
}

void qi8_score_launch(
    torch::Tensor q, torch::Tensor mu, torch::Tensor mus,
    torch::Tensor vv, torch::Tensor vvs, torch::Tensor sig2,
    torch::Tensor bt, torch::Tensor nsh, torch::Tensor score,
    double scale, int64_t n_req, int64_t n_kv, int64_t G, int64_t MP,
    int64_t zsplit)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)n_req, (unsigned)n_kv, (unsigned)zsplit);
    dim3 block((unsigned)(QI8_WARPS * 32));
    size_t smem = (size_t)QI8_WARPS * QI8_NSTAGE * SLOT_BYTES;
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(qi8_score_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    }
    qi8_score_kernel<<<grid, block, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        mu.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(mus.data_ptr()),
        vv.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(vvs.data_ptr()),
        reinterpret_cast<const __half*>(sig2.data_ptr()),
        bt.data_ptr<int>(), nsh.data_ptr<int>(),
        score.data_ptr<float>(),
        (float)scale, (int)n_kv, (int)G,
        (long)q.stride(0), (long)q.stride(1),
        (long)mu.stride(0), (long)mu.stride(1), (long)mus.stride(0),
        (long)vv.stride(0), (long)vv.stride(1), (long)vvs.stride(0),
        (long)sig2.stride(0),
        (int)bt.stride(0),
        (long)score.stride(0), (long)score.stride(1), (int)MP);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qi8_score_launch", &qi8_score_launch,
          "AMASS quad page score, int8 tensor core (Hopper)");
}
"""

_MODS = {}   # rp -> loaded extension


def _get(rp: int = 2):
    key = int(rp)
    if key in _MODS:
        return _MODS[key]
    from torch.utils.cpp_extension import load_inline
    flags = ["-O3", "--use_fast_math",
             f"-DQUAD_WARPS={_WARPS}", f"-DQUAD_NSTAGE={_NSTAGE}",
             f"-DQUAD_RP={key}",
             "-gencode=arch=compute_90a,code=sm_90a"]
    suffix = f"rp{key}_w{_WARPS}s{_NSTAGE}"
    _verbose = os.environ.get("QUAD_PTXAS_V", "0") != "0"
    if _verbose:
        flags.append("-Xptxas=-v")
        suffix += "_v"
    mod = load_inline(
        name=f"amass_quad_score_{suffix}",
        cpp_sources="", cuda_sources=_CUDA_SRC,
        extra_cuda_cflags=flags, verbose=_verbose)
    _MODS[key] = mod
    return mod


def _get_i8(v_bits: int = 8):
    """Build (once, hash-cached) the int8 tensor-core kernel (r'=2 only)."""
    key = ("i8", int(v_bits))
    if key in _MODS:
        return _MODS[key]
    from torch.utils.cpp_extension import load_inline
    flags = ["-O3", "--use_fast_math",
             f"-DQI8_WARPS={_I8_WARPS}", f"-DQI8_NSTAGE={_I8_NSTAGE}",
             f"-DQI8_V4={1 if v_bits == 4 else 0}",
             "-gencode=arch=compute_90a,code=sm_90a"]
    suffix = f"v{v_bits}_w{_I8_WARPS}s{_I8_NSTAGE}"
    _verbose = os.environ.get("QUAD_PTXAS_V", "0") != "0"
    if _verbose:
        flags.append("-Xptxas=-v")
        suffix += "_v"
    mod = load_inline(
        name=f"amass_quad_score_i8_{suffix}",
        cpp_sources="", cuda_sources=_CUDA_SRC_I8,
        extra_cuda_cflags=flags, verbose=_verbose)
    _MODS[key] = mod
    return mod


def _quad_layer_state(st, layer: int):
    """Per-layer (mu, mu_scale, V, V_scale, sig2) quad views.  Accepts either a
    QuadState (``quad_mu``/``quad_V``/``quad_sig2`` slabs) or an object exposing
    ``quad_layer_state(layer)``."""
    if hasattr(st, "quad_layer_state"):
        return st.quad_layer_state(layer)
    return (st.quad_mu[layer], st.mu_scale[layer], st.quad_V[layer],
            st.V_scale[layer], st.quad_sig2[layer])


def quad_score_cuda(st, layer: int, q, block_table, seq_lens, n_req: int,
                    scale: float) -> None:
    """CUDA drop-in for the Triton ``quad_score`` (same contract as
    ``r8_score_cuda``).  ``seq_lens`` is accepted for signature parity but the
    quad score needs no per-token validity mask (finalized pages are full).

    Dispatch: r'=2 (the production rank) runs the int8 TENSOR-CORE kernel
    (int8 or packed-int4 V, from the state's V shape); other ranks and the
    ``QUAD_CUDA_IMPL=f32`` escape run the fp32 CUDA-core kernel (int8 V only).
    """
    mu, mu_s, V, V_s, sig2 = _quad_layer_state(st, layer)
    # rank r': prefer explicit ``rp``/``r`` attrs (QuadState); int8 V last dim == r'.
    rp = int(getattr(st, "rp", None) or getattr(st, "r", V.shape[-1]))
    impl = os.environ.get("QUAD_CUDA_IMPL", "i8")
    if rp == 2 and impl != "f32":
        # V last dim: r' (int8) or r'/2 (int4 packed nibbles along r')
        v_bits = 8 if V.shape[-1] == rp else 4
        mod = _get_i8(v_bits)
        mod.qi8_score_launch(
            q, mu, mu_s, V, V_s, sig2,
            block_table, st.n_sel_hi, st.score,
            float(scale), int(n_req), int(st.n_kv), int(st.G),
            int(st.max_pages),
            int(_zsplit_i8(int(n_req), int(st.n_kv), int(st.max_pages))))
        return
    assert V.shape[-1] == rp, (
        "the f32 quad_score_cuda supports int8 V only (V last dim must == r'); "
        "int4 V needs the r'=2 i8 kernel or the Triton quad_score reference")
    mod = _get(rp)
    mod.quad_score_launch(
        q, mu, mu_s, V, V_s, sig2,
        block_table, st.n_sel_hi, st.score,
        float(scale), int(n_req), int(st.n_kv), int(st.G), int(st.max_pages),
        int(_zsplit(int(n_req), int(st.n_kv))))
