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

THREE implementations behind one wrapper (``QUAD_CUDA_IMPL`` = ``i8w``
default | ``i8`` | ``f32`` escape):

* **I8W (default, r'=2 int8 OR packed-int4 V, n_kv % 2 == 0)** -- the I8 math
  with WARP-OWNED FULL-SLAB page streams: each warp loads its pages' codes as
  contiguous KH-head slabs (KH=2 swept best; grid.y = n_kv/KH) and loops the
  KH heads over the slab.  Fixes the measured wall of the per-(req,kv-head) I8
  grid -- the 256B/128B per-head island gather ran at ~60% of the achievable
  HBM rate and was 70%+ of kernel time at bs>=16 (deep-opt-2,
  scratch_deepopt2/): 1.15-1.47x over I8 across the grid, bitwise-identical
  scores.  int4 V (``QuadState.v_bits=4``) is the QI8W_V4 compile variant:
  the V slab halves (128B/head); the mma is fed the biased nibble (nib^8,
  SHR + LOP3) and the -8 plane is folded out int32-exactly in the epilogue
  (comb -= 8*qsum) -> bitwise-equal to the I8 kernel's int4 path (mu STAYS
  int8).  Ships with a PDL trigger
  (griddepcontrol) so a dependent topb overlaps this kernel's drain
  (AMASS_PDL=1 default).
* **I8 (r'=2 int8/int4 V; fallback for odd n_kv / non-slab strides)** -- the whole
  (3 x 128 x 8-groups) page dot
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

Fixed launch shapes, no host sync, no allocation -> full-CUDA-graph safe
(PDL edges are graph-capturable programmatic dependencies).
Tunables: QUAD_CUDA_{WARPS,ZSPLIT,NSTAGE,RP} (f32), QI8_{WARPS,NSTAGE,ZSPLIT}
(i8), QI8W_{KH,ZSPLIT,MINCTAS} (i8w; WARPS/NSTAGE locked by measurement),
AMASS_PDL (default 1); QUAD_PTXAS_V=1 prints ptxas register/occupancy.
Measured negatives (deep-opt-2, do not retry): all-heads-per-CTA lockstep
(i8f), KH=8/4 slabs, packed-Af LDS.128 (register/occupancy), launch_bounds
minctas 12/14 (spills), NSTAGE>2 (needs a deeper prologue; static_assert).
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

_I8W_WARPS = int(os.environ.get("QI8W_WARPS", "4"))
_I8W_NSTAGE = int(os.environ.get("QI8W_NSTAGE", "2"))
_I8W_ZSPLIT = int(os.environ.get("QI8W_ZSPLIT", "0"))
_I8W_KH = int(os.environ.get("QI8W_KH", "2"))     # kv-heads per CTA (slab width)
_I8W_CMAXRREG = int(os.environ.get("QI8W_CLSE_MAXRREG", "0"))  # clse reg cap A/B
_I8W_CWAVE4 = int(os.environ.get("QI8W_CLSE_WAVE4", "0"))      # clse 4+4 tail A/B
_I8W_CSTAGE = int(os.environ.get("QI8W_CLSE_CSTAGE", "0"))     # smem-staged coords


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
    float*         __restrict__ score_h,     // (R,n_kv,G,MP)  (nrm combine)
    const float scale, const int n_kv, const int G,
    const long q_st, const long q_hs,
    const long mu_sb, const long mu_ks, const long mus_sb,
    const long v_sb,  const long v_ks,  const long vs_sb, const long sg_sb,
    const int bt_stride,
    const long sc_sr, const long sc_sh, const int MP,
    const long sh_sr, const long sh_sh, const long sh_sg, const int write_h)
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

        if (write_h) {
            // nrm combine: emit the per-head score S[g] (one sublane owns it).
            if (sub == 0 && gl < G)
                score_h[(long)req * sh_sr + (long)kh * sh_sh
                        + (long)gl * sh_sg + p] = S;
        } else {
            // ---- kv-union group max over the 8 groups (lane bits 4,8,16) -- //
            S = fmaxf(S, __shfl_xor_sync(~0u, S, 4));
            S = fmaxf(S, __shfl_xor_sync(~0u, S, 8));
            S = fmaxf(S, __shfl_xor_sync(~0u, S, 16));
            if (lane == 0)
                score[(long)req * sc_sr + (long)kh * sc_sh + p] = S;
        }
    }
}

void quad_score_launch(
    torch::Tensor q, torch::Tensor mu, torch::Tensor mus,
    torch::Tensor vv, torch::Tensor vvs, torch::Tensor sig2,
    torch::Tensor bt, torch::Tensor nsh, torch::Tensor score,
    torch::Tensor score_h,
    double scale, int64_t n_req, int64_t n_kv, int64_t G, int64_t MP,
    int64_t zsplit, int64_t write_h)
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
        score_h.data_ptr<float>(),
        (float)scale, (int)n_kv, (int)G,
        (long)q.stride(0), (long)q.stride(1),
        (long)mu.stride(0), (long)mu.stride(1), (long)mus.stride(0),
        (long)vv.stride(0), (long)vv.stride(1), (long)vvs.stride(0),
        (long)sig2.stride(0),
        (int)bt.stride(0),
        (long)score.stride(0), (long)score.stride(1), (int)MP,
        (long)score_h.stride(0), (long)score_h.stride(1),
        (long)score_h.stride(2), (int)write_h);
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
    float*         __restrict__ score_h,     // (R,n_kv,G,MP)  (nrm combine)
    const float scale, const int n_kv, const int G,
    const long q_st, const long q_hs,
    const long mu_sb, const long mu_ks, const long mus_sb,
    const long v_sb,  const long v_ks,  const long vs_sb, const long sg_sb,
    const int bt_stride,
    const long sc_sr, const long sc_sh, const int MP,
    const long sh_sr, const long sh_sh, const long sh_sg, const int write_h)
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
        if (write_h) {
            // nrm combine: the t4==0 lane owns S[gid,p0], t4==2 owns S[gid,p1].
            if (t4 == 0 && gid < G)
                score_h[(long)req * sh_sr + (long)kh * sh_sh
                        + (long)gid * sh_sg + p0] = S;
            if (t4 == 2 && gid < G && p1 < nselhi)
                score_h[(long)req * sh_sr + (long)kh * sh_sh
                        + (long)gid * sh_sg + p1] = S;
        } else {
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
}

void qi8_score_launch(
    torch::Tensor q, torch::Tensor mu, torch::Tensor mus,
    torch::Tensor vv, torch::Tensor vvs, torch::Tensor sig2,
    torch::Tensor bt, torch::Tensor nsh, torch::Tensor score,
    torch::Tensor score_h,
    double scale, int64_t n_req, int64_t n_kv, int64_t G, int64_t MP,
    int64_t zsplit, int64_t write_h)
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
        score_h.data_ptr<float>(),
        (float)scale, (int)n_kv, (int)G,
        (long)q.stride(0), (long)q.stride(1),
        (long)mu.stride(0), (long)mu.stride(1), (long)mus.stride(0),
        (long)vv.stride(0), (long)vv.stride(1), (long)vvs.stride(0),
        (long)sig2.stride(0),
        (int)bt.stride(0),
        (long)score.stride(0), (long)score.stride(1), (int)MP,
        (long)score_h.stride(0), (long)score_h.stride(1),
        (long)score_h.stride(2), (int)write_h);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qi8_score_launch", &qi8_score_launch,
          "AMASS quad page score, int8 tensor core (Hopper)");
}
"""

# =========================================================================== #
# I8W kernel: the QI8 math with WARP-OWNED FULL-SLAB page streams (deep-opt-2).
#
# WHY (measured, scratch_deepopt2/score_decomp.py + slab probe + i8f_bench):
# the per-(req,kv-head) grid of the I8 kernel gathers each page's codes as 8
# separate islands (V 256B + mu 128B + 3 scattered scalars per head) -> the
# gather runs at ~60% of the achievable HBM rate and is 70%+ of kernel time at
# bs>=16 (load-only 125us vs 56us for the same bytes read as whole slabs,
# bs16/64K).  The B-fragment-build issue cost was measured NOT the wall
# (planar-deinterleave A/B neutral; compute-only 67us << load-only 125us).
# A CTA-lockstep all-heads-per-CTA variant (i8f) was BITWISE-correct but
# SLOWER (0.38-0.99x): the per-pair __syncthreads convoy at ~4 CTAs/SM erased
# the slab-gather win -> the fix must keep INDEPENDENT per-warp streams.
#
# I8W: one WARP owns a page-PAIR stream (grid (req,1,Z), 4 warps/CTA, worker
# stride Z*4 -- the I8 shape) and loads each page's FULL contiguous slabs
#   V  blk*(n_kv*VHB)B | mu blk*(n_kv*128)B | vvs+sig2+mus (3 16B chunks)
# into its own DOUBLE-BUFFERED ring (prefetch pair k+1 issued BEFORE
# cp.async.wait_group 1 of pair k), then loops the n_kv heads over the slab:
# per head it re-reads the hoistable A fragments from the q-code smem (staged
# once per CTA for all n_kv*G <= 64 heads) and runs the I8 kernel's exact
# 4-mma + epilogue.  8x coarser gather (DRAM-row friendly), 24 scattered
# requests/page -> 5, zero cross-warp sync in the hot loop.  Math, epilogue
# float order, and stores are IDENTICAL to the I8 kernel -> scores BITWISE
# EQUAL (gated in tests + scratch_deepopt2/i8f_bench.py).
#
# QI8W_V4=1 compiles the int4-V variant (VHB 256 -> 128: V arrives as 2 signed
# nibbles/byte along r').  The B fragment feeds the mma the BIASED nibble
# (nib ^ 8) in 0..15 (SHR + one LOP3 per word; the __vsub4 sign-extend of the
# I8 kernel's QI8_V4 path is a ~5-op emulation that measured +37% at the
# issue-bound bs16/16K cell), and the uniform -8 plane is folded out EXACTLY
# in the integer epilogue: signed = (nib^8) - 8 for every V byte, so
# comb_true = comb - 8*qsum with qsum = sum_d qi[d] staged per head in the
# quant prologue (qi == 128*hi + lo exactly under the clamps; all terms
# < 2^26 -> int32-exact).  Same mma, same epilogue float order -> scores
# BITWISE EQUAL to the I8 kernel's QI8_V4 path (gated in tests).  mu STAYS
# int8 in both variants.
# =========================================================================== #
_CUDA_SRC_I8W = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef QI8W_WARPS
#define QI8W_WARPS 4
#endif
#ifndef QI8W_NSTAGE
#define QI8W_NSTAGE 2
#endif
// The ring is a hard double buffer: the prologue primes ONE slot and the loop
// prefetches pair k+1 then cp.async.wait_group 1.  Deeper NSTAGE would need
// an (NSTAGE-1)-deep prologue + wait<NSTAGE-1>; building with NSTAGE=3 as-is
// reads slot k before its load lands (measured BITWISE-FAIL, i8w_knobs.log)
// and showed no speed win -- locked out.
static_assert(QI8W_NSTAGE == 2, "qi8w ring is a hard double buffer");
#ifndef QI8W_KH            // kv-heads per CTA (slab width); 2, 4 or 8
#define QI8W_KH 4
#endif
#ifndef QI8W_V4
#define QI8W_V4 0          // 1 = int4 V (2 signed nibbles/byte along r'=2)
#endif
#ifndef QI8W_CLSE
#define QI8W_CLSE 0        // 1 = CLSE score: per-key coord LSE tail + iso-resid
#endif
#if QI8W_CLSE
#ifndef QI8W_CB
#define QI8W_CB 4          // coord bits: 4 (2 signed nibbles/byte) | 8
#endif
#ifndef QI8W_CTOK
#define QI8W_CTOK 1        // 1 = token-grain c_scale (fp16/key), 0 = page-grain
#endif
#ifndef QI8W_CWAVE4
#define QI8W_CWAVE4 0      // 1 = 4+4 online tail (register A/B)
#endif
#ifndef QI8W_CSTAGE
#define QI8W_CSTAGE 0      // 1 = stage decoded+scaled coords in smem once per
                           //     (pair, head): kills the 8x gid-redundant
                           //     nibble decode + per-token cs multiply
#endif
#if QI8W_CB == 4
#define CBH 16             // c-code slab bytes / kv-head (page * r'/2)
#else
#define CBH 32             // c-code slab bytes / kv-head (page * r')
#endif
#if QI8W_CTOK
#define CSBH 32            // c_scale bytes / kv-head (page fp16)
#else
#define CSBH 2             // one fp16 scale / (block, kv-head)
#endif
#endif

#define DMAX 128
#define PAGE 16
#define GMX  8
#define RP   2
#define QLIM 16383
#if QI8W_V4
#define VHB 128            // V slab bytes / kv-head (d nibble-pairs, packed)
#else
#define VHB 256            // V slab bytes / kv-head (d * r' int8)
#endif
#define QROW 36            // q-code row stride in words (128B + 16B pad):
// the per-(pair,kh) A-fragment re-reads index a DIFFERENT row per lane
// (hg = kh*G + gid) -- an unpadded 32-word stride puts all 8 gid rows on the
// same bank (8-way conflict on EVERY Af load, ~1000 serialized LSU cycles per
// pair, the measured killer of i8w-v1); 36 staggers rows 4 banks apart.
// A PACKED layout serving the re-read as 4x LDS.128 was MEASURED SLOWER
// (bs16/16K 41.7 -> 59.1us): the uint4 temporaries push registers 48 -> 72,
// occupancy 10 -> 7 CTAs/SM, and the kernel is OCCUPANCY-bound, not
// LDS-count-bound (scratch_deepopt2/i8w_af_ab.py).  Keep 16x LDS.32.

// slot geometry (compile-time from QI8W_KH; VHB = V slab bytes per head):
//   V slab   KH*VHB B @ 0        | mu slab KH*128 B @ MUOFF (+32B stagger)
//   scales   vvs @ SCOFF, sig2 @ SCOFF+32, mus @ SCOFF+64 (32B strides)
//   CLSE     c codes @ CLOFF (KH*CBH), c_scale @ CSOFF, resid @ RSOFF
//   page B   @ PBLK = round128(end) + 64  (16-bank offset from page A)
#define MUOFF (QI8W_KH * VHB + 32)
#define SCOFF (MUOFF + QI8W_KH * 128)
#if QI8W_CLSE
#define CLOFF (SCOFF + 96)
#define CSOFF (CLOFF + QI8W_KH * CBH)
#define RSOFF (CSOFF + (((QI8W_KH * CSBH) + 15) / 16) * 16)
#define SLOTEND (RSOFF + (((QI8W_KH * 2) + 15) / 16) * 16)
#define PBLK  (((SLOTEND + 127) / 128) * 128 + 64)
#else
#define PBLK  ((((SCOFF + 96) + 127) / 128) * 128 + 64)
#endif
#define SLOT_BYTES (2 * PBLK)

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
__device__ __forceinline__ void mma_s8(int d[4], const unsigned a[4],
                                       const unsigned b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

#ifndef QI8W_MINCTAS
#define QI8W_MINCTAS 0     // optional launch_bounds minBlocks (occupancy A/B)
#endif
#if QI8W_MINCTAS > 0
__global__ __launch_bounds__(QI8W_WARPS * 32, QI8W_MINCTAS) void
#else
__global__ __launch_bounds__(QI8W_WARPS * 32) void
#endif
qi8w_score_kernel(
    const __nv_bfloat16* __restrict__ q,     // (T,H,d)
    const int8_t*  __restrict__ mu,          // (NB,n_kv,d)
    const __half*  __restrict__ mus,         // (NB,n_kv)
    const int8_t*  __restrict__ vv,          // (NB,n_kv,d,r')
    const __half*  __restrict__ vvs,         // (NB,n_kv,r')
    const __half*  __restrict__ sig2,        // (NB,n_kv,r')
    const int8_t*  __restrict__ cc,          // (NB,n_kv,page,rc)   [CLSE]
    const __half*  __restrict__ cs,          // (NB,n_kv[,page])    [CLSE]
    const __half*  __restrict__ rs,          // (NB,n_kv)           [CLSE]
    const int*     __restrict__ bt,          // (n_req, bt_stride)
    const int*     __restrict__ nsh,         // (R,)
    float*         __restrict__ score,       // (R,n_kv,MP)
    float*         __restrict__ score_h,     // (R,n_kv,G,MP)  (nrm combine)
    const float scale, const float iso_coef, const int n_kv, const int G,
    const long q_st, const long q_hs,
    const long mu_sb, const long mus_sb, const long v_sb,
    const long vs_sb, const long sg_sb,
    const long c_sb, const long cs_sb, const long rs_sb,
    const int bt_stride,
    const long sc_sr, const long sc_sh, const int MP,
    const long sh_sr, const long sh_sh, const long sh_sg, const int write_h)
{
    const int req  = blockIdx.x;
    const int kh0  = blockIdx.y * QI8W_KH;     // first kv-head of this CTA
    const int zsp  = blockIdx.z;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nselhi = nsh[req];
    const int gid  = lane >> 2;                // 0..7
    const int t4   = lane & 3;                 // 0..3

    // dynamic smem: per-warp slab rings | q codes (hi, lo, qsc) for KH*G rows
    extern __shared__ int8_t dsmem[];
    int8_t* ring = dsmem;
    unsigned* qhi_w = reinterpret_cast<unsigned*>(
        dsmem + (long)QI8W_WARPS * QI8W_NSTAGE * SLOT_BYTES);
    unsigned* qlo_w = qhi_w + (long)QI8W_KH * GMX * QROW;
    float*    qsc_sh = reinterpret_cast<float*>(qlo_w + (long)QI8W_KH * GMX * QROW);
#if QI8W_V4
    // per-head q-code sums (sum_d qi[d], exact): the int4 B bytes are fed to
    // the mma as (nib ^ 8) in 0..15 -- signed = (nib^8) - 8 uniformly -- and
    // the -8 plane is folded out in the integer epilogue: comb -= 8*qsum.
    int*      qsum_sh = reinterpret_cast<int*>(qsc_sh + QI8W_KH * GMX);
#endif
#if QI8W_CLSE
    // per-head fp32 |q|^2 (the CLSE iso-residual term needs it)
#if QI8W_V4
    float*    qsq_sh = reinterpret_cast<float*>(qsum_sh + QI8W_KH * GMX);
#else
    float*    qsq_sh = qsc_sh + QI8W_KH * GMX;
#endif
#endif

    // ---- q -> 2-limb int8 for this CTA's KH*G heads (once per CTA) --------- //
    const int nrows = QI8W_KH * G;
    for (int h = warp; h < QI8W_KH * GMX; h += QI8W_WARPS) {
        float v4[4] = {0.f, 0.f, 0.f, 0.f};
        if (h < nrows) {
            const __nv_bfloat16* qg =
                q + (long)req * q_st + (long)(kh0 * G + h) * q_hs;
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
#if QI8W_V4
        int qsl = 0;       // sum of this lane's qi (== 128*hi + lo exactly)
#endif
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int qi = __float2int_rn(v4[j] / qs);
            qi = max(-QLIM, min(QLIM, qi));
            int h2 = __float2int_rn(qi * (1.f / 128.f));
            h2 = max(-127, min(127, h2));
            int l = max(-127, min(127, qi - h2 * 128));
            hw |= (unsigned)((unsigned char)(signed char)h2) << (8 * j);
            lw |= (unsigned)((unsigned char)(signed char)l) << (8 * j);
#if QI8W_V4
            qsl += qi;
#endif
        }
        qhi_w[h * QROW + lane] = (h < nrows) ? hw : 0u;
        qlo_w[h * QROW + lane] = (h < nrows) ? lw : 0u;
        if (lane == 0) qsc_sh[h] = qs;
#if QI8W_V4
        qsl = __reduce_add_sync(~0u, qsl);     // REDUX.SYNC (1 instr, sm_80+)
        if (lane == 0) qsum_sh[h] = (h < nrows) ? qsl : 0;
#endif
#if QI8W_CLSE
        float sq = v4[0] * v4[0] + v4[1] * v4[1] + v4[2] * v4[2]
                 + v4[3] * v4[3];
        #pragma unroll
        for (int o = 16; o >= 1; o >>= 1)
            sq += __shfl_xor_sync(~0u, sq, o);
        if (lane == 0) qsq_sh[h] = (h < nrows) ? sq : 0.f;
#endif
    }
    __syncthreads();

    const float qcoef = scale * scale / (2.f * (float)PAGE);
    const int half = t4 >> 1;                  // this lane's page-half of C
#if QI8W_CLSE && QI8W_CSTAGE
    // decoded+scaled coords, one (pair, head) at a time: [page][token][col]
    __shared__ float cstg_sh[QI8W_WARPS][2][PAGE][2];
#endif

    // ---- per-warp double-buffered ring of page-PAIR slabs ------------------ //
    int8_t* wring = ring + (long)warp * QI8W_NSTAGE * SLOT_BYTES;
    const int nworker = gridDim.z * QI8W_WARPS;
    const int wid = zsp * QI8W_WARPS + warp;
    const int npair_hi = (nselhi + 1) >> 1;

    auto load_page = [&](int p, int8_t* dst) {
        const int blk = bt[(long)req * bt_stride + p];
        const int8_t* vg = vv + (long)blk * v_sb  + (long)kh0 * VHB;
        const int8_t* mg = mu + (long)blk * mu_sb + (long)kh0 * 128;
        #pragma unroll
        for (int i = lane; i < QI8W_KH * (VHB / 16); i += 32)
            cp_async16(dst + i * 16, vg + i * 16);
        #pragma unroll
        for (int i = lane; i < QI8W_KH * 8; i += 32)
            cp_async16(dst + MUOFF + i * 16, mg + i * 16);
        // scales as 4B chunks (uniform across KH; 3 tiny arrays)
        const int8_t* sv = (const int8_t*)(vvs  + (long)blk * vs_sb + kh0 * RP);
        const int8_t* sg = (const int8_t*)(sig2 + (long)blk * sg_sb + kh0 * RP);
        const int8_t* sm = (const int8_t*)(mus  + (long)blk * mus_sb + kh0);
        if (lane < QI8W_KH)
            cp_async4(dst + SCOFF + lane * 4, sv + lane * 4);
        else if (lane < 2 * QI8W_KH)
            cp_async4(dst + SCOFF + 32 + (lane - QI8W_KH) * 4,
                      sg + (lane - QI8W_KH) * 4);
        else if (lane < 2 * QI8W_KH + QI8W_KH / 2)
            cp_async4(dst + SCOFF + 64 + (lane - 2 * QI8W_KH) * 4,
                      sm + (lane - 2 * QI8W_KH) * 4);
#if QI8W_CLSE
        // CLSE slabs: c codes (KH*CBH B), c_scale (KH*CSBH B), resid (KH*2 B)
        const int8_t* cg = cc + (long)blk * c_sb + (long)kh0 * CBH;
        #pragma unroll
        for (int i = lane; i < QI8W_KH * (CBH / 16); i += 32)
            cp_async16(dst + CLOFF + i * 16, cg + i * 16);
#if QI8W_CTOK
        const int8_t* csg = (const int8_t*)cs + ((long)blk * cs_sb + kh0 * PAGE) * 2;
        #pragma unroll
        for (int i = lane; i < QI8W_KH * (CSBH / 16); i += 32)
            cp_async16(dst + CSOFF + i * 16, csg + i * 16);
#else
        const int8_t* csg = (const int8_t*)cs + ((long)blk * cs_sb + kh0) * 2;
        if (lane >= 24 && lane < 24 + QI8W_KH / 2)
            cp_async4(dst + CSOFF + (lane - 24) * 4, csg + (lane - 24) * 4);
#endif
        const int8_t* rg = (const int8_t*)rs + ((long)blk * rs_sb + kh0) * 2;
        if (lane >= 28 && lane < 28 + QI8W_KH / 2)
            cp_async4(dst + RSOFF + (lane - 28) * 4, rg + (lane - 28) * 4);
#endif
    };
    auto load_pair = [&](int pp, int slot) {
        int8_t* sl8 = wring + (long)slot * SLOT_BYTES;
        const int p0 = 2 * pp, p1 = 2 * pp + 1;
        load_page(p0, sl8);
        if (p1 < nselhi) load_page(p1, sl8 + PBLK);
    };

    int npairs = 0;
    for (int pp = wid; pp < npair_hi; pp += nworker) npairs++;
    if (npairs > 0) { load_pair(wid, 0); }
    cp_commit();

    for (int k = 0; k < npairs; ++k) {
        const int pp = wid + k * nworker;
        const int slot = k % QI8W_NSTAGE;
        // prefetch pair k+1 BEFORE waiting on pair k (double buffer)
        if (k + 1 < npairs)
            load_pair(wid + (k + 1) * nworker, (k + 1) % QI8W_NSTAGE);
        cp_commit();
        cp_wait<QI8W_NSTAGE - 1>();
        __syncwarp();
        const int8_t* sl8 = wring + (long)slot * SLOT_BYTES;
        const int p0 = 2 * pp, p1 = 2 * pp + 1;
        const int pmy = half ? p1 : p0;
        const int8_t* pb8 = sl8 + (half ? PBLK : 0);
        const int  pvalid = (pmy < nselhi);

        // ---- loop this CTA's kv-heads over the slab -------------------- //
        #pragma unroll
        for (int kh = 0; kh < QI8W_KH; ++kh) {
            // A fragments for head (kh0+kh, gid) re-read from the q-code smem
            const int hg = (kh * G + ((gid < G) ? gid : 0)) * QROW;
            unsigned Af[4][4];
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                Af[kk][0] = (gid < G) ? qhi_w[hg + kk * 8 + t4]     : 0u;
                Af[kk][1] = (gid < G) ? qlo_w[hg + kk * 8 + t4]     : 0u;
                Af[kk][2] = (gid < G) ? qhi_w[hg + kk * 8 + t4 + 4] : 0u;
                Af[kk][3] = (gid < G) ? qlo_w[hg + kk * 8 + t4 + 4] : 0u;
            }
            const float qs = (gid < G) ? qsc_sh[kh * G + gid] : 0.f;
            float musf = 0.f;
            if (pvalid && (t4 & 1) == 1)
                musf = __half2float(
                    reinterpret_cast<const __half*>(pb8 + SCOFF + 64)[kh]);

            int  acc[4] = {0, 0, 0, 0};
            const int col = gid & 3;
            const int pb  = gid >> 2;
            const int8_t* base   = sl8 + (pb ? PBLK : 0);
            const int8_t* vbase  = base + kh * VHB;
            const int8_t* mubase = base + MUOFF + kh * 128;
#if QI8W_V4
            const unsigned nsh4 = (col == 1) ? 4u : 0u;   // nibble shift for V col
#else
            const unsigned psel = (col == 1) ? 0x7531u : 0x6420u;
#endif
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                unsigned b[2];
                if (col < 2) {
#if QI8W_V4
                    // packed nibbles: byte d holds V cols {0 lo, 1 hi}.  Feed
                    // the mma the BIASED nibble (nib ^ 8) in 0..15 -- one SHR
                    // + one LOP3 per word, no __vsub4 emulation (its ~5-op
                    // chain measured +37% at the issue-bound bs16/16K cell).
                    // signed = (nib ^ 8) - 8 UNIFORMLY, so the -8 plane is an
                    // exact integer rank-1 term folded out in the epilogue:
                    // comb_true = comb - 8*qsum (qsum staged per head).
                    const unsigned* vw = reinterpret_cast<const unsigned*>(
                        vbase + kk * 32 + t4 * 4);
                    b[0] = ((vw[0] >> nsh4) & 0x0F0F0F0Fu) ^ 0x08080808u;
                    b[1] = ((vw[4] >> nsh4) & 0x0F0F0F0Fu) ^ 0x08080808u;
#else
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

            // ---- epilogue: IDENTICAL float sequence to the I8 kernel ---- //
            const int comb0 = 128 * acc[0] + acc[2];
            const int comb1 = 128 * acc[1] + acc[3];
            const __half2 vvs2 = *reinterpret_cast<const __half2*>(
                pb8 + SCOFF + kh * RP * 2);
            float S = -CUDART_INF_F;
            float qmu_own = (float)comb0 * qs * musf * scale;
            float qmu = __shfl_sync(~0u, qmu_own, lane | 1);
#if QI8W_CLSE
            // ---- CLSE tail: S = qmu + logsumexp_t[(qh.c_t) s] + iso ------ //
            // qh0/qh1 (dequantized r'=2 projections) live on the EVEN t4
            // lanes (V-column combs); broadcast to the odd pair lane, then
            // the 16-token LSE is SPLIT across the (even, odd) pair (8 tokens
            // each) and merged with one shuffle -- halves the tail chain and
            // puts the otherwise-idle mu-column lanes to work.
            {
                const float2 vs = __half22float2(vvs2);
#if QI8W_V4
                const int cq8 = 8 * qsum_sh[kh * G + ((gid < G) ? gid : 0)];
                float qh0 = (float)(comb0 - cq8) * qs * vs.x;
                float qh1 = (float)(comb1 - cq8) * qs * vs.y;
#else
                float qh0 = (float)comb0 * qs * vs.x;
                float qh1 = (float)comb1 * qs * vs.y;
#endif
                qh0 = __shfl_sync(~0u, qh0, lane & ~1);
                qh1 = __shfl_sync(~0u, qh1, lane & ~1);
                const int tbase = (t4 & 1) * 8;        // even: 0-7, odd: 8-15
#if QI8W_CSTAGE
                // ---- cooperative coord decode: the c bytes are per (page,
                // head) -- SHARED across the 8 gid lanes -- so decode+scale
                // them ONCE per (pair, head) (lane l -> page l>>4, token
                // l&15, both cols) into a tiny smem plane instead of 8x
                // redundantly per lane (the cvt lesson).  Association matches
                // Triton (c*cs first).
                __syncwarp();
                {
                    const int sp2 = lane >> 4;         // staged page half
                    const int stk = lane & 15;         // staged token
                    const int8_t* sb = sl8 + (sp2 ? PBLK : 0);
#if QI8W_CTOK
                    const float scs = __half2float(reinterpret_cast<const __half*>(
                        sb + CSOFF + kh * CSBH)[stk]);
#else
                    const float scs = __half2float(reinterpret_cast<const __half*>(
                        sb + CSOFF)[kh]);
#endif
#if QI8W_CB == 4
                    const unsigned byt = ((unsigned)(unsigned char)
                        (sb + CLOFF + kh * CBH)[stk]) ^ 0x88u;
                    const float f0 = (__uint_as_float(0x4B000000u | (byt & 0xFu))
                                      - (8388608.f + 8.f)) * scs;
                    const float f1 = (__uint_as_float(0x4B000000u | (byt >> 4))
                                      - (8388608.f + 8.f)) * scs;
#else
                    const int8_t* cb2 = sb + CLOFF + kh * CBH + stk * 2;
                    const unsigned b0 = ((unsigned)(unsigned char)cb2[0]) ^ 0x80u;
                    const unsigned b1 = ((unsigned)(unsigned char)cb2[1]) ^ 0x80u;
                    const float f0 = (__uint_as_float(0x4B000000u | b0)
                                      - (8388608.f + 128.f)) * scs;
                    const float f1 = (__uint_as_float(0x4B000000u | b1)
                                      - (8388608.f + 128.f)) * scs;
#endif
                    cstg_sh[warp][sp2][stk][0] = f0;
                    cstg_sh[warp][sp2][stk][1] = f1;
                }
                __syncwarp();
                const float A = qh0 * scale, B2 = qh1 * scale;
                float tok[8];
                {
                    const float2* cv2 = reinterpret_cast<const float2*>(
                        &cstg_sh[warp][half][0][0]) + tbase;
                    #pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        const float2 cv = cv2[t];
                        tok[t] = A * cv.x + B2 * cv.y;
                    }
                }
                float m0 = tok[0];
                #pragma unroll
                for (int t = 1; t < 8; ++t) m0 = fmaxf(m0, tok[t]);
                float s0 = 0.f;
                #pragma unroll
                for (int t = 0; t < 8; ++t) s0 += __expf(tok[t] - m0);
#else
#if QI8W_CTOK
                const float A = qh0 * scale, B2 = qh1 * scale;
                const __half* csp = reinterpret_cast<const __half*>(
                    pb8 + CSOFF + kh * CSBH);
#else
                const float csf = __half2float(reinterpret_cast<const __half*>(
                    pb8 + CSOFF)[kh]);
                const float A = qh0 * scale * csf, B2 = qh1 * scale * csf;
#endif
#if QI8W_CWAVE4
                // 4+4 register-lean tail: two waves of 4 tokens, online merge
                // (saves the tok[8] live range; +2 MUFU for the wave fold).
                float m0 = -CUDART_INF_F, s0 = 0.f;
                #pragma unroll
                for (int w2 = 0; w2 < 2; ++w2) {
                    float tk[4];
#if QI8W_CB == 4
                    const unsigned* cw = reinterpret_cast<const unsigned*>(
                        pb8 + CLOFF + kh * CBH + tbase);
                    const unsigned wv = cw[w2] ^ 0x88888888u;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const unsigned byt = (wv >> (8 * j)) & 0xFFu;
                        const float f0 = __uint_as_float(0x4B000000u | (byt & 0xFu))
                                         - (8388608.f + 8.f);
                        const float f1 = __uint_as_float(0x4B000000u | (byt >> 4))
                                         - (8388608.f + 8.f);
                        tk[j] = A * f0 + B2 * f1;
                    }
#else
                    const unsigned* cw = reinterpret_cast<const unsigned*>(
                        pb8 + CLOFF + kh * CBH + tbase * 2);
                    #pragma unroll
                    for (int w = 0; w < 2; ++w) {
                        const unsigned wv = cw[w2 * 2 + w];
                        const float c00 = __uint_as_float(
                            0x4B000000u | ((wv & 0xFFu) ^ 0x80u)) - (8388608.f + 128.f);
                        const float c01 = __uint_as_float(
                            0x4B000000u | (((wv >> 8) & 0xFFu) ^ 0x80u)) - (8388608.f + 128.f);
                        const float c10 = __uint_as_float(
                            0x4B000000u | (((wv >> 16) & 0xFFu) ^ 0x80u)) - (8388608.f + 128.f);
                        const float c11 = __uint_as_float(
                            0x4B000000u | ((wv >> 24) ^ 0x80u)) - (8388608.f + 128.f);
                        tk[w * 2]     = A * c00 + B2 * c01;
                        tk[w * 2 + 1] = A * c10 + B2 * c11;
                    }
#endif
#if QI8W_CTOK
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        tk[j] *= __half2float(csp[tbase + w2 * 4 + j]);
#endif
                    const float mw = fmaxf(fmaxf(tk[0], tk[1]),
                                           fmaxf(tk[2], tk[3]));
                    const float sw = __expf(tk[0] - mw) + __expf(tk[1] - mw)
                                   + __expf(tk[2] - mw) + __expf(tk[3] - mw);
                    if (w2 == 0) { m0 = mw; s0 = sw; }
                    else {
                        const float mn = fmaxf(m0, mw);
                        s0 = s0 * __expf(m0 - mn) + sw * __expf(mw - mn);
                        m0 = mn;
                    }
                }
#else
                float tok[8];
#if QI8W_CB == 4
                // magic-fp nibble decode: two's-complement nibble n ->
                // as_float(0x4B000000 | (n ^ 8)) - (2^23 + 8) == signed(n)
                // EXACTLY ((n^8)-8 is the sign fold), no I2F pipe.
                const unsigned* cw = reinterpret_cast<const unsigned*>(
                    pb8 + CLOFF + kh * CBH + tbase);
                #pragma unroll
                for (int w = 0; w < 2; ++w) {
                    const unsigned wv = cw[w] ^ 0x88888888u;   // fold both nibbles
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const unsigned byt = (wv >> (8 * j)) & 0xFFu;
                        const float f0 = __uint_as_float(0x4B000000u | (byt & 0xFu))
                                         - (8388608.f + 8.f);
                        const float f1 = __uint_as_float(0x4B000000u | (byt >> 4))
                                         - (8388608.f + 8.f);
                        tok[w * 4 + j] = A * f0 + B2 * f1;
                    }
                }
#else
                // int8 coords: (c0,c1) byte pair / token; signed via ^0x80 fold
                const unsigned* cw = reinterpret_cast<const unsigned*>(
                    pb8 + CLOFF + kh * CBH + tbase * 2);
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const unsigned wv = cw[w];
                    const float c00 = __uint_as_float(
                        0x4B000000u | ((wv & 0xFFu) ^ 0x80u)) - (8388608.f + 128.f);
                    const float c01 = __uint_as_float(
                        0x4B000000u | (((wv >> 8) & 0xFFu) ^ 0x80u)) - (8388608.f + 128.f);
                    const float c10 = __uint_as_float(
                        0x4B000000u | (((wv >> 16) & 0xFFu) ^ 0x80u)) - (8388608.f + 128.f);
                    const float c11 = __uint_as_float(
                        0x4B000000u | ((wv >> 24) ^ 0x80u)) - (8388608.f + 128.f);
                    tok[w * 2]     = A * c00 + B2 * c01;
                    tok[w * 2 + 1] = A * c10 + B2 * c11;
                }
#endif
#if QI8W_CTOK
                #pragma unroll
                for (int t = 0; t < 8; ++t)
                    tok[t] *= __half2float(csp[tbase + t]);
#endif
                float m0 = tok[0];
                #pragma unroll
                for (int t = 1; t < 8; ++t) m0 = fmaxf(m0, tok[t]);
                float s0 = 0.f;
                #pragma unroll
                for (int t = 0; t < 8; ++t) s0 += __expf(tok[t] - m0);
#endif
#endif  // QI8W_CSTAGE
                const float m1 = __shfl_xor_sync(~0u, m0, 1);
                const float s1 = __shfl_xor_sync(~0u, s0, 1);
                const float mm = fmaxf(m0, m1);
                const float ss = s0 * __expf(m0 - mm) + s1 * __expf(m1 - mm);
                if ((t4 & 1) == 0 && pvalid && gid < G) {
                    const float lse = mm + __logf(ss);
                    const float rsf = __half2float(
                        reinterpret_cast<const __half*>(pb8 + RSOFF)[kh]);
                    const float iso = fmaxf(
                        qsq_sh[kh * G + gid] - (qh0 * qh0 + qh1 * qh1), 0.f)
                        * rsf * iso_coef;
                    S = qmu + lse + iso;
                }
            }
#else
            const __half2 sg2  = *reinterpret_cast<const __half2*>(
                pb8 + SCOFF + 32 + kh * RP * 2);
            if ((t4 & 1) == 0 && pvalid && gid < G) {
                const float2 vs = __half22float2(vvs2);
                const float2 sg = __half22float2(sg2);
#if QI8W_V4
                // fold out the +8 nibble bias EXACTLY (int32): the V-column
                // combs (even-t4 lanes only; the mu column fed exact int8
                // bytes) carry +8*qsum from the biased B -> subtract before
                // the float convert.  All terms < 2^26 -> exact, so scores
                // stay BITWISE-equal to the signed-nibble reference.
                const int cq8 = 8 * qsum_sh[kh * G + gid];
                const float qh0 = (float)(comb0 - cq8) * qs * vs.x;
                const float qh1 = (float)(comb1 - cq8) * qs * vs.y;
#else
                const float qh0 = (float)comb0 * qs * vs.x;
                const float qh1 = (float)comb1 * qs * vs.y;
#endif
                S = qmu + qcoef * (sg.x * qh0 * qh0 + sg.y * qh1 * qh1);
            }
#endif
            if (write_h) {
                // nrm combine: t4==0 lane owns S[gid,p0], t4==2 owns S[gid,p1].
                if (t4 == 0 && gid < G)
                    score_h[(long)req * sh_sr + (long)(kh0 + kh) * sh_sh
                            + (long)gid * sh_sg + p0] = S;
                if (t4 == 2 && gid < G && p1 < nselhi)
                    score_h[(long)req * sh_sr + (long)(kh0 + kh) * sh_sh
                            + (long)gid * sh_sg + p1] = S;
            } else {
                S = fmaxf(S, __shfl_xor_sync(~0u, S, 4));
                S = fmaxf(S, __shfl_xor_sync(~0u, S, 8));
                S = fmaxf(S, __shfl_xor_sync(~0u, S, 16));
                if (lane == 0)
                    score[(long)req * sc_sr + (long)(kh0 + kh) * sc_sh + p0] = S;
                if (lane == 2 && p1 < nselhi)
                    score[(long)req * sc_sr + (long)(kh0 + kh) * sc_sh + p1] = S;
            }
        }
    }
    // PDL: release dependents (e.g. topb launched with the programmatic-
    // stream-serialization attribute) as this grid drains; the instruction
    // is a no-op when no dependent is armed.  Prior stores of this thread
    // are visible to a dependent that executes griddepcontrol.wait.
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

void qi8w_score_launch(
    torch::Tensor q, torch::Tensor mu, torch::Tensor mus,
    torch::Tensor vv, torch::Tensor vvs, torch::Tensor sig2,
    torch::Tensor cc, torch::Tensor cs, torch::Tensor rs,
    torch::Tensor bt, torch::Tensor nsh, torch::Tensor score,
    torch::Tensor score_h,
    double scale, double iso_coef, int64_t n_req, int64_t n_kv, int64_t G,
    int64_t MP, int64_t zsplit, int64_t write_h)
{
    TORCH_CHECK(n_kv % QI8W_KH == 0, "qi8w: n_kv must be a multiple of KH");
    TORCH_CHECK(G <= 8, "qi8w: G must be <= 8");
    auto stream = at::cuda::getCurrentCUDAStream();
    size_t qcodes = (size_t)QI8W_KH * GMX * QROW * 2 * 4 + QI8W_KH * GMX * 4;
#if QI8W_V4
    qcodes += (size_t)QI8W_KH * GMX * 4;       // per-head qsum plane
#endif
#if QI8W_CLSE
    qcodes += (size_t)QI8W_KH * GMX * 4;       // per-head |q|^2 plane
#endif
    size_t smem = (size_t)QI8W_WARPS * QI8W_NSTAGE * SLOT_BYTES + qcodes;
    dim3 grid((unsigned)n_req, (unsigned)(n_kv / QI8W_KH), (unsigned)zsplit);
    dim3 block((unsigned)(QI8W_WARPS * 32));
    if (smem > 48 * 1024) {
        static size_t recorded = 0;
        if (smem > recorded) {
            cudaFuncSetAttribute(qi8w_score_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            recorded = smem;
        }
    }
    qi8w_score_kernel<<<grid, block, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        mu.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(mus.data_ptr()),
        vv.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(vvs.data_ptr()),
        reinterpret_cast<const __half*>(sig2.data_ptr()),
        cc.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(cs.data_ptr()),
        reinterpret_cast<const __half*>(rs.data_ptr()),
        bt.data_ptr<int>(), nsh.data_ptr<int>(),
        score.data_ptr<float>(),
        score_h.data_ptr<float>(),
        (float)scale, (float)iso_coef, (int)n_kv, (int)G,
        (long)q.stride(0), (long)q.stride(1),
        (long)mu.stride(0), (long)mus.stride(0), (long)vv.stride(0),
        (long)vvs.stride(0), (long)sig2.stride(0),
        (long)cc.stride(0), (long)cs.stride(0), (long)rs.stride(0),
        (int)bt.stride(0),
        (long)score.stride(0), (long)score.stride(1), (int)MP,
        (long)score_h.stride(0), (long)score_h.stride(1),
        (long)score_h.stride(2), (int)write_h);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qi8w_score_launch", &qi8w_score_launch,
          "AMASS quad page score, int8 tensor core, warp-owned slab streams");
}
"""


# =========================================================================== #
# Fused CUDA nrm-combine: ONE kernel replaces the two Triton passes            #
# (per-head page-max, then score[p] = sum_g exp(S[g,p] - hmax[g])).            #
#                                                                              #
# WHY (final-opt profile): the Triton passes read score_h TWICE from DRAM and  #
# monolithically tile P_PAD = next_pow2(MP) (a (G, 8192) fp32 block per        #
# program at 64K ctx) -> 55-57us at bs16/64K vs a ~19us byte floor, plus two   #
# extra launches on the bs1 latency path.  Here one CTA owns a (req, kv-head)  #
# row: phase 1 sweeps S[g, 0:nsh) for the per-g max (bringing the row into     #
# L2: G*MP*4 = 131KB/row), phase 2 re-reads it L2-hot for the exp-sum.  Same   #
# math, deterministic (fixed reduction order), no hmax global buffer, no       #
# alloc, fixed shapes -> graph-safe.  PDL: waits on the score kernel's drain   #
# and releases the dependent topb (the score->nrm->topb->decode chain).        #
# =========================================================================== #
_CUDA_SRC_NRM = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <ATen/cuda/CUDAContext.h>

#define GMX 8

__global__ __launch_bounds__(1024) void
nrm_combine_kernel(
    const float* __restrict__ score_h,   // (R, n_kv, G, MP)
    const int*   __restrict__ nsh,       // (R,)
    float*       __restrict__ score,     // (R, n_kv, MP)
    const int G,
    const long sh_sr, const long sh_sh, const long sh_sg,
    const long sc_sr, const long sc_sh)
{
    const int req = blockIdx.x;
    const int kh  = blockIdx.y;
    const int tid = threadIdx.x;
    const int BLK = blockDim.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    const int nwarp = BLK >> 5;

    __shared__ float wmax[32][GMX];      // per-warp per-g maxima
    __shared__ float hmax_sh[GMX];

    // PDL: start while the score grid drains; fence before reading score_h.
    asm volatile("griddepcontrol.wait;" ::: "memory");

    const int n = nsh[req];
    const float* shp = score_h + (long)req * sh_sr + (long)kh * sh_sh;

    // ---- phase 1: per-g max over the selectable region ------------------- //
    float lmax[GMX];
    #pragma unroll
    for (int g = 0; g < GMX; ++g) lmax[g] = -CUDART_INF_F;
    for (int p = tid; p < n; p += BLK) {
        #pragma unroll
        for (int g = 0; g < GMX; ++g)
            if (g < G)
                lmax[g] = fmaxf(lmax[g], shp[(long)g * sh_sg + p]);
    }
    #pragma unroll
    for (int g = 0; g < GMX; ++g) {
        #pragma unroll
        for (int o = 16; o >= 1; o >>= 1)
            lmax[g] = fmaxf(lmax[g], __shfl_xor_sync(~0u, lmax[g], o));
        if (lane == 0) wmax[wid][g] = lmax[g];
    }
    __syncthreads();
    if (wid == 0 && lane < GMX) {
        float m = -CUDART_INF_F;
        for (int w = 0; w < nwarp; ++w) m = fmaxf(m, wmax[w][lane]);
        hmax_sh[lane] = m;
    }
    __syncthreads();

    // ---- phase 2: score[p] = sum_g exp(S[g,p] - hmax[g])  (L2-hot reread) - //
    float hm[GMX];
    #pragma unroll
    for (int g = 0; g < GMX; ++g) hm[g] = hmax_sh[g];
    float* scp = score + (long)req * sc_sr + (long)kh * sc_sh;
    for (int p = tid; p < n; p += BLK) {
        float acc = 0.f;
        #pragma unroll
        for (int g = 0; g < GMX; ++g)
            if (g < G)
                acc += __expf(shp[(long)g * sh_sg + p] - hm[g]);
        scp[p] = acc;
    }
    // release the dependent (topb) as this grid drains
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

void nrm_combine_launch(torch::Tensor score_h, torch::Tensor nsh,
                        torch::Tensor score, int64_t n_req, int64_t n_kv,
                        int64_t G, int64_t blk, int64_t pdl)
{
    TORCH_CHECK(G <= GMX, "nrm_combine: G must be <= 8");
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)n_req, (unsigned)n_kv);
    dim3 block((unsigned)blk);
    if (pdl) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = 0;
        cfg.stream = stream;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        cudaLaunchKernelEx(&cfg, nrm_combine_kernel,
            score_h.data_ptr<float>(), nsh.data_ptr<int>(),
            score.data_ptr<float>(), (int)G,
            (long)score_h.stride(0), (long)score_h.stride(1),
            (long)score_h.stride(2),
            (long)score.stride(0), (long)score.stride(1));
        return;
    }
    nrm_combine_kernel<<<grid, block, 0, stream>>>(
        score_h.data_ptr<float>(), nsh.data_ptr<int>(),
        score.data_ptr<float>(), (int)G,
        (long)score_h.stride(0), (long)score_h.stride(1),
        (long)score_h.stride(2),
        (long)score.stride(0), (long)score.stride(1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nrm_combine_launch", &nrm_combine_launch,
          "AMASS fused nrm GQA-combine (Hopper)");
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


def _zsplit_i8w(n_req: int, n_kv: int, max_pages: int, kh: int = None) -> int:
    """grid.z for the warp-owned-slab kernel (4-warp CTAs, worker = warp,
    grid.y = n_kv/KH).  Heavier workers than the I8 kernel (each pair is a
    KH-head slab): ~1024 warp streams saturates; z <= MP/16 keeps >= ~4 pairs
    per stream so the double buffer stays primed.  Swept on H200
    (scratch_deepopt2/i8f_bench.py I8W section)."""
    if _I8W_ZSPLIT > 0:
        return _I8W_ZSPLIT
    if kh is None:
        kh = _I8W_KH
    # Fitted to the H200 z-sweep (scratch_deepopt2/i8f_bench.json): streams
    # want ~1 pair each at low batch (latency-bound, fill the machine) and
    # ~8 pairs each once n_req*khg covers the SMs (bs>=16).
    per = max(1, n_req * max(1, n_kv // kh))
    pairs = max(1, max_pages // 2)
    div = min(8, max(1, per // 8))
    z = (pairs + div * 4 - 1) // (div * 4)
    return max(4, min(256, z))


def _get_i8w(kh: int = None, v_bits: int = 8, clse: bool = False,
             c_bits: int = 4, c_grain: str = "token"):
    """Build (once, hash-cached) the warp-owned-slab int8 tensor-core kernel.
    ``v_bits=4`` compiles the QI8W_V4 variant (packed-nibble V, unpacked into
    the SAME int8 mma B fragment in-kernel; mu stays int8).  ``clse=True``
    compiles the QI8W_CLSE variant (per-key coord logsumexp tail + iso-resid;
    ``c_bits``/``c_grain`` select the coord quant/scale-grain sub-variants)."""
    if kh is None:
        kh = _I8W_KH
    key = ("i8w", _I8W_WARPS, _I8W_NSTAGE, kh, int(v_bits),
           bool(clse), int(c_bits), c_grain)
    if key in _MODS:
        return _MODS[key]
    from torch.utils.cpp_extension import load_inline
    flags = ["-O3", "--use_fast_math",
             f"-DQI8W_WARPS={_I8W_WARPS}", f"-DQI8W_NSTAGE={_I8W_NSTAGE}",
             f"-DQI8W_KH={kh}",
             "-gencode=arch=compute_90a,code=sm_90a"]
    suffix = f"w{_I8W_WARPS}s{_I8W_NSTAGE}k{kh}"
    if v_bits == 4:
        flags.append("-DQI8W_V4=1")
        suffix += "v4"
    if clse:
        flags += ["-DQI8W_CLSE=1", f"-DQI8W_CB={int(c_bits)}",
                  f"-DQI8W_CTOK={1 if c_grain == 'token' else 0}"]
        suffix += f"_clse{int(c_bits)}{'t' if c_grain == 'token' else 'p'}"
        if _I8W_CWAVE4:
            flags.append("-DQI8W_CWAVE4=1")
            suffix += "w4"
        if _I8W_CSTAGE:
            flags.append("-DQI8W_CSTAGE=1")
            suffix += "cs"
        if _I8W_CMAXRREG > 0:
            flags.append(f"-Xptxas=-maxrregcount={_I8W_CMAXRREG}")
            suffix += f"r{_I8W_CMAXRREG}"
    _verbose = os.environ.get("QUAD_PTXAS_V", "0") != "0"
    if _verbose:
        flags.append("-Xptxas=-v")
        suffix += "_v"
    mod = load_inline(
        name=f"amass_quad_score_i8w_{suffix}",
        cpp_sources="", cuda_sources=_CUDA_SRC_I8W,
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


_NRM_CUDA = None   # latch: None untried, module when built, False = fallback


def _get_nrm():
    global _NRM_CUDA
    if _NRM_CUDA is None:
        from torch.utils.cpp_extension import load_inline
        _NRM_CUDA = load_inline(
            name="amass_nrm_combine_cuda",
            cpp_sources="", cuda_sources=_CUDA_SRC_NRM,
            extra_cuda_cflags=["-O3", "--use_fast_math",
                               "-gencode=arch=compute_90a,code=sm_90a"],
            verbose=False)
    return _NRM_CUDA


def _nrm_from_cuda(st, n_req: int) -> None:
    """Reduce the per-head scores the CUDA score kernel wrote into
    ``st.score_h`` into ``st.score``.  Runs the FUSED single-kernel CUDA
    combine (phase-1 per-head max brings the row into L2, phase-2 exp-sum
    re-reads it hot; PDL edges to the score kernel and topb) and latches to
    the Triton two-pass reference if the build fails."""
    global _NRM_CUDA
    if _NRM_CUDA is not False:
        try:
            mod = _get_nrm()
            # BLK=1024 measured best at every cell (bs1/16K 5.7->5.2us,
            # bs16/16K 6.1->5.7, 64K unchanged): more parallel loads shorten
            # the phase-1 latency chains; 128/256 leave them exposed.
            blk = int(os.environ.get("AMASS_NRM_BLK", "1024"))
            pdl = int(os.environ.get("AMASS_PDL", "1"))
            mod.nrm_combine_launch(st.score_h, st.n_sel_hi, st.score,
                                   int(n_req), int(st.n_kv), int(st.G),
                                   blk, pdl)
            return
        except Exception as e:  # noqa: BLE001
            _NRM_CUDA = False
            print(f"[amass] CUDA nrm combine unavailable "
                  f"({type(e).__name__}: {e}) -> Triton passes", flush=True)
    from .quad_score import _nrm_launch
    _nrm_launch(st, n_req)


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
    impl = os.environ.get("QUAD_CUDA_IMPL", "i8w")
    # GQA combine.  combine="nrm" (default): the kernel writes the per-head score
    # into st.score_h, then the shared Triton nrm passes reduce it into st.score
    # (bitwise-identical combine to the Triton quad reference).  combine="max":
    # the kernel group-maxes in-warp and writes st.score directly (score_h unused,
    # a dummy pointer + write_h=0).
    nrm = getattr(st, "combine", "max") == "nrm"
    score_h = st.score_h if nrm else st.score
    write_h = 1 if nrm else 0
    if rp == 2 and impl != "f32":
        # V last dim: r' (int8) or r'/2 (int4 packed nibbles along r')
        v_bits = 8 if V.shape[-1] == rp else 4
        n_kv, G = int(st.n_kv), int(st.G)
        # warp-owned-SLAB kernel (i8w, the default): each warp reads each of
        # its pages' codes as contiguous KH-head slabs (grid.y = n_kv/KH) and
        # loops the KH heads -> KHx coarser gather (the measured wall;
        # scratch_deepopt2).  Covers BOTH v_bits (int4 = the QI8W_V4 compile
        # variant, so int4 keeps the fast kernel).  Guards: r'=2, n_kv
        # divisible by KH, contiguous per-block slabs (V slab = 128*r' B/head
        # int8, halved for packed int4).
        vhb = 128 * rp if v_bits == 8 else 64 * rp   # V bytes per (block, head)
        if (impl == "i8w" and n_kv % _I8W_KH == 0
                and G <= 8
                and V.stride(1) == vhb and mu.stride(1) == 128):
            mod = _get_i8w(v_bits=v_bits)
            mod.qi8w_score_launch(
                q, mu, mu_s, V, V_s, sig2,
                mu, mu_s, mu_s,          # cc/cs/rs unused in quad builds
                block_table, st.n_sel_hi, st.score, score_h,
                float(scale), 0.0, int(n_req), n_kv, G,
                int(st.max_pages),
                int(_zsplit_i8w(int(n_req), n_kv, int(st.max_pages))),
                int(write_h))
            if nrm:
                _nrm_from_cuda(st, int(n_req))
            return
        mod = _get_i8(v_bits)
        mod.qi8_score_launch(
            q, mu, mu_s, V, V_s, sig2,
            block_table, st.n_sel_hi, st.score, score_h,
            float(scale), int(n_req), int(st.n_kv), int(st.G),
            int(st.max_pages),
            int(_zsplit_i8(int(n_req), int(st.n_kv), int(st.max_pages))),
            int(write_h))
        if nrm:
            _nrm_from_cuda(st, int(n_req))
        return
    assert V.shape[-1] == rp, (
        "the f32 quad_score_cuda supports int8 V only (V last dim must == r'); "
        "int4 V needs the r'=2 i8 kernel or the Triton quad_score reference")
    mod = _get(rp)
    mod.quad_score_launch(
        q, mu, mu_s, V, V_s, sig2,
        block_table, st.n_sel_hi, st.score, score_h,
        float(scale), int(n_req), int(st.n_kv), int(st.G), int(st.max_pages),
        int(_zsplit(int(n_req), int(st.n_kv))), int(write_h))
    if nrm:
        _nrm_from_cuda(st, int(n_req))


def clse_score_cuda(st, layer: int, q, block_table, seq_lens, n_req: int,
                    scale: float) -> None:
    """CUDA drop-in for the Triton ``clse_score`` (same contract): writes
    ``st.score`` (R, n_kv, MP) fp32 over the selectable region, with the
    state's GQA combine (max in-kernel | nrm via the shared Triton passes).

    Runs the QI8W_CLSE compile variant of the warp-owned-slab tensor-core
    kernel: the quad mma front end (2-limb int8 q; B cols [V0 V1 mu 0]) plus a
    per-key coord logsumexp tail (the 16 tokens split across the (even, odd)
    lane pair, magic-constant nibble->fp decode) and the iso-residual term.
    Covers r'=2, int8/int4 V, int8/int4 coords, token/page-grain coord scales,
    n_kv %% KH == 0, G <= 8 -- the production CLSE configs.  Anything else
    must stay on the Triton reference (the caller latches the fallback).
    """
    mu, mu_s, V, V_s, sig2 = _quad_layer_state(st, layer)
    c, c_s, resid = st.quad_c[layer], st.c_scale[layer], st.quad_resid[layer]
    rp = int(getattr(st, "rp", None) or getattr(st, "r", 2))
    n_kv, G = int(st.n_kv), int(st.G)
    page, d = int(st.page), int(st.d)
    v_bits = 8 if V.shape[-1] == rp else 4
    c_bits = int(getattr(st, "c_bits", 4))
    c_grain = getattr(st, "c_grain", "token")
    vhb = 128 * rp if v_bits == 8 else 64 * rp
    cbh = page * rp if c_bits == 8 else page * rp // 2
    cs_hs = page if c_grain == "token" else 1
    if not (rp == 2 and page == 16 and n_kv % _I8W_KH == 0 and G <= 8
            and V.stride(1) == vhb and mu.stride(1) == 128
            and c.stride(1) == cbh and c_s.stride(1) == cs_hs
            and resid.stride(1) == 1):
        raise RuntimeError(
            f"clse_score_cuda: unsupported geometry (r'={rp} page={page} "
            f"n_kv={n_kv} G={G}) -- use the Triton clse_score")
    nrm = getattr(st, "combine", "max") == "nrm"
    score_h = st.score_h if nrm else st.score
    write_h = 1 if nrm else 0
    iso_coef = float(scale) * float(scale) / (2.0 * page * max(d - rp, 1))
    mod = _get_i8w(v_bits=v_bits, clse=True, c_bits=c_bits, c_grain=c_grain)
    mod.qi8w_score_launch(
        q, mu, mu_s, V, V_s, sig2, c, c_s, resid,
        block_table, st.n_sel_hi, st.score, score_h,
        float(scale), iso_coef, int(n_req), n_kv, G,
        int(st.max_pages),
        int(_zsplit_i8w(int(n_req), n_kv, int(st.max_pages))),
        int(write_h))
    if nrm:
        _nrm_from_cuda(st, int(n_req))
