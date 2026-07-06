"""Stage B, hand-CUDA split-K sparse paged decode (Hopper sm_90a, FA3-style).

Drop-in CUDA replacement for the Triton ``sparse_paged_decode_batched`` in
``attention/decode.py``: identical signature/semantics. TWO code paths behind one
drop-in wrapper (``sparse_paged_decode_batched_cuda``), dispatched on batch:

  * split-K (``amass_decode_split`` + Triton ``merge_splits_kernel``): grid
    (split, n_kv, n_req), the low-batch default. Deep cp.async ring, many CTAs
    hide HBM latency; writes ``st.m_part/l_part/acc_part`` partials, a second
    launch recombines them.
  * FUSED / merge-FOLDED (``amass_decode_fused``): grid (n_kv, n_req), ONE
    launch, no ``acc_part`` HBM round-trip and NO separate merge -- the
    cross-split flash-combine is folded into the decode epilogue THROUGH SHARED
    MEMORY. Each CTA owns one (req,kv-head); NPG=WARPS_F/4 page-GROUPS (4 D-split
    warps each) walk disjoint page stripes {pg, pg+NPG, ...} and their partials
    are recombined in smem. The page-group layout keeps the split kernel's low
    register pressure (acc[4][4]) + 128-thread coalesced page load while raising
    warps/CTA to the occupancy the long intra-CTA page loop needs. Chosen when
    ``n_req >= AMASS_CUDA_FUSED_MINREQ`` (default 8): at high batch n_kv*n_req
    already fills the machine, so folding the merge + doubling in-flight page
    parallelism wins. Measured (H200, ~10% pages) fused vs split+merge:
    bs16/16K 96->62us (24%->37% roofline, 1.54x), bs16/32K 155->111 (29%->41%),
    bs16/64K 273->210 (33%->43%), bs32/16K 173->120 (1.45x). At bs1/bs4 the
    fused CTA count is too small to hide HBM latency over the long serial page
    loop, so split-K stays the default there. WARPS_F swept -> 16 (NPG=4) wins.

Both paths are bitwise-close to the Triton golden (max|Δ| <= 2.5e-4) and to fp32
dense (all-pages / restricted top-b); both CUDA-graph capture/replay (alloc-free,
host-sync-free -- verified).

VARIABLE per-unit page counts: every path here is DATA-DEPENDENT on
``st.page_cnt[r, kh]`` -- the split kernel derives
``pps = max(cdiv(cnt, split), pps_min)`` and its loop bounds per (req, kv-head),
the fused kernel walks ``nwaves = cdiv(cnt, NPG)``, and the merge recomputes the
active-split count from the same ``cnt`` -- with a FIXED allocation (page_table
rows are MP wide, the per-unit cap). So a selection where counts vary
per (req, kv-head), from 0 selectable pages up to all of them, is correct and
CUDA-graph safe with NO kernel change (loop bounds are device-side values; the
launch shapes never change). Gated by the skewed-count kernel test. Work across
CTAs becomes count-imbalanced under skew, but the total pages read is conserved
(= the matched total budget), so throughput at occupancy is unchanged; split-K
already subdivides any oversized unit into ~pps-page chunks.

Kernel math == the Triton golden reference (online-softmax flash decode over the
Stage-A-selected pages):
  * K always resident (engine K half), read from the paged cache by physical
    block id -- NO gather.
  * V through the compile-time ``V_SRC`` seam (AMASS_DESIGN 8.1). V_SRC==0 =
    resident engine V half (implemented). V_SRC==1 = mem tier: a documented
    ``#if V_SRC==1`` hook that today falls back to the resident-shaped load, so
    the tier slots in later without touching this call site.

Hopper features used:
  * bf16 tensor cores (``mma.sync.aligned.m16n8k16``) for QK^T; fp16 tensor
    cores for P.V -- matching the Triton bf16-QK / fp16-PV dtype split so the
    numerics stay bitwise-close (max |Δ| vs Triton <=2.5e-4, ==0 at matched
    split-K). M=16 (GQA group padded) maps to the mma m16 tile; the N=16 page
    score becomes the K=16 contraction of the P.V mma with NO cross-thread
    repack (the m16n8k16 C-fragment layout for two n8 tiles is exactly the
    A-fragment layout of the next mma).
  * ``cp.async`` (``cp.async.ca.shared.global``) multi-stage K/V prefetch ring
    (NSTAGE deep): the next page's K AND V DMA overlaps this page's mma+softmax.
    The latency-hiding lever -- both kernels are latency/occupancy bound at the
    decode shapes, not HBM-bandwidth bound (~1.1 TB/s, ~23% of peak).
  * WARP SPECIALIZATION: one CTA = one (req, kv-head, split); WARPS=4 warps
    share ONE cp.async ring (128-thread cooperative loads = 4x load MLP), then
    split the P.V output D=128 into 4 disjoint 32-channel slices so all 4 SM
    sub-partitions run tensor cores in parallel -- 4x warps/SM at the same smem
    footprint as 1 warp/CTA. QK^T runs redundantly per warp (cheap, shared K).
  * smem row padded to 136 (vs 128) so mma fragment gathers are bank-conflict
    free; Q A-fragments hoisted to registers (constant across pages).
  * fixed launch shapes, host-sync-free, allocation-free, +
    cudaFuncSetAttribute opt-in for >48KB smem -> FULL CUDA-graph capture/replay.

Hopper-feature assessment (why NOT wgmma / TMA here): this is a GQA-DECODE tile
-- 1 query token x G<=8 heads, padded to M=16. ``wgmma`` needs M=64 (a warp
GROUP), so it would run at <=25% utilisation on 4x-padded rows: ``mma.sync``
m16n8k16 is the CORRECT tensor-core primitive for M=16, not a shortcut. TMA
(``cp.async.bulk.tensor``) helps large contiguous tiles issued from few
instructions; here the loads are already ``cp.async`` and the kernel is
LATENCY/OCCUPANCY bound (see the fused floor at low batch), not load-ISSUE bound,
so TMA's win over cp.async is marginal and it fights the per-page ``page_table``
indirection + graph capture. The levers that actually moved roofline were
(1) folding the merge (kills the acc_part round-trip) and (2) the page-group
occupancy fix -- both landed above. A cross-warp QK split, a hand-CUDA split-K
merge, and a full-D-per-warp fused layout were tried and LOST (barrier /
occupancy / register cost); a bf16 P.V was neutral (behind ``-DPV_BF16``).

If the extension is unavailable / a layout surprise appears, the public wrapper
raises (callers that want a fallback should catch and call the Triton entry).
"""
from __future__ import annotations

import os

import torch
import triton

from .decode import _split_kv  # reuse the golden helper
from .merge import merge_splits_kernel   # reuse the golden merge (bitwise)

# --------------------------------------------------------------------------- #
# CUDA source                                                                 #
# --------------------------------------------------------------------------- #
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define PAGE 16
#define DMAX 128
// Padded smem row stride: 128 is a multiple of the 32 smem banks, so a
// row-major [token][ch] tile makes every mma fragment gather (rows differing by
// 128 elems) collide on the same bank (K 8-way, V 4-way conflicts). Pad the row
// to 136 (128 + 8, keeps 16B cp.async alignment) so consecutive rows step 4
// banks apart -> K conflict-free, V 4-way -> 2-way.
#define SSTRIDE 136
#ifndef NSTAGE
#define NSTAGE 3          // cp.async ring depth (compile-time tunable)
#endif

// ---- mma.sync m16n8k16 wrappers ----------------------------------------- //
__device__ __forceinline__ void mma_bf16(float d[4], const unsigned a[4],
                                          const unsigned b[2]) {
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_f16(float d[4], const unsigned a[4],
                                         const unsigned b[2]) {
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return *reinterpret_cast<const unsigned*>(p);
}

// ---- V_SRC==1: the mem tier's 3-way ptr-select (hot | staged | pinned) ---- //
// Mirrors tier/decode_mem.py::_mem_decode_split_kernel EXACTLY: ONE scalar
// source decision per (page, kv-head) -- complete+resident -> hot buffer,
// vbo>=0 -> staging pool, else pinned host pool over UVA -- with int16
// pure-bit addressing (the cp.async copies bytes; bf16 is recovered by the
// smem read), so the tiered bytes are IDENTICAL to the resident bytes by the
// tier's invariants.  K follows the same select under k_tier (mem-kv);
// mem-v keeps K resident.
struct TierArgs {
  const int16_t* hotv; const int16_t* stgv; const int16_t* poolv;
  const int16_t* hotk; const int16_t* stgk; const int16_t* poolk;
  const int* p2s; const int* vbo;
  long svb, svt, svh, NB, S;
  int lidx, k_tier;
};
struct TieredPage {
  const __nv_bfloat16* v; long vt;
  const __nv_bfloat16* k; long kt;
};
__device__ __forceinline__ TieredPage tier_page(
    const TierArgs& ta, int pt, long blk, int kh, int seq_len, int n_kv) {
  const bool complete = ((long)(pt + 1) * PAGE) <= (long)seq_len;
  const int slot = ta.p2s[(long)kh * ta.NB + blk];
  const int vbs  = ta.vbo[blk];
  const bool use_hot = complete && (slot >= 0);
  const bool use_stg = (!use_hot) && (vbs >= 0);
  const long hot_off = ((long)kh * ta.S + (use_hot ? slot : 0)) * (PAGE * DMAX);
  const long stg_off = (long)(use_stg ? vbs : 0) * ta.svb + (long)kh * ta.svh;
  const long pin_off = (((long)ta.lidx * ta.NB + blk) * n_kv + kh)
                       * (PAGE * DMAX);
  TieredPage tp;
  tp.v = reinterpret_cast<const __nv_bfloat16*>(
      use_hot ? ta.hotv + hot_off
              : use_stg ? ta.stgv + stg_off : ta.poolv + pin_off);
  tp.vt = use_hot ? DMAX : use_stg ? ta.svt : DMAX;
  if (ta.k_tier) {
    tp.k = reinterpret_cast<const __nv_bfloat16*>(
        use_hot ? ta.hotk + hot_off
                : use_stg ? ta.stgk + stg_off : ta.poolk + pin_off);
    tp.kt = tp.vt;
  } else { tp.k = nullptr; tp.kt = 0; }
  return tp;
}

// ---- tier-pack decode: 15 int64s -> TierArgs (see the Python wrappers) --- //
static TierArgs make_tier_args(const std::vector<int64_t>& tp) {
  TierArgs ta;
  if (tp.size() < 15) {          // V_SRC==0: dead args, never dereferenced
    ta.hotv = ta.stgv = ta.poolv = ta.hotk = ta.stgk = ta.poolk = nullptr;
    ta.p2s = ta.vbo = nullptr;
    ta.svb = ta.svt = ta.svh = ta.NB = ta.S = 0;
    ta.lidx = ta.k_tier = 0;
    return ta;
  }
  ta.hotv  = reinterpret_cast<const int16_t*>(tp[0]);
  ta.stgv  = reinterpret_cast<const int16_t*>(tp[1]);
  ta.poolv = reinterpret_cast<const int16_t*>(tp[2]);
  ta.hotk  = reinterpret_cast<const int16_t*>(tp[3]);
  ta.stgk  = reinterpret_cast<const int16_t*>(tp[4]);
  ta.poolk = reinterpret_cast<const int16_t*>(tp[5]);
  ta.p2s   = reinterpret_cast<const int*>(tp[6]);
  ta.vbo   = reinterpret_cast<const int*>(tp[7]);
  ta.svb = tp[8]; ta.svt = tp[9]; ta.svh = tp[10];
  ta.NB = tp[11]; ta.S = tp[12];
  ta.lidx = (int)tp[13]; ta.k_tier = (int)tp[14];
  return ta;
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
               "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;\n");
}
template <int N> __device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

#ifndef WARPS
#define WARPS 4          // warps per CTA; PV D-dim is split WARPS ways
#endif
#define DPW (DMAX / WARPS)          // 32 output channels / warp
#define DDPW (DPW / 8)              // 4 n8 tiles / warp

// One CTA = one (req, kv-head, split); WARPS warps share ONE cp.async K/V ring.
// The 4 warps redundantly run QK^T + online-softmax (cheap, all read the shared
// K tile), then split the P.V output D=128 into WARPS disjoint slices (warp w
// owns channels [w*32, w*32+32)) so all 4 SM sub-partitions run tensor cores in
// parallel -- 4x the warps/SM at the same shared-memory footprint as 1 warp/CTA.
// K bf16 resident; V through the V_SRC seam. Grid (split, n_kv, n_req).
template <int V_SRC>
__global__ __launch_bounds__(WARPS * 32) void amass_decode_split(
    const __nv_bfloat16* __restrict__ q,     // (T,H,d)
    const __nv_bfloat16* __restrict__ k,     // (nb,page,n_kv,d)
    const __nv_bfloat16* __restrict__ v,     // resident V half (V_SRC==0)
    const int* __restrict__ bt,              // (R, btr)
    const int* __restrict__ tab,             // (R, n_kv, MP)
    const int* __restrict__ cnt,             // (R, n_kv)
    const int* __restrict__ sl,              // (R,)
    float* __restrict__ m_part, float* __restrict__ l_part,
    float* __restrict__ acc_part,
    const float scale, const int n_kv, const int G, const int split,
    const int pps_min,
    const long q_st, const long q_sh,
    const long kb, const long kt, const long kh_s,
    const long vb, const long vt, const long vh_s,
    const long btr, const long tabr, const long tabh,
    const long cntr, const long stride_pr, const TierArgs ta) {
  const int sp = blockIdx.x;
  const int kh = blockIdx.y;
  const int r  = blockIdx.z;
  const int tid  = threadIdx.x;            // 0..127
  const int warp = tid >> 5;               // 0..WARPS-1 (owns D slice)
  const int lane = tid & 31;               // 0..31
  const int gid  = lane >> 2;              // 0..7  (mma row group)
  const int t4   = lane & 3;               // 0..3  (mma col/thread-in-group)

  const int c = cnt[(long)r * cntr + kh];  // selected pages for (r,kh)
  int pps = (c + split - 1) / split;
  if (pps < pps_min) pps = pps_min;
  const int j0 = sp * pps;
  if (j0 >= c) return;                     // inactive split: no store
  const int j1 = min(j0 + pps, c);
  const int seq_len = sl[r];

  // ---- shared memory: Q (bf16 16x128) + cp.async ring of K/V pages -------- //
  // K and V both bf16, non-transposed (Ks[stage][t*128+d], Vs[stage][t*128+d]),
  // filled by cp.async so the next page's K AND V DMA overlap this page's mma.
  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* Qs = smem;                                   // [16*128]
  __nv_bfloat16* Ks = Qs + 16 * DMAX;                         // [NSTAGE*16*SSTRIDE]
  __nv_bfloat16* Vs = Ks + NSTAGE * PAGE * SSTRIDE;           // [NSTAGE*16*SSTRIDE]

  // stage Q (rows >= G zeroed), bf16 straight from the engine q (all threads)
  const __nv_bfloat16* qb = q + (long)r * q_st + (long)kh * G * q_sh;
  const __nv_bfloat16 zbf = __float2bfloat16(0.0f);
  for (int i = tid; i < 16 * DMAX; i += WARPS * 32) {
    int row = i >> 7, col = i & 127;
    Qs[i] = (row < G) ? qb[(long)row * q_sh + col] : zbf;
  }

  // ---- cp.async loader: page slot -> ring stage ((j-j0)%NSTAGE) ----------- //
  // All WARPS*32 threads cooperate on one page tile (4x load MLP vs 1 warp).
  auto load_page = [&](int j) {
    const int stage = (j - j0) % NSTAGE;
    const int pt  = tab[(long)r * tabr + (long)kh * tabh + j];
    const long blk = bt[(long)r * btr + pt];
    __nv_bfloat16* Kd = Ks + (long)stage * PAGE * SSTRIDE;
    __nv_bfloat16* Vd = Vs + (long)stage * PAGE * SSTRIDE;
    const __nv_bfloat16* kg = k + blk * kb + (long)kh * kh_s;
    const __nv_bfloat16* vg = v + blk * vb + (long)kh * vh_s;
    long ktk = kt, vtk = vt;
    if (V_SRC == 1) {
      const TieredPage tp = tier_page(ta, pt, blk, kh, seq_len, n_kv);
      vg = tp.v; vtk = tp.vt;
      if (ta.k_tier) { kg = tp.k; ktk = tp.kt; }
    }
    for (int i = tid; i < PAGE * DMAX / 8; i += WARPS * 32) {
      int t = (i * 8) >> 7, d = (i * 8) & 127;
      cp_async16(&Kd[t * SSTRIDE + d], &kg[(long)t * ktk + d]);
      cp_async16(&Vd[t * SSTRIDE + d], &vg[(long)t * vtk + d]);
    }
  };

  // token base of a page slot (for seq_len masking)
  auto page_tok0 = [&](int j) -> long {
    return (long)tab[(long)r * tabr + (long)kh * tabh + j] * PAGE;
  };

  // ---- prime the cp.async ring -------------------------------------------- //
  int nq = j1 - j0;
  int prol = min(NSTAGE - 1, nq);
  for (int s = 0; s < prol; ++s) { load_page(j0 + s); cp_commit(); }
  for (int s = prol; s < NSTAGE - 1; ++s) cp_commit();   // dummy groups
  __syncthreads();       // Q visible to all warps before the Qf hoist

  // ---- state (each warp owns DPW output channels = DDPW n8 tiles) --------- //
  float m_i[2] = {-INFINITY, -INFINITY};
  float l_i[2] = {0.f, 0.f};
  float acc[DDPW][4];
#pragma unroll
  for (int dd = 0; dd < DDPW; ++dd) { acc[dd][0] = acc[dd][1] = acc[dd][2] = acc[dd][3] = 0.f; }

  // Q A-fragments are CONSTANT across pages -> hoist to registers (removes 32
  // smem loads/page). Each warp runs the FULL QK^T (all 8 d-chunks) redundantly
  // -- cheap, all warps read the shared K tile; a cross-warp QK split was tried
  // and lost (the per-page reduction barrier cost more than the K-read saving).
  const __nv_bfloat16* Qrow0 = Qs + (long)gid * DMAX;
  const __nv_bfloat16* Qrow8 = Qs + (long)(gid + 8) * DMAX;
  unsigned Qf[8][4];
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
    const __nv_bfloat16* q0 = Qrow0 + kk * 16 + t4 * 2;
    const __nv_bfloat16* q8 = Qrow8 + kk * 16 + t4 * 2;
    Qf[kk][0] = smem_u32(q0);      Qf[kk][1] = smem_u32(q8);
    Qf[kk][2] = smem_u32(q0 + 8);  Qf[kk][3] = smem_u32(q8 + 8);
  }

  for (int j = j0; j < j1; ++j) {
    const int st = (j - j0) % NSTAGE;
    cp_wait<NSTAGE - 2>();
    __syncthreads();     // all warps' loads of this stage visible (shared ring)
    const __nv_bfloat16* Kd = Ks + (long)st * PAGE * SSTRIDE;
    const __nv_bfloat16* Vd = Vs + (long)st * PAGE * SSTRIDE;

    // ---- QK^T (bf16 mma): accS0=nn0 (t 0..7), accS1=nn1 (t 8..15) --------- //
    float accS0[4] = {0, 0, 0, 0};
    float accS1[4] = {0, 0, 0, 0};
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      unsigned b0[2], b1[2];
      const __nv_bfloat16* kr0 = Kd + (long)gid * SSTRIDE + kk * 16 + t4 * 2;
      const __nv_bfloat16* kr1 = Kd + (long)(gid + 8) * SSTRIDE + kk * 16 + t4 * 2;
      b0[0] = smem_u32(kr0);     b0[1] = smem_u32(kr0 + 8);
      b1[0] = smem_u32(kr1);     b1[1] = smem_u32(kr1 + 8);
      mma_bf16(accS0, Qf[kk], b0);
      mma_bf16(accS1, Qf[kk], b1);
    }
    // scale + seq_len mask (col t = nn*8 + t4*2 + {0,1})
    const long tok0 = page_tok0(j);
#pragma unroll
    for (int i = 0; i < 4; ++i) { accS0[i] *= scale; accS1[i] *= scale; }
    if (tok0 + t4 * 2 + 0 >= seq_len) { accS0[0] = -INFINITY; accS0[2] = -INFINITY; }
    if (tok0 + t4 * 2 + 1 >= seq_len) { accS0[1] = -INFINITY; accS0[3] = -INFINITY; }
    if (tok0 + 8 + t4 * 2 + 0 >= seq_len) { accS1[0] = -INFINITY; accS1[2] = -INFINITY; }
    if (tok0 + 8 + t4 * 2 + 1 >= seq_len) { accS1[1] = -INFINITY; accS1[3] = -INFINITY; }

    // prefetch the page NSTAGE-1 ahead into its ring slot ((jn-j0)%NSTAGE)
    const int jn = j + (NSTAGE - 1);
    if (jn < j1) load_page(jn);
    cp_commit();

    // ---- online softmax + P.V (fp16 mma), per row rr in {gid, gid+8} ------ //
#ifdef PV_BF16
    unsigned Pab[4];     // P A-fragment (bf16x2) for the bf16 P.V mma
#else
    unsigned Pa[4];      // P A-fragment (fp16x2) for the P.V mma
#endif
    float alpha[2];
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      float s0 = accS0[rr * 2 + 0], s1 = accS0[rr * 2 + 1];
      float s2 = accS1[rr * 2 + 0], s3 = accS1[rr * 2 + 1];
      float lmax = fmaxf(fmaxf(s0, s1), fmaxf(s2, s3));
      lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, 1));
      lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, 2));   // row max
      float a_sc = 1.f; float p0 = 0, p1 = 0, p2 = 0, p3 = 0;
      if (lmax > -INFINITY) {
        float mp = m_i[rr];
        float mn = fmaxf(mp, lmax);
        a_sc = (mp == -INFINITY) ? 0.f : __expf(mp - mn);
        p0 = __expf(s0 - mn); p1 = __expf(s1 - mn);
        p2 = __expf(s2 - mn); p3 = __expf(s3 - mn);
        float ls = p0 + p1 + p2 + p3;
        ls += __shfl_xor_sync(0xffffffffu, ls, 1);
        ls += __shfl_xor_sync(0xffffffffu, ls, 2);                 // row sum
        l_i[rr] = l_i[rr] * a_sc + ls;
        m_i[rr] = mn;
      }
      alpha[rr] = a_sc;
      // pack P (fp16) into the A-fragment slots for this row:
      //   a[rr]      = P[row][t = t4*2 + {0,1}]   (nn0)
      //   a[rr + 2]  = P[row][t = t4*2 + {8,9}]   (nn1)
#ifdef PV_BF16
      __nv_bfloat162 h01 = __floats2bfloat162_rn(p0, p1);
      __nv_bfloat162 h89 = __floats2bfloat162_rn(p2, p3);
      Pab[rr]     = *reinterpret_cast<unsigned*>(&h01);
      Pab[rr + 2] = *reinterpret_cast<unsigned*>(&h89);
#else
      __half2 h01 = __floats2half2_rn(p0, p1);
      __half2 h89 = __floats2half2_rn(p2, p3);
      Pa[rr]     = *reinterpret_cast<unsigned*>(&h01);
      Pa[rr + 2] = *reinterpret_cast<unsigned*>(&h89);
#endif
    }

    // P.V for THIS warp's D slice: global n8 tile dd = warp*DDPW + ddl.
#pragma unroll
    for (int ddl = 0; ddl < DDPW; ++ddl) {
      const int dd = warp * DDPW + ddl;
      acc[ddl][0] *= alpha[0]; acc[ddl][1] *= alpha[0];   // rows gid   (c0,c1)
      acc[ddl][2] *= alpha[1]; acc[ddl][3] *= alpha[1];   // rows gid+8 (c2,c3)
      // V B-fragment: b0={V[t4*2+0][col], V[t4*2+1][col]}, b1={V[t4*2+8],[+9]},
      // col = dd*8+gid; bf16 smem -> fp16 (lossless), pack into half2 regs.
      const int col = dd * 8 + gid;
      const __nv_bfloat16 b0v = Vd[(t4 * 2 + 0) * SSTRIDE + col];
      const __nv_bfloat16 b1v = Vd[(t4 * 2 + 1) * SSTRIDE + col];
      const __nv_bfloat16 b8v = Vd[(t4 * 2 + 8) * SSTRIDE + col];
      const __nv_bfloat16 b9v = Vd[(t4 * 2 + 9) * SSTRIDE + col];
      unsigned vb2[2];
#ifdef PV_BF16
      // bf16 P.V: feed V straight to the tensor core (no bf16->fp16 convert).
      __nv_bfloat162 hb0 = __halves2bfloat162(b0v, b1v);
      __nv_bfloat162 hb1 = __halves2bfloat162(b8v, b9v);
      vb2[0] = *reinterpret_cast<unsigned*>(&hb0);
      vb2[1] = *reinterpret_cast<unsigned*>(&hb1);
      mma_bf16(acc[ddl], Pab, vb2);
#else
      // fp16 P.V (matches Triton's fp16 dot; bf16->fp16 is lossless): convert.
      __half2 hb0 = __halves2half2(__float2half(__bfloat162float(b0v)),
                                   __float2half(__bfloat162float(b1v)));
      __half2 hb1 = __halves2half2(__float2half(__bfloat162float(b8v)),
                                   __float2half(__bfloat162float(b9v)));
      vb2[0] = *reinterpret_cast<unsigned*>(&hb0);
      vb2[1] = *reinterpret_cast<unsigned*>(&hb1);
      mma_f16(acc[ddl], Pa, vb2);
#endif
    }
  }

  // ---- store partials: each warp writes its own D slice; warp 0 writes m/l  //
  const long pbase = (long)r * stride_pr + (long)(kh * split + sp) * G;
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    int row = gid + rr * 8;
    if (row >= G) continue;
    if (warp == 0 && t4 == 0) {
      m_part[pbase + row] = m_i[rr]; l_part[pbase + row] = l_i[rr];
    }
    float* ao = acc_part + (pbase + row) * DMAX;
#pragma unroll
    for (int ddl = 0; ddl < DDPW; ++ddl) {
      const int dd = warp * DDPW + ddl;
      ao[dd * 8 + t4 * 2 + 0] = acc[ddl][rr * 2 + 0];
      ao[dd * 8 + t4 * 2 + 1] = acc[ddl][rr * 2 + 1];
    }
  }
}

// =========================================================================== //
// FUSED decode (merge folded): one CTA = one (req, kv-head), PAGE-STRIPE across
// warps.  Warp w processes pages {w, w+WF, w+2WF, ...} of the (req,kv)'s
// selected set -- so up to WF pages are in flight at once WITHIN the CTA (the
// page parallelism the split-K CTAs gave us, moved inside the block).  Each
// warp keeps a FULL online-softmax state (m,l, acc over ALL D=128) for its
// stripe; at loop exit the per-warp partials are flash-combined THROUGH SHARED
// MEMORY (no HBM acc_part round-trip) and the CTA writes the final output
// directly (no separate merge launch).  Grid (n_kv, n_req).
//
//   * K/V wave ring: all threads cooperatively cp.async a WAVE of WF pages
//     (one page per warp-slot) into a double-buffered smem ring, so warp w's
//     page for wave wv sits at Ksw[buf][w].  cp.async load of wave wv+1 overlaps
//     the mma/softmax of wave wv.
//   * V through the V_SRC seam (constexpr), identical semantics to the split
//     kernel; QK bf16-mma / PV fp16-mma (or bf16 under -DPV_BF16) match Triton.
#ifndef WARPS_F
#define WARPS_F 16          // total warps/CTA = NPG page-groups x 4 D-split warps
#endif
#ifndef FWAVES
#define FWAVES 2            // K/V wave-ring depth (2 = double buffer)
#endif
#define WFT   (WARPS_F * 32)
#define NPG   (WARPS_F / 4)              // page-groups in flight (page parallelism)
#define PGSLOT (PAGE * SSTRIDE)          // one page tile (padded) in bf16
// One CTA = one (req, kv-head).  NPG page-GROUPS each own 4 D-split warps that
// cooperatively load ONE page and run the split kernel's exact inner loop (low
// register acc[4][4], 128-thread coalesced page load).  Group pg walks the page
// stripe {pg, pg+NPG, pg+2NPG, ...}; at loop exit the NPG group partials are
// flash-combined in smem (merge folded, no HBM acc_part round-trip).  Grid
// (n_kv, n_req).  More warps/CTA than the full-D layout -> the occupancy the
// long intra-CTA page loop needs, at low register pressure.
template <int V_SRC>
__global__ __launch_bounds__(WFT) void amass_decode_fused(
    const __nv_bfloat16* __restrict__ q,     // (T,H,d)
    const __nv_bfloat16* __restrict__ k,     // (nb,page,n_kv,d)
    const __nv_bfloat16* __restrict__ v,     // resident V half (V_SRC==0)
    const int* __restrict__ bt,              // (R, btr)
    const int* __restrict__ tab,             // (R, n_kv, MP)
    const int* __restrict__ cnt,             // (R, n_kv)
    const int* __restrict__ sl,              // (R,)
    __nv_bfloat16* __restrict__ out,         // (T,H,d)
    const float scale, const int n_kv, const int G,
    const long q_st, const long q_sh,
    const long kb, const long kt, const long kh_s,
    const long vb, const long vt, const long vh_s,
    const long btr, const long tabr, const long tabh,
    const long cntr, const long stride_ot, const long stride_oh,
    const TierArgs ta) {
  const int kh = blockIdx.x;
  const int r  = blockIdx.y;
  const int tid  = threadIdx.x;            // 0..WFT-1
  const int warp = tid >> 5;               // 0..WARPS_F-1
  const int pg   = warp >> 2;              // 0..NPG-1  (page group / stripe)
  const int dw   = warp & 3;               // 0..3      (D-split warp in group)
  const int lane = tid & 31;
  const int gid  = lane >> 2;              // 0..7  (mma row group)
  const int t4   = lane & 3;               // 0..3

  // ---- shared memory layout --------------------------------------------- //
  // Qs[16*128] | Ks[FWAVES][NPG][16*136] | Vs[FWAVES][NPG][16*136]
  //   | (fp32) accsm[NPG][16*128] | msm[NPG*16] | lsm[NPG*16] | Msm[16] | Lsm[16]
  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* Qs = smem;                                   // [16*128]
  __nv_bfloat16* Ks = Qs + 16 * DMAX;                         // [FWAVES*NPG*PGSLOT]
  __nv_bfloat16* Vs = Ks + FWAVES * NPG * PGSLOT;             // [FWAVES*NPG*PGSLOT]
  float* accsm = reinterpret_cast<float*>(Vs + FWAVES * NPG * PGSLOT);  // [NPG*16*128]
  float* msm   = accsm + NPG * 16 * DMAX;                     // [NPG*16]
  float* lsm   = msm + NPG * 16;                              // [NPG*16]
  float* Msm   = lsm + NPG * 16;                              // [16]
  float* Lsm   = Msm + 16;                                    // [16]

  // stage Q FIRST (selection-independent) so a programmatic dependent launch
  // overlaps it with the tail of topb, then fence before reading topb's
  // page_table/page_cnt (griddepcontrol.wait is a no-op when not armed).
  const __nv_bfloat16* qb = q + (long)r * q_st + (long)kh * G * q_sh;
  const __nv_bfloat16 zbf = __float2bfloat16(0.0f);
  for (int i = tid; i < 16 * DMAX; i += WFT) {
    int row = i >> 7, col = i & 127;
    Qs[i] = (row < G) ? qb[(long)row * q_sh + col] : zbf;
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");

  const int c = cnt[(long)r * cntr + kh];  // selected pages for (r,kh)
  const int seq_len = sl[r];
  const int nwaves = (c + NPG - 1) / NPG;

  // cooperative loader: WAVE wv = NPG pages, one per group, into ring `buf`.
  // group pg loads page (wv*NPG+pg) with its own 128 threads (4 D-warps); a
  // group whose page >= c is idle (its warps skip below).
  auto load_wave = [&](int wv, int buf) {
    const int j = wv * NPG + pg;
    if (j >= c) return;
    const int pt  = tab[(long)r * tabr + (long)kh * tabh + j];
    const long blk = bt[(long)r * btr + pt];
    const __nv_bfloat16* kg = k + blk * kb + (long)kh * kh_s;
    const __nv_bfloat16* vg = v + blk * vb + (long)kh * vh_s;
    long ktk = kt, vtk = vt;
    if (V_SRC == 1) {
      const TieredPage tp = tier_page(ta, pt, blk, kh, seq_len, n_kv);
      vg = tp.v; vtk = tp.vt;
      if (ta.k_tier) { kg = tp.k; ktk = tp.kt; }
    }
    __nv_bfloat16* Kb = Ks + ((long)buf * NPG + pg) * PGSLOT;
    __nv_bfloat16* Vb = Vs + ((long)buf * NPG + pg) * PGSLOT;
    const int gtid = dw * 32 + lane;         // 0..127 within the group
    for (int i = gtid; i < PAGE * DMAX / 8; i += 128) {
      int t = (i * 8) >> 7, d = (i * 8) & 127;
      cp_async16(&Kb[t * SSTRIDE + d], &kg[(long)t * ktk + d]);
      cp_async16(&Vb[t * SSTRIDE + d], &vg[(long)t * vtk + d]);
    }
  };

  {
    int prol = min(FWAVES - 1, nwaves);
    for (int s = 0; s < prol; ++s) { load_wave(s, s); cp_commit(); }
    for (int s = prol; s < FWAVES - 1; ++s) cp_commit();   // dummy groups
  }
  __syncthreads();     // Q visible

  // ---- per-warp online-softmax state (D-slice of 4 n8 tiles = 32 chans) -- //
  float m_i[2] = {-INFINITY, -INFINITY};
  float l_i[2] = {0.f, 0.f};
  float acc[4][4];
#pragma unroll
  for (int dd = 0; dd < 4; ++dd) { acc[dd][0] = acc[dd][1] = acc[dd][2] = acc[dd][3] = 0.f; }

  // Q A-fragments (constant across pages, identical for every warp)
  const __nv_bfloat16* Qrow0 = Qs + (long)gid * DMAX;
  const __nv_bfloat16* Qrow8 = Qs + (long)(gid + 8) * DMAX;
  unsigned Qf[8][4];
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
    const __nv_bfloat16* q0 = Qrow0 + kk * 16 + t4 * 2;
    const __nv_bfloat16* q8 = Qrow8 + kk * 16 + t4 * 2;
    Qf[kk][0] = smem_u32(q0);      Qf[kk][1] = smem_u32(q8);
    Qf[kk][2] = smem_u32(q0 + 8);  Qf[kk][3] = smem_u32(q8 + 8);
  }

  for (int wv = 0; wv < nwaves; ++wv) {
    const int buf = wv % FWAVES;
    cp_wait<FWAVES - 2>();   // wave wv fully loaded (FWAVES-deep ring)
    __syncthreads();
    // prefetch wave wv+FWAVES-1 into its ring slot (overlaps this wave's mma;
    // that slot's consumers finished at iteration wv-1, fenced by its barrier)
    if (wv + FWAVES - 1 < nwaves)
      load_wave(wv + FWAVES - 1, (wv + FWAVES - 1) % FWAVES);
    cp_commit();

    const int j = wv * NPG + pg;
    if (j < c) {
      const __nv_bfloat16* Kd = Ks + ((long)buf * NPG + pg) * PGSLOT;
      const __nv_bfloat16* Vd = Vs + ((long)buf * NPG + pg) * PGSLOT;

      // ---- QK^T (bf16 mma), full page (all 4 D-warps redundant) --------- //
      float accS0[4] = {0, 0, 0, 0};
      float accS1[4] = {0, 0, 0, 0};
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        unsigned b0[2], b1[2];
        const __nv_bfloat16* kr0 = Kd + (long)gid * SSTRIDE + kk * 16 + t4 * 2;
        const __nv_bfloat16* kr1 = Kd + (long)(gid + 8) * SSTRIDE + kk * 16 + t4 * 2;
        b0[0] = smem_u32(kr0);     b0[1] = smem_u32(kr0 + 8);
        b1[0] = smem_u32(kr1);     b1[1] = smem_u32(kr1 + 8);
        mma_bf16(accS0, Qf[kk], b0);
        mma_bf16(accS1, Qf[kk], b1);
      }
      const long tok0 = (long)tab[(long)r * tabr + (long)kh * tabh + j] * PAGE;
#pragma unroll
      for (int i = 0; i < 4; ++i) { accS0[i] *= scale; accS1[i] *= scale; }
      if (tok0 + t4 * 2 + 0 >= seq_len) { accS0[0] = -INFINITY; accS0[2] = -INFINITY; }
      if (tok0 + t4 * 2 + 1 >= seq_len) { accS0[1] = -INFINITY; accS0[3] = -INFINITY; }
      if (tok0 + 8 + t4 * 2 + 0 >= seq_len) { accS1[0] = -INFINITY; accS1[2] = -INFINITY; }
      if (tok0 + 8 + t4 * 2 + 1 >= seq_len) { accS1[1] = -INFINITY; accS1[3] = -INFINITY; }

      // ---- online softmax + P.V (this warp's 4 n8 D-tiles) -------------- //
#ifdef PV_BF16
      unsigned Pab[4];
#else
      unsigned Pa[4];
#endif
      float alpha[2];
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        float s0 = accS0[rr * 2 + 0], s1 = accS0[rr * 2 + 1];
        float s2 = accS1[rr * 2 + 0], s3 = accS1[rr * 2 + 1];
        float lmax = fmaxf(fmaxf(s0, s1), fmaxf(s2, s3));
        lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, 1));
        lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, 2));
        float a_sc = 1.f; float p0 = 0, p1 = 0, p2 = 0, p3 = 0;
        if (lmax > -INFINITY) {
          float mp = m_i[rr];
          float mn = fmaxf(mp, lmax);
          a_sc = (mp == -INFINITY) ? 0.f : __expf(mp - mn);
          p0 = __expf(s0 - mn); p1 = __expf(s1 - mn);
          p2 = __expf(s2 - mn); p3 = __expf(s3 - mn);
          float ls = p0 + p1 + p2 + p3;
          ls += __shfl_xor_sync(0xffffffffu, ls, 1);
          ls += __shfl_xor_sync(0xffffffffu, ls, 2);
          l_i[rr] = l_i[rr] * a_sc + ls;
          m_i[rr] = mn;
        }
        alpha[rr] = a_sc;
#ifdef PV_BF16
        __nv_bfloat162 h01 = __floats2bfloat162_rn(p0, p1);
        __nv_bfloat162 h89 = __floats2bfloat162_rn(p2, p3);
        Pab[rr]     = *reinterpret_cast<unsigned*>(&h01);
        Pab[rr + 2] = *reinterpret_cast<unsigned*>(&h89);
#else
        __half2 h01 = __floats2half2_rn(p0, p1);
        __half2 h89 = __floats2half2_rn(p2, p3);
        Pa[rr]     = *reinterpret_cast<unsigned*>(&h01);
        Pa[rr + 2] = *reinterpret_cast<unsigned*>(&h89);
#endif
      }
      // P.V for THIS warp's D slice: global tile dd = dw*4 + ddl.
#pragma unroll
      for (int ddl = 0; ddl < 4; ++ddl) {
        const int dd = dw * 4 + ddl;
        acc[ddl][0] *= alpha[0]; acc[ddl][1] *= alpha[0];
        acc[ddl][2] *= alpha[1]; acc[ddl][3] *= alpha[1];
        const int col = dd * 8 + gid;
        const __nv_bfloat16 b0v = Vd[(t4 * 2 + 0) * SSTRIDE + col];
        const __nv_bfloat16 b1v = Vd[(t4 * 2 + 1) * SSTRIDE + col];
        const __nv_bfloat16 b8v = Vd[(t4 * 2 + 8) * SSTRIDE + col];
        const __nv_bfloat16 b9v = Vd[(t4 * 2 + 9) * SSTRIDE + col];
        unsigned vb2[2];
#ifdef PV_BF16
        __nv_bfloat162 hb0 = __halves2bfloat162(b0v, b1v);
        __nv_bfloat162 hb1 = __halves2bfloat162(b8v, b9v);
        vb2[0] = *reinterpret_cast<unsigned*>(&hb0);
        vb2[1] = *reinterpret_cast<unsigned*>(&hb1);
        mma_bf16(acc[ddl], Pab, vb2);
#else
        __half2 hb0 = __halves2half2(__float2half(__bfloat162float(b0v)),
                                     __float2half(__bfloat162float(b1v)));
        __half2 hb1 = __halves2half2(__float2half(__bfloat162float(b8v)),
                                     __float2half(__bfloat162float(b9v)));
        vb2[0] = *reinterpret_cast<unsigned*>(&hb0);
        vb2[1] = *reinterpret_cast<unsigned*>(&hb1);
        mma_f16(acc[ddl], Pa, vb2);
#endif
      }
    }
    __syncthreads();       // all warps done reading buf before it is reused
  }

  // ---- fold the merge: flash-combine the NPG group partials in smem ------ //
  // each warp dumps its D-slice; (dw==0) writes the group's m/l once.
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int row = gid + rr * 8;
    float* asl = accsm + ((long)pg * 16 + row) * DMAX;
#pragma unroll
    for (int ddl = 0; ddl < 4; ++ddl) {
      const int dd = dw * 4 + ddl;
      asl[dd * 8 + t4 * 2 + 0] = acc[ddl][rr * 2 + 0];
      asl[dd * 8 + t4 * 2 + 1] = acc[ddl][rr * 2 + 1];
    }
    if (dw == 0 && t4 == 0) { msm[pg * 16 + row] = m_i[rr]; lsm[pg * 16 + row] = l_i[rr]; }
  }
  __syncthreads();

  // per-row global max + denominator over the NPG groups (16 rows)
  if (tid < 16) {
    const int row = tid;
    float M = -INFINITY;
#pragma unroll 1
    for (int w = 0; w < NPG; ++w)
      if (lsm[w * 16 + row] > 0.f) M = fmaxf(M, msm[w * 16 + row]);
    float L = 0.f;
#pragma unroll 1
    for (int w = 0; w < NPG; ++w) {
      float lw = lsm[w * 16 + row];
      if (lw > 0.f) L += lw * __expf(msm[w * 16 + row] - M);
    }
    Msm[row] = M; Lsm[row] = L;
  }
  __syncthreads();

  // write final output: each thread owns a set of (row,d) cells
  const long obase = (long)r * stride_ot + (long)(kh * G) * stride_oh;
  for (int e = tid; e < 16 * DMAX; e += WFT) {
    const int row = e >> 7, d = e & 127;
    if (row >= G) continue;
    const float M = Msm[row], L = Lsm[row];
    float val = 0.f;
    if (L > 0.f) {
      float s = 0.f;
#pragma unroll 1
      for (int w = 0; w < NPG; ++w) {
        float lw = lsm[w * 16 + row];
        if (lw > 0.f) s += accsm[((long)w * 16 + row) * DMAX + d]
                          * __expf(msm[w * 16 + row] - M);
      }
      val = s / L;
    }
    out[obase + (long)row * stride_oh + d] = __float2bfloat16(val);
  }
}

void amass_decode_fused_launch(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bt,
    torch::Tensor tab, torch::Tensor cnt, torch::Tensor sl, torch::Tensor out,
    double scale, int64_t n_req, int64_t n_kv, int64_t G, int64_t v_src,
    std::vector<int64_t> tier_pack) {
  const TierArgs ta = make_tier_args(tier_pack);
  const int d = 128;
  TORCH_CHECK(k.size(-1) == d, "amass_decode_fused: d must be 128");
  TORCH_CHECK(k.size(1) == PAGE, "amass_decode_fused: page must be 16");
  TORCH_CHECK(G <= 8, "amass_decode_fused: G must be <= 8");
  dim3 grid((unsigned)n_kv, (unsigned)n_req);
  dim3 block(WFT);
  size_t smem_bf16 = (size_t)(16 * DMAX + 2 * FWAVES * NPG * PGSLOT)
                   * sizeof(__nv_bfloat16);
  size_t smem_f32 = (size_t)(NPG * 16 * DMAX + 2 * NPG * 16 + 32)
                  * sizeof(float);
  size_t smem = smem_bf16 + smem_f32;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto kptr = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  auto vptr = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
  auto qptr = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  auto optr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const bool pdl = [] {
    const char* e = getenv("AMASS_PDL");
    return e == nullptr || e[0] != '0';
  }();
  auto run = [&](auto kern) {
    if (smem > 48 * 1024) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          (const void*)kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
          (int)smem));
    }
    if (pdl) {
      // programmatic dependent launch: overlap the Q-staging prologue with
      // the tail of topb (the kernel fences via griddepcontrol.wait before
      // reading page_table/page_cnt).  Graph-capturable (CUDA >= 12.0).
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = grid;
      cfg.blockDim = block;
      cfg.dynamicSmemBytes = smem;
      cfg.stream = stream;
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[0].val.programmaticStreamSerializationAllowed = 1;
      cfg.attrs = attr;
      cfg.numAttrs = 1;
      cudaLaunchKernelEx(&cfg, kern,
        qptr, kptr, vptr, bt.data_ptr<int>(), tab.data_ptr<int>(),
        cnt.data_ptr<int>(), sl.data_ptr<int>(), optr, (float)scale,
        (int)n_kv, (int)G,
        (long)q.stride(0), (long)q.stride(1),
        (long)k.stride(0), (long)k.stride(1), (long)k.stride(2),
        (long)v.stride(0), (long)v.stride(1), (long)v.stride(2),
        (long)bt.stride(0), (long)tab.stride(0), (long)tab.stride(1),
        (long)cnt.stride(0), (long)out.stride(0), (long)out.stride(1), ta);
      return;
    }
    kern<<<grid, block, smem, stream>>>(
      qptr, kptr, vptr, bt.data_ptr<int>(), tab.data_ptr<int>(),
      cnt.data_ptr<int>(), sl.data_ptr<int>(), optr, (float)scale,
      (int)n_kv, (int)G,
      (long)q.stride(0), (long)q.stride(1),
      (long)k.stride(0), (long)k.stride(1), (long)k.stride(2),
      (long)v.stride(0), (long)v.stride(1), (long)v.stride(2),
      (long)bt.stride(0), (long)tab.stride(0), (long)tab.stride(1),
      (long)cnt.stride(0), (long)out.stride(0), (long)out.stride(1), ta);
  };
  if (v_src == 0) run(amass_decode_fused<0>);
  else            run(amass_decode_fused<1>);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// =========================================================================== //
// SPLIT2: the split-K decode with the merge FOLDED INTO THE SAME LAUNCH via a
// last-CTA-arrives reduction (the canonical CUB pattern).  Each (req,kv,split)
// CTA runs the identical page loop as amass_decode_split and stores its
// m/l/acc partials; it then __threadfence()s and atomicAdd-arrives on a
// per-(req,kv) counter.  The LAST CTA to arrive re-reads the (L2-hot) partials
// of all ACTIVE splits, flash-combines them, writes the final output row, and
// RESETS the counter to zero for the next step (graph-replay safe: capture
// does not execute, every replay starts from 0; n_act may change per step).
// Kills the separate merge launch + its cold acc_part re-read.  c==0 edge:
// the sp==0 CTA zero-fills the output row (matches the merge kernel).
// =========================================================================== //
template <int V_SRC>
__global__ __launch_bounds__(WARPS * 32) void amass_decode_split2(
    const __nv_bfloat16* __restrict__ q,     // (T,H,d)
    const __nv_bfloat16* __restrict__ k,     // (nb,page,n_kv,d)
    const __nv_bfloat16* __restrict__ v,     // resident V half (V_SRC==0)
    const int* __restrict__ bt,              // (R, btr)
    const int* __restrict__ tab,             // (R, n_kv, MP)
    const int* __restrict__ cnt,             // (R, n_kv)
    const int* __restrict__ sl,              // (R,)
    float* __restrict__ m_part, float* __restrict__ l_part,
    float* __restrict__ acc_part,
    int* __restrict__ arrive,                // (R, n_kv) zero-init counters
    __nv_bfloat16* __restrict__ out,         // (T,H,d)
    const float scale, const int n_kv, const int G, const int split,
    const int pps_min,
    const long q_st, const long q_sh,
    const long kb, const long kt, const long kh_s,
    const long vb, const long vt, const long vh_s,
    const long btr, const long tabr, const long tabh,
    const long cntr, const long stride_pr,
    const long stride_ot, const long stride_oh, const TierArgs ta) {
  const int sp = blockIdx.x;
  const int kh = blockIdx.y;
  const int r  = blockIdx.z;
  const int tid  = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int gid  = lane >> 2;
  const int t4   = lane & 3;

  const int c = cnt[(long)r * cntr + kh];
  int pps = (c + split - 1) / split;
  if (pps < pps_min) pps = pps_min;
  const int n_act = (c + pps - 1) / pps;     // active splits (0 if c==0)
  if (c == 0) {                              // empty selection: zero output
    if (sp == 0) {
      const long obase = (long)r * stride_ot + (long)(kh * G) * stride_oh;
      for (int i = tid; i < G * DMAX; i += WARPS * 32) {
        int row = i >> 7, d = i & 127;
        out[obase + (long)row * stride_oh + d] = __float2bfloat16(0.f);
      }
    }
    return;
  }
  const int j0 = sp * pps;
  if (j0 >= c) return;                       // inactive split
  const int j1 = min(j0 + pps, c);
  const int seq_len = sl[r];

  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* Qs = smem;
  __nv_bfloat16* Ks = Qs + 16 * DMAX;
  __nv_bfloat16* Vs = Ks + NSTAGE * PAGE * SSTRIDE;

  const __nv_bfloat16* qb = q + (long)r * q_st + (long)kh * G * q_sh;
  const __nv_bfloat16 zbf = __float2bfloat16(0.0f);
  for (int i = tid; i < 16 * DMAX; i += WARPS * 32) {
    int row = i >> 7, col = i & 127;
    Qs[i] = (row < G) ? qb[(long)row * q_sh + col] : zbf;
  }

  auto load_page = [&](int j) {
    const int stage = (j - j0) % NSTAGE;
    const int pt  = tab[(long)r * tabr + (long)kh * tabh + j];
    const long blk = bt[(long)r * btr + pt];
    __nv_bfloat16* Kd = Ks + (long)stage * PAGE * SSTRIDE;
    __nv_bfloat16* Vd = Vs + (long)stage * PAGE * SSTRIDE;
    const __nv_bfloat16* kg = k + blk * kb + (long)kh * kh_s;
    const __nv_bfloat16* vg = v + blk * vb + (long)kh * vh_s;
    long ktk = kt, vtk = vt;
    if (V_SRC == 1) {
      const TieredPage tp = tier_page(ta, pt, blk, kh, seq_len, n_kv);
      vg = tp.v; vtk = tp.vt;
      if (ta.k_tier) { kg = tp.k; ktk = tp.kt; }
    }
    for (int i = tid; i < PAGE * DMAX / 8; i += WARPS * 32) {
      int t = (i * 8) >> 7, d = (i * 8) & 127;
      cp_async16(&Kd[t * SSTRIDE + d], &kg[(long)t * ktk + d]);
      cp_async16(&Vd[t * SSTRIDE + d], &vg[(long)t * vtk + d]);
    }
  };
  auto page_tok0 = [&](int j) -> long {
    return (long)tab[(long)r * tabr + (long)kh * tabh + j] * PAGE;
  };

  int nq = j1 - j0;
  int prol = min(NSTAGE - 1, nq);
  for (int s = 0; s < prol; ++s) { load_page(j0 + s); cp_commit(); }
  for (int s = prol; s < NSTAGE - 1; ++s) cp_commit();
  __syncthreads();

  float m_i[2] = {-INFINITY, -INFINITY};
  float l_i[2] = {0.f, 0.f};
  float acc[DDPW][4];
#pragma unroll
  for (int dd = 0; dd < DDPW; ++dd) { acc[dd][0] = acc[dd][1] = acc[dd][2] = acc[dd][3] = 0.f; }

  const __nv_bfloat16* Qrow0 = Qs + (long)gid * DMAX;
  const __nv_bfloat16* Qrow8 = Qs + (long)(gid + 8) * DMAX;
  unsigned Qf[8][4];
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
    const __nv_bfloat16* q0 = Qrow0 + kk * 16 + t4 * 2;
    const __nv_bfloat16* q8 = Qrow8 + kk * 16 + t4 * 2;
    Qf[kk][0] = smem_u32(q0);      Qf[kk][1] = smem_u32(q8);
    Qf[kk][2] = smem_u32(q0 + 8);  Qf[kk][3] = smem_u32(q8 + 8);
  }

  for (int j = j0; j < j1; ++j) {
    const int st = (j - j0) % NSTAGE;
    cp_wait<NSTAGE - 2>();
    __syncthreads();
    const __nv_bfloat16* Kd = Ks + (long)st * PAGE * SSTRIDE;
    const __nv_bfloat16* Vd = Vs + (long)st * PAGE * SSTRIDE;

    float accS0[4] = {0, 0, 0, 0};
    float accS1[4] = {0, 0, 0, 0};
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      unsigned b0[2], b1[2];
      const __nv_bfloat16* kr0 = Kd + (long)gid * SSTRIDE + kk * 16 + t4 * 2;
      const __nv_bfloat16* kr1 = Kd + (long)(gid + 8) * SSTRIDE + kk * 16 + t4 * 2;
      b0[0] = smem_u32(kr0);     b0[1] = smem_u32(kr0 + 8);
      b1[0] = smem_u32(kr1);     b1[1] = smem_u32(kr1 + 8);
      mma_bf16(accS0, Qf[kk], b0);
      mma_bf16(accS1, Qf[kk], b1);
    }
    const long tok0 = page_tok0(j);
#pragma unroll
    for (int i = 0; i < 4; ++i) { accS0[i] *= scale; accS1[i] *= scale; }
    if (tok0 + t4 * 2 + 0 >= seq_len) { accS0[0] = -INFINITY; accS0[2] = -INFINITY; }
    if (tok0 + t4 * 2 + 1 >= seq_len) { accS0[1] = -INFINITY; accS0[3] = -INFINITY; }
    if (tok0 + 8 + t4 * 2 + 0 >= seq_len) { accS1[0] = -INFINITY; accS1[2] = -INFINITY; }
    if (tok0 + 8 + t4 * 2 + 1 >= seq_len) { accS1[1] = -INFINITY; accS1[3] = -INFINITY; }

    const int jn = j + (NSTAGE - 1);
    if (jn < j1) load_page(jn);
    cp_commit();

#ifdef PV_BF16
    unsigned Pab[4];
#else
    unsigned Pa[4];
#endif
    float alpha[2];
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      float s0 = accS0[rr * 2 + 0], s1 = accS0[rr * 2 + 1];
      float s2 = accS1[rr * 2 + 0], s3 = accS1[rr * 2 + 1];
      float lmax = fmaxf(fmaxf(s0, s1), fmaxf(s2, s3));
      lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, 1));
      lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, 2));
      float a_sc = 1.f; float p0 = 0, p1 = 0, p2 = 0, p3 = 0;
      if (lmax > -INFINITY) {
        float mp = m_i[rr];
        float mn = fmaxf(mp, lmax);
        a_sc = (mp == -INFINITY) ? 0.f : __expf(mp - mn);
        p0 = __expf(s0 - mn); p1 = __expf(s1 - mn);
        p2 = __expf(s2 - mn); p3 = __expf(s3 - mn);
        float ls = p0 + p1 + p2 + p3;
        ls += __shfl_xor_sync(0xffffffffu, ls, 1);
        ls += __shfl_xor_sync(0xffffffffu, ls, 2);
        l_i[rr] = l_i[rr] * a_sc + ls;
        m_i[rr] = mn;
      }
      alpha[rr] = a_sc;
#ifdef PV_BF16
      __nv_bfloat162 h01 = __floats2bfloat162_rn(p0, p1);
      __nv_bfloat162 h89 = __floats2bfloat162_rn(p2, p3);
      Pab[rr]     = *reinterpret_cast<unsigned*>(&h01);
      Pab[rr + 2] = *reinterpret_cast<unsigned*>(&h89);
#else
      __half2 h01 = __floats2half2_rn(p0, p1);
      __half2 h89 = __floats2half2_rn(p2, p3);
      Pa[rr]     = *reinterpret_cast<unsigned*>(&h01);
      Pa[rr + 2] = *reinterpret_cast<unsigned*>(&h89);
#endif
    }

#pragma unroll
    for (int ddl = 0; ddl < DDPW; ++ddl) {
      const int dd = warp * DDPW + ddl;
      acc[ddl][0] *= alpha[0]; acc[ddl][1] *= alpha[0];
      acc[ddl][2] *= alpha[1]; acc[ddl][3] *= alpha[1];
      const int col = dd * 8 + gid;
      const __nv_bfloat16 b0v = Vd[(t4 * 2 + 0) * SSTRIDE + col];
      const __nv_bfloat16 b1v = Vd[(t4 * 2 + 1) * SSTRIDE + col];
      const __nv_bfloat16 b8v = Vd[(t4 * 2 + 8) * SSTRIDE + col];
      const __nv_bfloat16 b9v = Vd[(t4 * 2 + 9) * SSTRIDE + col];
      unsigned vb2[2];
#ifdef PV_BF16
      __nv_bfloat162 hb0 = __halves2bfloat162(b0v, b1v);
      __nv_bfloat162 hb1 = __halves2bfloat162(b8v, b9v);
      vb2[0] = *reinterpret_cast<unsigned*>(&hb0);
      vb2[1] = *reinterpret_cast<unsigned*>(&hb1);
      mma_bf16(acc[ddl], Pab, vb2);
#else
      __half2 hb0 = __halves2half2(__float2half(__bfloat162float(b0v)),
                                   __float2half(__bfloat162float(b1v)));
      __half2 hb1 = __halves2half2(__float2half(__bfloat162float(b8v)),
                                   __float2half(__bfloat162float(b9v)));
      vb2[0] = *reinterpret_cast<unsigned*>(&hb0);
      vb2[1] = *reinterpret_cast<unsigned*>(&hb1);
      mma_f16(acc[ddl], Pa, vb2);
#endif
    }
  }

  // ---- store partials (same layout as amass_decode_split) ---------------- //
  const long pbase = (long)r * stride_pr + (long)(kh * split + sp) * G;
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    int row = gid + rr * 8;
    if (row >= G) continue;
    if (warp == 0 && t4 == 0) {
      m_part[pbase + row] = m_i[rr]; l_part[pbase + row] = l_i[rr];
    }
    float* ao = acc_part + (pbase + row) * DMAX;
#pragma unroll
    for (int ddl = 0; ddl < DDPW; ++ddl) {
      const int dd = warp * DDPW + ddl;
      ao[dd * 8 + t4 * 2 + 0] = acc[ddl][rr * 2 + 0];
      ao[dd * 8 + t4 * 2 + 1] = acc[ddl][rr * 2 + 1];
    }
  }

  // ---- last-CTA merge (folded): arrive, last re-reads + combines --------- //
  __threadfence();                            // partials visible device-wide
  __shared__ int s_last;
  __syncthreads();                            // all warps' stores issued
  if (tid == 0) {
    int old = atomicAdd(arrive + (long)r * n_kv + kh, 1);
    s_last = (old == n_act - 1) ? 1 : 0;
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();                            // acquire: see all partials

  // this CTA merges: one warp per query-head row g (G<=8 <= warps*?); loop.
  for (int g = warp; g < G; g += WARPS) {
    const long base = (long)r * stride_pr + (long)kh * split * G + g;
    float m_max = -INFINITY;
    for (int s = 0; s < n_act; ++s) {
      float m = m_part[base + (long)s * G];
      float l = l_part[base + (long)s * G];
      if (l > 0.f) m_max = fmaxf(m_max, m);
    }
    float l_tot = 0.f;
    float o[4] = {0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < n_act; ++s) {
      const long idx = base + (long)s * G;
      float m = m_part[idx], l = l_part[idx];
      float w = (l > 0.f) ? __expf(m - m_max) : 0.f;
      l_tot += l * w;
      const float* a = acc_part + idx * DMAX;
#pragma unroll
      for (int i = 0; i < 4; ++i) o[i] += a[lane + i * 32] * w;
    }
    const float inv = (l_tot > 0.f) ? 1.f / l_tot : 0.f;
    __nv_bfloat16* op = out + (long)r * stride_ot + (long)(kh * G + g) * stride_oh;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      op[lane + i * 32] = __float2bfloat16(o[i] * inv);
  }
  if (tid == 0) arrive[(long)r * n_kv + kh] = 0;   // reset for the next step
}

void amass_decode_split2_launch(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bt,
    torch::Tensor tab, torch::Tensor cnt, torch::Tensor sl,
    torch::Tensor m_part, torch::Tensor l_part, torch::Tensor acc_part,
    torch::Tensor arrive, torch::Tensor out,
    double scale, int64_t n_req, int64_t n_kv, int64_t G, int64_t split,
    int64_t pps_min, int64_t v_src, std::vector<int64_t> tier_pack) {
  const TierArgs ta = make_tier_args(tier_pack);
  const int d = 128;
  TORCH_CHECK(k.size(-1) == d, "amass_decode_split2: d must be 128");
  TORCH_CHECK(k.size(1) == PAGE, "amass_decode_split2: page must be 16");
  TORCH_CHECK(G <= 8, "amass_decode_split2: G must be <= 8");
  dim3 grid((unsigned)split, (unsigned)n_kv, (unsigned)n_req);
  dim3 block(WARPS * 32);
  size_t smem = (size_t)(16 * DMAX + 2 * NSTAGE * PAGE * SSTRIDE)
              * sizeof(__nv_bfloat16);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto kptr = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  auto vptr = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
  auto qptr = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  auto optr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  auto run = [&](auto kern) {
    if (smem > 48 * 1024) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          (const void*)kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
          (int)smem));
    }
    kern<<<grid, block, smem, stream>>>(
      qptr, kptr, vptr, bt.data_ptr<int>(), tab.data_ptr<int>(),
      cnt.data_ptr<int>(), sl.data_ptr<int>(),
      m_part.data_ptr<float>(), l_part.data_ptr<float>(),
      acc_part.data_ptr<float>(), arrive.data_ptr<int>(), optr,
      (float)scale, (int)n_kv, (int)G,
      (int)split, (int)pps_min,
      (long)q.stride(0), (long)q.stride(1),
      (long)k.stride(0), (long)k.stride(1), (long)k.stride(2),
      (long)v.stride(0), (long)v.stride(1), (long)v.stride(2),
      (long)bt.stride(0), (long)tab.stride(0), (long)tab.stride(1),
      (long)cnt.stride(0), (long)m_part.stride(0),
      (long)out.stride(0), (long)out.stride(1), ta);
  };
  if (v_src == 0) run(amass_decode_split2<0>);
  else            run(amass_decode_split2<1>);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- split-K merge (port of attention/merge.py) ------------------------- //
// One warp per (req, query-head); recombines the ACTIVE splits only (n_act,
// derived on device from page_cnt) with the flash rescale. Leaner than the
// Triton merge, which masks a fixed SPLIT_PAD=128-wide reduction.
__global__ __launch_bounds__(32) void amass_merge(
    const float* __restrict__ m_part, const float* __restrict__ l_part,
    const float* __restrict__ acc_part, const int* __restrict__ cnt,
    __nv_bfloat16* __restrict__ out,
    const int G, const int split, const int pps_min, const int D,
    const long stride_ot, const long stride_oh, const long stride_pr,
    const long cntr) {
  const int r = blockIdx.y;
  const int h = blockIdx.x;
  const int kh = h / G, g = h % G;
  const int lane = threadIdx.x;
  const int c = cnt[(long)r * cntr + kh];
  int pps = (c + split - 1) / split; if (pps < pps_min) pps = pps_min;
  const int n_act = (c + pps - 1) / pps;            // active splits (0 if c==0)

  float m_max = -INFINITY;
  for (int s = 0; s < n_act; ++s) {
    float m = m_part[(long)r * stride_pr + (long)(kh * split + s) * G + g];
    float l = l_part[(long)r * stride_pr + (long)(kh * split + s) * G + g];
    if (l > 0.f) m_max = fmaxf(m_max, m);
  }
  float l_tot = 0.f;
  float o[4] = {0.f, 0.f, 0.f, 0.f};                // D=128 -> 4 chans/lane
  for (int s = 0; s < n_act; ++s) {
    const long idx = (long)r * stride_pr + (long)(kh * split + s) * G + g;
    float m = m_part[idx], l = l_part[idx];
    float w = (l > 0.f) ? __expf(m - m_max) : 0.f;
    l_tot += l * w;
    const float* a = acc_part + idx * D;
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] += a[lane + i * 32] * w;
  }
  const float inv = (l_tot > 0.f) ? 1.f / l_tot : 0.f;
  __nv_bfloat16* op = out + (long)r * stride_ot + (long)h * stride_oh;
#pragma unroll
  for (int i = 0; i < 4; ++i)
    op[lane + i * 32] = __float2bfloat16(o[i] * inv);
}

void amass_merge_launch(torch::Tensor m_part, torch::Tensor l_part,
                        torch::Tensor acc_part, torch::Tensor cnt,
                        torch::Tensor out, int64_t n_req, int64_t n_kv,
                        int64_t G, int64_t split, int64_t pps_min) {
  const int D = 128;
  TORCH_CHECK(out.size(-1) == D, "amass_merge: D must be 128");
  dim3 grid((unsigned)(n_kv * G), (unsigned)n_req);
  auto stream = at::cuda::getCurrentCUDAStream();
  amass_merge<<<grid, 32, 0, stream>>>(
      m_part.data_ptr<float>(), l_part.data_ptr<float>(),
      acc_part.data_ptr<float>(), cnt.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      (int)G, (int)split, (int)pps_min, D,
      (long)out.stride(0), (long)out.stride(1), (long)m_part.stride(0),
      (long)cnt.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- launcher ----------------------------------------------------------- //
void amass_decode_split_launch(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bt,
    torch::Tensor tab, torch::Tensor cnt, torch::Tensor sl,
    torch::Tensor m_part, torch::Tensor l_part, torch::Tensor acc_part,
    double scale, int64_t n_req, int64_t n_kv, int64_t G, int64_t split,
    int64_t pps_min, int64_t v_src, std::vector<int64_t> tier_pack) {
  const TierArgs ta = make_tier_args(tier_pack);
  const int d = 128;
  TORCH_CHECK(k.size(-1) == d, "amass_decode_cuda: d must be 128");
  TORCH_CHECK(k.size(1) == PAGE, "amass_decode_cuda: page must be 16");
  TORCH_CHECK(G <= 8, "amass_decode_cuda: G must be <= 8");

  dim3 grid((unsigned)split, (unsigned)n_kv, (unsigned)n_req);
  dim3 block(WARPS * 32);
  size_t smem = (size_t)(16 * DMAX + 2 * NSTAGE * PAGE * SSTRIDE)
              * sizeof(__nv_bfloat16);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto kptr = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  auto vptr = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
  auto qptr = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());

  auto run = [&](auto kern) {
    if (smem > 48 * 1024) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          (const void*)kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
          (int)smem));
    }
    kern<<<grid, block, smem, stream>>>(
      qptr, kptr, vptr, bt.data_ptr<int>(), tab.data_ptr<int>(),
      cnt.data_ptr<int>(), sl.data_ptr<int>(),
      m_part.data_ptr<float>(), l_part.data_ptr<float>(),
      acc_part.data_ptr<float>(), (float)scale, (int)n_kv, (int)G,
      (int)split, (int)pps_min,
      (long)q.stride(0), (long)q.stride(1),
      (long)k.stride(0), (long)k.stride(1), (long)k.stride(2),
      (long)v.stride(0), (long)v.stride(1), (long)v.stride(2),
      (long)bt.stride(0), (long)tab.stride(0), (long)tab.stride(1),
      (long)cnt.stride(0), (long)m_part.stride(0), ta);
  };
  if (v_src == 0) run(amass_decode_split<0>);
  else            run(amass_decode_split<1>);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_split", &amass_decode_split_launch, "AMASS sparse decode split-K");
  m.def("decode_split2", &amass_decode_split2_launch,
        "AMASS sparse decode split-K, merge folded (last-CTA reduction)");
  m.def("decode_fused", &amass_decode_fused_launch,
        "AMASS sparse decode, merge folded (1 CTA per (req,kv))");
  m.def("merge", &amass_merge_launch, "AMASS split-K merge");
}
"""

def _tier_pack(vsource):
    """15-int64 tier pointer pack for the ``V_SRC==1`` kernels ([] = resident).

    Mirrors ``tier/decode_mem.py::mem_decode_ptrsel``'s argument derivation:
    hot/staged/pinned int16 bases + page2slot/vbo keys + staging strides +
    (lidx, NB, S) addressing scalars + the mem-kv K flag.  All buffers are
    tier-owned, alloc-once, address-stable -> the baked pointers are CUDA-graph
    safe (contents update device-side via tier.step each step)."""
    if getattr(vsource, "SRC_ID", 0) != 1:
        return []
    u = vsource.union()
    stgv = u["v_pool"].view(torch.int16)
    k_tier = 1 if getattr(vsource, "K_SRC", 0) else 0
    if k_tier:
        hot_k, poolk = vsource.k_tier_ptrs()
        kp, skb, skt, skh = vsource.k_args()
        assert (skb, skt, skh) == (stgv.stride(0), stgv.stride(1),
                                   stgv.stride(2)), \
            "tiered CUDA decode assumes k_pool strides == v_pool strides"
        hotk_ptr = hot_k.view(torch.int16).data_ptr()
        stgk_ptr = kp.view(torch.int16).data_ptr()
        poolk_ptr = poolk.data_ptr()
    else:
        hotk_ptr = stgk_ptr = poolk_ptr = 0
    return [u["hot_i16"].data_ptr(), stgv.data_ptr(), u["pool"].data_ptr(),
            hotk_ptr, stgk_ptr, poolk_ptr,
            u["page2slot"].data_ptr(), u["vbo"].data_ptr(),
            stgv.stride(0), stgv.stride(1), stgv.stride(2),
            int(u["NB"]), int(u["S"]), int(u["lidx"]), k_tier]


_MODS: dict = {}
_DEFAULT_NSTAGE = int(os.environ.get("AMASS_CUDA_NSTAGE", "6"))
_DEFAULT_PVBF16 = bool(int(os.environ.get("AMASS_CUDA_PVBF16", "0")))


_DEFAULT_WARPS = int(os.environ.get("AMASS_CUDA_WARPS", "4"))
# Fused-decode page-groups x 4 D-warps (total warps/CTA); must be a multiple of 4.
_DEFAULT_WARPSF = int(os.environ.get("AMASS_CUDA_WARPSF", "16"))
# Fused-decode K/V wave-ring depth (2 = double buffer).  3/4 were MEASURED
# WORSE (bs16/64K: 253 -> 268 -> 277us): the extra in-flight wave delays its
# own arrival and burns smem without adding CTAs.  Knob kept for re-testing on
# an idle GPU.
_DEFAULT_FWAVES = int(os.environ.get("AMASS_CUDA_FWAVES", "2"))


def _get(nstage: int = None, pvbf16: bool = None, warps: int = None,
         warpsf: int = None, fwaves: int = None):
    """Build (once per config, hash-cached) and return the extension, or None."""
    if nstage is None:
        nstage = _DEFAULT_NSTAGE
    if pvbf16 is None:
        pvbf16 = _DEFAULT_PVBF16
    if warps is None:
        warps = _DEFAULT_WARPS
    if warpsf is None:
        warpsf = _DEFAULT_WARPSF
    if fwaves is None:
        fwaves = _DEFAULT_FWAVES
    key = (nstage, pvbf16, warps, warpsf, fwaves)
    if key in _MODS:
        return _MODS[key]
    try:
        from torch.utils.cpp_extension import load_inline
        flags = ["-O3", "--use_fast_math", f"-DNSTAGE={nstage}",
                 f"-DWARPS={warps}", f"-DWARPS_F={warpsf}",
                 f"-DFWAVES={fwaves}",
                 "-gencode=arch=compute_90a,code=sm_90a"]
        if pvbf16:
            flags.append("-DPV_BF16")
        mod = load_inline(
            name=(f"amass_decode_cuda_s{nstage}w{warps}f{warpsf}"
                  f"v{fwaves}_{'b' if pvbf16 else 'h'}"),
            cpp_sources="", cuda_sources=_CUDA_SRC, extra_cuda_cflags=flags,
            verbose=bool(int(os.environ.get("AMASS_CUDA_VERBOSE", "0"))))
        print(f"[amass_decode_cuda] hand-CUDA sparse decode ACTIVE "
              f"(NSTAGE={nstage} WARPS={warps} FWAVES={fwaves} "
              f"PV={'bf16' if pvbf16 else 'fp16'})", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[amass_decode_cuda] build failed ({type(e).__name__}: {e})",
              flush=True)
        mod = None
    _MODS[key] = mod
    return mod


def sparse_paged_decode_batched_cuda(q: torch.Tensor, kv_cache: torch.Tensor,
                                     block_table: torch.Tensor,
                                     seq_lens: torch.Tensor, st,
                                     out: torch.Tensor, vsource, *,
                                     scale: float = None) -> None:
    """Hand-CUDA Stage B: CUDA split kernel + the Triton merge. Same contract
    as ``attention.decode.sparse_paged_decode_batched`` (drop-in).

    DISPATCH: at high enough batch (``n_req >= AMASS_CUDA_FUSED_MINREQ``, default
    8) the merge-FOLDED single-launch path (``sparse_paged_decode_fused_cuda``)
    wins -- it removes the acc_part HBM round-trip + the separate merge launch,
    and its intra-CTA page-group parallelism keeps ~2x the split path's roofline
    once n_kv*n_req fills the machine (H200 measured bs16/16K: 96us split+merge ->
    62us fused, 24%->37% roofline). At low batch the split-K path keeps more CTAs
    in flight (deep cp.async ring hides HBM latency the few fused CTAs cannot), so
    it stays the default there. Force with ``AMASS_CUDA_FUSED`` = 1 (always) / 0
    (never)."""
    _ns = os.environ.get("AMASS_CUDA_NSTAGE", "")
    _pv = os.environ.get("AMASS_CUDA_PVBF16", "")
    _wp = os.environ.get("AMASS_CUDA_WARPS", "")
    _wf = os.environ.get("AMASS_CUDA_WARPSF", "")
    if scale is None:
        scale = st.scale
    n_req = seq_lens.shape[0]
    # merge-fold dispatch (see docstring)
    _fused = os.environ.get("AMASS_CUDA_FUSED", "")
    _minreq = int(os.environ.get("AMASS_CUDA_FUSED_MINREQ", "8"))
    use_fused = (_fused == "1") or (_fused != "0" and n_req >= _minreq)
    if use_fused:
        return sparse_paged_decode_fused_cuda(
            q, kv_cache, block_table, seq_lens, st, out, vsource, scale=scale)
    mod = _get(int(_ns) if _ns else None,
               bool(int(_pv)) if _pv else None,
               int(_wp) if _wp else None,
               int(_wf) if _wf else None)
    if mod is None:
        raise RuntimeError("amass_decode_cuda extension unavailable")
    k_view, _ = _split_kv(kv_cache)
    _, page, n_kv, d = k_view.shape
    G = st.G
    split = st.split
    # CUDA path likes finer split-K than the Triton default (more CTAs -> more
    # warps to hide HBM latency with the deep cp.async ring). Swept on H200:
    # 8 (bs>1) / 4 (bs=1) win. Overridable via AMASS_CUDA_PPS for tuning.
    pps_min = 8 if n_req > 1 else 4
    _ov = os.environ.get("AMASS_CUDA_PPS", "")
    if _ov:
        pps_min = int(_ov)

    v_ptr, _svb, _svt, _svh = vsource.args()
    V_SRC = vsource.SRC_ID
    tier_pack = _tier_pack(vsource)

    # SPLIT2 (opt-in, MEASURED SLOWER -- kept for the record): merge FOLDED into
    # the decode launch via a last-CTA reduction.  Bitwise-equal to split+merge
    # (max|d| <= 1.2e-4) but the CONCENTRATED merge (one CTA walks n_act splits
    # of latency-chained L2 loads) + a device threadfence per CTA cost MORE than
    # the merge launch it saves: bs1/16K 30.6us vs 17.6 (split+merge) vs 11.7
    # (Triton).  The separate merge kernel parallelizes the combine over
    # (req, kh*G) CTAs; folding it serializes it.  AMASS_CUDA_SPLIT2=1 enables.
    if os.environ.get("AMASS_CUDA_SPLIT2", "0") == "1":
        arrive = getattr(st, "_split2_arrive", None)
        if arrive is None or arrive.shape[0] < st.max_reqs * n_kv:
            # allocated once at eager warmup (never inside graph capture)
            arrive = torch.zeros(st.max_reqs * n_kv, dtype=torch.int32,
                                 device=q.device)
            st._split2_arrive = arrive
        mod.decode_split2(
            q, k_view, v_ptr, block_table, st.page_table, st.page_cnt,
            seq_lens, st.m_part, st.l_part, st.acc_part, arrive, out,
            float(scale), n_req, n_kv, G, split, pps_min, V_SRC, tier_pack)
        return

    mod.decode_split(
        q, k_view, v_ptr, block_table, st.page_table, st.page_cnt, seq_lens,
        st.m_part, st.l_part, st.acc_part, float(scale),
        n_req, n_kv, G, split, pps_min, V_SRC, tier_pack)

    # The Triton merge is the default: a hand-CUDA merge (mod.merge, opt-in via
    # AMASS_CUDA_CMERGE=1) was measured SLOWER in-pipeline (32-thread CTAs -> low
    # occupancy) than the Triton merge's vectorized fixed-width reduction.
    if os.environ.get("AMASS_CUDA_CMERGE", "0") == "1":
        mod.merge(st.m_part, st.l_part, st.acc_part, st.page_cnt, out,
                  n_req, n_kv, G, split, pps_min)
    else:
        merge_splits_kernel[(n_req, n_kv * G)](
            st.m_part, st.l_part, st.acc_part, st.page_cnt, out,
            out.stride(0), out.stride(1), st.m_part.stride(0),
            st.page_cnt.stride(0), G=G, D=d, SPLIT=split,
            SPLIT_PAD=triton.next_power_of_2(split), PPS_MIN=pps_min)


def sparse_paged_decode_fused_cuda(q: torch.Tensor, kv_cache: torch.Tensor,
                                   block_table: torch.Tensor,
                                   seq_lens: torch.Tensor, st,
                                   out: torch.Tensor, vsource, *,
                                   scale: float = None) -> None:
    """Merge-FOLDED Stage B: ONE fused launch, no split-K partials, no merge.

    One CTA per (req, kv-head) loops every selected page and writes the final
    attention output directly (the split-K cross-partial reduction is folded
    into the decode epilogue -- no ``acc_part`` HBM round-trip, no second
    kernel). Same drop-in contract as ``sparse_paged_decode_batched_cuda``.
    Best when ``n_kv*n_req`` fills the machine; ``sparse_paged_decode_batched_cuda``
    (split+merge) stays the low-batch fallback."""
    _ns = os.environ.get("AMASS_CUDA_NSTAGE", "")
    _pv = os.environ.get("AMASS_CUDA_PVBF16", "")
    _wp = os.environ.get("AMASS_CUDA_WARPS", "")
    _wf = os.environ.get("AMASS_CUDA_WARPSF", "")
    mod = _get(int(_ns) if _ns else None,
               bool(int(_pv)) if _pv else None,
               int(_wp) if _wp else None,
               int(_wf) if _wf else None)
    if mod is None:
        raise RuntimeError("amass_decode_cuda extension unavailable")
    if scale is None:
        scale = st.scale
    n_req = seq_lens.shape[0]
    k_view, _ = _split_kv(kv_cache)
    _, page, n_kv, d = k_view.shape
    G = st.G
    v_ptr, _svb, _svt, _svh = vsource.args()
    V_SRC = vsource.SRC_ID
    mod.decode_fused(
        q, k_view, v_ptr, block_table, st.page_table, st.page_cnt, seq_lens,
        out, float(scale), n_req, n_kv, G, V_SRC, _tier_pack(vsource))
