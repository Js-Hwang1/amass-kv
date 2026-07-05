"""r8 build / refresh — the query-INDEPENDENT low-rank int8 page summary.

Runs OUTSIDE the CUDA graph (torch + ``linalg.eigh``, data-dependent shapes),
once per page-finalize, exactly like the reference ``dynkv_plugin.py:185-213``
and following the content-tag invalidation of ``vllm/kernels/selection_state.py``
(rebuild ONLY finalized pages whose stored tag no longer matches the live K).

For a FINALIZED page every page token is valid (``(p+1)*page <= seq_len``), so
no per-token validity mask is needed here (the partial tail page is never built
- it sits inside the always-attended window). Per (block, kv-head):

    mu = mean_t K            (d,)                    page centroid
    dc = K - mu              (page, d)               centered keys
    Gm = dc @ dcᵀ            (page, page)            page-gram (cheap eigh)
    S², U = eigh(Gm)                                 ascending eigenpairs
    S8 = sqrt(clamp(S²[-r:]))                        (r,)   top-r singular values
    U8 = U[..., -r:]         (page, r)
    c  = U8 * S8             (page, r)               coords
    Vk = dcᵀ @ U8 / S8       (d, r)                  right singular axes

Then int8 fake-quant at the configured grain (default: per-column Vk, per-page
c, per-(block,head) mu -- matching the R8State scale layouts) and scatter into
the layer's slab; the page-final key's leading TAGW channels become the new tag.
"""
from __future__ import annotations

import os

import torch

from .state import R8State

# --------------------------------------------------------------------------- #
# Custom 16x16 cyclic-Jacobi eigensolver (Piece C).                            #
#                                                                              #
# ``torch.linalg.eigh`` on the batched page-grams routes to cusolver           #
# ``syevBatched``, which FAILS (CUSOLVER_STATUS_INTERNAL_ERROR) at >= 65536    #
# grams -- i.e. bs16 @ 16K, bs1 @ 131K, bs4 @ 64K -- a correctness blocker for #
# long-ctx / high-batch (measured: scratch_audit/build_probe.py).  The page is #
# 16 tokens, so every gram is a symmetric 16x16.  This one-warp-per-gram        #
# cyclic-Jacobi solver has NO batch-size ceiling and removes the ~33 ms         #
# cusolver overhead.  Eigenvalues are sorted ASCENDING in torch to match the    #
# ``torch.linalg.eigh`` contract the build epilogue expects.                    #
# --------------------------------------------------------------------------- #
_JAC_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef JAC_WARPS
#define JAC_WARPS 8
#endif
#ifndef JAC_SWEEPS
#define JAC_SWEEPS 14
#endif
#define JN 16              // page = 16 -> symmetric 16x16 gram
#define JPAD 17            // smem row pad -> conflict-free

// One warp per gram.  A (16x16 symmetric) + V (eigenvectors, columns) live in
// smem; cyclic-Jacobi sweeps zero each off-diagonal (p,q) in turn, lanes j<16
// applying the two-sided rotation column-parallel.  diag(A) -> eigenvalues,
// columns of V -> eigenvectors (unsorted; torch sorts ascending after).
__global__ void jacobi_eigh16(const float* __restrict__ Gm,   // (M,16,16)
                              float* __restrict__ W,           // (M,16)
                              float* __restrict__ Vout,        // (M,16,16)
                              const int M) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = blockIdx.x * JAC_WARPS + warp;
    if (g >= M) return;
    __shared__ float As[JAC_WARPS][JN][JPAD];
    __shared__ float Vs[JAC_WARPS][JN][JPAD];
    float (*A)[JPAD] = As[warp];
    float (*V)[JPAD] = Vs[warp];
    const float* Gg = Gm + (long)g * JN * JN;
    for (int i = lane; i < JN * JN; i += 32) {
        int r = i / JN, c = i % JN;
        A[r][c] = Gg[i];
        V[r][c] = (r == c) ? 1.f : 0.f;
    }
    __syncwarp();
    for (int sweep = 0; sweep < JAC_SWEEPS; ++sweep) {
        for (int p = 0; p < JN - 1; ++p) {
            for (int q = p + 1; q < JN; ++q) {
                float apq = A[p][q];
                if (fabsf(apq) > 1e-20f) {
                    float app = A[p][p], aqq = A[q][q];
                    float theta = (aqq - app) / (2.f * apq);
                    float t = (theta >= 0.f ? 1.f : -1.f)
                              / (fabsf(theta) + sqrtf(theta * theta + 1.f));
                    float cc = rsqrtf(t * t + 1.f);
                    float ss = t * cc;
                    // A <- J^T A J : rotate columns p,q then rows p,q.
                    if (lane < JN) {
                        int k = lane;
                        float akp = A[k][p], akq = A[k][q];
                        A[k][p] = cc * akp - ss * akq;
                        A[k][q] = ss * akp + cc * akq;
                    }
                    __syncwarp();
                    if (lane < JN) {
                        int k = lane;
                        float apk = A[p][k], aqk = A[q][k];
                        A[p][k] = cc * apk - ss * aqk;
                        A[q][k] = ss * apk + cc * aqk;
                    }
                    if (lane < JN) {
                        int k = lane;
                        float vkp = V[k][p], vkq = V[k][q];
                        V[k][p] = cc * vkp - ss * vkq;
                        V[k][q] = ss * vkp + cc * vkq;
                    }
                    __syncwarp();
                }
            }
        }
    }
    // write eigenvalues (diag) + eigenvector columns.
    if (lane < JN) W[(long)g * JN + lane] = A[lane][lane];
    for (int i = lane; i < JN * JN; i += 32)
        Vout[(long)g * JN * JN + i] = V[i / JN][i % JN];
}

void jacobi_eigh16_launch(torch::Tensor Gm, torch::Tensor W, torch::Tensor V,
                          int64_t M) {
    const int block = JAC_WARPS * 32;
    const int grid = (int)((M + JAC_WARPS - 1) / JAC_WARPS);
    auto stream = at::cuda::getCurrentCUDAStream();
    jacobi_eigh16<<<grid, block, 0, stream>>>(
        Gm.data_ptr<float>(), W.data_ptr<float>(), V.data_ptr<float>(), (int)M);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("jacobi_eigh16_launch", &jacobi_eigh16_launch, "16x16 Jacobi eigh");
}
"""

_JAC_MOD = None
_JAC_TRIED = False


def _jac_mod():
    global _JAC_MOD, _JAC_TRIED
    if _JAC_TRIED:
        return _JAC_MOD
    _JAC_TRIED = True
    try:
        from torch.utils.cpp_extension import load_inline
        _JAC_MOD = load_inline(
            name="amass_jacobi_eigh16", cpp_sources="", cuda_sources=_JAC_SRC,
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
            verbose=False)
    except Exception:
        _JAC_MOD = None
    return _JAC_MOD


def _eigh_jacobi(Gm: torch.Tensor):
    """Drop-in for ``torch.linalg.eigh`` on batched symmetric 16x16 grams.
    Returns (S2, U) with S2 ASCENDING and U the eigenvector columns, matching
    the ``torch.linalg.eigh`` contract.  Falls back to torch.linalg.eigh when
    the extension is unavailable or the page is not 16 (kernel is 16x16-only)."""
    mod = _jac_mod()
    if mod is None or Gm.shape[-1] != 16 or Gm.shape[-2] != 16:
        return torch.linalg.eigh(Gm)
    *lead, n, _ = Gm.shape
    A = Gm.reshape(-1, 16, 16).contiguous().to(torch.float32)
    M = A.shape[0]
    W = torch.empty(M, 16, device=A.device, dtype=torch.float32)
    V = torch.empty(M, 16, 16, device=A.device, dtype=torch.float32)
    mod.jacobi_eigh16_launch(A, W, V, M)
    # sort ascending (eigh contract) + reorder eigenvector columns to match.
    order = W.argsort(dim=-1)                                  # (M,16)
    S2 = torch.gather(W, -1, order)
    U = torch.gather(V, -1, order.unsqueeze(-2).expand(M, 16, 16))
    return (S2.reshape(*lead, 16).to(Gm.dtype),
            U.reshape(*lead, 16, 16).to(Gm.dtype))


# --------------------------------------------------------------------------- #
# int8/int4 fake-quant helpers.  code = round(x / s), symmetric levels          #
# +/-(2^(bits-1)-1) (int8: +/-127, int4: +/-7), s = amax(|x|, grain)/lvl.  int4  #
# packs two signed nibbles per int8 byte along the LAST (rank) axis -> the code  #
# tensor's last dim halves; the fp16 scale layout is bit-width-independent.      #
# --------------------------------------------------------------------------- #
def _quant(x: torch.Tensor, dim, bits=8):
    lvl = float((1 << (bits - 1)) - 1)                    # 127 (int8) / 7 (int4)
    s = (x.abs().amax(dim=dim, keepdim=True) / lvl).clamp_min(1e-8)
    code = (x / s).round().clamp_(-lvl, lvl).to(torch.int8)
    return code, s


def _pack_i4(code: torch.Tensor) -> torch.Tensor:
    """Pack an int8-typed int4 code (..., r) into (..., r//2) int8: byte =
    lo_nibble | (hi_nibble << 4), each nibble a 4-bit two's-complement value."""
    r = code.shape[-1]
    assert r % 2 == 0
    nib = (code.to(torch.int32) & 0xF)                    # 4-bit two's complement
    lo = nib[..., 0::2]
    hi = nib[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8).view(torch.int8).contiguous()


def _quant_mu(mu):
    # mu (..., d) -> scale per (block, head): amax over d.  mu STAYS int8.
    code, s = _quant(mu, dim=-1, bits=8)
    return code, s.squeeze(-1).to(torch.float16)          # (..., )


def _quant_vk(Vk, grain, bits=8):
    # Vk (..., d, r).  "col" (default) = per-column r (amax over d);
    # "tensor" = one scale per (block, head), broadcast into the r slots.
    if grain == "col":
        code, s = _quant(Vk, dim=-2, bits=bits)           # s (...,1,r)
        sc = s.squeeze(-2).to(torch.float16)              # (..., r)
    else:
        code, s = _quant(Vk, dim=(-2, -1), bits=bits)     # s (...,1,1)
        sc = s.squeeze(-2).expand(*Vk.shape[:-2], Vk.shape[-1])
        sc = sc.contiguous().to(torch.float16)            # (..., r)
    if bits == 4:
        code = _pack_i4(code)                             # (..., d, r//2)
    return code, sc


def _quant_c(c, grain, bits=8):
    # c (..., page, r).  "page" (default) = per-page-token (amax over r);
    # "tensor" = one scale per (block, head), broadcast into the page slots.
    if grain == "page":
        code, s = _quant(c, dim=-1, bits=bits)            # s (...,page,1)
        sc = s.squeeze(-1).to(torch.float16)              # (..., page)
    else:
        code, s = _quant(c, dim=(-2, -1), bits=bits)      # s (...,1,1)
        sc = s.squeeze(-1).expand(*c.shape[:-2], c.shape[-2])
        sc = sc.contiguous().to(torch.float16)            # (..., page)
    if bits == 4:
        code = _pack_i4(code)                             # (..., page, r//2)
    return code, sc


# --------------------------------------------------------------------------- #
def r8_build_refresh(st: R8State, layer: int, K: torch.Tensor,
                     block_table: torch.Tensor, seq_lens: torch.Tensor,
                     n_req: int, *, force: bool = False) -> int:
    """(Re)build the r8 summary for every STALE finalized page of the batch.

    K            per-layer key view (NB, page, n_kv, d), the engine K half.
    block_table  (n_req, max_blocks) int32, logical page -> physical block.
    seq_lens     (n_req,) int32.
    force        ignore tags and rebuild all finalized pages (tests).

    Returns the number of physical blocks rebuilt (0 in the steady state).
    NOT graph-safe by design (eigh + boolean gather); called on page-finalize,
    off the hot path, like ``derive_page_params``.
    """
    device = K.device
    page, n_kv, d, r, tagw = st.page, st.n_kv, st.d, st.r, st.tagw
    if n_req == 0:
        return 0
    sl = seq_lens[:n_req].to(torch.int64)
    n_fin = sl // page                                    # finalized pages/req
    max_fin = int(n_fin.max().item())
    if max_fin == 0:
        return 0

    pidx = torch.arange(max_fin, device=device)
    fin_mask = pidx[None, :] < n_fin[:, None]             # (n_req, max_fin)
    bt = block_table[:n_req, :max_fin].to(torch.int64)    # (n_req, max_fin)
    blocks_flat = bt[fin_mask]                            # (n_valid,)
    if blocks_flat.numel() == 0:
        return 0

    if force:
        stale = torch.ones(blocks_flat.numel(), dtype=torch.bool, device=device)
    else:
        # content tag = leading TAGW channels of the page-final token's key.
        tag_cur = K[blocks_flat, page - 1, :, :tagw].float()   # (n_valid,n_kv,W)
        tag_old = st.r8_tag[layer, blocks_flat].float()
        stale = (tag_cur != tag_old).any(dim=(1, 2))           # NaN-init -> True

    sb = torch.unique(blocks_flat[stale])                 # physical blocks
    if sb.numel() == 0:
        return 0

    # ---- eigh page-gram summary (fp32), per (block, kv-head) ------------- #
    Kp = K[sb].permute(0, 2, 1, 3).float()                # (n_stale,n_kv,page,d)
    mu = Kp.mean(dim=2)                                    # (n_stale,n_kv,d)
    dc = Kp - mu[:, :, None, :]                            # (n_stale,n_kv,page,d)
    Gm = torch.matmul(dc, dc.transpose(-1, -2))           # (..,page,page)
    # Custom 16x16 Jacobi (Piece C): no cusolver batch ceiling (torch.linalg.eigh
    # -> syevBatched FAILS at >=65536 grams).  AMASS_EIGH=torch forces cusolver.
    if page == 16 and os.environ.get("AMASS_EIGH", "jacobi") != "torch":
        S2, U = _eigh_jacobi(Gm)                          # ascending
    else:
        S2, U = torch.linalg.eigh(Gm)                     # ascending
    S8 = S2[..., -r:].clamp_min(1e-10).sqrt()             # (..,r)
    U8 = U[..., -r:]                                       # (..,page,r)
    c = U8 * S8[..., None, :]                              # (..,page,r)
    Vk = torch.matmul(dc.transpose(-1, -2), U8) / S8[..., None, :]  # (..,d,r)

    # ---- int8/int4 fake-quant at the configured grain + bit width -------- #
    mu_code, mu_s = _quant_mu(mu)
    Vk_code, Vk_s = _quant_vk(Vk, st.vk_grain, st.vk_bits)
    c_code, c_s = _quant_c(c, st.c_grain, st.c_bits)

    # ---- scatter into the layer slab + refresh tags --------------------- #
    st.r8_mu[layer, sb] = mu_code
    st.mu_scale[layer, sb] = mu_s
    st.r8_Vk[layer, sb] = Vk_code
    st.Vk_scale[layer, sb] = Vk_s
    st.r8_c[layer, sb] = c_code
    st.c_scale[layer, sb] = c_s
    st.r8_tag[layer, sb] = K[sb, page - 1, :, :tagw].to(st.r8_tag.dtype)
    return int(sb.numel())
