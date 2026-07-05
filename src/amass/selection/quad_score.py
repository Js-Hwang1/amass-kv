"""quad_score — HOT Triton reference for the Gaussian-MGF quadratic page score.

Golden math (bitwise reference = ``dynkv_plugin.py:224-229`` +
``scripts/quadratic_score_poc.py::s_quad``), per (request, kv-head, page),
dequantizing the int8 summary in-kernel; s = 1/sqrt(d) (= ``scale``), P = page:

    q~[g, :] = Vᵀ q_g                                 (over d)      (G, r')
    quad[g]  = (s^2 / (2P)) * sum_{k<r'} sig2[k] q~[g,k]^2          (G,)
    qmu[g]   = mu·q_g * s                                          (G,)
    S[g]     = qmu[g] + quad[g]                                    (G,)
    page score = max_g S[g]   (kv-union: group max)  ->  score[req, kv, page]

sig2[k] = sigma_k^2 = the k-th eigenvalue of the page's centered-key scatter.
There is NO per-key logsumexp tail (the r8 dependency-chain-bound half of the
kernel): quad is just Vᵀq (a d x r' mma) + r' scaled squared adds + mu·q.

Dequant: mu = mu_code * mu_scale (per (block,head)); V = V_code * V_scale
(per-column r').  The score is written for the SELECTABLE+sink region
[0, n_sel_hi) only (every such page is finalized -> its quad summary exists).

Grid (n_req, n_kv, cdiv(MP, PT)); fixed launch shapes, no host sync, no
allocation -> full-CUDA-graph safe.  q~ is realised with ``tl.dot`` at fp32
(allow_tf32=False) so the kernel matches the torch einsum reference to
floating-point tolerance; this is the bitwise target for ``quad_score_cuda``.
"""
from __future__ import annotations

import triton
import triton.language as tl

from .quad_state import QuadState
from .select import derive_page_params, topb_select


@triton.jit
def _quad_score_kernel(q_ptr, mu_ptr, mus_ptr, v_ptr, vs_ptr, sig2_ptr,
                       bt_ptr, sl_ptr, nsh_ptr, score_ptr, scale, page_inv2,
                       stride_qt, stride_qh,
                       stride_mub, stride_muh,
                       stride_musb,
                       stride_vb, stride_vh, stride_vd,
                       stride_vsb,
                       stride_sig2b,
                       stride_btr, stride_sr, stride_sh,
                       G: tl.constexpr, GD: tl.constexpr, D: tl.constexpr,
                       R: tl.constexpr, RD: tl.constexpr, PT: tl.constexpr,
                       V_BITS: tl.constexpr):
    req = tl.program_id(0)
    kh = tl.program_id(1)
    tb = tl.program_id(2)
    n_sel_hi = tl.load(nsh_ptr + req)
    if tb * PT >= n_sel_hi:                       # padded / window-only tile
        return

    offs_g = tl.arange(0, GD)
    offs_d = tl.arange(0, D)
    offs_r = tl.arange(0, RD)
    gmask = offs_g < G
    rmask = offs_r < R

    # q for the whole GQA group: (GD, D) fp32 (padded heads -> 0).
    q = tl.load(q_ptr + req * stride_qt
                + (kh * G + offs_g)[:, None] * stride_qh + offs_d[None, :],
                mask=gmask[:, None], other=0.0).to(tl.float32)

    for i in tl.static_range(PT):
        p = tb * PT + i
        if p < n_sel_hi:
            blk = tl.load(bt_ptr + req * stride_btr + p).to(tl.int64)
            # ---- dequant mu (D,), V (D, RD), sig2 (RD,) ---------------- #
            mu_c = tl.load(mu_ptr + blk * stride_mub + kh * stride_muh
                           + offs_d).to(tl.float32)
            mu_s = tl.load(mus_ptr + blk * stride_musb + kh).to(tl.float32)
            mu = mu_c * mu_s
            # V codes: int8 direct, or int4 unpacked (2 signed nibbles/byte
            # along r'; col 2j = low nibble, 2j+1 = high nibble).
            if V_BITS == 4:
                rp = offs_r // 2
                nib = offs_r % 2
                v_b = tl.load(v_ptr + blk * stride_vb + kh * stride_vh
                              + offs_d[:, None] * stride_vd + rp[None, :],
                              mask=rmask[None, :], other=0).to(tl.int32)
                v_n = (v_b >> (nib[None, :] * 4)) & 0xF
                v_c = tl.where(v_n >= 8, v_n - 16, v_n).to(tl.float32)
            else:
                v_c = tl.load(v_ptr + blk * stride_vb + kh * stride_vh
                              + offs_d[:, None] * stride_vd + offs_r[None, :],
                              mask=rmask[None, :], other=0.0).to(tl.float32)
            v_s = tl.load(vs_ptr + blk * stride_vsb + kh * R + offs_r,
                          mask=rmask, other=0.0).to(tl.float32)
            V = v_c * v_s[None, :]                             # (D, RD)
            sig2 = tl.load(sig2_ptr + blk * stride_sig2b + kh * R + offs_r,
                           mask=rmask, other=0.0).to(tl.float32)

            # ---- q~ = Vᵀ q ; quad = (s^2/2P) sig2·q~^2 ; S = mu·q s + quad #
            qt = tl.dot(q, V, allow_tf32=False)               # (GD, RD)
            quad = tl.sum(sig2[None, :] * qt * qt, axis=1) * (scale * scale
                                                              * page_inv2)  # (GD,)
            qmu = tl.sum(q * mu[None, :], axis=1) * scale     # (GD,)
            S = tl.where(gmask, qmu + quad, float("-inf"))
            score = tl.max(S, axis=0)                         # kv-union group max
            tl.store(score_ptr + req * stride_sr + kh * stride_sh + p, score)


def quad_score(st: QuadState, layer: int, q, block_table, seq_lens, n_req: int,
               scale: float) -> None:
    """HOT Stage-A.1: write ``st.score`` (R, n_kv, MP) fp32 for the selectable
    region.  ``st.n_sel_hi`` must have been refreshed for this step
    (``derive_page_params``).  q (T, H, d), block_table (n_req, max_blocks)
    int32, seq_lens (n_req,) int32 -- straight from FlashAttentionMetadata."""
    mu, mu_s, V, V_s, sig2, _ = st.layer_state(layer)
    n_kv, G, D, page, r, MP = st.n_kv, st.G, st.d, st.page, st.r, st.max_pages
    PT = 4
    grid = (n_req, n_kv, triton.cdiv(MP, PT))
    _quad_score_kernel[grid](
        q, mu, mu_s, V, V_s, sig2,
        block_table, seq_lens, st.n_sel_hi, st.score, scale, 1.0 / (2.0 * page),
        q.stride(0), q.stride(1),
        mu.stride(0), mu.stride(1),
        mu_s.stride(0),
        V.stride(0), V.stride(1), V.stride(2),
        V_s.stride(0),
        sig2.stride(0),
        block_table.stride(0), st.score.stride(0), st.score.stride(1),
        G=G, GD=max(16, triton.next_power_of_2(G)), D=D,
        R=r, RD=max(16, triton.next_power_of_2(r)), PT=PT,
        V_BITS=getattr(st, "v_bits", 8),
        num_warps=2)


def select_pages_quad(st: QuadState, layer: int, q, block_table, seq_lens,
                      n_req: int, scale: float, *, derive: bool = True) -> None:
    """One-call Stage-A entry (quad score mode): (derive) + quad_score +
    topb_select.  ``derive`` refreshes the per-request page params from seq_lens;
    set False when the metadata builder already called ``derive_page_params``
    this step (it is per-step, not per-layer).  Writes st.page_table /
    st.page_cnt.  The topb radix tau-select is REUSED from the r8 path unchanged
    (it reads only st.score + the derived params, both quad-agnostic)."""
    if derive:
        derive_page_params(st, seq_lens, n_req)
    quad_score(st, layer, q, block_table, seq_lens, n_req, scale)
    topb_select(st, n_req)
