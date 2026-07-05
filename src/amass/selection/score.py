"""r8_score — HOT Triton reference for the low-rank int8 page score.

Golden math (bitwise reference = ``dynkv_plugin.py:216-221``), per
(request, kv-head, page), dequantizing the int8 summary in-kernel:

    q~[g, :] = Vkᵀ q_g                          (over d)       (G, r)
    tok[g, t] = (c q~[g]) * scale               (over r)       (G, page)
    S[g]     = mu·q_g * scale + logsumexp_t tok[g, t]          (G,)
    page score = max_g S[g]   (kv-union: group max)  ->  score[req, kv, page]

Dequant: mu = mu_code * mu_scale (per (block,head)); Vk = Vk_code * Vk_scale
(per-column r); c = c_code * c_scale (per-page token).  The score is written for
the SELECTABLE+sink region [0, n_sel_hi) only -- every such page is finalized
(window >= 1 page), so its r8 summary exists.  The always-attended window /
partial-tail pages are never scored (``topb_select`` force-keeps them).

Grid (n_req, n_kv, cdiv(MP, PT)); fixed launch shapes, no host sync, no
allocation -> full-CUDA-graph safe.  q~ / tok are realised with ``tl.dot`` at
fp32 (allow_tf32=False) so the kernel matches the torch einsum reference to
floating-point tolerance.
"""
from __future__ import annotations

import triton
import triton.language as tl

from .state import R8State


@triton.jit
def _r8_score_kernel(q_ptr, mu_ptr, mus_ptr, vk_ptr, vks_ptr, c_ptr, cs_ptr,
                     bt_ptr, sl_ptr, nsh_ptr, score_ptr, scale,
                     stride_qt, stride_qh,
                     stride_mub, stride_muh,
                     stride_musb,
                     stride_vkb, stride_vkh, stride_vkd,
                     stride_vksb,
                     stride_cb, stride_ch, stride_ct,
                     stride_csb,
                     stride_btr, stride_sr, stride_sh,
                     G: tl.constexpr, GD: tl.constexpr, D: tl.constexpr,
                     PAGE: tl.constexpr, PGD: tl.constexpr,
                     R: tl.constexpr, RD: tl.constexpr, PT: tl.constexpr,
                     VK_BITS: tl.constexpr, C_BITS: tl.constexpr):
    req = tl.program_id(0)
    kh = tl.program_id(1)
    tb = tl.program_id(2)
    n_sel_hi = tl.load(nsh_ptr + req)
    if tb * PT >= n_sel_hi:                       # padded / window-only tile
        return
    seq_len = tl.load(sl_ptr + req)

    offs_g = tl.arange(0, GD)
    offs_d = tl.arange(0, D)
    offs_r = tl.arange(0, RD)
    offs_t = tl.arange(0, PGD)
    gmask = offs_g < G
    rmask = offs_r < R
    pmask = offs_t < PAGE

    # q for the whole GQA group: (GD, D) fp32 (padded heads -> 0).
    q = tl.load(q_ptr + req * stride_qt
                + (kh * G + offs_g)[:, None] * stride_qh + offs_d[None, :],
                mask=gmask[:, None], other=0.0).to(tl.float32)

    for i in tl.static_range(PT):
        p = tb * PT + i
        if p < n_sel_hi:
            blk = tl.load(bt_ptr + req * stride_btr + p).to(tl.int64)
            # ---- dequant mu (D,), Vk (D, RD), c (PGD, RD) --------------- #
            mu_c = tl.load(mu_ptr + blk * stride_mub + kh * stride_muh
                           + offs_d).to(tl.float32)
            mu_s = tl.load(mus_ptr + blk * stride_musb + kh).to(tl.float32)
            mu = mu_c * mu_s
            # Vk codes: int8 direct, or int4 unpacked (2 signed nibbles/byte along
            # r).  int4 shares one byte between column 2j (low) and 2j+1 (high).
            if VK_BITS == 4:
                rp = offs_r // 2
                nib = offs_r % 2
                vk_b = tl.load(vk_ptr + blk * stride_vkb + kh * stride_vkh
                               + offs_d[:, None] * stride_vkd + rp[None, :],
                               mask=rmask[None, :], other=0).to(tl.int32)
                vk_n = (vk_b >> (nib[None, :] * 4)) & 0xF
                vk_c = tl.where(vk_n >= 8, vk_n - 16, vk_n).to(tl.float32)
            else:
                vk_c = tl.load(vk_ptr + blk * stride_vkb + kh * stride_vkh
                               + offs_d[:, None] * stride_vkd + offs_r[None, :],
                               mask=rmask[None, :], other=0.0).to(tl.float32)
            vk_s = tl.load(vks_ptr + blk * stride_vksb + kh * R + offs_r,
                           mask=rmask, other=0.0).to(tl.float32)
            Vk = vk_c * vk_s[None, :]                          # (D, RD)
            if C_BITS == 4:
                rp = offs_r // 2
                nib = offs_r % 2
                c_b = tl.load(c_ptr + blk * stride_cb + kh * stride_ch
                              + offs_t[:, None] * stride_ct + rp[None, :],
                              mask=(pmask[:, None] & rmask[None, :]),
                              other=0).to(tl.int32)
                c_n = (c_b >> (nib[None, :] * 4)) & 0xF
                c_c = tl.where(c_n >= 8, c_n - 16, c_n).to(tl.float32)
            else:
                c_c = tl.load(c_ptr + blk * stride_cb + kh * stride_ch
                              + offs_t[:, None] * stride_ct + offs_r[None, :],
                              mask=(pmask[:, None] & rmask[None, :]),
                              other=0.0).to(tl.float32)
            c_s = tl.load(cs_ptr + blk * stride_csb + kh * PAGE + offs_t,
                          mask=pmask, other=0.0).to(tl.float32)
            cc = c_c * c_s[:, None]                            # (PGD, RD)

            # ---- q~ = Vkᵀ q ; tok = c q~ ; S = mu·q + lse_t(tok) -------- #
            qt = tl.dot(q, Vk, allow_tf32=False)              # (GD, RD)
            tok = tl.dot(qt, tl.trans(cc), allow_tf32=False) * scale  # (GD,PGD)
            tvalid = ((p * PAGE + offs_t) < seq_len) & pmask
            tok = tl.where(tvalid[None, :], tok, float("-inf"))
            m = tl.max(tok, axis=1)
            lse = m + tl.log(tl.sum(tl.exp(tok - m[:, None]), axis=1))
            qmu = tl.sum(q * mu[None, :], axis=1) * scale     # (GD,)
            S = tl.where(gmask, qmu + lse, float("-inf"))
            score = tl.max(S, axis=0)                         # kv-union group max
            tl.store(score_ptr + req * stride_sr + kh * stride_sh + p, score)


def r8_score(st: R8State, layer: int, q, block_table, seq_lens, n_req: int,
             scale: float) -> None:
    """HOT Stage-A.1: write ``st.score`` (R, n_kv, MP) fp32 for the selectable
    region.  ``st.n_sel_hi`` must have been refreshed for this step
    (``derive_page_params``).  q (T, H, d), block_table (n_req, max_blocks)
    int32, seq_lens (n_req,) int32 -- straight from FlashAttentionMetadata."""
    mu, mu_s, Vk, Vk_s, c, c_s, _ = st.layer_state(layer)
    n_kv, G, D, page, r, MP = st.n_kv, st.G, st.d, st.page, st.r, st.max_pages
    PT = 4
    grid = (n_req, n_kv, triton.cdiv(MP, PT))
    _r8_score_kernel[grid](
        q, mu, mu_s, Vk, Vk_s, c, c_s,
        block_table, seq_lens, st.n_sel_hi, st.score, scale,
        q.stride(0), q.stride(1),
        mu.stride(0), mu.stride(1),
        mu_s.stride(0),
        Vk.stride(0), Vk.stride(1), Vk.stride(2),
        Vk_s.stride(0),
        c.stride(0), c.stride(1), c.stride(2),
        c_s.stride(0),
        block_table.stride(0), st.score.stride(0), st.score.stride(1),
        G=G, GD=max(16, triton.next_power_of_2(G)), D=D,
        PAGE=page, PGD=max(16, triton.next_power_of_2(page)),
        R=r, RD=max(16, triton.next_power_of_2(r)), PT=PT,
        VK_BITS=getattr(st, "vk_bits", 8), C_BITS=getattr(st, "c_bits", 8),
        num_warps=2)
