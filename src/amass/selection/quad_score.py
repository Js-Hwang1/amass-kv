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
                       bt_ptr, sl_ptr, nsh_ptr, score_ptr, sh_ptr,
                       scale, page_inv2,
                       stride_qt, stride_qh,
                       stride_mub, stride_muh,
                       stride_musb,
                       stride_vb, stride_vh, stride_vd,
                       stride_vsb,
                       stride_sig2b,
                       stride_btr, stride_sr, stride_sh,
                       stride_shr, stride_shh, stride_shg,
                       G: tl.constexpr, GD: tl.constexpr, D: tl.constexpr,
                       R: tl.constexpr, RD: tl.constexpr, PT: tl.constexpr,
                       V_BITS: tl.constexpr, WRITE_H: tl.constexpr):
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
            if WRITE_H:                                       # nrm combine path
                tl.store(sh_ptr + req * stride_shr + kh * stride_shh
                         + offs_g * stride_shg + p, S, mask=gmask)


def quad_score(st: QuadState, layer: int, q, block_table, seq_lens, n_req: int,
               scale: float) -> None:
    """HOT Stage-A.1: write ``st.score`` (R, n_kv, MP) fp32 for the selectable
    region.  ``st.n_sel_hi`` must have been refreshed for this step
    (``derive_page_params``).  q (T, H, d), block_table (n_req, max_blocks)
    int32, seq_lens (n_req,) int32 -- straight from FlashAttentionMetadata."""
    mu, mu_s, V, V_s, sig2, _ = st.layer_state(layer)
    n_kv, G, D, page, r, MP = st.n_kv, st.G, st.d, st.page, st.r, st.max_pages
    nrm = getattr(st, "combine", "max") == "nrm"
    sh = st.score_h if nrm else st.score              # dummy ptr when unused
    PT = 4
    grid = (n_req, n_kv, triton.cdiv(MP, PT))
    _quad_score_kernel[grid](
        q, mu, mu_s, V, V_s, sig2,
        block_table, seq_lens, st.n_sel_hi, st.score, sh,
        scale, 1.0 / (2.0 * page),
        q.stride(0), q.stride(1),
        mu.stride(0), mu.stride(1),
        mu_s.stride(0),
        V.stride(0), V.stride(1), V.stride(2),
        V_s.stride(0),
        sig2.stride(0),
        block_table.stride(0), st.score.stride(0), st.score.stride(1),
        sh.stride(0), sh.stride(1), sh.stride(2) if nrm else 0,
        G=G, GD=max(16, triton.next_power_of_2(G)), D=D,
        R=r, RD=max(16, triton.next_power_of_2(r)), PT=PT,
        V_BITS=getattr(st, "v_bits", 8), WRITE_H=nrm,
        num_warps=2)
    if nrm:
        _nrm_launch(st, n_req)


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


# =========================================================================== #
# SIGNAL-RESEARCH extension (scratch_signal): CLSE score = quad's storage     #
# geometry + rank-r' PER-KEY COORDS.  Per (request, kv-head, page):           #
#                                                                             #
#   q~[g,:]  = Vᵀ q_g                                   (G, r')               #
#   tok[g,t] = (c q~[g]) * s                            (G, page)             #
#   iso[g]   = (|q_g|² − |q~[g]|²) · resid · s²/(2P(d−r'))    (G,)            #
#   S[g]     = mu·q_g s + logsumexp_t tok[g,t] + iso[g]                       #
#   score    = max_g S[g]              (combine="max", graph-static b)        #
#            | Σ_g exp(S[g] − max_page S[g])   (combine="nrm", 3 launches)    #
#                                                                             #
# c is the rank-r' coordinate block (U8·S8, int8/int4, token- or page-grain   #
# scale); resid = trace − Σ_k sig2_k is the residual scatter energy: the      #
# logsumexp restores the peaky single-key mass the Gaussian drop-c form       #
# misses, iso restores the out-of-subspace variance.  Same fixed launch       #
# shapes / no host sync / no allocation as quad_score (full-CUDA-graph safe). #
# =========================================================================== #
@triton.jit
def _clse_score_kernel(q_ptr, mu_ptr, mus_ptr, v_ptr, vs_ptr, c_ptr, cs_ptr,
                       rs_ptr, bt_ptr, sl_ptr, nsh_ptr, score_ptr, sh_ptr,
                       scale, iso_coef,
                       stride_qt, stride_qh,
                       stride_mub, stride_muh,
                       stride_musb,
                       stride_vb, stride_vh, stride_vd,
                       stride_vsb,
                       stride_cb, stride_ch, stride_ct,
                       stride_csb,
                       stride_rsb,
                       stride_btr, stride_sr, stride_sh,
                       stride_shr, stride_shh, stride_shg,
                       G: tl.constexpr, GD: tl.constexpr, D: tl.constexpr,
                       PAGE: tl.constexpr, PGD: tl.constexpr,
                       R: tl.constexpr, RD: tl.constexpr, PT: tl.constexpr,
                       V_BITS: tl.constexpr, C_BITS: tl.constexpr,
                       C_TOKEN_GRAIN: tl.constexpr, WRITE_H: tl.constexpr):
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

    q = tl.load(q_ptr + req * stride_qt
                + (kh * G + offs_g)[:, None] * stride_qh + offs_d[None, :],
                mask=gmask[:, None], other=0.0).to(tl.float32)
    qsq = tl.sum(q * q, axis=1)                            # (GD,)

    for i in tl.static_range(PT):
        p = tb * PT + i
        if p < n_sel_hi:
            blk = tl.load(bt_ptr + req * stride_btr + p).to(tl.int64)
            mu_c = tl.load(mu_ptr + blk * stride_mub + kh * stride_muh
                           + offs_d).to(tl.float32)
            mu_s = tl.load(mus_ptr + blk * stride_musb + kh).to(tl.float32)
            mu = mu_c * mu_s
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
            if C_TOKEN_GRAIN:
                c_s = tl.load(cs_ptr + blk * stride_csb + kh * PAGE + offs_t,
                              mask=pmask, other=0.0).to(tl.float32)
                cc = c_c * c_s[:, None]                        # (PGD, RD)
            else:
                c_s = tl.load(cs_ptr + blk * stride_csb + kh).to(tl.float32)
                cc = c_c * c_s
            resid = tl.load(rs_ptr + blk * stride_rsb + kh).to(tl.float32)

            qt = tl.dot(q, V, allow_tf32=False)               # (GD, RD)
            tok = tl.dot(qt, tl.trans(cc), allow_tf32=False) * scale  # (GD,PGD)
            tvalid = ((p * PAGE + offs_t) < seq_len) & pmask
            tok = tl.where(tvalid[None, :], tok, float("-inf"))
            m = tl.max(tok, axis=1)
            lse = m + tl.log(tl.sum(tl.exp(tok - m[:, None]), axis=1))
            qhsq = tl.sum(qt * qt, axis=1)                     # (GD,)
            iso = tl.maximum(qsq - qhsq, 0.0) * resid * iso_coef
            qmu = tl.sum(q * mu[None, :], axis=1) * scale      # (GD,)
            S = tl.where(gmask, qmu + lse + iso, float("-inf"))
            score = tl.max(S, axis=0)                          # group max
            tl.store(score_ptr + req * stride_sr + kh * stride_sh + p, score)
            if WRITE_H:
                tl.store(sh_ptr + req * stride_shr + kh * stride_shh
                         + offs_g * stride_shg + p, S, mask=gmask)


@triton.jit
def _nrm_hmax_kernel(sh_ptr, nsh_ptr, hmax_ptr,
                     stride_shr, stride_shh, stride_shg,
                     stride_hr, stride_hh,
                     G: tl.constexpr, P_PAD: tl.constexpr):
    """Per (req, kv-head, g): max of the per-head score over [0, n_sel_hi)."""
    req = tl.program_id(0)
    kh = tl.program_id(1)
    g = tl.program_id(2)
    n_sel_hi = tl.load(nsh_ptr + req)
    offs = tl.arange(0, P_PAD)
    v = tl.load(sh_ptr + req * stride_shr + kh * stride_shh + g * stride_shg
                + offs, mask=offs < n_sel_hi, other=float("-inf"))
    tl.store(hmax_ptr + req * stride_hr + kh * stride_hh + g,
             tl.max(v, axis=0))


@triton.jit
def _nrm_combine_kernel(sh_ptr, hmax_ptr, nsh_ptr, score_ptr,
                        stride_shr, stride_shh, stride_shg,
                        stride_hr, stride_hh,
                        stride_sr, stride_sh,
                        G: tl.constexpr, GD: tl.constexpr,
                        P_PAD: tl.constexpr):
    """score[p] = sum_g exp(S[g,p] - hmax[g]) over the selectable region."""
    req = tl.program_id(0)
    kh = tl.program_id(1)
    n_sel_hi = tl.load(nsh_ptr + req)
    offs_g = tl.arange(0, GD)
    offs_p = tl.arange(0, P_PAD)
    gmask = offs_g < G
    pmask = offs_p < n_sel_hi
    S = tl.load(sh_ptr + req * stride_shr + kh * stride_shh
                + offs_g[:, None] * stride_shg + offs_p[None, :],
                mask=gmask[:, None] & pmask[None, :], other=float("-inf"))
    hm = tl.load(hmax_ptr + req * stride_hr + kh * stride_hh + offs_g,
                 mask=gmask, other=0.0)
    sc = tl.sum(tl.where(gmask[:, None], tl.exp(S - hm[:, None]), 0.0), axis=0)
    tl.store(score_ptr + req * stride_sr + kh * stride_sh + offs_p, sc,
             mask=pmask)


def clse_score(st: QuadState, layer: int, q, block_table, seq_lens,
               n_req: int, scale: float) -> None:
    """Stage-A.1 (CLSE mode): write ``st.score`` (R, n_kv, MP) fp32 for the
    selectable region; combine="nrm" additionally writes ``st.score_h`` and
    re-combines with the per-head-normalized mass sum (3 fixed launches)."""
    mu, mu_s, V, V_s, _sig2, _ = st.layer_state(layer)
    c, c_s, resid = st.quad_c[layer], st.c_scale[layer], st.quad_resid[layer]
    n_kv, G, D, page, r, MP = st.n_kv, st.G, st.d, st.page, st.r, st.max_pages
    nrm = st.combine == "nrm"
    sh = st.score_h if nrm else st.score           # dummy ptr when unused
    token_grain = st.c_grain == "token"
    import os as _os
    if r == 2 and _os.environ.get("AMASS_CLSE_GENERIC", "0") != "1":
        # fast block-sweep reference (chunked-K dots; ~2 orders faster at
        # long-MP/mns16 shapes; bitwise-equivalent within fp tolerance).
        import os as _os2
        PB = int(_os2.environ.get("AMASS_CLSE_PB", "32"))
        grid = (n_req, n_kv, triton.cdiv(MP, PB))
        _clse_score_fast_kernel[grid](
            q, mu, mu_s, V, V_s, c, c_s, resid,
            block_table, st.n_sel_hi, st.score, sh,
            scale, scale * scale / (2.0 * page * max(D - r, 1)),
            q.stride(0), q.stride(1),
            mu.stride(0), mu.stride(1),
            mu_s.stride(0),
            V.stride(0), V.stride(1), V.stride(2),
            V_s.stride(0),
            c.stride(0), c.stride(1), c.stride(2),
            c_s.stride(0),
            resid.stride(0),
            block_table.stride(0), st.score.stride(0), st.score.stride(1),
            sh.stride(0), sh.stride(1), sh.stride(2) if nrm else 0,
            G=G, GD=max(16, triton.next_power_of_2(G)), D=D, DK=32,
            PAGE=page, PB=PB,
            V_BITS=getattr(st, "v_bits", 8), C_BITS=st.c_bits,
            C_TOKEN_GRAIN=token_grain, WRITE_H=nrm,
            num_warps=4)
        if nrm:
            _nrm_launch(st, n_req)
        return
    PT = 4
    grid = (n_req, n_kv, triton.cdiv(MP, PT))
    _clse_score_kernel[grid](
        q, mu, mu_s, V, V_s, c, c_s, resid,
        block_table, seq_lens, st.n_sel_hi, st.score, sh,
        scale, scale * scale / (2.0 * page * max(D - r, 1)),
        q.stride(0), q.stride(1),
        mu.stride(0), mu.stride(1),
        mu_s.stride(0),
        V.stride(0), V.stride(1), V.stride(2),
        V_s.stride(0),
        c.stride(0), c.stride(1), c.stride(2),
        c_s.stride(0),
        resid.stride(0),
        block_table.stride(0), st.score.stride(0), st.score.stride(1),
        sh.stride(0), sh.stride(1), sh.stride(2) if nrm else 0,
        G=G, GD=max(16, triton.next_power_of_2(G)), D=D,
        PAGE=page, PGD=max(16, triton.next_power_of_2(page)),
        R=r, RD=max(16, triton.next_power_of_2(r)), PT=PT,
        V_BITS=getattr(st, "v_bits", 8), C_BITS=st.c_bits,
        C_TOKEN_GRAIN=token_grain, WRITE_H=nrm,
        num_warps=2)
    if nrm:
        _nrm_launch(st, n_req)


def _nrm_launch(st: QuadState, n_req: int) -> None:
    """The two nrm-combine passes (per-head max over pages, exp-sum)."""
    G, MP = st.G, st.max_pages
    P_PAD = triton.next_power_of_2(MP)
    _nrm_hmax_kernel[(n_req, st.n_kv, G)](
        st.score_h, st.n_sel_hi, st.hmax,
        st.score_h.stride(0), st.score_h.stride(1), st.score_h.stride(2),
        st.hmax.stride(0), st.hmax.stride(1),
        G=G, P_PAD=P_PAD, num_warps=4)
    _nrm_combine_kernel[(n_req, st.n_kv)](
        st.score_h, st.hmax, st.n_sel_hi, st.score,
        st.score_h.stride(0), st.score_h.stride(1), st.score_h.stride(2),
        st.hmax.stride(0), st.hmax.stride(1),
        st.score.stride(0), st.score.stride(1),
        G=G, GD=max(2, triton.next_power_of_2(G)), P_PAD=P_PAD,
        num_warps=4)


def select_pages_clse(st: QuadState, layer: int, q, block_table, seq_lens,
                      n_req: int, scale: float, *, derive: bool = True) -> None:
    """One-call Stage-A entry (CLSE score mode): (derive) + clse_score +
    topb_select (the radix tau-select reused unchanged)."""
    if derive:
        derive_page_params(st, seq_lens, n_req)
    clse_score(st, layer, q, block_table, seq_lens, n_req, scale)
    topb_select(st, n_req)


# --------------------------------------------------------------------------- #
# FAST CLSE reference (r'=2): one CTA per (req, kv-head, PB-page block).      #
# The generic per-page kernel above is the bitwise-intent reference; at e2e   #
# shapes (mns16, MP~2500) its per-page tl.dot structure is ~100x off the CUDA #
# quad score, which stalls long generations.  This kernel batches PB=32 pages #
# per CTA with THREE (GD,D)x(D,PB) dots (qmu, qh0, qh1) + an online 16-step   #
# LSE sweep -- same math, ~30-60x fewer CTAs / dots.  r'=2 only.              #
# --------------------------------------------------------------------------- #
@triton.jit
def _clse_score_fast_kernel(q_ptr, mu_ptr, mus_ptr, v_ptr, vs_ptr, c_ptr,
                            cs_ptr, rs_ptr, bt_ptr, nsh_ptr, score_ptr, sh_ptr,
                            scale, iso_coef,
                            stride_qt, stride_qh,
                            stride_mub, stride_muh,
                            stride_musb,
                            stride_vb, stride_vh, stride_vd,
                            stride_vsb,
                            stride_cb, stride_ch, stride_ct,
                            stride_csb,
                            stride_rsb,
                            stride_btr, stride_sr, stride_sh,
                            stride_shr, stride_shh, stride_shg,
                            G: tl.constexpr, GD: tl.constexpr, D: tl.constexpr,
                            DK: tl.constexpr, PAGE: tl.constexpr,
                            PB: tl.constexpr,
                            V_BITS: tl.constexpr, C_BITS: tl.constexpr,
                            C_TOKEN_GRAIN: tl.constexpr,
                            WRITE_H: tl.constexpr):
    req = tl.program_id(0)
    kh = tl.program_id(1)
    tb = tl.program_id(2)
    n_sel_hi = tl.load(nsh_ptr + req)
    if tb * PB >= n_sel_hi:
        return
    offs_g = tl.arange(0, GD)
    offs_k = tl.arange(0, DK)
    offs_p = tb * PB + tl.arange(0, PB)
    gmask = offs_g < G
    pmask = offs_p < n_sel_hi

    blk = tl.load(bt_ptr + req * stride_btr + offs_p, mask=pmask,
                  other=0).to(tl.int64)                          # (PB,)

    # ---- K-chunked dots: qmu_raw, qh0_raw, qh1_raw (GD, PB); qsq (GD,) --- #
    qmu = tl.zeros([GD, PB], tl.float32)
    qh0 = tl.zeros([GD, PB], tl.float32)
    qh1 = tl.zeros([GD, PB], tl.float32)
    qsq = tl.zeros([GD], tl.float32)
    for k0 in tl.static_range(0, D, DK):
        qk = tl.load(q_ptr + req * stride_qt
                     + (kh * G + offs_g)[:, None] * stride_qh
                     + (k0 + offs_k)[None, :],
                     mask=gmask[:, None], other=0.0).to(tl.float32)  # (GD,DK)
        qsq += tl.sum(qk * qk, axis=1)
        mu_c = tl.load(mu_ptr + blk[:, None] * stride_mub + kh * stride_muh
                       + (k0 + offs_k)[None, :], mask=pmask[:, None],
                       other=0).to(tl.float32)                   # (PB, DK)
        qmu += tl.dot(qk, tl.trans(mu_c), allow_tf32=False)
        if V_BITS == 4:
            v_b = tl.load(v_ptr + blk[:, None] * stride_vb + kh * stride_vh
                          + (k0 + offs_k)[None, :] * stride_vd,
                          mask=pmask[:, None], other=0).to(tl.int32)
            v0n = v_b & 0xF
            v1n = (v_b >> 4) & 0xF
            v0 = tl.where(v0n >= 8, v0n - 16, v0n).to(tl.float32)
            v1 = tl.where(v1n >= 8, v1n - 16, v1n).to(tl.float32)
        else:
            v0 = tl.load(v_ptr + blk[:, None] * stride_vb + kh * stride_vh
                         + (k0 + offs_k)[None, :] * stride_vd,
                         mask=pmask[:, None], other=0.0).to(tl.float32)
            v1 = tl.load(v_ptr + blk[:, None] * stride_vb + kh * stride_vh
                         + (k0 + offs_k)[None, :] * stride_vd + 1,
                         mask=pmask[:, None], other=0.0).to(tl.float32)
        qh0 += tl.dot(qk, tl.trans(v0), allow_tf32=False)
        qh1 += tl.dot(qk, tl.trans(v1), allow_tf32=False)

    mu_s = tl.load(mus_ptr + blk * stride_musb + kh, mask=pmask,
                   other=0.0).to(tl.float32)
    vs0 = tl.load(vs_ptr + blk * stride_vsb + kh * 2, mask=pmask,
                  other=0.0).to(tl.float32)
    vs1 = tl.load(vs_ptr + blk * stride_vsb + kh * 2 + 1, mask=pmask,
                  other=0.0).to(tl.float32)
    qmu = qmu * (mu_s * scale)[None, :]
    qh0 = qh0 * vs0[None, :]
    qh1 = qh1 * vs1[None, :]

    # ---- coords (PB, PAGE) x2 + scales ------------------------------------ #
    offs_t = tl.arange(0, PAGE)
    if C_BITS == 4:
        c_b = tl.load(c_ptr + blk[:, None] * stride_cb + kh * stride_ch
                      + offs_t[None, :] * stride_ct,
                      mask=pmask[:, None], other=0).to(tl.int32)  # (PB, PAGE)
        c0n = c_b & 0xF
        c1n = (c_b >> 4) & 0xF
        c0 = tl.where(c0n >= 8, c0n - 16, c0n).to(tl.float32)
        c1 = tl.where(c1n >= 8, c1n - 16, c1n).to(tl.float32)
    else:
        c0 = tl.load(c_ptr + blk[:, None] * stride_cb + kh * stride_ch
                     + offs_t[None, :] * stride_ct,
                     mask=pmask[:, None], other=0.0).to(tl.float32)
        c1 = tl.load(c_ptr + blk[:, None] * stride_cb + kh * stride_ch
                     + offs_t[None, :] * stride_ct + 1,
                     mask=pmask[:, None], other=0.0).to(tl.float32)
    if C_TOKEN_GRAIN:
        c_s = tl.load(cs_ptr + blk[:, None] * stride_csb + kh * PAGE
                      + offs_t[None, :], mask=pmask[:, None],
                      other=0.0).to(tl.float32)                   # (PB, PAGE)
        c0 = c0 * c_s
        c1 = c1 * c_s
    else:
        c_s = tl.load(cs_ptr + blk * stride_csb + kh, mask=pmask,
                      other=0.0).to(tl.float32)                   # (PB,)
        c0 = c0 * c_s[:, None]
        c1 = c1 * c_s[:, None]

    # ---- online LSE over the PAGE tokens (register column-select) -------- #
    m = tl.full([GD, PB], float("-inf"), tl.float32)
    sacc = tl.zeros([GD, PB], tl.float32)
    for t in tl.static_range(PAGE):
        c0t = tl.sum(tl.where(offs_t[None, :] == t, c0, 0.0), axis=1)  # (PB,)
        c1t = tl.sum(tl.where(offs_t[None, :] == t, c1, 0.0), axis=1)
        v = (qh0 * c0t[None, :] + qh1 * c1t[None, :]) * scale
        m_new = tl.maximum(m, v)
        sacc = sacc * tl.exp(m - m_new) + tl.exp(v - m_new)
        m = m_new
    lse = m + tl.log(sacc)

    resid = tl.load(rs_ptr + blk * stride_rsb + kh, mask=pmask,
                    other=0.0).to(tl.float32)                     # (PB,)
    qhsq = qh0 * qh0 + qh1 * qh1                                  # (GD, PB)
    iso = tl.maximum(qsq[:, None] - qhsq, 0.0) * resid[None, :] * iso_coef
    S = tl.where(gmask[:, None] & pmask[None, :], qmu + lse + iso,
                 float("-inf"))                                   # (GD, PB)
    score = tl.max(S, axis=0)                                     # (PB,)
    tl.store(score_ptr + req * stride_sr + kh * stride_sh + offs_p, score,
             mask=pmask)
    if WRITE_H:
        tl.store(sh_ptr + req * stride_shr + kh * stride_shh
                 + offs_g[:, None] * stride_shg + offs_p[None, :], S,
                 mask=gmask[:, None] & pmask[None, :])
