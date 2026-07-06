"""quad build / refresh — the query-INDEPENDENT quad page summary.

Mirror of :func:`r8_build_refresh` (same page-gram eigh, same content-tag
invalidation, same off-hot-path lifecycle) with exactly ONE change: it stores
the r' EIGENVALUES ``sig2`` of the centered-key scatter instead of the r8
per-key coordinate block ``c``.  That is the whole r8->quad saving (P*r coords
-> r values).  Per (block, kv-head), for a finalized page:

    mu   = mean_t K            (d,)                 page centroid
    dc   = K - mu              (page, d)            centered keys
    Gm   = dc @ dcᵀ            (page, page)         page-gram (cheap 16x16 eigh)
    S², U = eigh(Gm)                                ascending eigenpairs
    S8   = sqrt(clamp(S²[-r':]))                    (r',)  top-r' singular values
    U8   = U[..., -r':]        (page, r')
    Vk   = dcᵀ @ U8 / S8       (d, r')              right singular axes (== r8 V)
    sig2 = S²[-r':]            (r',)                the eigenvalues sigma_k^2

``sig2[k] = sigma_k^2 = (c_k^2).sum_t`` because ``c = U8 * S8`` and U8's columns
are orthonormal over the page dim, so ``sum_t c_k^2 = S8_k^2 = S²_k``.  We store
S²[-r':] directly and NEVER materialise or quantise ``c``.

int8 fake-quant reuses the r8 helpers (mu per-vector int8, V per-column int8/
int4) unchanged; sig2 stays fp16 (r' small, DC-magnitude sensitive).  NOT
graph-safe by design (eigh + boolean gather); called on page-finalize.
"""
from __future__ import annotations

import os

import torch

from .build import (_build_blocks, _bulk_blocks, _delta_select, _eigh_jacobi,
                    _page_eigh, _quant_c, _quant_mu, _quant_vk, _scatter_index,
                    _tail_blocks)


def _clse_extras(st, mu, S2r, U8, S8, Kp_flat=None, trace=None):
    """CLSE coords + residual energy from the eigh factors.  c = U8 * S8
    exactly (dc @ Vk = U8 S8, U orthonormal over the page dim).  trace of the
    centered scatter = sum ||k||^2 - P ||mu||^2 (cheap, avoids re-centering);
    resid = trace - sum(top-r' eigenvalues) >= 0."""
    c = U8 * S8[..., None, :]                              # (..., page, r')
    grain = "page" if st.c_grain == "token" else "tensor"
    c_code, c_s = _quant_c(c, grain, st.c_bits)
    if trace is None:
        trace = (Kp_flat.float().pow(2).sum(dim=(1, 3))
                 - st.page * mu.pow(2).sum(-1))            # (N, n_kv)
    resid = (trace - S2r.sum(-1)).clamp_min(0)
    if st.c_grain == "page":
        c_s = c_s[..., 0]                                  # one scale/(blk,head)
    return c_code, c_s, resid
from .quad_state import QuadState


# --------------------------------------------------------------------------- #
def _quad_write(st: QuadState, Kp_flat: torch.Tensor, lidx: torch.Tensor,
                bidx: torch.Tensor) -> None:
    """Summarize + quantize + scatter ``Kp_flat`` (N, page, n_kv, d) into the
    quad slabs at the flat (layer, block) index pairs.  Sync-free."""
    page, r, tagw = st.page, st.r, st.tagw
    tag_new = Kp_flat[:, page - 1, :, :tagw]
    mu, _U8, _S8, S2r, Vk = _page_eigh(Kp_flat, r)
    mu_code, mu_s = _quant_mu(mu)
    Vk_code, Vk_s = _quant_vk(Vk, st.v_grain, st.v_bits)
    st.quad_mu[lidx, bidx] = mu_code
    st.mu_scale[lidx, bidx] = mu_s
    st.quad_V[lidx, bidx] = Vk_code
    st.V_scale[lidx, bidx] = Vk_s
    st.quad_sig2[lidx, bidx] = S2r.to(st.quad_sig2.dtype)
    st.quad_tag[lidx, bidx] = tag_new.to(st.quad_tag.dtype)
    if st.coords == "lse":
        c_code, c_s, resid = _clse_extras(st, mu, S2r, _U8, _S8,
                                          Kp_flat=Kp_flat)
        st.quad_c[lidx, bidx] = c_code
        st.c_scale[lidx, bidx] = c_s.to(st.c_scale.dtype)
        st.quad_resid[lidx, bidx] = resid.to(st.quad_resid.dtype)


def quad_build_tail(st: QuadState, K_layers, block_table: torch.Tensor,
                    seq_lens: torch.Tensor, n_req: int, rows=None) -> int:
    """Batched all-layer rebuild of the last finalized page of ``rows`` (or of
    every request).  quad twin of :func:`amass.selection.build.r8_build_tail`
    (see the rationale there): zero device syncs, idempotent.  Called by the
    builder on steady finalize steps only."""
    if n_req == 0:
        return 0
    L = len(K_layers)
    blocks = _tail_blocks(block_table, seq_lens, n_req, st.page, rows)
    if blocks.numel() == 0:
        return 0
    Kp_raw = torch.stack([K[blocks] for K in K_layers])   # (L,n,page,n_kv,d)
    n = blocks.shape[0]
    lidx, bidx = _scatter_index(L, 0, blocks)
    _quad_write(st, Kp_raw.reshape(L * n, st.page, st.n_kv, st.d), lidx, bidx)
    return L * n


def quad_build_bulk(st: QuadState, K_layers, block_table: torch.Tensor,
                    seq_lens: torch.Tensor, n_req: int, max_fin: int,
                    chunk_pairs: int = 8192) -> int:
    """Rebuild ALL finalized pages for ALL layers, zero syncs; quad twin of
    :func:`amass.selection.build.r8_build_bulk` (rationale there)."""
    if n_req == 0 or max_fin <= 0:
        return 0
    blocks = _bulk_blocks(block_table, seq_lens, n_req, page=st.page,
                          max_fin=max_fin)
    return _build_blocks(_quad_write, st, K_layers, blocks, chunk_pairs)


def quad_build_delta(st: QuadState, K_layers, block_table: torch.Tensor,
                     seq_lens: torch.Tensor, n_req: int, max_fin: int,
                     chunk_pairs: int = 8192) -> int:
    """Composition-change rebuild: only blocks whose content tag went stale;
    quad twin of :func:`amass.selection.build.r8_build_delta` (rationale
    there: one batched all-layer tag pass, rebuild stale blocks only)."""
    if n_req == 0 or max_fin <= 0:
        return 0
    blocks = _delta_select(st.quad_tag, K_layers, block_table, seq_lens,
                           n_req, st.page, max_fin, st.tagw)
    return _build_blocks(_quad_write, st, K_layers, blocks, chunk_pairs)


# --------------------------------------------------------------------------- #
def quad_build_refresh(st: QuadState, layer: int, K: torch.Tensor,
                       block_table: torch.Tensor, seq_lens: torch.Tensor,
                       n_req: int, *, force: bool = False) -> int:
    """(Re)build the quad summary for every STALE finalized page of the batch.

    K            per-layer key view (NB, page, n_kv, d), the engine K half.
    block_table  (n_req, max_blocks) int32, logical page -> physical block.
    seq_lens     (n_req,) int32.
    force        ignore tags and rebuild all finalized pages (tests).

    Returns the number of physical blocks rebuilt (0 in the steady state).
    Same tag-gated, off-hot-path contract as ``r8_build_refresh``.
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
        tag_old = st.quad_tag[layer, blocks_flat].float()
        stale = (tag_cur != tag_old).any(dim=(1, 2))           # NaN-init -> True

    sb = torch.unique(blocks_flat[stale])                 # physical blocks
    if sb.numel() == 0:
        return 0

    # ---- eigh page-gram summary (fp32), per (block, kv-head) ------------- #
    Kp = K[sb].permute(0, 2, 1, 3).float()                # (n_stale,n_kv,page,d)
    mu = Kp.mean(dim=2)                                    # (n_stale,n_kv,d)
    dc = Kp - mu[:, :, None, :]                            # (n_stale,n_kv,page,d)
    Gm = torch.matmul(dc, dc.transpose(-1, -2))           # (..,page,page)
    # Custom 16x16 Jacobi (Piece C); AMASS_EIGH=torch forces cusolver.
    if page == 16 and os.environ.get("AMASS_EIGH", "jacobi") != "torch":
        S2, U = _eigh_jacobi(Gm)                          # ascending
    else:
        S2, U = torch.linalg.eigh(Gm)                     # ascending
    S2r = S2[..., -r:].clamp_min(1e-10)                   # (..,r') eigenvalues
    S8 = S2r.sqrt()                                        # (..,r')
    U8 = U[..., -r:]                                       # (..,page,r')
    Vk = torch.matmul(dc.transpose(-1, -2), U8) / S8[..., None, :]  # (..,d,r')

    # ---- int8/int4 fake-quant (reuse the r8 helpers) -------------------- #
    mu_code, mu_s = _quant_mu(mu)
    Vk_code, Vk_s = _quant_vk(Vk, st.v_grain, st.v_bits)

    # ---- scatter into the layer slab + refresh tags -------------------- #
    st.quad_mu[layer, sb] = mu_code
    st.mu_scale[layer, sb] = mu_s
    st.quad_V[layer, sb] = Vk_code
    st.V_scale[layer, sb] = Vk_s
    st.quad_sig2[layer, sb] = S2r.to(st.quad_sig2.dtype)
    st.quad_tag[layer, sb] = K[sb, page - 1, :, :tagw].to(st.quad_tag.dtype)
    if st.coords == "lse":
        c_code, c_s, resid = _clse_extras(st, mu, S2r, U8, S8,
                                          trace=S2.sum(-1))
        st.quad_c[layer, sb] = c_code
        st.c_scale[layer, sb] = c_s.to(st.c_scale.dtype)
        st.quad_resid[layer, sb] = resid.to(st.quad_resid.dtype)
    return int(sb.numel())
