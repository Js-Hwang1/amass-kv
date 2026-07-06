"""decode_mem -- the DYNAMIC-budget MEM tiered decode (Stage B for mem-v/mem-kv).

This module WIRES the previously-shadow mem tiered decode: it consumes the
Stage-A selection interface

    st.page_table   (R, n_kv, MP) int32   ascending logical page ids, -1 padded
    st.page_cnt     (R, n_kv)     int32   VARIABLE per (layer, kv-head) unit

where the DYNAMIC budget makes ``page_cnt`` differ per unit (Sum_u b_u
conserved by the selector; sink + window + partial tail forced into every
unit's set, exactly like the static selector), and decodes over the tier's
3-way KV union:

    hot buffer   (VRAM, exact-LRU residency)        complete + resident pages
    staging pool (VRAM, bounded working set)        partial tail / un-flushed
    pinned pool  (host DRAM, UVA zero-copy)         flushed pages (fallback)

per the ``V_SRC==1`` / ``K_SRC==1`` seam already compiled into the Triton
split-K decode (attention/decode.py). Two launches per layer:

  1. residency gather for THIS layer's selection: ``gather_plan`` (miss
     classification + exact-LRU victim assignment) then the miss FETCH
     (pinned -> hot over PCIe, staged -> hot in-VRAM). Both are graph-safe
     and VARIABLE-COUNT NATIVE: every kernel guards on the per-unit
     ``page_cnt``/``miss_cnt`` device values; no launch shape depends on b_u.
  2. the tiered split-K decode + merge (V from hot|staged|pinned; K likewise
     for mem-kv), reading the SAME page_table/page_cnt.

Fetch transports (``fetch=``):

  * ``"kernel"``  the UVA gather kernel: SM-resident zero-copy reads.
                  GRAPH-SAFE (fixed grid, no host sync) -> the only transport
                  legal inside CUDA-graph capture. Holds SMs while stalled on
                  PCIe (measured ~13 GB/s effective on scattered pages).
  * ``"dma"``     copy-engine DMA (cudaMemcpy2DAsync over coalesced physical-
                  block runs) for the FLUSHED misses + the kernel for STAGED
                  misses. Host-planned (device->host miss list sync) -> NOT
                  graph-capturable, but the transfer itself runs on the copy
                  engine at ~2-3x the UVA kernel's bandwidth and does not
                  steal SMs from the decode.
  * ``"off"``     no gather. Decode still correct (complete non-resident
                  pages fall through to pinned zero-copy reads) but every
                  cold page pays the UVA latency -- the hot buffer is the
                  performance play, not a correctness requirement.

The decode kernel is the golden Triton ``sparse_decode_split_kernel`` --
bitwise-equal output to the resident (V_SRC==0) path given equal bytes, which
is exactly what the tier guarantees (pure int16 bit copies end to end). The
hand-CUDA decode (decode_cuda.py, owned by the fast path) has no tiered-load
arm yet, so the mem path pins the Triton kernel deliberately.
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from ..attention.decode import _pps_min, sparse_paged_decode_batched
from ..attention.merge import merge_splits_kernel

_FETCH_MODES = ("kernel", "dma", "off")


# --------------------------------------------------------------------------- #
# PTR-SELECT tiered split-K decode: one K load + one V load per page.          #
#                                                                              #
# The shared kernel (attention/decode.py, V_SRC==1) issues THREE predicated    #
# loads + two tl.where merges per tensor per page (hot | staged | pinned).     #
# Measured in-graph on H200 that costs +49% over the resident decode (44.3 vs  #
# 29.8us, 4x16K reqs @10%). Here the 3-way select happens on the ADDRESS       #
# (scalar per page) instead of on the DATA: all three sources are addressed    #
# as int16 (pure bits; hot_i16 / v_pool.view(int16) / pinned pool int16), one  #
# load fetches the tile, one bitcast recovers bf16. Bit-identical values by    #
# construction (same bytes, same math), fewer memory instructions.             #
# --------------------------------------------------------------------------- #
@triton.jit
def _mem_decode_split_kernel(q_ptr, bt_ptr, tab_ptr, cnt_ptr, sl_ptr,
                             m_ptr, l_ptr, acc_ptr, sm_scale,
                             stride_qt, stride_qh,
                             kres_ptr, stride_kb, stride_kt, stride_kh,
                             hotv_ptr, hotk_ptr,
                             stgv_ptr, stgk_ptr, svb, svt, svh,
                             poolv_ptr, poolk_ptr,
                             p2s_ptr, vbo_ptr,
                             stride_btr, stride_tabr, stride_tabh,
                             stride_cntr, stride_pr,
                             t_lidx, t_NB, t_S,
                             G: tl.constexpr, G_PAD: tl.constexpr,
                             PAGE: tl.constexpr, D: tl.constexpr,
                             SPLIT: tl.constexpr, PPS_MIN: tl.constexpr,
                             K_TIER: tl.constexpr, NKV: tl.constexpr):
    r = tl.program_id(0)
    kh = tl.program_id(1)
    sp = tl.program_id(2)
    cnt = tl.load(cnt_ptr + r * stride_cntr + kh)
    pps = tl.maximum(tl.cdiv(cnt, SPLIT), PPS_MIN)
    j0 = sp * pps
    if j0 >= cnt:
        return
    j1 = tl.minimum(j0 + pps, cnt)
    offs_g = tl.arange(0, G_PAD)
    offs_d = tl.arange(0, D)
    offs_t = tl.arange(0, PAGE)
    gmask = offs_g < G

    seq_len = tl.load(sl_ptr + r)
    q = tl.load(q_ptr + r * stride_qt
                + (kh * G + offs_g)[:, None] * stride_qh + offs_d[None, :],
                mask=gmask[:, None], other=0.0)

    m_i = tl.full([G_PAD], float("-inf"), tl.float32)
    l_i = tl.zeros([G_PAD], tl.float32)
    acc = tl.zeros([G_PAD, D], tl.float32)

    for j in tl.range(j0, j1, num_stages=3):
        pt = tl.load(tab_ptr + r * stride_tabr + kh * stride_tabh + j
                     ).to(tl.int64)
        blk = tl.load(bt_ptr + r * stride_btr + pt).to(tl.int64)
        tok = pt * PAGE + offs_t
        tmask = tok < seq_len
        # ---- one 3-way source decision per PAGE (scalars) ----------------- #
        complete = ((pt + 1) * PAGE) <= seq_len
        slot = tl.load(p2s_ptr + kh * t_NB + blk)
        vb = tl.load(vbo_ptr + blk)
        use_hot = complete & (slot >= 0)
        use_stg = (use_hot == 0) & (vb >= 0)
        hot_off = ((kh * t_S + tl.maximum(slot, 0)).to(tl.int64) * (PAGE * D)
                   + offs_t[:, None] * D + offs_d[None, :])
        stg_off = (tl.maximum(vb, 0).to(tl.int64) * svb + kh * svh
                   + offs_t[:, None] * svt + offs_d[None, :])
        pin_off = ((((t_lidx * t_NB + blk) * NKV + kh).to(tl.int64)
                    * (PAGE * D)) + offs_t[:, None] * D + offs_d[None, :])
        # ---- K: resident (K_TIER=0) or address-selected tier read --------- #
        if K_TIER == 0:
            k = tl.load(kres_ptr + blk * stride_kb + kh * stride_kh
                        + offs_t[:, None] * stride_kt + offs_d[None, :],
                        mask=tmask[:, None], other=0.0)
        else:
            kp = tl.where(use_hot, hotk_ptr + hot_off,
                          tl.where(use_stg, stgk_ptr + stg_off,
                                   poolk_ptr + pin_off))
            k = tl.load(kp, mask=tmask[:, None], other=0
                        ).to(tl.bfloat16, bitcast=True)
        # ---- V: address-selected tier read (always) ------------------------ #
        vp = tl.where(use_hot, hotv_ptr + hot_off,
                      tl.where(use_stg, stgv_ptr + stg_off,
                               poolv_ptr + pin_off))
        v = tl.load(vp, mask=tmask[:, None], other=0
                    ).to(tl.bfloat16, bitcast=True)
        s = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale
        s = tl.where(tmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = tl.dot(p.to(tl.float16), v.to(tl.float16),
                     acc=acc * alpha[:, None])
        m_i = m_new

    base = r * stride_pr + (kh * SPLIT + sp) * G
    tl.store(m_ptr + base + offs_g, m_i, mask=gmask)
    tl.store(l_ptr + base + offs_g, l_i, mask=gmask)
    tl.store(acc_ptr + (base + offs_g)[:, None] * D + offs_d[None, :], acc,
             mask=gmask[:, None])


def mem_decode_ptrsel(q: torch.Tensor, kv_cache: Optional[torch.Tensor],
                      block_table: torch.Tensor, seq_lens: torch.Tensor,
                      st, out: torch.Tensor, tier, lidx: int, *,
                      scale: float = None) -> None:
    """PTR-SELECT tiered Stage-B decode (split + merge), reading K/V straight
    from the tier's pools (no VSource indirection). Bit-identical output to
    the shared V_SRC==1 kernel (same bytes, same flash math)."""
    if scale is None:
        scale = st.scale
    n_req = seq_lens.shape[0]
    G = st.G
    split = st.split
    pps_min = _pps_min(n_req)
    r5 = tier.residency
    stg = tier.staging
    vp16 = stg.v_pool[lidx].view(torch.int16)
    has_k = tier.offload_k
    if has_k:
        kres = vp16                            # dead placeholder
        skb = skt = skh = 0
        kp16 = stg.k_pool[lidx].view(torch.int16)
        hotk = r5.hot_k_i16[lidx]
        poolk = tier.pool_k.dev_view
        n_kv = r5.n_kv
    else:
        from ..attention.decode import _split_kv
        kres, _ = _split_kv(kv_cache)
        skb, skt, skh = kres.stride(0), kres.stride(1), kres.stride(2)
        n_kv = kres.shape[2]
        kp16 = vp16
        hotk = r5.hot_i16[lidx]
        poolk = tier.pool.dev_view
    d = r5.d
    _mem_decode_split_kernel[(n_req, n_kv, split)](
        q, block_table, st.page_table, st.page_cnt, seq_lens,
        st.m_part, st.l_part, st.acc_part, scale,
        q.stride(0), q.stride(1),
        kres, skb, skt, skh,
        r5.hot_i16[lidx], hotk,
        vp16, kp16,
        vp16.stride(0), vp16.stride(1), vp16.stride(2),
        tier.pool.dev_view, poolk,
        r5.page2slot[lidx], stg.vbo,
        block_table.stride(0), st.page_table.stride(0),
        st.page_table.stride(1), st.page_cnt.stride(0), st.m_part.stride(0),
        lidx, r5.NB, r5.S,
        G=G, G_PAD=max(16, triton.next_power_of_2(G)), PAGE=r5.page, D=d,
        SPLIT=split, PPS_MIN=pps_min, K_TIER=1 if has_k else 0, NKV=n_kv,
        num_warps=4)
    merge_splits_kernel[(n_req, n_kv * G)](
        st.m_part, st.l_part, st.acc_part, st.page_cnt, out,
        out.stride(0), out.stride(1), st.m_part.stride(0),
        st.page_cnt.stride(0), G=G, D=d, SPLIT=split,
        SPLIT_PAD=triton.next_power_of_2(split), PPS_MIN=pps_min)


def mem_dynamic_decode(q: torch.Tensor, kv_cache: Optional[torch.Tensor],
                       block_table: torch.Tensor, seq_lens: torch.Tensor,
                       st, out: torch.Tensor, tier, lidx: int, *,
                       scale: float = None, fetch: str = "kernel",
                       impl: str = "ptrsel") -> None:
    """One layer of the mem tiered decode with VARIABLE per-unit fetch.

    Reads ``st.page_table`` / ``st.page_cnt`` (this layer's Stage-A output;
    counts vary per (request, kv-head)), brings every selected complete page
    into the VRAM hot cache (miss-only fetch, exact-LRU victims), then runs
    the tiered split-K decode + merge into ``out`` rows [0, n_req).

    ``kv_cache`` is the resident engine cache; under mem-kv (``tier.offload_k``)
    it is never dereferenced for K/V (the tier serves both) and may be a K-only
    or dummy tensor. ``fetch`` selects the miss transport (see module doc);
    only ``"kernel"`` is CUDA-graph capturable.

    ``impl``: ``"ptrsel"`` (default) = the address-selected single-load tiered
    TRITON kernel in this module. Measured in-graph on H200 (4x16K reqs, 10%
    budget, 98% hot-hit): mem-kv 44.3 -> 31.6us (+6.8% over the resident
    decode's 29.6us, down from +49%), mem-v 35.8 -> 31.2us (+6.5%); output
    BITWISE equal to both the shared kernel and the resident decode.
    ``"cuda"`` = the hand-CUDA decode's ``V_SRC==1`` tiered arm (FINAL-OPT):
    the SAME per-page ptr-select (hot | staged | pinned, int16 pure-bit
    addressing) inside ``attention/decode_cuda.py``'s split/fused kernels --
    the fast path's kernel speed with tiered bytes, batch-dispatched like the
    resident CUDA decode, bitwise-equal to the resident CUDA decode on equal
    bytes.  ``"shared"`` = the ``V_SRC==1`` 3-way-data-select seam kernel in
    attention/decode.py (kept as the cross-check reference).
    """
    if fetch not in _FETCH_MODES:
        raise ValueError(f"fetch must be one of {_FETCH_MODES}, got {fetch!r}")
    n_req = seq_lens.shape[0]
    if fetch == "kernel":
        tier.step(lidx, st.page_table, st.page_cnt, block_table, seq_lens,
                  n_req)
    elif fetch == "dma":
        tier.step_dma(lidx, st.page_table, st.page_cnt, block_table, seq_lens,
                      n_req, wait=True)
    if impl == "ptrsel":
        mem_decode_ptrsel(q, kv_cache, block_table, seq_lens, st, out, tier,
                          lidx, scale=scale)
    elif impl == "cuda":
        from ..attention.decode_cuda import sparse_paged_decode_batched_cuda
        sparse_paged_decode_batched_cuda(q, kv_cache, block_table, seq_lens,
                                         st, out, tier.vsource(lidx),
                                         scale=scale)
    else:
        sparse_paged_decode_batched(q, kv_cache, block_table, seq_lens, st,
                                    out, tier.vsource(lidx), scale=scale)
