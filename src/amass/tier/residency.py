"""Residency -- exact-LRU hot-buffer residency for the selected pages.

Per (layer, kv-head) the tier keeps a bounded VRAM ``hot`` buffer of ``S`` page
slots. Each decode step, the pages Stage A selected are looked up in the
residency map: HITS already sit in ``hot``; MISSES are gathered (from the pinned
pool if flushed, else the staging pool) into victim slots chosen by exact LRU.
Stage B then reads V from ``hot`` (+ the staging/pinned fallback for the partial
tail). This is the P1 machinery of the prototype, rewritten around two written
invariants:

  I2 (hot == residency): ``hot[l,kv,slot]`` ALWAYS holds the V of
     ``slot2page[l,kv,slot]``. Structurally enforced by making the gather the
     SINGLE owner of the (hot[slot], page2slot, slot2page) transition: it writes
     the hot bytes and re-points both maps in the same operation, and it reads
     from the correct source (pinned if ``pool_valid`` else staged via ``vbo``).
     The prototype's bug #2 was a gather whose staged-source path filled hot
     with the wrong page's V; here there is exactly one gather code path and its
     source selection is explicit. ``assert_maps_inverse`` checks the
     content-independent structural half every debug step.

  I3 (no aliasing, residency side): ``page2slot`` and ``slot2page`` are mutual
     inverses -- at most one slot per page, one page per slot. (The staging
     free-set half of I3 lives in staging.py.)

SCAFFOLD STATUS. The persistent buffers + invariant asserts are real. The
per-step map/gather update is a CORRECT torch reference (host-synchronizing, not
graph-safe) marked TODO to port to the graph-safe Triton/CUDA kernels that
already exist in ``vllm/kernels/dram_tier.py`` (``_vt_miss_diff_kernel`` /
``_select_victims_kernel`` / ``_p2_gather_kernel``). Because the mem-v Stage-B
tiered V-load is the documented FOLLOW-UP, ``gather`` is not yet called on the
hot path; it is provided complete so the follow-up wires decode without
re-architecting. See tier/README.md.
"""
from __future__ import annotations

import contextlib

import torch
import triton
import triton.language as tl

_I32_MAX = tl.constexpr(2147483647)


@triton.jit
def _miss_diff_kernel(tab_ptr, cnt_ptr, bt_ptr, sl_ptr,
                      p2s_ptr, lru_ptr, clock_ptr,
                      mpage_ptr, mcnt_ptr,
                      NB, S, MP, NKV, stride_btr, stride_tr, stride_cr,
                      PAGE: tl.constexpr, BLOCK: tl.constexpr):
    """Per (request, kv-head): classify each selected COMPLETE page as a hot HIT
    (page2slot>=0, touch its LRU so it cannot be evicted this step) or a MISS
    (compact its physical block into mpage). Partial-tail pages are neither (the
    decode reads them from staged v_pool). Ports ``_vt_miss_diff_kernel``."""
    r = tl.program_id(0)
    kv = tl.program_id(1)
    n = tl.load(cnt_ptr + r * stride_cr + kv)
    sl = tl.load(sl_ptr + r)
    clk = tl.load(clock_ptr)
    n_miss = clk * 0
    for off in range(0, n, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        valid = idx < n
        pt = tl.load(tab_ptr + r * stride_tr + kv * MP + idx, mask=valid,
                     other=0)
        blk = tl.load(bt_ptr + r * stride_btr + pt, mask=valid, other=0)
        complete = ((pt + 1) * PAGE) <= sl
        slot = tl.load(p2s_ptr + kv * NB + blk, mask=valid, other=-1)
        is_hit = valid & complete & (slot >= 0)
        is_miss = valid & complete & (slot < 0)
        m32 = is_miss.to(tl.int32)
        mpos = tl.cumsum(m32, axis=0) - m32
        tl.store(mpage_ptr + r * stride_tr + kv * MP + n_miss + mpos, blk,
                 mask=is_miss)
        tl.store(lru_ptr + kv * S + slot, clk, mask=is_hit)
        n_miss += tl.sum(m32, axis=0)
    tl.store(mcnt_ptr + r * stride_cr + kv, n_miss)


@triton.jit
def _select_victims_kernel(mpage_ptr, mcnt_ptr,
                           p2s_ptr, s2p_ptr, lru_ptr, clock_ptr,
                           victim_ptr, ovf_ptr,
                           n_req, NB, S, MP, stride_tr, stride_cr,
                           BLOCK_S: tl.constexpr):
    """Per kv-head, sequential over (request, miss): pick the exact-LRU victim
    slot (min lru_clock among slots NOT touched this step), evict its old page,
    install the miss page, dedup pages already resident (shared prefix). SINGLE
    owner of the (page2slot, slot2page) transition (I2/I3). Ports
    ``_vt_select_victims_kernel``."""
    kv = tl.program_id(0)
    clk = tl.load(clock_ptr)
    for r in range(0, n_req):
        m = tl.load(mcnt_ptr + r * stride_cr + kv)
        for i in range(0, m):
            blk = tl.load(mpage_ptr + r * stride_tr + kv * MP + i)
            cur = tl.load(p2s_ptr + kv * NB + blk)
            if cur >= 0:
                tl.store(victim_ptr + r * stride_tr + kv * MP + i, -1)
            else:
                best_c = clk * 0 + _I32_MAX
                best_s = clk * 0 - 1
                for s0 in range(0, S, BLOCK_S):
                    offs = s0 + tl.arange(0, BLOCK_S)
                    ck = tl.load(lru_ptr + kv * S + offs, mask=offs < S,
                                 other=_I32_MAX)
                    ck = tl.where(ck >= clk, _I32_MAX, ck)
                    bc = tl.min(ck, axis=0)
                    bs = tl.argmin(ck, axis=0).to(tl.int32) + s0
                    take = bc < best_c
                    best_s = tl.where(take, bs, best_s)
                    best_c = tl.where(take, bc, best_c)
                if best_s < 0:
                    tl.store(ovf_ptr, 1)
                    tl.store(victim_ptr + r * stride_tr + kv * MP + i, -1)
                else:
                    old = tl.load(s2p_ptr + kv * S + best_s)
                    if old >= 0:
                        tl.store(p2s_ptr + kv * NB + old, -1)
                    tl.store(p2s_ptr + kv * NB + blk, best_s)
                    tl.store(s2p_ptr + kv * S + best_s, blk)
                    tl.store(lru_ptr + kv * S + best_s, clk)
                    tl.store(victim_ptr + r * stride_tr + kv * MP + i, best_s)


@triton.jit
def _gather_kernel(pool_ptr, hot_ptr, vp_ptr, vbo_ptr, valid_ptr,
                   poolk_ptr, hotk_ptr, kp_ptr,
                   mpage_ptr, mcnt_ptr, victim_ptr,
                   lidx, NB, S, MP, NKV, stride_tr, stride_cr,
                   svb, svt, svh,
                   PAGE: tl.constexpr, D: tl.constexpr, HAS_K: tl.constexpr):
    """Bring each miss page's V (and K, mem-kv) into its victim ``hot`` slot:
    from the pinned pool if flushed (zero-copy int16 UVA read), else from the
    staged pool via vbo (+ lazy write-through to the pinned pool). K and V share
    the miss list + victim slot. SINGLE owner of the hot bytes; keeps hot ==
    residency (I2). Grid (n_req*MP, n_kv). Ports ``_p2_gather_kernel``."""
    j = tl.program_id(0)
    kv = tl.program_id(1)
    r = j // MP
    i = j % MP
    m = tl.load(mcnt_ptr + r * stride_cr + kv)
    if i < m:
        slot = tl.load(victim_ptr + r * stride_tr + kv * MP + i)
        if slot >= 0:
            blk = tl.load(mpage_ptr + r * stride_tr + kv * MP + i)
            vld = tl.load(valid_ptr + kv * NB + blk)
            offs_t = tl.arange(0, PAGE)
            offs_d = tl.arange(0, D)
            tile = offs_t[:, None] * D + offs_d[None, :]
            pool_off = (((lidx * NB + blk) * NKV + kv).to(tl.int64) * (PAGE * D))
            hot_off = ((kv * S + slot).to(tl.int64) * (PAGE * D))
            if vld > 0:
                tl.store(hot_ptr + hot_off + tile,
                         tl.load(pool_ptr + pool_off + tile))
                if HAS_K:
                    tl.store(hotk_ptr + hot_off + tile,
                             tl.load(poolk_ptr + pool_off + tile))
            else:
                vb = tl.load(vbo_ptr + blk)
                if vb >= 0:
                    stg = vb.to(tl.int64) * svb + kv * svh \
                        + offs_t[:, None] * svt + offs_d[None, :]
                    v16 = tl.load(vp_ptr + stg).to(tl.int16, bitcast=True)
                    tl.store(hot_ptr + hot_off + tile, v16)
                    tl.store(pool_ptr + pool_off + tile, v16)
                    if HAS_K:
                        k16 = tl.load(kp_ptr + stg).to(tl.int16, bitcast=True)
                        tl.store(hotk_ptr + hot_off + tile, k16)
                        tl.store(poolk_ptr + pool_off + tile, k16)
                    tl.store(valid_ptr + kv * NB + blk, tl.cast(1, tl.int8))
                else:
                    tl.store(hot_ptr + hot_off + tile,
                             tl.load(pool_ptr + pool_off + tile))
                    if HAS_K:
                        tl.store(hotk_ptr + hot_off + tile,
                                 tl.load(poolk_ptr + pool_off + tile))


@triton.jit
def _invalidate_kernel(bt_ptr, sl_ptr, qsl_ptr,
                       p2s_ptr, s2p_ptr, valid_ptr,
                       NB, S, stride_btr,
                       PAGE: tl.constexpr, NKV: tl.constexpr,
                       KV_PAD: tl.constexpr):
    """Drop pool_valid + hot residency for every page (re)written this step
    (block reuse / chunked prefill make both the hot copy AND the pinned copy
    stale). Grid (L, n_req), OUTSIDE the graph before flush/gather. Ports
    ``_vt_invalidate_kernel``."""
    l = tl.program_id(0)
    r = tl.program_id(1)
    sl = tl.load(sl_ptr + r)
    if sl > 0:
        ql = tl.load(qsl_ptr + r + 1) - tl.load(qsl_ptr + r)
        p0 = (sl - ql) // PAGE
        p1 = (sl - 1) // PAGE
        offs_kv = tl.arange(0, KV_PAD)
        kmask = offs_kv < NKV
        for p in range(p0, p1 + 1):
            blk = tl.load(bt_ptr + r * stride_btr + p)
            rows = (l * NKV + offs_kv) * NB + blk
            tl.store(valid_ptr + rows, tl.zeros([KV_PAD], tl.int8), mask=kmask)
            slot = tl.load(p2s_ptr + rows, mask=kmask, other=-1)
            res = kmask & (slot >= 0)
            tl.store(p2s_ptr + rows, tl.full([KV_PAD], -1, tl.int32), mask=res)
            tl.store(s2p_ptr + (l * NKV + offs_kv) * S + slot,
                     tl.full([KV_PAD], -1, tl.int32), mask=res)


class Residency:
    """Exact-LRU hot-buffer residency state (all layers, all kv-heads)."""

    def __init__(self, *, num_layers: int, num_blocks: int, n_kv: int,
                 hot_slots: int, page: int, d: int, max_reqs: int,
                 max_pages: int, dtype: torch.dtype, device,
                 offload_k: bool = False):
        self.L, self.NB, self.n_kv = num_layers, num_blocks, n_kv
        self.S, self.page, self.d = hot_slots, page, d
        self.R, self.MP = max_reqs, max_pages
        self.dtype = dtype
        self.offload_k = offload_k
        dev = torch.device(device)
        self.device = dev
        i32, i8 = torch.int32, torch.int8

        # ---- VRAM hot buffer: bounded by S (working set), NOT NB ---------- #
        # THE only KV bytes resident under mem-kv (besides the r8 summary): the
        # bounded hot cache of fetched pages. mem-kv adds a parallel K hot cache.
        self.hot = torch.zeros((num_layers, n_kv, hot_slots, page, d),
                               dtype=dtype, device=dev)
        self.hot_i16 = self.hot.view(torch.int16)   # bit-exact kernel view
        self.hot_k = (torch.zeros((num_layers, n_kv, hot_slots, page, d),
                                  dtype=dtype, device=dev) if offload_k else None)
        self.hot_k_i16 = self.hot_k.view(torch.int16) if offload_k else None

        # ---- residency maps (NB-sized VRAM bookkeeping; int8/int32) ------- #
        # page2slot[l,kv,blk] = hot slot holding this physical block, or -1.
        self.page2slot = torch.full((num_layers, n_kv, num_blocks), -1,
                                    dtype=i32, device=dev)
        # slot2page[l,kv,slot] = physical block resident in this slot, or -1.
        self.slot2page = torch.full((num_layers, n_kv, hot_slots), -1,
                                    dtype=i32, device=dev)
        # exact-LRU clock per slot; a slot touched/claimed this step == clock.
        self.lru_clock = torch.zeros((num_layers, n_kv, hot_slots),
                                     dtype=i32, device=dev)
        # pool_valid[l,kv,blk] = 1 once this block's V is flushed to the pinned
        # pool (a zero-copy read is then legal). Owned jointly with staging.
        self.pool_valid = torch.zeros((num_layers, n_kv, num_blocks),
                                      dtype=i8, device=dev)
        self.clock = torch.ones((1,), dtype=i32, device=dev)
        self.overflow = torch.zeros((1,), dtype=i32, device=dev)

        # ---- per-step gather scratch (request-row keyed) ------------------ #
        self.miss_pages = torch.zeros((max_reqs, n_kv, max_pages),
                                      dtype=i32, device=dev)
        self.miss_cnt = torch.zeros((max_reqs, n_kv), dtype=i32, device=dev)
        self.victim_slots = torch.zeros((max_reqs, n_kv, max_pages),
                                        dtype=i32, device=dev)

        elem = torch.empty(0, dtype=dtype).element_size()
        hot_n = self.hot.numel() + (self.hot_k.numel() if offload_k else 0)
        self.hot_gib = hot_n * elem / 2**30
        self.maps_gib = sum(t.numel() * t.element_size() for t in (
            self.page2slot, self.slot2page, self.lru_clock, self.pool_valid,
            self.miss_pages, self.miss_cnt, self.victim_slots)) / 2**30

    # --------------------------------------------------------------------- #
    # Per-step lifecycle (outside the graph unless noted)                   #
    # --------------------------------------------------------------------- #
    def invalidate_written(self, block_table, seq_lens, query_start_loc) -> None:
        """Drop residency + pool validity for every page (re)written this step
        (block reuse / chunked prefill make the hot copy AND the pinned copy
        stale). Runs before flush/gather so a rewritten page re-stages.

        SCAFFOLD: torch reference. TODO port ``_vt_invalidate_kernel``
        (graph-safe, no host sync). Keyed by physical block so it clears both
        maps and ``pool_valid`` for the affected blocks across all layers/heads.
        """
        n_req = seq_lens.shape[0]
        kv_pad = max(1, triton.next_power_of_2(self.n_kv))
        _invalidate_kernel[(self.L, n_req)](
            block_table, seq_lens, query_start_loc,
            self.page2slot, self.slot2page, self.pool_valid,
            self.NB, self.S, block_table.stride(0),
            PAGE=self.page, NKV=self.n_kv, KV_PAD=kv_pad, num_warps=1)

    def gather(self, layer, page_table, page_cnt, block_table, seq_lens,
               n_req, staging, pool, poolk=None) -> None:
        """Bring every selected MISS page into ``hot`` (I2), choosing victim
        slots by exact LRU (I3). Source: pinned pool if ``pool_valid`` else the
        staging ``v_pool`` via ``vbo`` (with lazy write-through to the pinned
        pool). SINGLE owner of the (hot, page2slot, slot2page) transition.

        Graph-safe (fixed grids, no host sync). ``pool`` is the MappedHostVPool
        device view (int16). Fuses miss_diff + select_victims + gather; the LRU
        clock advances once per call so the previous step's touches become
        evictable. Ports _vt_miss_diff/_select_victims/_p2_gather (dram_tier.py).
        """
        self.gather_plan(layer, page_table, page_cnt, block_table, seq_lens,
                         n_req)
        self.gather_fetch(layer, staging, pool, n_req, poolk=poolk)

    def gather_plan(self, layer, page_table, page_cnt, block_table, seq_lens,
                    n_req) -> None:
        """miss_diff + select_victims: decide which selected pages MISS the hot
        buffer and which victim slot each takes (updates page2slot/slot2page/LRU
        and writes miss_cnt/victim_slots). No V movement -> cheap; the fetch is
        gather_fetch. Split out so (a) the residency HIT RATE = 1 - miss_cnt/sel
        is measurable and (b) the PCIe fetch can be overlapped with compute."""
        l = layer
        n_kv, S, NB, MP = self.n_kv, self.S, self.NB, self.MP
        self.clock += 1
        _miss_diff_kernel[(n_req, n_kv)](
            page_table, page_cnt, block_table, seq_lens,
            self.page2slot[l], self.lru_clock[l], self.clock,
            self.miss_pages, self.miss_cnt,
            NB, S, MP, n_kv, block_table.stride(0), page_table.stride(0),
            page_cnt.stride(0), PAGE=self.page, BLOCK=256, num_warps=4)
        _select_victims_kernel[(n_kv,)](
            self.miss_pages, self.miss_cnt,
            self.page2slot[l], self.slot2page[l], self.lru_clock[l], self.clock,
            self.victim_slots, self.overflow,
            n_req, NB, S, MP, self.miss_pages.stride(0),
            self.miss_cnt.stride(0), BLOCK_S=1024, num_warps=4)

    def gather_fetch(self, layer, staging, pool, n_req, stream=None,
                     poolk=None) -> None:
        """Fetch the planned MISS pages' V (and K, mem-kv) into their hot victim
        slots (pinned zero-copy or staged write-through). The ONLY PCIe-touching
        step; runs on ``stream`` if given (async prefetch / double-buffer)."""
        l = layer
        n_kv, S, NB, MP = self.n_kv, self.S, self.NB, self.MP
        vp = staging.v_pool[l]                     # (NV, page, n_kv, d) bf16
        has_k = self.hot_k_i16 is not None
        kp = staging.k_pool[l] if has_k else vp
        hotk = self.hot_k_i16[l] if has_k else self.hot_i16[l]
        pk = poolk if has_k else pool
        ctx = torch.cuda.stream(stream) if stream is not None \
            else contextlib.nullcontext()
        with ctx:
            _gather_kernel[(n_req * MP, n_kv)](
                pool, self.hot_i16[l], vp, staging.vbo, self.pool_valid[l],
                pk, hotk, kp,
                self.miss_pages, self.miss_cnt, self.victim_slots,
                l, NB, S, MP, n_kv, self.miss_pages.stride(0),
                self.miss_cnt.stride(0),
                vp.stride(0), vp.stride(1), vp.stride(2),
                PAGE=self.page, D=self.d, HAS_K=has_k, num_warps=4)

    def build_dma_plan(self, layer, pool, n_req, poolk=None):
        """Build the copy-engine descriptor list for the FLUSHED (pinned) miss
        pages: coalesce contiguous physical-block RUNS per kv-head into strided
        2-D copies (``dst`` = hot slots at ``page*d`` stride, ``src`` = pinned
        pool at ``n_kv*page*d`` block stride). Returns (runs, has_staged) where
        ``runs`` is a list of (dst, src, dpitch, spitch, width_bytes, height,
        dstk, srck) tuples. This is the HOST-side, D2H-syncing, python-loop part
        -- for a PREFETCH it runs during the PREVIOUS step's compute so its cost
        is hidden; ``issue_dma_plan`` is the cheap per-step copy issue.

        Bit-exact: pure int16 byte copy (pool + hot are the int16 view), identical
        bits to the UVA kernel; only the transport (copy engine vs SM) differs.
        """
        l = layer
        n_kv, S, NB = self.n_kv, self.S, self.NB
        page, d = self.page, self.d
        pbytes = page * d * 2
        blk_stride_b = n_kv * page * d * 2
        slot_stride_b = page * d * 2
        has_k = self.hot_k_i16 is not None
        mcnt = self.miss_cnt.to("cpu")
        mpage = self.miss_pages.to("cpu")
        vict = self.victim_slots.to("cpu")
        valid = self.pool_valid[l].to("cpu")
        pool_base = pool.data_ptr()
        hot_base = self.hot_i16[l].data_ptr()
        poolk_base = poolk.data_ptr() if (has_k and poolk is not None) else None
        hotk_base = self.hot_k_i16[l].data_ptr() if has_k else None
        pool_l_off = ((l * NB) * n_kv) * page * d * 2
        runs = []
        has_staged = False
        for r in range(n_req):
            for kv in range(n_kv):
                m = int(mcnt[r, kv])
                i = 0
                while i < m:
                    slot = int(vict[r, kv, i]); blk = int(mpage[r, kv, i])
                    if slot < 0:
                        i += 1; continue
                    if int(valid[kv, blk]) == 0:
                        has_staged = True; i += 1; continue
                    run = 1
                    while (i + run < m and int(vict[r, kv, i + run]) == slot + run
                           and int(mpage[r, kv, i + run]) == blk + run
                           and int(valid[kv, blk + run]) == 1):
                        run += 1
                    src = pool_base + pool_l_off + (blk * n_kv + kv) * pbytes
                    dst = hot_base + (kv * S + slot) * pbytes
                    dstk = (hotk_base + (kv * S + slot) * pbytes) if has_k else 0
                    srck = (poolk_base + pool_l_off + (blk * n_kv + kv) * pbytes) \
                        if has_k else 0
                    runs.append((dst, src, slot_stride_b, blk_stride_b, pbytes,
                                 run, dstk, srck))
                    i += run
        return runs, has_staged

    def issue_dma_plan(self, runs, copy_engine, stream) -> None:
        """Issue a prebuilt DMA plan on ``stream`` (pure copy-engine; no D2H, no
        python-per-miss beyond iterating the coalesced runs). This is the part
        that overlaps decode."""
        has_k = self.hot_k_i16 is not None
        for dst, src, dpitch, spitch, wbytes, height, dstk, srck in runs:
            if height == 1:
                copy_engine.memcpy_async(dst, src, wbytes, stream)
                if has_k:
                    copy_engine.memcpy_async(dstk, srck, wbytes, stream)
            else:
                copy_engine.memcpy_2d_async(dst, dpitch, src, spitch, wbytes,
                                            height, stream)
                if has_k:
                    copy_engine.memcpy_2d_async(dstk, dpitch, srck, spitch,
                                                wbytes, height, stream)

    def gather_fetch_dma(self, layer, staging, pool, n_req, copy_engine,
                         poolk=None, stream=None) -> "torch.cuda.Event":
        """Copy-engine miss fetch (per-step convenience = build + issue). The
        FLUSHED pages go through the copy engine; STAGED (unflushed) pages stay on
        the device-to-device kernel. Returns the side-stream event. NOTE the
        host-side ``build_dma_plan`` dominates per-step; the amortised path is
        prefetch (build during step N-1, ``issue_dma_plan`` in step N)."""
        stream = stream if stream is not None else copy_engine.next_stream()
        runs, has_staged = self.build_dma_plan(layer, pool, n_req, poolk=poolk)
        with torch.cuda.stream(stream):
            self.issue_dma_plan(runs, copy_engine, stream)
        ev = torch.cuda.Event(); ev.record(stream)
        if has_staged:
            self._gather_staged_only(layer, staging, pool, n_req, poolk)
        return ev

    def _gather_staged_only(self, layer, staging, pool, n_req, poolk=None):
        """Fallback for STAGED (unflushed) miss pages: the existing gather kernel
        (device-to-device from v_pool, no PCIe). Flushed pages it re-copies are a
        harmless idempotent overwrite (same bytes); to avoid that we could mask,
        but the staged set is tiny (only the last ~chunk of un-flushed pages)."""
        n_kv, S, NB, MP = self.n_kv, self.S, self.NB, self.MP
        vp = staging.v_pool[layer]
        has_k = self.hot_k_i16 is not None
        kp = staging.k_pool[layer] if has_k else vp
        hotk = self.hot_k_i16[layer] if has_k else self.hot_i16[layer]
        pk = poolk if has_k else pool
        _gather_kernel[(n_req * MP, n_kv)](
            pool, self.hot_i16[layer], vp, staging.vbo, self.pool_valid[layer],
            pk, hotk, kp,
            self.miss_pages, self.miss_cnt, self.victim_slots,
            layer, NB, S, MP, n_kv, self.miss_pages.stride(0),
            self.miss_cnt.stride(0),
            vp.stride(0), vp.stride(1), vp.stride(2),
            PAGE=self.page, D=self.d, HAS_K=has_k, num_warps=4)

    # --------------------------------------------------------------------- #
    # Invariant checks (call in debug; cheap, content-independent)          #
    # --------------------------------------------------------------------- #
    def assert_maps_inverse(self, layer: int | None = None) -> None:
        """I2/I3 (residency, structural half): page2slot and slot2page are
        mutual inverses -- one slot per page, one page per slot. Content ("hot
        holds the right bytes") is enforced by the single-owner gather; this
        checks the addressing half without touching V."""
        layers = range(self.L) if layer is None else [layer]
        for l in layers:
            for kv in range(self.n_kv):
                s2p = self.slot2page[l, kv]
                occ = s2p >= 0
                pages = s2p[occ]
                # one page per slot: occupied slots hold distinct pages
                if pages.numel() != torch.unique(pages).numel():
                    raise AssertionError(
                        f"I3 residency: layer {l} kv {kv} has two slots holding "
                        "the same page (slot2page not injective)")
                # mutual inverse: page2slot[slot2page[slot]] == slot
                slots = torch.nonzero(occ, as_tuple=False).flatten()
                back = self.page2slot[l, kv][pages]
                if not torch.equal(back, slots.to(back.dtype)):
                    raise AssertionError(
                        f"I2 residency: layer {l} kv {kv} page2slot/slot2page "
                        "are not mutual inverses (gather broke single-owner)")

    def bytes_report(self) -> str:
        return (f"hot={self.hot_gib:.3f}GiB (S={self.S} slots/kv, "
                f"working-set bound) maps={self.maps_gib:.3f}GiB "
                f"(NB={self.NB} int8/int32 bookkeeping)")
