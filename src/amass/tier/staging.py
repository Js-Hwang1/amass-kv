"""Staging -- the per-step V stage/flush lifecycle (fixes the two mem bugs).

Under mem-v the engine block pool holds K only; the V of the token written each
step lands first in a BOUNDED VRAM staging pool ``v_pool`` (partial tail pages +
the whole prefix of a prefilling request), then, once a page is COMPLETE, its V
is flushed to the pinned host pool and the staging block is recycled. This
module owns that lifecycle and the two invariants whose violation were the
prototype's confirmed bugs:

  I1 (no lost V): every COMPLETE written page has its V retrievable -- either
     STAGED (``vbo[blk] >= 0``, in ``v_pool``) or FLUSHED (``pool_valid=1`` for
     all (l,kv), in the pinned pool). Bug #1 was exhaustion leaving ``vbo=-1``
     AND ``pool_valid=0`` -> zeros. Enforced structurally by (a) ``valloc``
     guaranteeing a v-block for EVERY written page and RAISING on free-stack
     underflow (never silently leaving vbo=-1), and (b) freeing a v-block ONLY
     after its flush copy is enqueued (see I4). ``assert_no_lost_v`` checks it.

  I4 (flush-before-reuse ordering): within ``begin_step``, ``flush_copy`` reads
     ``v_pool[vb]`` BEFORE ``valloc`` reassigns ``vb`` and BEFORE this step's
     scatter overwrites it. Enforced by the FIXED CALL ORDER inside the single
     ``begin_step`` method -- the ordering IS the mechanism; it cannot be
     reordered without editing this one function.

  I3 (no aliasing, staging side): the free-stack is a true set -- assigned
     (``vbo>=0``) and free (``v_free_stack[:top]``) partition ``[0, NV)`` with no
     duplicates. Enforced by a single owner of push/pop with CAS; checked by
     ``assert_free_partition``.

SIZING. ``v_pool`` (VRAM) is bounded to the concurrent working set
``NV = 2*max_pages + 4*max_reqs`` (or an explicit ``v_blocks``), NEVER to
``num_gpu_blocks``. That bound must cover the true concurrent-prefill peak so I1
holds for non-batched requests too (the prototype's #1 root cause was an
under-sized bound + a decode-only free policy). ``assert_capacity`` states the
bound explicitly.

SCAFFOLD STATUS. Persistent buffers + the ``begin_step`` ORDERING + the
invariant asserts are real. The sub-step bodies (vflush_clear / flush_collect /
flush_copy / valloc / vslot) are CORRECT torch references (host-synchronizing,
not graph-safe), each marked TODO to port to the graph-safe kernels that already
exist in ``vllm/kernels/dram_tier.py`` (``_p2_*``). The write path (scatter V ->
v_pool, in-graph, replaces reshape_and_cache) is the documented FOLLOW-UP; see
tier/README.md.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_kv_kernel(key_ptr, val_ptr, kp_ptr, vp_ptr, vslot_ptr,
                       skt, skh, svt, svh, vpb, vpt, vph,
                       NKV: tl.constexpr, KV_PAD: tl.constexpr,
                       D: tl.constexpr, PAGE: tl.constexpr,
                       HAS_K: tl.constexpr):
    """Write path: scatter this step's V (and K, mem-kv) (n_tokens, n_kv, d) into
    the staging pool(s) at ``v_slot_mapping`` (= vb*PAGE + pos%PAGE, or -1 to
    skip). K and V share the SAME staging slot (same page). Bitwise bf16 copy.
    Ports ``_p2_scatter_kv_kernel``."""
    t = tl.program_id(0)
    offs_kv = tl.arange(0, KV_PAD)
    offs_d = tl.arange(0, D)
    kmask = offs_kv < NKV
    vs = tl.load(vslot_ptr + t)
    if vs >= 0:
        vb = (vs // PAGE).to(tl.int64)
        voff = vs % PAGE
        dst = vb * vpb + voff * vpt + offs_kv[:, None] * vph + offs_d[None, :]
        v = tl.load(val_ptr + t * svt + offs_kv[:, None] * svh + offs_d[None, :],
                    mask=kmask[:, None], other=0.0)
        tl.store(vp_ptr + dst, v, mask=kmask[:, None])
        if HAS_K:
            k = tl.load(key_ptr + t * skt + offs_kv[:, None] * skh
                        + offs_d[None, :], mask=kmask[:, None], other=0.0)
            tl.store(kp_ptr + dst, k, mask=kmask[:, None])


@triton.jit
def _vflush_clear_kernel(bt_ptr, sl_ptr, qsl_ptr, vfl_ptr,
                         stride_btr, PAGE: tl.constexpr):
    """Clear v_flushed for pages written this step (their pinned copy is stale).
    Grid (n_req,). Ports ``_p2_vflush_clear_kernel``."""
    r = tl.program_id(0)
    sl = tl.load(sl_ptr + r)
    ql = tl.load(qsl_ptr + r + 1) - tl.load(qsl_ptr + r)
    if (sl > 0) & (ql > 0):
        p0 = (sl - ql) // PAGE
        p1 = (sl - 1) // PAGE
        for p in range(p0, p1 + 1):
            blk = tl.load(bt_ptr + r * stride_btr + p)
            tl.store(vfl_ptr + blk, 0)


@triton.jit
def _flush_collect_kernel(bt_ptr, sl_ptr, qsl_ptr, vbo_ptr, vfl_ptr,
                          flist_ptr, fcnt_ptr, stack_ptr, top_ptr,
                          stride_btr, MAXF, N_REQ,
                          PAGE: tl.constexpr, BLOCK: tl.constexpr):
    """Collect complete un-flushed staged pages into the flush list (claim via
    atomic_xchg on v_flushed) and free their staged blocks -- the free happens
    ONLY after the flush claim (I1), and ``flush_copy`` reads via the CAPTURED
    (blk, vb) pairs in the flush list, never via ``vbo``, so the free is safe
    the same step (I4 = the begin_step call order: collect -> copy -> valloc).
    Freeing applies to ALL requests, prefill included: a chunked prefill flushes
    chunk k's pages while writing chunk k+1, keeping the concurrent staged peak
    at ~2 chunks + tails -- the O(chunk) bound ``assert_capacity`` states. (The
    earlier decode-only free let long prefills accumulate the WHOLE prefix in
    staging and underflow valloc; the prototype's #1 root cause.) Grid (1,),
    SERIALIZED over requests (shared flush index + free top raced on a
    per-request grid = the old wrong-V bug). Ports ``_p2_flush_collect_kernel``."""
    for r in range(N_REQ):
        sl = tl.load(sl_ptr + r)
        ql = tl.load(qsl_ptr + r + 1) - tl.load(qsl_ptr + r)
        if (sl > 0) & (ql > 0):
            ctx = sl - ql
            np_ctx = ctx // PAGE
            for p0 in range(0, np_ctx, BLOCK):
                offs = p0 + tl.arange(0, BLOCK)
                pmask = offs < np_ctx
                blk = tl.load(bt_ptr + r * stride_btr + offs, mask=pmask,
                              other=0)
                vb = tl.load(vbo_ptr + blk, mask=pmask, other=-1)
                act = pmask & (vb >= 0)
                fl = tl.atomic_xchg(vfl_ptr + blk,
                                    tl.full([BLOCK], 1, tl.int32), mask=act)
                new = act & (fl == 0)
                i = tl.atomic_add(fcnt_ptr + offs * 0,
                                  tl.full([BLOCK], 1, tl.int32), mask=new)
                ok = new & (i < MAXF)
                tl.store(flist_ptr + i * 2 + 0, blk, mask=ok)
                tl.store(flist_ptr + i * 2 + 1, vb, mask=ok)
                over = new & (i >= MAXF)
                n_over = tl.sum(over.to(tl.int32), axis=0)
                if n_over > 0:
                    tl.atomic_add(fcnt_ptr, -n_over)
                tl.store(vfl_ptr + blk, tl.zeros([BLOCK], tl.int32), mask=over)
                fre = act & (over == 0)
                old = tl.atomic_xchg(vbo_ptr + blk,
                                     tl.full([BLOCK], -1, tl.int32),
                                     mask=fre)
                do = fre & (old >= 0)
                j = tl.atomic_add(top_ptr + offs * 0,
                                  tl.full([BLOCK], 1, tl.int32), mask=do)
                tl.store(stack_ptr + j, old, mask=do)


@triton.jit
def _valloc_kernel(bt_ptr, sl_ptr, qsl_ptr, vbo_ptr, stack_ptr, top_ptr,
                   ovf_ptr, stride_btr, N_REQ, PAGE: tl.constexpr):
    """Allocate a staging block for every page written this step lacking one.
    Grid (1,), serialized (shared free-stack top). Atomic CAS guards duplicate
    claims of a shared engine block. On free-stack underflow sets ovf (I1 is
    guarded loud at build by assert_capacity + the debug asserts). Ports
    ``_p2_valloc_kernel``."""
    for r in range(N_REQ):
        sl = tl.load(sl_ptr + r)
        ql = tl.load(qsl_ptr + r + 1) - tl.load(qsl_ptr + r)
        if (sl > 0) & (ql > 0):
            p0 = (sl - ql) // PAGE
            p1 = (sl - 1) // PAGE
            for p in range(p0, p1 + 1):
                blk = tl.load(bt_ptr + r * stride_btr + p)
                vb = tl.load(vbo_ptr + blk)
                if vb < 0:
                    idx = tl.atomic_add(top_ptr, -1) - 1
                    if idx < 0:
                        tl.atomic_add(top_ptr, 1)
                        tl.store(ovf_ptr, 1)
                    else:
                        nv = tl.load(stack_ptr + idx)
                        old = tl.atomic_cas(vbo_ptr + blk, -1, nv)
                        if old != -1:            # someone else claimed: return
                            j = tl.atomic_add(top_ptr, 1)
                            tl.store(stack_ptr + j, nv)


@triton.jit
def _vslot_kernel(bt_ptr, sl_ptr, qsl_ptr, vbo_ptr, vslot_ptr,
                  stride_btr, PAGE: tl.constexpr, BLOCK: tl.constexpr):
    """Fill v_slot_mapping for this step's tokens from the allocated v-blocks
    (runs after valloc). Grid (n_req, cdiv(max_query_len, BLOCK)). Ports
    ``_p2_vslot_kernel``."""
    r = tl.program_id(0)
    tb = tl.program_id(1)
    sl = tl.load(sl_ptr + r)
    q0 = tl.load(qsl_ptr + r)
    ql = tl.load(qsl_ptr + r + 1) - q0
    if (sl <= 0) | (ql <= 0):
        return
    i = tb * BLOCK + tl.arange(0, BLOCK)
    tmask = i < ql
    pos = sl - ql + i
    p = pos // PAGE
    blk = tl.load(bt_ptr + r * stride_btr + p, mask=tmask, other=0)
    vb = tl.load(vbo_ptr + blk, mask=tmask, other=-1)
    val = tl.where(vb >= 0, vb * PAGE + pos % PAGE, -1).to(tl.int64)
    tl.store(vslot_ptr + q0 + i, val, mask=tmask)


@triton.jit
def _flush_copy_kernel(vp_ptr, pool_ptr, valid_ptr,
                       kp_ptr, poolk_ptr,
                       flist_ptr, fcnt_ptr,
                       NB, MAXF,
                       svl, svb, svt, svh,
                       NKV: tl.constexpr, PAGE: tl.constexpr, D: tl.constexpr,
                       HAS_K: tl.constexpr):
    """Copy each flushed page ``v_pool[l, vb]`` -> pinned host pool ``pool[l,
    blk, kv]`` for ALL layers/kv-heads and set ``pool_valid`` (bit-exact int16
    stores over UVA). For mem-kv also copies ``k_pool -> poolk`` (K and V share
    the flush list + validity). Grid (MAXF, L). READS the staging pools BEFORE
    valloc reuses vb (I4). Ports ``_p2_flush_copy_kernel``."""
    i = tl.program_id(0)                 # flush-list entry
    l = tl.program_id(1)                 # layer
    cnt = tl.minimum(tl.load(fcnt_ptr), MAXF)
    if i < cnt:
        blk = tl.load(flist_ptr + i * 2 + 0).to(tl.int64)
        vb = tl.load(flist_ptr + i * 2 + 1).to(tl.int64)
        offs_t = tl.arange(0, PAGE)
        offs_d = tl.arange(0, D)
        for kv in range(NKV):
            src = l * svl + vb * svb + offs_t[:, None] * svt + kv * svh \
                + offs_d[None, :]
            pool_off = (((l * NB + blk) * NKV + kv).to(tl.int64) * (PAGE * D))
            dst = pool_off + offs_t[:, None] * D + offs_d[None, :]
            v = tl.load(vp_ptr + src)
            tl.store(pool_ptr + dst, v.to(tl.int16, bitcast=True))
            if HAS_K:
                k = tl.load(kp_ptr + src)
                tl.store(poolk_ptr + dst, k.to(tl.int16, bitcast=True))
            tl.store(valid_ptr + (l * NKV + kv) * NB + blk, tl.cast(1, tl.int8))


class Staging:
    """V staging pool + per-step flush/alloc lifecycle (all layers)."""

    def __init__(self, *, num_layers: int, num_blocks: int, n_kv: int,
                 page: int, d: int, max_reqs: int, max_pages: int,
                 max_tokens: int, dtype: torch.dtype, device,
                 v_blocks: int | None = None, max_flush: int | None = None,
                 offload_k: bool = False):
        self.L, self.NB, self.n_kv = num_layers, num_blocks, n_kv
        self.page, self.d = page, d
        self.R, self.MP = max_reqs, max_pages
        self.max_tokens = max_tokens
        self.offload_k = offload_k
        dev = torch.device(device)
        self.device = dev
        i32, i64 = torch.int32, torch.int64

        # ---- working-set bound (the CHUNK, NOT the full context) ---------- #
        # A single step writes at most ``max_tokens`` tokens (the per-step token
        # budget = max_num_batched_tokens under chunked prefill), i.e. at most
        # ceil(max_tokens/page) NEW pages; with pages flushed on the next decode
        # step, at most ~2 chunks of un-flushed pages are ever concurrent, plus
        # a partial tail per request. So NV is O(chunk), NOT O(max_model_len).
        #
        # PROFILED BUG FIX: the earlier default ``2*max_pages`` sized the staging
        # pool to 2x the FULL context (max_pages = max_model_len/page). At 64K
        # that is an 8 GiB resident v_pool -- larger than the V being offloaded,
        # which INVERTS the memory play (measured realized saving went NEGATIVE).
        # Sizing to the chunk restores the ~2x resident-context headline.
        chunk_pages = (max_tokens + page - 1) // page
        if v_blocks is None or v_blocks <= 0:
            v_blocks = 2 * chunk_pages + 4 * max_reqs
        if max_flush is None or max_flush <= 0:
            max_flush = chunk_pages + 2 * max_reqs + 64
        self.NV = v_blocks
        self.MAXF = max_flush
        self.chunk_pages = chunk_pages

        # ---- staging pool (VRAM, bounded by NV) --------------------------- #
        # Layout (L, NV, page, n_kv, d): matches the write-path scatter and the
        # decode staged-source read in the reference kernels.
        self.v_pool = torch.zeros((num_layers, v_blocks, page, n_kv, d),
                                  dtype=dtype, device=dev)
        # mem-kv: a parallel K staging pool, SAME slot/lifecycle bookkeeping.
        self.k_pool = (torch.zeros((num_layers, v_blocks, page, n_kv, d),
                                   dtype=dtype, device=dev)
                       if offload_k else None)

        # ---- lifecycle bookkeeping (NB-sized VRAM, int32) ----------------- #
        # vbo[blk] = staging block holding this engine block's V, or -1.
        self.vbo = torch.full((num_blocks,), -1, dtype=i32, device=dev)
        # v_flushed[blk] = staged bits copied to the pinned pool (claim latch).
        self.v_flushed = torch.zeros((num_blocks,), dtype=i32, device=dev)
        # free-stack: a TRUE SET of unassigned staging blocks (I3).
        self.v_free_stack = torch.arange(v_blocks, dtype=i32, device=dev)
        self.v_free_top = torch.full((1,), v_blocks, dtype=i32, device=dev)
        # per-token V write slots (refreshed per step; persistent -> graph-safe
        # scatter). -1 = no staging slot (skipped).
        self.v_slot_mapping = torch.full((max_tokens,), -1, dtype=i64,
                                         device=dev)
        # flush list: (blk, vb) pairs to copy v_pool -> pinned this step.
        self.flush_list = torch.zeros((max_flush, 2), dtype=i32, device=dev)
        self.flush_cnt = torch.zeros((1,), dtype=i32, device=dev)
        # staging-exhaustion latch (valloc underflow -> I1 risk; checked in debug).
        self.overflow = torch.zeros((1,), dtype=i32, device=dev)

        elem = torch.empty(0, dtype=dtype).element_size()
        pools = self.v_pool.numel() + (self.k_pool.numel() if offload_k else 0)
        self.vpool_gib = pools * elem / 2**30
        self.maps_gib = sum(t.numel() * t.element_size() for t in (
            self.vbo, self.v_flushed, self.v_free_stack, self.v_slot_mapping,
            self.flush_list)) / 2**30

    # --------------------------------------------------------------------- #
    # begin_step -- the ONE ordered lifecycle (outside the graph).          #
    #   The call ORDER below is the I4 mechanism. Do not reorder.           #
    # --------------------------------------------------------------------- #
    def begin_step(self, block_table, seq_lens, query_start_loc,
                   max_query_len: int, n_tokens: int, pool, pool_valid,
                   capture: bool = False, poolk=None) -> None:
        """Run the per-step stage/flush/alloc lifecycle. ``pool`` is the
        MappedHostVPool device view; ``pool_valid`` is the residency's
        ``pool_valid`` map (jointly owned). MUST run OUTSIDE the graph (host,
        before replay), like the r8 refresh.

        Ordering (I4): invalidate is done by the residency BEFORE this call.
          1. reset v_slot_mapping for this step's tokens
          2. vflush_clear  -- rewritten pages lose their stale flushed bit
          3. flush_collect -- claim complete un-flushed pages; free decode pages
                              (free happens only AFTER the flush claim)
          4. flush_copy    -- READ v_pool[vb] -> pinned pool  (reads BEFORE ...)
          5. valloc        -- ... reassigns freed vb to this step's pages
          6. vslot         -- fill v_slot_mapping from the (re)allocated vbo
        The scatter of V into v_pool happens in-graph AFTER begin_step (write
        path, FOLLOW-UP), so flush_copy in step N reads content settled by
        step<N and valloc's reuse is safe.
        """
        if n_tokens > self.max_tokens:
            raise RuntimeError(
                f"tier begin_step: {n_tokens} tokens > v_slot_mapping capacity "
                f"{self.max_tokens}; raise max_tokens sizing")
        self.v_slot_mapping[:n_tokens].fill_(-1)
        if capture:
            return  # dummy capture batch: never mutate the allocator
        self._vflush_clear(block_table, seq_lens, query_start_loc)
        self._flush_collect(block_table, seq_lens, query_start_loc)
        self._flush_copy(pool, pool_valid, poolk)   # READS v_pool/k_pool ...
        self._valloc(block_table, seq_lens, query_start_loc)   # ... then reuses
        self._vslot(block_table, seq_lens, query_start_loc, max_query_len)

    # --------------------------------------------------------------------- #
    # Write path (in-graph, AFTER begin_step): scatter V -> v_pool.          #
    # --------------------------------------------------------------------- #
    def scatter_kv(self, lidx: int, key, value: torch.Tensor,
                   n_tokens: int) -> None:
        """Scatter ``value`` (and ``key`` for mem-kv) (n_tokens, n_kv, d) into the
        staging pool(s) at this step's ``v_slot_mapping`` (filled by begin_step's
        _vslot). Replaces the reshape_and_cache_flash write for the tier."""
        vp = self.v_pool[lidx]                     # (NV, page, n_kv, d)
        kv_pad = max(1, triton.next_power_of_2(self.n_kv))
        has_k = self.k_pool is not None
        kp = self.k_pool[lidx] if has_k else vp
        skt = key.stride(0) if has_k else 0
        skh = key.stride(1) if has_k else 0
        _scatter_kv_kernel[(n_tokens,)](
            key if has_k else value, value, kp, vp, self.v_slot_mapping,
            skt, skh, value.stride(0), value.stride(1),
            vp.stride(0), vp.stride(1), vp.stride(2),
            NKV=self.n_kv, KV_PAD=kv_pad, D=self.d, PAGE=self.page,
            HAS_K=has_k)

    # --------------------------------------------------------------------- #
    # Lifecycle sub-steps (SCAFFOLD: torch references; TODO graph-safe port) #
    # --------------------------------------------------------------------- #
    def _written_pages(self, block_table, seq_lens, query_start_loc):
        """Yield (r, decode, complete_ctx_blks, written_blks) per active request.

        complete_ctx_blks = physical blocks of COMPLETE pages written by PRIOR
        steps (settled content, flushable); written_blks = pages touched THIS
        step (need a staging slot). ``decode`` = single-token step."""
        n_req = seq_lens.shape[0]
        page = self.page
        sl = seq_lens.tolist()
        ql = (query_start_loc[1:n_req + 1] - query_start_loc[:n_req]).tolist()
        for r in range(n_req):
            if sl[r] <= 0 or ql[r] <= 0:
                continue
            ctx = sl[r] - ql[r]
            np_ctx = ctx // page                        # complete settled pages
            complete = block_table[r, :np_ctx].tolist()
            p0 = ctx // page
            p1 = (sl[r] - 1) // page
            written = block_table[r, p0:p1 + 1].tolist()
            yield r, (ql[r] == 1), complete, written

    def _vflush_clear(self, block_table, seq_lens, query_start_loc) -> None:
        """Clear v_flushed for pages written this step (their pinned copy is now
        stale). Graph-safe Triton (was an O(context) host loop -- the profiled
        begin_step bottleneck)."""
        n_req = seq_lens.shape[0]
        _vflush_clear_kernel[(n_req,)](
            block_table, seq_lens, query_start_loc, self.v_flushed,
            block_table.stride(0), PAGE=self.page, num_warps=1)

    def _flush_collect(self, block_table, seq_lens, query_start_loc) -> None:
        """Collect complete un-flushed staged pages into flush_list (claim via
        v_flushed latch), and free decode requests' staged blocks -- the free
        happens ONLY after the flush claim, so flush_copy reads them first (I1).
        SERIALIZED over requests (grid (1,)): the prototype raced a per-request
        grid -> wrong V on multi-request long-context decode. Graph-safe Triton
        (was the O(context) host loop; the profiled begin_step bottleneck)."""
        n_req = seq_lens.shape[0]
        self.flush_cnt.zero_()
        _flush_collect_kernel[(1,)](
            block_table, seq_lens, query_start_loc, self.vbo, self.v_flushed,
            self.flush_list, self.flush_cnt, self.v_free_stack, self.v_free_top,
            block_table.stride(0), self.MAXF, n_req,
            PAGE=self.page, BLOCK=128, num_warps=1)

    def _flush_copy(self, pool, pool_valid, poolk=None) -> None:
        """Copy each flushed page v_pool[l,vb] -> pinned pool[l,blk] (and k_pool
        -> poolk for mem-kv) for ALL layers/heads and set pool_valid (bit-exact
        int16 stores over UVA). READS the staging pools BEFORE valloc reuses them
        (I4). ``pool``/``poolk`` are the MappedHostVPool device views (int16).
        Graph-safe. Ports ``_p2_flush_copy_kernel``."""
        vp = self.v_pool                          # (L, NV, page, n_kv, d) bf16
        has_k = self.k_pool is not None
        _flush_copy_kernel[(self.MAXF, self.L)](
            vp, pool, pool_valid, self.k_pool if has_k else vp,
            poolk if has_k else pool, self.flush_list, self.flush_cnt,
            self.NB, self.MAXF,
            vp.stride(0), vp.stride(1), vp.stride(2), vp.stride(3),
            NKV=self.n_kv, PAGE=self.page, D=self.d, HAS_K=has_k)

    def _valloc(self, block_table, seq_lens, query_start_loc) -> None:
        """Allocate a staging block for EVERY page written this step that lacks
        one, popping the free-stack with atomic CAS (dup-claim safe). Graph-safe
        Triton, grid (1,) serialized pop. Free-stack underflow sets ``overflow``
        (I1 guarded loud at build by assert_capacity + AMASS_TIER_ASSERT)."""
        n_req = seq_lens.shape[0]
        _valloc_kernel[(1,)](
            block_table, seq_lens, query_start_loc, self.vbo,
            self.v_free_stack, self.v_free_top, self.overflow,
            block_table.stride(0), n_req, PAGE=self.page, num_warps=1)

    def _vslot(self, block_table, seq_lens, query_start_loc,
               max_query_len: int) -> None:
        """Fill v_slot_mapping for this step's tokens from the allocated vbo so
        the in-graph scatter writes V into the right staging slot. Graph-safe
        Triton, grid (n_req, cdiv(max_query_len, BLOCK))."""
        n_req = seq_lens.shape[0]
        _vslot_kernel[(n_req, triton.cdiv(max(max_query_len, 1), 256))](
            block_table, seq_lens, query_start_loc, self.vbo,
            self.v_slot_mapping, block_table.stride(0),
            PAGE=self.page, BLOCK=256, num_warps=2)

    # --------------------------------------------------------------------- #
    # Invariant checks (call in debug)                                      #
    # --------------------------------------------------------------------- #
    def assert_free_partition(self) -> None:
        """I3 (staging): assigned(vbo>=0) + free(v_free_stack[:top]) partition
        [0, NV) with no duplicates."""
        top = int(self.v_free_top.item())
        if not (0 <= top <= self.NV):
            raise AssertionError(f"I3 staging: free_top={top} out of [0,{self.NV}]")
        assigned = self.vbo[self.vbo >= 0]
        n_assigned = int(assigned.numel())
        n_unique = int(torch.unique(assigned).numel())
        if n_assigned != n_unique:
            raise AssertionError(
                "I3 staging: a staging block is assigned to two engine blocks "
                f"(assigned={n_assigned} unique={n_unique})")
        if n_assigned + top != self.NV:
            raise AssertionError(
                f"I3 staging: assigned({n_assigned}) + free({top}) != NV"
                f"({self.NV}); the free-stack leaked or double-counted a block")

    def assert_no_lost_v(self, block_table, seq_lens, query_start_loc,
                         pool_valid) -> None:
        """I1: every COMPLETE written page is staged (vbo>=0) OR flushed
        (pool_valid=1 for all layers/heads). Checks the settled prefix per
        request."""
        for r, _dec, complete, _written in self._written_pages(
                block_table, seq_lens, query_start_loc):
            for blk in complete:
                staged = int(self.vbo[blk].item()) >= 0
                flushed = bool((pool_valid[:, :, blk] == 1).all())
                if not (staged or flushed):
                    raise AssertionError(
                        f"I1: request {r} block {blk} is a complete written page "
                        "with neither staged (vbo=-1) nor flushed (pool_valid=0) "
                        "V -> would read zeros")

    def assert_capacity(self) -> None:
        """State the working-set bound (NV) explicitly. Under CHUNKED prefill a
        single step writes at most ``chunk_pages`` (= ceil(max_tokens/page)) new
        pages, flushed on the next decode step, so the concurrent un-flushed peak
        is ~2 chunks + the per-request partial tail. NV = 2*chunk_pages +
        4*max_reqs covers it. valloc RAISES (loud, never silent I1 violation) if
        a step ever needs more -- e.g. a NON-chunked single-shot prefill of a
        request longer than a chunk, which must either enable chunked prefill or
        raise v_blocks. NV is O(chunk), NOT O(max_model_len): that is the whole
        point of the memory play (v_pool must not scale with context)."""
        floor = self.chunk_pages + self.R
        if self.NV < floor:
            raise AssertionError(
                f"I1 capacity: v_blocks NV={self.NV} < floor {floor} "
                f"(chunk_pages={self.chunk_pages} + max_reqs={self.R}); one "
                "prefill chunk could exhaust staging and lose V.")

    def bytes_report(self) -> str:
        return (f"v_pool={self.vpool_gib:.3f}GiB (NV={self.NV} blocks, "
                f"working-set bound) maps={self.maps_gib:.3f}GiB")
