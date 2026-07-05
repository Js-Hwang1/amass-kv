"""Tier -- the DRAM offload facade + the ``TierVSource`` Stage-B seam.

Composes the three owned pieces into one object the backend drives:

  pool       MappedHostVPool  -- pinned host V (keyed by physical block, DRAM)
  residency  Residency        -- hot buffer + exact-LRU + gather (I2/I3)
  staging    Staging          -- v_pool + per-step flush/alloc lifecycle (I1/I4)

``begin_step`` is the ONE per-step host entry (called by the builder OUTSIDE the
graph, like the r8 refresh): it runs the residency invalidation then the staging
stage/flush/alloc lifecycle in the fixed I4 order, and (in debug) checks all four
invariants. ``TierVSource`` is what Stage B consumes as the ``V_SRC==1`` source,
mirroring ``ResidentVSource`` -- it carries the tier's pointer union WITHOUT the
decode kernel knowing any tier internals (AMASS_DESIGN.md 8.1).

SCAFFOLD: the tier ALLOCATES and its lifecycle bookkeeping RUNS (invariants
exercised). The actual V byte movement (write-path scatter, flush copy, residency
gather) and the ``V_SRC==1`` tiered decode load are the documented FOLLOW-UP; see
tier/README.md. mem-v therefore runs correct-by-fallback (stock FA over resident
V) today while the tier machinery is validated in shadow.
"""
from __future__ import annotations

import os
from typing import Tuple

import torch

from .pool import MappedHostVPool
from .residency import Residency
from .staging import Staging

# Debug latch: run the (host-syncing) invariant asserts inside begin_step.
_ASSERT = os.environ.get("AMASS_TIER_ASSERT") not in (None, "", "0")


class Tier:
    """DRAM V-offload tier (mem-v). Facade over pool + residency + staging."""

    def __init__(self, *, num_layers: int, num_blocks: int, n_kv: int,
                 page: int, d: int, hot_slots: int, max_reqs: int,
                 max_pages: int, max_tokens: int, dtype: torch.dtype, device,
                 v_blocks: int | None = None, max_pool_gb: float = 256.0,
                 offload_k: bool = False):
        self.L, self.NB, self.n_kv = num_layers, num_blocks, n_kv
        self.page, self.d = page, d
        self.device = torch.device(device)
        self.dtype = dtype
        self.offload_k = offload_k

        # V pinned host pool (mem-v + mem-kv); K pinned host pool (mem-kv only).
        self.pool = MappedHostVPool(
            num_layers=num_layers, num_blocks=num_blocks, n_kv=n_kv, page=page,
            d=d, dtype=dtype, device=device, max_pool_gb=max_pool_gb)
        self.pool_k = (MappedHostVPool(
            num_layers=num_layers, num_blocks=num_blocks, n_kv=n_kv, page=page,
            d=d, dtype=dtype, device=device, max_pool_gb=max_pool_gb)
            if offload_k else None)
        self.residency = Residency(
            num_layers=num_layers, num_blocks=num_blocks, n_kv=n_kv,
            hot_slots=hot_slots, page=page, d=d, max_reqs=max_reqs,
            max_pages=max_pages, dtype=dtype, device=device, offload_k=offload_k)
        self.staging = Staging(
            num_layers=num_layers, num_blocks=num_blocks, n_kv=n_kv, page=page,
            d=d, max_reqs=max_reqs, max_pages=max_pages, max_tokens=max_tokens,
            dtype=dtype, device=device, v_blocks=v_blocks, offload_k=offload_k)
        # capacity is a static property of the sizing -> check once at build.
        self.staging.assert_capacity()

    @classmethod
    def from_config(cls, cfg, *, num_layers, num_blocks, n_kv, page, d,
                    max_reqs, max_pages, max_tokens, dtype, device) -> "Tier":
        """Build from the single AmassConfig (mem variants only)."""
        if not cfg.is_mem:
            raise ValueError(f"Tier requires a mem variant, got {cfg.variant!r}")
        return cls(
            num_layers=num_layers, num_blocks=num_blocks, n_kv=n_kv, page=page,
            d=d, hot_slots=cfg.hot_slots, max_reqs=max_reqs, max_pages=max_pages,
            max_tokens=max_tokens, dtype=dtype, device=device,
            v_blocks=cfg.v_blocks, max_pool_gb=cfg.max_pool_gb,
            offload_k=cfg.offload_k)

    # --------------------------------------------------------------------- #
    # Per-step host entry (OUTSIDE the graph; builder drives it).           #
    # --------------------------------------------------------------------- #
    def begin_step(self, block_table, seq_lens, query_start_loc,
                   max_query_len: int, n_tokens: int,
                   capture: bool = False) -> None:
        """Residency invalidation + staging stage/flush/alloc lifecycle. Fixed
        order: invalidate (drops stale residency/validity) THEN staging (whose
        internal order is the I4 mechanism)."""
        self.residency.invalidate_written(block_table, seq_lens,
                                           query_start_loc)
        self.staging.begin_step(
            block_table, seq_lens, query_start_loc, max_query_len, n_tokens,
            pool=self.pool.dev_view, pool_valid=self.residency.pool_valid,
            capture=capture,
            poolk=self.pool_k.dev_view if self.pool_k is not None else None)
        if _ASSERT and not capture:
            self.assert_invariants(block_table, seq_lens, query_start_loc)

    def assert_invariants(self, block_table, seq_lens, query_start_loc) -> None:
        """Check I1-I4's structurally-checkable halves (host-syncing; debug)."""
        # Graph-safe valloc cannot raise; the exhaustion latch makes I1's loud
        # half a debug check (build-time assert_capacity is the static guard).
        if int(self.staging.overflow.item()) != 0:
            raise AssertionError(
                "I1 capacity: staging free-stack UNDERFLOWED (a written page got "
                "no staging block -> would lose V). Raise v_blocks to cover the "
                "concurrent-prefill peak (see Staging.assert_capacity).")
        self.staging.assert_free_partition()                       # I3 staging
        self.residency.assert_maps_inverse()                       # I2/I3 resid
        self.staging.assert_no_lost_v(block_table, seq_lens,       # I1
                                      query_start_loc,
                                      self.residency.pool_valid)
        # I4 is enforced by begin_step's call order (not a state predicate).

    # --------------------------------------------------------------------- #
    # FOLLOW-UP hooks (documented; not wired in the scaffold).              #
    # --------------------------------------------------------------------- #
    def write_kv(self, lidx, value, n_tokens, key=None) -> None:
        """In-graph write path: scatter this step's V (and K, mem-kv) into the
        staging pool(s) via ``staging.v_slot_mapping``. Runs AFTER begin_step so
        the scatter targets the (re)allocated staging slots. For mem-v, K stays
        in the engine cache (stock reshape_and_cache); only V is offloaded."""
        self.staging.scatter_kv(lidx, key, value, n_tokens)

    def step(self, lidx, page_table, page_cnt, block_table, seq_lens,
             n_req) -> None:
        """Per-layer residency gather feeding Stage B's hot buffer: miss pages
        (selected, complete, not resident) are fetched from pinned/staged into
        exact-LRU victim slots; hits stay resident. Turns the per-step full-
        shortlist PCIe fetch into a MISS-only fetch (K+V for mem-kv). Runs before
        the layer's Stage-B decode reads K/V."""
        poolk = self.pool_k.dev_view if self.pool_k is not None else None
        self.residency.gather(lidx, page_table, page_cnt, block_table, seq_lens,
                              n_req, self.staging, self.pool.dev_view,
                              poolk=poolk)

    @property
    def copy_engine(self):
        """Lazy copy-engine DMA (double-buffered side streams)."""
        ce = getattr(self, "_copy_engine", None)
        if ce is None:
            from .dma import CopyEngine
            ce = CopyEngine()
            self._copy_engine = ce
        return ce

    def step_dma(self, lidx, page_table, page_cnt, block_table, seq_lens,
                 n_req, wait: bool = True):
        """Copy-engine variant of ``step``: plan the misses on device, then fetch
        the FLUSHED miss pages through the hardware COPY ENGINE (not the SM-holding
        UVA kernel), so the PCIe transfer overlaps decode. Returns the side-stream
        event; if ``wait`` the caller's stream waits on it (in-layer, dependent),
        else the caller overlaps it (prefetch). STAGED misses go to the kernel."""
        poolk = self.pool_k.dev_view if self.pool_k is not None else None
        self.residency.gather_plan(lidx, page_table, page_cnt, block_table,
                                   seq_lens, n_req)
        ev = self.residency.gather_fetch_dma(
            lidx, self.staging, self.pool.dev_view, n_req, self.copy_engine,
            poolk=poolk)
        if wait:
            torch.cuda.current_stream().wait_event(ev)
        return ev

    def vsource(self, lidx: int) -> "TierVSource":
        """The Stage-B KV source for layer ``lidx`` (V_SRC==1, and K_SRC==1 for
        mem-kv)."""
        return TierVSource(self, lidx)

    def bytes_report(self) -> str:
        tag = "mem-kv" if self.offload_k else "mem-v"
        kpool = (f" + K-pool {self.pool_k.pool_gib:.2f}GiB DRAM"
                 if self.pool_k is not None else "")
        return (f"[amass tier {tag}] {self.pool.vram_note()}{kpool} | "
                f"{self.residency.bytes_report()} | "
                f"{self.staging.bytes_report()}")


class TierVSource:
    """AMASS-mem V source (mirrors ``ResidentVSource``): the ``V_SRC==1`` seam.

    Stage B consumes this exactly like the fast path's ResidentVSource -- same
    ``SRC_ID`` + ``args()`` protocol -- so ``attention/decode.py`` stays tier-
    blind. The tier's 3-way V read (hot buffer | staged v_pool | pinned pool) is
    a compile-time ``V_SRC==1`` branch inside the decode kernel; this object
    supplies the pointer UNION that branch needs.

    SCAFFOLD. ``args()`` returns the staging ``v_pool`` layer view as the primary
    base so the shared decode wrapper's ``v_ptr, svb, svt, svh = vsource.args()``
    line is satisfied and addresses valid memory (the current V_SRC==1 arm is a
    documented resident-shaped fallback). The FULL union the follow-up kernel
    consumes is exposed by ``union()``; wiring the extra pointers into the
    kernel's V_SRC==1 arm is the follow-up. Until then mem-v decode delegates to
    stock FA (backend/attn.py), so ``args()`` is never on a live tiered path.
    """

    SRC_ID = 1

    def __init__(self, tier: Tier, lidx: int):
        self._tier = tier
        self._lidx = lidx
        self._vp = tier.staging.v_pool[lidx]        # (NV, page, n_kv, d)
        # mem-kv: K is offloaded too -> the decode kernel's K_SRC==1 arm reads K
        # from the tier (staged k_pool | hot_k | pinned K pool). mem-v: K_SRC=0.
        self.K_SRC = 1 if tier.offload_k else 0

    def args(self) -> Tuple[torch.Tensor, int, int, int]:
        # Primary base = staging v_pool layer view (the STAGED source and the
        # decode kernel's ``v_ptr``); block/token/head strides match resident V.
        vp = self._vp
        return vp, vp.stride(0), vp.stride(1), vp.stride(2)

    def k_args(self) -> Tuple[torch.Tensor, int, int, int]:
        """mem-kv K base = the staging k_pool layer view (the STAGED K source and
        the decode kernel's ``k_ptr`` under K_SRC==1); same paged layout as V."""
        kp = self._tier.staging.k_pool[self._lidx]
        return kp, kp.stride(0), kp.stride(1), kp.stride(2)

    def k_tier_ptrs(self):
        """The K-side (hot_k, pinned-K-pool) pointers the K_SRC==1 arm consumes
        (page2slot/vbo/pool_valid are SHARED with V -> already in tier_ptrs)."""
        t = self._tier
        return t.residency.hot_k[self._lidx], t.pool_k.dev_view

    def tier_ptrs(self):
        """The extra pointers the ``V_SRC==1`` decode arm consumes beyond
        ``args()``: (hot, page2slot, vbo, pinned-pool, lidx, NB, S).

          * hot        r.hot[lidx]        bf16 (n_kv, S, page, d)  -- resident hits
          * page2slot  r.page2slot[lidx]  int32 (n_kv, NB)         -- hot addressing
          * vbo        s.vbo              int32 (NB,)              -- staged addressing
          * pinned     pool.dev_view      int16 UVA device ptr     -- flushed, zero-copy
        The kernel keys the 3-way select on ``page2slot>=0`` (+ page complete),
        ``vbo>=0``, else the pinned pool at ``page_offset(lidx, blk, kv)``."""
        t = self._tier
        l = self._lidx
        return (t.residency.hot[l], t.residency.page2slot[l], t.staging.vbo,
                t.pool.dev_view, l, t.NB, t.residency.S)

    def union(self) -> dict:
        """Full pointer union for the FOLLOW-UP V_SRC==1 decode load.

        The kernel selects, per (block, kv-head, token), among:
          * hot buffer     hot[lidx]      via page2slot[lidx][blk]  (complete+resident)
          * staged v_pool  v_pool[lidx]   via vbo[blk]              (partial/unflushed)
          * pinned pool    pool.dev_view  at pool.page_offset(...)  (flushed, zero-copy)
        keyed by ``pool_valid[lidx]`` + ``vbo``. See _p2_sparse_decode_split_kernel
        in vllm/kernels/dram_tier.py for the exact selection + masking.
        """
        t = self._tier
        l = self._lidx
        r, s = t.residency, t.staging
        return dict(
            lidx=l, NB=t.NB, S=r.S, n_kv=t.n_kv, page=t.page, d=t.d,
            hot=r.hot[l], hot_i16=r.hot_i16[l],
            page2slot=r.page2slot[l], pool_valid=r.pool_valid[l],
            v_pool=s.v_pool[l], vbo=s.vbo,
            pool=t.pool.dev_view, pool_base=t.pool.page_offset(l, 0, 0),
        )
