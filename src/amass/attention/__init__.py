"""Stage B — sparse paged attention over the selected pages.

Variant-agnostic: the decode kernel attends the pages chosen by Stage A and
reads V through an injected V-SOURCE (§8.1 of AMASS_DESIGN.md), so this module
has ZERO knowledge of the DRAM tier:

  * fast  -> ResidentVSource: the engine V half.
  * mem-v -> TierVSource: hot buffer | staged v_pool | pinned pool (3-way).

The V source is a compile-time (``constexpr``) source-id plus the union of
pointers, so there is ONE split-K decode kernel with a specialized V load — not
a python branch per tile. Two fixed-shape, host-sync-free launches (split +
merge), CUDA-graph-safe.

Contract (ported from sparse_decode.py):
  sparse_paged_decode_batched(q, kv_cache, block_table, seq_lens, st, out,
                              vsource)         2 launches, writes out in place
  merge_splits_kernel                          split-K merge
  ResidentVSource / VSource                    the fast/mem V-source seam
"""
from __future__ import annotations

from .decode import (  # noqa: F401
    ResidentVSource,
    VSource,
    ensure_stage_b_buffers,
    sparse_paged_decode_batched,
)
from .merge import merge_splits_kernel  # noqa: F401

__all__ = [
    "sparse_paged_decode_batched",
    "ResidentVSource",
    "VSource",
    "ensure_stage_b_buffers",
    "merge_splits_kernel",
]
