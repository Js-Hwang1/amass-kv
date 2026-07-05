"""DRAM offload tier (AMASS-mem only) — the memory play.

V (mem-v) or K+V (mem-kv) live in a pinned, device-mapped host pool; only a
small hot buffer + a bounded staging pool stay resident. Per decode step, the
selected pages' V is gathered from the tier into the hot buffer and served to
Stage B.

Structured as a clean state machine with WRITTEN INVARIANTS (AMASS_DESIGN.md
§8.3) — the prototype's tangled ``dram_tier.py`` lifecycle is being rewritten
here to satisfy them (and fix the two known mem bugs):

  I1  no lost V     — every complete written page is staged OR flushed
  I2  hot==residency— hot[slot] holds slot2page[slot]'s V
  I3  no aliasing   — free-stack is a true set; one live block per staging slot
  I4  ordering      — flush_copy reads v_pool[vb] before valloc/scatter reuse it

Modules:
  pool.py        MappedHostVPool — pinned device-mapped host V pool (UVA)
  residency.py   Residency — hot-buffer residency + exact-LRU + gather (I2/I3)
  staging.py     Staging — V staging pool + begin_step lifecycle (I1/I4)
  tier.py        Tier facade + TierVSource (the V_SRC==1 Stage-B seam)

Public surface (imported lazily; torch/CUDA only touched at construction):
  Tier, TierVSource, MappedHostVPool, Residency, Staging
"""
from __future__ import annotations

from .pool import MappedHostVPool  # noqa: F401
from .residency import Residency  # noqa: F401
from .staging import Staging  # noqa: F401
from .tier import Tier, TierVSource  # noqa: F401

__all__ = [
    "Tier",
    "TierVSource",
    "MappedHostVPool",
    "Residency",
    "Staging",
]
