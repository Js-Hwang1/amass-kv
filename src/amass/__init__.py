"""AMASS — decode-time certified page selection for KV-cache attention, with an
optional DRAM offload tier. Two build targets share one algorithm core:

  * AMASS-fast  — K+V resident (the speed play).
  * AMASS-mem   — V (or K+V) offloaded to DRAM (default; the memory play).

Public API:
  from amass import AmassConfig, register        # register() = vLLM plugin hook

Architecture (ours_doc/AMASS_DESIGN.md):
  config.py     AmassConfig — single source of truth (no scattered env vars)
  selection/    Stage A: decode-time page selection (SHARED by both variants)
  attention/    Stage B: sparse paged attention (V source injected as a seam)
  tier/         DRAM offload state machine (mem variants only)
  backend/      vLLM integration: attention impl, metadata builder, register()
"""
from __future__ import annotations

from .config import AmassConfig  # noqa: F401

__all__ = ["AmassConfig", "register"]

__version__ = "0.1.0"


def register() -> None:
    """vLLM general-plugin entry point. Thin re-export of
    :func:`amass.backend.register.register` so the entry point is stable while
    the backend is ported."""
    from .backend.register import register as _register
    _register()
