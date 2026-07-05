"""vLLM integration — the plugin boundary.

Registers AMASS as a vLLM general-plugin (entry point
``vllm.general_plugins`` -> ``amass.backend.register:register``): patches
``CudaPlatform.get_attn_backend_cls`` so the FlashAttention backend is replaced
by :class:`AmassAttentionBackend`, and (mem variants) patches the KV-cache spec
to K-only. All behaviour is driven by a single :class:`amass.AmassConfig`.

Modules:
  attn.py       AmassAttentionImpl — dispatch fast|mem, decode/prefill forwards
  builder.py    AmassMetadataBuilder — persistent buffers + per-step refresh
  register.py   register() + get_kv_cache_spec patch

Decode is a subclass of FlashAttentionImpl: every non-pure-decode path (prefill,
mixed, cascade, profiling) inherits stock FA; only pure single-token decode runs
the AMASS Stage-A + Stage-B pipeline (+ tier for mem).
"""
from __future__ import annotations

from .attn import AmassAttentionImpl  # noqa: F401
from .builder import AmassMetadataBuilder  # noqa: F401
from .register import AmassAttentionBackend, register  # noqa: F401

__all__ = [
    "register",
    "AmassAttentionBackend",
    "AmassAttentionImpl",
    "AmassMetadataBuilder",
]
