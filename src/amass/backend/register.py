"""register() -- the vLLM general-plugin entry point.

Builds the single :class:`AmassConfig` (from an optional ``AMASS_CONFIG`` json/
yaml plus any explicit overrides) and patches ``CudaPlatform.get_attn_backend_cls``
so that wherever vLLM would pick the FlashAttention backend it gets
:class:`AmassAttentionBackend` instead -- a FlashAttentionBackend subclass, so
every non-decode path inherits stock FA.

``AMASS_DISABLE=1`` makes register() inert (the FullKV / stock-FA reference line).
This is the ONE place besides ``AmassConfig.load`` that reads the environment.
"""
from __future__ import annotations

import os

from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

from ..config import AmassConfig
from . import _runtime
from .attn import AmassAttentionImpl
from .builder import AmassMetadataBuilder

_APPLIED = False


class AmassAttentionBackend(FlashAttentionBackend):
    """FlashAttentionBackend with the AMASS impl + builder swapped in.

    FAST variant keeps the stock FA KV-cache shape (K+V resident). The mem
    variants' K-only spec patch will be added here alongside the tier.

    NOTE: get_name() is deliberately inherited (-> "FLASH_ATTN"): vLLM 0.24
    looks the name up in AttentionBackendEnum, and only the registered FA name
    resolves. The impl/builder still come from THIS subclass."""

    @staticmethod
    def get_impl_cls() -> type[AmassAttentionImpl]:
        return AmassAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[AmassMetadataBuilder]:
        return AmassMetadataBuilder


_FA_PATH = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
_OUR_PATH = f"{__name__}.AmassAttentionBackend"
_ANNOUNCED = False


def register(overrides: dict | None = None) -> None:
    """vLLM general-plugin entry point (runs in every engine process)."""
    global _APPLIED
    if _APPLIED or os.environ.get("AMASS_DISABLE"):
        return

    cfg = AmassConfig.load(overrides)
    _runtime.set_config(cfg)

    from vllm.platforms import cuda
    # 0.24: get_attn_backend_cls is defined on CudaPlatformBase (CudaPlatform is
    # the Nvml/NonNvml alias). Patch the class in the MRO that defines it.
    target = next(k for k in cuda.CudaPlatform.__mro__
                  if "get_attn_backend_cls" in k.__dict__)
    orig = target.__dict__["get_attn_backend_cls"].__func__

    def get_attn_backend_cls(cls, *args, **kwargs):
        path = orig(cls, *args, **kwargs)
        if path == _FA_PATH:
            global _ANNOUNCED
            if not _ANNOUNCED:
                _ANNOUNCED = True
                print(f"[amass] ACTIVE variant={cfg.variant} "
                      f"budget={cfg.budget} coverage={cfg.coverage} "
                      f"sink={cfg.sink_pages} window={cfg.window_pages} "
                      f"-> {_OUR_PATH}", flush=True)
            return _OUR_PATH
        return path

    target.get_attn_backend_cls = classmethod(get_attn_backend_cls)
    _APPLIED = True
    print("[amass] registered (patched CudaPlatform.get_attn_backend_cls)",
          flush=True)
    # mem variants: the K-only KV-cache-spec patch (head_size_v=0) that realizes
    # the VRAM saving is the documented FOLLOW-UP (see _patch_kv_cache_spec_k_only
    # and tier/README.md). NOT applied in the scaffold: with K+V still resident,
    # mem-v runs correct-by-fallback (stock FA) while the tier is validated in
    # shadow. The follow-up applies it together with the V_SRC==1 tiered load.


def _patch_kv_cache_spec_k_only() -> None:
    """FOLLOW-UP (mem variants): patch ``Attention.get_kv_cache_spec`` to
    ``head_size_v=0`` so the engine block pool holds K only and num_gpu_blocks
    DOUBLES at the same gpu_memory_utilization. Port from the prototype
    (dynkv_plugin ``_patch_kv_cache_spec``): replace the FullAttentionSpec with
    ``dataclasses.replace(spec, head_size_v=0)`` for plain symmetric-head layers,
    rejecting asymmetric / non-FullAttentionSpec layers loudly. Must be applied
    ONLY once the tier serves V (V_SRC==1) -- otherwise attention loses its V
    half with no replacement. Intentionally NOT called by register() yet."""
    raise NotImplementedError(
        "K-only spec patch is the mem-v follow-up; apply alongside the V_SRC==1 "
        "tiered decode load. See tier/README.md 'FOLLOW-UP'.")
