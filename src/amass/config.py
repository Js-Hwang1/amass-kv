"""AmassConfig — the single source of truth for all AMASS behaviour.

Replaces the ~50 scattered ``KVCOMP_*`` environment variables of the research
prototype. Built ONCE at plugin registration from the vLLM config plus an
optional ``AMASS_CONFIG`` (a JSON/YAML file path or an inline JSON string), and
threaded explicitly to every component. Kernels receive plain scalars/constexprs
derived from this object and NEVER read the environment themselves.

Two build targets, one algorithm core (see ours_doc/AMASS_DESIGN.md):

  * ``fast``   — K+V resident; decode-time page selection + sparse attention.
  * ``mem-v``  — V offloaded to a DRAM tier (default; the memory play).
  * ``mem-kv`` — K+V offloaded (adds the low-rank r8 selector). [P3]
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Literal, Optional

Variant = Literal["fast", "mem-v", "mem-kv"]
Table = Literal["kv-union", "q-head", "kv-shared"]


@dataclasses.dataclass(frozen=True)
class AmassConfig:
    # ---- variant --------------------------------------------------------- #
    variant: Variant = "mem-v"

    # ---- Stage A: decode-time page selection ----------------------------- #
    # Adaptive per-head nucleus coverage over EXACT page masses (flagship).
    coverage: float = 0.95
    # Alternative to coverage: a fixed per-step selected-page FRACTION (1-cr).
    # When set, overrides `coverage`.
    budget: Optional[float] = None
    table: Table = "kv-union"
    sink_pages: int = 1
    window_pages: int = 1
    # Use the r8 low-rank screen for selection instead of the full bf16 K scan.
    # Required for mem-kv (K is not resident); optional for the others.
    r8_screen: bool = False
    r8_rank: int = 8
    # Selection SUMMARY / score mode.  "quad" (default, LongBench-CERTIFIED) =
    # the Gaussian-MGF quadratic page score (ours_doc/QUAD_SUMMARY_METHOD.md):
    # drops the per-key coords c, stores r' eigenvalues instead, and shrinks the
    # resident selector 3.28x (15.6% -> 4.7% of the KV; measured 1201 vs 3943 MiB)
    # while beating r8 on LongBench-v1 (quad@5% -0.22 vs r8@5% -0.38 vs FullKV),
    # with a cheaper tail-free hot-path kernel.  "r8" = the backward-safe fallback
    # (per-key low-rank logsumexp page-mass estimate mu/Vk/c, rank r8_rank).
    score: Literal["r8", "quad"] = "quad"
    quad_rank: int = 2             # quad rank r' (default 2; rank-insensitive)

    # ---- Stage B / tier (mem variants only) ------------------------------ #
    hot_slots: int = 2048          # hot-buffer slots per (layer, kv-head)
    v_blocks: Optional[int] = None  # V-staging pool blocks (None = auto-size)
    max_pool_gb: float = 256.0     # pinned host-pool cap

    # ---- engine / kernels ------------------------------------------------ #
    # Use the hand-CUDA Hopper (sm_90a) kernels for the hot path (r8_score,
    # topb_select, sparse decode) instead of the Triton reference. The CUDA
    # r8_score is ~7x faster; the kernels build once via cpp_extension at first
    # eager warmup (before graph capture) and are bitwise-matched to the Triton
    # reference. Falls back to Triton per-kernel if a build fails.
    use_cuda: bool = True
    force_eager: bool = False      # documented escape hatch (never CUDA graphs)

    # --------------------------------------------------------------------- #
    @property
    def is_mem(self) -> bool:
        return self.variant in ("mem-v", "mem-kv")

    @property
    def offload_k(self) -> bool:
        return self.variant == "mem-kv"

    def __post_init__(self) -> None:
        if self.variant not in ("fast", "mem-v", "mem-kv"):
            raise ValueError(f"unknown variant {self.variant!r}")
        if self.offload_k and not self.r8_screen:
            # mem-kv cannot scan resident K -> must use the r8 screen
            object.__setattr__(self, "r8_screen", True)
        if not (0.0 < self.coverage <= 1.0):
            raise ValueError(f"coverage must be in (0,1], got {self.coverage}")
        if self.budget is not None and not (0.0 < self.budget <= 1.0):
            raise ValueError(f"budget must be in (0,1], got {self.budget}")
        if self.score not in ("r8", "quad"):
            raise ValueError(f"score must be 'r8' or 'quad', got {self.score!r}")
        if self.quad_rank < 1:
            raise ValueError(f"quad_rank must be >= 1, got {self.quad_rank}")

    # ---- construction ---------------------------------------------------- #
    @classmethod
    def load(cls, overrides: Optional[dict] = None) -> "AmassConfig":
        """Build from the optional ``AMASS_CONFIG`` (file path or inline JSON)
        plus explicit ``overrides``. This is the ONLY place AMASS reads the
        environment for configuration."""
        data: dict = {}
        spec = os.environ.get("AMASS_CONFIG")
        if spec:
            if os.path.isfile(spec):
                text = open(spec).read()
                if spec.endswith((".yaml", ".yml")):
                    import yaml
                    data = yaml.safe_load(text) or {}
                else:
                    data = json.loads(text)
            else:
                data = json.loads(spec)
        if overrides:
            data.update({k: v for k, v in overrides.items() if v is not None})
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - fields
        if unknown:
            raise ValueError(f"unknown AmassConfig keys: {sorted(unknown)}")
        return cls(**data)
