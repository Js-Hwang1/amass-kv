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
    # Budget ALLOCATION across the (layer, kv-head) units: every unit gets
    # exactly b = ceil(budget * n_selectable) pages (the per-unit top-b radix of
    # select.py). This STATIC allocation is the only budget path. (A conserved-
    # total "dynamic" per-unit budget was measured to give ~0 task headroom over
    # static -- kept as a paper negative, code removed; see task #44.)
    table: Table = "kv-union"
    sink_pages: int = 1
    window_pages: int = 1
    # Use the r8 low-rank screen for selection instead of the full bf16 K scan.
    # Required for mem-kv (K is not resident); optional for the others.
    r8_screen: bool = False
    r8_rank: int = 8
    # Selection SUMMARY / score mode.
    #   "quad" (DEFAULT, fast CUDA, LongBench-CERTIFIED) = the Gaussian-MGF
    #     quadratic page score (ours_doc/QUAD_SUMMARY_METHOD.md): drops the
    #     per-key coords c, stores r' eigenvalues instead, and shrinks the
    #     resident selector 3.28x (15.6% -> ~3.4% int4-V of the KV) while beating
    #     r8 on LongBench-v1, with a cheaper tail-free hot-path kernel that runs
    #     on the hand-CUDA Hopper tensor-core kernel.
    #   "clse" (DEFAULT, reasoning-optimized) = quad's storage geometry PLUS
    #     rank-r' per-key coords (int4, ~18 B/page) + a residual scalar, scored
    #     with the r'-projected logsumexp + isotropic residual.  Recovers the
    #     peaky single-key mass the Gaussian drop-c "quad" form misses (reasoning
    #     recall .68 -> .80; matches the EXACT-LSE selector on AIME within noise).
    #     ~3.44% resident.  Ships a hand-CUDA Hopper kernel (v0.1.1, ~1.1-1.5x the
    #     quad score time; 3.2-8.9x over the Triton reference) with a Triton
    #     fallback.  See ours_doc CLSE signal note.
    #   "quad" = speed-first (drops the per-key coords c; slightly faster hot
    #     path, but loses peaky reasoning mass).  Set score="quad" for max
    #     throughput when reasoning-losslessness is not required.
    #   "r8" = the backward-safe fallback (per-key low-rank logsumexp page-mass
    #     estimate mu/Vk/c, rank r8_rank).
    score: Literal["r8", "quad", "clse"] = "clse"
    quad_rank: int = 2             # quad rank r' (default 2; rank-insensitive)
    # quad V-summary quant bits (production default = int4).  int4-V = ~3.4%
    # resident selector (vs int8's 4.7%): a free memory reduction, RULER
    # task-equivalent + proxy-lossless per the quant study, neutral speed on the
    # i8w kernel per DEEP-OPT.  int8 stays available via AMASS_CONFIG
    # {"quad_v_bits": 8}.
    quad_v_bits: int = 4
    # GQA combine over the group's per-head page scores S[g] (quad AND clse):
    #   "nrm" (DEFAULT) = per-head-normalized mass sum, sum_g exp(S[g] - max_page
    #     S[g]).  The confirmed win (+5.8pt AIME, LongBench-safe; tasks #42/#47/
    #     #48): group-max was the WORST combine.  Graph-safe (fixed extra passes).
    #   "max" = the plain GQA group max (kept reachable for ablation).
    quad_combine: Literal["max", "nrm"] = "nrm"
    # CLSE (score="clse") coord quantization.  quad_c_bits: 4 (default) | 8;
    # quad_c_grain: "token" (per-key scale, default) | "page" (one scale/page).
    # Ignored unless score == "clse".
    quad_c_bits: int = 4
    quad_c_grain: Literal["token", "page"] = "token"

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
        if self.score not in ("r8", "quad", "clse"):
            raise ValueError(
                f"score must be 'r8', 'quad' or 'clse', got {self.score!r}")
        if self.quad_rank < 1:
            raise ValueError(f"quad_rank must be >= 1, got {self.quad_rank}")
        if self.quad_v_bits not in (4, 8):
            raise ValueError(
                f"quad_v_bits must be 4 or 8, got {self.quad_v_bits}")
        if self.quad_combine not in ("max", "nrm"):
            raise ValueError(
                f"quad_combine must be 'max' or 'nrm', got {self.quad_combine!r}")
        if self.quad_c_bits not in (4, 8):
            raise ValueError(
                f"quad_c_bits must be 4 or 8, got {self.quad_c_bits}")
        if self.quad_c_grain not in ("token", "page"):
            raise ValueError(
                f"quad_c_grain must be 'token' or 'page', got "
                f"{self.quad_c_grain!r}")

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
