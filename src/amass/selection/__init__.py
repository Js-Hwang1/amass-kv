"""Stage A — decode-time page selection (SHARED by AMASS-fast and AMASS-mem).

STATIC r8-ranked pipeline (ours_doc/AMASS_R8_STATIC_SPEC.md): a query-INDEPENDENT
low-rank int8 page summary (mu / Vk / c from the page-gram eigh, built on
page-finalize with content-tag invalidation) scores every page DIRECTLY, and a
static top-b keeps the highest-scoring pages per kv-head (kv-union = group max),
always attending the sink pages + recent window + partial tail.  No second exact
LSE pass, no adaptive per-head coverage.

Pipeline (per decode step, per layer):

  r8_build_refresh(st, layer, K, block_table, seq_lens, n_req)   # page-finalize
  derive_page_params(st, seq_lens, n_req)                        # once/step
  select_pages_r8(st, layer, q, block_table, seq_lens, n_req, scale)
      = r8_score(...)   -> st.score      (R, n_kv, MP) fp32
      + topb_select(...) -> st.page_table (R, n_kv, MP) int32 / st.page_cnt

All persistent state lives in :class:`R8State` (allocated once from engine
maxima).  ``r8_score`` and ``topb_select`` are fixed-shape, host-sync-free,
allocation-free (full-CUDA-graph safe); ``r8_build_refresh`` runs off the hot
path (torch + eigh).  These are the golden Triton reference for the later
hand-CUDA (Hopper) kernels.
"""
from __future__ import annotations

from .build import (r8_build_bulk, r8_build_delta, r8_build_refresh,
                    r8_build_tail)
from .quad_build import (quad_build_bulk, quad_build_delta,
                         quad_build_refresh, quad_build_tail)
from .quad_score import (clse_score, quad_score, select_pages_clse,
                         select_pages_quad)
from .quad_state import QuadState
from .score import r8_score
from .select import derive_page_params, select_pages_r8, topb_select
from .state import R8State, TAGW

__all__ = [
    "R8State",
    "TAGW",
    "r8_build_refresh",
    "r8_build_tail",
    "r8_build_bulk",
    "r8_build_delta",
    "derive_page_params",
    "r8_score",
    "topb_select",
    "select_pages_r8",
    # ---- quad score mode (Gaussian-MGF quadratic; 3.4x smaller selector) ---- #
    "QuadState",
    "quad_build_refresh",
    "quad_build_tail",
    "quad_build_bulk",
    "quad_build_delta",
    "quad_score",
    "select_pages_quad",
    # ---- clse score mode (quad geometry + rank-r' per-key coords + resid) --- #
    "clse_score",
    "select_pages_clse",
]
