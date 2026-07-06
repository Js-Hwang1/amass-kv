"""Backend runtime singletons: the one AmassConfig and the Stage-A resolver.

register() builds the single :class:`AmassConfig` and stashes it here; the impl
and builder read it (avoids a register<->attn import cycle). ``resolve_selection``
returns the Stage-A entry points, preferring the REAL graph-safe ``amass.selection``
(the STATIC r8-ranked pipeline) and falling back to the eager validation bridge in
``_selection_bridge.py`` when it is not available (or ``AMASS_FORCE_BRIDGE=1``).

The real Stage-A (``amass.selection``, AMASS_R8_STATIC_SPEC.md) is the STATIC
r8-ranked pipeline: ``R8State`` (r8 codes keyed by physical block, L-major),
``derive_page_params(st, seq_lens, n_req)`` (per-step, one launch), and
``select_pages_r8(st, layer, q, block_table, seq_lens, n_req, scale)`` which reads
the per-layer r8 codes -- codes that must first be BUILT from finalized pages via
``r8_build_refresh(st, layer, K, block_table, seq_lens, n_req)`` (eigh-based, NOT
graph-safe, tag-gated, runs on page-finalize off the hot path).

WIRING (this agent). The lifecycle the earlier scaffold flagged is now built in
``builder.py``:

  * R8State needs ``num_gpu_blocks`` (unknown at builder __init__) -> the builder
    constructs it LAZILY at the first build() where the cache is allocated, and
    keys a ``kv_cache.data_ptr() -> layer index`` map from
    ``compilation_config.static_forward_context`` (the same tensor object the
    impl.forward receives -- verified in vLLM 0.24 attention.get_attention_context).
  * the eigh build/refresh runs in build() OUTSIDE the graph (host, before graph
    replay), per layer, over the layer's resident K half, tag-gated so only
    newly-finalized pages rebuild. Skipped on cudagraph-capture dummy batches
    (``build_for_cudagraph_capture`` sets a capture flag), exactly like the DRAM
    tier's ``begin_step(capture=...)``.
  * ``derive_page_params`` runs once per step in build() (graph-safe, one launch),
    so the per-layer select kernels inside the graph just read the refreshed
    persistent buffers.

The impl (``attn.py``) calls the select fn returned here with the BRIDGE-compatible
signature ``(q, kv_cache, block_table, seq_lens, st, n_req, scale, cfg)`` and never
sees a layer index, so :func:`_r8_select_adapter` recovers the layer from the
``st._layer_of`` map (populated by the builder) and calls ``select_pages_r8`` with
``derive=False`` (the builder already derived this step). Stage-B decode is
selector-agnostic: it consumes ``st.page_table`` / ``st.page_cnt`` plus the Stage-B
partials, so swapping the bridge for the r8 path does not touch attention/.
"""
from __future__ import annotations

import os
from typing import Optional

from ..config import AmassConfig

_config: Optional[AmassConfig] = None

# The real graph-safe r8 Stage A is now wired (builder constructs R8State at first
# build, runs the r8_build_refresh page-finalize hook outside the graph, and
# derives per-step params). Set AMASS_FORCE_BRIDGE=1 to force the eager validation
# bridge instead (kept as a fallback / A-B reference; not CUDA-graph capturable).
_R8_WIRED = True

# mem-v Stage-B readiness. The tier (tier/) ALLOCATES and its per-step
# stage/flush/residency lifecycle RUNS (builder drives begin_step, invariants
# exercised) as soon as the mem-v tier is wired. But the actual tiered V DELIVERY
# to attention -- the decode kernel's ``V_SRC==1`` 3-way load (hot | staged |
# pinned) AND the K-only KV-cache-spec patch that realizes the VRAM saving -- is
# the documented FOLLOW-UP (see tier/README.md). While False, mem-v decode
# delegates to stock FA over the still-resident V (correct-by-fallback; the tier
# runs in shadow so its state machine is validated against the resident path).
# The follow-up flips this True together with (a) the V_SRC==1 kernel arm,
# (b) tier.write_kv, (c) tier.step gather, (d) the K-only spec patch.
_MEM_STAGEB_WIRED = False


def mem_stageb_wired() -> bool:
    return _MEM_STAGEB_WIRED


def set_config(cfg: AmassConfig) -> None:
    global _config
    _config = cfg


def get_config() -> Optional[AmassConfig]:
    return _config


def r8_selection_available() -> bool:
    """True when amass.selection exposes the full r8 Stage-A surface."""
    try:
        from .. import selection as sel
    except Exception:
        return False
    return all(getattr(sel, n, None) is not None for n in
               ("R8State", "derive_page_params", "select_pages_r8",
                "r8_build_refresh"))


def quad_selection_available() -> bool:
    """True when amass.selection exposes the full quad + clse Stage-A surface."""
    try:
        from .. import selection as sel
    except Exception:
        return False
    return all(getattr(sel, n, None) is not None for n in
               ("QuadState", "derive_page_params", "select_pages_quad",
                "quad_build_refresh", "clse_score", "select_pages_clse"))


def use_quad() -> bool:
    """The active config selects a QuadState-backed score mode (``quad``,
    default, or ``clse``); r8 stays the backward-safe fallback."""
    cfg = _config
    return cfg is not None and getattr(cfg, "score", "r8") in ("quad", "clse")


def _force_bridge() -> bool:
    return os.environ.get("AMASS_FORCE_BRIDGE") not in (None, "", "0")


# Kernel-backend latches: None = untried, True = CUDA in use, False = Triton
# fallback (a build failed on the first eager call; never re-tried).
_SEL_CUDA = None
_DEC_CUDA = None
_QSEL_CUDA = None


def _layer_index(st, kv_cache) -> int:
    layer_of = getattr(st, "_layer_of", None)
    if layer_of is None:
        raise RuntimeError(
            "amass r8: R8State has no _layer_of map (builder did not wire it)")
    lidx = layer_of.get(kv_cache.data_ptr())
    if lidx is None:
        raise RuntimeError(
            "amass r8: kv_cache pointer not in the builder's layer map "
            f"(ptr={kv_cache.data_ptr():#x}); the r8 state is L-major and needs "
            "the physical layer index. Check static_forward_context wiring.")
    return lidx


def _r8_select_adapter(q, kv_cache, block_table, seq_lens, st, n_req, scale,
                       cfg) -> None:
    """Bridge-signature Stage-A entry for the REAL r8 path.

    Dispatches score+topb to the hand-CUDA Hopper kernels when ``cfg.use_cuda``
    (the CUDA r8_score is ~7x faster), else the Triton reference. The CUDA
    extensions build once at the first eager call (warmup, before graph
    capture); a build failure latches to Triton. ``derive=False``: the builder
    derived the per-step page params already. Writes st.page_table/page_cnt."""
    from .. import selection as sel

    lidx = _layer_index(st, kv_cache)
    global _SEL_CUDA
    if cfg.use_cuda and _SEL_CUDA is not False:
        try:
            from ..selection import score_cuda
            # MEASURED-BEST routing (H200; scratch_fastopt + scratch_topb grids):
            #   * r8_score -> hand-CUDA (2-pass Hopper); the fused kernel does not
            #     beat the tuned 2-pass in int8 (mma.sync is synchronous, can't
            #     overlap the logsumexp tail); score_cuda picks 2-pass for int8.
            #   * topb -> hand-CUDA RADIX tau-select: now beats Triton at every
            #     size (9.2 vs 13.2us @16K, 14.3 vs 95.9us @64K = 6.7x) after the
            #     O(n) radix rewrite replaced the O(n log^2 n) bitonic sort.
            #     Env AMASS_TOPB_TRITON=1 forces the Triton top-b (A-B checks).
            score_cuda.r8_score_cuda(st, lidx, q, block_table, seq_lens, n_req,
                                     scale)
            if os.environ.get("AMASS_TOPB_TRITON", "0") == "1":
                from ..selection.select import topb_select
                topb_select(st, n_req)
            else:
                from ..selection import select_cuda
                select_cuda.topb_select_cuda(st, n_req)
            if _SEL_CUDA is None:
                _SEL_CUDA = True
                print("[amass] selection kernels: CUDA r8_score + CUDA radix "
                      "top-b", flush=True)
            return
        except Exception as e:  # noqa: BLE001
            _SEL_CUDA = False
            print(f"[amass] CUDA selection unavailable "
                  f"({type(e).__name__}: {e}) -> Triton", flush=True)
    sel.select_pages_r8(st, lidx, q, block_table, seq_lens, n_req, scale,
                        derive=False)


def _quad_select_adapter(q, kv_cache, block_table, seq_lens, st, n_req, scale,
                         cfg) -> None:
    """Bridge-signature Stage-A entry for the quad / clse score modes.

    Dispatches the quadratic score to the hand-CUDA Hopper kernel (``quad_score_
    cuda``, bitwise target = the Triton reference, nrm combine folded in) when
    ``cfg.use_cuda`` and it builds; else the Triton ``quad_score``.  The top-b
    radix tau-select is score-agnostic, so it reuses the SAME CUDA/Triton
    selector as the r8 path.  ``derive=False``: the builder derived the per-step
    page params already.

    score="clse" (``st.coords == "lse"``) has no hand-CUDA kernel yet, so it
    always runs the graph-safe Triton ``clse_score`` reference + the stock
    (CUDA or Triton) top-b selector."""
    from .. import selection as sel

    lidx = _layer_index(st, kv_cache)

    def _topb():
        if os.environ.get("AMASS_TOPB_TRITON", "0") == "1":
            from ..selection.select import topb_select
            topb_select(st, n_req)
        else:
            from ..selection import select_cuda
            select_cuda.topb_select_cuda(st, n_req)

    # CLSE: Triton reference score (graph-safe) + top-b.  The nrm combine is
    # applied inside ``clse_score`` when ``st.combine == "nrm"``.
    if getattr(st, "coords", "none") == "lse":
        sel.clse_score(st, lidx, q, block_table, seq_lens, n_req, scale)
        if cfg.use_cuda:
            try:
                _topb()
                return
            except Exception:  # noqa: BLE001
                from ..selection.select import topb_select
                topb_select(st, n_req)
                return
        from ..selection.select import topb_select
        topb_select(st, n_req)
        return

    global _QSEL_CUDA
    if cfg.use_cuda and _QSEL_CUDA is not False:
        try:
            from ..selection import quad_score_cuda
            quad_score_cuda.quad_score_cuda(st, lidx, q, block_table, seq_lens,
                                            n_req, scale)
            _topb()
            if _QSEL_CUDA is None:
                _QSEL_CUDA = True
                print("[amass] selection kernels: CUDA quad_score + CUDA radix "
                      "top-b", flush=True)
            return
        except Exception as e:  # noqa: BLE001
            _QSEL_CUDA = False
            print(f"[amass] CUDA quad selection unavailable "
                  f"({type(e).__name__}: {e}) -> Triton quad_score", flush=True)
    sel.select_pages_quad(st, lidx, q, block_table, seq_lens, n_req, scale,
                          derive=False)


def decode_dispatch(q, kv_cache, block_table, seq_lens, st, out, vsource,
                    scale, cfg) -> None:
    """Stage-B decode: hand-CUDA Hopper kernel when ``cfg.use_cuda`` (latched
    fallback to the Triton reference on build failure)."""
    from ..attention.decode import sparse_paged_decode_batched
    global _DEC_CUDA
    if cfg.use_cuda and _DEC_CUDA is not False:
        try:
            from ..attention import decode_cuda
            decode_cuda.sparse_paged_decode_batched_cuda(
                q, kv_cache, block_table, seq_lens, st, out, vsource,
                scale=scale)
            if _DEC_CUDA is None:
                _DEC_CUDA = True
                print("[amass] decode kernel: CUDA (Hopper)", flush=True)
            return
        except Exception as e:  # noqa: BLE001
            _DEC_CUDA = False
            print(f"[amass] CUDA decode unavailable "
                  f"({type(e).__name__}: {e}) -> Triton", flush=True)
    sparse_paged_decode_batched(q, kv_cache, block_table, seq_lens, st, out,
                                vsource, scale=scale)


def resolve_selection():
    """(StateClass, derive_fn, select_fn, graph_safe).

    Returns the real graph-safe r8 path when it is wired and available;
    otherwise the eager validation bridge. ``select_fn`` always has the
    bridge-compatible call signature the impl uses."""
    if _R8_WIRED and not _force_bridge() and r8_selection_available():
        from .. import selection as sel
        if use_quad() and quad_selection_available():
            return (sel.QuadState, sel.derive_page_params,
                    _quad_select_adapter, True)
        return (sel.R8State, sel.derive_page_params, _r8_select_adapter, True)
    from ._selection_bridge import BridgeState, bridge_derive, bridge_select
    return BridgeState, bridge_derive, bridge_select, False
