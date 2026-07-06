"""AmassMetadataBuilder -- FlashAttentionMetadataBuilder + persistent AMASS state.

Owns the r8 Stage-A lifecycle:

  * Allocates ``R8State`` ONCE (so every buffer address is stable across
    CUDA-graph replays), LAZILY at the first build() where the KV cache is
    allocated -- ``R8State`` is sized from ``cache_config.num_gpu_blocks``, which
    is unknown at builder __init__ (same reason the DRAM tier defers to
    ``_ensure_dram`` at first build). At that first build it also keys a
    ``kv_cache.data_ptr() -> physical layer index`` map from
    ``compilation_config.static_forward_context`` (the exact tensor object the
    impl.forward receives), so the impl's layer-index-free select call can find
    the right L-major slab.

  * Runs the eigh ``r8_build_refresh`` OUTSIDE the graph, in build(), per layer,
    over that layer's resident K half -- tag-gated so only newly-finalized pages
    rebuild (steady-state cost = one tag compare/layer). Skipped on cudagraph
    capture dummy batches (``build_for_cudagraph_capture`` sets ``_capturing``),
    mirroring the DRAM tier's ``begin_step(capture=...)``.

  * Refreshes the per-step derived selection params (``derive_page_params``) once
    per step (graph-safe, one launch) and attaches the state as ``md.amass`` for
    the impl to consume.

CUDA-graph support: with the real graph-safe Stage A the pure-decode pipeline is
fixed-shape / allocation-free / host-sync-free, so the builder declares
``UNIFORM_SINGLE_TOKEN_DECODE`` -> pure-decode batches replay as FULL graphs
(select + decode INSIDE the graph, zero python per step). The eigh build never
runs inside the graph; only the persistent r8/derived buffers it refreshes
outside the graph are read at replay. The eager validation bridge (torch topk ->
not capturable) forces ``NEVER`` -> attention runs eagerly between graph pieces.
"""
from __future__ import annotations

import os
import traceback

import torch

# Profiling-only NVTX ranges (AMASS_NVTX=1): attribute the per-step host
# prologue (FA build / refresh gate / derive) on an nsys timeline. Zero-cost
# when off (module-level constant, no per-step env read).
_NVTX = os.environ.get("AMASS_NVTX", "0") == "1"
_nvtx_push = torch.cuda.nvtx.range_push
_nvtx_pop = torch.cuda.nvtx.range_pop

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder

from ..attention.decode import _split_kv, ensure_stage_b_buffers
from . import _runtime


def _seq_lens_host(cam):
    """Per-step HOST (CPU) seq lens, sync-free, WITHOUT the deprecated
    ``seq_lens_cpu`` property.

    vLLM 0.24 decorates ``CommonAttentionMetadata.seq_lens_cpu`` with
    ``@typing_extensions.deprecated``; inside the engine that access pays the
    warnings machinery EVERY step -- measured ~6.9 ms/decode-step (nsys +
    perf_counter, scratch_deepopt3), the single largest AMASS host CPU cost.
    The runner populates the same pinned CPU tensor in the plain (non-property)
    field ``seq_lens_cpu_upper_bound`` (exact outside async spec decode, which
    never reaches the pure-decode AMASS path); ``_seq_lens_cpu`` is the
    deprecated-path backing store. Read those directly; fall back to None (gate
    then refreshes every step, correct but slow)."""
    sl = getattr(cam, "seq_lens_cpu_upper_bound", None)
    if sl is None:
        sl = getattr(cam, "_seq_lens_cpu", None)
    return sl

# split=128: measured Stage-B bandwidth knee at bs=1 (2064 vs 1674 GB/s @
# split64) with no bs=4 regression (pps floors at PPS_MIN so the active-split
# count is unchanged there).
_SPLIT = 128


class AmassMetadataBuilder(FlashAttentionMetadataBuilder):
    """FA builder + persistent AMASS decode state + per-step param refresh."""

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        cfg = _runtime.get_config()
        _State, _derive, _select, graph_safe = _runtime.resolve_selection()
        if cfg is not None and cfg.variant == "fast" and graph_safe \
                and not cfg.force_eager:
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
        return AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._cfg = _runtime.get_config()
        self._device = device
        self._vllm_config = vllm_config
        self._layer_names = list(layer_names)
        self._state = None
        self._derive = None
        self._r8 = False          # True => real graph-safe r8 path
        self._capturing = False   # set during cudagraph-capture dummy builds
        self._layer_kv = None     # per-layer resident kv_cache tensor views
        self._layer_k = None      # per-layer resident K-half views (tail build)
        self._prev_seq_lens = None  # host seq_lens snapshot (refresh-skip gate)
        self._built_once = False  # first full build -> bulk; later -> tag delta
        self._tier = None         # amass.tier.Tier (mem variants; lazy)
        self._tier_tried = False
        # mem-v/mem-kv: the DRAM tier is built lazily at first build (needs
        # num_gpu_blocks) and its per-step stage/flush lifecycle runs OUTSIDE the
        # graph in build(), exactly like the r8 refresh. Attention still
        # delegates to stock FA until the V_SRC==1 tiered load lands
        # (_runtime.mem_stageb_wired()); the tier runs in shadow meanwhile.
        if self._cfg is not None and self._cfg.is_mem:
            print("[amass] mem tier WIRED (shadow); Tier + begin_step deferred "
                  "to first build (num_gpu_blocks not set yet)", flush=True)
            return
        # Only the fast variant allocates r8 selection state here.
        if self._cfg is None or self._cfg.variant != "fast":
            return

        State, derive, _select, graph_safe = _runtime.resolve_selection()
        self._r8 = graph_safe
        self._derive = derive
        if self._r8:
            # R8State is sized from num_gpu_blocks (unknown until the cache is
            # allocated) and needs the per-layer kv_cache tensors -> defer the
            # whole construction to the first build() (see _ensure_state).
            print("[amass] r8 selection WIRED (graph-safe); R8State + layer map "
                  "deferred to first build (num_gpu_blocks not set yet)",
                  flush=True)
            return

        # --- bridge fallback: no num_blocks needed, construct now -------------
        try:
            max_reqs = vllm_config.scheduler_config.max_num_seqs
            max_model_len = vllm_config.model_config.max_model_len
            bs = self.block_size
            max_pages = (max_model_len + bs - 1) // bs
            n_kv = self.num_heads_kv
            G = self.num_heads_q // n_kv
            self._state = State(device, self._cfg, max_reqs, n_kv, G,
                                self.headdim, max_pages, _SPLIT)
            ensure_stage_b_buffers(self._state, device, _SPLIT)
            print(f"[amass] BRIDGE buffers: reqs={max_reqs} n_kv={n_kv} G={G} "
                  f"d={self.headdim} max_pages={max_pages} split={_SPLIT} "
                  f"graph_safe={graph_safe}", flush=True)
        except Exception as e:
            print("[amass] bridge buffer alloc FAILED:", flush=True)
            traceback.print_exception(type(e), e, e.__traceback__)
            self._state = None

    # --------------------------------------------------------------------- #
    # Lazy R8State construction (num_gpu_blocks known only at first build).  #
    # --------------------------------------------------------------------- #
    def _ensure_state(self):
        if self._state is not None or not self._r8:
            return self._state
        nb = self._vllm_config.cache_config.num_gpu_blocks
        if not nb:                       # profiling run: cache not allocated yet
            return None
        # Resolve the per-layer resident kv_cache tensors (the SAME tensor object
        # the impl.forward receives: attention.get_attention_context returns
        # attn_layer.kv_cache directly) and key the ptr -> layer-index map.
        sfc = self._vllm_config.compilation_config.static_forward_context
        layer_kv = []
        ptr2layer = {}
        for lidx, name in enumerate(self._layer_names):
            mod = sfc.get(name)
            kvc = getattr(mod, "kv_cache", None) if mod is not None else None
            if not isinstance(kvc, torch.Tensor) or kvc.numel() == 0:
                return None              # not bound yet -> retry next build
            layer_kv.append(kvc)
            ptr2layer[kvc.data_ptr()] = lidx

        try:
            cfg = self._cfg
            # Score mode selects the summary state: r8 (backward-safe fallback)
            # or the 3.4x-smaller QuadState selector (score="quad", default, or
            # "clse", the coords-augmented reasoning signal). Both mirror the
            # same alloc-once / graph-safe layout, so the rest is identical.
            quad = getattr(cfg, "score", "r8") in ("quad", "clse")
            if quad:
                from ..selection import QuadState as _State
                rank = cfg.quad_rank
            else:
                from ..selection import R8State as _State
                rank = cfg.r8_rank
            n_kv = self.num_heads_kv
            G = self.num_heads_q // n_kv
            max_reqs = self._vllm_config.scheduler_config.max_num_seqs
            max_model_len = self._vllm_config.model_config.max_model_len
            page = self.block_size
            max_pages = (max_model_len + page - 1) // page
            # STATIC selection is budget-driven; if only a coverage was given,
            # fall back to a fixed fraction (no adaptive per-head b).
            budget = cfg.budget if cfg.budget is not None else 0.1
            # quad V-summary quant bits (int4 prod default / int8 opt-in) + the
            # GQA combine (nrm default) + the clse coords.  R8State takes none of
            # these, so only thread them through the quad branch.
            extra_state = {}
            if quad:
                extra_state["v_bits"] = cfg.quad_v_bits
                extra_state["combine"] = cfg.quad_combine
                if getattr(cfg, "score", "quad") == "clse":
                    extra_state.update(coords="lse", c_bits=cfg.quad_c_bits,
                                       c_grain=cfg.quad_c_grain)
            st = _State(
                self._device, num_layers=len(self._layer_names),
                num_blocks=int(nb), n_kv=n_kv, G=G, head_dim=self.headdim,
                page=page, max_reqs=max_reqs, max_pages=max_pages,
                rank=rank, budget=budget,
                sink_pages=cfg.sink_pages, window_pages=cfg.window_pages,
                **extra_state)
            # Stage-B ownership lives in attention/; it reads st.D / st.max_reqs /
            # st.n_kv / st.G. R8State exposes .d (lowercase) -> alias it, and give
            # it a .scale slot (unused: the impl passes scale explicitly).
            st.D = st.d
            st.scale = None
            ensure_stage_b_buffers(st, self._device, _SPLIT)
            st._layer_of = ptr2layer
            self._layer_kv = layer_kv
            # Per-layer resident K-half views (stable storage): the steady-state
            # tail rebuild gathers from every layer each finalize step.
            self._layer_k = [_split_kv(kvc)[0] for kvc in layer_kv]
            self._state = st
            print(f"[amass] {'QuadState' if quad else 'R8State'} ALLOCATED "
                  f"(score={getattr(cfg, 'score', 'r8')}): layers="
                  f"{st.L} num_blocks={st.NB} reqs={max_reqs} n_kv={n_kv} G={G} "
                  f"d={self.headdim} page={page} max_pages={max_pages} "
                  f"rank={st.r} v_bits={getattr(st, 'v_bits', '-')} "
                  f"combine={getattr(st, 'combine', '-')} "
                  f"coords={getattr(st, 'coords', '-')} "
                  f"budget={budget} sink={cfg.sink_pages} "
                  f"window={cfg.window_pages} split={_SPLIT} "
                  f"selector_MiB={st.bytes_per_layer() * st.L / 2**20:.1f}",
                  flush=True)
        except Exception as e:
            print("[amass] R8State alloc FAILED (decode will delegate to stock "
                  "FA / FullKV):", flush=True)
            traceback.print_exception(type(e), e, e.__traceback__)
            self._state = None
        return self._state

    # --------------------------------------------------------------------- #
    # Lazy Tier construction (mem variants; num_gpu_blocks known at build).  #
    # --------------------------------------------------------------------- #
    def _ensure_tier(self):
        if self._tier is not None or self._tier_tried:
            return self._tier
        nb = self._vllm_config.cache_config.num_gpu_blocks
        if not nb:                        # profiling run: cache not allocated
            return None
        # Resolve the per-layer resident kv_cache tensors (dtype/device source);
        # bind only once every layer's cache tensor is materialised.
        sfc = self._vllm_config.compilation_config.static_forward_context
        layer_kv = []
        for name in self._layer_names:
            mod = sfc.get(name)
            kvc = getattr(mod, "kv_cache", None) if mod is not None else None
            if not isinstance(kvc, torch.Tensor) or kvc.numel() == 0:
                return None               # not bound yet -> retry next build
            layer_kv.append(kvc)
        self._tier_tried = True
        try:
            from ..tier import Tier
            cfg = self._cfg
            sched = self._vllm_config.scheduler_config
            comp = self._vllm_config.compilation_config
            max_reqs = sched.max_num_seqs
            max_model_len = self._vllm_config.model_config.max_model_len
            page = self.block_size
            max_pages = (max_model_len + page - 1) // page
            max_tokens = max(
                int(getattr(sched, "max_num_batched_tokens", 0) or 0),
                int(getattr(comp, "max_cudagraph_capture_size", 0) or 0),
                max_reqs) + 8
            dtype = layer_kv[0].dtype
            self._tier = Tier.from_config(
                cfg, num_layers=len(self._layer_names), num_blocks=int(nb),
                n_kv=self.num_heads_kv, page=page, d=self.headdim,
                max_reqs=max_reqs, max_pages=max_pages, max_tokens=max_tokens,
                dtype=dtype, device=self._device)
            self._layer_kv = layer_kv
            print(f"[amass] Tier ALLOCATED (mem-v shadow): {self._tier.bytes_report()}",
                  flush=True)
        except Exception as e:
            print("[amass] Tier alloc FAILED (mem-v -> stock FA):", flush=True)
            traceback.print_exception(type(e), e, e.__traceback__)
            self._tier = None
        return self._tier

    # --------------------------------------------------------------------- #
    def build_for_cudagraph_capture(self, common_attn_metadata):
        # Capture/warmup dummy batches carry dummy block tables / seq lens: the
        # eigh r8_build_refresh must NOT run on them (it would write garbage r8
        # codes for dummy blocks). The captured graph reads the persistent r8 /
        # derived buffers the REAL build() refreshes; only their addresses are
        # baked at capture, so skipping the eager build here is correct.
        self._capturing = True
        try:
            return super().build_for_cudagraph_capture(common_attn_metadata)
        finally:
            self._capturing = False

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        if _NVTX:
            _nvtx_push("amass.fa_build")
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if _NVTX:
            _nvtx_pop()
        # mem variants: run the tier stage/flush/alloc lifecycle OUTSIDE the
        # graph (host), every step (staging happens during prefill AND decode).
        # Skipped on cudagraph-capture dummy batches (capture=True). Attach the
        # tier so the impl can build a TierVSource once the tiered load is wired.
        if self._cfg is not None and self._cfg.is_mem:
            tier = self._ensure_tier()
            if tier is not None:
                try:
                    tier.begin_step(
                        md.block_table, md.seq_lens, md.query_start_loc,
                        md.max_query_len, md.slot_mapping.shape[0],
                        capture=self._capturing)
                    md.amass_tier = tier
                except Exception as e:
                    print("[amass] tier.begin_step FAILED (mem-v this step -> "
                          "stock FA):", flush=True)
                    traceback.print_exception(type(e), e, e.__traceback__)
            return md
        st = self._ensure_state()
        if st is None or md.max_query_len != 1:
            return md
        n_req = md.seq_lens.shape[0]
        if self._r8:
            # 1. (Re)build the r8 summary of newly-finalized pages OUTSIDE the
            #    graph, per layer. Tag-gated: steady state rebuilds nothing.
            #    At the first decode build every prompt page's K is committed
            #    (the current decode token is not yet in the cache, but it lives
            #    in the always-attended window, so it is never scored), so the
            #    whole prompt is summarized correctly here in one shot; scored
            #    pages are ALWAYS a subset of finalized pages (window >= 1).
            need_derive = True
            if not self._capturing:
                if _NVTX:
                    _nvtx_push("amass.refresh")
                need_derive = self._maybe_refresh(st, md,
                                                  common_attn_metadata, n_req)
                if _NVTX:
                    _nvtx_pop()
            # 2. Per-step derived selection params (graph-safe, one launch).
            #    All derive outputs are pure functions of ceil(seq/page)
            #    (select.py::_derive_params_kernel), which only changes on
            #    steps where an advanced row hits seq % page == 1 -> skip the
            #    launch otherwise (the persistent buffers already hold the
            #    identical values). Capture steps keep the unconditional
            #    (dummy) derive, matching the pre-gating behaviour.
            if need_derive:
                from ..selection import derive_page_params
                if _NVTX:
                    _nvtx_push("amass.derive")
                derive_page_params(st, md.seq_lens, n_req)
                if _NVTX:
                    _nvtx_pop()
        else:
            self._derive(md.seq_lens, st, self._cfg)   # bridge no-op
        md.amass = st
        return md

    def _maybe_refresh(self, st, md, cam, n_req):
        """Refresh-skip gate. The summary codes only change when a page
        FINALIZES or a request slot is reused; both are detectable on the HOST
        with no device sync from the runner's CPU seq lens:

          * steady step, no boundary crossed  -> NOTHING to do (skip).
          * steady step, some rows crossed a page boundary (cur % page == 0)
            -> ``*_build_tail`` for exactly THOSE rows: batched, sync-free
            rebuild of each one's last finalized page (idempotent; ~55
            launches vs the full refresh's measured ~99 ms sync-storm).
          * anything else (first step, shape change, slot reuse, preemption)
            -> ``*_build_bulk``: batched zero-sync rebuild of ALL finalized
            pages of the batch (idempotent; covers slot/block reuse WITHOUT
            the tag scan; the measured 737 ms first-decode sync-storm becomes
            a few dozen bandwidth-bound launches).
          * no host lens at all -> legacy per-layer tag-scan refresh.

        Padded uniform-decode rows (full-cudagraph batches pad n_req up to the
        capture size) sit at ``cur == prev == 0`` and are treated as steady;
        they never cross a boundary so the tail skips them. (The pre-iter-2
        gate treated padded batches as never steady, silently running the full
        refresh EVERY step whenever ``n_req < capture size``.)

        Returns ``need_derive``: whether the derive params can have changed
        this step (ceil(seq/page) increments on ``cur % page == 1`` rows)."""
        page = self.block_size
        sl_cpu = _seq_lens_host(cam)
        if sl_cpu is None:                     # no host lens -> conservative
            self._prev_seq_lens = None
            self._r8_refresh(st, md, n_req)
            return True
        cur = sl_cpu[:n_req]
        prev = self._prev_seq_lens
        full = True
        crossed = None
        if prev is not None and prev.shape == cur.shape:
            adv = cur == prev + 1              # rows that decoded one token
            pad = (cur == 0) & (prev == 0)     # padded uniform-decode rows
            # frozen rows (cur == prev, e.g. vLLM end-of-generate drain steps
            # replaying padded decodes with discarded tokens) change nothing.
            frozen = cur == prev
            if bool(torch.all(adv | pad | frozen)):   # steady batch
                full = False
                crossed = (cur % page == 0) & adv
        if os.environ.get("AMASS_DEBUG_GATE", "0") == "1":
            print(f"[amass gate] full={full} "
                  f"crossed={crossed.sum().item() if crossed is not None else '-'} "
                  f"n_req={n_req} cur={cur[:4].tolist()} "
                  f"prev={prev[:4].tolist() if prev is not None else None}",
                  flush=True)
        self._prev_seq_lens = cur.clone()
        need_derive = True
        if full:
            max_fin = int(cur.max().item()) // page   # CPU tensor: free
            # First full build: zero-sync bulk (everything is stale anyway).
            # Every later composition change (arrival, drain shuffle, slot
            # reuse, preemption resume): batched TAG-SCAN delta -- rebuild
            # only blocks whose content tag went stale (the measured ~800 ms
            # bulk per transition at bs16/16K becomes ~tag-pass + new work).
            if self._built_once:
                self._delta_refresh(st, md, n_req, max_fin)
            else:
                self._bulk_refresh(st, md, n_req, max_fin)
                self._built_once = True
        else:
            if bool(crossed.any()):
                rows = torch.nonzero(crossed).squeeze(-1).to(self._device)
                self._tail_refresh(st, md, n_req, rows)
            # derive params change only when ceil(sl/page) does (% page == 1).
            need_derive = bool(((cur % page == 1) & adv).any())
        return need_derive

    def _r8_refresh(self, st, md, n_req):
        """Run r8_build_refresh for every layer over its resident K half.

        NOT graph-safe (eigh + boolean gather) -> only ever called here, on the
        host, before graph replay. First decode step / slot changes only; the
        steady-state finalize path is ``_tail_refresh``."""
        if getattr(self._cfg, "score", "r8") in ("quad", "clse"):
            from ..selection import quad_build_refresh as _refresh
        else:
            from ..selection import r8_build_refresh as _refresh
        bt = md.block_table
        sl = md.seq_lens
        for lidx, kvc in enumerate(self._layer_kv):
            K, _ = _split_kv(kvc)          # (NB, page, n_kv, d) resident K half
            _refresh(st, lidx, K, bt, sl, n_req)

    def _tail_refresh(self, st, md, n_req, rows):
        """Steady-state finalize step: batched all-layer sync-free rebuild of
        the crossing rows' last finalized page (selection.build.r8_build_tail)."""
        if getattr(self._cfg, "score", "r8") in ("quad", "clse"):
            from ..selection import quad_build_tail as _tail
        else:
            from ..selection import r8_build_tail as _tail
        _tail(st, self._layer_k, md.block_table, md.seq_lens, n_req, rows)

    def _bulk_refresh(self, st, md, n_req, max_fin):
        """First decode step / slot change: batched zero-sync rebuild of ALL
        finalized pages, all layers (selection.build.r8_build_bulk)."""
        if getattr(self._cfg, "score", "r8") in ("quad", "clse"):
            from ..selection import quad_build_bulk as _bulk
        else:
            from ..selection import r8_build_bulk as _bulk
        _bulk(st, self._layer_k, md.block_table, md.seq_lens, n_req, max_fin)

    def _delta_refresh(self, st, md, n_req, max_fin):
        """Batch-composition change: batched all-layer tag scan, rebuild only
        stale blocks (selection.build.r8_build_delta)."""
        if getattr(self._cfg, "score", "r8") in ("quad", "clse"):
            from ..selection import quad_build_delta as _delta
        else:
            from ..selection import r8_build_delta as _delta
        _delta(st, self._layer_k, md.block_table, md.seq_lens, n_req, max_fin)
