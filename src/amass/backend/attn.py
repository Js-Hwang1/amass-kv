"""AmassAttentionImpl -- FlashAttentionImpl + AMASS decode-time page selection.

Only a plain pure single-token decode step on a standard causal decoder layer
runs the AMASS Stage-A + Stage-B pipeline; every other path (prefill, mixed
batch, cascade, encoder, profiling, quantized cache, sliding window, ...)
inherits stock FlashAttention unchanged (this is a FlashAttentionImpl subclass).

FAST variant (this file): pure decode does
    select_pages_r8(...)                 -> st.page_table / st.page_cnt  (Stage A)
    sparse_paged_decode_batched(...,     -> out                          (Stage B)
        vsource=ResidentVSource(kv_cache))
K is read from the resident cache; V through the ResidentVSource seam (the engine
V half). The mem variants swap ONLY the VSource (a later agent) -- this call site
does not change. Non-fast variants delegate to stock FA here until the tier lands.
"""
from __future__ import annotations

from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

from ..attention.decode import ResidentVSource
from . import _runtime

_ANNOUNCED = False


class AmassAttentionImpl(FlashAttentionImpl):
    """FlashAttentionImpl + AMASS-fast pure-decode page selection."""

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        m = attn_metadata
        cfg = _runtime.get_config()
        # mem variants: the tier ran begin_step in the builder (shadow). Until
        # the V_SRC==1 tiered decode load lands, V is still resident -> delegate
        # to stock FA for CORRECT output. This is the seam where the follow-up
        # swaps ResidentVSource for TierVSource without touching decode.py.
        if cfg is not None and cfg.is_mem:
            return self._forward_mem(layer, query, key, value, kv_cache, m,
                                     output, output_scale, output_block_scale)
        st = getattr(m, "amass", None) if m is not None else None
        # Delegate everything that is not a plain pure-decode step on a standard
        # causal decoder layer with an unquantized resident cache to stock FA.
        if (cfg is None or cfg.variant != "fast" or st is None
                or m is None or output is None
                or m.max_query_len != 1
                or kv_cache.numel() == 0
                or getattr(m, "use_cascade", False)
                or m.causal is not True
                or self.attn_type != AttentionType.DECODER
                or self.sliding_window != (-1, -1)
                or self.alibi_slopes is not None
                or self.sinks is not None
                or getattr(self, "dcp_world_size", 1) > 1
                or is_quantized_kv_cache(self.kv_cache_dtype)):
            return super().forward(layer, query, key, value, kv_cache, m,
                                   output, output_scale=output_scale,
                                   output_block_scale=output_block_scale)

        # --- pure single-token decode: Stage A + Stage B (in-graph when the
        # real graph-safe selection is installed; eager/piecewise on the bridge).
        # KV of the new token is already in the paged cache (0.24 writes it via a
        # separate op before attention). ---------------------------------------
        _State, _derive, select_pages, _graph_safe = _runtime.resolve_selection()
        n_req = m.seq_lens.shape[0]
        out3 = (output if output.dim() == 3
                else output.view(output.shape[0], self.num_heads, -1))

        select_pages(query, kv_cache, m.block_table, m.seq_lens, st, n_req,
                     self.scale, cfg)
        _runtime.decode_dispatch(
            query, kv_cache, m.block_table, m.seq_lens, st, out3,
            ResidentVSource(kv_cache), self.scale, cfg)

        global _ANNOUNCED
        if not _ANNOUNCED:
            _ANNOUNCED = True
            print(f"[amass] fast decode path ACTIVE (n_req={n_req} "
                  f"graph_safe={_graph_safe} "
                  f"budget={cfg.budget} coverage={cfg.coverage})", flush=True)
        return output

    # ---- mem-v (the memory play) ----------------------------------------- #
    def _forward_mem(self, layer, query, key, value, kv_cache, m, output,
                     output_scale, output_block_scale):
        """mem-v attention seam.

        SCAFFOLD: the tier's stage/flush/residency lifecycle already ran in the
        builder (md.amass_tier). The tiered V DELIVERY (decode ``V_SRC==1`` load)
        is the FOLLOW-UP, so today we delegate to stock FA over the still-resident
        V -- correct output while the tier is validated in shadow.

        FOLLOW-UP (when ``_runtime.mem_stageb_wired()``): build a TierVSource for
        this layer and run the SAME Stage-A select + Stage-B decode as fast, only
        swapping ResidentVSource -> tier.vsource(lidx). Sketch left in place so
        the wiring is a fill, not a redesign."""
        tier = getattr(m, "amass_tier", None) if m is not None else None
        if (not _runtime.mem_stageb_wired() or tier is None or output is None
                or m is None or m.max_query_len != 1 or kv_cache.numel() == 0):
            return super().forward(layer, query, key, value, kv_cache, m,
                                   output, output_scale=output_scale,
                                   output_block_scale=output_block_scale)
        # --- FOLLOW-UP tiered decode (unreachable until the flag flips) ----- #
        # lidx = _runtime.tier_layer_index(tier, kv_cache)          # ptr -> layer
        # st = getattr(m, "amass", None)                            # Stage-A state
        # n_req = m.seq_lens.shape[0]
        # out3 = output if output.dim() == 3 else output.view(
        #     output.shape[0], self.num_heads, -1)
        # select_pages(query, kv_cache, m.block_table, m.seq_lens, st, n_req,
        #              self.scale, _runtime.get_config())
        # tier.step(lidx, st.page_table, st.page_cnt, m.block_table,
        #           m.seq_lens, n_req)              # residency gather -> hot
        # _runtime.decode_dispatch(query, kv_cache, m.block_table, m.seq_lens,
        #                          st, out3, tier.vsource(lidx), self.scale,
        #                          _runtime.get_config())
        # return output
        raise RuntimeError("mem-v tiered decode reached with the wiring flag "
                           "set but the follow-up path is still a sketch")
