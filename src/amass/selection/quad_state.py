"""QuadState — persistent buffers for the Gaussian-MGF QUADRATIC page score.

The quad score mode (``ours_doc/QUAD_SUMMARY_METHOD.md`` /
``ours_doc/QUAD_IMPL_SPEC.md``) replaces the r8 per-key logsumexp page-mass
estimate with its 2nd-order (Gaussian / MGF) expansion, a linear + low-rank
quadratic form in the query.  That drops the per-key coordinate block ``c`` (the
whole r8->quad saving) and lets the rank fall to r'=2, so the resident selector
shrinks from ~16% of the KV to ~4.7% (3.2% int4).

QuadState MIRRORS :class:`R8State` (alloc-once / graph-safe / physical-block
keyed, one L-major slab per layer) with exactly ONE structural change to the
per-layer selector state: the r8 ``c`` (page, r) block is replaced by the r'
eigenvalues ``sig2`` (the k-th eigenvalue of the page's centered-key scatter):

  quad_mu    (L, NB, n_kv, d)       int8   + mu_scale  (L, NB, n_kv)      fp16
  quad_V     (L, NB, n_kv, d, r')   int8   + V_scale   (L, NB, n_kv, r')  fp16 (per-col)
  quad_sig2  (L, NB, n_kv, r')      fp16   (eigenvalues sigma_k^2; small, fp16)
  quad_tag   (L, NB, n_kv, TAGW)    bf16   (content-tag invalidation, NaN-init)

The per-step outputs / scratch (score, page_table, page_cnt, n_pages, n_sel_hi,
b_fix, budget64) are BYTE-IDENTICAL to R8State's, so ``derive_page_params`` and
``topb_select`` (the radix tau-select) are reused unchanged: quad only swaps the
summary state + the build + the score kernel.

Footprint per (page, kv-head) at r'=2, int8 V:
  mu 128 + V(d*r')=256 + sig2(r' fp16)=4 + scales(mu 2 + V 4)=6 + tag 16 = ~410 B
= ~5% of the 8 KB bf16 K+V page (vs r8's ~1.33 KB / 16%): ~3.1-3.4x smaller.
"""
from __future__ import annotations

import torch

from .state import TAGW


class QuadState:
    """Persistent quad selection state + per-step derived params.

    Allocated once (metadata-builder init) from engine maxima; every tensor
    address is stable across decode steps (CUDA-graph capture requirement).
    ``rank`` here is the QUAD rank r' (default 2), NOT the r8 rank-8.
    """

    def __init__(self, device, *, num_layers: int, num_blocks: int,
                 n_kv: int, G: int, head_dim: int, page: int,
                 max_reqs: int, max_pages: int,
                 rank: int = 2, budget: float = 0.1,
                 sink_pages: int = 1, window_pages: int = 1,
                 tagw: int = TAGW,
                 v_grain: str = "col", v_bits: int = 8, mu_bits: int = 8):
        assert window_pages >= 1, "window_pages>=1: the partial tail must live " \
            "inside the always-attended window so the selectable region is exact"
        assert v_grain in ("col", "tensor")
        assert v_bits in (4, 8) and mu_bits == 8, \
            "quad config: V int4 or int8, mu STAYS int8 (int4 mu breaks the DC " \
            "term -- see RECALL_GATE)"
        assert rank >= 1
        if v_bits == 4:
            assert rank % 2 == 0, "int4 V packs 2 nibbles/byte along r' -> r' even"
        L, NB, D, r, MP, R = num_layers, num_blocks, head_dim, rank, max_pages, \
            max_reqs
        self.L, self.NB, self.n_kv, self.G, self.d = L, NB, n_kv, G, D
        self.page, self.r, self.tagw = page, r, tagw
        self.max_reqs, self.max_pages = R, MP
        self.budget = float(budget)
        self.sink_pages, self.window_pages = int(sink_pages), int(window_pages)
        self.v_grain = v_grain
        self.v_bits, self.mu_bits = int(v_bits), 8
        self.device = device
        i8, f16, bf16 = torch.int8, torch.float16, torch.bfloat16
        i32, f32, f64 = torch.int32, torch.float32, torch.float64
        # int4 packs two signed nibbles per int8 byte along the r' (rank) axis.
        rv = r // 2 if v_bits == 4 else r

        # ---- quad selector state (per layer, physical-block keyed) -------- #
        self.quad_mu = torch.zeros(L, NB, n_kv, D, device=device, dtype=i8)
        self.mu_scale = torch.zeros(L, NB, n_kv, device=device, dtype=f16)
        self.quad_V = torch.zeros(L, NB, n_kv, D, rv, device=device, dtype=i8)
        self.V_scale = torch.zeros(L, NB, n_kv, r, device=device, dtype=f16)
        self.quad_sig2 = torch.zeros(L, NB, n_kv, r, device=device, dtype=f16)
        # NaN-init tag: NaN != anything -> every page builds on first touch.
        self.quad_tag = torch.full((L, NB, n_kv, tagw), float("nan"),
                                   device=device, dtype=bf16)

        # ---- per-step outputs / scratch (request-row keyed) -------------- #
        # IDENTICAL to R8State so derive_page_params + topb_select are reused.
        self.score = torch.empty(R, n_kv, MP, device=device, dtype=f32)
        self.page_table = torch.empty(R, n_kv, MP, device=device, dtype=i32)
        self.page_cnt = torch.zeros(R, n_kv, device=device, dtype=i32)

        # ---- per-request derived params (BatchedDecodeState-style) ------- #
        self.n_pages = torch.zeros(R, device=device, dtype=i32)
        self.n_sel_hi = torch.zeros(R, device=device, dtype=i32)
        self.b_fix = torch.ones(R, device=device, dtype=i32)
        self.budget64 = torch.tensor([self.budget], device=device, dtype=f64)

    # --------------------------------------------------------------------- #
    def bytes_per_layer(self) -> int:
        return (self.quad_mu[0].numel() + self.quad_V[0].numel()
                + self.quad_sig2[0].numel() * 2
                + self.mu_scale[0].numel() * 2 + self.V_scale[0].numel() * 2
                + self.quad_tag[0].numel() * 2)

    def layer_state(self, layer: int):
        """Return the per-layer (mu, mu_scale, V, V_scale, sig2, tag) views for
        ``layer`` (contiguous slabs; in-place writes propagate)."""
        return (self.quad_mu[layer], self.mu_scale[layer], self.quad_V[layer],
                self.V_scale[layer], self.quad_sig2[layer], self.quad_tag[layer])
