"""R8State — persistent buffers for the STATIC r8-ranked selection pipeline.

Layout is fixed by ``ours_doc/AMASS_R8_STATIC_SPEC.md`` ("Data layouts"); every
tensor is allocated ONCE from engine maxima so its address is stable across
decode steps (a hard requirement for CUDA-graph capture). The hot kernels
(``r8_score``, ``topb_select``) read these buffers with fixed launch shapes, no
host sync, no allocation.

R8 state is keyed by PHYSICAL block id and stored L-major (one slab per layer):

  r8_mu    (L, NB, n_kv, d)       int8   + mu_scale  (L, NB, n_kv)      fp16
  r8_Vk    (L, NB, n_kv, d, r)    int8   + Vk_scale  (L, NB, n_kv, r)   fp16
  r8_c     (L, NB, n_kv, page, r) int8   + c_scale   (L, NB, n_kv, page) fp16
  r8_tag   (L, NB, n_kv, TAGW)    bf16   (content-tag invalidation, NaN-init)

Per-step outputs / scratch (keyed by REQUEST row, sized from engine maxima):

  score      (R, n_kv, MP)  fp32     group-maxed r8 page score (kv-union)
  page_table (R, n_kv, MP)  int32    selected logical page ids, -1 padded
  page_cnt   (R, n_kv)      int32
  n_pages / n_sel_hi / b_fix (R,) int32   derived once per step (BatchedDecode-
                                          State-style; ``static b`` per request,
                                          identical across heads and layers)

Footprint of the selector state per (page, kv-head): d + d*r + page*r int8
= 128 + 1024 + 128 = 1.25 KB, ~3.2x smaller than a 4 KB bf16 K page.
"""
from __future__ import annotations

import torch

TAGW = 8      # bf16 leading key channels of the page-final token = content tag


class R8State:
    """Persistent r8 selection state + per-step derived params.

    Allocated once (metadata-builder init) from engine maxima. ``page``,
    ``sink_pages``, ``window_pages`` and ``budget`` are configuration
    constants captured here so the per-step ``derive_page_params`` needs only
    ``seq_lens``.
    """

    def __init__(self, device, *, num_layers: int, num_blocks: int,
                 n_kv: int, G: int, head_dim: int, page: int,
                 max_reqs: int, max_pages: int,
                 rank: int = 8, budget: float = 0.1,
                 sink_pages: int = 1, window_pages: int = 1,
                 tagw: int = TAGW,
                 vk_grain: str = "col", c_grain: str = "page",
                 vk_bits: int = 8, c_bits: int = 8, mu_bits: int = 8):
        assert window_pages >= 1, "window_pages>=1: the partial tail must live " \
            "inside the always-attended window so the selectable region is exact"
        assert vk_grain in ("col", "tensor")
        assert c_grain in ("page", "tensor")
        assert vk_bits in (4, 8) and c_bits in (4, 8) and mu_bits == 8, \
            "Piece B lossless config: Vk/c int4 or int8, mu STAYS int8 (int4 mu " \
            "breaks the DC term -- RECALL_GATE)"
        L, NB, D, r, MP, R = num_layers, num_blocks, head_dim, rank, max_pages, \
            max_reqs
        self.L, self.NB, self.n_kv, self.G, self.d = L, NB, n_kv, G, D
        self.page, self.r, self.tagw = page, r, tagw
        self.max_reqs, self.max_pages = R, MP
        self.budget = float(budget)
        self.sink_pages, self.window_pages = int(sink_pages), int(window_pages)
        self.vk_grain, self.c_grain = vk_grain, c_grain
        self.vk_bits, self.c_bits, self.mu_bits = int(vk_bits), int(c_bits), 8
        self.device = device
        i8, f16, bf16 = torch.int8, torch.float16, torch.bfloat16
        i32, f32, f64 = torch.int32, torch.float32, torch.float64
        # int4 packs two signed nibbles per int8 byte along the r (rank) axis, so
        # the code tensor's last dim halves; the fp16 scale layout is unchanged.
        rvk = r // 2 if vk_bits == 4 else r
        rc = r // 2 if c_bits == 4 else r

        # ---- r8 selector state (per layer, physical-block keyed) --------- #
        self.r8_mu = torch.zeros(L, NB, n_kv, D, device=device, dtype=i8)
        self.mu_scale = torch.zeros(L, NB, n_kv, device=device, dtype=f16)
        self.r8_Vk = torch.zeros(L, NB, n_kv, D, rvk, device=device, dtype=i8)
        self.Vk_scale = torch.zeros(L, NB, n_kv, r, device=device, dtype=f16)
        self.r8_c = torch.zeros(L, NB, n_kv, page, rc, device=device, dtype=i8)
        self.c_scale = torch.zeros(L, NB, n_kv, page, device=device, dtype=f16)
        # NaN-init tag: NaN != anything -> every page builds on first touch.
        self.r8_tag = torch.full((L, NB, n_kv, tagw), float("nan"),
                                 device=device, dtype=bf16)

        # ---- per-step outputs / scratch (request-row keyed) -------------- #
        self.score = torch.empty(R, n_kv, MP, device=device, dtype=f32)
        self.page_table = torch.empty(R, n_kv, MP, device=device, dtype=i32)
        self.page_cnt = torch.zeros(R, n_kv, device=device, dtype=i32)

        # ---- per-request derived params (BatchedDecodeState-style) ------- #
        self.n_pages = torch.zeros(R, device=device, dtype=i32)
        self.n_sel_hi = torch.zeros(R, device=device, dtype=i32)
        self.b_fix = torch.ones(R, device=device, dtype=i32)
        # fp64 budget in a 1-elem tensor: Triton float scalars are fp32 and
        # ceil(budget*S) must match a host float64 ceil (v1 parity).
        self.budget64 = torch.tensor([self.budget], device=device, dtype=f64)

    # --------------------------------------------------------------------- #
    def bytes_per_layer(self) -> int:
        return (self.r8_mu[0].numel() + self.r8_Vk[0].numel()
                + self.r8_c[0].numel()
                + self.mu_scale[0].numel() * 2 + self.Vk_scale[0].numel() * 2
                + self.c_scale[0].numel() * 2 + self.r8_tag[0].numel() * 2)

    def layer_state(self, layer: int):
        """Return the per-layer (mu, mu_scale, Vk, Vk_scale, c, c_scale, tag)
        views for ``layer`` (contiguous slabs; in-place writes propagate)."""
        return (self.r8_mu[layer], self.mu_scale[layer], self.r8_Vk[layer],
                self.Vk_scale[layer], self.r8_c[layer], self.c_scale[layer],
                self.r8_tag[layer])
