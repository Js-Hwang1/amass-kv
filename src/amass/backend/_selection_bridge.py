"""TEMPORARY Stage-A validation bridge (delete when ``amass.selection`` lands).

An eager, torch-only exact page-mass selector so the AMASS-fast Stage-B path is
runnable and testable end-to-end BEFORE the sibling agent's graph-safe
``select_pages_r8`` exists. It is intentionally simple and correct, not fast:
per (request, kv-head) it scores every page by exact ``logsumexp`` mass, keeps
the top-b of the selectable region unioned across the GQA group, plus the
always-attended sink + recent-window pages -- the same set semantics as the
prototype's ``kvunion`` mode. When ``cfg.budget`` covers everything selectable
(e.g. budget=1.0) it keeps ALL pages, i.e. exact dense -- the bitwise-vs-FullKV
gate for the decode kernel.

Because it uses data-dependent torch ops it is NOT CUDA-graph-safe; the builder
runs it PIECEWISE/eager (get_cudagraph_support -> NEVER). The real r8 path
replaces it with fixed-shape launches and re-enables full graphs.
"""
from __future__ import annotations

import math

import torch

from ..attention.decode import _split_kv, ensure_stage_b_buffers


class BridgeState:
    """Persistent buffers matching the Stage-A contract (see _runtime.py).

    Stage-B partials are allocated by ``ensure_stage_b_buffers`` (owned by
    attention/), so the exact same buffer layout serves the real path."""

    def __init__(self, device, cfg, max_reqs, n_kv, G, head_dim, max_pages,
                 split):
        R, MP, D = max_reqs, max_pages, head_dim
        self.max_reqs, self.n_kv, self.G, self.D = R, n_kv, G, D
        self.max_pages = MP
        self.cfg = cfg
        i32 = torch.int32
        self.page_table = torch.empty(R, n_kv, MP, device=device, dtype=i32)
        self.page_cnt = torch.zeros(R, n_kv, device=device, dtype=i32)
        ensure_stage_b_buffers(self, device, split)   # m_part/l_part/acc_part
        # Unused on the bridge path: the impl passes scale=self.scale explicitly
        # to both Stage A and Stage B, so decode never falls back to st.scale.
        self.scale = None


def bridge_derive(seq_lens, st, cfg) -> None:
    """No-op: the eager selector recomputes n_pages from seq_lens itself."""
    return None


def bridge_select(q, kv_cache, block_table, seq_lens, st, n_req, scale,
                  cfg) -> None:
    """Exact per-(req, kv-head) page-mass top-b -> GQA-union -> packed table.

    Writes rows [0, n_req) of ``st.page_table`` (ascending logical page indices,
    -1 pad) and ``st.page_cnt``. q is (T, H, d); rows [0, n_req) are the decode
    queries."""
    K, _ = _split_kv(kv_cache)
    _, page, n_kv, d = K.shape
    G = st.G
    sink = cfg.sink_pages
    win = cfg.window_pages
    dev = q.device
    seq_list = seq_lens[:n_req].tolist()

    st.page_table[:n_req].fill_(-1)
    st.page_cnt[:n_req].zero_()

    for r in range(n_req):
        seq_len = int(seq_list[r])
        if seq_len <= 0:
            continue
        n_pages = (seq_len + page - 1) // page
        bt = block_table[r, :n_pages].long()
        Kr = K[bt]                                   # (P, page, n_kv, d)
        Kf = Kr.permute(2, 0, 1, 3).reshape(n_kv, n_pages * page, d)
        qr = q[r].view(n_kv, G, d).to(Kf.dtype)
        logits = torch.einsum("kgd,ktd->kgt", qr, Kf).float() * scale
        logits[..., seq_len:] = float("-inf")        # cache-page padding
        pm = logits.view(n_kv, G, n_pages, page)
        pmass = torch.logsumexp(pm, dim=-1)          # (n_kv, G, P)

        n_sel_hi = n_pages - win
        n_selectable = n_sel_hi - sink
        keep = torch.zeros(n_kv, n_pages, dtype=torch.bool, device=dev)
        if sink > 0:
            keep[:, :sink] = True
        if n_sel_hi < n_pages:
            keep[:, n_sel_hi:] = True

        if n_selectable <= 1:                        # short ctx: attend all
            keep[:, :] = True
        else:
            if cfg.budget is not None:
                b = min(max(1, math.ceil(cfg.budget * n_selectable)),
                        n_selectable)
            else:                                    # coverage -> nucleus set
                b = _nucleus_b(pmass, sink, n_sel_hi, cfg.coverage)
            if b >= n_selectable:
                keep[:, sink:n_sel_hi] = True
            else:
                sel = pmass[:, :, sink:n_sel_hi]     # (n_kv, G, S)
                top = sel.topk(b, dim=-1).indices    # (n_kv, G, b)
                um = torch.zeros(n_kv, n_selectable, dtype=torch.bool,
                                 device=dev)
                um.scatter_(-1, top.reshape(n_kv, G * b), True)  # GQA union
                keep[:, sink:n_sel_hi] |= um

        for kh in range(n_kv):
            idx = keep[kh].nonzero(as_tuple=False).flatten().to(torch.int32)
            c = int(idx.numel())
            st.page_table[r, kh, :c] = idx
            st.page_cnt[r, kh] = c


def _nucleus_b(pmass, sink, n_sel_hi, coverage) -> int:
    """Smallest b s.t. the top-b selectable pages cover `coverage` of the mass
    for the WORST query head (so every head is covered) -- a scalar b for the
    static bridge. Real per-head adaptivity is the r8 path's job."""
    probs = torch.softmax(pmass, dim=-1)             # (n_kv, G, P)
    selp = probs[:, :, sink:n_sel_hi]                # (n_kv, G, S)
    always = 1.0 - selp.sum(-1)                      # sinks+window mass per head
    need = (coverage - always).clamp_min(0.0)
    sp, _ = selp.sort(dim=-1, descending=True)
    csum = sp.cumsum(-1)
    reached = csum >= need.unsqueeze(-1)             # (n_kv, G, S)
    # first index reaching `need`, per head; b = max over heads (+1 for 1-based)
    S = selp.shape[-1]
    first = torch.where(reached.any(-1),
                        reached.float().argmax(-1),
                        torch.full_like(need, S - 1).long())
    return int(first.max().item()) + 1
