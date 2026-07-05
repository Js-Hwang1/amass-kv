"""Stage B, launch 2 of 2: the split-K merge.

The split kernel (``attention/decode.py``) writes one online-softmax partial
``(m, l, acc)`` per (request, kv-head, split). This kernel recombines the active
splits of each (request, query-head) into the final attention output with the
standard flash-decode rescale, and writes ``out`` in place.

Active-split count is recomputed on device from ``page_cnt`` (the same
``pps = max(cdiv(cnt, SPLIT), PPS_MIN)`` rule the split kernel used), so inactive
splits that stored nothing are masked out and never read. Launch shape is fixed
(``(n_req, n_kv*G)``); every dynamic quantity is a device-side value -> the whole
Stage B stays inside a FULL CUDA graph. Padded request rows (``page_cnt == 0``
everywhere) write zeros, never NaN.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def merge_splits_kernel(m_ptr, l_ptr, acc_ptr, cnt_ptr, o_ptr,
                        stride_ot, stride_oh, stride_pr, stride_cntr,
                        G: tl.constexpr, D: tl.constexpr,
                        SPLIT: tl.constexpr, SPLIT_PAD: tl.constexpr,
                        PPS_MIN: tl.constexpr):
    r = tl.program_id(0)
    h = tl.program_id(1)
    kh = h // G
    g = h % G
    cnt = tl.load(cnt_ptr + r * stride_cntr + kh)
    pps = tl.maximum(tl.cdiv(cnt, SPLIT), PPS_MIN)
    n_act = tl.cdiv(cnt, pps)              # splits that actually stored a partial
    offs_s = tl.arange(0, SPLIT_PAD)
    offs_d = tl.arange(0, D)
    smask = offs_s < n_act
    idx = r * stride_pr + (kh * SPLIT + offs_s) * G + g
    m = tl.load(m_ptr + idx, mask=smask, other=float("-inf"))
    l = tl.load(l_ptr + idx, mask=smask, other=0.0)
    acc = tl.load(acc_ptr + idx[:, None] * D + offs_d[None, :],
                  mask=smask[:, None], other=0.0)
    m_max = tl.max(m, axis=0)
    # empty splits carry (m=-inf, l=0): weight 0 (the `where` guards -inf - -inf)
    w = tl.where(l > 0, tl.exp(m - m_max), 0.0)
    l_tot = tl.sum(l * w, axis=0)
    o = tl.sum(acc * w[:, None], axis=0)
    # padded request rows (cnt = 0 everywhere): write 0, never NaN
    o = tl.where(l_tot > 0, o / l_tot, 0.0)
    tl.store(o_ptr + r * stride_ot + h * stride_oh + offs_d,
             o.to(o_ptr.dtype.element_ty))
