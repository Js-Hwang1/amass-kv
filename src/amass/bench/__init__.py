"""``amass.bench`` -- the AMASS internal benchmarker (module + CLI).

A trustworthy, reusable benchmarker that ships INSIDE the ``amass`` package. It
measures the two things that matter for the release claim, with reproducible
CUDA-event timing (warmup + medians, eager AND CUDA-graph replay, roofline
context) so results are defensible rather than lucky:

  * **Per-kernel latency** (:mod:`amass.bench.kernels`): each hot kernel
    (``r8_score`` / ``topb`` / ``decode``) hand-CUDA-vs-Triton, decode also vs a
    dense-FlashAttention baseline, with a CUDA-vs-Triton agreement check.
  * **End-to-end TPOT** (:mod:`amass.bench.e2e`): decode-step latency +
    throughput in a real vLLM run (FullKV vs AMASS-fast CUDA vs Triton), via the
    two-length-diff method under full cudagraph.

Public API
----------
    from amass.bench import run_kernels, run_e2e
    results = run_kernels(batches=[4], contexts=[16384], budgets=[0.1])
    e2e     = run_e2e(configs=["fullkv", "fast_cuda"], ctx=16384)

CLI
---
    amass-bench kernels            # per-kernel sweep (--quick for a fast subset)
    amass-bench e2e --ctx 16384    # end-to-end TPOT
    amass-bench all --json out.json
    python -m amass.bench kernels  # equivalent

See ``PACKAGING.md`` for the ``[project.scripts]`` entry-point line and
``README.md`` for the environment / usage details.
"""
from __future__ import annotations

from .kernels import (bench_decode, bench_r8_score, bench_topb,  # noqa: F401
                      run_kernels)
from .e2e import run_e2e  # noqa: F401
from .timing import (TimeResult, bench_graph, bench_median,  # noqa: F401
                     hbm_roofline_us, pct_of_peak_bw)

__all__ = [
    "run_kernels",
    "run_e2e",
    "bench_r8_score",
    "bench_topb",
    "bench_decode",
    "bench_median",
    "bench_graph",
    "hbm_roofline_us",
    "pct_of_peak_bw",
    "TimeResult",
]
