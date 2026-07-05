"""Reproducible GPU timing primitives for the AMASS benchmarker.

Everything here uses **CUDA-event** timing (never wall clock) with an explicit
warmup, and reports a **median over repeated batches** so a single unlucky
launch cannot move the number. Two flavours:

  * :func:`bench_median` -- eager (stream) timing: median over ``reps`` batches,
    each batch the mean of ``iters`` back-to-back launches inside one event pair
    (amortizes the ~5 us event/launch overhead the hot kernels live near).
  * :func:`bench_graph`  -- CUDA-graph replay timing: capture ``fn()`` once
    (side-stream warmup, as CUDA capture requires) then time ``g.replay()`` the
    same way. This is the production number for a graph-captured decode step and
    it strips per-launch CPU overhead entirely. Returns ``ok=False`` (never
    raises) if the callable is not capture-safe.

The roofline helpers report the HBM-bandwidth floor so a bandwidth-bound kernel's
result carries its "% of peak BW" context (H200 HBM3e ~4.8 TB/s peak).

These are the SAME methodology the build agents used (``scratch_r8d/bench_cuda``,
``scratch_r8f/bench_cuda``): event pair around a warm loop, medians, and a
capture/replay path -- packaged once so every kernel and the e2e path share it.
"""
from __future__ import annotations

import dataclasses
import statistics
from typing import Callable, List, Optional

import torch

# H200 SXM/NVL HBM3e peak bandwidth (TB/s). Used only for the roofline context;
# the measured "% of peak" is reported against this, not asserted.
H200_HBM_TBS = 4.8


@dataclasses.dataclass
class TimeResult:
    """A timing measurement in microseconds (median of repeated batches)."""

    us: float                     # median per-call latency (us)
    min_us: float                 # fastest batch (us)
    reps_us: List[float]          # per-batch medians (us), sorted
    mode: str                     # "eager" | "graph"
    ok: bool = True               # False => capture failed (graph mode)
    err: Optional[str] = None     # capture error, if any

    @property
    def std_us(self) -> float:
        return statistics.pstdev(self.reps_us) if len(self.reps_us) > 1 else 0.0

    def __repr__(self) -> str:  # pragma: no cover - display only
        if not self.ok:
            return f"<TimeResult {self.mode} FAILED: {self.err}>"
        return (f"<TimeResult {self.mode} {self.us:.1f}us "
                f"(min {self.min_us:.1f}, +/-{self.std_us:.1f})>")


def _event_loop_us(fn: Callable[[], None], iters: int) -> float:
    """Mean per-call latency (us) of ``iters`` back-to-back launches, one CUDA
    event pair around the whole loop (so event overhead is amortized)."""
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1e3  # ms -> us


def bench_median(fn: Callable[[], None], *, warmup: int = 25, iters: int = 50,
                 reps: int = 7) -> TimeResult:
    """Eager CUDA-event timing: median over ``reps`` batches of ``iters`` calls.

    ``warmup`` launches run first (build JIT, warm caches, reach steady clocks),
    then each of ``reps`` batches times ``iters`` back-to-back launches inside a
    single event pair. The reported ``us`` is the median batch, ``min_us`` the
    fastest -- report both so a noisy tail is visible.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    reps_us = sorted(_event_loop_us(fn, iters) for _ in range(reps))
    return TimeResult(us=statistics.median(reps_us), min_us=reps_us[0],
                      reps_us=reps_us, mode="eager")


def bench_graph(fn: Callable[[], None], *, warmup: int = 10, iters: int = 50,
                reps: int = 7) -> TimeResult:
    """CUDA-graph replay timing: capture ``fn()`` once, time ``g.replay()``.

    Warmup runs on a side stream (required before capture); the graph is then
    captured and replayed. Timing uses the same median-of-batches scheme. If the
    callable is not capture-safe (host sync / allocation / dynamic shape) this
    returns ``ok=False`` with the error instead of raising -- the caller decides
    whether that is expected (e.g. a dense-SDPA baseline is not captured here).
    """
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        for _ in range(warmup):
            g.replay()
        torch.cuda.synchronize()
        reps_us = sorted(_event_loop_us(g.replay, iters) for _ in range(reps))
        # keep `g` referenced until timing is done (it is, above)
        return TimeResult(us=statistics.median(reps_us), min_us=reps_us[0],
                          reps_us=reps_us, mode="graph")
    except Exception as e:  # noqa: BLE001
        return TimeResult(us=float("nan"), min_us=float("nan"), reps_us=[],
                          mode="graph", ok=False, err=f"{type(e).__name__}: {e}")


def hbm_roofline_us(bytes_moved: float, hbm_tbs: float = H200_HBM_TBS) -> float:
    """Lower-bound latency (us) to move ``bytes_moved`` at ``hbm_tbs`` TB/s."""
    return bytes_moved / (hbm_tbs * 1e12) * 1e6


def pct_of_peak_bw(us: float, bytes_moved: float,
                   hbm_tbs: float = H200_HBM_TBS) -> float:
    """Achieved HBM bandwidth as a percentage of ``hbm_tbs`` peak."""
    if us <= 0:
        return float("nan")
    achieved_tbs = bytes_moved / (us * 1e-6) / 1e12
    return achieved_tbs / hbm_tbs * 100.0
