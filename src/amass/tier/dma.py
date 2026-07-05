"""dma -- copy-engine DMA for the miss fetch (kills the UVA-read-kernel SM steal).

The prototype gather reads the pinned host pool with a Triton kernel over UVA:
that kernel HOLDS SMs while it stalls on PCIe, so it CONTENDS with the concurrent
decode (measured: running the UVA gather on a side stream alongside decode is
SLOWER than serial -- negative overlap; scratch_memopt2/contention.py). A
``cudaMemcpyAsync`` on a side stream instead drives the hardware COPY ENGINE,
which does not occupy SMs, so a *contiguous* H2D transfer overlaps decode ~80%
(measured 56 GB/s vs 20 GB/s UVA). The batched ``cudaMemcpyBatchAsync`` (CUDA 13)
issues many scattered copies in one host call, but for small (4 KB) pages it is
implemented as an SM-internal copy kernel -> it does NOT overlap (measured -1%).

So the copy engine only helps when the transfer is CONTIGUOUS. This module wraps
both primitives (single contiguous ``memcpy_async`` and scattered
``memcpy_batch``) on a dedicated side stream with a double-buffered event, and
lets the residency gather route the FLUSHED (pinned) miss pages through the copy
engine while the (rare, recent) STAGED pages stay on the cheap in-VRAM kernel.
The residency exposes contiguous physical-block RUNS (the vLLM allocator hands
out runs), which coalesce into a strided 2-D copy the copy engine overlaps.
"""
from __future__ import annotations

import ctypes
import glob
import os

import torch

_H2D = 1                      # cudaMemcpyHostToDevice
_libcudart = None


def _cudart():
    global _libcudart
    if _libcudart is not None:
        return _libcudart
    names = ["libcudart.so", "libcudart.so.13", "libcudart.so.12"]
    names += sorted(glob.glob(os.path.join(
        os.path.dirname(torch.__file__), "..", "nvidia", "cuda_runtime", "lib",
        "libcudart.so*")))
    last = None
    for name in names:
        try:
            lib = ctypes.CDLL(name)
            lib.cudaMemcpyAsync.restype = ctypes.c_int
            lib.cudaMemcpyAsync.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                ctypes.c_int, ctypes.c_void_p]
            lib.cudaMemcpy2DAsync.restype = ctypes.c_int
            lib.cudaMemcpy2DAsync.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                ctypes.c_int, ctypes.c_void_p]
            _has_batch = hasattr(lib, "cudaMemcpyBatchAsync")
            if _has_batch:
                lib.cudaMemcpyBatchAsync.restype = ctypes.c_int
                lib.cudaMemcpyBatchAsync.argtypes = (
                    [ctypes.c_void_p] * 3 + [ctypes.c_size_t]
                    + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                       ctypes.c_void_p])
            lib._amass_has_batch = _has_batch
            _libcudart = lib
            return lib
        except (OSError, AttributeError) as exc:  # pragma: no cover
            last = exc
    raise RuntimeError(f"could not load libcudart for DMA: {last}")


class _BatchAttr(ctypes.Structure):
    # cudaMemcpyAttributes: srcAccessOrder + {src,dst}LocHint{type,id} + flags
    _fields_ = [("srcAccessOrder", ctypes.c_int),
                ("srcLocType", ctypes.c_int), ("srcLocId", ctypes.c_int),
                ("dstLocType", ctypes.c_int), ("dstLocId", ctypes.c_int),
                ("flags", ctypes.c_uint)]


class CopyEngine:
    """Double-buffered copy-engine DMA on dedicated side streams.

    ``memcpy_async`` = one contiguous H2D (true copy engine, overlaps decode).
    ``memcpy_2d_async`` = strided 2-D H2D (a coalesced physical-block run for one
    kv-head; still copy engine). ``memcpy_batch`` = scattered one-call fallback
    (CUDA 13; SM-internal for small pages -> use only when no run coalescing).
    Double-buffered: alternate streams so step N+1's prefetch does not serialise
    behind step N's consume.
    """

    def __init__(self, n_buffers: int = 2):
        self.lib = _cudart()
        self.streams = [torch.cuda.Stream() for _ in range(n_buffers)]
        self.events = [torch.cuda.Event() for _ in range(n_buffers)]
        self._i = 0
        # one attribute set: source accessed in stream order (Stream=1).
        self._attr = _BatchAttr(1, 0, 0, 0, 0, 0)
        self._aidx = (ctypes.c_size_t * 1)(0)
        self.has_batch = getattr(self.lib, "_amass_has_batch", False)

    def next_stream(self) -> torch.cuda.Stream:
        s = self.streams[self._i]
        self._i = (self._i + 1) % len(self.streams)
        return s

    def _sptr(self, stream):
        return ctypes.c_void_p(stream.cuda_stream)

    def memcpy_async(self, dst_ptr: int, src_ptr: int, nbytes: int,
                     stream) -> None:
        rc = self.lib.cudaMemcpyAsync(
            ctypes.c_void_p(dst_ptr), ctypes.c_void_p(src_ptr), nbytes,
            _H2D, self._sptr(stream))
        if rc != 0:
            raise RuntimeError(f"cudaMemcpyAsync rc={rc}")

    def memcpy_2d_async(self, dst_ptr: int, dpitch: int, src_ptr: int,
                        spitch: int, width_bytes: int, height: int,
                        stream) -> None:
        """Strided 2-D H2D: ``height`` rows of ``width_bytes``, dst rows every
        ``dpitch`` bytes, src rows every ``spitch`` bytes."""
        rc = self.lib.cudaMemcpy2DAsync(
            ctypes.c_void_p(dst_ptr), dpitch, ctypes.c_void_p(src_ptr), spitch,
            width_bytes, height, _H2D, self._sptr(stream))
        if rc != 0:
            raise RuntimeError(f"cudaMemcpy2DAsync rc={rc}")

    def memcpy_batch(self, dsts, srcs, sizes, count: int, stream) -> None:
        """One-call scattered batch (CUDA 13). ``dsts/srcs`` are ctypes
        ``c_void_p`` arrays, ``sizes`` a ``c_size_t`` array."""
        if not self.has_batch:
            raise RuntimeError("cudaMemcpyBatchAsync unavailable (< CUDA 12.8)")
        rc = self.lib.cudaMemcpyBatchAsync(
            dsts, srcs, sizes, count,
            ctypes.byref(self._attr), self._aidx, 1, self._sptr(stream))
        if rc != 0:
            raise RuntimeError(f"cudaMemcpyBatchAsync rc={rc}")

    def record(self, stream) -> torch.cuda.Event:
        ev = self.events[self.streams.index(stream)] if stream in self.streams \
            else torch.cuda.Event()
        ev.record(stream)
        return ev
