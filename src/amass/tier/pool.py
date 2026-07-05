"""MappedHostVPool -- the pinned, device-mapped host V pool (UVA zero-copy).

The memory play: the full-precision V of every physical KV block lives in cheap
HOST DRAM, mapped into the GPU address space (cudaHostGetDevicePointer / UVA) so
decode kernels can gather it zero-copy over PCIe. Only a bounded hot buffer +
staging pool stay in VRAM (see residency.py / staging.py).

SIZING DISCIPLINE (the flagged-bug boundary).
  * This pinned pool is keyed by PHYSICAL block id and sized (L, NB, n_kv,
    page, d) -- i.e. by ``num_gpu_blocks``. That is BY DESIGN and correct: this
    is HOST DRAM, the offload TARGET; it must hold the exact V bits of every
    physical block that could be live, so any flushed page is retrievable (I1).
    Host DRAM is abundant and cheap, which is the entire point of the tier.
  * The flagged 17GiB bug was sizing VRAM-RESIDENT per-block state by
    ``num_gpu_blocks`` (the fast path's R8State: r8_Vk was L*NB*n_kv*d*r int8).
    The tier NEVER does that: its VRAM structures (the hot buffer, the staging
    v_pool) are bounded to the WORKING SET (hot_slots S, v_blocks NV), NOT NB;
    the only NB-sized VRAM is the tiny int8/int32 residency bookkeeping
    (~1-4 bytes/block), not KV-sized. See ``MappedHostVPool.vram_note``.
  * ``max_pool_gb`` caps the HOST allocation and RAISES loudly rather than
    silently truncating (a truncated pool would violate I1: some block's V would
    have nowhere to live).

Ported clean from the prototype ``vllm/kernels/dram_tier.py`` (MappedHostPool /
host_device_pointer / DevPtrTensor); the debug/telemetry cruft is dropped.
"""
from __future__ import annotations

import ctypes
import glob
import os

import torch

# --------------------------------------------------------------------------- #
# libcudart -> cudaHostGetDevicePointer (UVA device address of pinned host mem) #
# --------------------------------------------------------------------------- #
_libcudart = None


def _cudart():
    """Load libcudart once and bind cudaHostGetDevicePointer. Searched on the
    system paths plus the torch-bundled nvidia wheel (the container's stack)."""
    global _libcudart
    if _libcudart is not None:
        return _libcudart
    last = None
    names = ["libcudart.so", "libcudart.so.13", "libcudart.so.12"]
    names += sorted(glob.glob(os.path.join(
        os.path.dirname(torch.__file__), "..", "nvidia", "cuda_runtime", "lib",
        "libcudart.so*")))
    for name in names:
        try:
            lib = ctypes.CDLL(name)
            lib.cudaHostGetDevicePointer.restype = ctypes.c_int
            lib.cudaHostGetDevicePointer.argtypes = [
                ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
            _libcudart = lib
            return lib
        except (OSError, AttributeError) as exc:  # pragma: no cover
            last = exc
    raise RuntimeError(f"could not load libcudart: {last}")


def host_device_pointer(host_tensor: torch.Tensor) -> int:
    """Device-side address of a pinned host tensor (UVA zero-copy).

    For torch pinned memory under UVA this equals ``host_tensor.data_ptr()``, but
    we still round-trip through ``cudaHostGetDevicePointer`` so a non-UVA / weird
    allocator setup fails LOUDLY here instead of faulting inside a kernel."""
    if not host_tensor.is_pinned():
        raise ValueError("pool tensor must be pinned (pin_memory=True)")
    dptr = ctypes.c_void_p(0)
    rc = _cudart().cudaHostGetDevicePointer(
        ctypes.byref(dptr), ctypes.c_void_p(host_tensor.data_ptr()), 0)
    if rc != 0:
        raise RuntimeError(f"cudaHostGetDevicePointer failed rc={rc} "
                           "(pinned memory not device-mapped?)")
    return dptr.value


class DevPtrTensor:
    """Duck-typed tensor: lets Triton / a raw-pointer kernel launch on a device
    address that has no ``torch.Tensor`` wrapper.

    Triton's JIT only needs ``.data_ptr()`` + ``.dtype`` from a tensor argument
    (plus the pointer value for alignment specialization), so this is enough to
    hand a device-mapped host pool to a kernel. The CUDA extension takes the raw
    ``data_ptr()`` directly."""

    __slots__ = ("_ptr", "dtype", "device", "shape")

    def __init__(self, ptr: int, dtype: torch.dtype, device: torch.device,
                 shape=()):
        self._ptr = int(ptr)
        self.dtype = dtype
        self.device = device
        self.shape = shape

    def data_ptr(self) -> int:
        return self._ptr


class MappedHostVPool:
    """Pinned host V pool + its kernel-launchable device-pointer view.

    Layout ``(L, NB, n_kv, page, d)`` -- L-major, then physical block, so a
    (layer, block, kv-head) tile is contiguous. The pool address of a
    (l, blk, kv) page is ``((l*NB + blk)*n_kv + kv) * page * d`` (element index),
    matching the reference gather/flush kernels' ``pool_off`` arithmetic.

    The device view is exposed as ``int16`` so kernels do PURE BIT copies (no
    float canonicalization) -- V bytes must survive the host round-trip exactly
    for byte-identical output vs the resident path (the equality gate).
    """

    def __init__(self, *, num_layers: int, num_blocks: int, n_kv: int,
                 page: int, d: int, dtype: torch.dtype = torch.bfloat16,
                 device=None, max_pool_gb: float = 256.0):
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"pool dtype must be bf16/fp16, got {dtype}")
        self.L, self.NB, self.n_kv = num_layers, num_blocks, n_kv
        self.page, self.d, self.dtype = page, d, dtype
        elem = torch.empty(0, dtype=dtype).element_size()
        pool_bytes = num_layers * num_blocks * n_kv * page * d * elem
        if pool_bytes > max_pool_gb * (1 << 30):
            raise RuntimeError(
                f"pinned V pool would be {pool_bytes / 2**30:.1f} GiB "
                f"(> cap max_pool_gb={max_pool_gb}); reduce num_gpu_blocks "
                "(num_gpu_blocks_override) or raise AmassConfig.max_pool_gb. "
                "NOTE this is HOST DRAM (the offload target), not VRAM.")
        dev = torch.device(device) if device is not None else \
            torch.device("cuda", torch.cuda.current_device())
        # (L, NB, n_kv, page, d) pinned host allocation.
        self.host = torch.zeros((num_layers, num_blocks, n_kv, page, d),
                                dtype=dtype, pin_memory=True)
        # int16 device view: bit-exact copies, no float canonicalize.
        self.dev_view = DevPtrTensor(host_device_pointer(self.host),
                                     torch.int16, dev, tuple(self.host.shape))
        self.pool_gib = pool_bytes / 2**30
        self.device = dev

    # ---- offsets (element indices; kernels take the raw dev pointer) ------- #
    def page_offset(self, layer: int, blk: int, kv: int) -> int:
        """Element offset of the (layer, block, kv-head) page in the pool."""
        return (((layer * self.NB + blk) * self.n_kv + kv) * self.page * self.d)

    def vram_note(self) -> str:
        """One-line reminder of the sizing discipline (used in logs)."""
        return (f"host_pool={self.pool_gib:.2f}GiB (DRAM, keyed by NB={self.NB}; "
                "VRAM hot/staging bounded to working set, see residency/staging)")
