# AMASS

**Decode-time sparse KV-cache attention for vLLM.** AMASS keeps LLM decoding
fast and memory-light at long context by attending, at every decode step, to
only the pages that actually carry attention mass: a compact per-page score
selects a small working set (typically 5 to 10 percent of the KV cache), and a
sparse paged-attention kernel reads just those pages. Prefill is untouched, so
time-to-first-token is unchanged.

> **License note (provisional).** This release is distributed under Apache-2.0
> as a placeholder pending final confirmation by the authors. See `LICENSE`.

## What is in the box

AMASS has two parts that share one selection core:

1. **The quad selector (Stage A).** A quadratic, Gaussian-MGF page score built
   from a tiny per-page summary. "quad" drops the per-key coordinates and stores
   only `r'` eigenvalues per page, which shrinks the resident selector state by
   **3.28x** (from 15.6 percent down to 4.7 percent of the KV cache; measured
   1201 MiB vs 3943 MiB) while remaining accurate enough to pick the right pages.
2. **The DRAM value tier (Stage B).** A pinned, device-mapped host-DRAM pool for
   the value cache, with a bounded resident hot buffer plus staging pool. Only
   the ~5 percent quad summary and the pages actually called each step stay in
   VRAM; the rest of V (or K+V) lives in host memory.

### Two variants, one algorithm

| Variant       | KV residency                                  | The play          |
| ------------- | --------------------------------------------- | ----------------- |
| **AMASS-fast** | K+V resident in VRAM                          | speed             |
| **AMASS-mem**  | V (mem-v) or K+V (mem-kv) offloaded to DRAM; only the quad summary + called pages resident | memory + speed |

## Install

```bash
pip install amass-kv
```

The import name is `amass`:

```python
import amass
print(amass.__version__)
```

`amass-kv` declares `torch` and `triton` as its runtime dependencies. vLLM is an
**optional extra** (see [Dependencies](#dependencies)); install AMASS into your
existing vLLM environment, or pull a compatible engine with:

```bash
pip install "amass-kv[vllm]"     # engine + plugin
pip install "amass-kv[all]"      # + transformers (bench) + pyyaml (yaml configs)
```

Requires Python >= 3.10 and an NVIDIA GPU. The hot-path CUDA kernels are
compiled once at first use via `torch.utils.cpp_extension` (targeting Hopper,
`sm_90a`); the reference Triton kernels are the automatic fallback.

## Quickstart

AMASS installs as a vLLM **general plugin**, so once the wheel is present vLLM
auto-registers it at engine startup with no code change: AMASS patches itself in
wherever vLLM would have chosen the FlashAttention backend. Configuration is a
single `AmassConfig`, sourced from the `AMASS_CONFIG` environment variable (an
inline JSON string, or a path to a JSON/YAML file):

```bash
# Run any vLLM entry point (serve / offline) with AMASS active.
export AMASS_CONFIG='{"variant": "fast", "coverage": 0.95}'
vllm serve meta-llama/Llama-3.1-8B-Instruct

# The memory play: offload V to DRAM, keep K resident.
export AMASS_CONFIG='{"variant": "mem-v", "coverage": 0.95}'

# Turn AMASS off (stock FlashAttention / FullKV reference line):
export AMASS_DISABLE=1
```

Key `AmassConfig` fields: `variant` (`fast` / `mem-v` / `mem-kv`), `coverage`
(adaptive per-head nucleus coverage over exact page masses; default 0.95) or
`budget` (a fixed selected-page fraction), `score` (`quad` default, or the `r8`
low-rank fallback), and `use_cuda` (hand-CUDA hot path vs Triton reference).

## Headline results (measured)

- **3.28x smaller selector state.** The quad summary is 4.7 percent of the KV
  cache vs 15.6 percent for the low-rank `r8` baseline (1201 vs 3943 MiB
  measured), with a cheaper tail-free hot-path kernel.
- **Near-lossless on LongBench.** quad at a 5 percent per-step budget lands
  within noise of FullKV (LongBench-v1 delta about -0.22 for quad vs -0.38 for
  r8 at the same budget; about -0.29 at coverage 0.95 across 14 subsets).
- **Faster in the regime that matters.** AMASS-fast beats dense FlashAttention-3
  in the **long-context, high-batch** decode regime by up to about **2x**, and
  is at **parity at batch size 1** (the hot path is bandwidth-bound and only
  pulls ahead once the dense KV read dominates). It does not claim >=2x
  everywhere.
- **76 to 92 percent VRAM saved** in the mem variant, by holding only the quad
  summary plus the called pages resident and serving the rest of V from DRAM.

Numbers are per-kernel and end-to-end reproducible with `amass-bench` (below).
See the paper for the full protocol, benchmarks (RULER, LongBench v1/v2,
SCBench, a reasoning suite), and ablations.

- Paper: see the project page linked in this repository.

## Benchmarking: `amass-bench`

The wheel ships a benchmarker so the latency claims are reproducible on any
Hopper box with one command:

```bash
amass-bench kernels --quick    # per-kernel: r8_score / topb / decode, CUDA vs Triton vs dense-FA
amass-bench e2e --ctx 16384    # end-to-end decode-step TPOT in a real vLLM run
amass-bench all --json out.json
python -m amass.bench kernels  # identical to the console entry point
```

`amass-bench e2e` needs the `bench` + `vllm` extras (it tokenizes a real model
and spins up an engine). Timing is CUDA-event based with warmup and medians, and
reports both eager and CUDA-graph-replay speedups plus an HBM roofline for the
bandwidth-bound score kernel. Full flag reference: `amass-bench --help`.

## Dependencies

| Dependency     | Why                                              | How it is declared        |
| -------------- | ------------------------------------------------ | ------------------------- |
| `torch`        | tensors + runtime CUDA (`load_inline`) kernels   | hard (`>=2.4`)            |
| `triton`       | reference / fallback kernels                     | hard (`>=3.0`)            |
| `vllm`         | plugin host + attention backend                  | extra `[vllm]` (`>=0.8`)  |
| `transformers` | `amass-bench e2e` tokenizer (lazy import)         | extra `[bench]`           |
| `pyyaml`       | YAML `AMASS_CONFIG` files (JSON needs no dep)      | extra `[yaml]`            |

`numpy` is **not** a direct dependency: nothing under `amass/` imports it; it
arrives transitively through torch. vLLM is an extra rather than a hard floor
because it is a heavy, CUDA-ABI-sensitive wheel that production deployments pin
themselves, and a hard floor could silently upgrade a pinned engine.

## License

Apache-2.0 (provisional; see the note at the top). See `LICENSE` for the full
text.
