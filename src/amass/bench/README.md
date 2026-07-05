# `amass.bench` — the AMASS internal benchmarker

A reusable, packaged benchmarker for the AMASS KV-cache sparse-attention hot path.
It ships inside the `amass` pip package (module **and** `amass-bench` CLI) so the
release's latency claims can be reproduced by anyone, on any Hopper box, with one
command — not from a scratch script.

It measures the two things the release claim rests on, with defensible timing:

| what | how |
|---|---|
| **Per-kernel latency** | each hot kernel (`r8_score` / `topb` / `decode`) hand-CUDA vs its golden Triton reference; `decode` also vs a dense-FlashAttention baseline. CUDA-event timed, warmup + medians, **eager and CUDA-graph-replay**, with the HBM byte-roofline for the bandwidth-bound `r8_score`. A CUDA-vs-Triton numerical agreement check rides along so a fast-but-wrong result is caught. |
| **End-to-end TPOT** | decode-step latency + throughput in a **real vLLM run** (FullKV vs AMASS-fast CUDA vs AMASS-fast Triton) via the two-length-diff method `TPOT=(t(N2)-t(N1))/(N2-N1)`, under full cudagraph by default. |

## Layout

```
amass/bench/
  __init__.py   public API: run_kernels, run_e2e, bench_{r8_score,topb,decode}, timing helpers
  timing.py     CUDA-event timing (warmup + medians), CUDA-graph replay timing, HBM roofline
  kernels.py    standalone per-kernel benchmarks (build R8State + synthetic inputs)
  e2e.py        real-vLLM two-length-diff TPOT (one subprocess per config)
  cli.py        argparse CLI: subcommands kernels / e2e / all
  __main__.py   `python -m amass.bench`
  README.md / PACKAGING.md
```

## Public API

```python
from amass.bench import run_kernels, run_e2e

# per-kernel sweep -> list of result dicts
res = run_kernels(which=["r8_score", "decode"], batches=[4], contexts=[16384],
                  budgets=[0.1], n_kv=8, G=8, graph=True)

# end-to-end TPOT (spawns one vLLM subprocess per config) -> list of dicts
e2e = run_e2e(configs=["fullkv", "fast_cuda"], ctx=16384, nreq=1, budget=0.10)
```

Single-shot helpers are also public: `bench_r8_score(bs, ctx, ...)`,
`bench_topb(bs, ctx, budget=..., ...)`, `bench_decode(bs, ctx, budget=..., ...)`,
and the timing primitives `bench_median`, `bench_graph`, `hbm_roofline_us`.

## CLI

```bash
amass-bench kernels                 # full sweep: batch{1,2,4,8} x ctx{4K,16K,32K} x budget{.02,.05,.1,1.0}
amass-bench kernels --quick         # fast subset: bs{1,4} / ctx 16K / budget .1
amass-bench kernels --which r8_score --batches 4 --contexts 16384
amass-bench e2e --ctx 16384         # FullKV vs AMASS-fast CUDA vs Triton
amass-bench all --json out.json     # kernels then e2e, dump structured results
python -m amass.bench kernels       # identical to the console entry point
```

Key flags: `--which` (kernels subset), `--batches/--contexts/--budgets` (sweep
axes), `--n-kv/--group` (GQA shape; **G=8** is the spec target, **G=4** is real
Llama-3.1-8B), `--no-graph`, `--no-dense`, `--configs`/`--ctx`/`--nreq`/`--budget`
(e2e), `--eager` (piecewise instead of full cudagraph), `--json PATH`.

## Running on the cluster (finicky container)

Everything runs inside the NGC container via `scripts/container_exec.sh` with the
canonical `/tmp/kvcomp_vllm24` stack env (see `scratch_dram_repro/run_eq.sh`).
Use **`CUDA_VISIBLE_DEVICES=0`** on `h200x4-04` (GPU 0). The hand-CUDA kernels
build once (~2–3 min each) via `torch.utils.cpp_extension` at first use, cached
under `TORCH_EXTENSIONS_DIR` (node-local `HOME=/tmp/kvcomp_home`).

```bash
CUDA_VISIBLE_DEVICES=0 scripts/container_exec.sh bash -lc '
  export HOME=/tmp/kvcomp_home
  export LD_LIBRARY_PATH=$(ls -d /tmp/kvcomp_vllm24/nvidia/*/lib | tr "\n" ":")$LD_LIBRARY_PATH
  export XDG_CACHE_HOME=/tmp/kvcomp_cache24 TRITON_CACHE_DIR=/tmp/kvcomp_cache24/triton
  export TRITON_LIBCUDA_PATH=/.singularity.d/libs HF_HOME=$KVCOMP_ROOT/hf_cache
  export TORCH_EXTENSIONS_DIR=/tmp/kvcomp_home/torch_ext KVCOMP_VLLM_SITE=/tmp/kvcomp_vllm24
  PYTHONPATH=/tmp/kvcomp_vllm24:$KVCOMP_ROOT/src python -m amass.bench kernels --quick'
```

## Reliability notes

- **Timing is CUDA-event based, never wall clock** (except the e2e two-length
  diff, which is designed to cancel fixed overhead). Every number is the median
  of repeated batches after an explicit warmup; the fastest batch and spread are
  also recorded.
- **Graph vs eager is always labeled.** The kernel table reports both an eager
  speedup and a CUDA-graph-replay speedup; the e2e run marks `[graph]`/`[eager]`.
- **Roofline context** is attached to `r8_score` (bandwidth-bound): the table
  shows the HBM byte-roofline µs, achieved % of peak BW, and the ×-roofline gap.
- **Self-check:** `r8_score` at `bs=4 / ctx=16384 / G=8` reproduces the documented
  ~7× (≈60 µs CUDA vs ≈410 µs Triton, |Δ| ≈ 1e-4 vs the Triton reference).
- **e2e isolation:** each config runs in a fresh subprocess because the plugin's
  `register()` patches the CUDA platform once per process; three variants cannot
  share one interpreter.
