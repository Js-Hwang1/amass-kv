"""End-to-end decode-step latency (TPOT) + throughput in a REAL vLLM run.

Measures the production decode-step cost of AMASS-fast (Stage A + Stage B inside
the engine) against the stock-FlashAttention FullKV line, for three configs:

  * ``fullkv``      -- ``AMASS_DISABLE=1`` -> stock FlashAttention (reference).
  * ``fast_cuda``   -- AMASS-fast, hand-CUDA hot kernels (``use_cuda=True``).
  * ``fast_triton`` -- AMASS-fast, Triton hot kernels (``use_cuda=False``).

TPOT method: the **two-length difference** ``TPOT = (t(N2) - t(N1))/(N2 - N1)``
(median over reps), which cancels prefill + fixed per-request overhead and leaves
the pure per-decode-step latency (same method as ``scripts/lat_probe.py``). Under
FULL cudagraph by default (``eager=False``); pass ``eager=True`` for the piecewise
comparison. Throughput is reported as decode tokens/s (aggregate over the batch).

Isolation: each config runs in a FRESH subprocess. The plugin's ``register()``
patches the CUDA platform ONCE per process (module-global latch) and reads
``AMASS_CONFIG`` / ``AMASS_DISABLE`` at engine init, so three variants cannot share
one process -- the worker below runs exactly one config and the parent aggregates.
The subprocess inherits the container env (PYTHONPATH etc.) from the parent.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_MARK = "__E2E_JSON__ "


# --------------------------------------------------------------------------- #
# Plugin install (vLLM general-plugin entry point discovery)                  #
# --------------------------------------------------------------------------- #
def _install_amass_plugin() -> None:
    """Write the ``amass`` dist-info + entry point into the vLLM site so vLLM's
    plugin loader discovers ``amass.backend.register:register`` at engine init
    (mirrors ``scratch_r8g/run_r8.install_amass_plugin``)."""
    site = Path(os.environ.get("KVCOMP_VLLM_SITE", "/tmp/kvcomp_vllm24"))
    shutil.rmtree(site / "dynkv_plugin-0.1.dist-info", ignore_errors=True)
    di = site / "amass-0.1.dist-info"
    di.mkdir(parents=True, exist_ok=True)
    (di / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: amass\nVersion: 0.1\n")
    (di / "entry_points.txt").write_text(
        "[vllm.general_plugins]\namass = amass.backend.register:register\n")
    (di / "RECORD").write_text("")
    stub = site / "flash_attn"
    stub.mkdir(exist_ok=True)
    (stub / "__init__.py").write_text(
        'raise ModuleNotFoundError("flash_attn stubbed out (amass)")\n')


# --------------------------------------------------------------------------- #
# In-subprocess single-config worker                                          #
# --------------------------------------------------------------------------- #
def _worker(cfg: Dict) -> Dict:
    """Run ONE config in this process and return the measurement dict.

    Called only inside a freshly-spawned worker (see :func:`run_e2e`)."""
    import torch
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    mode = cfg["mode"]
    _install_amass_plugin()
    if mode == "fullkv":
        os.environ["AMASS_DISABLE"] = "1"
        os.environ.pop("AMASS_CONFIG", None)
    else:
        os.environ.pop("AMASS_DISABLE", None)
        os.environ["AMASS_CONFIG"] = json.dumps({
            "variant": "fast", "budget": cfg["budget"],
            "sink_pages": cfg["sink"], "window_pages": cfg["window"],
            "use_cuda": bool(cfg["use_cuda"]),
            "force_eager": bool(cfg["eager"])})

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    hf_id = cfg["hf_id"]
    ctx = cfg["ctx"]
    tok = AutoTokenizer.from_pretrained(hf_id)

    # synthetic prompts sized to ~ctx tokens (deterministic, no dataset needed).
    # DISTINCT per request (unique header) + prefix caching OFF so every request
    # owns its physical KV blocks -- otherwise identical prompts share prefix
    # blocks and the measured KV-read volume is 1/nreq of the honest workload.
    unit = ("The archivists of the floating city catalogued every brass device "
            "with meticulous, unhurried care. ")
    ntok_unit = max(1, len(tok.encode(unit, add_special_tokens=False)))
    prompt = unit * max(1, (ctx // ntok_unit))
    from vllm.inputs import TokensPrompt
    prompts = []
    for i in range(cfg["nreq"]):
        hdr = f"Archive shelf {i} of {cfg['nreq']} begins here. "
        ids = tok.encode(hdr + prompt, add_special_tokens=False)[: ctx - 256]
        prompts.append(TokensPrompt(prompt_token_ids=ids))
    prompt_ids = prompts[0]["prompt_token_ids"]

    llm = LLM(model=hf_id, gpu_memory_utilization=cfg["gpu_util"],
              max_model_len=min(ctx + 512, 131072), enforce_eager=cfg["eager"],
              max_num_seqs=cfg["nreq"], enable_prefix_caching=False,
              num_gpu_blocks_override=(cfg["gpu_blocks"]
                                       if mode != "fullkv" else None))

    def timed_gen(n_tok: int) -> float:
        sp = SamplingParams(temperature=0, max_tokens=n_tok, ignore_eos=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    timed_gen(8)  # warmup: graph capture + JIT build at both shapes
    timed_gen(cfg["n2"])
    if os.environ.get("AMASS_CUDAPROF", "0") == "1":
        # nsys attribution mode (scratch_deepopt3): profile exactly one short
        # decode-heavy generate between cudaProfilerApi start/stop, then exit.
        torch.cuda.profiler.start()
        t = timed_gen(64)
        torch.cuda.profiler.stop()
        return {"mode": mode, "ctx": ctx, "nreq": cfg["nreq"], "prof_s": t,
                "tpot_ms": float("nan"), "tok_per_s": 0.0, "eager": cfg["eager"],
                "graph": not cfg["eager"], "budget": cfg["budget"],
                "use_cuda": cfg["use_cuda"], "n_prompt_tok": len(prompt_ids)}
    N1, N2, reps = cfg["n1"], cfg["n2"], cfg["reps"]
    diffs = []
    for _ in range(reps):
        t1 = timed_gen(N1)
        t2 = timed_gen(N2)
        diffs.append((t2 - t1) / (N2 - N1))
    diffs.sort()
    tpot_s = diffs[len(diffs) // 2]
    tpot_ms = tpot_s * 1000.0
    agg_tok_s = cfg["nreq"] / tpot_s          # batch decode tokens / s
    return {
        "mode": mode, "ctx": ctx, "n_prompt_tok": len(prompt_ids),
        "nreq": cfg["nreq"], "budget": cfg["budget"], "use_cuda": cfg["use_cuda"],
        "eager": cfg["eager"], "graph": not cfg["eager"],
        "tpot_ms": tpot_ms, "tok_per_s": agg_tok_s,
        "tok_per_s_per_req": 1.0 / tpot_s,
        "reps_ms": [round(d * 1000, 4) for d in diffs],
    }


# --------------------------------------------------------------------------- #
# Parent-side driver (spawns one subprocess per config)                       #
# --------------------------------------------------------------------------- #
def _config(mode: str, **base) -> Dict:
    use_cuda = mode == "fast_cuda"
    engine_mode = "fullkv" if mode == "fullkv" else "fast"
    return {"mode": engine_mode, "label": mode, "use_cuda": use_cuda, **base}


def run_e2e(*, configs: Sequence[str] = ("fullkv", "fast_cuda", "fast_triton"),
            ctx: int = 16384, nreq: int = 1, budget: float = 0.10,
            sink: int = 1, window: int = 1, eager: bool = False,
            n1: int = 32, n2: int = 160, reps: int = 3,
            hf_id: str = "meta-llama/Llama-3.1-8B-Instruct",
            gpu_util: float = 0.55, gpu_blocks: int = 12000,
            verbose: bool = True) -> List[Dict]:
    """Run the end-to-end TPOT comparison; return a list of measurement dicts.

    Each config is measured in its own subprocess (see module docstring). The
    parent parses the ``__E2E_JSON__`` line each worker prints. Public API:
    ``amass.bench.run_e2e``.
    """
    base = dict(ctx=ctx, nreq=nreq, budget=budget, sink=sink, window=window,
                eager=eager, n1=n1, n2=n2, reps=reps, hf_id=hf_id,
                gpu_util=gpu_util, gpu_blocks=gpu_blocks)
    results: List[Dict] = []
    for name in configs:
        cfg = _config(name, **base)
        if verbose:
            print(f"\n[e2e] launching worker: {name} "
                  f"(ctx={ctx} nreq={nreq} budget={budget} "
                  f"{'eager' if eager else 'cudagraph'})", flush=True)
        payload = json.dumps({**cfg, "use_cuda": cfg["use_cuda"]})
        proc = subprocess.run(
            [sys.executable, "-m", "amass.bench", "__e2e_worker__", payload],
            env=os.environ.copy(), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)
        out = proc.stdout or ""
        parsed = None
        for line in out.splitlines():
            if line.startswith(_MARK):
                parsed = json.loads(line[len(_MARK):])
                break
        if parsed is None:
            tail = "\n".join(out.splitlines()[-25:])
            print(f"[e2e] worker {name} FAILED (rc={proc.returncode}); "
                  f"last output:\n{tail}", flush=True)
            results.append({"mode": name, "error": True, "rc": proc.returncode})
            continue
        parsed["label"] = name
        results.append(parsed)
        if verbose:
            print(f"[e2e] {name}: TPOT {parsed['tpot_ms']:.3f} ms  "
                  f"{parsed['tok_per_s']:.1f} tok/s "
                  f"({'graph' if parsed['graph'] else 'eager'})", flush=True)

    _print_summary(results)
    return results


def _print_summary(results: List[Dict]) -> None:
    ok = [r for r in results if not r.get("error")]
    if not ok:
        return
    base = next((r for r in ok if r["label"] == "fullkv"), None)
    print("\n=== e2e TPOT (decode-step latency) ===", flush=True)
    for r in ok:
        rel = ""
        if base and r is not base and base["tpot_ms"]:
            rel = f"  ({base['tpot_ms'] / r['tpot_ms']:.2f}x vs FullKV)"
        print(f"  {r['label']:12s} ctx={r['ctx']:>5d} nreq={r['nreq']}  "
              f"TPOT {r['tpot_ms']:7.3f} ms  {r['tok_per_s']:7.1f} tok/s  "
              f"[{'graph' if r['graph'] else 'eager'}]{rel}", flush=True)


def _worker_main(argv: Sequence[str]) -> None:
    """Entry for the ``__e2e_worker__`` subprocess dispatch (see cli / __main__)."""
    cfg = json.loads(argv[0])
    res = _worker(cfg)
    print(_MARK + json.dumps(res), flush=True)
