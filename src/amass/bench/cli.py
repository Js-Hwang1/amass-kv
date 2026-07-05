"""``amass-bench`` command-line interface.

Subcommands:
  * ``kernels`` -- per-kernel latency sweep (r8_score / topb / decode, CUDA vs
    Triton vs dense-FA). ``--quick`` restricts to a fast headline subset.
  * ``e2e``     -- end-to-end decode-step TPOT + throughput in a real vLLM run
    (FullKV vs AMASS-fast CUDA vs AMASS-fast Triton).
  * ``all``     -- kernels then e2e.

Both ``amass-bench <sub>`` (console entry point) and ``python -m amass.bench
<sub>`` work. ``--json PATH`` dumps the structured results next to the table.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional, Sequence


def _int_list(s: str) -> List[int]:
    return [int(x) for x in s.replace(" ", "").split(",") if x]


def _float_list(s: str) -> List[float]:
    return [float(x) for x in s.replace(" ", "").split(",") if x]


def _add_kernel_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--which", default="r8_score,topb,decode",
                   help="kernels to bench (comma list)")
    p.add_argument("--batches", type=_int_list, default=None,
                   help="batch sizes (default 1,2,4,8; quick: 1,4)")
    p.add_argument("--contexts", type=_int_list, default=None,
                   help="context lengths (default 4096,16384,32768; quick: 16384)")
    p.add_argument("--budgets", type=_float_list, default=None,
                   help="page budgets (default 0.02,0.05,0.1,1.0; quick: 0.1)")
    p.add_argument("--n-kv", type=int, default=8, help="KV heads (Llama=8)")
    p.add_argument("--group", type=int, default=8,
                   help="GQA group size G (spec target=8; Llama real=4)")
    p.add_argument("--no-graph", action="store_true",
                   help="skip CUDA-graph replay timing")
    p.add_argument("--no-dense", action="store_true",
                   help="skip the dense-FlashAttention decode baseline")
    p.add_argument("--quick", action="store_true",
                   help="fast subset (bs 1,4 / ctx 16384 / budget 0.1)")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--reps", type=int, default=7)


def _add_e2e_args(p: argparse.ArgumentParser, reps_name: str = "--reps") -> None:
    # ``reps_name`` is "--e2e-reps" under the ``all`` subcommand (where kernels
    # already own "--reps"); both map to dest ``e2e_reps``.
    p.add_argument("--configs", default="fullkv,fast_cuda,fast_triton",
                   help="e2e configs (comma list)")
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--nreq", type=int, default=1)
    p.add_argument("--budget", type=float, default=0.10)
    p.add_argument("--sink", type=int, default=1)
    p.add_argument("--window", type=int, default=1)
    p.add_argument("--eager", action="store_true",
                   help="enforce_eager (piecewise) instead of full cudagraph")
    p.add_argument("--n1", type=int, default=32)
    p.add_argument("--n2", type=int, default=160)
    p.add_argument(reps_name, dest="e2e_reps", type=int, default=3,
                   help="two-length TPOT reps")
    p.add_argument("--hf-id", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--gpu-util", type=float, default=0.55)
    p.add_argument("--gpu-blocks", type=int, default=12000)


def _run_kernels(args) -> List[Dict]:
    from . import kernels
    if args.quick:
        batches = args.batches or [1, 4]
        contexts = args.contexts or [16384]
        budgets = args.budgets or [0.1]
    else:
        batches = args.batches or list(kernels._DEFAULT_BATCHES)
        contexts = args.contexts or list(kernels._DEFAULT_CONTEXTS)
        budgets = args.budgets or list(kernels._DEFAULT_BUDGETS)
    print(f"[amass-bench] kernels: which={args.which} batches={batches} "
          f"contexts={contexts} budgets={budgets} n_kv={args.n_kv} G={args.group}"
          f" graph={not args.no_graph} dense={not args.no_dense}", flush=True)
    return kernels.run_kernels(
        which=args.which.split(","), batches=batches, contexts=contexts,
        budgets=budgets, n_kv=args.n_kv, G=args.group, graph=not args.no_graph,
        dense=not args.no_dense, warmup=args.warmup, iters=args.iters,
        reps=args.reps)


def _run_e2e(args) -> List[Dict]:
    from . import e2e
    return e2e.run_e2e(
        configs=args.configs.split(","), ctx=args.ctx, nreq=args.nreq,
        budget=args.budget, sink=args.sink, window=args.window, eager=args.eager,
        n1=args.n1, n2=args.n2, reps=args.e2e_reps, hf_id=args.hf_id,
        gpu_util=args.gpu_util, gpu_blocks=args.gpu_blocks)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # internal dispatch: the e2e parent spawns `python -m amass.bench
    # __e2e_worker__ <json>` -> run one config in this fresh process.
    if argv and argv[0] == "__e2e_worker__":
        from . import e2e
        e2e._worker_main(argv[1:])
        return 0

    ap = argparse.ArgumentParser(
        prog="amass-bench",
        description="AMASS internal benchmarker (kernels + end-to-end TPOT).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pk = sub.add_parser("kernels", help="per-kernel latency sweep")
    _add_kernel_args(pk)
    pe = sub.add_parser("e2e", help="end-to-end decode-step TPOT")
    _add_e2e_args(pe)
    pa = sub.add_parser("all", help="kernels then e2e")
    _add_kernel_args(pa)
    _add_e2e_args(pa, reps_name="--e2e-reps")
    # --json goes AFTER the subcommand: `amass-bench e2e --json out.json`
    for p in (pk, pe, pa):
        p.add_argument("--json", default=None,
                       help="write structured results to this JSON path")

    args = ap.parse_args(argv)
    out: Dict[str, List[Dict]] = {}
    if args.cmd in ("kernels", "all"):
        out["kernels"] = _run_kernels(args)
    if args.cmd in ("e2e", "all"):
        if args.cmd == "all":
            # free any kernel-bench allocations before the engine spins up
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        out["e2e"] = _run_e2e(args)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[amass-bench] wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
