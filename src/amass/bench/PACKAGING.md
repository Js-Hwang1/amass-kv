# Packaging note — wiring `amass-bench`

This benchmarker ships **inside** the `amass` package (`amass.bench`). It exposes
a console entry point. When the `amass` package's own `pyproject.toml` is
finalized (a later step — do **not** touch the repo-root `pyproject.toml`, which
belongs to the separate `kvcomp` harness), add the `console_scripts` entry point:

```toml
[project.scripts]
amass-bench = "amass.bench.cli:main"
```

That is the single line the finalized `amass` `pyproject.toml` needs so that
`pip install amass` puts an `amass-bench` executable on `PATH`. It calls
`amass.bench.cli.main`, which also backs `python -m amass.bench` (via
`amass/bench/__main__.py`), so both invocation styles stay in lock-step.

## Full context for the `amass` pyproject

The bench module is pure-Python and adds **no** new runtime dependencies beyond
what the rest of `amass` already needs (`torch`, `triton`; `transformers` +
`vllm` are only imported by the `e2e` subcommand, lazily). If the `amass`
pyproject uses hatchling like the repo root, the wheel already includes
`amass.bench` as long as the package glob covers `amass` (e.g.
`packages = ["src/amass"]`). No separate data files.

For reference, the vLLM plugin entry point already lives in the same package and
is registered the same way (documented in `ours_doc/AMASS_DESIGN.md` §9):

```toml
[project.entry-points."vllm.general_plugins"]
amass = "amass.backend.register:register"
```

## Summary — lines the `amass` pyproject should contain

```toml
[project.scripts]
amass-bench = "amass.bench.cli:main"

[project.entry-points."vllm.general_plugins"]
amass = "amass.backend.register:register"
```
