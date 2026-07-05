"""``python -m amass.bench`` -> the CLI (same as the ``amass-bench`` entry point)."""
from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
