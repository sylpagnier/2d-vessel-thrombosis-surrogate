"""Thin shim for ``python -m src.tools.diagnostics``."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.tools.diagnostics.__main__ import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
