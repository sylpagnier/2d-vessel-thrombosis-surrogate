"""Thin shim for ``python -m src.tools.diagnostics``."""

from __future__ import annotations


from src.tools.diagnostics.__main__ import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
