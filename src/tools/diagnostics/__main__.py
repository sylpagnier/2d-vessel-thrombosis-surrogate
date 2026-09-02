"""Unified diagnostic CLI.

Usage:
  python -m src.tools.diagnostics list
  python -m src.tools.diagnostics clot-free-headroom --cache smoke --tag smoke
"""

from __future__ import annotations

import sys

from src.tools.diagnostics.registry import DIAGNOSTICS, resolve_main


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip(), flush=True)
        print("\nSlugs:", ", ".join(sorted(DIAGNOSTICS)), flush=True)
        return 0 if not argv else 2
    if argv[0] == "list":
        for slug, mod in sorted(DIAGNOSTICS.items()):
            print(f"  {slug:24s}  {mod}", flush=True)
        return 0
    slug = argv[0]
    try:
        run = resolve_main(slug)
    except (KeyError, AttributeError) as exc:
        print(f"[ERR] {exc}", flush=True)
        return 2
    return int(run(argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
