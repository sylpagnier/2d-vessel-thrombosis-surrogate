"""Strip AI tool co-author / attribution trailers from commit messages."""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Match common AI attribution trailers (case-insensitive).
_PATTERNS = (
    re.compile(r"^Co-Authored-By:.*(?:claude|cursor|anthropic|cursoragent|openai|copilot).*$", re.I | re.M),
    re.compile(r"^Made with Cursor.*$", re.I | re.M),
)


def strip_ai_attribution(message: str) -> str:
    out = message
    for pattern in _PATTERNS:
        out = pattern.sub("", out)
    # Collapse runs of blank lines left after removal.
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.rstrip() + "\n"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: strip_ai_coauthor.py <commit-msg-file>", file=sys.stderr)
        return 1
    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8", errors="replace")
    path.write_text(strip_ai_attribution(text), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
