"""Self-contained stdin filter for git filter-branch (no relative imports)."""
from __future__ import annotations

import re
import sys

_PATTERNS = (
    re.compile(r"^Co-Authored-By:.*(?:claude|cursor|anthropic|cursoragent|openai|copilot).*$", re.I | re.M),
    re.compile(r"^Made with Cursor.*$", re.I | re.M),
)

msg = sys.stdin.read()
for pattern in _PATTERNS:
    msg = pattern.sub("", msg)
msg = re.sub(r"\n{3,}", "\n\n", msg).rstrip() + "\n"
sys.stdout.write(msg)
