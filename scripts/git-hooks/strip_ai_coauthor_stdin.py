"""stdin/stdout filter for git filter-branch --msg-filter."""
from __future__ import annotations

import sys

from strip_ai_coauthor import strip_ai_attribution

if __name__ == "__main__":
    sys.stdout.write(strip_ai_attribution(sys.stdin.read()))
