"""No tracked function may read a name that nothing in reach ever binds.

A `NameError` in a rarely-taken branch is the cheapest bug in this codebase to write and among
the most expensive to find.  One shipped here: refactoring the deploy training pool into
`deploy_training_pool()` deleted the local `banned`, and a log line four lines below still read
it.  That branch only runs for a Stage-A training call with `verbose=True`, so it survived the
whole test suite and crashed the first cross-fit arm three hours into an overnight chain.

No linter is installed in this environment -- `ruff`, `flake8` and `pyflakes` are all absent --
so the check lives here rather than in CI config.

It is deliberately conservative.  A name is reported only when it is bound NOWHERE the function
could see it: not a parameter, not assigned anywhere in the function or a nested scope, not a
comprehension / `with` / `except` / `for` / match target, not a module-level binding or import,
not a builtin, and not declared `global` or `nonlocal`.  A name bound on ANY path counts as
bound, so "assigned only inside the if" is never flagged.  The point is zero false positives:
a guard people learn to ignore is worse than no guard.
"""

from __future__ import annotations

import ast
import builtins
import subprocess
from pathlib import Path

import pytest

from src.utils.paths import get_project_root

REPO = get_project_root()
BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__package__", "self", "cls"}


def _bound_in(node: ast.AST) -> set[str]:
    """Every name this subtree binds, at any depth and by any binding form."""
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out.update(n.names)
        elif isinstance(n, (ast.MatchAs, ast.MatchStar)) and getattr(n, "name", None):
            out.add(n.name)
    return out


def check(path: Path) -> list[str]:
    """Findings for one file; empty when it cannot be parsed, which is not this test's job."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []
    module_bound = _bound_in(tree) | BUILTINS
    bad: list[str] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        visible = module_bound | _bound_in(fn)
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in visible:
                bad.append(f"{path.name}:{n.lineno}: {fn.name}() reads undefined {n.id!r}")
    return bad


def _tracked_python() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO, capture_output=True, text=True)
    if out.returncode != 0:
        return []
    return [REPO / f for f in out.stdout.split()]


def test_no_tracked_function_reads_an_unbound_name():
    files = _tracked_python()
    if not files:
        pytest.skip("git not available")
    hits: list[str] = []
    for f in files:
        hits += check(f)
    assert not hits, (
        "these functions read names nothing binds, which is a NameError the moment the branch "
        "runs:\n  " + "\n  ".join(sorted(set(hits))))


def test_the_checker_catches_the_defect_it_exists_for(tmp_path):
    """A guard that cannot fail is worse than no guard, so prove this one fires."""
    bad = tmp_path / "bug.py"
    bad.write_text(
        "def load_packs(verbose=True):\n"
        "    stems = ['a']\n"
        "    if verbose:\n"
        "        print(sorted(banned & set(stems)))\n"
        "    return stems\n",
        encoding="utf-8")
    found = check(bad)
    assert len(found) == 1 and "banned" in found[0], found


def test_the_checker_does_not_flag_ordinary_binding_forms(tmp_path):
    """The false-positive control: every binding form the real code actually uses."""
    ok = tmp_path / "fine.py"
    ok.write_text(
        "import os\n"
        "def f(x):\n"
        "    total = 0\n"
        "    for i in range(x):\n"
        "        total += i\n"
        "    with open(os.devnull) as fh:\n"
        "        data = fh.read()\n"
        "    try:\n"
        "        pass\n"
        "    except ValueError as exc:\n"
        "        print(exc)\n"
        "    return [y for y in range(3)], total, data\n",
        encoding="utf-8")
    assert check(ok) == []
