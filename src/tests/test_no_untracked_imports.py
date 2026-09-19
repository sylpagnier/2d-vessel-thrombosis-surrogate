"""Committed code must not import modules that are not committed.

`scripts/stage_a/` and `src/tools/diagnostics/` are deliberately gitignored -- they are
exploration, and `008e1a2` ("Publish only what a researcher can use; keep our exploration
local") is the decision that put them there.  The hazard is that they still IMPORT cleanly on
the machine that wrote them, so a committed script can grow a dependency on one and nothing
notices until someone clones the repo.

That happened three times: `eval_flow_source_paired.py`, `eval_domain_routed.py` and
`calibrate_residual_scale.py` all resolved "has the flow model already seen this vessel?"
through `scripts/stage_a/crossfit_halves.py`.  On a fresh clone each is an ImportError in the
middle of an evaluation -- not at import time, because the import sits inside the branch that
needs it, so it survives every smoke test and fails only when someone asks for the holdout
panel.  The fix was to name the definition in committed code
(`kinematics_select_packs.deploy_training_pool`); this test is what stops it coming back.

Imports are found textually rather than by executing anything, so a module that is expensive or
GPU-bound to import is still checked.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

from src.utils.paths import get_project_root

REPO = get_project_root()

#: `from src.x import y` / `import src.x.y` -- only first-party roots can be untracked.
IMPORT = re.compile(r"^\s*(?:from|import)\s+((?:src|scripts)[\w\.]*)", re.M)


def _tracked_python() -> set[str]:
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO, capture_output=True, text=True)
    if out.returncode != 0:
        return set()
    return {p.replace("\\", "/") for p in out.stdout.split()}


def _module_file(dotted: str) -> pathlib.Path | None:
    """Deepest prefix of ``dotted`` that exists as a .py file, or None."""
    parts = dotted.split(".")
    for n in range(len(parts), 1, -1):
        cand = REPO.joinpath(*parts[:n]).with_suffix(".py")
        if cand.is_file():
            return cand
    return None


def test_no_committed_file_imports_an_untracked_module():
    tracked = _tracked_python()
    if not tracked:
        pytest.skip("git not available, or nothing tracked")

    offences: list[str] = []
    for rel in sorted(tracked):
        path = REPO / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for match in IMPORT.finditer(text):
            target = _module_file(match.group(1))
            if target is None:
                continue                       # a package, a third-party name, or absent
            target_rel = target.relative_to(REPO).as_posix()
            if target_rel not in tracked:
                offences.append(f"{rel} imports {match.group(1)} ({target_rel}, untracked)")

    assert not offences, (
        "committed code depends on modules that are not committed, so a fresh clone breaks:\n  "
        + "\n  ".join(sorted(set(offences)))
        + "\n\nMove the definition into tracked code (see "
          "`kinematics_select_packs.deploy_training_pool` for the pattern) rather than "
          "un-ignoring the exploration directory.")
