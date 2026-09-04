"""Freeze the environment-knob surface so configuration sprawl cannot grow.

This suite does not care that 471 knobs exist -- that is history.  It cares that
the number stops going up, and that new configuration is added as a typed field
instead of a fresh ``os.environ.get`` at a call site.

If one of these fails because you added a knob: add a field to the relevant
typed config (``BiochemRuntimeConfig`` / ``PushforwardConfig`` / ``PhysicsConfig``
/ ``VesselConfig``) and bind it in ``runtime_config.RUNTIME_ENV_TO_FIELD``.  If it
fails because you *removed* one, delete it from ``_env_allowlist.json`` too --
that direction is always welcome.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from src.utils.env_registry import known_env, typed_env, untyped_env
from src.utils.paths import get_project_root

READ = re.compile(
    r"""os\.(?:environ\.get|getenv)\(\s*["']([A-Z0-9_]+)["']"""
    r"""|os\.environ\[\s*["']([A-Z0-9_]+)["']\s*\]"""
)
WRITE = re.compile(
    r"""os\.environ\[\s*["']([A-Z0-9_]+)["']\s*\]\s*="""
    r"""|os\.environ\.setdefault\(\s*["']([A-Z0-9_]+)["']"""
)


def _tracked_python() -> list[Path]:
    root = get_project_root()
    out = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=root, capture_output=True, text=True
    )
    if out.returncode != 0:
        return []
    return [root / p for p in out.stdout.split() if (root / p).is_file()]


def _env_names_in_tree() -> set[str]:
    names: set[str] = set()
    for path in _tracked_python():
        if path.name == "test_env_registry.py":
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (UnicodeDecodeError, OSError):
            continue
        for pattern in (READ, WRITE):
            for m in pattern.finditer(text):
                names.add(m.group(1) or m.group(2))
    return names


def test_no_new_environment_knobs():
    found = _env_names_in_tree()
    if not found:  # no git available; nothing to assert against
        return
    added = sorted(found - known_env())
    assert not added, (
        "New ad-hoc environment knobs were introduced:\n  "
        + "\n  ".join(added)
        + "\n\nAdd a typed config field instead (see src/utils/env_registry.py). "
        "If this really must be an environment variable, bind it in "
        "runtime_config.RUNTIME_ENV_TO_FIELD and add it to _env_allowlist.json."
    )


def test_allowlist_has_no_stale_entries():
    """The inventory should track reality, so removals get recorded too."""
    found = _env_names_in_tree()
    if not found:
        return
    stale = sorted(known_env() - found)
    assert not stale, (
        f"{len(stale)} knob(s) are in the allowlist but no longer read anywhere. "
        "Delete them from src/utils/_env_allowlist.json:\n  " + "\n  ".join(stale)
    )


def test_typed_knobs_are_actually_bound():
    from src.architecture.runtime_config import RUNTIME_ENV_TO_FIELD

    unbound = sorted(typed_env() - set(RUNTIME_ENV_TO_FIELD))
    assert not unbound, f"listed as typed but not bound in RUNTIME_ENV_TO_FIELD: {unbound}"


def test_untyped_surface_does_not_grow():
    """The ad-hoc surface is a debt figure; it may shrink, never grow."""
    ceiling = json.loads(
        (Path(__file__).resolve().parents[1] / "utils" / "_env_allowlist.json").read_text(
            encoding="utf-8"
        )
    )["total"]
    assert len(known_env()) <= ceiling, (
        f"environment surface grew to {len(known_env())} (ceiling {ceiling})"
    )
    # Sanity: the two partitions cover the whole inventory.
    assert untyped_env() | typed_env() == known_env()
