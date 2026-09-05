"""Every knob a Stage-A arm script exports must actually reach the code.

The 2026-09-04 configuration sweep replaced ~275 "undiscoverable" environment reads with
plain module constants, on the premise that nothing in the tree set them.  True of the tree;
false of `scripts/stage_a/run_*.sh`, which is where the whole E-series ladder lives.  Seventeen
knobs -- `KINEMATICS_BC_ENVELOPE`, `_DECAY`, `_DECODER_SKIP`, `_RESIDUAL_GAIN`, `_REZERO`, the
loss weights, the cohort switches -- were swept, so every arm script became a silent no-op for
them: `run_E5_band_gateup.sh` would have trained a plain-envelope model wearing E5's name and
written it to E5's output directory.

This test does not care HOW a knob is read, only that setting it changes what the code sees.
So it stays true if the constants are later replaced by a config object, and it fails the
moment an arm's setting stops arriving.
"""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from src.utils.paths import get_project_root

ARM_DIR = get_project_root() / "scripts" / "stage_a"

#: knob -> (module, attribute).  An arm knob that no module exposes under its own name is not
#: checkable this way; list it here as it gains a home rather than dropping the assertion.
HOMES: dict[str, tuple[str, str]] = {
    "KINEMATICS_BC_ENVELOPE": ("src.architecture.ginodeq", "KINEMATICS_BC_ENVELOPE"),
    "KINEMATICS_BC_ENVELOPE_DECAY": ("src.architecture.ginodeq", "KINEMATICS_BC_ENVELOPE_DECAY"),
    "KINEMATICS_BC_ENVELOPE_FLOOR": ("src.architecture.ginodeq", "KINEMATICS_BC_ENVELOPE_FLOOR"),
    "KINEMATICS_DECODER_SKIP": ("src.architecture.ginodeq", "KINEMATICS_DECODER_SKIP"),
    "KINEMATICS_RESIDUAL_GAIN": ("src.architecture.ginodeq", "KINEMATICS_RESIDUAL_GAIN"),
    "KINEMATICS_RESIDUAL_REZERO": ("src.architecture.ginodeq", "KINEMATICS_RESIDUAL_REZERO"),
    "KINEMATICS_BAND_ON_CORNERS": ("src.utils.kinematics_physics_terms", "KINEMATICS_BAND_ON_CORNERS"),
    "KINEMATICS_BAND_SHEAR_FLOOR": ("src.utils.kinematics_physics_terms", "KINEMATICS_BAND_SHEAR_FLOOR"),
    "KINEMATICS_BAND_FLOOR_WEIGHT": ("src.training.train_kinematics_predictor", "KINEMATICS_BAND_FLOOR_WEIGHT"),
    "KINEMATICS_DEPLOY_PACKS_ONLY": ("src.training.train_kinematics_predictor", "KINEMATICS_DEPLOY_PACKS_ONLY"),
    "KINEMATICS_DEPLOY_TRAIN_STEMS": ("src.utils.kinematics_select_packs", "KINEMATICS_DEPLOY_TRAIN_STEMS"),
    "KINEMATICS_ELEVATE_P2": ("src.training.train_kinematics_predictor", "KINEMATICS_ELEVATE_P2"),
    "KINEMATICS_GATE_WEIGHT": ("src.training.train_kinematics_predictor", "KINEMATICS_GATE_WEIGHT"),
    "KINEMATICS_MAX_NODES": ("src.training.train_kinematics_predictor", "KINEMATICS_MAX_NODES"),
    "KINEMATICS_PRIOR_FLOOR_WEIGHT": ("src.training.train_kinematics_predictor", "KINEMATICS_PRIOR_FLOOR_WEIGHT"),
    "KINEMATICS_SELECT_PATIENCE": ("src.training.train_kinematics_predictor", "KINEMATICS_SELECT_PATIENCE"),
    "KINEMATICS_WALL_SHEAR_WEIGHT": ("src.training.train_kinematics_predictor", "KINEMATICS_WALL_SHEAR_WEIGHT"),
    "KINEMATICS_SELECT_MAX_GRAPHS": ("src.utils.kinematics_select_packs", "KINEMATICS_SELECT_MAX_GRAPHS"),
    "KINEMATICS_VAL_EVERY": ("src.utils.kinematics_console", "KINEMATICS_VAL_EVERY"),
}

#: read where they are used rather than bound to a module constant, so the import-time probe
#: below cannot see them.  Each is exercised by the arm itself.
NOT_MODULE_CONSTANTS = {
    "KINEMATICS_BC_LAMBDA", "KINEMATICS_COORD_MODE", "KINEMATICS_LOSS_WEIGHTS",
    "KINEMATICS_NORMALIZE_SHEAR_GRAD", "KINEMATICS_OUTPUT_DIR", "KINEMATICS_PREPARED_CACHE",
    "KINEMATICS_BAND_DSRX_ABS", "SPECIES_PRIOR_SOURCE", "BIOCHEM_GRAD_CACHE_CPU",
}


def arm_knobs() -> set[str]:
    """Every ``KINEMATICS_*`` / ``SPECIES_*`` / ``BIOCHEM_*`` an arm script exports.

    ``scripts/stage_a/`` is gitignored (it is exploration, not published surface), so on a
    fresh clone there is nothing to parse.  Returning an empty set would make every test in
    this module pass vacuously, which is worse than not running them -- `pytest.skip` at the
    call sites says so out loud instead.
    """
    pat = re.compile(r"^export ((?:KINEMATICS|SPECIES|BIOCHEM)_[A-Z0-9_]+)", re.M)
    out: set[str] = set()
    for sh in sorted(ARM_DIR.glob("*.sh")):
        out |= set(pat.findall(sh.read_text(encoding="utf-8")))
    return out


def _knobs_or_skip() -> set[str]:
    knobs = arm_knobs()
    if not knobs:
        pytest.skip(f"no Stage-A arm scripts under {ARM_DIR} (gitignored; nothing to check)")
    return knobs


def test_every_arm_knob_has_a_home_or_is_declared_inline():
    unknown = _knobs_or_skip() - set(HOMES) - NOT_MODULE_CONSTANTS
    assert not unknown, (
        "Stage-A arm scripts export knobs this test cannot check: %s.  Give each a "
        "(module, attribute) entry in HOMES, or add it to NOT_MODULE_CONSTANTS with a note "
        "saying where it is read." % sorted(unknown))


@pytest.mark.parametrize("knob", sorted(set(HOMES) & arm_knobs()))
def test_arm_knob_reaches_the_code(knob):
    """Setting the knob in the environment must change the value the module resolves.

    Run in a subprocess: these are import-time constants, so a value set after this test
    session imported the module would not be observable in-process.
    """
    _knobs_or_skip()
    mod, attr = HOMES[knob]

    def resolve(value: str) -> str:
        probe = (
            "import os, sys, importlib;"
            "os.environ[%r] = %r;"
            "m = importlib.import_module(%r);"
            "sys.stdout.write(str(getattr(m, %r)))" % (knob, value, mod, attr)
        )
        env = {**os.environ, "PYTHONPATH": str(get_project_root())}
        got = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                             env=env, cwd=str(get_project_root()))
        assert got.returncode == 0, got.stderr[-2000:]
        return got.stdout

    # Two settings rather than one sentinel: a boolean knob coerces any unrecognised string to
    # its default, so a single probe cannot distinguish "ignored" from "read and rejected".
    on, off = resolve("1"), resolve("0")
    assert on != off, (
        f"{knob} is exported by a Stage-A arm script but {mod}.{attr} resolves to {off!r} "
        f"whether it is set to '1' or '0' -- the arm is a silent no-op for it.")

@pytest.mark.parametrize("knob", sorted(set(HOMES) & arm_knobs()))
def test_arm_knob_is_actually_consumed(knob):
    """The constant must be READ somewhere, not merely defined.

    The sweep did two things, and restoring the first does not undo the second: it froze the
    constants, and at some call sites it inlined the RESOLVED value
    (`deploy_only = False`, where an `os.environ.get(...) in ("1", ...)` had been).  A knob can
    therefore round-trip through its module perfectly -- which is all
    `test_arm_knob_reaches_the_code` checks -- while nothing consumes it, which is how
    `KINEMATICS_DEPLOY_PACKS_ONLY=1` still loaded the whole synthetic corpus and died resolving
    a `vessel_0` mesh against a deploy pack.
    """
    _knobs_or_skip()
    mod, attr = HOMES[knob]
    path = get_project_root() / (mod.replace(".", "/") + ".py")
    body = path.read_text(encoding="utf-8")
    uses = [ln for ln in body.splitlines()
            if re.search(rf"\b{re.escape(attr)}\b", ln)
            and not re.match(rf"\s*{re.escape(attr)}\s*=", ln)]
    assert uses, (
        f"{mod}.{attr} is defined but never read there.  Either a call site inlined its "
        f"resolved value (search the module for the behaviour it should gate) or the knob is "
        f"genuinely dead and should be deleted from the arm scripts and the allowlist.")
