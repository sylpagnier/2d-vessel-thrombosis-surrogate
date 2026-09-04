"""Pins for the single source of artifact identity (:mod:`src.clot_ml.artifacts`).

What this protects: identity used to live in ~20 places and they drifted.  The shipped stack
was `DeployClot2_0` while `eval_clot_ml_0.py --baseline` still defaulted to `clot_gnn_v5w`,
and the customer UI was served the legacy stub for a whole sweep campaign.  A role resolved
from the pointer cannot drift, because there is one of it.
"""
from __future__ import annotations

import json

import pytest

from src.clot_ml import artifacts as A


def test_roles_are_derived_from_one_pointer_and_agree():
    """The three roles must come from a single manifest chain, so they cannot disagree."""
    if A.pointer_name() is None:
        pytest.skip("no promoted artifact in this checkout")
    uni, wound, base = A.shipped(A.UNIFIED), A.shipped(A.WOUND), A.shipped(A.BASE)
    assert uni and wound and base
    assert len({uni, wound, base}) == 3, "the chain collapsed onto one artifact"

    # each link is the *declared* parent of the one above it, not a naming guess
    uni_man = json.loads((A.LOCKED / uni / "manifest.json").read_text())
    assert (uni_man.get("v0") or {}).get("base_model") == wound
    wound_man = json.loads((A.LOCKED / wound / "manifest.json").read_text())
    assert wound_man.get("base_model") == base

    # and each is the kind its role claims
    for role, name in ((A.UNIFIED, uni), (A.WOUND, wound), (A.BASE, base)):
        man = json.loads((A.LOCKED / name / "manifest.json").read_text())
        assert man.get("kind") == A.KIND_FOR_ROLE[role], (
            f"{name} is kind {man.get('kind')!r} but fills the {role!r} role")


def test_an_explicit_name_is_never_retargeted():
    """Pinned comparisons against a named past generation are why ablation tables are
    readable.  Resolution must never quietly point one at the current model."""
    for name in ("DeployClot_0", "clot_gnn_v6", "some_future_artifact"):
        assert A.resolve(name) == name
        assert A.resolve(name, A.WOUND) == name


def test_aliases_and_none_follow_the_pointer():
    if A.pointer_name() is None:
        pytest.skip("no promoted artifact in this checkout")
    want = A.shipped(A.UNIFIED)
    for alias in (None, "", "clot_ml_0", "clot_ml_v0"):
        assert A.resolve(alias) == want


def test_a_missing_pointer_degrades_instead_of_raising(tmp_path, monkeypatch):
    """An ordinary checkout with no promoted artifact must still import and load.

    UNIFIED falls back to the compiled-in name; the derived roles raise instead, because a
    chain that cannot be walked is a broken promotion and guessing there is how the wrong
    model gets served for an entire campaign.
    """
    monkeypatch.setattr(A, "POINTER", tmp_path / "absent.json")
    assert A.pointer() == {}
    assert A.pointer_name() is None
    assert A.resolve(None) == A.DEFAULT_NAME
    for role in (A.WOUND, A.BASE):
        with pytest.raises(LookupError):
            A.shipped(role)


def test_a_pointer_of_the_wrong_kind_is_ignored(tmp_path, monkeypatch):
    bad = tmp_path / "ptr.json"
    bad.write_text(json.dumps({"kind": "temporal_v4", "name": "DeployClot2"}), encoding="utf-8")
    monkeypatch.setattr(A, "POINTER", bad)
    assert A.pointer_name() is None


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError):
        A.shipped("wall")


def test_v0_reexports_stay_wired_to_the_registry():
    """`v0` re-exports these for its many existing importers; they must not fork."""
    from src.clot_ml import v0

    assert v0.resolve_clot_ml_name is A.resolve
    assert v0.pointer_v0_name is A.pointer_name
    assert v0.DEFAULT_NAME == A.DEFAULT_NAME
    assert v0.POINTER == A.POINTER
