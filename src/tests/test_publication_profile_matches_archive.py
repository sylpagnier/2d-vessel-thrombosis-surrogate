"""A figure profile must describe the OOF archive it actually reads.

`ensure_oof_series` only regenerates the archive when the file is MISSING, so a profile whose
`oof_arms` / `oof_cache` disagree with the archive on disk is invisible: every figure is built
from the archive that is there, while the profile -- and anything quoting it -- names a
different model.  The mismatch only surfaces if someone deletes the file, at which point the
figures silently change generation.

That had happened.  The `shipped` profile declared `oof_arms=("v5a,v5b,v5c",)` on the `gt`
cache while `clot_ml_0_oof_series.npz` recorded
``{"arms": ["dc_fem_cfw025"], "cache": "v5_fem", "flow": "fem"}`` -- the GT-flow research arms
named over the deployed FEM ones.  Figures 3, 4 and 6 all rest on that archive, and Table 4's
caption is generated from its `flow` field.

The test is skipped when the archive is absent, because a fresh clone has no outputs; it is an
agreement check between two things that exist, not a demand that they exist.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.utils.paths import get_project_root

REPO = get_project_root()


def _profiles():
    from scripts.publication.config import PROFILES
    return PROFILES


@pytest.mark.parametrize("profile", sorted(_profiles()))
def test_profile_agrees_with_its_archive(profile):
    profiles = _profiles()
    spec = profiles[profile]
    path = REPO / "outputs" / spec["subdir"] / "data" / spec["oof_series_name"]
    if not path.is_file():
        pytest.skip(f"{path.relative_to(REPO)} not built in this checkout")

    with np.load(path, allow_pickle=False) as z:
        assert "meta" in z.files, f"{path.name} has no metadata to check against"
        meta = json.loads(str(z["meta"][0]))

    # Compared VERBATIM, not comma-split: `eval_strict_temporal --arms` takes one arm per
    # entry and each arm is itself a comma-separated tag list, so `("a,b",)` means one arm
    # averaging two seed replicates while `("a", "b")` means two independent arms.  Splitting
    # on commas would call those equal and hide a real change of protocol.
    want_arms = sorted(str(a) for a in spec["oof_arms"])
    got_arms = sorted(str(a) for a in meta.get("arms", []))
    assert got_arms == want_arms, (
        f"profile {profile!r} declares arms {want_arms} but {path.name} was built from "
        f"{got_arms}. The figures come from the FILE, so the profile is the thing that is "
        f"wrong -- fix PROFILES, or delete the archive and regenerate it.")

    assert str(meta.get("cache")) == spec["oof_cache"], (
        f"profile {profile!r} declares cache {spec['oof_cache']!r} but {path.name} was built "
        f"from {meta.get('cache')!r}.")


@pytest.mark.parametrize("profile", sorted(_profiles()))
def test_archive_excludes_the_sealed_split(profile):
    """No FINAL_HALF vessel may appear in a published archive, whatever the profile."""
    from src.core_physics.wall_cohort_splits import SEALED

    spec = _profiles()[profile]
    path = REPO / "outputs" / spec["subdir"] / "data" / spec["oof_series_name"]
    if not path.is_file():
        pytest.skip(f"{path.relative_to(REPO)} not built in this checkout")
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"][0]))
    leaked = sorted(set(SEALED) & {str(v) for v in meta.get("vessels", [])})
    assert not leaked, f"{path.name} exports SEALED trajectories: {leaked}"
