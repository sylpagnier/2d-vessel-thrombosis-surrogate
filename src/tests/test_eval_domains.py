"""The evaluation domains, decided 2026-08-22 (roadmap item A3).

WOUND_PROGRESS 9 / 12.3 left open whether the WALL scoring domain should be `mask_wall` or
`solid_boundary_mask`.  Two measurements settled it, and both are pinned here because the
whole decision rests on them:

  * **no cohort pack carries a wound**, so the two candidate masks are identical on every
    vessel any published number was computed on -- the choice cannot move a figure;
  * **`mask_wound` is 100% GT clot on all three wound packs**, so a domain containing it
    measures COVERAGE, not skill (WOUND_PROGRESS 13).

Hence: wall stays `mask_wall`; **off-wall becomes `~solid` (true lumen) rather than `~wall`**,
which is where those 100%-GT nodes were silently sitting; and the wound keeps its own
purpose-built split (`wound_region_masks` -> wnd / w_reg / w_lum).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.clot_ml.data import eval_domains, off_domain, wall_domain, wound_of

PACKS = Path(__file__).resolve().parents[2] / "data/processed/graphs_biochem_anchors"


def _sample(n=10, wall_idx=(0, 1), wound_idx=()):
    wall = np.zeros(n, dtype=bool)
    wall[list(wall_idx)] = True
    solid = wall.copy()
    solid[list(wound_idx)] = True
    return dict(wall=wall, solid=solid)


def test_off_wall_is_true_lumen_not_just_not_wall():
    S = _sample(wound_idx=(2, 3))
    wall, off = eval_domains(S)
    assert wall.tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert off[2] == off[3] == False, "wound nodes must be in NEITHER global domain"
    assert off[4:].all(), "everything past the boundary is lumen"
    assert wound_of(S).tolist() == [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]


def test_every_node_is_in_exactly_one_of_the_three():
    S = _sample(wound_idx=(2, 3))
    wall, off = eval_domains(S)
    assert np.array_equal(wall.astype(int) + off.astype(int) + wound_of(S).astype(int),
                          np.ones(len(wall), int))


def test_a_sample_without_a_wound_is_untouched():
    S = _sample()
    wall, off = eval_domains(S)
    assert np.array_equal(off, ~S["wall"])
    assert not wound_of(S).any()


def test_a_cache_predating_the_geometry_union_falls_back_exactly():
    """Older caches carry no `solid` key; `~wall` must be reproduced bit for bit."""
    S = {"wall": np.array([True, False, False, True])}
    wall, off = eval_domains(S)
    assert np.array_equal(off, ~S["wall"])
    assert np.array_equal(wall_domain(S), S["wall"])
    assert np.array_equal(off_domain(S), ~S["wall"])


# --------------------------------------------------------------------------- on disk
def test_no_cohort_pack_carries_a_wound():
    """The fact the A3 decision rests on: it makes the change inert on every published number.

    If this ever fires, a wound vessel has entered the cohort and the bare `~wall` sites in
    `scripts/eval_strict_temporal.py`'s mask builders must be converted to `off_domain`.
    """
    from src.core_physics.wall_cohort_splits import CLOT_FREE, DEV, FIT, SEALED

    seen, offenders = 0, []
    for a in list(FIT) + list(DEV) + list(SEALED) + list(CLOT_FREE):
        p = PACKS / f"{a}.pt"
        if not p.exists():
            continue
        seen += 1
        d = torch.load(p, map_location="cpu", weights_only=False)
        m = getattr(d, "mask_wound", None)
        if torch.is_tensor(m) and m.numel() and bool(m.reshape(-1).bool().any()):
            offenders.append(a)
    if seen < 10:
        pytest.skip("cohort packs not present in this checkout")
    assert not offenders, (
        "cohort packs now carry a wound: %s.  The A3 domain decision was justified on the "
        "fact that they did not -- re-read src/clot_ml/data.eval_domains before trusting any "
        "wall/off number for these." % offenders)


def test_the_wound_mask_is_all_gt_clot_which_is_why_it_is_not_a_domain():
    """The other half of the justification (WOUND_PROGRESS 13)."""
    packs = sorted(PACKS.glob("wound_patient*.pt"))
    if not packs:
        pytest.skip("no wound packs on disk")
    for p in packs:
        d = torch.load(p, map_location="cpu", weights_only=False)
        names = d.y_channel_names.split(",")
        mat = np.expm1(
            d.y[-1, :, names.index("Mat_log1p_nd")].double().numpy()) * 7e10
        w = d.mask_wound.reshape(-1).bool().numpy()
        frac = float((mat >= 2e7)[w].mean())
        assert frac == 1.0, (
            "%s: %.1f%% of the wound is GT clot.  It was 100%%, which is why the wound is "
            "scored by `wound_region_masks` and not folded into a global domain -- if that "
            "has changed, revisit the A3 decision." % (p.name, 100 * frac))
