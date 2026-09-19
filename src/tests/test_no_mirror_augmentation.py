"""Mirror-Y augmentation is retired: the packs were never valid reflections.

The six `*_mirror_y` packs on disk were written under an older channel layout, so under
the current schema they carried negated `node_type` one-hots, a negated `wss_prior_nd`
magnitude, a negated `AP_log1p_nd` concentration, an unflipped `v_prior` / `v0_pred`, and
stale WLS operators (`G_y`, `M_inv`, `V`, `W`) built on the un-mirrored graph.  The only
A/B ever run on them (`WG_prec_mirror` 0.3529 vs `WG_prec_iter` 0.3536, n=1 holdout) was
far inside the cohort noise floor, and a mirror sitting in its twin's LOO fold has already
produced one false burden result (WALL_MODEL_PLAN.md 2.6).

These tests fail loudly if either the packs or the augmentation plumbing come back.
"""
from __future__ import annotations

from pathlib import Path

from src.utils.paths import anchor_packs_dir, get_project_root

ROOT = get_project_root()
ANCHOR_DIR = anchor_packs_dir()


def test_no_mirror_packs_on_disk():
    if not ANCHOR_DIR.is_dir():
        return
    found = sorted(p.name for p in ANCHOR_DIR.glob("*mirror*"))
    assert not found, (
        f"mirror packs are retired but found {found}; see this module's docstring before "
        "regenerating any of them"
    )


def test_no_mirror_augmentation_plumbing():
    """`scripts/archive` is exempt: it is a frozen record of runs, not live code."""
    needles = ("augment_mirror_y", "SPECIES_AUGMENT_MIRROR_Y")
    hits: list[str] = []
    for base in (ROOT / "src", ROOT / "scripts"):
        for pattern in ("*.py", "*.ps1"):
            for path in base.rglob(pattern):
                if path.name == Path(__file__).name or "archive" in path.parts:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                hits += [f"{path.relative_to(ROOT)}: {n}" for n in needles if n in text]
    assert not hits, f"mirror-Y augmentation plumbing is retired: {hits}"
