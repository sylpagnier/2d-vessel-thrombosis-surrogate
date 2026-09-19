"""Geometry-stratified protocol for the wall cohort.

WHY THE OLD CUT HAD TO GO.  `src/core_physics/wall_cohort_splits.py` puts 040/041/044 in DEV
and everything else in FIT.  Measured (`geometry_class.py`), that is **exactly** the
stenosis/aneurysm set against an all-baseline FIT, so every FIT-vs-DEV number in
`docs/PHASE9_ML.md` is confounded with geometry class: the model reads DEV off-wall 0.80 and
FIT 0.64, and that is a comparison of three pathological vessels against ten normal ones,
not evidence of generalisation.

WHY A FIXED RE-CUT CANNOT FIX IT.  `comsol039`-`comsol047` are the STRAIGHT vessels -- no
bend -- confirmed by manual read 2026-09-07 (`USER_DESIGNATED` in `geometry_class.py`):

    comsol039  aneurysm (T=92)   comsol040  aneurysm         comsol041  stenosis
    comsol042  stenosis          comsol043  aneurysm (SEALED) comsol044  stenosis
    comsol045  stenosis          comsol046  stenosis          comsol047  aneurysm

Of these, `comsol039` is excluded everywhere -- T = 92 is a truncated run, a different
quantity (PHASE6_RESULTS 6.2) -- and `comsol043` is in SEALED and stays there.  That leaves
**seven** non-SEALED, full-horizon priority vessels: two aneurysm (040, 047), five stenosis
(041, 042, 044, 045, 046) -- not the three this module was first written against.

**Straight is not the same as clean.** Being unbent is what makes 039-047 the unconfounded
axis; it is not evidence the other vessels are geometrically normal. A bent vessel can carry
real stenotic or aneurysmal geometry that was never hand-flagged because the bend dominates
the mesh's visual read -- `comsol048` reads slightly stenotic and `comsol012` is practically
a stenosis with a lot of bend (its measured `narrowing` already sits inside the designated
stenoses' range; see `geometry_class.py`). Neither is in `USER_DESIGNATED`. Anything reported
as `baseline` below means "not hand-flagged," not "confirmed normal" -- state that in any
caption built from this split.

**A fixed FIT/DEV cut still cannot do this well.** Even with two non-SEALED aneurysms
(040, 047) and five stenoses, a single two-way split puts each priority vessel on only one
side: whichever ones land in DEV are measured but never trained on, and vice versa. K-fold
is the protocol that gets every vessel both roles, not just the minimum fix for n=1.

WHAT THIS MODULE DOES INSTEAD.  Geometry-stratified K-fold over the whole eligible
non-SEALED pool.  Every vessel is held out exactly once, so:

  * every vessel has an honest out-of-fold score, including all seven priority vessels;
  * 040 is *trained on* in K-1 folds and *measured* in one -- both, rather than neither;
  * priority vessels land in different folds by construction, so each fold's training set
    contains at least two of them.

**UPDATE 2026-09-07.** With `comsol047` confirmed as a second non-SEALED aneurysm, this module
now CAN and does train on one aneurysm while measuring a different one: `stratified_folds`
deals `comsol040` and `comsol047` into different folds (verified -- they land in folds 0 and 1
respectively under the default `k=5`), so aneurysm out-of-fold performance is no longer an n=1
curiosity, it is a genuine n=2 held-out measurement with real train/test separation between the
two aneurysm vessels. `comsol039` (T=92, still excluded) and SEALED's `comsol043` remain the
only aneurysms this protocol cannot touch. Any prose still saying "aneurysm is n=1" describes
the pre-2026-09-07 designation and needs updating.
"""
from __future__ import annotations

from collections import defaultdict

import torch

from src.core_physics.wall_cohort_splits import DEV as OLD_DEV, FIT as OLD_FIT, MIN_T, SEALED

PRIORITY_CLASSES = ("aneurysm", "stenosis", "stenosis+aneurysm")


def eligible_pool() -> list[str]:
    """Non-SEALED, full-horizon, **clot-carrying** vessels, in a stable order.

    Deliberately excludes `wall_cohort_splits.CLOT_FREE`, which joined the cache on
    2026-08-22.  Every caller of this function averages a RECALL-bearing score, and an
    empty-GT vessel has no recall -- its evidence is about false positives and belongs on a
    separate row (`eval_strict.py --clot-free`).  The training pool is a different question
    and is taken from the cache itself, in `run_phase9_cv.py`.
    """
    return sorted(set(OLD_FIT) | set(OLD_DEV))


def classes_for(anchors, pack_dir) -> dict[str, str]:
    """anchor -> geometry class, using the measured classifier with its documented abstain."""
    from src.clot_ml.geometry_class import USER_DESIGNATED, classify, width_stats

    out = {}
    for a in anchors:
        p = pack_dir / f"{a}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < MIN_T:
            continue
        s = width_stats(d)
        cls = classify(s, a)
        if cls == "unknown":
            # width_nd is unusable here; fall back to the human designation, and treat an
            # unlabelled vessel as baseline for STRATIFICATION only (never for reporting).
            cls = USER_DESIGNATED.get(a, "unknown")
        out[a] = cls
    return out


def is_priority(cls: str) -> bool:
    return cls in PRIORITY_CLASSES


def stratified_folds(classes: dict[str, str], k: int = 5) -> list[list[str]]:
    """K held-out sets, dealing each geometry class round-robin so priority vessels spread.

    Deterministic: vessels are dealt in sorted order within class, and the classes are
    processed rarest-first so the scarce priority vessels choose their folds before the
    plentiful baseline ones fill the space.
    """
    by_cls: dict[str, list[str]] = defaultdict(list)
    for a, c in classes.items():
        by_cls[c].append(a)
    folds: list[list[str]] = [[] for _ in range(k)]
    order = sorted(by_cls, key=lambda c: (len(by_cls[c]), c))
    slot = 0
    for c in order:
        for a in sorted(by_cls[c]):
            folds[slot % k].append(a)
            slot += 1
    return [sorted(f) for f in folds]


def describe(classes: dict[str, str], folds: list[list[str]]) -> str:
    lines = []
    for i, f in enumerate(folds):
        tag = ", ".join("%s[%s]" % (a, classes.get(a, "?")[:4]) for a in f)
        n_prio = sum(is_priority(classes.get(a, "")) for a in f)
        lines.append("  fold %d (n=%d, priority=%d): %s" % (i, len(f), n_prio, tag))
    return "\n".join(lines)
