"""Cache loading, splits, and the shared readout from a per-node score to a mask."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core_physics.wall_cohort_splits import DEV, FIT

from src.utils.paths import get_project_root

REPO = get_project_root()
def load_cache(flow: str = "gt") -> dict[str, dict]:
    """Load a feature cache, warning loudly if it predates the 2026-08-22 rebuild.

    STALE CACHES ARE THE EXPENSIVE FAILURE HERE, because they do not announce themselves --
    the arrays load, the shapes are right, and every downstream number is quietly computed on
    superseded features.  There is an exact marker: `build_clot_ml_cache.py` has written a
    `solid` key since the geometry union landed (MODEL_REVIEW 5b.5), so a cache without one
    was built before the pack repair changed `wall_normal`, `node_type_*`, `width_nd` and the
    v4 transport channels.

    NAMING, because it has already cost one wrong conclusion (2026-09-02).  The live 68-column
    GT cache is `clot_ml_cache_v5`, NOT `clot_ml_cache_v4`.  `_v4` is an orphan from the
    `clot_gnn_v1` era: built 08-17 from the then-current v3 features, 19 clot-carrying vessels
    and no clot-free ones, superseded five minutes after the 08-22 rebuild by `_v5` (31
    vessels, `solid` present).  Nothing defaults to `_v4` any more.  Reading it as the GT
    counterpart of `clot_ml_cache_v4_fem` -- which the `v4`/`v4_pred`/`v4_fem` naming invites
    -- compares 19 vessels against 31 and moves every precision-weighted score.

    `build_clot_ml_cache_v4.py --flow gt` still writes to the orphan unless `--out` names the
    live cache; see the note there.
    """
    root = REPO / f"outputs/clot_ml_cache_{flow}"
    out = {}
    for p in sorted(root.glob("*.npz")):
        z = np.load(p, allow_pickle=True)
        out[p.stem] = {k: z[k] for k in z.files}
    if out and not any("solid" in S for S in out.values()):
        import warnings
        warnings.warn(
            "feature cache %r predates the 2026-08-22 pack repair (no `solid` key): its "
            "wall_normal / node_type_* / width_nd / v4 transport channels are superseded. "
            "Rebuild with `python scripts/build_clot_ml_cache.py --flow gt --force` and "
            "`build_clot_ml_cache_v4.py --force`." % str(root.name), stacklevel=2)
    # Structural staleness check, replacing the single-event `solid`-key heuristic
    # above: a hash of the feature builders travels with every cache written since
    # 2026-09-03, so ANY later change to them is detectable rather than only the one
    # historical change somebody remembered to add a marker for.  Advisory, not fatal
    # -- refusing would strand every cache the shipped artifact was trained on.
    from src.clot_ml.feature_fingerprint import check as _fp_check
    _warn = _fp_check(out, f"clot_ml_cache_{flow}")
    if _warn:
        print(f"[!] {_warn}", flush=True)
    return out


def splits(cache: dict) -> tuple[list[str], list[str]]:
    fit = [a for a in FIT if a in cache]
    dev = [a for a in DEV if a in cache]
    return fit, dev


def standardiser(cache: dict, anchors: list[str]):
    X = np.concatenate([cache[a]["X"] for a in anchors], axis=0)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def mask_from_score(score: np.ndarray, thresh: float) -> np.ndarray:
    return score >= thresh


# ---------------------------------------------------------------------------
# THE EVALUATION DOMAINS -- decided 2026-08-22 (roadmap item A3)
# ---------------------------------------------------------------------------
# The question WOUND_PROGRESS 9/12.3 left open was whether the WALL scoring domain should be
# `mask_wall` (the healthy wall) or `solid_boundary_mask` (healthy wall + wound).  Measured
# before deciding, and both measurements point the same way:
#
#   * **No cohort pack carries a wound.**  All 42 FIT/DEV/SEALED/CLOT_FREE vessels have an
#     empty `mask_wound`, so on every vessel any number in this project was ever computed on,
#     the two candidate masks are IDENTICAL.  The choice cannot move a published figure.
#   * **`mask_wound` is 100% GT clot on all three wound packs** (80/80, 80/80, 26/26).
#     WOUND_PROGRESS 13 already established what that means: a domain made of guaranteed
#     positives measures COVERAGE, not skill, and any model that commits the patch scores
#     1.0 there -- the ungated physics law does it with nothing fitted.
#
# So folding the wound into the WALL domain would hand the wall score 80 free true positives
# on a wound pack.  That is the degeneracy 13 identified, and it is the reason the wall
# domain stays `mask_wall`.
#
# But the same argument condemns the status quo, one domain over: `~mask_wall` currently
# CONTAINS those 100%-GT nodes, so today they inflate (or, unpredicted, tank) the OFF-WALL
# score instead.  The off-wall domain is meant to be lumen.  So:
#
#     wall  = mask_wall                 the gated `srf1` law's own selection
#     off   = ~solid_boundary_mask      true lumen -- excludes healthy wall AND wound
#     wound = mask_wound                its own question, answered by `wound_region_masks`
#             ( `src/clot_ml/wound.py` -> wnd / w_reg / w_lum )
#
# Every node still belongs to exactly one of the three, and the undomained score
# (`domain=None`) still covers all of them, so nothing becomes invisible.
#
# On a no-wound pack `~solid == ~wall` bit-for-bit, which is why this is safe to land before
# the Phase B rebuild rather than after it.
def eval_domains(S: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(wall, off)`` for one cached sample.  See the block comment above.

    Falls back to ``~S["wall"]`` when the sample predates `solid` (caches built before the
    2026-08-22 geometry union).  On every no-wound pack the fallback is exact.
    """
    wall = np.asarray(S["wall"], dtype=bool)
    solid = np.asarray(S.get("solid", wall), dtype=bool)
    return wall, ~solid


def wall_domain(S: dict) -> np.ndarray:
    """The wall scoring domain.  A ``dom_of``-shaped callable, for the tuners."""
    return eval_domains(S)[0]


def off_domain(S: dict) -> np.ndarray:
    """The off-wall scoring domain: TRUE LUMEN.  A ``dom_of``-shaped callable.

    Not ``~S["wall"]``.  On a wound pack that would include the wound's 100%-GT nodes.
    """
    return eval_domains(S)[1]


def wound_of(S: dict) -> np.ndarray:
    """The wound nodes of a cached sample: solid boundary that is not healthy wall."""
    wall = np.asarray(S["wall"], dtype=bool)
    return np.asarray(S.get("solid", wall), dtype=bool) & ~wall


def physics_mask(S: dict) -> np.ndarray:
    """The shipped zero-parameter backbone's own full-mesh mask, from cached fields."""
    from src.config import BiochemConfig
    from src.core_physics.physics_lumen_model import adjacency, grow_into_lumen
    bio = BiochemConfig(phase="biochem")
    wall, ei = S["wall"], S["edge_index"]
    A = adjacency(ei, len(wall)).astype(np.int8)
    cur = (S["gate"] > 0) & wall
    adm = (S["sr"] < float(bio.lss) * 2.0) & wall
    for _ in range(20):
        cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
    off = grow_into_lumen(cur, wall, A, S["spd"], S["sr"], lumen_hops=2, speed_thresh=0.2)
    return cur | off


def attach_physics(cache: dict) -> dict:
    """Add the backbone mask, and expose it to the models as an extra feature column."""
    for S in cache.values():
        if "phys_mask" in S:
            continue
        m = physics_mask(S)
        S["phys_mask"] = m
        cols = [str(c) for c in S["cols"]]
        S["sdf"] = S["X"][:, cols.index("sdf_nd")].astype(np.float64)
        S["X"] = np.concatenate([S["X"], m.astype(np.float32).reshape(-1, 1)], axis=1)
        S["cols"] = np.array([str(c) for c in S["cols"]] + ["phys_mask"])
    return cache
