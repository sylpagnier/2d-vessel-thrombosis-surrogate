"""Mirror-branch vessels: where the steady flow is bistable and COMSOL tossed the coin.

On a mirror-symmetric vessel the steady Carreau problem can have two stable solutions that are
reflections of each other (a Coanda pitchfork: the jet hugs one wall or the other).  COMSOL
landed on one; our FEM solve can land on the other.  Nothing about the geometry prefers either,
so a clot prediction that is COMSOL's clot MIRRORED is not a model error, and scoring it as one
penalises an arbitrary branch choice (comsol045/046, first measured 2026-09-05).

THE RULE, and why it is decided from the FLOW, never from the clot score:

    A vessel is scored against its MIRRORED ground truth iff
      (1) its mesh maps onto its own reflection across the long axis (`mirror_permutation`), and
      (2) the solved t=0 flow is closer to COMSOL's t=0 flow reflected than to COMSOL's flow
          as given, by `BRANCH_MARGIN` (`flow_branch`).

Choosing the branch by "whichever clot score is higher" would forgive every symmetric vessel a
little, including ones already on COMSOL's branch, and bias every cohort mean upward.  The flow
test uses no clot label.  Once decided the choice is one per vessel: the same for every frame,
domain, metric and arm.

WHERE IT APPLIES.  EVALUATION ONLY -- every ground-truth mask a score, a readout tuner or a
figure compares a prediction against goes through `eval_gt`.  Training targets are untouched:
the shipped model was fitted on COMSOL's labels and re-labelling would be a different model.
`src/tests/test_mirror_branch.py` fails when a new evaluation path reads a raw label.

THE REGISTRY is `configs/mirror_branch.json`, written by `scripts/build_mirror_branch_registry.py`
and committed, so the decision is made once, reviewed in a diff, and read identically by every
script.  The node permutation is recomputed from positions on use and re-checked for symmetry,
so a registry entry can never be applied to a mesh it does not fit.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.spatial import cKDTree

from src.utils.paths import get_project_root

REGISTRY = get_project_root() / "configs" / "mirror_branch.json"
#: 99th-percentile reflection residual, in typical node spacings, below which a mesh is symmetric
SYMMETRY_TOL_SPACING = 0.75
#: fraction of nodes that must map to distinct partners
SYMMETRY_MIN_UNIQUE = 0.95
#: the mirrored flow must beat the unmirrored one by this factor to call the branch flipped
BRANCH_MARGIN = 2.0


@dataclass(frozen=True)
class MirrorMap:
    perm: np.ndarray            # node i's reflection partner
    normal: np.ndarray          # unit normal to the reflection axis (for vector fields)
    residual: float             # 99th-percentile reflection residual / typical node spacing
    unique_frac: float

    @property
    def symmetric(self) -> bool:
        return self.residual < SYMMETRY_TOL_SPACING and self.unique_frac > SYMMETRY_MIN_UNIQUE


def mirror_permutation(pos: np.ndarray) -> MirrorMap:
    """Reflect nodes across the vessel's principal axis and match each to its nearest node."""
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    c = pos.mean(0)
    X = pos - c
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    u, n = vt[0], vt[1]
    tree = cKDTree(pos)
    spacing = float(np.median(tree.query(pos, k=2)[0][:, 1]))
    d, idx = tree.query(c + np.outer(X @ u, u) - np.outer(X @ n, n))
    return MirrorMap(perm=idx, normal=n, residual=float(np.percentile(d, 99) / spacing),
                     unique_frac=float(len(np.unique(idx)) / len(idx)))


def reflect_vectors(uv: np.ndarray, m: MirrorMap) -> np.ndarray:
    """A nodal vector field (N, 2) reflected: permute nodes, flip the normal component."""
    w = np.asarray(uv, dtype=np.float64)[m.perm]
    return w - 2.0 * np.outer(w @ m.normal, m.normal)


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    nb = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / nb) if nb > 0 else float("nan")


def flow_branch(solved_uv: np.ndarray, comsol_uv: np.ndarray, m: MirrorMap) -> dict:
    """Which branch the solved flow is on -> rel-L2 both ways and the decision."""
    direct = _rel_l2(solved_uv, comsol_uv)
    mirrored = _rel_l2(solved_uv, reflect_vectors(comsol_uv, m))
    return dict(symmetric=m.symmetric, residual=round(m.residual, 3),
                unique_frac=round(m.unique_frac, 4), rel_l2_direct=round(direct, 4),
                rel_l2_mirrored=round(mirrored, 4),
                mirror_branch=bool(m.symmetric and mirrored * BRANCH_MARGIN < direct))


@lru_cache(maxsize=1)
def registry() -> dict:
    """{stem: audit row} for every symmetric vessel; a missing file means nothing is flipped."""
    if not REGISTRY.is_file():
        return {}
    return json.loads(REGISTRY.read_text(encoding="utf-8"))["vessels"]


def mirror_branch_stems() -> tuple[str, ...]:
    return tuple(sorted(s for s, r in registry().items() if r.get("mirror_branch")))


_PERMS: dict[tuple[str, int], np.ndarray] = {}


def branch_perm(stem: str | None, pos: np.ndarray) -> np.ndarray | None:
    """Permutation putting `stem`'s ground truth on the model's flow branch, or None."""
    stem = str(stem or "")
    if stem not in mirror_branch_stems():
        return None
    key = (stem, int(len(pos)))
    if key not in _PERMS:
        m = mirror_permutation(pos)
        if not m.symmetric:
            raise ValueError(f"{stem} is registered as a mirror-branch vessel but these "
                             f"{len(pos)} positions are not mirror-symmetric "
                             f"(residual {m.residual:.2f}); refusing to mirror its labels")
        _PERMS[key] = m.perm
    return _PERMS[key]


def eval_gt(stem: str | None, gt: np.ndarray, pos: np.ndarray, mirror: bool = True) -> np.ndarray:
    """THE evaluation label: ground truth on the model's branch.  (N,) or (T, N) boolean.

    Unchanged for every vessel not in the registry.  Pass positions in the same node order as
    `gt` -- a cache's `pos`, or a pack's `x[:, :2]`.

    ``mirror=False`` keeps COMSOL's labels as given.  It exists for ONE kind of arm: a
    GT-oracle coupling arm, whose features are built from COMSOL's own clot and whose prediction
    therefore follows COMSOL's branch rather than the solved flow's (`eval_strict --labels comsol`).
    """
    gt = np.asarray(gt) > 0.5 if np.asarray(gt).dtype != bool else np.asarray(gt)
    perm = branch_perm(stem, pos) if mirror else None
    if perm is None:
        return gt
    if gt.shape[-1] != len(perm):
        raise ValueError(f"{stem}: ground truth has {gt.shape[-1]} nodes but its positions have "
                         f"{len(perm)}; they must be in the same node order")
    return gt[..., perm]
