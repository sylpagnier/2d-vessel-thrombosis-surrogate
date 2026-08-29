"""Does the corpus reach deployment's wall-shear regime at P2 node resolution?

    python scripts/diag_corpus_node_resolution.py [--limit N]

`preflight_kine_cohort.py` answers the same question from the built `.pt` graphs.  This
answers it from the solver's `.npz` alone, for two reasons:

*   A solved corpus copied off the COMSOL box without the meshes it was solved on cannot
    be converted at all, and that is exactly when the answer is wanted -- before deciding
    whether to spend another 250 solves on the recipe.
*   The corpus is solved P2, so its `.npz` carries mid-side values (`mid_u`/`mid_v`) as
    well as corner values.  The union of the two node sets IS the corpus at half the mesh
    spacing, so the resolution question can be settled with no new CFD: measure the regime
    on corners alone, then on corners + mid-sides, and read the gain.

Connectivity is recovered by Delaunay with an alpha filter at 1.6h, which drops the
triangles Delaunay invents across the concave wall.  On the 25 FIT deploy packs -- where
the true edges are known -- that recovery reproduces `wall_dsrx_sd` to 0.89x and
`wall_sep_only` to 0.98x, so the numbers below carry roughly a 10-20% reconstruction
uncertainty and should not be read finer than that.

Everything is in the consumer's own convention (MLS `hops=3`, wall nodes, gate thresholds
from `BiochemConfig`), so it is directly comparable to `data/reference/deploy_wall_shear_band.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

KEYS = ("wall_sr_med", "wall_dsrx_sd", "wall_fire", "wall_sep", "wall_sep_only")


def edges_from_points(xy, alpha_mult: float = 1.6):
    """Undirected `(2, 2E)` edge index over a point cloud, minus the concavity-spanning cells."""
    import numpy as np
    from scipy.spatial import Delaunay

    T = Delaunay(xy).simplices
    lengths = np.stack([np.linalg.norm(xy[T[:, i]] - xy[T[:, j]], axis=1)
                        for i, j in ((0, 1), (1, 2), (2, 0))], axis=1)
    h = float(np.median(lengths))
    T = T[lengths.max(axis=1) < alpha_mult * h]
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=0)
    e = np.unique(np.sort(e, axis=1), axis=0)
    return np.concatenate([e, e[:, ::-1]], axis=0).T.astype("int64"), h


def regime_from_points(xy, u, v, u_ref, d_bar):
    """Consumer-convention wall gate statistics for a raw point cloud.  ``None`` if unusable."""
    import numpy as np

    from src.clot_ml.features import M_TO_CM
    from src.config import BiochemConfig
    from src.core_physics.mls_gradient import build_mls_gradient, shear_rate_2d

    bio = BiochemConfig(phase="biochem")
    lss, sgt = float(bio.lss), float(bio.sgt) / M_TO_CM

    ei, h = edges_from_points(xy)
    Dx, Dy = build_mls_gradient(xy / d_bar, ei, hops=3)
    un, vn = u / u_ref, v / u_ref
    sr = shear_rate_2d(Dx @ un, Dy @ un, Dx @ vn, Dy @ vn) * (u_ref / d_bar)
    dsrx = (Dx @ sr) / (d_bar * M_TO_CM)

    # No-slip identifies the wall; the corpus npz carries no mask.
    wall = np.hypot(u, v) < 1e-3 * u_ref
    if wall.sum() < 5:
        return None
    s, dx = sr[wall], dsrx[wall]
    lo, sep = s < lss, dx < sgt
    fire = lo | sep
    return {
        "wall_sr_med": float(np.median(s)),
        "wall_dsrx_sd": float(np.std(dx)),
        "wall_fire": float(fire.mean()),
        "wall_sep": float(sep.mean()),
        "wall_sep_only": float((sep & ~lo).sum() / max(fire.sum(), 1)),
        "h_nd": h / d_bar,
        "n": int(len(u)),
        "n_wall": int(wall.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/processed/cfd_results_kinematics/carreau")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="outputs/diag_corpus_node_resolution.json")
    a = ap.parse_args()

    import numpy as np

    from src.utils.wall_shear_regime import load_reference

    files = sorted((REPO / a.src).glob("*.npz"))
    if a.limit:
        files = files[: a.limit]
    if not files:
        print(f"[ERR] no .npz under {a.src}")
        return 1

    p1_rows, p2_rows = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        u_ref, d_bar = float(z["u_ref"]), float(z["d_bar"])
        corner = regime_from_points(np.stack([z["x"], z["y"]], 1).astype(float),
                                    z["u"].astype(float), z["v"].astype(float), u_ref, d_bar)
        if "mid_x" not in z:
            print(f"[warn] {f.name}: no mid-side samples (order_fluid < 2); corners only")
            if corner:
                p1_rows.append(corner)
            continue
        xy2 = np.concatenate([np.stack([z["x"], z["y"]], 1),
                              np.stack([z["mid_x"], z["mid_y"]], 1)], 0).astype(float)
        u2 = np.concatenate([z["u"], z["mid_u"]]).astype(float)
        v2 = np.concatenate([z["v"], z["mid_v"]]).astype(float)
        keep = np.unique(np.round(xy2, 12), axis=0, return_index=True)[1]
        both = regime_from_points(xy2[keep], u2[keep], v2[keep], u_ref, d_bar)
        if corner and both:
            p1_rows.append(corner)
            p2_rows.append(both)

    ref = (load_reference() or {}).get("summary", {})
    print(f"\n{len(p1_rows)} vessels from {a.src}\n")
    print(f"{'metric':<16}{'corners':>12}{'+mid-side':>12}{'gain':>7}{'deploy p50':>12}")
    for k in KEYS + ("h_nd", "n"):
        med1 = float(np.median([r[k] for r in p1_rows]))
        med2 = float(np.median([r[k] for r in p2_rows])) if p2_rows else float("nan")
        dep = ref.get(k, {}).get("p50", float("nan"))
        print(f"{k:<16}{med1:>12.4g}{med2:>12.4g}{(med2 / med1 if med1 else float('nan')):>7.2f}"
              f"{dep:>12.4g}")
    if p2_rows:
        print(f"\nvessels whose `dsrx` gate branch fires at all:  corners "
              f"{np.mean([r['wall_sep_only'] > 0 for r in p1_rows]):.2f}"
              f"   +mid-side {np.mean([r['wall_sep_only'] > 0 for r in p2_rows]):.2f}")

    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"src": a.src, "corners": p1_rows, "p2": p2_rows}, indent=2),
                   encoding="utf-8")
    print(f"[save] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
