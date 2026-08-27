"""Is ``wound_patient003``'s chemistry in the training support, the way its ``Mat`` is not?

v6 failed on 003 because wall ``Mat`` p90 is 27.78x crit -- the dataset maximum -- and the
residual collapsed onto the ODE.  The proposed next lever is a deploy-legal AP/RP field
feeding ``integrate_mat_trajectory(..., species=)``.  That plan has the same failure mode
if 003's *chemistry* is also the dataset maximum: a surrogate trained on the other 51 packs
would be asked to extrapolate on the test vessel.

This only reads ``data.y``.  AP/RP are reported as final / t=0 so the number is a depletion
ratio and does not depend on CGS vs SI scaling.  ``Mat`` p90 is reprinted so the two
distributions can be compared on the same vessels.

    python scripts/diag_species_ood.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig  # noqa: E402
from src.core_physics.clot_phi_simple import mat_si_for_gelation_from_log1p  # noqa: E402
from src.data_gen.lib.mesh_wls import solid_boundary_nodes  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"


def _nd(y, names, ch):
    return torch.expm1(y[:, :, names.index(ch)].clamp(-10, 8)).numpy()


def main() -> int:
    bio = BiochemConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    rows = []
    for p in sorted(PACKS.glob("*.pt")):
        if p.name.endswith(".prenormalfix"):
            continue
        stem = p.stem
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
            names = d.y_channel_names.split(",")
            T = int(d.y.shape[0])
            solid = solid_boundary_nodes(d)
            ap = _nd(d.y, names, "AP_log1p_nd")
            rp = _nd(d.y, names, "RP_log1p_nd")
            mat = mat_si_for_gelation_from_log1p(
                d.y[T - 1, :, names.index("Mat_log1p_nd")], bio).reshape(-1).numpy()
            ap0 = float(np.mean(ap[0, solid]))
            rp0 = float(np.mean(rp[0, solid]))
            ap_s = ap[T - 1, solid]
            rp_s = rp[T - 1, solid]
            # depletion: 1.0 means untouched; 0.0 means fully consumed
            ap_dep = ap_s / max(ap0, 1e-30)
            rp_dep = rp_s / max(rp0, 1e-30)
            # spatial contrast at the final frame (the thing a GNN would have to predict)
            ap_cv0 = float(np.std(ap[0, solid]) / max(ap0, 1e-30))
            ap_cvT = float(np.std(ap_s) / max(float(np.mean(ap_s)), 1e-30))
            rows.append(dict(
                stem=stem, T=T, n_solid=int(solid.sum()),
                ap_cv0=ap_cv0, ap_cvT=ap_cvT,
                ap_p10=float(np.percentile(ap_dep, 10)),
                ap_p50=float(np.percentile(ap_dep, 50)),
                ap_p90=float(np.percentile(ap_dep, 90)),
                ap_min=float(ap_dep.min()),
                rp_p10=float(np.percentile(rp_dep, 10)),
                rp_p50=float(np.percentile(rp_dep, 50)),
                mat_p90=float(np.percentile(mat[solid] / crit, 90)),
            ))
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {stem}: {type(exc).__name__} {exc}")
    # rank by how depleted AP gets (the deposition-response signature)
    rows.sort(key=lambda r: r["ap_p10"])
    print(f"{'stem':28s} {'T':>4s} {'solid':>6s} {'APcv0':>7s} {'APcvT':>7s} "
          f"{'AP p10':>8s} {'AP p50':>8s} {'AP min':>8s} {'RP p10':>8s} {'Mat p90':>8s}")
    for r in rows:
        mark = "  <<<" if r["stem"].startswith("wound_") else ""
        print(f"{r['stem']:28s} {r['T']:4d} {r['n_solid']:6d} {r['ap_cv0']:7.4f} "
              f"{r['ap_cvT']:7.3f} {r['ap_p10']:8.3f} {r['ap_p50']:8.3f} "
              f"{r['ap_min']:8.3f} {r['rp_p10']:8.3f} {r['mat_p90']:8.2f}{mark}")

    wound = [r for r in rows if r["stem"].startswith("wound_")]
    non = [r for r in rows if not r["stem"].startswith("wound_")]
    tgt = next((r for r in rows if r["stem"] == "wound_patient003"), None)
    print()
    print(f"AP t=0 CV  (should be ~0):  median {np.median([r['ap_cv0'] for r in rows]):.4f}  "
          f"max {max(r['ap_cv0'] for r in rows):.4f}")
    print(f"AP t=T  CV (contrast):      median {np.median([r['ap_cvT'] for r in non]):.3f}  "
          f"non-wound max {max(r['ap_cvT'] for r in non):.3f}  "
          f"wound max {max(r['ap_cvT'] for r in wound):.3f}")
    if tgt is None:
        return 0
    n_more_depleted = int(sum(r["ap_p10"] <= tgt["ap_p10"] for r in non))
    n_more_contrast = int(sum(r["ap_cvT"] >= tgt["ap_cvT"] for r in non))
    print(f"wound_patient003  AP p10(final/t0) = {tgt['ap_p10']:.3f}  "
          f"APcvT = {tgt['ap_cvT']:.3f}  Mat p90 = {tgt['mat_p90']:.2f}")
    print(f"  non-wound vessels at least this depleted (p10): {n_more_depleted}  -> "
          f"{'IN support' if n_more_depleted >= 3 else 'EXTRAPOLATION RISK'}")
    print(f"  non-wound vessels with at least this AP contrast (cvT): {n_more_contrast}  -> "
          f"{'IN support' if n_more_contrast >= 3 else 'EXTRAPOLATION RISK'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
