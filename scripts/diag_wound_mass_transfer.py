"""Is the wound's deposition rate mass-transfer limited?  The falsifiable version.

THE OPENING.  `wound_patient006` (docs/DEPLOYCLOT.md 5b) clots only 65.4% of its wound, the
third that never clots is the MOST stagnant (wall shear p50 0.7 /s against 3.5 /s on the part
that does clot), and the two-constant ODE under-predicts its wound `Mat` by 8.4x while
tracking every other wound vessel to within 8%.  Deleting the shear GATE -- which is what
`srf2` does -- is not the same as deleting shear DEPENDENCE: the ungated law still needs
`RP`/`AP` carried from the bulk to the wall, and in a dead zone that delivery is the
bottleneck.

WHY THE TEST IS CLEAN.  At `t = 0` the platelet fields are spatially FLAT -- `AP` has
coefficient of variation 0.0000 on every pack (docs/WOUND_PROGRESS.md 18.1) -- and the wound
law's bracket is

    J0_Mat = Da * 1 * [ Sat(M)*k_rs*RP + Sat(M)*k_as*AP + (Mas/M_inf)*k_aa*AP ] * step2t(t)

with `M = M_inf` and `Mas = 0` initially.  So **the current model predicts a spatially UNIFORM
initial deposition rate over the whole wound patch**, with no free parameter to explain
otherwise.  Any spatial structure in the observed initial rate is therefore unexplained by the
shipped physics, and its correlation with local shear is a direct test of the transport
hypothesis.

WHAT MASS TRANSFER PREDICTS.  Wall mass transfer in a shear boundary layer follows the Leveque
solution, `Sh ~ (wall shear)^(1/3)`, so the delivery flux goes as `sr^(1/3)`.  Two resistances
in series -- reaction then delivery -- give

    1/J = 1/J_rxn + 1/J_mt ,      J_mt = alpha * sr^(1/3) * C

which is flat wherever delivery is fast (`J_mt >> J_rxn`, the flowing wounds) and proportional
to `sr^(1/3)` wherever it is slow (the dead zone).  So the prediction is not "rate rises with
shear" -- it is a SATURATING curve with a **1/3** exponent in its rising limb and a plateau,
and the flowing vessels should sit on the plateau while `wound_patient006` straddles the knee.

    python scripts/diag_wound_mass_transfer.py --flow fem
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.wall_cohort_constants import WOUND_COHORT  # noqa: E402
from src.clot_ml.temporal import _flow_hops  # noqa: E402
from src.clot_ml.wound import wound_mask  # noqa: E402
from src.config import BiochemConfig  # noqa: E402
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
MAT_CH = 15          # Mat_log1p_nd
MAT_S = 7e10         # pack Mat_log1p_nd -> COMSOL model units
AP_CH, RP_CH = 5, 4  # AP_log1p_nd, RP_log1p_nd


def _fit_loglog(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Slope, intercept and Pearson r of log10(y) on log10(x), finite entries only."""
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if m.sum() < 8:
        return float("nan"), float("nan"), float("nan")
    lx, ly = np.log10(x[m]), np.log10(y[m])
    if lx.std() < 1e-12:
        return float("nan"), float("nan"), float("nan")
    s, b = np.polyfit(lx, ly, 1)
    r = float(np.corrcoef(lx, ly)[0, 1])
    return float(s), float(b), r


def probe(stem: str, flow: str) -> dict:
    d = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
    d.graph_stem = stem
    if flow == "fem":
        from src.clot_ml.v0 import solve_fem_into_pack
        solve_fem_into_pack(d)
    bio = BiochemConfig(phase="biochem")
    w = wound_mask(d)
    y = np.asarray(d.y)
    t = np.asarray(d.t).reshape(-1)
    f = t0_flow_fields(d, bio, hops=_flow_hops(flow), flow_source=flow)

    # INITIAL deposition rate, per node: the first stored interval, where M = M_inf and
    # Mas = 0 so the autocatalytic term is still off and the bracket is at its simplest.
    mat = np.expm1(y[:, :, MAT_CH]) * MAT_S
    dt = float(t[1] - t[0])
    rate0 = (mat[1] - mat[0]) / dt

    sr = np.asarray(f.sr)
    ap0 = np.expm1(y[0, :, AP_CH])
    rp0 = np.expm1(y[0, :, RP_CH])

    out = dict(stem=stem, n_wound=int(w.sum()),
               sr_p10=float(np.percentile(sr[w], 10)),
               sr_p50=float(np.median(sr[w])),
               sr_p90=float(np.percentile(sr[w], 90)),
               rate0_p50=float(np.median(rate0[w])),
               rate0_cv=float(np.std(rate0[w]) / max(np.mean(rate0[w]), 1e-30)),
               # the control: if these are flat, the shipped law predicts a flat rate
               ap0_cv=float(np.std(ap0[w]) / max(np.mean(ap0[w]), 1e-30)),
               rp0_cv=float(np.std(rp0[w]) / max(np.mean(rp0[w]), 1e-30)))
    s, _, r = _fit_loglog(sr[w], rate0[w])
    out.update(loglog_slope=s, loglog_r=r)
    out["_sr"] = sr[w].tolist()
    out["_rate0"] = rate0[w].tolist()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default="fem", choices=["gt", "pred", "fem"])
    ap.add_argument("--out", default="outputs/deployclot/wound_mass_transfer.json")
    args = ap.parse_args()

    rows = [probe(s, args.flow) for s in WOUND_COHORT]

    print("PER VESSEL -- wound nodes only, initial deposition rate against local wall shear")
    print(f"{'vessel':20s} {'n':>4s} {'sr p10':>8s} {'sr p50':>8s} {'sr p90':>8s} "
          f"{'rate0 CV':>9s} {'AP0 CV':>8s} {'slope':>7s} {'r':>7s}")
    for r in rows:
        print(f"{r['stem']:20s} {r['n_wound']:4d} {r['sr_p10']:8.2f} {r['sr_p50']:8.2f} "
              f"{r['sr_p90']:8.2f} {r['rate0_cv']:9.3f} {r['ap0_cv']:8.4f} "
              f"{r['loglog_slope']:7.3f} {r['loglog_r']:7.3f}")

    # --- pooled: do all six vessels' wound nodes collapse onto one saturating curve? -------
    sr = np.concatenate([np.asarray(r["_sr"]) for r in rows])
    rate = np.concatenate([np.asarray(r["_rate0"]) for r in rows])
    print(f"\nPOOLED, n={len(sr)} wound nodes over {len(rows)} vessels")
    print(f"{'shear band (1/s)':>20s} {'n':>5s} {'rate0 median':>14s} {'/ plateau':>10s}")
    edges = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 100.0, 200.0, 1e9]
    band = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (sr >= lo) & (sr < hi)
        if m.sum() < 3:
            continue
        band.append((lo, hi, int(m.sum()), float(np.median(rate[m]))))
    plateau = float(np.median([b[3] for b in band[-3:]])) if len(band) >= 3 else float("nan")
    for lo, hi, n, med in band:
        hi_s = "inf" if hi > 1e8 else f"{hi:g}"
        print(f"{f'[{lo:g}, {hi_s})':>20s} {n:5d} {med:14.4e} {med / plateau:10.3f}")

    # the rising limb: nodes below the knee, where delivery should be limiting
    lim = sr < 10.0
    s_lo, _, r_lo = _fit_loglog(sr[lim], rate[lim])
    s_hi, _, r_hi = _fit_loglog(sr[~lim], rate[~lim])
    print(f"\nlog-log slope BELOW 10 /s (delivery-limited?)  {s_lo:+.3f}  (r {r_lo:+.3f}, "
          f"n={int(lim.sum())})")
    print(f"log-log slope ABOVE 10 /s (reaction-limited?)  {s_hi:+.3f}  (r {r_hi:+.3f}, "
          f"n={int((~lim).sum())})")
    print("\nLeveque wall mass transfer predicts +0.333 on the rising limb and ~0 on the "
          "plateau.")

    payload = dict(flow=args.flow, per_vessel=[{k: v for k, v in r.items()
                                                if not k.startswith("_")} for r in rows],
                   pooled=dict(bands=[dict(lo=b[0], hi=b[1], n=b[2], rate0_median=b[3])
                                      for b in band],
                               plateau=plateau,
                               slope_below_10=s_lo, r_below_10=r_lo,
                               slope_above_10=s_hi, r_above_10=r_hi,
                               n_nodes=int(len(sr))))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[save] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
