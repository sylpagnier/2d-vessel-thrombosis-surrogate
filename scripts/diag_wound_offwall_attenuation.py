"""Is the off-wall depth rule's attenuation a constant, or is it a transport quantity?

THE RULE.  The wound off-wall readout commits a lumen node in shell ``d`` when
``att**d * Mat_owner >= crit`` with ``att = 0.23`` fitted once across the cohort
(:func:`src.clot_ml.v0.replace_depth_mask`).  docs/WOUND_PROGRESS.md's open item 5 says
plainly what is wrong with that: ``0.16**k * Mat >= crit`` is a magnitude threshold and the
physics is a growth front, so the rule "cannot reach past two shells even given perfect
``Mat``".

THE HYPOTHESIS.  ``att`` is standing in for TRANSPORT.  ``Mat`` is made at the surface and
has to survive convection to reach depth; how far it gets is set by the local flow, not by a
cohort constant.  High shear thins the concentration boundary layer and the field should die
fast with depth; a stagnation band lets it accumulate and the same wall ``Mat`` should reach
further.  That is exactly the axis that separates this corpus: ``wound_patient003`` and
``006`` are stagnation-regime wounds whose off-wall clot is three layers deep, while ``001``
and ``002`` are flowing wounds whose off-wall clot is exactly one layer (16.5).

    att_node = clip(att0 * (sr_ref / sr_node) ** beta, ATT_LO, ATT_HI)

``sr_ref`` is the vessel's OWN median wall shear, so the ratio is dimensionless and carries
no absolute scale between vessels; ``beta = 0`` returns the shipped constant exactly, so the
swept family strictly CONTAINS the baseline and cannot lose by construction -- only the
leave-one-vessel-out pick can.

WHAT IS HELD FIXED.  The chemistry field, the shells, the owner map, the replacement scope
and the monotone union are the shipped ones; the base GNN series is computed once per vessel
and reused across every arm, so the arms differ in the attenuation and in nothing else.

    python scripts/diag_wound_offwall_attenuation.py --flow fem
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.biochem_gnn.wall_cohort_constants import WOUND_COHORT  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"

#: Attenuation is a survival fraction per hop; outside this range the rule is degenerate
#: (nothing ever commits, or the whole shell does regardless of `Mat`).
ATT_LO, ATT_HI = 0.05, 0.95

ATT0_GRID = (0.16, 0.23, 0.30, 0.40, 0.55)
BETA_GRID = (0.0, 0.25, 0.5, 1.0)
DEPTH_GRID = (1, 2, 3, 4, 5)


def att_field(sr: np.ndarray, wall: np.ndarray, att0: float, beta: float) -> np.ndarray:
    """Per-node attenuation.  ``beta=0`` is the shipped constant, bit-for-bit."""
    if beta == 0.0:
        return np.full(sr.shape, float(att0), dtype=np.float64)
    ref = float(np.median(sr[wall])) if wall.any() else float(np.median(sr))
    ref = max(ref, 1e-12)
    ratio = ref / np.maximum(np.asarray(sr, dtype=np.float64), 1e-12)
    return np.clip(float(att0) * ratio ** float(beta), ATT_LO, ATT_HI)


def score_one(stem: str, flow: str, every: int, artifact: str,
              scope: str | None = None) -> dict:
    from eval_wound_complement import gt_series, score_domains

    from src.clot_ml.locked import build_sample, predict_temporal_v4_wound
    from src.clot_ml.temporal import _flow_hops
    from src.clot_ml.v0 import (
        _replace_target, chemistry_mat_trajectory, load_v0_bundle, replace_depth_mask,
        solve_fem_into_pack,
    )
    from src.clot_ml.wound import solid_mask, wound_region_masks
    from src.core_physics.physics_lumen_model import (
        first_corner_shell, solid_boundary_shells, topological_owner,
    )
    from src.core_physics.physics_wall_model import t0_flow_fields

    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    bundle = load_v0_bundle(artifact)
    cfg = bundle["cfg"]
    w = bundle["base"].get("wound") or {}
    wr = (float(w["g_pre"]), float(w["g_post"]))
    crit = float(bio.viscosity_mat_crit)

    data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
    data.graph_stem = stem
    if flow == "fem":
        solve_fem_into_pack(data)
    T = int(data.y.shape[0])
    times = sorted({*range(0, T, max(every, 1)), T - 1})
    S = build_sample(data, bio, flow=flow, variant="v4")
    ei = torch.tensor(np.asarray(S["edge_index"]))
    gts = gt_series(data, phys, times)
    wall = np.asarray(S["wall"], dtype=bool)
    solid = solid_mask(data)
    off = ~solid
    pos = np.asarray(S["pos"], dtype=np.float64)
    eix = np.asarray(S["edge_index"])
    shells, owner = solid_boundary_shells(pos, solid, eix,
                                          shell1=first_corner_shell(pos, solid, eix),
                                          town=topological_owner(pos, solid, eix),
                                          max_depth=max(DEPTH_GRID))
    f = t0_flow_fields(data, bio, hops=_flow_hops(flow), flow_source=flow)
    traj = chemistry_mat_trajectory(data, bio, cfg, flow=flow, sample=S, wound_rate=wr)

    # The GNN base is identical across every arm, so it is rolled out ONCE.
    base = predict_temporal_v4_wound(bundle["base"], data, times, flow=flow, sample=S)
    reg_m, lum_m, far_m = wound_region_masks(data)
    domains = dict(wall=wall, w_reg=reg_m, w_lum=lum_m, far=far_m)
    target = _replace_target(data, off, scope or cfg.replace_scope)
    last = times[-1]
    T_raw = int(traj.shape[0])

    out: dict[str, dict] = {}
    for att0 in ATT0_GRID:
        for beta in BETA_GRID:
            a = att_field(f.sr, wall, att0, beta)
            for depth in DEPTH_GRID:
                prev = np.zeros(int(data.num_nodes), dtype=bool)
                for ti in times:
                    m = np.asarray(base["series"][int(ti)], dtype=bool).copy()
                    fld = traj[int(np.clip(ti, 0, T_raw - 1))]
                    m[target] = replace_depth_mask(fld, shells, owner, crit=crit,
                                                   att=a, depth=depth)[target]
                    prev = m | prev
                out[f"{att0:g}/{beta:g}/{depth}"] = score_domains(prev, gts[last], ei,
                                                                 wall, domains)
    return out


def lovo_pick(rows: dict, held: str, key: str) -> tuple[str, float]:
    """Best arm on every vessel EXCEPT ``held``, scored by mean ``key``."""
    arms = list(next(iter(rows.values())))
    best, bv = arms[0], -np.inf
    for a in arms:
        vals = [rows[s][a][key] for s in rows if s != held]
        vals = [v for v in vals if v == v]
        m = float(np.mean(vals)) if vals else float("nan")
        if m == m and m > bv:
            best, bv = a, m
    return best, bv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default="fem", choices=["gt", "fem"])
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--stems", nargs="*")
    ap.add_argument("--key", default="w_lum",
                    help="domain the LOVO pick optimises")
    ap.add_argument("--scope", default=None,
                    help="override the artifact's replacement scope (all_lumen | wound_region)")
    ap.add_argument("--out", default="outputs/deployclot/wound_offwall_attenuation.json")
    args = ap.parse_args()

    rows = {}
    for stem in (args.stems or list(WOUND_COHORT)):
        if not (PACKS / f"{stem}.pt").exists():
            continue
        t0 = time.time()
        rows[stem] = score_one(stem, args.flow, args.every, args.artifact, args.scope)
        n = len(rows[stem])
        print(f"[{stem}] {n} arms  {time.time() - t0:.0f}s", flush=True)

    shipped = "0.23/0/3"
    all_picks: dict[str, dict] = {}
    for key in ("w_lum", "far"):
        print()
        print(f"=== {key}: shipped {shipped} vs the leave-one-vessel-out pick")
        print(f"{'vessel':20s} {'shipped':>9s} {'LOVO arm':>16s} {'LOVO':>9s} "
              f"{'best-here':>16s} {'oracle':>9s}")
        sh, lo, orc = [], [], []
        picks = {}
        for stem in rows:
            arm, _ = lovo_pick(rows, stem, key)
            picks[stem] = arm
            v_sh = rows[stem][shipped][key]
            v_lo = rows[stem][arm][key]
            cand = {a: rows[stem][a][key] for a in rows[stem]
                    if rows[stem][a][key] == rows[stem][a][key]}
            b_arm = max(cand, key=cand.get) if cand else "--"
            v_or = cand.get(b_arm, float("nan"))
            sh.append(v_sh); lo.append(v_lo); orc.append(v_or)
            print(f"{stem:20s} {v_sh:9.4f} {arm:>16s} {v_lo:9.4f} {b_arm:>16s} {v_or:9.4f}")
        print(f"{'MEAN':20s} {np.nanmean(sh):9.4f} {'':>16s} {np.nanmean(lo):9.4f} "
              f"{'':>16s} {np.nanmean(orc):9.4f}")
        all_picks[key] = picks

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        dict(flow=args.flow, artifact=args.artifact, shipped_arm=shipped,
             att0_grid=list(ATT0_GRID), beta_grid=list(BETA_GRID),
             depth_grid=list(DEPTH_GRID), lovo_picks=all_picks,
             per_vessel=rows), indent=2), encoding="utf-8")
    print(f"\n[save] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
