"""Which available ``Mat`` field can drive the off-wall magnitude rule?

WOUND_PROGRESS 16 localised the whole off-wall gap to one quantity: the shipped rule
``shell & 0.16*Mat_owner >= crit`` is correct (fed GT ``Mat`` it scores 0.9755/0.9755/0.7897)
and the ODE's ``Mat`` orders 003's far-field candidates at CHANCE (AUC 0.5048).

Before training anything, check the field that is already sitting in every locked checkpoint
and has never been read: ``ClotGNN``'s REGRESSION head, a zero-init residual on the backbone's
own ``Mat`` trained with smooth-L1 against ``log1p(Mat/crit)``.  ``locked.predict_mat`` exposes
it; the deploy readout uses the classifier instead.  If it already ranks the far field, the v6
build is a readout change rather than a new model.

Reports, per vessel: far-field AUC on the candidate set, and the end-to-end off-wall score with
that field substituted into the otherwise-unchanged rule.

    python scripts/diag_wound_mat_sources.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.evaluate import domain_score  # noqa: E402
from src.clot_ml.locked import (  # noqa: E402
    build_sample, load_temporal_v4_wound, predict_mat, predict_temporal_v4_wound,
)
from src.clot_ml.temporal import ode_trajectory  # noqa: E402
from src.clot_ml.wound import solid_mask, wound_region_masks  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_phi_simple import mat_si_for_gelation_from_log1p  # noqa: E402
from src.core_physics.physics_lumen_model import (  # noqa: E402
    first_corner_shell, topological_owner,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
STEMS = ("wound_patient001", "wound_patient002", "wound_patient003")


def auc(score, y):
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(np.asarray(score, float))) + 1.0
    npos, nneg = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=list(STEMS))
    ap.add_argument("--name", default="clot_gnn_v5w")
    args = ap.parse_args()

    phys, bio = PhysicsConfig(phase="biochem"), BiochemConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    bundle = load_temporal_v4_wound(name=args.name)
    acc: dict[str, list[float]] = {}

    for stem in args.stems:
        data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
        T = int(data.y.shape[0])
        ei_np = data.edge_index.detach().cpu().numpy()
        ei = torch.tensor(ei_np)
        solid = solid_mask(data)
        off = ~solid
        _, _, far = wound_region_masks(data)
        gt = gt_clot_phi_at_time(data, T - 1, phys).numpy() > 0.5
        pos = data.x[:, :2].detach().cpu().numpy().astype(np.float64)

        S = build_sample(data, bio, flow="gt", variant="v4")
        pred = predict_temporal_v4_wound(bundle, data, [0, T - 1], flow="gt",
                                         sample=S)["series"][T - 1]
        pw = pred & solid
        shell = first_corner_shell(pos, solid, ei_np)
        town = topological_owner(pos, solid, ei_np)
        has = town >= 0
        ow_c = np.zeros(len(solid), bool)
        ow_c[has] = pw[town[has]]
        cand = shell & off & far & ow_c

        wr = bundle["base"]["temporal"].get("wound_rate")
        traj, _ = ode_trajectory(data, bio, flow="gt",
                                 wound_rate=None if wr is None else tuple(wr))
        mi = data.y_channel_names.split(",").index("Mat_log1p_nd")
        gmat = mat_si_for_gelation_from_log1p(data.y[:, :, mi], bio).reshape(T, -1).numpy()
        # the reg head predicts log1p(Mat/crit); invert to model units so the SAME bar applies
        reg = np.expm1(np.asarray(predict_mat(bundle["base"]["ens"], S), np.float64)) * crit

        fields = {"ODE Mat (shipped)": traj[-1], "GNN reg head": reg, "GT Mat": gmat[-1]}
        print("=" * 100)
        print(f"{stem}  off GT+={int((gt & off).sum())}  far cand={int(cand.sum())} "
              f"(GT+ {int(gt[cand].sum())})")
        print(f"  {'Mat source':22s} {'wallp90':>8s} {'farAUC':>7s} {'off@0.16':>9s} "
              f"{'TP':>5s} {'FP':>5s} | {'off@att*':>9s} {'att*':>6s}")
        atts = np.geomspace(0.02, 4.0, 40)
        for tag, fld in fields.items():
            ow = np.zeros(len(solid))
            ow[has] = np.asarray(fld, np.float64)[town[has]]
            m = shell & off & (0.16 * ow >= crit)
            sc = domain_score(m | pw, gt, ei, off, solid)
            best = max((domain_score((shell & off & (a * ow >= crit)) | pw,
                                     gt, ei, off, solid), a) for a in atts)
            acc.setdefault(tag, []).append(sc)
            a_far = auc(np.asarray(fld, np.float64)[town[cand]], gt[cand]) if cand.any() else float("nan")
            print(f"  {tag:22s} {np.percentile(np.asarray(fld)[solid], 90) / crit:8.2f} "
                  f"{a_far:7.4f} {sc:9.4f} {int((m & gt & off).sum()):5d} "
                  f"{int((m & ~gt & off).sum()):5d} | {best[0]:9.4f} {best[1]:6.3f}")

    print("=" * 100)
    for tag, v in acc.items():
        print(f"  {tag:22s} mean off@0.16 {np.mean(v):.4f}   "
              + "  ".join(f"{x:.4f}" for x in v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
