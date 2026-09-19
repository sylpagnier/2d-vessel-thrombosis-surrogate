"""Turn strict nested-CV trajectories into a VIZ_STANDARD payload.

The promoted ``clot_ml_0`` weights train on the complete non-sealed pool.  They are useful
for deployment, but showing their prediction on a training vessel is not a generalization
visualization.  This script accepts only the ``--save-oof-series`` output of
``eval_strict_temporal.py``: each tab therefore renders the outer-fold trajectory from a
model and temporal readout that excluded that vessel.
"""
from __future__ import annotations
from src.utils.paths import anchor_packs_dir

import argparse
import json
from pathlib import Path

import numpy as np
import torch


from src.clot_ml.geometry_splits import classes_for
from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.mls_gradient import node_positions
from src.core_physics.physics_lumen_model import median_edge_length, wall_normal_projection
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.core_physics.wall_cohort_splits import SEALED
from src.evaluation.clot_relaxed_metrics import (
    clot_score_from_deploy_dict,
    compute_clot_relaxed_metrics,
    metrics_to_deploy_prefix,
)
from src.clot_ml.v0 import load_v0_bundle, predict_clot_ml_0
from src.clot_ml.wound import solid_mask, wound_mask, wound_region_masks

PACKS = anchor_packs_dir()
N_FRAMES = 13
MAX_BG_POINTS = 1800


def _times(data, every: int) -> list[int]:
    T = int(data.y.shape[0])
    return sorted(set(list(range(0, T, max(int(every), 1))) + [T - 1]))


def _vessel_payload(anchor: str, data, series: dict[int, np.ndarray], times: list[int],
                    *, classes: dict, edge_index: torch.Tensor, healthy_wall: np.ndarray,
                    solid: np.ndarray, region: np.ndarray, flow: str,
                    base_train_count: int, temporal_train_count: int,
                    provenance_note: str, model: str, fold: str = "LOVO") -> dict:
    """Convert a wound prediction series to the same compact VIZ_STANDARD shape as OOF data."""
    phys = PhysicsConfig(phase="biochem")
    gt = np.stack([
        gt_clot_phi_at_time(data, int(ti), phys, device=torch.device("cpu"))
        .reshape(-1).numpy() > 0.5 for ti in times
    ])
    pos = node_positions(data)
    dist_raw, _ = wall_normal_projection(pos, solid)
    h_edge = median_edge_length(pos, edge_index.numpy())
    dist = np.clip(dist_raw / (1.5 * max(h_edge, 1e-9)), 0.0, 1.0)
    pred = np.stack([np.asarray(series[int(ti)], dtype=bool) for ti in times])
    boundary_idx = np.flatnonzero(solid)
    interior = np.flatnonzero(~solid)
    lumen_idx = np.flatnonzero((pred[-1] | gt.any(axis=0)) & ~solid)
    bg = interior[::max(1, len(interior) // MAX_BG_POINTS)]
    frame = np.linspace(0, len(times) - 1, N_FRAMES).round().astype(int)

    def pts(nodes: np.ndarray) -> list[list[float]]:
        return [[round(float(pos[i, 0]), 4), round(float(pos[i, 1]), 4)] for i in nodes]

    return dict(
        model=model, flow=flow, geom_class=classes.get(anchor, "unknown"),
        out_of_fold=True, fold=fold,
        base_train_count=int(base_train_count), temporal_train_count=int(temporal_train_count),
        provenance_note=provenance_note, sealed_viz=False, was_viz_half=False,
        score_off_label="off-wall", score_off_domain="off",
        region_label="wound region", region_domain="w_reg",
        t_final=float(data.t.reshape(-1)[times[-1]].item()), n_wall=int(healthy_wall.sum()),
        bg=pts(bg), wall_pos=pts(boundary_idx), lumen_pos=pts(lumen_idx),
        lumen_dist=[round(float(x), 3) for x in dist[lumen_idx]],
        frame_t=[round(float(data.t.reshape(-1)[times[i]].item()), 1) for i in frame],
        frame_gt_wall=[list(map(bool, (gt[i] & solid)[boundary_idx])) for i in frame],
        frame_model_wall=[list(map(bool, (pred[i] & solid)[boundary_idx])) for i in frame],
        frame_gt_lumen=[list(map(bool, (gt[i] & ~solid)[lumen_idx])) for i in frame],
        frame_model_lumen=[list(map(bool, (pred[i] & ~solid)[lumen_idx])) for i in frame],
        score_t=[round(float(data.t.reshape(-1)[ti].item()), 1) for ti in times],
        score_wall=[round(_score(pred[i], gt[i], edge_index, healthy_wall, healthy_wall), 4)
                    for i in range(len(times))],
        score_offwall=[round(_score(pred[i], gt[i], edge_index, solid, ~solid), 4)
                       for i in range(len(times))],
        score_wound_region=[round(_score(pred[i], gt[i], edge_index, solid, region), 4)
                       for i in range(len(times))],
    )


def build_wound_payload(stems: list[str], *, every: int, flow: str) -> dict:
    """Build a wound/generalization payload using the unified v0 dispatcher.

    The v0 GNN was trained only on the separate non-wound pool.  Wound constants are evaluated
    leave-one-wound-vessel-out (the other two vessels), so these tabs are valid generalization
    views even though they do not come from the non-wound outer-fold NPZ.
    """
    bundle = load_v0_bundle()
    manifest = bundle.get("manifest", {})
    pool = manifest.get("validation", {}).get("training_pool", {})
    base_n = int(pool.get("clot_carrying", 0)) + int(pool.get("clot_free_false_positive_only", 0))
    classes = classes_for(stems, PACKS)
    out: dict[str, dict] = {"_meta": dict(
        schema_version=1, model="clot_ml_0 wound LOVO", mode="wound", flow=flow,
        off_label="off-wall", region_label="wound region",
        protocol="wound complement leave-one-vessel-out",
        final_half_excluded=sorted(SEALED),
        note=("Wound tabs use the unified v0 dispatcher. The base GNN was trained on the "
              "separate non-wound pool; wound-rate constants use the other two wound vessels."),
    )}
    bio = BiochemConfig(phase="biochem")
    for anchor in stems:
        data = torch.load(PACKS / f"{anchor}.pt", map_location="cpu", weights_only=False)
        times = _times(data, every)
        pred = predict_clot_ml_0(bundle, data, times, flow=flow)
        ei = data.edge_index.detach().cpu()
        healthy = data.mask_wall.reshape(-1).bool().numpy()
        solid = solid_mask(data)
        wnd = wound_mask(data)
        region, _lumen, _far = wound_region_masks(data)
        out[anchor] = _vessel_payload(
            anchor, data, pred["series"], times, classes=classes, edge_index=ei,
            healthy_wall=healthy, solid=solid, region=region, flow=flow,
            base_train_count=base_n, temporal_train_count=max(len(stems) - 1, 0),
            provenance_note=(f"wound LOVO: wound-rate fit excluded {anchor}; "
                             f"wound nodes={int(wnd.sum())}"),
            model="clot_ml_0 wound LOVO",
        )
        print(f"{anchor} [LOVO]: base_train={base_n} wound_rate_train={len(stems)-1}")
    return out


def _meta(z) -> dict:
    if "meta" not in z:
        raise ValueError("series archive has no metadata; use eval_strict_temporal.py --save-oof-series")
    return json.loads(str(z["meta"][0]))


def _score(pred: np.ndarray, gt: np.ndarray, edge_index: torch.Tensor,
           wall: np.ndarray, domain: np.ndarray) -> float:
    pred_d = torch.tensor((pred & domain).astype(np.float32))
    gt_d = torch.tensor((gt & domain).astype(np.float32))
    m = compute_clot_relaxed_metrics(pred_d, gt_d, edge_index,
                                     wall_mask=torch.tensor(wall))
    return float(clot_score_from_deploy_dict(metrics_to_deploy_prefix(m)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", default="",
                    help="NPZ written by eval_strict_temporal.py --save-oof-series")
    ap.add_argument("--out", default="", help="output JSON payload")
    ap.add_argument("--wound", action="store_true",
                    help="build wound_comsol* tabs using the unified v0 dispatcher and wound LOVO")
    ap.add_argument("--stems", nargs="*",
                    default=["wound_comsol001", "wound_comsol002", "wound_comsol003"],
                    help="wound stems to include with --wound")
    ap.add_argument("--every", type=int, default=2, help="wound time-grid stride")
    ap.add_argument("--flow", choices=("gt", "pred"), default="gt",
                    help="t=0 flow source for --wound (default: gt)")
    args = ap.parse_args()

    if args.wound:
        out_path = Path(args.out or "outputs/clot_ml_0_wound_temporal_data.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(build_wound_payload(args.stems, every=args.every,
                                                           flow=args.flow)), encoding="utf-8")
        print("wrote %s (%d wound vessels)" % (out_path, len(args.stems)))
        return
    if not args.series:
        raise SystemExit("--series is required unless --wound is selected")

    with np.load(args.series, allow_pickle=False) as z:
        meta = _meta(z)
        vessels = [str(v) for v in meta.get("vessels", [])]
        final_half = set(str(v) for v in meta.get("final_half_excluded", []))
        if not vessels:
            raise ValueError("series archive contains no held-out vessels")
        if set(vessels) & set(SEALED):
            raise RuntimeError("FINAL_HALF must never enter an OOF visualization")
        if final_half != set(SEALED):
            raise RuntimeError("series archive does not declare the current FINAL_HALF exclusion")
        arrays = {key: z[key].copy() for key in z.files if key != "meta"}

    classes = classes_for(vessels, PACKS)
    phys = PhysicsConfig(phase="biochem")
    out: dict[str, dict] = {"_meta": dict(
        schema_version=1,
        model="clot_ml_0 strict OOF",
        flow=meta.get("flow", "gt"),
        protocol=meta.get("purpose"),
        source_series=str(Path(args.series)),
        final_half_excluded=sorted(final_half),
        note=("Every tab is an outer-fold prediction. Its GNN base and temporal readout "
              "excluded the displayed vessel; FINAL_HALF is not present."),
    )}

    for anchor in vessels:
        provenance_key, masks_key, times_key = (f"provenance|{anchor}", f"masks|{anchor}",
                                                f"times|{anchor}")
        if any(key not in arrays for key in (provenance_key, masks_key, times_key)):
            raise ValueError(f"{anchor}: incomplete OOF series archive")
        provenance = json.loads(str(arrays[provenance_key][0]))
        if provenance.get("held_out") != anchor or anchor in provenance.get("base_train", []):
            raise RuntimeError(f"{anchor}: OOF provenance does not prove base-model exclusion")

        data = torch.load(PACKS / f"{anchor}.pt", map_location="cpu", weights_only=False)
        wall = data.mask_wall.reshape(-1).bool().numpy()
        solid = getattr(data, "solid_boundary_mask", data.mask_wall).reshape(-1).bool().numpy()
        off = ~solid
        masks = np.asarray(arrays[masks_key], dtype=bool)
        idx = np.asarray(arrays[times_key], dtype=np.int64)
        if masks.ndim != 2 or masks.shape[1] != len(wall) or len(idx) != masks.shape[0]:
            raise ValueError(f"{anchor}: OOF mask shape does not match its graph")
        if np.any(idx < 0) or np.any(idx >= int(data.y.shape[0])):
            raise ValueError(f"{anchor}: OOF time indices are outside its simulation")

        pos = node_positions(data)
        edge_index = data.edge_index
        t = data.t.reshape(-1).detach().cpu().numpy().astype(np.float64)
        gt = np.stack([
            gt_clot_phi_at_time(data, int(ti), phys, device=torch.device("cpu"))
            .reshape(-1).numpy() > 0.5
            for ti in idx
        ])
        dist_raw, _ = wall_normal_projection(pos, solid)
        h_edge = median_edge_length(pos, edge_index.numpy())
        dist = np.clip(dist_raw / (1.5 * max(h_edge, 1e-9)), 0.0, 1.0)
        interior = np.flatnonzero(off)
        bg = interior[::max(1, len(interior) // MAX_BG_POINTS)]
        wall_idx = np.flatnonzero(wall)
        lumen_idx = np.flatnonzero(((masks[-1] | gt.any(axis=0)) & off))
        frame = np.linspace(0, len(idx) - 1, N_FRAMES).round().astype(int)

        def pts(nodes: np.ndarray) -> list[list[float]]:
            return [[round(float(pos[i, 0]), 4), round(float(pos[i, 1]), 4)] for i in nodes]

        out[anchor] = dict(
            model="clot_ml_0 strict OOF",
            flow=meta.get("flow", "gt"),
            geom_class=classes.get(anchor, "unknown"),
            out_of_fold=True,
            fold=int(provenance["fold"]),
            base_train_count=len(provenance["base_train"]),
            temporal_train_count=len(provenance["temporal_train"]),
            sealed_viz=False,
            was_viz_half=False,
            t_final=float(t[idx[-1]]),
            n_wall=int(wall.sum()),
            bg=pts(bg), wall_pos=pts(wall_idx), lumen_pos=pts(lumen_idx),
            lumen_dist=[round(float(x), 3) for x in dist[lumen_idx]],
            frame_t=[round(float(t[idx[i]]), 1) for i in frame],
            frame_gt_wall=[[(bool(x)) for x in (gt[i] & wall)[wall_idx]] for i in frame],
            frame_model_wall=[[(bool(x)) for x in (masks[i] & wall)[wall_idx]] for i in frame],
            frame_gt_lumen=[[(bool(x)) for x in (gt[i] & off)[lumen_idx]] for i in frame],
            frame_model_lumen=[[(bool(x)) for x in (masks[i] & off)[lumen_idx]] for i in frame],
            score_t=[round(float(t[ti]), 1) for ti in idx],
            score_wall=[round(_score(masks[i], gt[i], edge_index, wall, wall), 4)
                        for i in range(len(idx))],
            score_offwall=[round(_score(masks[i], gt[i], edge_index, wall, off), 4)
                           for i in range(len(idx))],
        )
        print("%s [OOF fold %d]: base_train=%d temporal_train=%d" %
              (anchor, provenance["fold"], len(provenance["base_train"]),
               len(provenance["temporal_train"])))

    out_path = Path(args.out or "outputs/clot_ml_0_oof_temporal_data.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print("wrote %s (%d OOF vessels)" % (out_path, len(vessels)))


if __name__ == "__main__":
    main()
