"""Build the JSON payload for the clot_gnn_v4 time-lapse visualization (VIZ_STANDARD).

v4 supersedes v3: same "time is a direct model input" design, plus a new advective-
transport feature (COMSOL's own advection operator solved on the mesh, live per query
time) and a stricter, nested nothing-selected-on-held-out-data protocol
(docs/PHASE10_V4.md). Off-wall gains over v3 are statistically significant under a paired
vessel bootstrap; wall gains are not and sit inside the cohort's own noise floor.

HONESTY NOTE on the training-pool vessels below: per docs/PHASE10_V4.md 8b, both the GNN
ensemble and the temporal readout are fitted on the full 19-vessel eligible pool, so
"there is no held-out vessel left to score these exact weights against." Those 5 vessels
are IN-SAMPLE. The strict-CV number in the artifact (manifest.json scores_strict_cv) is
the real generalization estimate for that part.

SEALED IS SPLIT -- see docs/SEALED_SPLIT.md before changing anything below.
WALL_COHORT_V2_GENERALIZATION (8 vessels) was split in half, deterministically, by
sorted-ID parity: VIZ_HALF may be opened for visualization (never for model selection);
FINAL_HALF stays fully closed, reserved for the project's one true final read. This
script includes the two most useful VIZ_HALF vessels (patient042, the one non-truncated
stenosis outside the training pool; patient001, a genuinely never-trained-on baseline
vessel) and asserts FINAL_HALF is never touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.locked import load_default, predict_default_series
from src.config import PhysicsConfig
from src.core_physics.mls_gradient import node_positions
from src.core_physics.physics_lumen_model import median_edge_length, wall_normal_projection
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.evaluation.clot_relaxed_metrics import (
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = Path("data/processed/graphs_biochem_anchors")

# docs/SEALED_SPLIT.md -- the durable, deterministic split of WALL_COHORT_V2_GENERALIZATION.
# 2026-08-22: VIZ_HALF was RELEASED from SEALED into TRAIN (docs/SEALED_SPLIT.md).  The name
# is kept because the viz still labels these four vessels distinctly, but they are no longer
# held out -- a model promoted after that date may have trained on them, so their score is
# no longer out-of-sample evidence.  FINAL_HALF is unchanged and still sealed.
VIZ_HALF_SEALED = ["patient001", "patient010", "patient014", "patient042"]
FINAL_HALF_SEALED = ["patient007", "patient013", "patient031", "patient043"]

# Training-pool vessels (in-sample for clot_gnn_v4, per data/reference/clot_gnn_locked.json).
PRIORITY_ANCHORS = ["patient040", "patient041", "patient044"]   # aneurysm, stenosis, stenosis
BASELINE_ANCHORS = ["patient012", "patient032"]

# The two VIZ_HALF vessels actually shown: the one non-truncated stenosis outside the
# training pool (042) and one genuinely never-trained-on baseline vessel (001).
SEALED_VIZ_SHOWN = ["patient042", "patient001"]

VESSELS = SEALED_VIZ_SHOWN + PRIORITY_ANCHORS + BASELINE_ANCHORS
CLASS_OF = {"patient040": "aneurysm", "patient041": "stenosis", "patient044": "stenosis",
           "patient012": "baseline", "patient032": "baseline",
           "patient042": "stenosis", "patient001": "unknown"}
SEALED_GUARD = set(FINAL_HALF_SEALED)
N_FRAMES = 13
MAX_BG_POINTS = 1800


def main() -> None:
    phys = PhysicsConfig(phase="biochem")
    bundle, kind = load_default()
    assert kind in ("temporal_v4", "temporal_v4_wound"), f"expected a v4-family model shipped, got {kind}"
    # temporal_v4_wound (e.g. clot_gnn_v5w) is byte-identical to its base model on any pack
    # with no wound mask -- these VESSELS have none, so this exercises the base ensemble.
    model_name = (bundle.get("manifest") or {}).get("name", "clot_gnn_v4")
    # Read WHO this exact shipped model actually trained on straight off its own manifest,
    # rather than trusting a hardcoded VIZ_HALF/FINAL_HALF split that only held for
    # clot_gnn_v3/v4/v4w. docs/SEALED_SPLIT.md's 2026-08-22 amendment moved patient001/010/
    # 014/042 from held-out into TRAIN -- a model promoted after that date (v5, v5w) has
    # seen them, so badging them "SEALED" for those models would be a lie the viz tells.
    base_ens_manifest = (bundle.get("ens") or (bundle.get("base") or {}).get("ens") or {}).get("manifest", {})
    training_pool = set(base_ens_manifest.get("training_pool", []))

    out = {}
    for anchor in VESSELS:
        assert anchor not in SEALED_GUARD, "FINAL_HALF is SEALED, do not open -- docs/SEALED_SPLIT.md"
        is_held_out = bool(training_pool) and anchor not in training_pool
        is_sealed_viz = is_held_out
        d = torch.load(DIR / f"{anchor}.pt", map_location="cpu", weights_only=False)
        pos = node_positions(d)
        wall = d.mask_wall.reshape(-1).bool().numpy()
        n = len(wall)
        ei = d.edge_index.numpy()
        interior = np.where(~wall)[0]
        stride = max(1, len(interior) // MAX_BG_POINTS)
        bg = interior[::stride]

        t = d.t.reshape(-1).detach().cpu().numpy().astype(np.float64)
        times = list(range(len(t)))
        # No `sample=` override: v4 needs its own 69-column, variant="v4" feature build
        # (build_sample(..., variant="v4")) -- the v3-era cache is the wrong shape for it.
        res = predict_default_series(bundle, kind, d, times, flow="gt")
        series = res["series"]                                   # {ti: bool mask [N]}
        model_hot = np.stack([series[i] for i in range(len(t))], axis=0)  # [T, N]

        gt_hot = np.zeros((len(t), n), dtype=bool)
        for i in range(len(t)):
            gt_hot[i] = gt_clot_phi_at_time(d, i, phys, device=torch.device("cpu")).numpy() > 0.5

        dist_raw, _ = wall_normal_projection(pos, wall)
        h_edge = median_edge_length(pos, ei)
        dist_norm = np.clip(dist_raw / (1.5 * max(h_edge, 1e-9)), 0.0, 1.0)

        model_final = model_hot[-1]
        gt_ever = gt_hot.any(axis=0)
        lumen_render_set = np.where((model_final | gt_ever) & ~wall)[0]
        wall_idx = np.where(wall)[0]
        frame_idx = np.linspace(0, len(t) - 1, N_FRAMES).round().astype(int)

        def pts(idx):
            return [[round(float(pos[i, 0]), 4), round(float(pos[i, 1]), 4)] for i in idx]

        ei_t = d.edge_index
        wall_f = torch.tensor(wall.astype(np.float32))
        off_f = torch.tensor((~wall).astype(np.float32))

        def domain_score(pred_hot, gt_hot_t, domain_f):
            pred_d = torch.tensor(pred_hot.astype(np.float32)) * domain_f
            gt_d = torch.tensor(gt_hot_t.astype(np.float32)) * domain_f
            m = compute_clot_relaxed_metrics(pred_d, gt_d, ei_t, wall_mask=torch.tensor(wall))
            return clot_score_from_deploy_dict(metrics_to_deploy_prefix(m))

        score_wall = [domain_score(model_hot[i], gt_hot[i], wall_f) for i in range(len(t))]
        score_offwall = [domain_score(model_hot[i], gt_hot[i], off_f) for i in range(len(t))]

        out[anchor] = {
            "model": model_name,
            "flow": "gt",
            "geom_class": CLASS_OF[anchor],
            "sealed_viz": is_sealed_viz,
            "was_viz_half": anchor in VIZ_HALF_SEALED,
            "t_final": float(t[-1]),
            "n_wall": int(wall.sum()),
            "bg": pts(bg),
            "wall_pos": pts(wall_idx),
            "lumen_pos": pts(lumen_render_set),
            "lumen_dist": [round(float(x), 3) for x in dist_norm[lumen_render_set]],
            "frame_t": [round(float(t[i]), 1) for i in frame_idx],
            "frame_gt_wall": [[bool(x) for x in (gt_hot[i] & wall)[wall_idx]] for i in frame_idx],
            "frame_model_wall": [[bool(x) for x in (model_hot[i] & wall)[wall_idx]] for i in frame_idx],
            "frame_gt_lumen": [[bool(x) for x in (gt_hot[i] & ~wall)[lumen_render_set]] for i in frame_idx],
            "frame_model_lumen": [[bool(x) for x in (model_hot[i] & ~wall)[lumen_render_set]] for i in frame_idx],
            "score_t": [round(float(x), 1) for x in t],
            "score_wall": [round(float(x), 4) for x in score_wall],
            "score_offwall": [round(float(x), 4) for x in score_offwall],
        }
        tag = "HELD-OUT" if is_sealed_viz else ("in-sample, ex-VIZ_HALF" if anchor in VIZ_HALF_SEALED else "in-sample")
        print(f"{anchor} [{CLASS_OF[anchor]}, {tag}]: wall={wall.sum()} lumen_render={len(lumen_render_set)}  "
              f"final wall score={score_wall[-1]:.3f}  final offwall score={score_offwall[-1]:.3f}")

    out_path = Path("outputs/v4_temporal_data.json")
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
