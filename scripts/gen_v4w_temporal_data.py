"""Build the JSON payload for the unified ``clot_ml_v0`` wound time-lapse.

v4w supersedes v4: on a pack with no wound it returns v4's output bit-for-bit (pinned by
src/tests/test_wound_complement.py). On a wound pack it adds a THIRD domain -- the wound
collar (`mask_wound`, COMSOL's `sel1`) -- which the ordinary shear gates never see (0% gate
coverage on all 3 known wound vessels; COMSOL's own wound law drops the gates entirely).
The complement forces the set on the wound domain (nearly free -- it is ~100% GT clot) and
supplies TIMING there via two fitted scalars (G_pre, G_post), which is the actual result:
docs/WOUND_PROGRESS.md 12.2 measures MOT wound 0.533 -> 0.944.

HONESTY NOTE. Only 3 wound vessels exist right now (wound_patient001/002/003) -- a
completely separate, tiny cohort from the 19-vessel patient* pool v4's GNN backbone was
trained on, so the backbone itself has never seen a wound. The two wound-specific scalars
are fit leave-one-vessel-out (LOVO, n=3) -- shown per vessel, not in-sample. wound_patient003
is a known outlier (externally triggered, docs/WOUND_PROGRESS.md 11) -- shown, not hidden.
No wound vessel is SEALED yet; the user is bringing one -- see the VESSELS list below for
where to add it once it lands.
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

from src.clot_ml.evaluate import domain_score
from src.clot_ml.v0 import load_v0_bundle, predict_clot_ml_v0
from src.clot_ml.wound import solid_mask, wound_mask, wound_region_masks
from src.config import PhysicsConfig
from src.core_physics.mls_gradient import node_positions
from src.core_physics.physics_lumen_model import median_edge_length, wall_normal_projection
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time

DIR = Path("data/processed/graphs_biochem_anchors")
# The 3 known wound vessels -- all LOVO, none SEALED (docs/WOUND_PROGRESS.md).
# Add a genuinely held-out vessel here (as its own list, badged sealed_viz=True in the
# loop below) once one exists -- do not just append it to VESSELS unbadged.
VESSELS = ["wound_patient001", "wound_patient002", "wound_patient003"]
OUTLIER = {"wound_patient003"}
N_FRAMES = 13
MAX_BG_POINTS = 1800


def main() -> None:
    phys = PhysicsConfig(phase="biochem")
    bundle = load_v0_bundle("clot_ml_v0")
    model_name = bundle["manifest"]["name"]
    base_name = bundle["manifest"]["base_model"]

    # v0 is currently an alias of clot_gnn_v6w: its manifest intentionally has the
    # release metadata, while the retained wound validation table lives on that source.
    source_name = bundle["manifest"].get("alias_of", base_name)
    source_manifest = json.loads(
        (REPO / "outputs/clot_ml/locked" / source_name / "manifest.json").read_text())
    scores_wound = source_manifest.get("scores_wound")

    out = {"_meta": {
        "model": model_name,
        "base_model": base_name,
        "scores_wound": scores_wound,
        "release_status": bundle["manifest"].get("release_status", {}),
        "validation": bundle["manifest"].get("validation", {}),
    }}
    for anchor in VESSELS:
        d = torch.load(DIR / f"{anchor}.pt", map_location="cpu", weights_only=False)
        pos = node_positions(d)
        wall = d.mask_wall.reshape(-1).bool().numpy()
        wound = wound_mask(d)
        solid = solid_mask(d)
        off = ~solid
        region, wound_lumen, _far_lumen = wound_region_masks(d)
        n = len(wall)
        ei = d.edge_index.numpy()
        interior = np.where(off)[0]
        stride = max(1, len(interior) // MAX_BG_POINTS)
        bg = interior[::stride]

        t = d.t.reshape(-1).detach().cpu().numpy().astype(np.float64)
        times = list(range(len(t)))
        res = predict_clot_ml_v0(bundle, d, times, flow="gt")
        series = res["series"]
        model_hot = np.stack([series[i] for i in range(len(t))], axis=0)  # [T, N]

        gt_hot = np.zeros((len(t), n), dtype=bool)
        for i in range(len(t)):
            gt_hot[i] = gt_clot_phi_at_time(d, i, phys, device=torch.device("cpu")).numpy() > 0.5

        dist_raw, _ = wall_normal_projection(pos, wall | wound)
        h_edge = median_edge_length(pos, ei)
        dist_norm = np.clip(dist_raw / (1.5 * max(h_edge, 1e-9)), 0.0, 1.0)

        model_final = model_hot[-1]
        gt_ever = gt_hot.any(axis=0)
        lumen_render_set = np.where((model_final | gt_ever) & off)[0]
        wall_idx = np.where(wall)[0]
        wound_idx = np.where(wound)[0]
        frame_idx = np.linspace(0, len(t) - 1, N_FRAMES).round().astype(int)

        def pts(idx):
            return [[round(float(pos[i, 0]), 4), round(float(pos[i, 1]), 4)] for i in idx]

        ei_t = d.edge_index
        wall_f = torch.tensor(wall.astype(np.float32))
        wound_f = torch.tensor(wound.astype(np.float32))
        off_f = torch.tensor(off.astype(np.float32))
        wall_hop_ref = torch.tensor(wall)

        def dscore(pred_hot, gt_hot_t, domain):
            gt_has = (gt_hot_t & domain).sum() > 0
            if not gt_has:
                return float("nan")
            return domain_score(pred_hot, gt_hot_t, ei_t, domain, wall)

        score_wall = [dscore(model_hot[i], gt_hot[i], wall) for i in range(len(t))]
        # NOT scored on raw `wound` (mask_wound / COMSOL sel1): that domain is 100% GT clot
        # on every vessel, so it can't discriminate a model from the ungated law -- see
        # wound_region_masks() in src/clot_ml/wound.py. `region` (boundary + lumen out to
        # WOUND_REGION_HOPS) is the domain that actually contains a mix of clot/no-clot.
        score_wound = [dscore(model_hot[i], gt_hot[i], region) for i in range(len(t))]
        score_wound_off = [dscore(model_hot[i], gt_hot[i], wound_lumen) for i in range(len(t))]
        score_off = [dscore(model_hot[i], gt_hot[i], off) for i in range(len(t))]

        out[anchor] = {
            "model": model_name,
            "flow": "gt",
            "outlier": anchor in OUTLIER,
            "t_final": float(t[-1]),
            "n_wall": int(wall.sum()),
            "n_wound": int(wound.sum()),
            "bg": pts(bg),
            "wall_pos": pts(wall_idx),
            "wound_pos": pts(wound_idx),
            "lumen_pos": pts(lumen_render_set),
            "lumen_dist": [round(float(x), 3) for x in dist_norm[lumen_render_set]],
            "frame_t": [round(float(t[i]), 1) for i in frame_idx],
            "frame_gt_wall": [[bool(x) for x in (gt_hot[i] & wall)[wall_idx]] for i in frame_idx],
            "frame_model_wall": [[bool(x) for x in (model_hot[i] & wall)[wall_idx]] for i in frame_idx],
            "frame_gt_wound": [[bool(x) for x in (gt_hot[i] & wound)[wound_idx]] for i in frame_idx],
            "frame_model_wound": [[bool(x) for x in (model_hot[i] & wound)[wound_idx]] for i in frame_idx],
            "frame_gt_lumen": [[bool(x) for x in (gt_hot[i] & off)[lumen_render_set]] for i in frame_idx],
            "frame_model_lumen": [[bool(x) for x in (model_hot[i] & off)[lumen_render_set]] for i in frame_idx],
            "score_t": [round(float(x), 1) for x in t],
            "score_wall": [None if v != v else round(float(v), 4) for v in score_wall],
            "score_wound": [None if v != v else round(float(v), 4) for v in score_wound],
            "score_wound_off": [None if v != v else round(float(v), 4) for v in score_wound_off],
            "score_off": [None if v != v else round(float(v), 4) for v in score_off],
        }
        print(f"{anchor}{' [OUTLIER]' if anchor in OUTLIER else ''}: "
              f"wall={wall.sum()} wound={wound.sum()} lumen_render={len(lumen_render_set)}  "
              f"final wall={score_wall[-1]:.3f}  wound={score_wound[-1]:.3f}  "
              f"wound_off={score_wound_off[-1]:.3f}  off={score_off[-1]:.3f}")

    out_path = Path("outputs/clot_ml_v0_temporal_data.json")
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
