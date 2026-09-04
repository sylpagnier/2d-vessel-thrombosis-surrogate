"""Out-of-fold clot_ml_0 trajectories for publication figures.

Publication clot panels must use strict nested-CV masks from
``eval_strict_temporal.py --save-oof-series``, not the promoted full-pool
``predict_clot_ml_0`` artifact (in-sample on training vessels) and not
``flow=pred`` surrogate flow (collapses wall scores).
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval_wound_complement import gt_series
from src.clot_ml.data import off_domain, wall_domain
from src.clot_ml.locked import build_sample
from src.clot_ml.wound import has_wound, solid_mask, wound_region_masks
from src.config import BiochemConfig, PhysicsConfig
from src.evaluation.clot_relaxed_metrics import (
    clot_score_from_deploy_dict,
    compute_clot_relaxed_metrics,
    metrics_to_deploy_prefix,
)


@dataclass(frozen=True)
class OofVessel:
    stem: str
    masks: np.ndarray  # [T_oof, N] bool
    times: np.ndarray  # simulation indices
    fold: int
    base_train_count: int
    temporal_train_count: int
    flow: str


@dataclass(frozen=True)
class OofArchive:
    path: Path
    flow: str
    vessels: dict[str, OofVessel]

    def get(self, stem: str) -> OofVessel | None:
        return self.vessels.get(stem)


def oof_series_path(cfg) -> Path:
    path = getattr(cfg, "oof_series_path", None)
    if path is None:
        return REPO / "outputs" / "publication" / "data" / "clot_ml_0_oof_series.npz"
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def _meta(z: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "meta" not in z:
        raise ValueError("series archive has no metadata; run eval_strict_temporal.py --save-oof-series")
    return json.loads(str(z["meta"][0]))


def load_oof_archive(path: Path) -> OofArchive:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing OOF series: {path}")

    with np.load(path, allow_pickle=False) as z:
        meta = _meta(z)
        flow = str(meta.get("flow", "gt"))
        vessels: dict[str, OofVessel] = {}
        for stem in meta.get("vessels", []):
            stem = str(stem)
            prov_key, masks_key, times_key = (
                f"provenance|{stem}",
                f"masks|{stem}",
                f"times|{stem}",
            )
            if any(k not in z for k in (prov_key, masks_key, times_key)):
                continue
            provenance = json.loads(str(z[prov_key][0]))
            if provenance.get("held_out") != stem or stem in provenance.get("base_train", []):
                raise RuntimeError(f"{stem}: OOF provenance does not prove base-model exclusion")
            vessels[stem] = OofVessel(
                stem=stem,
                masks=np.asarray(z[masks_key], dtype=bool),
                times=np.asarray(z[times_key], dtype=np.int64),
                fold=int(provenance["fold"]),
                base_train_count=len(provenance.get("base_train", [])),
                temporal_train_count=len(provenance.get("temporal_train", [])),
                flow=flow,
            )

    if not vessels:
        raise ValueError(f"{path}: no vessels in OOF archive")
    return OofArchive(path=path, flow=flow, vessels=vessels)


def score_deploy(pred: np.ndarray, gt: np.ndarray, edge_index: torch.Tensor,
                 wall: np.ndarray, domain: np.ndarray) -> float:
    pred_d = torch.tensor((pred & domain).astype(np.float32))
    gt_d = torch.tensor((gt & domain).astype(np.float32))
    m = compute_clot_relaxed_metrics(
        pred_d, gt_d, edge_index, wall_mask=torch.tensor(wall))
    return float(clot_score_from_deploy_dict(metrics_to_deploy_prefix(m)))


def build_vessel_figure_data(
    stem: str,
    oof: OofVessel,
    *,
    data=None,
) -> dict:
    """Build publication plot payload for one OOF vessel."""
    if data is None:
        from scripts.publication.utils import get_pack_path

        data = torch.load(get_pack_path(stem), map_location="cpu", weights_only=False)
    data.graph_stem = stem

    bio = BiochemConfig(phase="biochem")
    phys = PhysicsConfig(phase="biochem")
    # The archive's own flow decides how the sample is built, and a SOLVED flow has to be
    # solved first: `features.build_features` reads `data.u0_pred`, which only the FEM solve
    # writes.  This was latent while the archive mislabelled itself as `gt`
    # (eval_strict_temporal hardcoded it); with the label fixed, every consumer that replays
    # the archive needs the field the archive was actually built on.
    if oof.flow == "fem" and not hasattr(data, "u0_pred"):
        from src.clot_ml.v0 import solve_fem_into_pack

        solve_fem_into_pack(data)
    S = build_sample(data, bio, flow=oof.flow, variant="v4")
    wall = np.asarray(S["wall"], dtype=bool)
    solid = solid_mask(data)
    off = off_domain(S)
    edge_index = data.edge_index.detach().cpu()

    times = [int(t) for t in oof.times.tolist()]
    gts = gt_series(data, phys, times)

    is_wound = has_wound(data)
    wound_doms = None
    if is_wound:
        wound_doms = dict(zip(("region", "lumen", "far"), wound_region_masks(data)))

    frames: dict[int, dict] = {}
    for i, t in enumerate(times):
        gt_b = np.asarray(gts[t], dtype=bool)
        pred_b = np.asarray(oof.masks[i], dtype=bool)
        gt_phi = gt_b.astype(np.float64)
        pred_phi = pred_b.astype(np.float64)
        frames[t] = {
            "gt_phi": gt_phi,
            "pred_phi": pred_phi,
            "gt_mask": gt_b,
            "pred_mask": pred_b,
        }

    return dict(
        pos=data.x[:, 0:2].numpy(),
        wall=wall,
        times=times,
        is_wound=is_wound,
        out_of_fold=True,
        fold=oof.fold,
        flow=oof.flow,
        base_train_count=oof.base_train_count,
        temporal_train_count=oof.temporal_train_count,
        frames=frames,
        _score_ctx=dict(
            edge_index=edge_index,
            wall=wall,
            off=off,
            solid=solid,
            wound_doms=wound_doms,
        ),
    )


def metrics_rows_for_vessel(stem: str, payload: dict) -> list[dict]:
    """Expand OOF payload into per-timestep metric rows."""
    ctx = payload["_score_ctx"]
    edge_index = ctx["edge_index"]
    wall = ctx["wall"]
    off = ctx["off"]
    wound_doms = ctx["wound_doms"]
    rows: list[dict] = []

    for t in payload["times"]:
        fd = payload["frames"][t]
        pred_b = fd["pred_mask"]
        gt_b = fd["gt_mask"]
        row = {
            "vessel": stem,
            "time": t,
            "is_wound": payload["is_wound"],
            "out_of_fold": True,
            "fold": payload["fold"],
            "flow": payload["flow"],
        }
        if payload["is_wound"] and wound_doms is not None:
            row["wall"] = score_deploy(pred_b, gt_b, edge_index, wall, wall)
            for key, dom in (("w_reg", wound_doms["region"]),
                             ("w_lum", wound_doms["lumen"]),
                             ("far", wound_doms["far"])):
                row[key] = score_deploy(pred_b, gt_b, edge_index, wall, dom)
        else:
            row["wall"] = score_deploy(pred_b, gt_b, edge_index, wall, wall)
            row["off"] = score_deploy(pred_b, gt_b, edge_index, wall, off)
        rows.append(row)
    return rows


def export_oof_series(cfg) -> Path:
    """Run eval_strict_temporal.py to write the OOF NPZ (canonical v5 ensemble)."""
    out = oof_series_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    arms = getattr(cfg, "oof_arms", ("v5a,v5b,v5c",))
    cache = getattr(cfg, "oof_cache", "gt")
    head_seeds = int(getattr(cfg, "oof_head_seeds", 4))
    set_masks = getattr(cfg, "oof_set_masks", "outputs/v4_set_masks.npz")

    cmd = [
        sys.executable,
        str(REPO / "scripts" / "eval_strict_temporal.py"),
        "--arms", *arms,
        "--cache", cache,
        "--head-seeds", str(head_seeds),
        "--owner-lag", "--learn-lag", "--lag-anchor", "ode",
        "--two-stage",
        "--set-masks", str(REPO / set_masks if not Path(set_masks).is_absolute() else set_masks),
        "--save-oof-series", str(out),
    ]
    print("[i] Exporting OOF series for publication figures")
    print("    " + " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True)
    return out


def ensure_oof_series(cfg, *, regenerate: bool = False) -> Path:
    path = oof_series_path(cfg)
    if regenerate or not path.is_file():
        export_oof_series(cfg)
    return path
