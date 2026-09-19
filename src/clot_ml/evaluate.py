"""The metric of record, wired once so every arm is scored identically.

Domain-restricted ``deploy_clot_score`` exactly as ``scripts/eval_domain_targets.py``
computes it: zero the prediction and the GT outside the domain, then the canonical
relaxed score.  Targets: wall > 0.9, off-wall > 0.7.
"""
from __future__ import annotations

import numpy as np
import torch

from src.clot_ml.data import eval_domains
from src.clot_ml.mirror_branch import eval_gt
from src.evaluation.clot_relaxed_metrics import (
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

WALL_TARGET = 0.9
OFF_TARGET = 0.7


def domain_score(pred: np.ndarray, gt: np.ndarray, ei: torch.Tensor,
                 domain: np.ndarray, wall: np.ndarray) -> float:
    if int((gt & domain).sum()) == 0:
        return float("nan")
    dom = torch.tensor(domain.astype(np.float32))
    m = compute_clot_relaxed_metrics(
        torch.tensor(pred.astype(np.float32)) * dom,
        torch.tensor(gt.astype(np.float32)) * dom,
        ei, wall_mask=torch.tensor(wall))
    return float(clot_score_from_deploy_dict(metrics_to_deploy_prefix(m)))


def full_score(pred: np.ndarray, gt: np.ndarray, ei: torch.Tensor, wall: np.ndarray) -> float:
    m = compute_clot_relaxed_metrics(
        torch.tensor(pred.astype(np.float32)), torch.tensor(gt.astype(np.float32)),
        ei, wall_mask=torch.tensor(wall))
    return float(clot_score_from_deploy_dict(metrics_to_deploy_prefix(m)))


def strict_f1(pred: np.ndarray, gt: np.ndarray, domain: np.ndarray | None = None) -> float:
    """Node-exact F1 inside ``domain``: no tolerance, no grace. NaN when the domain holds no GT.

    The manuscript's "strict F1" -- the score BATC is contrasted with (figure 0.3) and reported
    beside it (coupling_frequency). One definition, shared by every script that quotes it.
    """
    if domain is not None:
        pred, gt = pred & domain, gt & domain
    n_p, n_g = int(pred.sum()), int(gt.sum())
    if n_g == 0:
        return float("nan")
    tp = int((pred & gt).sum())
    return 0.0 if n_p == 0 or tp == 0 else 2 * tp / (n_p + n_g)


def f1(pred: np.ndarray, gt: np.ndarray) -> float:
    if gt.sum() == 0:
        return float("nan")
    tp = int((pred & gt).sum())
    p, r = tp / max(int(pred.sum()), 1), tp / max(int(gt.sum()), 1)
    return 2 * p * r / max(p + r, 1e-9)


def score_vessel(pred: np.ndarray, S: dict, stem: str | None = None) -> dict:
    """``S`` is a sample dict from the cache; ``pred`` a boolean full-mesh mask.

    Pass ``stem`` so a mirror-branch vessel is scored on its flow branch (`mirror_branch.eval_gt`).
    """
    ei = torch.tensor(S["edge_index"])
    gt = eval_gt(stem, S["y"], S["pos"])
    # `off` is TRUE LUMEN (`~solid`), not `~wall` -- see `src/clot_ml/data.eval_domains`.
    # Identical on every no-wound pack; on a wound pack it keeps the wound's 100%-GT nodes
    # out of the off-wall score, which is the whole point of the A3 decision.
    wall, off = eval_domains(S)
    return dict(
        wall=domain_score(pred, gt, ei, wall, wall),
        off=domain_score(pred, gt, ei, off, wall),
        full=full_score(pred, gt, ei, wall),
        wall_f1=f1(pred & wall, gt & wall),
        off_f1=f1(pred & off, gt & off),
    )


def summarise(rows: dict[str, dict], anchors_fit, anchors_dev) -> dict:
    out = {}
    for split, anchors in (("fit", anchors_fit), ("dev", anchors_dev)):
        vals = {k: [] for k in ("wall", "off", "full", "wall_f1", "off_f1")}
        for a in anchors:
            if a not in rows:
                continue
            for k in vals:
                v = rows[a].get(k, float("nan"))
                if v == v:
                    vals[k].append(v)
        out[split] = {k: (float(np.mean(v)) if v else float("nan")) for k, v in vals.items()}
        out[split]["n"] = len(
            [a for a in anchors if a in rows and rows[a].get("wall", float("nan")) == rows[a].get("wall", float("nan"))])
    return out


def banner(tag: str, s: dict) -> str:
    f, d = s["fit"], s["dev"]
    return ("%-26s | FIT wall %.4f off %.4f full %.4f | DEV wall %.4f off %.4f full %.4f"
            % (tag, f["wall"], f["off"], f["full"], d["wall"], d["off"], d["full"]))


# --- helpers the deploy probe and the eval scripts share -------------------------
# These lived in `scripts/eval_clot_ml_0.py` and `scripts/eval_wound_complement.py`,
# which meant `src/utils/kinematics_deploy_probe.py` imported from a script.


def time_grid(data, every: int) -> list[int]:
    """Evaluation time indices: every ``every`` steps, always including the last."""
    n_times = int(data.y.shape[0])
    grid = list(range(0, n_times, max(int(every), 1)))
    if grid[-1] != n_times - 1:
        grid.append(n_times - 1)
    return grid


def gt_series(data, phys, times) -> dict:
    """EVALUATION ground-truth clot mask at each requested time index, on the model's flow branch.

    Mirrored for a registered mirror-branch vessel (`src/clot_ml/mirror_branch.py`), keyed by
    ``data.graph_stem`` -- set it before calling.  Not for training targets.
    """
    from src.core_physics.t0_mu_physics import gt_clot_phi_at_time

    stem, pos = getattr(data, "graph_stem", None), data.x[:, :2].detach().cpu().numpy()
    return {int(ti): eval_gt(stem, gt_clot_phi_at_time(data, int(ti), phys).numpy() > 0.5, pos)
            for ti in times}


def score_domains(pred: np.ndarray, gt: np.ndarray, ei, wall_for_hops: np.ndarray,
                  domains: dict) -> dict:
    """Per-domain relaxed score plus strict F1."""
    out = {}
    for name, dom in domains.items():
        out[name] = domain_score(pred, gt, ei, dom, wall_for_hops)
        out[name + "_f1"] = f1(pred & dom, gt & dom)
    return out
