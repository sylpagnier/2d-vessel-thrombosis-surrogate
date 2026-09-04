"""Wound A/B example: wound_comsol005 vs. its matched no-wound twin comsol048.

Same underlying vessel geometry (.nas), simulated once with the wound boundary
condition and once without -- confirmed by identical node bounding box. Predicts
with the CURRENTLY SHIPPED model (load_default -> clot_gnn_v6w), not the
strict-OOF ensemble the rest of this pipeline uses, because v6w is wound-aware
and the strict-OOF cohort excludes wound vessels entirely.

CAVEAT, and it belongs in the figure caption: neither vessel's held-out status
against the base GNN's own training set has been independently re-verified here
(only the wound-rate law's LOVO fit on 001/002/003 is confirmed not to include
either of these two). Treat this as an illustrative example, not a scored result
-- the wound section is frozen pending PUBLICATION_NOTES §7.0 Q1-3.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.publication.config import CONFIG, DATA_DIR  # noqa: E402
from scripts.publication.utils import get_pack_path  # noqa: E402
from src.clot_ml.data import off_domain  # noqa: E402
from src.clot_ml.evaluate import domain_score  # noqa: E402
from src.clot_ml.locked import build_sample, load_default, predict_default_series  # noqa: E402
from src.clot_ml.wound import has_wound, solid_mask, wound_region_masks  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PAIR = ("wound_comsol005", "comsol048")


def build_one(stem: str, bundle, kind: str, bio, phys) -> dict:
    data = torch.load(get_pack_path(stem), map_location="cpu", weights_only=False)
    data.graph_stem = stem

    S = build_sample(data, bio, flow="gt", variant="v4")
    wall = S["wall"]
    solid = solid_mask(data)
    is_wound = has_wound(data)

    T = data.y.shape[0]
    # Match horizon to what the wound run actually has (truncated sims are common
    # -- WOUND_PROGRESS.md 7); pick 4 roughly-even sampled steps within it.
    last = T - 1
    times = sorted({0, last // 3, (2 * last) // 3, last})

    pred = predict_default_series(bundle, kind, data, times, flow="gt", sample=S)
    series = pred["series"]
    ei = data.edge_index.detach().cpu()

    wound_doms = None
    if is_wound:
        wound_doms = dict(zip(("region", "lumen", "far"), wound_region_masks(data)))
    off = None if is_wound else off_domain(S)

    frames = {}
    scores = {}
    for t in times:
        gt_b = gt_clot_phi_at_time(data, int(t), phys).numpy() > 0.5
        pred_b = series[int(t)]
        frames[int(t)] = dict(gt_mask=gt_b, pred_mask=pred_b,
                              gt_phi=gt_b.astype(float), pred_phi=pred_b.astype(float))
        row = {"wall": domain_score(pred_b, gt_b, ei, wall, wall)}
        if is_wound:
            row["w_reg"] = domain_score(pred_b, gt_b, ei, wound_doms["region"], wall)
            row["w_lum"] = domain_score(pred_b, gt_b, ei, wound_doms["lumen"], wall)
        else:
            row["off"] = domain_score(pred_b, gt_b, ei, off, wall)
        scores[int(t)] = row

    return dict(
        pos=data.x[:, 0:2].numpy(), wall=wall, times=times, is_wound=is_wound,
        T_total=T, frames=frames, wound_doms=wound_doms, scores=scores,
        biochem_variant=getattr(data, "biochem_variant", "?"),
        source_mph=getattr(data, "source_mph", "?"),
    )


def main() -> None:
    print("[i] Generating wound A/B example data (wound_comsol005 vs. comsol048)")
    bundle, kind = load_default()
    print(f"    shipped model: kind={kind}")
    bio = BiochemConfig(phase="biochem")
    phys = PhysicsConfig(phase="biochem")

    for stem in PAIR:
        print(f"  -> {stem} ...")
        payload = build_one(stem, bundle, kind, bio, phys)
        print(f"     is_wound={payload['is_wound']}  T={payload['T_total']}  "
              f"times={payload['times']}  source={payload['source_mph']}")
        torch.save(payload, DATA_DIR / f"wound_ab_{stem}.pt")
        print(f"     [OK] Saved wound_ab_{stem}.pt")


if __name__ == "__main__":
    main()
