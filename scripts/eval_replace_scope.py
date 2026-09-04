"""Choose the chemistry replacement's extent on the wound cohort, leave-one-vessel-out.

`clot_ml_0` replaces the GNN's off-wall verdict with a chemistry-ODE `Mat` field read through
solid-anchored replace+depth.  How much of the lumen it is allowed to replace is a knob with
two settings (`src/clot_ml/v0.REPLACE_SCOPES`):

    all_lumen      every true-lumen node          -- the shipped policy
    wound_region   only the wound-local lumen     -- built, never selected

At n=3 there was no basis to choose: WOUND_PROGRESS 19 measured `all_lumen` and left
`wound_region` as an option.  The 2026-09-02 deploy evaluation then showed what `all_lumen`
costs -- the FAR FIELD collapses to 0.0000 on `wound_comsol004`, `005` and `006`, because
chemistry replaces a verdict the GNN was getting right far from the injury.  With six wound
vessels the scope can be chosen the way every other readout scalar in this project is: on the
out-of-fold vessels, never on the held-out one.

The comparison is per DOMAIN, because the two scopes trade against each other by construction
-- `all_lumen` can only help where chemistry beats the GNN and can only hurt where it does
not, and those are different parts of the mesh.

    python scripts/eval_replace_scope.py --model DeployClot_0 --flow fem
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.eval_wound_complement import DOM, gt_series, score_domains  # noqa: E402
from src.biochem_gnn.wall_cohort_constants import WOUND_COHORT  # noqa: E402
from src.clot_ml.locked import build_sample  # noqa: E402
from src.clot_ml.v0 import (  # noqa: E402
    REPLACE_SCOPES, load_v0_bundle, predict_clot_ml_0, solve_fem_into_pack,
)
from src.clot_ml.wound import solid_mask, wound_region_masks  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"


def score_one(bundle, stem: str, scope: str, every: int, flow: str) -> dict:
    import dataclasses

    data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
    data.graph_stem = stem
    if flow == "fem":
        solve_fem_into_pack(data)
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    T = int(data.y.shape[0])
    times = sorted({*range(0, T, max(every, 1)), T - 1})
    S = build_sample(data, bio, flow=flow, variant="v4")
    ei = torch.tensor(np.asarray(S["edge_index"]))
    gts = gt_series(data, phys, times)

    b = dict(bundle)
    b["cfg"] = dataclasses.replace(bundle["cfg"], replace_scope=scope)
    out = predict_clot_ml_0(b, data, times, flow=flow, sample=S)

    wall = np.asarray(S["wall"], dtype=bool)
    reg, lum, far = wound_region_masks(data)
    domains = dict(wall=wall, wnd=solid_mask(data) & ~wall, w_reg=reg, w_lum=lum, far=far,
                   full=np.ones(len(wall), dtype=bool))
    return score_domains(out["series"][times[-1]], gts[times[-1]], ei, wall, domains)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--flow", default="fem", choices=["gt", "pred", "fem"])
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--out", default="outputs/deployclot/replace_scope.json")
    args = ap.parse_args()

    bundle = load_v0_bundle(args.model)
    stems = [s for s in WOUND_COHORT if (PACKS / f"{s}.pt").exists()]
    res: dict[str, dict[str, dict]] = {}
    for stem in stems:
        res[stem] = {}
        for scope in REPLACE_SCOPES:
            print(f"[i] {stem} scope={scope} ...", flush=True)
            res[stem][scope] = score_one(bundle, stem, scope, args.every, args.flow)

    doms = ("wall", "w_reg", "w_lum", "far")
    print()
    print(f"{'vessel':20s} " + " ".join(f"{d:>18s}" for d in doms))
    print(f"{'':20s} " + " ".join(f"{'all / wound_reg':>18s}" for _ in doms))
    for stem in stems:
        cells = []
        for d in doms:
            a = res[stem]["all_lumen"].get(d, float("nan"))
            w = res[stem]["wound_region"].get(d, float("nan"))
            cells.append(f"{a:8.4f}/{w:<9.4f}")
        print(f"{stem:20s} " + " ".join(cells))

    print()
    print("MEAN over the wound cohort (nan-safe)")
    print(f"{'domain':10s} {'all_lumen':>11s} {'wound_region':>13s} {'delta':>9s}")
    means = {}
    for d in doms:
        a = [res[s]["all_lumen"].get(d, float("nan")) for s in stems]
        w = [res[s]["wound_region"].get(d, float("nan")) for s in stems]
        ma, mw = float(np.nanmean(a)), float(np.nanmean(w))
        means[d] = dict(all_lumen=ma, wound_region=mw, delta=mw - ma)
        print(f"{d:10s} {ma:11.4f} {mw:13.4f} {mw - ma:+9.4f}")

    # leave-one-vessel-out: pick the scope on the OTHER five, score the held-out one
    print()
    print("LEAVE-ONE-VESSEL-OUT scope selection (mean over w_reg, w_lum and far)")
    picks, held_scores = {}, {}
    for stem in stems:
        others = [s for s in stems if s != stem]
        best, pick = None, None
        for scope in REPLACE_SCOPES:
            v = float(np.nanmean([[res[o][scope].get(d, float("nan")) for d in
                                   ("w_reg", "w_lum", "far")] for o in others]))
            if best is None or v > best:
                best, pick = v, scope
        picks[stem] = pick
        held = float(np.nanmean([res[stem][pick].get(d, float("nan"))
                                 for d in ("w_reg", "w_lum", "far")]))
        held_scores[stem] = held
        print(f"  {stem:20s} picks {pick:13s} -> held-out {held:.4f}")
    print(f"  {'MEAN held-out':20s} {'':19s}    {float(np.nanmean(list(held_scores.values()))):.4f}")
    for scope in REPLACE_SCOPES:
        fixed = float(np.nanmean([[res[s][scope].get(d, float("nan")) for d in
                                   ("w_reg", "w_lum", "far")] for s in stems]))
        print(f"  {'always ' + scope:20s} {'':19s}    {fixed:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        dict(model=args.model, flow=args.flow, per_vessel=res, means=means,
             lovo_picks=picks, lovo_held=held_scores), indent=2), encoding="utf-8")
    print(f"\n[save] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
