"""Score the wound complement against `clot_gnn_v4` alone, on the metric of record.

Domains are reported separately because they answer different questions:

    wall       healthy wall -- v4's territory, must be UNCHANGED; the regression check
    wnd        the wound boundary itself (COMSOL ``sel1``)
    w_reg      WOUND REGION: every node within 8 hops of the wound, boundary and lumen
    w_lum      the LUMEN part of that region -- the clot the wound pushes into the flow
    far        off-boundary and beyond 8 hops: everything the wound did not cause
    full       the deliverable

**Do not read ``wnd`` as a score.** The wound boundary is 100% GT clot on every vessel, so
any model that commits the patch reads 1.0000 there and the ungated law does that for free.
It is a COVERAGE diagnostic. ``w_reg`` and ``w_lum`` are the scores -- they carry real
negatives (positive rate 0.19 / 0.19 / 0.33 and 0.10 / 0.10 / 0.25 respectively).

Both ``mean-over-time`` and the ``final`` time point are quoted, per AGENTS.md -- they
disagree and the last point is the fully-formed clot a reader acts on.

The BASELINE arm is pinned to ``clot_gnn_v4`` and is NOT read from the locked pointer -- once
``clot_gnn_v4w`` ships the pointer resolves to the wound-capable model, and a pointer-followed
baseline would already contain the arm under test.

The wound arm is scored **leave-one-vessel-out**: each vessel is predicted with the
``(G_pre, G_post)`` fitted on the other two, read from ``outputs/clot_ml/wound_rate/lovo.json``.
Quoting the all-three refit here would be a selection leak of exactly the kind
docs/PHASE10_V4.md 1 removed from v3.

Usage:
    python scripts/eval_wound_complement.py
    python scripts/eval_wound_complement.py --stems wound_comsol001
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
# Repo root by marker, not by depth: this file may move between
# scripts/ and scripts/<subdir>/ without silently resolving one level off.
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.evaluate import domain_score, f1  # noqa: E402
from src.clot_ml.locked import load_temporal_v4, predict_temporal_v4
from src.clot_ml.wound import (
    G_POST0, G_PRE0, WOUND_REGION_HOPS, compose_with_v4, predict_wound_series, prepare_vessel,
    solid_mask, wound_mask, wound_region_masks,
)
from src.config import BiochemConfig, PhysicsConfig

GRAPH_DIR = Path("data/processed/graphs_biochem_anchors")
LOVO = Path("outputs/clot_ml/wound_rate/lovo.json")
WOUND_STEMS = ("wound_comsol001", "wound_comsol002", "wound_comsol003")
#: Column order. ``wnd`` sits between the two domains it is confused with, so the table
#: itself shows it is not the score.
DOM = ("wall", "wnd", "w_reg", "w_lum", "far", "full")


def gt_series(data, phys, times) -> dict:
    from src.core_physics.t0_mu_physics import gt_clot_phi_at_time

    return {int(ti): gt_clot_phi_at_time(data, int(ti), phys).numpy() > 0.5 for ti in times}


def score_domains(pred: np.ndarray, gt: np.ndarray, ei, wall_for_hops: np.ndarray,
                  domains: dict) -> dict:
    out = {}
    for name, dom in domains.items():
        out[name] = domain_score(pred, gt, ei, dom, wall_for_hops)
        out[name + "_f1"] = f1(pred & dom, gt & dom)
    return out


def mean_over_time(series: dict, gts: dict, ei, wall_for_hops, domains: dict) -> dict:
    acc: dict[str, list] = {}
    for ti, m in series.items():
        row = score_domains(m, gts[ti], ei, wall_for_hops, domains)
        for k, v in row.items():
            if v == v:
                acc.setdefault(k, []).append(v)
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    # An empty domain (a no-wound vessel's `wound`) scores nan at every time and drops out
    # of `acc` entirely; keep the key so the table stays rectangular.
    for name in domains:
        out.setdefault(name, float("nan"))
        out.setdefault(name + "_f1", float("nan"))
    return out


def lovo_constants(stem: str) -> tuple[float, float, str]:
    """The (G_pre, G_post) fitted WITHOUT this vessel."""
    if not LOVO.exists():
        return G_PRE0, G_POST0, "defaults (run scripts/train_wound_rate.py)"
    blob = json.loads(LOVO.read_text())
    folds = blob.get("folds") or {}
    if stem in folds:
        return float(folds[stem]["g_pre"]), float(folds[stem]["g_post"]), "LOVO"
    fa = blob.get("fitted_all", {})
    return float(fa.get("g_pre", G_PRE0)), float(fa.get("g_post", G_POST0)), "all-3 refit (LEAKY)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=list(WOUND_STEMS))
    ap.add_argument("--every", type=int, default=2, help="subsample the time grid for speed")
    ap.add_argument("--hops", type=int, default=WOUND_REGION_HOPS,
                    help="radius of the wound region, in mesh-graph hops (2 hops = 1 corner shell)")
    ap.add_argument("--base", default="clot_gnn_v4",
                    help="baseline artifact; PINNED so a repointed locked model cannot leak in")
    ap.add_argument("--lumen", default="shell", choices=("shell", "transport", "union", "recursive"),
                    help="how the off-boundary nodes are decided. 'shell' is the shipped "
                         "rule (crit/0.16 then a 4%% lag); 'transport' replaces BOTH "
                         "constants with COMSOL's own operator (C1); 'union' is shell OR "
                         "transport, which is monotone and therefore the safe first read.")
    ap.add_argument("--trigger", default="self", choices=("self", "wall", "oracle", "model"),
                    help="what may open the two-regime gate; 'oracle' is a ceiling, not a model")
    args = ap.parse_args()

    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    # The baseline is pinned to `clot_gnn_v4` rather than followed from the locked pointer.
    # Once `clot_gnn_v4w` ships, the pointer resolves to the wound-capable model and the
    # "alone" arm would silently become v4w -- a baseline already containing the arm under
    # test, which reads as "the complement does nothing".
    bundle = load_temporal_v4(name=args.base)
    print(f"[i] baseline: {args.base} (pinned, not read from the locked pointer)\n")

    rows: dict[str, dict[str, dict]] = {}
    for stem in args.stems:
        data = torch.load(GRAPH_DIR / f"{stem}.pt", map_location="cpu", weights_only=False)
        T = int(data.y.shape[0])
        times = sorted(set(list(range(0, T, args.every)) + [T - 1]))
        ei = torch.tensor(data.edge_index.detach().cpu().numpy())
        wnd, solid = wound_mask(data), solid_mask(data)
        wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
        region, lumen, far = wound_region_masks(data, k_hops=args.hops)
        domains = {"wall": wall, "wnd": wnd, "w_reg": region, "w_lum": lumen,
                   "far": far, "full": np.ones_like(wall)}

        gts = gt_series(data, phys, times)
        base = predict_temporal_v4(bundle, data, times, flow="gt")
        V = prepare_vessel(data, bio, flow="gt")

        g_pre, g_post, src = lovo_constants(stem)
        base_onset = base.get("onset")
        if base_onset is not None:
            base_onset = np.asarray(base_onset, dtype=np.float64).copy()
            from src.clot_ml.temporal import ode_trajectory
            from src.core_physics.physics_wall_model import first_crossing
            traj_stall, _ = ode_trajectory(data, bio, flow="gt", stall=True, wound_source=True)
            onset_stall = first_crossing(traj_stall, float(bio.viscosity_mat_crit))
            wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
            stall_ign = (onset_stall >= 0) & wall
            update = stall_ign & ((base_onset < 0) | (onset_stall < base_onset))
            base_onset[update] = onset_stall[update]
            
        arms = {
            f"{args.base} alone": base,
            "v4 + wound physics (G=1)": compose_with_v4(
                base, predict_wound_series(data, bio, times, g_pre=1.0, g_post=1.0,
                                           prepared=V, base_onset=base_onset), times, data, bio),
            f"v4 + wound two-regime [{src}, trig={args.trigger}, lum={args.lumen}]":
                compose_with_v4(
                    base, predict_wound_series(data, bio, times, g_pre=g_pre, g_post=g_post,
                                               prepared=V, trigger=args.trigger,
                                               lumen=args.lumen, base_onset=base_onset), times, data, bio),
        }
        print("=" * 118)
        print(f"{stem}   T={T}  wound={int(wnd.sum())}  healthy wall={int(wall.sum())}  "
              f"G_pre={g_pre:.2f} G_post={g_post:.2f} ({src})")
        gt_fin = gts[times[-1]]
        print(f"    region = {int(region.sum())} nodes within {args.hops} hops, "
              f"{int(lumen.sum())} of them lumen | GT+ rate: w_reg {gt_fin[region].mean():.3f}, "
              f"w_lum {gt_fin[lumen].mean():.3f}, wnd {gt_fin[wnd].mean():.3f}")
        print(f"  {'arm':38s}" + "".join(f"{k:>8s}" for k in DOM)
              + "   |" + "".join(f"{k:>8s}" for k in DOM))
        rows[stem] = {}
        for tag, out in arms.items():
            fin = score_domains(out["series"][times[-1]], gt_fin, ei, solid, domains)
            mot = mean_over_time(out["series"], gts, ei, solid, domains)
            rows[stem][tag] = dict(final=fin, mean=mot)
            print(f"  {tag:38s}" + "".join(f"{fin[k]:8.4f}" for k in DOM)
                  + "   |" + "".join(f"{mot[k]:8.4f}" for k in DOM))

    if len(rows) > 1:
        print("\n" + "=" * 126)
        print(f"{'COHORT MEAN (n=%d)' % len(rows):40s}" + "".join(f"{k:>8s}" for k in DOM)
              + "   |" + "".join(f"{k:>8s}" for k in DOM))
        print(f"{'':40s}{'-------- FINAL --------':^48s}   {'--- MEAN OVER TIME ---':^48s}")
        for tag in next(iter(rows.values())):
            def mean(kind_, key):
                return float(np.nanmean([rows[s][tag][kind_][key] for s in rows]))
            print(f"  {tag:38s}" + "".join(f"{mean('final', k):8.4f}" for k in DOM)
                  + "   |" + "".join(f"{mean('mean', k):8.4f}" for k in DOM))
        print("\n[i] 'wall' must be identical across arms -- the complement never touches the"
              " healthy wall. Any drift there is a bug, not a result.")
        print("[i] 'wnd' is COVERAGE, not skill: that domain is 100% GT clot on every vessel,"
              " so any model that commits the patch reads 1.0. Read w_reg and w_lum.")


if __name__ == "__main__":
    main()
