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
from src.biochem_gnn.wall_cohort_constants import WOUND_LOVO_COHORT  # noqa: E402
from src.utils.paths import anchor_packs_dir

import argparse
import json
from pathlib import Path

import numpy as np
import torch


from src.clot_ml.evaluate import gt_series, score_domains  # noqa: E402
from src.clot_ml.locked import load_temporal_v4, predict_temporal_v4
from src.clot_ml.wound import (
    G_POST0, G_PRE0, WOUND_REGION_HOPS, compose_with_v4, predict_wound_series, prepare_vessel,
    solid_mask, wound_mask, wound_region_masks,
)
from src.config import BiochemConfig, PhysicsConfig

GRAPH_DIR = anchor_packs_dir()
LOVO = Path("outputs/clot_ml/wound_rate/lovo.json")
#: The LOVO subset, not the full cohort -- see wall_cohort_constants.
WOUND_STEMS = WOUND_LOVO_COHORT
#: Column order. ``wnd`` sits between the two domains it is confused with, so the table
#: itself shows it is not the score.
DOM = ("wall", "wnd", "w_reg", "w_lum", "far", "full")


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


def lovo_constants(stem: str, lovo_path: Path | None = None) -> tuple[float, float, str]:
    """The (G_pre, G_post) fitted WITHOUT this vessel."""
    LOVO = lovo_path or globals()["LOVO"]
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
    ap.add_argument("--save-series", default="",
                    help="write per-timestep masks, GT and per-domain scores for the SHIPPED "
                         "arm to this .npz -- the wound counterpart of the OOF series archive "
                         "(docs/publication/EXPERIMENTS.md E7). Held out leave-one-vessel-out, "
                         "NOT out-of-fold; the file records that.")
    ap.add_argument("--save-summary", default="",
                    help="write every arm's cohort means to this .json so ledger rows can "
                         "resolve against a file rather than a transcribed console number")
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
    ap.add_argument("--flow", default="gt", help="flow source for the rollout. 'fem' is the "
                    "DEPLOYED field; pair it with the matching --lovo file, because constants "
                    "fitted on one field and applied to the other are a train/test mismatch.")
    ap.add_argument("--lovo", default="", help="path to the LOVO constants json. Default is the "
                    "n=3 GT-flow fit (outputs/clot_ml/wound_rate/lovo.json); the n=6 FEM fit is "
                    "outputs/clot_ml/wound_rate_fem/lovo.json.")
    args = ap.parse_args()

    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    # The baseline is pinned to `clot_gnn_v4` rather than followed from the locked pointer.
    # Once `clot_gnn_v4w` ships, the pointer resolves to the wound-capable model and the
    # "alone" arm would silently become v4w -- a baseline already containing the arm under
    # test, which reads as "the complement does nothing".
    bundle = load_temporal_v4(name=args.base)
    print(f"[i] baseline: {args.base} (pinned, not read from the locked pointer)\n")

    rows: dict[str, dict[str, dict]] = {}
    SERIES: dict[str, dict] = {}
    for stem in args.stems:
        data = torch.load(GRAPH_DIR / f"{stem}.pt", map_location="cpu", weights_only=False)
        data.graph_stem = stem   # keys the evaluation label (gt_series -> mirror_branch.eval_gt)
        T = int(data.y.shape[0])
        times = sorted(set(list(range(0, T, args.every)) + [T - 1]))
        ei = torch.tensor(data.edge_index.detach().cpu().numpy())
        wnd, solid = wound_mask(data), solid_mask(data)
        wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
        region, lumen, far = wound_region_masks(data, k_hops=args.hops)
        domains = {"wall": wall, "wnd": wnd, "w_reg": region, "w_lum": lumen,
                   "far": far, "full": np.ones_like(wall)}

        gts = gt_series(data, phys, times)
        base = predict_temporal_v4(bundle, data, times, flow=args.flow)
        V = prepare_vessel(data, bio, flow=args.flow)

        g_pre, g_post, src = lovo_constants(stem, Path(args.lovo) if args.lovo else None)
        base_onset = base.get("onset")
        if base_onset is not None:
            base_onset = np.asarray(base_onset, dtype=np.float64).copy()
            from src.clot_ml.temporal import ode_trajectory
            from src.core_physics.physics_wall_model import first_crossing
            traj_stall, _ = ode_trajectory(data, bio, flow=args.flow, stall=True,
                                           wound_source=True)
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

        if args.save_series:
            # The SHIPPED arm's trajectory, plus GT and a per-timestep score in each domain --
            # the wound counterpart of `clot_ml_0_oof_series.npz`, which carries no wound
            # vessel (docs/publication/EXPERIMENTS.md E7).
            #
            # PROTOCOL, and it must travel with the file: this is NOT out-of-fold. The GNN base
            # never trained on ANY wound vessel (no wound pack is in the feature cache), and the
            # two complement scalars are leave-one-vessel-out. That is a stronger hold-out than
            # the intact cohort's fold split, but it is a DIFFERENT protocol and a figure that
            # puts the two side by side has to say so.
            shipped_tag = [t for t in arms if "two-regime" in t][0]
            ser = arms[shipped_tag]["series"]
            per_time = {k: [] for k in DOM}
            for ti in times:
                sc = score_domains(ser[ti], gts[ti], ei, solid, domains)
                for k in DOM:
                    per_time[k].append(float(sc.get(k, np.nan)))
            SERIES[stem] = dict(
                masks=np.stack([np.asarray(ser[ti], dtype=bool) for ti in times]),
                gt=np.stack([np.asarray(gts[ti], dtype=bool) for ti in times]),
                times=np.asarray(times, dtype=np.int32),
                scores={k: np.asarray(v, dtype=np.float64) for k, v in per_time.items()},
                domains={k: np.asarray(v, dtype=bool) for k, v in domains.items()},
                g_pre=float(g_pre), g_post=float(g_post), const_source=src, arm=shipped_tag,
            )

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

    if args.save_summary and rows:
        # Every arm's cohort means, straight from `rows`, so a ledger row can resolve against
        # a file instead of a transcribed console number. The baseline arm matters as much as
        # the shipped one: the contrast IS the claim in STORY.md Leg 0.2.
        tags = list(next(iter(rows.values())))
        shipped = [t for t in tags if "two-regime" in t][0]
        base_tag = [t for t in tags if "alone" in t][0]
        phys_tag = [t for t in tags if "physics" in t][0]

        def cohort(tag: str, kind: str) -> dict:
            return {k: round(float(np.nanmean([rows[s][tag][kind][k] for s in rows])), 4)
                    for k in DOM}

        summ = dict(
            protocol=("leave-one-vessel-out on (G_pre, G_post); the GNN base never trained on "
                      "ANY wound vessel, so these vessels are unseen entirely. NOT out-of-fold "
                      "-- a different and stronger hold-out than the intact cohort's folds."),
            metric="clot_ml.evaluate.score_domains == BATC_0 (unadjusted), NOT the BATC setting",
            base=args.base, flow=args.flow, lovo=args.lovo or str(LOVO), every=args.every,
            lumen_rule=args.lumen, trigger=args.trigger,
            n_vessels=len(rows), vessels=sorted(rows),
            arm=shipped,
            final=cohort(shipped, "final"), mean_over_time=cohort(shipped, "mean"),
            baseline_arm=base_tag,
            baseline_final=cohort(base_tag, "final"),
            baseline_mean_over_time=cohort(base_tag, "mean"),
            physics_only_arm=phys_tag,
            physics_only_final=cohort(phys_tag, "final"),
            per_vessel_final={s: {k: round(float(rows[s][shipped]["final"][k]), 4) for k in DOM}
                              for s in rows},
        )
        sp = Path(args.save_summary)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summ, indent=2), encoding="utf-8")
        print(f"[i] wrote {sp}")

    if args.save_series and SERIES:
        out = Path(args.save_series)
        out.parent.mkdir(parents=True, exist_ok=True)
        blob: dict[str, np.ndarray] = {}
        for stem, d in SERIES.items():
            blob[f"masks|{stem}"] = d["masks"]
            blob[f"gt|{stem}"] = d["gt"]
            blob[f"times|{stem}"] = d["times"]
            for k, v in d["scores"].items():
                blob[f"score|{stem}|{k}"] = v
            for k, v in d["domains"].items():
                blob[f"domain|{stem}|{k}"] = v
        blob["meta"] = np.array([json.dumps({
            "schema_version": 1,
            "purpose": "held-out wound trajectories for visualisation and BATC-over-time",
            "protocol": (
                "NOT out-of-fold. The GNN base never trained on any wound vessel -- no wound "
                "pack is in the feature cache -- and the two complement scalars (G_pre, G_post) "
                "are leave-one-vessel-out. A stronger hold-out than the intact cohort's fold "
                "split, but a DIFFERENT protocol: any figure showing both must say so."),
            "metric": ("per-domain score from clot_ml.evaluate.score_domains == BATC_0 "
                       "(unadjusted). NOT the BATC setting the CV tables report."),
            "arm": next(iter(SERIES.values()))["arm"],
            "base": args.base,
            "flow": args.flow,
            "lovo": args.lovo or str(LOVO),
            "every": args.every,
            "lumen_rule": args.lumen,
            "trigger": args.trigger,
            "vessels": sorted(SERIES),
            "constants": {s: {"g_pre": d["g_pre"], "g_post": d["g_post"],
                              "source": d["const_source"]} for s, d in SERIES.items()},
        })])
        np.savez_compressed(out, **blob)
        print(f"\n[i] wrote {out}  ({len(SERIES)} wound vessels, "
              f"{sum(len(d['times']) for d in SERIES.values())} timesteps total)")


if __name__ == "__main__":
    main()
