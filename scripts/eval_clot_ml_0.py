"""Compare ``clot_ml_0`` against the best past baseline (default: ``clot_gnn_v5w``).

Scores the metric of record (domain-restricted severity), not sklearn F1.  Times are
pack indices, not seconds.  Wound vessels use the WOUND_PROGRESS 13 domains
(``wall`` / ``w_reg`` / ``w_lum`` / ``far``); non-wound vessels use global
``wall`` / ``off`` (true lumen).  SEALED is never in the default list.

    python scripts/eval_clot_ml_0.py
    python scripts/eval_clot_ml_0.py --stems wound_comsol003 comsol012
    python scripts/eval_clot_ml_0.py --cohort
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval_wound_complement import (  # noqa: E402
    DOM, gt_series, mean_over_time, score_domains,
)
from src.clot_ml.data import eval_domains  # noqa: E402
from src.clot_ml.evaluate import domain_score  # noqa: E402
from src.clot_ml.locked import (  # noqa: E402
    build_sample, load_temporal_v4_wound, predict_temporal_v4_wound,
)
from src.clot_ml.v0 import load_v0_bundle, predict_clot_ml_0  # noqa: E402
from src.clot_ml.wound import has_wound, solid_mask, wound_region_masks  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.biochem_gnn.wall_cohort_constants import WOUND_AB_PAIR, WOUND_COHORT  # noqa: E402
from src.core_physics.wall_cohort_splits import CLOT_FREE, DEV, FIT, SEALED  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
DEFAULT_STEMS = tuple(WOUND_COHORT) + (
    "comsol012", "comsol020", "comsol032", "comsol041", "comsol044",
    WOUND_AB_PAIR[1],
)


def _times(data, every: int) -> list[int]:
    T = int(data.y.shape[0])
    grid = list(range(0, T, max(int(every), 1)))
    if grid[-1] != T - 1:
        grid.append(T - 1)
    return grid


def _cohort_stems() -> list[str]:
    skip = set(SEALED) | set(CLOT_FREE)
    out = []
    for a in list(FIT) + list(DEV):
        if a in skip:
            continue
        if (PACKS / f"{a}.pt").exists():
            out.append(a)
    for s in WOUND_COHORT:
        if (PACKS / f"{s}.pt").exists() and s not in out:
            out.append(s)
    return out


def _score_nowound(pred, gt, ei, S) -> dict:
    """Both metrics, on the same mask, always.

    `domain_score` (the DEPLOY metric) and `SeverityScorer` (what the strictly-nested CV
    table reports) are different numbers for the same prediction -- systematically 0.19-0.22
    apart off-wall on this cohort.  Reporting only the first is what let the SEALED read be
    quoted against a cross-validated severity number and manufacture a 0.22 "off-wall gap"
    that does not exist (docs/DEPLOYCLOT.md 22).  Emitting both costs one extra call and
    makes the mismatch impossible to make again.
    """
    from src.clot_ml.severity_metric import DEFAULT, SeverityScorer

    wall, off = eval_domains(S)
    vs = SeverityScorer(S["edge_index"], gt, len(S["wall"]), DEFAULT)
    return dict(
        wall=domain_score(pred, gt, ei, wall, wall),
        off=domain_score(pred, gt, ei, off, wall),
        wall_sev=vs.score(pred & wall, wall),
        off_sev=vs.score(pred & off, off),
    )


def _score_one(bundle_base, bundle_v0, stem: str, every: int, flow: str) -> dict:
    data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
    # Deploy packs carry neither `graph_stem` nor `path`, so nothing on the object says which
    # vessel it is.  `flow="fem"` has to find the mesh the pack was built from; without this it
    # cannot, and dies inside the rollout.
    if getattr(data, "graph_stem", None) is None:
        data.graph_stem = stem
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    times = _times(data, every)
    if flow == "fem":
        # `predict_clot_ml_0` no longer runs the FEM solve internally (doing it there left
        # the sample and the baseline on GT while the run was labelled `fem`).  Solve here so
        # that EVERY downstream call -- build_sample, baseline, v0 rollout -- sees the FEM field.
        # flow stays "fem"; features.py/temporal.py give it the GT treatment (hops=3, gain=1.0).
        from src.clot_ml.v0 import solve_fem_into_pack

        solve_fem_into_pack(data)
    S = build_sample(data, bio, flow=flow, variant="v4")
    ei = torch.tensor(np.asarray(S["edge_index"]))
    gts = gt_series(data, phys, times)

    print(f"  [{stem}] baseline ...", flush=True)
    base = predict_temporal_v4_wound(bundle_base, data, times, flow=flow, sample=S)
    print(f"  [{stem}] v0 ...", flush=True)
    v0 = predict_clot_ml_0(bundle_v0, data, times, flow=flow, sample=S)

    last = times[-1]
    gt_last = gts[last]
    row = dict(stem=stem, T=int(data.y.shape[0]), n_times=len(times), wound=has_wound(data))

    if has_wound(data):
        wall = np.asarray(S["wall"], dtype=bool)
        domains = dict(zip(("region", "lumen", "far"), wound_region_masks(data)))
        domains = dict(wall=wall, wnd=solid_mask(data) & ~wall,
                       w_reg=domains["region"], w_lum=domains["lumen"],
                       far=domains["far"], full=np.ones(len(wall), dtype=bool))
        b_fin = score_domains(base["series"][last], gt_last, ei, wall, domains)
        v_fin = score_domains(v0["series"][last], gt_last, ei, wall, domains)
        b_mot = mean_over_time(base["series"], gts, ei, wall, domains)
        v_mot = mean_over_time(v0["series"], gts, ei, wall, domains)
        for d in DOM:
            row[f"base_fin_{d}"] = b_fin.get(d, float("nan"))
            row[f"v0_fin_{d}"] = v_fin.get(d, float("nan"))
            row[f"base_mot_{d}"] = b_mot.get(d, float("nan"))
            row[f"v0_mot_{d}"] = v_mot.get(d, float("nan"))
    else:
        b_w, b_o, v_w, v_o = [], [], [], []
        b_ws, b_os, v_ws, v_os = [], [], [], []
        for ti in times:
            gt = gts[ti]
            sb = _score_nowound(base["series"][ti], gt, ei, S)
            sv = _score_nowound(v0["series"][ti], gt, ei, S)
            b_w.append(sb["wall"])
            b_o.append(sb["off"])
            v_w.append(sv["wall"])
            v_o.append(sv["off"])
            b_ws.append(sb["wall_sev"])
            b_os.append(sb["off_sev"])
            v_ws.append(sv["wall_sev"])
            v_os.append(sv["off_sev"])
        row["base_fin_wall"] = b_w[-1]
        row["v0_fin_wall"] = v_w[-1]
        row["base_fin_off"] = b_o[-1]
        row["v0_fin_off"] = v_o[-1]
        row["base_mot_wall"] = float(np.nanmean(b_w))
        row["v0_mot_wall"] = float(np.nanmean(v_w))
        row["base_mot_off"] = float(np.nanmean(b_o))
        row["v0_mot_off"] = float(np.nanmean(v_o))
        row["base_fin_wall_sev"] = b_ws[-1]
        row["v0_fin_wall_sev"] = v_ws[-1]
        row["base_fin_off_sev"] = b_os[-1]
        row["v0_fin_off_sev"] = v_os[-1]
    return row


def _fmt(x) -> str:
    if x is None or x != x:
        return "   nan"
    return f"{float(x):7.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0", default=None,
                    help="unified_v0 artifact; default follows the locked pointer")
    ap.add_argument("--baseline", default=None,
                    help="pinned past baseline; not the locked pointer")
    ap.add_argument("--stems", nargs="*", default=None)
    ap.add_argument("--cohort", action="store_true",
                    help="all FIT+DEV clot-carrying packs plus the 3 wound vessels")
    ap.add_argument("--every", type=int, default=4,
                    help="subsample the time grid (1 = every frame)")
    ap.add_argument("--flow", default="gt")
    ap.add_argument("--force-invalid", action="store_true",
                    help="score an artifact stamped metrics_invalid anyway; the output is "
                         "in-sample and must be labelled as such")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    stems = args.stems
    if args.cohort:
        stems = _cohort_stems()
    elif not stems:
        stems = [s for s in DEFAULT_STEMS if (PACKS / f"{s}.pt").exists()]
    if not stems:
        print("[ERR] no stems to score")
        return 1

    print(f"[i] baseline={args.baseline}  v0={args.v0}  n={len(stems)}  every={args.every}",
          flush=True)
    bundle_base = load_temporal_v4_wound(args.baseline)
    bundle_v0 = load_v0_bundle(args.v0)

    # A production artifact trains on the whole corpus, SEALED included.  Every number this
    # script could print for it would be in-sample, and an in-sample number that looks like
    # the validated one is the single most publishable-looking mistake available here.
    from src.clot_ml.v0 import metrics_invalid_reason

    why = metrics_invalid_reason(bundle_v0)
    if why and not args.force_invalid:
        print(f"[ERR] {args.v0} must not be scored: {why}\n"
              "      Score the validated sibling instead, or pass --force-invalid if you "
              "genuinely want an in-sample sanity number and will label it as one.",
              flush=True)
        return 2
    if why:
        print(f"[!] IN-SAMPLE ONLY -- {args.v0}: {why}", flush=True)

    rows = []
    for stem in stems:
        if not (PACKS / f"{stem}.pt").exists():
            print(f"  [skip] {stem}: pack missing", flush=True)
            continue
        rows.append(_score_one(bundle_base, bundle_v0, stem, args.every, args.flow))

    print()
    print("NON-WOUND (global wall / true-lumen off) -- DEPLOY metric "
          "(`evaluate.domain_score`); the severity metric the CV table reports is "
          "`*_sev` in the saved JSON and runs 0.19-0.22 HIGHER off-wall. Do not "
          "compare one against the other (docs/DEPLOYCLOT.md 22).")
    print(f"{'vessel':22s} {'B wall':>8s} {'v0 wall':>8s} {'dW':>7s} "
          f"{'B off':>8s} {'v0 off':>8s} {'dO':>7s}")
    nw = [r for r in rows if not r["wound"]]
    for r in nw:
        dw = r["v0_fin_wall"] - r["base_fin_wall"]
        do = r["v0_fin_off"] - r["base_fin_off"]
        print(f"{r['stem']:22s} {_fmt(r['base_fin_wall'])} {_fmt(r['v0_fin_wall'])} "
              f"{dw:+7.4f} {_fmt(r['base_fin_off'])} {_fmt(r['v0_fin_off'])} {do:+7.4f}")
    if nw:
        def _mean(key):
            vs = [r[key] for r in nw if r[key] == r[key]]
            return float(np.mean(vs)) if vs else float("nan")
        print(f"{'MEAN':22s} {_fmt(_mean('base_fin_wall'))} {_fmt(_mean('v0_fin_wall'))} "
              f"{_mean('v0_fin_wall')-_mean('base_fin_wall'):+7.4f} "
              f"{_fmt(_mean('base_fin_off'))} {_fmt(_mean('v0_fin_off'))} "
              f"{_mean('v0_fin_off')-_mean('base_fin_off'):+7.4f}")

    print()
    print("WOUND (final; w_reg / w_lum are the scores, wnd is coverage)")
    print(f"{'vessel':22s} {'B wall':>8s} {'v0 wall':>8s} "
          f"{'B w_reg':>8s} {'v0 w_reg':>8s} {'B w_lum':>8s} {'v0 w_lum':>8s} "
          f"{'B far':>8s} {'v0 far':>8s}")
    wd = [r for r in rows if r["wound"]]
    for r in wd:
        print(f"{r['stem']:22s} {_fmt(r.get('base_fin_wall'))} {_fmt(r.get('v0_fin_wall'))} "
              f"{_fmt(r.get('base_fin_w_reg'))} {_fmt(r.get('v0_fin_w_reg'))} "
              f"{_fmt(r.get('base_fin_w_lum'))} {_fmt(r.get('v0_fin_w_lum'))} "
              f"{_fmt(r.get('base_fin_far'))} {_fmt(r.get('v0_fin_far'))}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"[save] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
