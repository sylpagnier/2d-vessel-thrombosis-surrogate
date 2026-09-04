"""Why does off-wall land at 0.618 on SEALED when the cohort reads 0.836?

THIS SPENDS NOTHING.  The SEALED read was taken 2026-09-03 and those vessels are burnt.
Recomputing a deterministic prediction that was already made is not a second read; what
would be one is SELECTING anything on them, and nothing here does -- every rule this script
reports is scored, never fitted, on SEALED.

WHAT IS ALREADY RULED OUT (DEPLOYCLOT.md 20, and the runs that produced this script):

    the cut rule        seven label-free per-vessel rules, strictly nested: none beats the
                        shipped `resid` by more than noise, and the in-cohort headroom
                        against the per-vessel oracle is +0.0385, half the noise floor.
    off-wall burden     spearman(n_GT, severity) = -0.07 across the cohort.  Low-burden
                        cohort vessels score 0.8372, the same as the mean.
    clot depth          the median GT off-wall node is 2 hops off the boundary on 17 of 20
                        cohort vessels; there is no depth axis to correlate with.
    missing features    SEALED is present in the temporal transport cache.

WHAT SURVIVES.  The one statistic that tracks the outcome is the SCORE LEVEL OF THE GT
NODES THEMSELVES -- spearman(GT p50, severity) = +0.62.  On the cohort the model puts its
off-wall GT nodes at a median of 0.94-0.99 and the shipped cut (0.98 inside the physics
mask, 0.92 outside) sits below them.  On `patient007` and `patient013` the same field puts
them at 0.64 and 0.63, and the same cut sits *above* them, so recall collapses while
precision stays perfect.  The ranking is intact either way -- off-wall AUC on SEALED is
0.9964 -- which is exactly why a threshold is the wrong instrument to look at and the
CONFIDENCE is the right one.

This script measures that decomposition on both sets on the same footing, and writes the
per-vessel score distributions the report renders.

    python scripts/diag_offwall_score_geography.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_strict import apply_adapt, load_scores  # noqa: E402
from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.evaluate import domain_score  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer  # noqa: E402
from src.core_physics.wall_cohort_splits import SEALED  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
GRID = np.round(np.linspace(0.02, 0.98, 49), 4)


def _spec():
    """The SHIPPED off-wall committed-set spec, read off the promoted artifact."""
    import pickle

    root = REPO / "outputs/clot_ml/locked/DeployClot"
    with (root / "temporal.pkl").open("rb") as fh:
        t = pickle.load(fh)
    return t["off_spec"]


def _viz(stem: str, sc, gt, S, m_off, m_wall, group: str, max_bg: int = 2600) -> dict:
    """Per-node payload for the report's map: TP / FP / FN, plus a background sample.

    Positions come straight off the sample, so what is drawn is the mesh the model saw.
    The background is subsampled because a 20k-node vessel is 20k points the eye cannot
    resolve anyway; every GT, every predicted and every missed node is kept in full.
    """
    pos = np.asarray(S["pos"], dtype=np.float64)
    wall = np.asarray(S["wall"], dtype=bool)
    off = ~wall
    m = np.asarray(m_off, bool) | np.asarray(m_wall, bool)
    gt = np.asarray(gt, bool)
    rng = np.random.default_rng(0)
    interesting = m | gt | wall
    bg = np.flatnonzero(~interesting)
    if bg.size > max_bg:
        bg = rng.choice(bg, max_bg, replace=False)
    r2 = lambda a: [[round(float(x), 5), round(float(y), 5)] for x, y in pos[a]]  # noqa: E731
    return dict(
        stem=stem, group=group,
        bounds=[float(pos[:, 0].min()), float(pos[:, 0].max()),
                float(pos[:, 1].min()), float(pos[:, 1].max())],
        bg=r2(bg), wall=r2(np.flatnonzero(wall & ~gt & ~m)),
        # off-wall verdicts -- the domain under discussion
        tp=r2(np.flatnonzero(off & gt & m)),
        fp=r2(np.flatnonzero(off & ~gt & m)),
        fn=r2(np.flatnonzero(off & gt & ~m)),
        # wall verdicts, for context on the same picture
        w_tp=r2(np.flatnonzero(wall & gt & m)),
        w_fp=r2(np.flatnonzero(wall & ~gt & m)),
        w_fn=r2(np.flatnonzero(wall & gt & ~m)),
    )


def _row(stem: str, sc: np.ndarray, gt: np.ndarray, S: dict, spec: dict,
         group: str, viz: list | None = None) -> dict:
    d = ~np.asarray(S["wall"], dtype=bool)
    n_gt = int((gt & d).sum())
    vs = SeverityScorer(S["edge_index"], gt, len(S["wall"]), DEFAULT)
    off_of = (lambda X: ~np.asarray(X["wall"], dtype=bool))
    m = apply_adapt(S, sc, "resid", tuple(spec["th"]), off_of, spec["b"], spec["med"],
                    spec.get("lo"), spec.get("hi")) & d
    tp, npred = int((m & gt & d).sum()), int(m.sum())
    best, bt = -1.0, float("nan")
    for t in GRID:
        v = vs.score(d & (sc >= t), d)
        if v == v and v > best:
            best, bt = float(v), float(t)
    v_off = sc[d]
    v_gt = sc[gt & d] if n_gt else np.array([])
    v_bg = sc[d & ~gt]
    bar = float(np.mean(spec["th"][2:]))          # the two off-wall cuts
    # BOTH metrics, on the same mask.  `sev` is `SeverityScorer` -- what the strictly-nested
    # CV table reports.  `dep` is `evaluate.domain_score` -- what `eval_clot_ml_0.py`, and
    # therefore the SEALED read, reports.  They are different numbers for the same
    # prediction, and quoting one against the other is what manufactured a 0.22 "SEALED
    # off-wall gap" that does not exist (DEPLOYCLOT.md 22).
    ei_t = torch.tensor(np.asarray(S["edge_index"]))
    # The WALL domain on the same footing, so the metric offset can be read per domain --
    # it is not the same size on the two, and that is worth seeing.
    wall = np.asarray(S["wall"], dtype=bool)
    m_w = apply_adapt(S, sc, "resid", tuple(spec["th"]),
                      (lambda X: np.asarray(X["wall"], dtype=bool)),
                      spec["b"], spec["med"], spec.get("lo"), spec.get("hi")) & wall
    has_w = bool((gt & wall).any())
    if viz is not None:
        viz.append(_viz(stem, sc, gt, S, m, m_w, group))
    return dict(
        wall_sev=(float(vs.score(m_w, wall)) if has_w else float("nan")),
        wall_dep=(float(domain_score(m_w, gt, ei_t, wall, wall)) if has_w else float("nan")),
        n_wall_gt=int((gt & wall).sum()),
        stem=stem, group=group, n_off=int(d.sum()), n_gt=n_gt,
        sev=float(vs.score(m, d)),
        dep=float(domain_score(m, gt, ei_t, d, np.asarray(S["wall"], dtype=bool))),
        oracle=best, oracle_cut=bt,
        n_pred=npred, tp=tp,
        precision=(tp / npred if npred else float("nan")),
        recall=(tp / n_gt if n_gt else float("nan")),
        off_max=float(v_off.max()), off_p999=float(np.percentile(v_off, 99.9)),
        bg_p999=float(np.percentile(v_bg, 99.9)) if v_bg.size else float("nan"),
        gt_p10=float(np.percentile(v_gt, 10)) if n_gt else float("nan"),
        gt_p50=float(np.median(v_gt)) if n_gt else float("nan"),
        gt_p90=float(np.percentile(v_gt, 90)) if n_gt else float("nan"),
        gt_above_bar=(float(np.mean(v_gt >= bar)) if n_gt else float("nan")),
        bar=bar,
        # a coarse histogram of the GT nodes' scores, for the report
        gt_hist=(np.histogram(v_gt, bins=20, range=(0.0, 1.0))[0].tolist()
                 if n_gt else []),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="dc_fem_c0")
    ap.add_argument("--cache", default="v5_fem")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="outputs/deployclot/offwall_score_geography.json")
    ap.add_argument("--viz-out", default="outputs/deployclot/offwall_viz.json")
    ap.add_argument("--viz-stems", nargs="*",
                    default=["patient007", "patient013", "patient043",
                             "patient032", "patient012", "patient041", "patient001"],
                    help="vessels to dump a per-node map for")
    args = ap.parse_args()

    spec = _spec()
    print(f"[i] shipped off-wall spec: {spec['kind']} th={spec['th']}", flush=True)
    rows, viz = [], []
    want = set(args.viz_stems or [])

    cache = attach_physics(load_cache(args.cache))
    pool, folds, sc_all = load_scores(args.tags.split(","))
    fo = {a: k for k, h in folds.items() for a in h}
    for a in [x for x in pool if x in cache]:
        S = cache[a]
        gt = np.asarray(S["y"]) > 0.5
        if not (gt & ~np.asarray(S["wall"], bool)).any():
            continue
        rows.append(_row(a, sc_all[(fo[a], a)], gt, S, spec, "cohort",
                         viz if a in want else None))

    # SEALED: the promoted ensemble, on the packs, exactly as the final read ran it.
    from src.clot_ml.locked import build_sample, predict_scores
    from src.clot_ml.v0 import load_v0_bundle, solve_fem_into_pack
    from src.config import BiochemConfig, PhysicsConfig
    from src.core_physics.t0_mu_physics import gt_clot_phi_at_time

    bundle = load_v0_bundle(args.model)
    ens = bundle["base"]["base"]["ens"]
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    for stem in SEALED:
        p = PACKS / f"{stem}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        d.graph_stem = stem
        solve_fem_into_pack(d)
        S = build_sample(d, bio, flow="fem", variant="v4")
        S = dict(S)
        S.setdefault("y", np.zeros(len(S["wall"]), dtype=np.float32))
        from src.clot_ml.data import physics_mask

        if "phys_mask" not in S:
            S["phys_mask"] = physics_mask(S)
        T = int(d.y.shape[0])
        gt = (gt_clot_phi_at_time(d, T - 1, phys, device=torch.device("cpu"))
              .reshape(-1).numpy() > 0.5)
        if not (gt & ~np.asarray(S["wall"], bool)).any():
            print(f"  [skip] {stem}: no off-wall GT", flush=True)
            continue
        rows.append(_row(stem, predict_scores(ens, S), gt, S, spec, "SEALED",
                         viz if stem in want else None))
        print(f"  [ok] {stem}", flush=True)

    hdr = (f"{'vessel':12s}{'set':8s}{'nGT':>5s}{'prec':>6s}{'rec':>6s}"
           f"{'off sev':>9s}{'off dep':>9s}{'wall sev':>9s}{'wall dep':>9s}")
    print("\n" + hdr)
    for g in ("cohort", "SEALED"):
        sub = [r for r in rows if r["group"] == g]
        for r in sorted(sub, key=lambda x: x["gt_p50"]):
            print(f"{r['stem']:12s}{r['group']:8s}{r['n_gt']:5d}"
                  f"{r['precision']:6.2f}{r['recall']:6.2f}{r['sev']:9.4f}{r['dep']:9.4f}"
                  f"{r['wall_sev']:9.4f}{r['wall_dep']:9.4f}")
        if sub:
            f = lambda k: float(np.nanmean([r[k] for r in sub]))  # noqa: E731
            print(f"{'MEAN':12s}{g:8s}{'':5s}{f('precision'):6.2f}"
                  f"{f('recall'):6.2f}{f('sev'):9.4f}{f('dep'):9.4f}"
                  f"{f('wall_sev'):9.4f}{f('wall_dep'):9.4f}" + chr(10))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(spec=spec, per_vessel=rows), indent=2,
                                         default=float), encoding="utf-8")
    print(f"[save] {args.out}", flush=True)
    if args.viz_out and viz:
        Path(args.viz_out).parent.mkdir(parents=True, exist_ok=True)
        by = {r["stem"]: r for r in rows}
        for v in viz:
            r = by.get(v["stem"], {})
            v["scores"] = {k: r.get(k) for k in ("sev", "dep", "wall_sev", "wall_dep",
                                                 "n_gt", "n_wall_gt", "precision",
                                                 "recall", "oracle")}
        Path(args.viz_out).write_text(json.dumps(viz, default=float), encoding="utf-8")
        kb = Path(args.viz_out).stat().st_size // 1024
        print(f"[save] {args.viz_out}  ({len(viz)} vessels, {kb} KB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
