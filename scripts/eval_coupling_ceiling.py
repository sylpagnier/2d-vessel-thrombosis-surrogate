"""Does a PERFECTLY COUPLED feature set have a higher ceiling than the uncoupled one?

THE QUESTION THIS EXISTS TO ANSWER, and why E1 could not answer it.  `eval_closed_loop_oracle.py`
fed ground-truth clot back into the flow and scored the result with the SHIPPED readout -- a
readout whose cuts were tuned on open-loop features.  Off-wall it came out -0.1604, and that
number is uninterpretable: it confounds "coupling does not help" with "our readout cannot consume
a field it was never tuned for."  A fair test has to give each feature set its OWN best readout.

THE INSTRUMENT.  For every vessel and domain, rank the nodes by that arm's own score field and
sweep the committed set size k, keeping the BEST score any k achieves.  This is the `oracle_cut`
idea of `eval_expected_score_readout.py` -- *"the best score any single cut reaches on THIS
vessel: the ceiling, never an arm"* -- in the ranking form the tiny off-wall burdens here
require (see `_ceiling`).  Computing it per arm removes the tuning mismatch by construction:

    open-loop  ceiling   what the uncoupled feature set could reach with a perfect readout
    closed-loop ceiling  what the perfectly-coupled feature set could reach with a perfect readout

**The comparison is CEILING vs CEILING, never ceiling vs shipped.**  An oracle cut is not
deployable -- it is chosen with knowledge of the answer -- so neither number may be quoted as a
model score.  Only the DIFFERENCE is meaningful, and only as "designing for coupling would/would
not raise the attainable ceiling."

READ THE RESULT LIKE THIS:
  * ceiling_closed  >  ceiling_open   -> coupling has real headroom our fixed-cut model cannot
                                        reach; STORY Leg 2 weakens and a coupled-native readout
                                        becomes worth building.
  * ceiling_closed ~= ceiling_open    -> the oracle field carries no extra reachable signal;
                                        E1's off-wall regression was tuning mismatch, and Leg 2's
                                        "solve once" stands on a fair test.
  * ceiling_closed  <  ceiling_open   -> the coupled field is genuinely worse to predict from,
                                        which would be a finding in its own right.

**THE SWEEP IS PER ARM AND PER VESSEL, and that is load-bearing.**  Closing the loop shifts the
score distribution, so a threshold grid shared between arms would hand one of them an advantage it
did not earn.  Ranking sidesteps the scale entirely: each arm is judged on the ORDER it puts the
nodes in, which is the thing a better-informed feature set ought to improve.

**WHAT THIS CANNOT SETTLE.**  It compares FEATURE SETS under a perfect readout, not ARCHITECTURES.
A network trained from scratch on closed-loop features might extract signal the shipped GNN's
score field does not expose.  This test says the oracle-coupled field carries no extra REACHABLE
signal for the shipped representation -- strong evidence against building a coupled-native
readout, not proof no coupled architecture could ever win.

TWO SUBPROCESSES, for the same reason as `eval_closed_loop_oracle.py`: `CLOT_ML_ORACLE_BLOCKAGE`
is captured at module import time (`src/clot_ml/features.py:59`), so each arm needs a fresh
interpreter with the environment set BEFORE import.  The parent asserts the two arms' score
fields actually differ; identical fields mean the toggle is dead, not that coupling is inert.

    python scripts/eval_coupling_ceiling.py --stems comsol005 comsol006 comsol012
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.utils.paths import get_project_root

REPO = get_project_root()


def _dump_arm(stems, flow: str, every: int, out: Path) -> None:
    """WORKER: predict each vessel and save the continuous score field, GT and domains."""
    import torch

    from scripts.eval_clot_ml_0 import PACKS
    from scripts.eval_wound_complement import gt_series
    from src.clot_ml.evaluate import time_grid as _times
    from src.clot_ml.locked import build_sample
    from src.clot_ml.v0 import load_v0_bundle, predict_clot_ml_0
    from src.clot_ml.wound import solid_mask
    from src.config import BiochemConfig, PhysicsConfig

    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    bundle = load_v0_bundle(None)
    blob: dict[str, np.ndarray] = {}
    for stem in stems:
        data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
        if getattr(data, "graph_stem", None) is None:
            data.graph_stem = stem
        if flow == "fem":
            from src.clot_ml.v0 import solve_fem_into_pack
            solve_fem_into_pack(data)
        times = _times(data, every)
        S = build_sample(data, bio, flow=flow, variant="v4")
        out_pred = predict_clot_ml_0(bundle, data, times, flow=flow, sample=S)
        sc = out_pred.get("score")
        if sc is None:
            print(f"  [skip] {stem}: no continuous score field")
            continue
        gts = gt_series(data, phys, times)
        wall = np.asarray(data.mask_wall).reshape(-1).astype(bool)
        off = ~np.asarray(solid_mask(data)).reshape(-1).astype(bool)
        blob[f"score|{stem}"] = np.asarray(sc, dtype=np.float64).reshape(-1)
        blob[f"gt|{stem}"] = np.asarray(gts[times[-1]], dtype=bool).reshape(-1)
        blob[f"wall|{stem}"] = wall
        blob[f"off|{stem}"] = off
        blob[f"ei|{stem}"] = np.asarray(data.edge_index)
        print(f"  [ok] {stem}", flush=True)
    np.savez_compressed(out, **blob)


def _ceiling(score, gt, dom, ei, n) -> tuple[float, float]:
    """Best BATC reachable by RANKING this arm's score field, and the set size that achieved it.

    TOP-K, NOT A QUANTILE GRID, and the reason matters.  Off-wall burdens here are tiny against
    huge domains -- `comsol005` carries 4 GT positives among 9623 lumen nodes -- so a global
    quantile threshold cannot express the answer at all: even the top 0.5% commits 48 nodes
    against 4 true ones.  A quantile sweep therefore reports a "ceiling" BELOW the score the real
    pipeline achieves, which is proof the family is wrong, not that the model is better than its
    ceiling.  Sweeping the top-k instead asks the question that actually matters here: **does the
    coupled field RANK clot nodes better than the uncoupled one**, independent of any cut rule.
    """
    from src.clot_ml.severity_metric import BATC, dilation_operator, severity_components

    m = int(dom.sum())
    b = int((gt & dom).sum())
    if m == 0 or b == 0:
        return float("nan"), float("nan")
    D = dilation_operator(ei, n, hops=BATC.relax_hops)
    idx = np.flatnonzero(dom)
    order = idx[np.argsort(-score[idx], kind="stable")]
    # dense near the burden, then coarse out to a generous multiple of it
    ks = sorted({int(k) for k in np.concatenate([
        np.arange(1, min(m, max(3 * b, 40)) + 1),
        np.unique(np.geomspace(max(3 * b, 40), m, 60).astype(int)),
    ]) if 1 <= int(k) <= m})
    best, at = float("nan"), float("nan")
    for k in ks:
        pred = np.zeros(n, dtype=bool)
        pred[order[:k]] = True
        c = severity_components(pred, gt & dom, D, cfg=BATC)
        x = float(c.get("score", float("nan")))
        if x == x and (best != best or x > best):
            best, at = x, float(k)
    return best, at


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=["comsol005", "comsol006", "comsol012"])
    ap.add_argument("--flow", default="fem")
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--out", default="outputs/deployclot/coupling_ceiling.json")
    ap.add_argument("--dump", default="", help="internal: worker mode, write score fields here")
    ap.add_argument("--blockage", default="", help="internal: worker arm label")
    ap.add_argument("--reuse", action="store_true",
                    help="skip both prediction arms and recompute from the saved score dumps")
    args = ap.parse_args()

    if args.dump:
        _dump_arm(args.stems, args.flow, args.every, Path(args.dump))
        return 0

    tmp = REPO / "outputs" / "deployclot"
    tmp.mkdir(parents=True, exist_ok=True)
    paths = {}
    for arm, blk in (("open", False), ("closed", True)):
        p = tmp / f"_coupling_scores_{arm}.npz"
        if args.reuse and p.is_file():
            print(f"[i] reusing {p}")
            paths[arm] = p
            continue
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO)
        if blk:
            env["CLOT_ML_ORACLE_BLOCKAGE"] = "1"
        else:
            env.pop("CLOT_ML_ORACLE_BLOCKAGE", None)
        cmd = [sys.executable, "scripts/eval_coupling_ceiling.py", "--dump", str(p),
               "--flow", args.flow, "--every", str(args.every), "--stems", *args.stems]
        print(f"\n[i] === arm: {arm} ===", flush=True)
        r = subprocess.run(cmd, cwd=REPO, env=env)
        if r.returncode != 0:
            raise SystemExit(f"[ERR] {arm} arm exited {r.returncode}")
        paths[arm] = p

    Z = {a: np.load(p, allow_pickle=True) for a, p in paths.items()}
    stems = sorted({k.split("|")[1] for k in Z["open"].files if "|" in k}
                   & {k.split("|")[1] for k in Z["closed"].files if "|" in k})
    if not stems:
        raise SystemExit("[ERR] no vessels common to both arms")

    identical = all(np.array_equal(Z["open"][f"score|{s}"], Z["closed"][f"score|{s}"])
                    for s in stems)
    if identical:
        raise SystemExit(
            "[ERR] the two arms' score fields are BIT-IDENTICAL -- CLOT_ML_ORACLE_BLOCKAGE "
            "never reached the closed arm. Check src/clot_ml/features.py:59.")

    per, res = {}, {}
    for dom_name in ("wall", "off"):
        rows = []
        for s in stems:
            n = len(Z["open"][f"score|{s}"])
            ei = Z["open"][f"ei|{s}"]
            gt = Z["open"][f"gt|{s}"]
            dom = Z["open"][f"{dom_name}|{s}"]
            co, qo = _ceiling(Z["open"][f"score|{s}"], gt, dom, ei, n)
            cc, qc = _ceiling(Z["closed"][f"score|{s}"], gt, dom, ei, n)
            rows.append(dict(stem=s, ceiling_open=None if co != co else round(co, 4),
                             ceiling_closed=None if cc != cc else round(cc, 4),
                             delta=None if (co != co or cc != cc) else round(cc - co, 4),
                             k_open=None if qo != qo else int(qo),
                             k_closed=None if qc != qc else int(qc),
                             burden=int((Z["open"][f"gt|{s}"] & dom).sum()),
                             domain_nodes=int(dom.sum())))
        per[dom_name] = rows
        d = [r["delta"] for r in rows if r["delta"] is not None]
        res[dom_name] = dict(
            n=len(d),
            mean_ceiling_open=round(float(np.mean([r["ceiling_open"] for r in rows
                                                   if r["ceiling_open"] is not None])), 4) if d else None,
            mean_ceiling_closed=round(float(np.mean([r["ceiling_closed"] for r in rows
                                                     if r["ceiling_closed"] is not None])), 4) if d else None,
            delta=round(float(np.mean(d)), 4) if d else None,
            all_same_sign=bool(d and (all(x > 0 for x in d) or all(x < 0 for x in d))),
        )

    out = dict(
        question=("does a perfectly-coupled feature set have a HIGHER ATTAINABLE CEILING than the "
                  "uncoupled one? Each arm is scored at its OWN best per-vessel cut, so the "
                  "readout-tuning mismatch that made eval_closed_loop_oracle's off-wall number "
                  "uninterpretable is removed by construction."),
        caveat=("an oracle cut is chosen knowing the answer and is NOT deployable. Neither "
                "ceiling may be quoted as a model score; only the difference is meaningful."),
        metric="BATC (severity, the reported setting), final time",
        flow=args.flow, every=args.every, family="top-k sweep over each arm's own score ranking",
        n_vessels=len(stems), vessels=stems,
        result=res, per_vessel=per,
    )
    (REPO / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== CEILING vs CEILING (each arm at its own best cut) ===")
    print(f"{'domain':<8}{'open':>9}{'closed':>9}{'delta':>9}{'n':>4}  same-sign")
    for k, v in res.items():
        if v["delta"] is None:
            print(f"{k:<8}{'--':>9}{'--':>9}{'--':>9}{v['n']:>4}")
            continue
        print(f"{k:<8}{v['mean_ceiling_open']:>9.4f}{v['mean_ceiling_closed']:>9.4f}"
              f"{v['delta']:>+9.4f}{v['n']:>4}  {v['all_same_sign']}")
    print(f"\n[i] wrote {REPO / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
