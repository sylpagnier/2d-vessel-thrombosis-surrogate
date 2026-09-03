"""BATC (and BATC_0) wall + off-wall for the clot_free_w sweep, one shipped resid readout.

    python scripts/diag_batc_sweep_wall.py --tags dc_fem_c0 dc_fem_cfw025 dc_fem_cfw00
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_strict import GRID, load_scores, readout_resid, tune_resid  # noqa: E402
from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.severity_metric import BATC, BATC_0, SeverityScorer  # noqa: E402


def measure(tag: str, cache: dict) -> dict:
    pool, folds, sc = load_scores([tag])
    fo = {a: k for k, held in folds.items() for a in held}
    ves = [a for a in pool if a in cache]
    ves_off = [a for a in ves
               if (np.asarray(cache[a]["y"]) > 0.5)[~np.asarray(cache[a]["wall"], bool)].any()]
    out = {}
    for name, cfg in (("BATC", BATC), ("BATC_0", BATC_0)):
        VS = {a: SeverityScorer(cache[a]["edge_index"], np.asarray(cache[a]["y"]) > 0.5,
                                len(cache[a]["wall"]), cfg) for a in ves}
        W, O = [], []
        for k, held in sorted(folds.items()):
            tr = [a for a in ves if a not in held]
            te = [a for a in held if a in ves]
            if not te:
                continue
            th_w = tune_resid(cache, VS, tr, {a: sc[(fo[a], a)] for a in tr}, GRID)
            for a in te:
                wall = np.asarray(cache[a]["wall"], bool)
                d = ~wall
                s = sc[(fo[a], a)]
                m = readout_resid(cache[a], s, th_w)
                wv = VS[a].score(m & wall, wall)
                if wv == wv:
                    W.append(wv)
                if a in ves_off:
                    ov = VS[a].score(m & d, d)
                    if ov == ov:
                        O.append(ov)
            print(f"    [{tag}/{name}] fold {k} done", flush=True)
        out[name] = (float(np.mean(W)), float(np.mean(O)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--cache", default="v5_fem")
    ap.add_argument("--out", default="outputs/deployclot/batc_sweep.json")
    args = ap.parse_args()

    cache = attach_physics(load_cache(args.cache))
    results = {}
    print(f"{'tag':16s}{'BATC wall':>10s}{'BATC off':>10s}{'BATC0 wall':>11s}{'BATC0 off':>10s}",
          flush=True)
    for tag in args.tags:
        r = measure(tag, cache)
        results[tag] = r
        print(f"{tag:16s}{r['BATC'][0]:10.4f}{r['BATC'][1]:10.4f}"
              f"{r['BATC_0'][0]:11.4f}{r['BATC_0'][1]:10.4f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[save] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
