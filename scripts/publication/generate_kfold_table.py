"""Table 4: strict nested-CV out-of-fold generalization, per vessel and per geometry class.

This is the paper's primary evidence for the geometry-generalization claim.  Every vessel in the
eligible pool is held out exactly once, so each row is an honest out-of-fold score, and the
per-class rollup is what the claim is actually about.

TWO CAVEATS THAT MUST TRAVEL WITH THE TABLE.

1.  **These are GT-flow numbers.**  The OOF archive is built with `flow="gt"`
    (`oof_data.py` docstring: the strict masks are exported under ground-truth t=0 flow).  The
    shipped path uses a local FEM solve.  The paper's flow-requirement section is what licenses
    reading one as the other -- FEM sits inside noise of GT (0.705 vs 0.710) -- so this table
    measures *geometry* transfer of the clot model given good flow, which is exactly the claim,
    but the sentence has to appear in the caption or the table overstates its own scope.

2.  **Aneurysm is n=1.**  With one non-SEALED aneurysm, no fold trains on an aneurysm while
    measuring a different one (`src/clot_ml/geometry_splits.py`).  The per-class row for
    aneurysm is a single vessel and the table says so rather than hiding it in a mean.

Usage:
    python scripts/publication/generate_kfold_table.py
    python scripts/publication/generate_kfold_table.py --at final    # final-time only
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.publication.config import CONFIG, DATA_DIR  # noqa: E402
from scripts.publication.oof_data import (  # noqa: E402
    build_vessel_figure_data, ensure_oof_series, load_oof_archive, metrics_rows_for_vessel,
)

PACKS = REPO / "data/processed/graphs_biochem_anchors"


def _geometry_class(stem: str) -> str:
    """Measured class, with the user designation authoritative where it exists."""
    from src.clot_ml.geometry_class import classify, width_stats

    p = PACKS / f"{stem}.pt"
    if not p.is_file():
        return "unknown"
    try:
        d = torch.load(p, map_location="cpu", weights_only=False)
        # `anchor` is passed so the human designation stays authoritative -- without it the
        # measured stenosis cut silently reclassifies every designated stenosis to baseline
        # (geometry_class.py, the A2 consequence note).
        return str(classify(width_stats(d), anchor=stem))
    except Exception:
        return "unknown"


# Reproducibility floor from repeated fits of the same configuration (MODEL_REVIEW s9f).
# NOT the spread across vessels -- it bounds how much a number moves when the SAME configuration
# is refitted, so it says whether a difference is a real effect or a reseed.
NOISE_FLOOR = {"wall": 0.0037, "off": 0.0432}


def _summary(vals: list[float]) -> dict:
    """Mean with a dispersion the reader can actually use.

    `sem` and `ci95` are across VESSELS in the class -- "would another draw of vessels move this
    mean".  The config noise floor answers a different question -- "would another fit move it".
    The paper needs both; quoting one as if it were the other has already caused an over-claim
    here (docs/PUBLICATION_NOTES.md, standing rule 3).
    """
    a = np.asarray(vals, float)
    n = int(a.size)
    out = {"mean": float(a.mean()), "median": float(statistics.median(vals)), "n": n,
           "min": float(a.min()), "max": float(a.max())}
    if n > 1:
        sd = float(a.std(ddof=1))
        sem = sd / float(np.sqrt(n))
        out["sd"], out["sem"] = sd, sem
        # Normal approximation; at n=3 it is indicative, which is why `n` travels with it.
        out["ci95"] = [float(a.mean() - 1.96 * sem), float(a.mean() + 1.96 * sem)]
    else:
        out["sd"] = out["sem"] = float("nan")
        out["ci95"] = None
        out["warning"] = "n=1 -- a single vessel; no dispersion is estimable"
    return out


def _agg(rows: list[dict], key: str, at: str) -> float:
    vals = [r[key] for r in rows if key in r and r[key] == r[key]]
    if not vals:
        return float("nan")
    return float(vals[-1]) if at == "final" else float(np.mean(vals))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--at", default="both", choices=("final", "mean", "both"),
                    help="final-time, mean-over-time, or both (default)")
    ap.add_argument("--force", action="store_true", help="re-export the OOF archive first")
    ap.add_argument("--out", default=str(DATA_DIR / "table4_kfold.json"))
    a = ap.parse_args()

    path = ensure_oof_series(CONFIG, regenerate=a.force)
    archive = load_oof_archive(path)
    print(f"[table4] OOF archive: {path.name}  "
          f"({len(archive.vessels)} vessels, flow={archive.flow})")
    if archive.flow != "gt":
        print(f"[table4] NOTE: archive flow is {archive.flow!r}, not 'gt' -- "
              "check the caveat in this script's docstring still applies.")

    per_vessel, failed = [], []
    for stem, oof in sorted(archive.vessels.items()):
        try:
            payload = build_vessel_figure_data(stem, oof)
            rows = metrics_rows_for_vessel(stem, payload)
            # GT burden at final time, per domain.  A vessel can carry wall clot and NO off-wall
            # clot; its off-wall F1 is then an empty-GT false-positive score, not recall, and
            # averaging it into a class mean repeats the error the clot-free protocol exists to
            # prevent (MODEL_REVIEW s8b).  Measured here so the rollup can separate them.
            ctx = payload["_score_ctx"]
            gt_last = np.asarray(payload["frames"][payload["times"][-1]]["gt_mask"], bool)
            n_off = int((gt_last & np.asarray(ctx["off"], bool)).sum())
            n_wall = int((gt_last & np.asarray(ctx["wall"], bool)).sum())
            rec = {
                "stem": stem,
                "fold": payload["fold"],
                "is_wound": payload["is_wound"],
                "geometry_class": _geometry_class(stem),
                "n_times": len(payload["times"]),
                "gt_off_n": n_off,
                "gt_wall_n": n_wall,
                "off_gt_empty": n_off == 0,
            }
            for key in ("wall", "off", "w_reg", "w_lum", "far"):
                if any(key in r for r in rows):
                    if a.at in ("final", "both"):
                        rec[f"{key}_final"] = _agg(rows, key, "final")
                    if a.at in ("mean", "both"):
                        rec[f"{key}_mot"] = _agg(rows, key, "mean")
            per_vessel.append(rec)
            print(f"  {stem:<20} fold={rec['fold']}  class={rec['geometry_class']:<10} "
                  f"wall={rec.get('wall_final', float('nan')):.4f}  "
                  f"off={rec.get('off_final', float('nan')):.4f}", flush=True)
        except Exception as exc:
            failed.append({"stem": stem, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  {stem:<20} FAILED  {type(exc).__name__}: {exc}", flush=True)

    if not per_vessel:
        print("[table4] nothing scored")
        return 1

    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in per_vessel:
        by_class[r["geometry_class"]].append(r)

    per_class = {}
    for cls, rs in sorted(by_class.items()):
        entry = {"n": len(rs), "vessels": [r["stem"] for r in rs]}
        # Wall means run over every vessel; OFF-WALL means run only over vessels that HAVE
        # off-wall ground truth.  The rest are reported as a false-positive row instead.
        with_off = [r for r in rs if not r.get("off_gt_empty", False)]
        entry["n_off_scored"] = len(with_off)
        entry["off_gt_empty_vessels"] = [r["stem"] for r in rs if r.get("off_gt_empty", False)]
        for key in ("wall_final", "wall_mot"):
            vals = [r[key] for r in rs if key in r and r[key] == r[key]]
            if vals:
                entry[key] = _summary(vals)
        for key in ("off_final", "off_mot"):
            vals = [r[key] for r in with_off if key in r and r[key] == r[key]]
            if vals:
                entry[key] = _summary(vals)
            empt = [r[key] for r in rs
                    if r.get("off_gt_empty", False) and key in r and r[key] == r[key]]
            if empt:
                entry[key + "_empty_gt"] = {"mean": float(np.mean(empt)), "n": len(empt),
                                            "note": "empty off-wall GT: FP score, NOT recall"}
        # An n=1 class mean is a single vessel; label it so no reader averages it onward.
        if len(rs) == 1:
            entry["warning"] = "n=1 -- a single vessel, not a class mean"
        per_class[cls] = entry

    payload = {
        "source": str(path),
        "flow": archive.flow,
        "caveats": [
            "OOF masks are exported under GT t=0 flow; the shipped path uses a local FEM solve. "
            "The flow-requirement section licenses reading one as the other (FEM within noise "
            "of GT). State this in the caption.",
            "Aneurysm generalization is n=1: no fold trains on one aneurysm and measures "
            "another (src/clot_ml/geometry_splits.py).",
            "Off-wall means cover ONLY vessels with non-empty off-wall GT. Vessels with zero "
            "off-wall GT score false positives, not recall, and are reported separately in "
            "*_empty_gt. Do not merge the two.",
        ],
        "noise_floor": dict(NOISE_FLOOR,
                            note="config reproducibility floor (MODEL_REVIEW s9f); a difference "
                                 "smaller than this is a reseed, not an effect. Distinct from "
                                 "sem/ci95, which are across vessels."),
        "per_vessel": per_vessel,
        "per_class": per_class,
        "failed": failed,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv = Path(a.out).with_suffix(".csv")
    cols = ["stem", "fold", "geometry_class", "is_wound", "n_times",
            "wall_final", "off_final", "wall_mot", "off_mot"]
    with open(csv, "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in per_vessel:
            fh.write(",".join(
                f"{r[c]:.4f}" if isinstance(r.get(c), float) else str(r.get(c, ""))
                for c in cols) + "\n")

    print(f"\n=== out-of-fold by geometry class  (mean +/- SEM across vessels) ===")
    for cls, e in per_class.items():
        wf, of = e.get("wall_final", {}), e.get("off_final", {})
        w, o = wf.get("mean", float("nan")), of.get("mean", float("nan"))
        ws, oss = wf.get("sem", float("nan")), of.get("sem", float("nan"))
        n_off = e.get("n_off_scored", 0)
        flag = "   <-- n=1" if e["n"] == 1 else ""
        print(f"  {cls:<12} n={e['n']:<3} wall {w:.4f} +/-{ws:.4f}   "
              f"off {o:.4f} +/-{oss:.4f} (n={n_off}){flag}")
        if e.get("off_gt_empty_vessels"):
            fp = e.get("off_final_empty_gt", {})
            print(f"               empty off-wall GT: {len(e['off_gt_empty_vessels'])} vessel(s)"
                  f" -- FP score {fp.get('mean', float('nan')):.4f}, excluded from the mean")
    if failed:
        print(f"\n  {len(failed)} vessel(s) failed")
    print(f"\nwrote {a.out}\n     {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
