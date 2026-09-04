"""Assemble the flow-requirement dataset: paired GT/pred deploy scores + the tolerance curve.

This is the data behind the paper's methodological figure (Fig 9): *how accurate does the t=0
flow surrogate have to be, and what statistic tells you?*

Two panels, two inputs:

  (a) CORRELATION.  Per vessel, the wall-score DROP when the flow is swapped from ground truth
      to the surrogate, against candidate diagnostics of that vessel's flow quality.  The claim
      is that velocity rel-L2 is near-uninformative (r ~ -0.03) while the gate statistics are
      not (r ~ +0.61).  Built here from the paired `--flow gt` / `--flow pred` cohort runs.

  (b) TOLERANCE.  The GT->pred blend curve, `u(a) = (1-a)*u_gt + a*u_pred`, scored at each `a`.
      Produced by `scripts/diag_flow_sensitivity.py`; this script only collects it.

PROVENANCE WARNING.  `scripts/diag_flow_sensitivity.py` and `scripts/diag_wall_gate_health.py`
were deleted in commit b2eebb9 ("Fix customer_pipeline.py couple unpack error", 2026-09-01) and
restored on 2026-09-01.  `outputs/runs/` was empty at that point, so the artifacts named in
`docs/PUBLICATION_NOTES.md` s2 no longer existed either.  Everything this figure rests on must
therefore be REGENERATED before the number goes in a draft -- do not assume a stale JSON on
disk is the one the notes describe.

Inputs (produce them first; this script tells you how if they are missing):

    python scripts/eval_clot_ml_0.py --cohort --flow gt   --out outputs/runs/eval_gt.json
    python scripts/eval_clot_ml_0.py --cohort --flow pred --out outputs/runs/pred_all.json
    python scripts/diag_flow_sensitivity.py <stems...> --source pred \
        --out outputs/runs/flow_sensitivity.json

Usage:
    python scripts/publication/generate_fig9_data.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Repo root by marker, not by depth: this file may move between
# scripts/ and scripts/<subdir>/ without silently resolving one level off.
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.publication.config import DATA_DIR  # noqa: E402

RUNS = REPO / "outputs" / "runs"

# Candidate diagnostics, in the order the paper reports them.  `higher_is_better` records
# whether a LARGER value should mean a SMALLER drop, so the sign of each correlation can be
# read as "does this diagnostic behave the way you would want a health check to behave".
DIAGNOSTICS = (
    ("gate_jaccard", "gate Jaccard (fraction of ceiling)", True),
    ("fire_ratio", "wall-gate firing ratio", True),
    ("empty_gate", "empty-gate indicator", False),
    ("dsrx_corr", "dsrx correlation", True),
    ("rel_l2", "velocity rel-L2", False),
)


def _load(path: Path) -> list[dict] | None:
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict):
        for key in ("rows", "per_vessel", "vessels"):
            if isinstance(obj.get(key), list):
                return obj[key]
        return [dict(v, stem=k) for k, v in obj.items() if isinstance(v, dict)]
    return obj if isinstance(obj, list) else None


def _score(row: dict) -> float | None:
    """Wall score, whatever the eval harness called it in this run."""
    for k in ("v0_fin_wall", "wall", "wall_final", "deploy_clot_f1"):
        v = row.get(k)
        if v is not None and float(v) == float(v):
            return float(v)
    return None


def _pairs(gt_rows: list[dict], pred_rows: list[dict]) -> list[dict]:
    """Join GT and predicted-flow runs on vessel, and record the drop."""
    gt = {r["stem"]: r for r in gt_rows if "stem" in r}
    out = []
    for r in pred_rows:
        stem = r.get("stem")
        if stem is None or stem not in gt:
            continue
        s_gt, s_pr = _score(gt[stem]), _score(r)
        if s_gt is None or s_pr is None:
            continue
        row = {"stem": stem, "wall_gt": s_gt, "wall_pred": s_pr, "drop": s_gt - s_pr}
        for key, _label, _hib in DIAGNOSTICS:
            if key in r:
                row[key] = float(r[key])
        out.append(row)
    return out


def _corr(xs: list[float], ys: list[float]) -> dict | None:
    """Pearson r with n; None when the sample is too small or degenerate."""
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return None
    return {"r": float(np.corrcoef(x, y)[0, 1]), "n": int(x.size)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default=str(RUNS / "eval_gt.json"))
    ap.add_argument("--pred", default=str(RUNS / "pred_all.json"))
    ap.add_argument("--sensitivity", default=str(RUNS / "flow_sensitivity.json"))
    ap.add_argument("--diagnostics", default=str(RUNS / "flow_diagnostics.json"),
                    help="per-vessel flow-quality diagnostics "
                         "(scripts/publication/generate_flow_diagnostics.py)")
    ap.add_argument("--out", default=str(DATA_DIR / "flow_requirement.json"))
    a = ap.parse_args()

    payload: dict = {"panel_a": None, "panel_b": None, "missing": []}

    gt_rows, pred_rows = _load(Path(a.gt)), _load(Path(a.pred))
    if gt_rows is None or pred_rows is None:
        payload["missing"].append(
            "panel (a): paired cohort runs. Produce with:\n"
            f"    python scripts/eval_clot_ml_0.py --cohort --flow gt   --out {a.gt}\n"
            f"    python scripts/eval_clot_ml_0.py --cohort --flow pred --out {a.pred}")
    else:
        # Diagnostics live in their own artifact because the eval harness emits outcomes,
        # not candidate predictors.  Join them on vessel before correlating.
        diag = {r["stem"]: r for r in (_load(Path(a.diagnostics)) or []) if "stem" in r}
        if diag:
            for r in pred_rows:
                if r.get("stem") in diag:
                    r.update({k: v for k, v in diag[r["stem"]].items() if k != "stem"})
        pairs = _pairs(gt_rows, pred_rows)
        if not pairs:
            payload["missing"].append(
                "panel (a): GT and pred runs found but share no scorable vessel.")
        else:
            drops = [p["drop"] for p in pairs]
            corrs = {}
            for key, label, hib in DIAGNOSTICS:
                vals = [p.get(key, float("nan")) for p in pairs]
                c = _corr(vals, drops)
                if c is not None:
                    corrs[key] = {**c, "label": label, "higher_is_better": hib}
            payload["panel_a"] = {"pairs": pairs, "correlations": corrs,
                                  "n_vessels": len(pairs)}
            if not corrs:
                payload["missing"].append(
                    "panel (a): no diagnostic columns in the eval rows. The correlation panel "
                    "needs per-vessel gate statistics (gate_jaccard, fire_ratio, empty_gate, "
                    "dsrx_corr, rel_l2) joined onto the eval rows. Produce with:\n"
                    f"    python scripts/publication/generate_flow_diagnostics.py "
                    f"--out {a.diagnostics}")

    sens = _load(Path(a.sensitivity))
    if sens is None:
        payload["missing"].append(
            "panel (b): tolerance curve. Produce with:\n"
            f"    python scripts/diag_flow_sensitivity.py comsol010 comsol005 comsol020 "
            f"--source pred --out {a.sensitivity}")
    else:
        by_stem: dict[str, list[dict]] = {}
        for r in sens:
            by_stem.setdefault(str(r.get("stem", "?")), []).append(r)
        payload["panel_b"] = {
            "curves": {k: sorted(v, key=lambda r: float(r.get("alpha", 0)))
                       for k, v in by_stem.items()},
            "n_vessels": len(by_stem),
        }
        if len(by_stem) < 3:
            payload["missing"].append(
                f"panel (b): only {len(by_stem)} vessel(s) in the tolerance curve. The paper "
                "needs >=3 -- one vessel is an anecdote, not a curve.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[flow-req] wrote {a.out}")
    if payload["panel_a"]:
        pa = payload["panel_a"]
        print(f"  panel (a): {pa['n_vessels']} vessels")
        for key, c in sorted(pa["correlations"].items(), key=lambda kv: -abs(kv[1]["r"])):
            print(f"     {c['label']:<38} r = {c['r']:+.3f}  (n={c['n']})")
    if payload["panel_b"]:
        print(f"  panel (b): {payload['panel_b']['n_vessels']} tolerance curve(s)")
    if payload["missing"]:
        print("\n[flow-req] NOT COMPLETE -- outstanding:")
        for m in payload["missing"]:
            print("  - " + m)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
