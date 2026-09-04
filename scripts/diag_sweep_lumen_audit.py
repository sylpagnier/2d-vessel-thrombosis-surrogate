"""Audit every research-sweep arm for the collapsed-lumen signature, and FAIL if it recurs.

THE BUG THIS EXISTS FOR (found 2026-09-03).  In the 2026-09-01 sweep run, **all 15 wound arms
reported exactly `lumen_clot_pct = 0`, `max_occlusion_pct = 0`, `open_lumen_pct = 100`** while
all 43 non-wound arms reported non-zero lumen clot.  It read as physics -- `19_wound_vs_no_wound`
appeared to show a wound producing LESS clot than no wound -- and it was not:

  * `scripts/run_research_sweep.py` drives `CustomerDeployPipeline`, which asks for the
    artifact by the name `clot_ml_0`.  Until 2026-09-03 that name did not resolve through the
    locked pointer, so it fell through to the legacy `clot_ml_v0` stub (DEPLOYCLOT.md 21).
  * That stub records no `replace_scope`, so it inherited the then-current default
    `all_lumen`, under which the chemistry field REPLACES the GNN's verdict over the whole
    lumen rather than only the wound region.  Where the chemistry `Mat` does not clear
    `crit/att`, the replacement writes nothing -- and the GNN's own correct verdict, which the
    identical no-wound arm keeps, is discarded.  A wound could therefore only ever REDUCE
    predicted lumen clot.  DEPLOYCLOT.md 10 measured the same erasure on real packs
    (far field 0.0817 vs 0.2448); on synthetic geometry it goes all the way to zero.

Both causes are fixed.  This script is the standing check that they stay fixed, because the
failure is silent: every field is populated, every number is a valid float, and the arms look
like a monotone physical trend until you notice the zeros are EXACT.

    python scripts/diag_sweep_lumen_audit.py
    python scripts/diag_sweep_lumen_audit.py --allow-zero 04_inlet_width:w_0.020

Exit code 1 if any arm shows the signature and is not explicitly allowed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SWEEP_DIR = REPO / "outputs/research_sweeps"

#: The three fields that collapse together.  Any ONE of them at its degenerate value is
#: suspicious; all three at once is the signature.
SIG = (("lumen_clot_pct_final", 0.0), ("max_occlusion_pct_final", 0.0),
       ("open_lumen_pct_final", 100.0))


def audit(sweep_dir: Path) -> list[dict]:
    rows = []
    for d in sorted(sweep_dir.iterdir()):
        p = d / "summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        for a in s.get("arms", []):
            hits = [k for k, v in SIG if a.get(k) == v]
            rows.append(dict(
                sweep=d.name, arm=a.get("name", "?"),
                wall=float(a.get("wall_clot_pct_final", float("nan"))),
                lumen=float(a.get("lumen_clot_pct_final", float("nan"))),
                occl=float(a.get("max_occlusion_pct_final", float("nan"))),
                open_pct=float(a.get("open_lumen_pct_final", float("nan"))),
                collapsed=len(hits) == len(SIG),
                model=s.get("clot_model", "?"), flow=s.get("flow", "?"),
            ))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(SWEEP_DIR))
    ap.add_argument("--allow-zero", nargs="*", default=[],
                    help="sweep:arm pairs permitted to show the signature, e.g. a geometry "
                         "with genuinely no lumen clot. Each needs a reason in the docs.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = audit(Path(args.dir))
    if not rows:
        print("[ERR] no sweep summaries found -- has the sweep been run?", flush=True)
        return 1
    allow = set(args.allow_zero)
    bad = [r for r in rows if r["collapsed"] and f"{r['sweep']}:{r['arm']}" not in allow]

    print(f"{len(rows)} arms across {len({r['sweep'] for r in rows})} sweeps; "
          f"model={rows[0]['model']} flow={rows[0]['flow']}\n")
    print(f"{'sweep':28s}{'arm':22s}{'wall%':>8s}{'lumen%':>9s}{'occl%':>8s}{'open%':>8s}")
    for r in rows:
        flag = "  <-- COLLAPSED" if r["collapsed"] else ""
        print(f"{r['sweep']:28s}{r['arm']:22s}{r['wall']:8.3f}{r['lumen']:9.4f}"
              f"{r['occl']:8.3f}{r['open_pct']:8.3f}{flag}")

    n_coll = sum(1 for r in rows if r["collapsed"])
    print(f"\ncollapsed (lumen=0 AND occlusion=0 AND open=100): {n_coll} of {len(rows)}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"[save] {args.out}")

    if bad:
        print(f"\n[FAIL] {len(bad)} arm(s) show the collapsed-lumen signature:", flush=True)
        for r in bad:
            print(f"  {r['sweep']}:{r['arm']}  (wall {r['wall']:.2f}% but zero lumen)",
                  flush=True)
        print("\nA wound arm here almost certainly means the chemistry replacement erased the\n"
              "GNN's lumen verdict again -- check the artifact this ran against resolves through\n"
              "the pointer, and that its `replace_scope` is `wound_region`. See this file's\n"
              "docstring and DEPLOYCLOT.md 10 / 21 / 26.", flush=True)
        return 1
    print("\n[OK] no arm shows the collapsed-lumen signature.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
