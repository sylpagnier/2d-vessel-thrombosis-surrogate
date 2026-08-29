"""Derive the deployment wall-shear reference band a kinematics cohort is accepted against.

    python scripts/derive_deploy_wall_shear_band.py

**FIT ONLY, deliberately.**  This band becomes a design target for the synthetic corpus, so
deriving it from DEV or SEALED would fit the generator to the vessels those splits exist to
provide independent evidence on.  Matching an INPUT distribution is ordinary domain adaptation
and the project already does it (`preflight_kine_cohort.py` compares `h_nd` and `u_ref` against
deployment); this adds the statistics the clot gate is actually decided by, which is the one
class of check preflight was missing when it passed a cohort whose `dsrx` branch never fires.

Nothing here is used as a LABEL.  The deploy packs are never trained on.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch

    from src.core_physics.wall_cohort_splits import FIT
    from src.utils.kinematics_select_packs import KNOWN_BAD_STEMS, STALE_EXTRACTOR_STEMS
    from src.utils.wall_shear_regime import REFERENCE_PATH, REGIME_KEYS, summarise, wall_shear_regime

    src = REPO / args.src
    stems = sorted(set(FIT) - set(STALE_EXTRACTOR_STEMS) - set(KNOWN_BAD_STEMS))
    if args.limit:
        stems = stems[: args.limit]

    rows, skipped = [], []
    for stem in stems:
        f = src / f"{stem}.pt"
        if not f.is_file():
            skipped.append(stem)
            continue
        try:
            r = wall_shear_regime(torch.load(f, map_location="cpu", weights_only=False))
        except Exception as exc:
            print(f"[warn] {stem}: {type(exc).__name__}: {exc}")
            r = None
        if r is None:
            skipped.append(stem)
            continue
        r["stem"] = stem
        rows.append(r)
        print(f"  {stem:<14}" + "  ".join(f"{k.replace('wall_', '')}={r[k]:.4g}"
                                          for k in REGIME_KEYS))

    if not rows:
        print("[ERR] no usable FIT packs")
        return 1

    summary = summarise(rows)
    print(f"\n{'metric':<16}{'p10':>12}{'p50':>12}{'p90':>12}")
    for k in REGIME_KEYS:
        if k in summary:
            print(f"{k:<16}" + "".join(f"{summary[k][q]:>12.4g}" for q in ("p10", "p50", "p90")))

    out = Path(args.out) if args.out else REPO / REFERENCE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": "FIT",
        "note": "FIT only -- DEV and SEALED are held out so they stay independent evidence. "
                "Consumer convention: MLS hops=3 on GT, wall nodes.",
        "n": len(rows), "skipped": skipped, "summary": summary, "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\n[save] {out}  ({len(rows)} FIT packs" + (f", {len(skipped)} skipped)" if skipped else ")"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
