"""Cache per-node features + targets for the PHASE9 clot-ML stack.

    python scripts/build_clot_ml_cache.py --flow gt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.features import build_features, feature_matrix  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.wall_cohort_splits import CLOT_FREE, DEV, FIT, MIN_T  # noqa: E402

DIR = REPO / "data/processed/graphs_biochem_anchors"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default="gt", choices=["gt", "pred"])
    ap.add_argument("--out", default="")
    ap.add_argument("--force", action="store_true",
                    help="rebuild vessels already present.  Needed after any change to the "
                         "packs or to `features.build_features` -- without it a stale cache "
                         "is silently kept and every downstream number inherits it.")
    ap.add_argument("--only", default="",
                    help="comma-separated anchors, for resuming a partial rebuild or for "
                         "building a small cache to smoke-test the pipeline against")
    args = ap.parse_args()
    out = Path(args.out or f"outputs/clot_ml_cache_{args.flow}")
    out.mkdir(parents=True, exist_ok=True)
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")

    # CLOT_FREE joins the cache as of 2026-08-22 (docs/SEALED_SPLIT.md,
    # MODEL_REVIEW_2026-08-22 8b): those 8 vessels carry no recall but they are real evidence
    # about FALSE POSITIVES, and until they were cached nothing could be measured on them --
    # `SeverityScorer.score(..., empty_gt="score")` existed with no data to apply it to.
    todo = list(FIT) + list(DEV) + list(CLOT_FREE)
    only = [x.strip() for x in args.only.split(",") if x.strip()]
    if only:
        todo = [a for a in todo if a in only]
    for a in todo:
        p = DIR / f"{a}.pt"
        dst = out / f"{a}.npz"
        if dst.exists() and not args.force:
            print("[skip] %s" % a, flush=True)
            continue
        if not p.exists():
            print("[miss] %s" % a, flush=True)
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < MIN_T:
            print("[drop] %s T=%d" % (a, int(d.y.shape[0])), flush=True)
            continue
        t0 = time.time()
        try:
            S = build_features(d, bio, phys, flow=args.flow)
        except Exception as e:  # noqa: BLE001
            print("[ERR ] %s %s" % (a, e), flush=True)
            continue
        if S["y"].sum() == 0 and a not in CLOT_FREE:
            # An UNEXPECTED empty GT is still a data error and still gets dropped.  A vessel
            # on the clot-free list is empty by design, and dropping it was the reason the
            # 2026-08-22 cohort decision never reached the model.
            print("[drop] %s empty GT (not on the clot-free list)" % a, flush=True)
            continue
        X, cols = feature_matrix(S["F"])
        np.savez_compressed(
            dst, X=X, cols=np.array(cols), y=S["y"], mat_gt=S["mat_gt"],
            wall=S["wall"], solid=S["solid"], shell=S["shell"], owner=S["owner"],
            edge_index=S["edge_index"], pos=S["pos"], mat_phys=S["mat_phys"],
            gate=S["gate"], sr=S["sr"], spd=S["spd"], u=S["u"], v=S["v"])
        print("[ok  ] %-12s n=%6d feats=%d clot=%5d (off %4d)%s  %.1fs"
              % (a, S["n"], X.shape[1], int(S["y"].sum()),
                 int((S["y"] > 0.5).sum() - ((S["y"] > 0.5) & S["wall"]).sum()),
                 "  CLOT-FREE" if a in CLOT_FREE else "",
                 time.time() - t0), flush=True)
    print("done -> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
