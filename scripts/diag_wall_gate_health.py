"""Does a Stage-A checkpoint keep the WALL gate alive on the deploy packs?

    python scripts/diag_wall_gate_health.py --checkpoint outputs/runs/armF_best.pth

Gate Jaccard is NOT the quantity that decides the deploy collapse -- measured, fixing the
`dsrx` gain moved gate Jaccard 0.562 -> 0.835 and wall F1 by 0.006.  What decides it is
whether ``(gate > 0) & wall`` is non-empty: `clot_ml`'s `physics_mask` seeds from that set,
and when it is empty THIRTEEN physics/advection feature channels go identically zero.  This
reports that directly, per checkpoint, without writing anything onto the packs.
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
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import numpy as np
    import torch

    from src.data_gen.lib.legal_priors import apply_prior_source
    from src.utils.kinematics_select_packs import selection_pack_dir, selection_pack_stems
    from src.utils.kinematics_selection import wall_gate_health
    from src.architecture.kinematics_model_config import (
        build_rgp_deq_from_ctor, resolve_rgp_deq_ctor_kwargs)
    from src.config import PhysicsConfig

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    state = ck.get("model_state_dict", ck.get("state_dict", ck))
    ctor = resolve_rgp_deq_ctor_kwargs(ck.get("model_config"), {})
    model = build_rgp_deq_from_ctor(PhysicsConfig(phase="kinematics"), ctor)
    model.load_state_dict(state, strict=False)
    model = model.to(dev).eval()

    stems = selection_pack_stems()
    if a.limit:
        stems = stems[: a.limit]
    rows = []
    for s in stems:
        p = selection_pack_dir() / f"{s}.pt"
        if not p.is_file():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        d = apply_prior_source(d, "analytic")
        try:
            with torch.no_grad():
                out = model(d.to(dev), solver="anderson")
            pred = (out[0] if isinstance(out, tuple) else out).detach().cpu()
        except torch.OutOfMemoryError:
            print(f"  {s}: OOM, skipped")
            torch.cuda.empty_cache()
            continue
        h = wall_gate_health(pred[:, 0:2], d.to("cpu"))
        if h:
            h["stem"] = s
            rows.append(h)
        torch.cuda.empty_cache()

    if not rows:
        print("no packs scored")
        return 1
    g = lambda k: np.array([r[k] for r in rows], dtype=float)
    ratio = g("fire_pred") / np.maximum(g("fire_gt"), 1e-9)
    print(f"\n{len(rows)} deploy packs   checkpoint {Path(a.checkpoint).name}")
    print(f"  wall gate EMPTY (the failure):     {int(g('empty').sum())} / {len(rows)}")
    print(f"  fire_pred / fire_gt   median {np.median(ratio):.2f}   "
          f"within 0.5-2x: {int(((ratio >= .5) & (ratio <= 2)).sum())}")
    print(f"  p05_ratio    median {np.median(g('p05_ratio')):.2f}  (1.0 = tail matched)")
    print(f"  sr_min_ratio median {np.median(g('sr_min_ratio')):.2f}   "
          f"|log| mean {np.mean(np.abs(np.log(np.maximum(g('sr_min_ratio'), 1e-9)))):.2f}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"[save] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
