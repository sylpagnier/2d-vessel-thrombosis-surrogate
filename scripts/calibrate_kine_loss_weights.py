"""Set Stage-A loss weights from measured gradient share (see `src/utils/loss_calibration.py`).

    python scripts/calibrate_kine_loss_weights.py --graphs 6

One forward/backward per loss term per graph.  No training runs, no sweep.
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
    ap.add_argument("--graphs", type=int, default=6)
    ap.add_argument("--checkpoint", default="", help="reference state; default = resolved ckpt")
    ap.add_argument("--elevate", action="store_true", default=True)
    ap.add_argument("--out", default="outputs/kine_loss_weights.json")
    args = ap.parse_args()

    # Calibrate under the SAME environment training will use: the balance is a property of the
    # configured objective, not of the code alone.  `KINEMATICS_NORMALIZE_SHEAR_GRAD` in
    # particular moves `l_shear_grad`'s gradient by ~9 orders of magnitude.
    import os

    for k, v in (("KINEMATICS_NORMALIZE_SHEAR_GRAD", "1"), ("SPECIES_PRIOR_SOURCE", "analytic")):
        os.environ.setdefault(k, v)
    print(f"[i] NORMALIZE_SHEAR_GRAD={os.environ['KINEMATICS_NORMALIZE_SHEAR_GRAD']}"
          f"  PRIOR_SOURCE={os.environ['SPECIES_PRIOR_SOURCE']}")

    import torch

    from src.config import PhysicsConfig
    from src.core_physics.physics_kernels import PhysicsKernels
    from src.data_gen.lib.legal_priors import apply_prior_source
    from src.data_gen.lib.p2_elevation import elevate_to_p2
    from src.utils.anchor_mask import graph_has_anchor
    from src.utils.kinematics_inference import (
        load_kinematics_predictor, resolve_kinematics_checkpoint)
    from src.utils.kinematics_paths import kinematics_training_graph_dir
    from src.utils.loss_calibration import (
        DEFAULT_SHARES, measure_gradient_norms, weights_from_gradient_norms)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys = PhysicsConfig(phase="kinematics")
    kern = PhysicsKernels(phys_cfg=phys)
    ckpt = resolve_kinematics_checkpoint(args.checkpoint or None)
    model = load_kinematics_predictor(ckpt, dev, cache=False)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    graphs = []
    for f in sorted(kinematics_training_graph_dir(rheology="carreau").glob("*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        if not graph_has_anchor(d):
            continue
        graphs.append(apply_prior_source(elevate_to_p2(d) if args.elevate else d, "analytic"))
        if len(graphs) >= args.graphs:
            break
    if not graphs:
        print("[ERR] no anchor graphs found")
        return 1

    print(f"[i] reference state: {ckpt}")
    print(f"[i] {len(graphs)} graphs, elevate={args.elevate}\n")
    norms = measure_gradient_norms(model, graphs, kern, dev)
    spread = norms.pop("_spread", {})
    weights = weights_from_gradient_norms(norms)

    print(f"{'term':<16}{'|grad| median':>15}{'max/min':>10}{'share':>8}{'weight':>12}")
    for k in sorted(DEFAULT_SHARES, key=lambda x: -DEFAULT_SHARES[x]):
        w = weights.get(k)
        tag = "" if w is not None else "   DROPPED (inert)"
        print(f"{k:<16}{norms.get(k, float('nan')):15.4g}{spread.get(k, float('nan')):10.1f}"
              f"{DEFAULT_SHARES[k]:8.2f}{(w if w is not None else float('nan')):12.4g}{tag}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"checkpoint": str(ckpt), "grad_norms": norms, "grad_spread": spread,
         "shares": DEFAULT_SHARES, "weights": weights}, indent=2))
    print(f"\n[save] {args.out}")
    print("\nWeights are normalised to l_data_kine = 1.0; the absolute scale is the LR's job.")
    print("Re-run from a mid-training checkpoint if the balance drifts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
