"""Set Stage-A loss weights from measured gradient share (see `src/utils/loss_calibration.py`).

    python scripts/calibrate_kine_loss_weights.py --graphs 6

One forward/backward per loss term per graph.  No training runs, no sweep.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root by marker, not by depth: this file may move between
# scripts/ and scripts/<subdir>/ without silently resolving one level off.
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
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
    os.environ.setdefault("KINEMATICS_PDE_FLOOR", "1")
    max_nodes = int(os.environ.get("KINEMATICS_MAX_NODES", "0") or 0)
    pde_floor = os.environ["KINEMATICS_PDE_FLOOR"].strip().lower() not in ("0", "false", "no", "off")
    print(f"[i] NORMALIZE_SHEAR_GRAD={os.environ['KINEMATICS_NORMALIZE_SHEAR_GRAD']}"
          f"  PRIOR_SOURCE={os.environ['SPECIES_PRIOR_SOURCE']}"
          f"  PDE_FLOOR={int(pde_floor)}"
          f"  MAX_NODES={max_nodes or 'none'}")

    import torch

    from src.config import PhysicsConfig
    from src.core_physics.physics_kernels import PhysicsKernels
    from src.data_gen.lib.legal_priors import apply_prior_source
    from src.data_gen.lib.p2_elevation import elevate_to_p2
    from src.utils.anchor_mask import graph_has_anchor
    from src.utils.kinematics_inference import (
        load_kinematics_predictor, resolve_kinematics_checkpoint)
    from src.utils.kinematics_paths import kinematics_training_graph_dir
    from src.utils.kinematics_physics_terms import attach_pde_floors
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
        g = apply_prior_source(elevate_to_p2(d) if args.elevate else d, "analytic")
        # This script takes one backward PER TERM, so it peaks higher than a training step.
        # Same cap as training (`KINEMATICS_MAX_NODES`), honoured here because the script
        # builds its own list rather than going through `load_dataset`.
        if max_nodes and int(g.num_nodes) > max_nodes:
            continue
        # `l_cont` / `l_mom` are hinged against the labels' own PDE residual in training
        # (`_attach_pde_floors`).  Calibrating without the floor measures a different objective:
        # un-floored, both terms carry the labels' near-wall discretisation residual, which on
        # the severe-stenosis vessels is larger than anything the model contributes.
        if pde_floor:
            attach_pde_floors(g, kern)
        graphs.append(g)
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
