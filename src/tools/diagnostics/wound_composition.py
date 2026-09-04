"""Does the wound module DELETE clot that `clot_gnn_v4` had right?

`compose_with_v4` hands the wound module total control of `owned = wound U its first shell`:

    mask[owned]  = wound_out["mask"][owned]        # override, not union
    onset[owned] = wound_out["onset"][owned]

and inside the ODE `gate_fields` does the same thing one level down -- on a wound node the
healthy t=0 gate is DISCARDED and replaced by the scalar `g_pre` / `g_post`:

    pre = torch.where(wnd, g_pre, base)            # base = V["gate"] * solid

Both are only safe if the healthy law has nothing to say on the wound.  On
`wound_comsol001/002` it has not: the gate is 0% ON at every wound node, so override and
addition coincide bit-for-bit.  On `wound_comsol003` the gate is already 42% open at step 3
(WOUND_PROGRESS 11.1) -- the one vessel that carries all the residual error.

`WOUND_PROGRESS` 1 flags this exactly ("override vs additive is currently unobservable ...
it becomes a real question after gelation") and it was never measured.  This script measures
it, on the three quantities that decide whether it matters:

  1. how many wound nodes have a NON-ZERO healthy gate -- the override/additive discriminator;
  2. how many nodes the override REMOVES from v4's committed set, and how many of those are
     GT clot (i.e. true positives destroyed by composition);
  3. how many nodes the override commits LATER than v4 would have.

    python scripts/diag_wound_composition.py
"""
from __future__ import annotations

from src.tools.diagnostics._common import bootstrap, biochem_packs_dir, repo_root

import sys
from pathlib import Path

import numpy as np
import torch


from src.clot_ml.locked import load_temporal_v4, predict_temporal_v4  # noqa: E402
from src.clot_ml.wound import (  # noqa: E402
    prepare_vessel, predict_wound_series, wound_owned_masks,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402

PACKS = biochem_packs_dir()
VESSELS = ("wound_comsol001", "wound_comsol002", "wound_comsol003")

#: the shipped constants (`clot_gnn_v4w` manifest, WOUND_PROGRESS 12.1)
G_PRE, G_POST = 1.98, 14.28


def gt_final(data, phys) -> np.ndarray:
    from src.core_physics.deploy_time_index import resolve_deploy_eval_time_index
    from src.core_physics.t0_mu_physics import gt_clot_phi_at_time

    ti = resolve_deploy_eval_time_index(int(data.y.shape[0]))
    return (gt_clot_phi_at_time(data, ti, phys, device=torch.device("cpu"))
            .reshape(-1).numpy() > 0.5)


def main(argv: list[str] | None = None) -> int:
    bootstrap()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    bundle = load_temporal_v4("clot_gnn_v4")

    print("\n[1] THE OVERRIDE/ADDITIVE DISCRIMINATOR -- healthy gate at wound nodes (t=0)")
    print("%-18s %7s %9s %9s %10s" % ("vessel", "n_wnd", "gate>0", "frac", "med gate"))
    prepared = {}
    for name in VESSELS:
        p = PACKS / f"{name}.pt"
        if not p.exists():
            print("%-18s [skip] no pack" % name)
            continue
        data = torch.load(p, map_location="cpu", weights_only=False)
        V = prepare_vessel(data, bio, flow="gt")
        prepared[name] = (data, V)
        wnd = V["wound"]
        g = np.asarray(V["gate"].numpy())[wnd]
        print("%-18s %7d %9d %9.1f%% %10.4g"
              % (name, wnd.sum(), int((g > 0).sum()), 100.0 * float((g > 0).mean()),
                 float(np.median(g))))
    print("    gate>0 == 0 means override and addition are BIT-IDENTICAL on that vessel.")

    print("\n[2] WHAT THE OVERRIDE DOES TO v4's SET  (owned = wound U first shell)")
    print("%-18s %7s %7s %7s %8s %8s %8s"
          % ("vessel", "owned", "v4_own", "wnd_own", "DELETED", "of-which", "ADDED"))
    print("%-18s %7s %7s %7s %8s %8s %8s"
          % ("", "", "", "", "", "GT-pos", ""))
    for name, (data, V) in prepared.items():
        times = list(range(int(data.y.shape[0])))
        base = predict_temporal_v4(bundle, data, times, flow="gt")
        w = predict_wound_series(data, bio, times, g_pre=G_PRE, g_post=G_POST, prepared=V)
        gt = gt_final(data, phys)
        owned = w["owned"]
        bm, wm = np.asarray(base["mask"], bool), np.asarray(w["mask"], bool)
        deleted = bm & owned & ~wm
        added = ~bm & owned & wm
        print("%-18s %7d %7d %7d %8d %8d %8d"
              % (name, owned.sum(), int((bm & owned).sum()), int((wm & owned).sum()),
                 int(deleted.sum()), int((deleted & gt).sum()), int(added.sum())))

        # timing: among nodes BOTH arms commit, does the override push them later?
        both = bm & wm & owned
        if both.any():
            bo = np.asarray(base["onset"], float)[both]
            wo = np.asarray(w["onset"], float)[both]
            ok = (bo >= 0) & (wo >= 0)
            if ok.any():
                d = wo[ok] - bo[ok]
                print("%-18s   onset shift on %d co-committed nodes: "
                      "median %+.1f steps (later>0), max %+.1f"
                      % ("", int(ok.sum()), float(np.median(d)), float(d.max())))

        # how deep does GT wound clot run vs the ONE shell the module owns?
        _, owned_off, _ = wound_owned_masks(data)
        solid = V["solid"]
        deep = gt & ~solid & ~owned_off
        print("%-18s   GT off-boundary clot beyond the owned shell: %d nodes "
              "(v4's responsibility)" % ("", int(deep.sum())))
    print("\n    DELETED > 0 means composition is destroying v4 predictions.")
    print("    'of-which GT-pos' > 0 means it is destroying CORRECT ones.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
