"""Time-resolved advective transport of the wall source -- per-time physics for the head.

WHY.  The time-conditioned head (`docs/PHASE9_ML.md` 13.9) is given exactly two pieces of
time-varying information: the query time itself, and a **binary** "has the ODE fired by now"
at this node.  Everything else it sees is a t=0 static field.  Off the wall it is given
nothing time-varying at all, because the ODE is a wall object -- and that is precisely where
the temporal arm is weakest (mean-over-time off-wall 0.649 against an oracle 0.84).

PHASE9 12.2 tried to fix this with the owner-threshold rule (an off-wall node fires when its
owner crosses `crit/att`) and measured it **worse than doing nothing** (0.490 against
0.5015), diagnosing that the ODE's `Mat` is biased low so `crit/att` is unreachable.  That
diagnosis is about a hand-written threshold rule.  It says nothing about handing the model
the underlying field and letting it calibrate.

WHAT THIS COMPUTES.  The transport operator of `src/clot_ml/transport.py` is **linear and
time-independent** -- the flow is frozen at t=0, so only the source changes with time.  So
the whole time-resolved off-wall field costs one solve per stored time:

    mat_adv(t) = L^-1 [ Mat_ODE(t) restricted to the wall ]

which is the physics' own answer to "how much deposited species has reached this off-wall
node by time t", under COMSOL's own operator (PHASE7 1.1: `dMat/dt + u.grad(Mat) = 0`,
zero diffusion, wall flux BC).  Alongside it, the two wall-side per-time quantities the head
also never saw: the node's own ODE `Mat(t)` and its owner's, as continuous values rather
than the single fired/not-fired bit.

    python scripts/build_temporal_transport.py
"""
from __future__ import annotations
from src.utils.paths import anchor_packs_dir, get_project_root

import argparse
import time
from pathlib import Path as pathlib_Path

import numpy as np
import torch

REPO = get_project_root()

from src.clot_ml.data import load_cache  # noqa: E402
from src.clot_ml.temporal import ode_trajectory  # noqa: E402
from src.clot_ml.features_v4 import horizon_for  # noqa: E402
from src.clot_ml.transport import _node_volume, _solve_upwind, upwind_operator  # noqa: E402
from src.config import BiochemConfig  # noqa: E402
from src.core_physics.wall_cohort_splits import CLOT_FREE  # noqa: E402

PACKS = anchor_packs_dir()
OUT = REPO / "outputs/temporal_transport"

#: One directory per t=0 flow source.  The channels are an ODE trajectory pushed through an
#: upwind operator, and BOTH are built from the t=0 velocity, so a GT-flow cache is not a
#: valid input for a FEM-flow head -- it would train the timing head on a field the deploy
#: path never sees.  `gt` keeps the historical path so nothing already built moves.
OUT_FOR_FLOW = {"gt": OUT,
                "pred": REPO / "outputs/temporal_transport_pred",
                "fem": REPO / "outputs/temporal_transport_fem"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-times", type=int, default=11)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--flow", default="gt", choices=["gt", "pred", "fem"],
                    help="t=0 velocity for BOTH the ODE trajectory and the upwind operator")
    ap.add_argument("--cache", default="",
                    help="feature cache to read geometry/flow from (default: --flow)")
    ap.add_argument("--out", default="", help="default: per --flow")
    args = ap.parse_args()
    out_dir = pathlib_Path(args.out) if args.out else OUT_FOR_FLOW[args.flow]
    out_dir.mkdir(parents=True, exist_ok=True)
    bio = BiochemConfig(phase="biochem")
    cache = load_cache(args.cache or args.flow)

    for a, S in sorted(cache.items()):
        if a in CLOT_FREE:
            # No onset to resolve, and `eval_strict_temporal.py` excludes them outright.
            continue
        dst = out_dir / f"{a}.npz"
        if dst.exists() and not args.force:
            print("[skip] %s" % a, flush=True)
            continue
        t0 = time.time()
        d = torch.load(PACKS / f"{a}.pt", map_location="cpu", weights_only=False)
        if args.flow == "fem":
            from src.clot_ml.v0 import solve_fem_into_pack
            if not str(getattr(d, "graph_stem", "") or ""):
                d.graph_stem = a
            solve_fem_into_pack(d)
        T = int(d.y.shape[0])
        times = [int(round(x)) for x in np.linspace(0, T - 1, args.n_times)]
        traj, _ = ode_trajectory(d, bio, flow=args.flow)      # [T, N], wall-supported

        wall, ei, owner = S["wall"], S["edge_index"], S["owner"]
        pos = S["pos"].astype(np.float64)
        u, v = S["u"].astype(np.float64), S["v"].astype(np.float64)
        # ONE definition of the transport horizon (`features_v4.horizon_for`).  It
        # excludes the SOLID boundary from the bulk-speed median, not just the healthy
        # wall -- an inline `~wall` copy would compute a different horizon than the
        # cache builder on a wound pack (a silent train/deploy skew).
        H = horizon_for(pos, u, v, np.asarray(S.get("solid", wall), dtype=bool))

        # one factorisation-worth of work per time; the operator itself never changes
        F, out = upwind_operator(pos, ei, u, v)
        vol = _node_volume(pos, ei)
        adv = np.zeros((len(times), len(wall)), dtype=np.float32)
        own = np.zeros_like(adv)
        slf = np.zeros_like(adv)
        for j, ti in enumerate(times):
            src = np.zeros(len(wall))
            src[wall] = np.maximum(traj[ti][wall], 0.0)
            adv[j] = _solve_upwind(F, out, src * vol, vol, H).astype(np.float32)
            own[j] = traj[ti][owner].astype(np.float32)
            slf[j] = traj[ti].astype(np.float32)
        np.savez_compressed(dst, times=np.array(times), T=T,
                            mat_adv_t=adv, mat_owner_t=own, mat_self_t=slf)
        print("[ok  ] %-11s T=%d  %d times  %.1fs" % (a, T, len(times), time.time() - t0),
              flush=True)
    print("done -> %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
