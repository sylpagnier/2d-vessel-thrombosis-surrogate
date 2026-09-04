"""Extend the PHASE9 feature cache with the two physics families v3 never used.

Both come straight out of `docs/PHASE7_FINDINGS.md` and both were *measured* there and then
left unshipped.

**(A) ADVECTIVE TRANSPORT** (`src/clot_ml/transport.py`).  PHASE7 1.1 read the production
`.mph` and found `Mat` is a domain field under `dMat/dt + u.grad(Mat) = 0` with a wall flux
BC and **zero diffusion**.  Every off-wall channel v3 has -- `log_mat_owner`, `gate_owner`,
`is_shell`, `hop_wall` -- is a *nearest wall node* rule, i.e. it moves information along the
mesh NORMAL, the one direction the equation does not transport along.  Solving COMSOL's own
operator gives the field the equation actually predicts, and (measured, 19 vessels) it
orders off-wall GT `Mat` better than the owner rule on exactly the high-burden vessels that
dominate the off-wall score: p032 0.72 -> 0.83, p041 0.42 -> 0.65, p044 0.37 -> 0.61,
p037 0.58 -> 0.77.  It is worse on p020/p021/p035, so it is a **complement** to the owner
rule, not a replacement -- which is why both are kept as features and the network chooses.

**(B) THE SEPARATION BRANCH AS AN INDICATOR.**  PHASE7 12.3: the gate is `A + B` with
`A = (L/gamma_m)*|d(sr,x)|` a MAGNITUDE and `B = 1` an INDICATOR, and `A` outweighs `B` by
~50x wherever it fires.  Dropping `A` from the *rate* takes oracle ordering 0.492 -> 0.703
and removes every anti-correlation, because MLS `d(sr,x)` on a coarsened graph is accurate
in SIGN but not in magnitude (12.2: rank 0.992 at t=0, ratio 1.026 -- but the derivative
collapses to 0.346 once a clot exists).  PHASE7 12.5 ranked shipping this #2 and it never
happened.

Cap-to-zero is not usable as written: on 4 of 19 vessels `A` is the only source, so a zero
cap gives them no physics at all.  The variant built here instead makes `A` an **indicator**
like `B`,

    gate_ind = 1[dsrx < sgt] + 1[sr < lss]        against  gate = (L/gm)*|dsrx|*1[..] + 1[..]

which keeps `gate_ind > 0` exactly where `gate > 0` -- **the mask is bit-identical**, so
this changes only the rate, which is the separation 12.3 asks for -- and it introduces no
new parameter.

    python scripts/build_clot_ml_cache_v4.py
"""
from __future__ import annotations
from src.utils.units import M_TO_CM
from src.utils.paths import anchor_packs_dir, get_project_root

import argparse
import time

import numpy as np

from src.clot_ml import feature_fingerprint as _fp
import torch
from scipy.stats import spearmanr

REPO = get_project_root()

from src.clot_ml.features_v4 import (  # noqa: E402
    horizon_for, indicator_physics, new_channels,
)
from src.clot_ml.v0 import solve_fem_into_pack  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.wall_cohort_splits import CLOT_FREE, DEV, FIT, MIN_T, SEALED  # noqa: E402

PACKS = anchor_packs_dir()

#: the two halves must be built from the same velocity field -- `--flow pred` reads the
#: predicted-flow v3 cache, which `scripts/build_clot_ml_cache.py --flow pred` writes.
SRC_FOR_FLOW = {"gt": "outputs/clot_ml_cache_gt", "pred": "outputs/clot_ml_cache_pred",
                "fem": "outputs/clot_ml_cache_fem"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default="gt", choices=["gt", "pred", "fem"],
                    help="velocity field for BOTH the source v3 cache and the v4 block")
    ap.add_argument("--out", default="",
                    help="default: outputs/clot_ml_cache_v4 (gt) / _v4_pred (pred)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--src", default="",
                    help="override the source v3 cache (default: per --flow)")
    ap.add_argument("--only", default="", help="comma-separated anchors")
    ap.add_argument("--include-sealed", action="store_true",
                    help="also cache WALL_COHORT_V2_GENERALIZATION (007/013/031/043). SEALED is spent once (docs/SEALED_SPLIT.md): caching it is not spending it, but "
                         "training or selecting on it is -- keep it out of both.")
    args = ap.parse_args()
    SRC = REPO / (args.src or SRC_FOR_FLOW[args.flow])
    if not SRC.exists():
        print("[ERR] source v3 cache %s missing -- run "
              "`python scripts/build_clot_ml_cache.py --flow %s` first"
              % (SRC, args.flow), flush=True)
        return 2
    # The GT default names the ORPHAN, not the live cache.  `outputs/clot_ml_cache_v4` is the
    # 19-vessel 08-17 cache from the `clot_gnn_v1` era; the live 68-column GT cache that every
    # model since 08-23 trained on is `clot_ml_cache_v5`.  Left pointing at `_v4` so a bare
    # rerun cannot silently overwrite the live cache -- pass `--out outputs/clot_ml_cache_v5`
    # deliberately when that is what you mean.  See `src/clot_ml/data.load_cache`.
    _DEFAULT_OUT = {"gt": "outputs/clot_ml_cache_v4",
                    "pred": "outputs/clot_ml_cache_v4_pred",
                    "fem": "outputs/clot_ml_cache_v4_fem"}
    out = REPO / (args.out or _DEFAULT_OUT[args.flow])
    out.mkdir(parents=True, exist_ok=True)
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)

    diag = []
    todo = list(FIT) + list(DEV) + list(CLOT_FREE)
    if args.include_sealed:
        todo += [a for a in SEALED if a not in todo]
    only = [x.strip() for x in args.only.split(",") if x.strip()]
    if only:
        todo = [a for a in todo if a in only]
    for a in todo:
        src = SRC / f"{a}.npz"
        dst = out / f"{a}.npz"
        if not src.exists():
            continue
        if dst.exists() and not args.force:
            print("[skip] %s" % a, flush=True)
            continue
        d = torch.load(PACKS / f"{a}.pt", map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < MIN_T:
            continue
        if args.flow == "fem":
            # The v4 block differentiates the SAME field the v3 half was built from, so the
            # FEM solve has to be in the pack here too, not just in the source cache.
            if not str(getattr(d, "graph_stem", "") or ""):
                d.graph_stem = a
            solve_fem_into_pack(d)
        t0 = time.time()
        z = np.load(src, allow_pickle=True)
        S = {k: z[k] for k in z.files}
        wall = S["wall"]
        mat_ind, onset_ind, gate_ind = indicator_physics(d, bio, wall, flow=args.flow)
        NC = new_channels(S, mat_ind, onset_ind, gate_ind, crit)

        cols = [str(c) for c in S["cols"]]
        order = sorted(NC)
        X = np.concatenate([S["X"]] + [NC[k].reshape(-1, 1) for k in order], axis=1)
        S["X"] = X.astype(np.float32)
        S["cols"] = np.array(cols + order)
        S["mat_ind"] = np.asarray(mat_ind, np.float32)
        S["gate_ind"] = np.asarray(gate_ind, np.float32)
        # See src/clot_ml/feature_fingerprint.py: a content hash of the feature
        # builders, so a cache written before a features change is detectably stale.
        np.savez_compressed(dst, **_fp.stamp(dict(S)))

        # --- diagnostics: does either family order GT Mat better than what v3 has? ----
        mg = S["mat_gt"].astype(np.float64)
        mp = S["mat_phys"].astype(np.float64)
        owner = S["owner"]

        def r(x, m):
            return (float(spearmanr(x[m], mg[m]).statistic)
                    if m.sum() > 8 and np.ptp(x[m]) > 0 else float("nan"))

        offlive = (~wall) & (mg > 0)
        wlive = wall & (mg > 0)
        row = dict(a=a,
                   off_owner=r(mp[owner], offlive), off_adv=r(NC["log_mat_adv"], offlive),
                   off_est=r(NC["log_mat_off_est"], offlive),
                   w_phys=r(mp, wlive), w_ind=r(np.asarray(mat_ind, float), wlive))
        diag.append(row)
        print("[ok  ] %-11s +%d ch  | OFF owner %+.3f adv %+.3f est %+.3f | WALL phys %+.3f ind %+.3f  %.1fs"
              % (a, len(order), row["off_owner"], row["off_adv"], row["off_est"],
                 row["w_phys"], row["w_ind"], time.time() - t0), flush=True)

    if diag:
        print("\nmean rank vs GT Mat (n=%d)" % len(diag))
        for k in ("off_owner", "off_adv", "off_est", "w_phys", "w_ind"):
            print("  %-10s %+.4f" % (k, np.nanmean([d_[k] for d_ in diag])))
    print("done -> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
