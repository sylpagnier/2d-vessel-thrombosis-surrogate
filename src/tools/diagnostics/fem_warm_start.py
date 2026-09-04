#!/usr/bin/env python
"""Does seeding the local FEM solve with the RGP-DEQ field make it cheaper?

Picard's step is ``x <- A(x_k)^-1 b``: the iterate enters only as the wind the operator is
linearised about, so the initial guess cannot move the fixed point -- only the number of
iterations spent getting there.  The cold solve starts from ``x_0 = 0``, i.e. from a wind that
is wrong everywhere by the full magnitude of the answer, and pays an opening transient (which
is exactly what the ``damping=0.5`` under-relaxation exists to survive).  The RGP-DEQ already
predicts that field at ~0.12 rel-L2 on most anchors, in milliseconds.

Arms, all solved to the same relative tolerance on the same mesh with the same inlet BC:

    cold        x_0 = 0, damping 0.5                 -- today's shipped path
    warm_deq    x_0 = the pack's cached `u0_pred`    -- the experiment
    warm_deq_d1 x_0 = `u0_pred`, damping 1.0         -- warm start + no under-relaxation
    warm_ana    x_0 = the analytic Poiseuille prior  -- control: is it the DEQ, or any seed?
    warm_ana_d1 x_0 = the analytic prior, damping 1.0 -- the seed that needs no checkpoint

`same_fp` is the max-norm difference between an arm's converged field and the cold one,
relative to the cold field's own max-norm.  It is the claim being checked: a warm start that
changes the answer is a bug, not a speedup.

    python -m src.tools.diagnostics fem-warm-start --cohort --out outputs/diag_fem_warm_start.json
"""
from __future__ import annotations

from src.tools.diagnostics._common import biochem_packs_dir, gt_inlet

import argparse
import contextlib
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from src.config import PhysicsConfig  # noqa: E402
from src.core_physics.wall_cohort_splits import CLOT_FREE, DEV, FIT, SEALED  # noqa: E402

PACKS = biochem_packs_dir()

ARMS = ("cold", "warm_deq", "warm_deq_d1", "warm_ana", "warm_ana_d1")

NAN = float("nan")


def _cohort_stems() -> list[str]:
    skip = set(SEALED) | set(CLOT_FREE)
    out = [a for a in list(FIT) + list(DEV) if a not in skip and (PACKS / f"{a}.pt").exists()]
    for s in ("wound_comsol001", "wound_comsol002", "wound_comsol003"):
        if (PACKS / f"{s}.pt").exists() and s not in out:
            out.append(s)
    return out


def _deq_field(data):
    u0 = getattr(data, "u0_pred", None)
    v0 = getattr(data, "v0_pred", None)
    if u0 is None or v0 is None:
        return None
    f = np.stack([u0.reshape(-1).numpy(), v0.reshape(-1).numpy()], 1).astype(np.float64)
    return f if np.isfinite(f).all() else None


def _analytic_field(data):
    from src.data_gen.lib.legal_priors import build_analytic_priors

    try:
        u, v, _, _ = build_analytic_priors(data)
    except Exception:
        return None
    f = np.stack([u.reshape(-1).cpu().numpy(), v.reshape(-1).cpu().numpy()], 1).astype(np.float64)
    return f if np.isfinite(f).all() else None


def _run_arm(mesh_path, data, u_init, damping):
    """One solve, timed, with the iteration and factorisation counts read off the log."""
    from src.core_physics.local_fem_solver import solve_local_t0_flow

    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        u_dim = solve_local_t0_flow(
            mesh_path, data, PhysicsConfig(), max_iters=300, tol=1e-9,
            u_gt_inlet_nd=gt_inlet(data), damping=damping, u_init_nd=u_init, verbose=True,
        )
    secs = time.perf_counter() - t0
    log = buf.getvalue()
    lus = [int(m) for m in re.findall(r"lu=(\d+)", log)]
    if isinstance(u_dim, torch.Tensor):
        u_dim = u_dim.numpy()
    u_ref = float(data.u_ref.reshape(-1)[0])
    return dict(
        secs=round(secs, 2),
        iters=log.count("Iter "),
        n_lu=(lus[-1] if lus else 0),
        converged="did not converge" not in log,
        field=np.asarray(u_dim, dtype=np.float64) / u_ref,
    )


def _rel_l2(u, g):
    den = np.linalg.norm(g)
    return float(np.linalg.norm(u - g) / den) if den > 0 else NAN


def _score_one(stem, arms):
    from src.clot_ml.v0 import _resolve_anchor_mesh

    data = torch.load(PACKS / (stem + ".pt"), map_location="cpu", weights_only=False)
    if getattr(data, "graph_stem", None) is None:
        data.graph_stem = stem
    row = dict(stem=stem, n_nodes=int(data.num_nodes))

    g = data.y[0, :, 0:2].numpy().astype(np.float64)
    mesh_path = _resolve_anchor_mesh(data)

    seeds = {
        "cold": (None, 0.5),
        "warm_deq": (_deq_field(data), 0.5),
        "warm_deq_d1": (_deq_field(data), 1.0),
        "warm_ana": (_analytic_field(data), 0.5),
        "warm_ana_d1": (_analytic_field(data), 1.0),
    }
    # The seed's own accuracy, so a speedup can be read against how good a guess it was.
    for tag, key in (("deq", "warm_deq"), ("ana", "warm_ana")):
        f = seeds[key][0]
        row["seed_" + tag + "_rel_l2"] = _rel_l2(f, g) if f is not None else NAN

    cold_field = None
    for arm in arms:
        u_init, damp = seeds[arm]
        if arm != "cold" and u_init is None:
            row[arm + "_error"] = "no seed field on this pack"
            continue
        try:
            r = _run_arm(mesh_path, data, u_init, damp)
        except Exception as exc:
            row[arm + "_error"] = type(exc).__name__ + ": " + str(exc)
            continue
        f = r.pop("field")
        if arm == "cold":
            cold_field = f
        row[arm + "_secs"] = r["secs"]
        row[arm + "_iters"] = r["iters"]
        row[arm + "_n_lu"] = r["n_lu"]
        row[arm + "_converged"] = r["converged"]
        row[arm + "_rel_l2"] = _rel_l2(f, g)
        if cold_field is not None:
            scale = max(float(np.abs(cold_field).max()), 1e-30)
            row[arm + "_same_fp"] = float(np.abs(f - cold_field).max() / scale)
    return row


def _med(rows, key):
    vs = [r[key] for r in rows if isinstance(r.get(key), (int, float))
          and r[key] == r[key] and np.isfinite(r[key])]
    return float(np.median(vs)) if vs else NAN


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", nargs="*", default=None)
    ap.add_argument("--cohort", action="store_true",
                    help="FIT+DEV clot-carrying packs plus the wounds")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    stems = args.stems or (_cohort_stems() if args.cohort
                           else ["comsol001", "comsol020", "comsol041"])
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit("unknown arm " + repr(a) + "; known: " + ", ".join(ARMS))
    if "cold" not in arms:
        raise SystemExit("the 'cold' arm is the baseline every other arm is measured against")
    print("[i] n=%d arms=%s" % (len(stems), arms), flush=True)

    rows = []
    for stem in stems:
        r = _score_one(stem, arms)
        rows.append(r)
        parts = ["%-22s n=%6d" % (r["stem"], r["n_nodes"])]
        for a in arms:
            if a + "_error" in r:
                parts.append(a + "=ERR")
            else:
                parts.append("%s %3dit/%6.1fs L2=%.4f"
                             % (a, r[a + "_iters"], r[a + "_secs"], r[a + "_rel_l2"]))
        print("  ".join(parts), flush=True)
        for a in arms:
            if a + "_error" in r:
                print("    [ERR] %s: %s" % (a, r[a + "_error"]), flush=True)

    print("\n=== medians ===", flush=True)
    print("seed rel-L2 vs COMSOL: deq %.4f  analytic %.4f"
          % (_med(rows, "seed_deq_rel_l2"), _med(rows, "seed_ana_rel_l2")), flush=True)
    base_it, base_s = _med(rows, "cold_iters"), _med(rows, "cold_secs")
    print("%-12s %7s %8s %8s %8s %5s %7s %9s"
          % ("arm", "iters", "secs", "x iters", "x secs", "n_lu", "relL2", "same_fp"), flush=True)
    for a in arms:
        it, sc = _med(rows, a + "_iters"), _med(rows, a + "_secs")
        print("%-12s %7.1f %8.2f %8.2f %8.2f %5.1f %7.4f %9.2e"
              % (a, it, sc, (base_it / it) if it else NAN, (base_s / sc) if sc else NAN,
                 _med(rows, a + "_n_lu"), _med(rows, a + "_rel_l2"),
                 _med(rows, a + "_same_fp")), flush=True)

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(arms=arms, rows=rows), indent=2), encoding="utf-8")
        print("[OK] wrote " + str(p), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
