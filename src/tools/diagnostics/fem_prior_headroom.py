#!/usr/bin/env python
"""Is the local FEM field a better RGP-DEQ prior than the analytic Poiseuille one?

The model's hard BC is ``u = uv_prior + sdf * r``, so the prior is not just an input feature --
it is the *base point*, and ``r`` is the only thing the network has to produce.  Swapping
``uv_prior`` from the analytic Poiseuille field to the converged FEM field therefore changes
what is being learned from "the flow" to "what the FEM got wrong", without touching the
architecture.  This measures whether that residual is (a) smaller, (b) inside the dynamic range
the decoder can actually emit, and (c) structured enough to be learnable at all.

(b) is the crux.  ``ginodeq.py`` s17 measured the decoder's own residual tail at
``|r| p99/p50 ~ 9`` against labels needing ~25 under the analytic prior -- the LayerNorm shell
the DEQ equilibrium sits on cannot carry per-node amplitude, so the objective could never move
the wall-shear tail.  If the FEM-prior residual needs a tail ratio at or under ~9, that
architectural ceiling stops binding and the retrain is worth running as-is.

Prior arms:

    analytic    Poiseuille magnitude + potential-flow direction   (today's shipped prior)
    fem         local Carreau solve, inlet Dirichlet from COMSOL  (what the clot stack solves)
    fem_legal   local Carreau solve, ANALYTIC inlet profile       (deploy-legal: geometry+BC only)

`fem_legal` is the arm that matters for a shippable model: `fem` reads COMSOL's own inlet
velocity, which a customer vessel does not have.  If the two diverge, the headroom is not real.

    python -m src.tools.diagnostics fem-prior-headroom --cohort --out outputs/diag_fem_prior_headroom.json
"""
from __future__ import annotations

from src.tools.diagnostics._common import bootstrap, biochem_packs_dir

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from src.config import BiochemConfig, NodeFeat, PhysicsConfig  # noqa: E402
from src.core_physics.wall_cohort_splits import CLOT_FREE, DEV, FIT, SEALED  # noqa: E402

PACKS = biochem_packs_dir()

ARMS = ("analytic", "fem", "fem_legal")

NAN = float("nan")


def _cohort_stems():
    skip = set(SEALED) | set(CLOT_FREE)
    out = [a for a in list(FIT) + list(DEV) if a not in skip and (PACKS / (a + ".pt")).exists()]
    for s in ("wound_comsol001", "wound_comsol002", "wound_comsol003"):
        if (PACKS / (s + ".pt")).exists() and s not in out:
            out.append(s)
    return out


def _corr(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return NAN
    return float(np.corrcoef(a, b)[0, 1])


def _jaccard(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    u = int((a | b).sum())
    return float((a & b).sum() / u) if u else NAN


def _wall_band(data, hops=3):
    n = int(data.num_nodes)
    row, col = data.edge_index
    band = data.mask_wall.reshape(-1).bool().clone()
    for _ in range(hops):
        acc = torch.zeros(n, dtype=torch.bool)
        acc.index_put_((row,), band[col], accumulate=False)
        band = band | acc
    return band.numpy()


def _gt_inlet(data):
    if bool(getattr(data, "research_synthetic", False)):
        return None
    y = getattr(data, "y", None)
    if y is None or not torch.is_tensor(y) or y.numel() == 0 or y.shape[1] == 0:
        return None
    cand = y[0, :, 0:2].detach().cpu().numpy()
    if np.isfinite(cand).all() and float(np.abs(cand).max()) > 0.0:
        return cand
    return None


def _solve_fem(data, mesh_path, legal):
    """FEM velocity, non-dimensional, seeded with the analytic prior (same fixed point, fewer iters)."""
    from src.core_physics.local_fem_solver import solve_local_t0_flow

    u0, v0, _, _ = _analytic(data)
    seed = np.stack([u0, v0], 1)
    with contextlib.redirect_stdout(io.StringIO()):
        u_dim = solve_local_t0_flow(
            mesh_path, data, PhysicsConfig(), max_iters=300, tol=1e-9,
            u_gt_inlet_nd=(None if legal else _gt_inlet(data)),
            u_init_nd=seed, verbose=True,
        )
    if isinstance(u_dim, torch.Tensor):
        u_dim = u_dim.numpy()
    nd = np.asarray(u_dim, dtype=np.float64) / float(data.u_ref.reshape(-1)[0])
    return nd[:, 0], nd[:, 1]


def _analytic(data):
    from src.data_gen.lib.legal_priors import build_analytic_priors

    u, v, mu, wss = build_analytic_priors(data)
    return (u.reshape(-1).cpu().numpy().astype(np.float64),
            v.reshape(-1).cpu().numpy().astype(np.float64),
            mu.reshape(-1).cpu().numpy().astype(np.float64),
            wss.reshape(-1).cpu().numpy().astype(np.float64))


#: Envelope sharpness for the ``bc_envelope`` residual parameterisation, in units of 1/sdf_nd.
#: Larger reaches 1 sooner, so less of the residual's dynamic range is spent undoing the
#: division.  ``RGP_DEQ.bc_lambda`` is the trained knob these bracket.
ENVELOPE_LAMBDAS = (10.0, 40.0)


def _residual_stats(prior_uv, g, sdf, band):
    """What the decoder would have to emit, under each residual parameterisation on offer.

    The shipped hard BC is ``u = prior + sdf * r``, so ``r = (y - prior)/sdf`` -- and the
    division is what sets the dynamic range the decoder must span, because ``sdf`` runs over
    three or four decades between the wall and the core while the error itself does not.  That
    division exists to force ``u = prior`` at the wall.  A FEM prior already satisfies no-slip
    exactly (see ``wall_leak``), so for that arm the division buys nothing and costs the whole
    tail -- which is the quantity ``ginodeq.py`` s17 measured the decoder as unable to supply.

    Reported for three parameterisations, so the architecture question is answered by
    measurement rather than by argument:

      ``resid_*``       ``r = (y - prior)/sdf``                     -- today's hard BC
      ``residraw_*``    ``r = y - prior``                           -- no BC factor at all
      ``residenvL_*``   ``r = (y - prior)/(1 - exp(-L*sdf))``       -- ``bc_envelope`` at lambda L

    Nodes at ``sdf <= 0`` are excluded from the divided arms -- there the hard BC pins the
    answer to the prior and the network contributes nothing, so they say nothing about the
    required dynamic range.
    """
    out = {}
    live = sdf > 1e-6
    if not live.any():
        return out
    d = g - prior_uv

    def _tail(vec, tag, mask):
        mag = np.linalg.norm(vec[mask], axis=1)
        if mag.size < 3:
            return
        p50 = float(np.median(mag))
        out[tag + "_p50"] = p50
        out[tag + "_tail"] = float(np.percentile(mag, 99) / max(p50, 1e-30))

    _tail(d / np.maximum(sdf, 1e-12)[:, None], "resid", live)
    _tail(d, "residraw", np.ones_like(live))
    for lam in ENVELOPE_LAMBDAS:
        env = 1.0 - np.exp(-lam * np.maximum(sdf, 0.0))
        _tail(d / np.maximum(env, 1e-12)[:, None], "residenv%g" % lam, live)

    lb = live & band
    if lb.any():
        _tail(d / np.maximum(sdf, 1e-12)[:, None], "resid_wall", lb)
    return out


def _concentration(err, frac=0.10):
    """Share of total squared error carried by the worst ``frac`` of nodes."""
    e = np.sort(err ** 2)[::-1]
    k = max(1, int(round(frac * e.size)))
    tot = float(e.sum())
    return float(e[:k].sum() / tot) if tot > 0 else NAN


def _smoothness(err_vec, edge_index):
    """Correlation of a node's error with its 1-hop neighbour mean.

    A prior residual that is spatially structured is something a message-passing network can
    represent; one that decorrelates over a single edge is mesh noise and no architecture will
    fit it.  Reported per velocity component and averaged.
    """
    row, col = edge_index
    n = err_vec.shape[0]
    deg = np.bincount(row, minlength=n).astype(np.float64)
    ok = deg > 0
    cs = []
    for k in range(err_vec.shape[1]):
        nb = np.bincount(row, weights=err_vec[col, k], minlength=n)
        nb[ok] /= deg[ok]
        cs.append(_corr(err_vec[ok, k], nb[ok]))
    cs = [c for c in cs if c == c]
    return float(np.mean(cs)) if cs else NAN


def _gate(data, bio, u, v, wall, hops, gain):
    """Gate agreement of a candidate field against COMSOL's own gate, in the wall band."""
    import os

    from src.core_physics.physics_wall_model import t0_flow_fields

    gt = t0_flow_fields(data, bio, hops=3, flow_source="gt")
    # Not every pack carries a cached surrogate field, and `data.u0_pred` RAISES rather than
    # returning None on a PyG store that lacks the key -- which killed the sweep 25 vessels in.
    keep_u = getattr(data, "u0_pred", None)
    keep_v = getattr(data, "v0_pred", None)
    data.u0_pred = torch.tensor(u, dtype=torch.float32)
    data.v0_pred = torch.tensor(v, dtype=torch.float32)
    old = os.environ.get("CLOT_PRED_DSRX_GAIN", "")
    os.environ["CLOT_PRED_DSRX_GAIN"] = str(float(gain))
    try:
        pr = t0_flow_fields(data, bio, hops=int(hops), flow_source="pred")
    finally:
        if old:
            os.environ["CLOT_PRED_DSRX_GAIN"] = old
        else:
            os.environ.pop("CLOT_PRED_DSRX_GAIN", None)
        for key, val in (("u0_pred", keep_u), ("v0_pred", keep_v)):
            if val is None:
                del data[key]
            else:
                data[key] = val
    w = np.asarray(wall, dtype=bool)
    return dict(
        dsrx_corr=_corr(pr.dsrx[w], gt.dsrx[w]),
        gate_jaccard=_jaccard(np.asarray(pr.gate)[w] > 0, np.asarray(gt.gate)[w] > 0),
    )


def _score_one(stem, arms, hops, gain):
    from src.clot_ml.v0 import _resolve_anchor_mesh

    data = torch.load(PACKS / (stem + ".pt"), map_location="cpu", weights_only=False)
    if getattr(data, "graph_stem", None) is None:
        data.graph_stem = stem
    bio = BiochemConfig(phase="biochem")
    row = dict(stem=stem, n_nodes=int(data.num_nodes))

    g = data.y[0, :, 0:2].numpy().astype(np.float64)
    sdf = data.x[:, NodeFeat.SDF].reshape(-1).numpy().astype(np.float64)
    wall = np.asarray(data.mask_wall.reshape(-1).bool().numpy())
    band = _wall_band(data, hops=3)
    ei = data.edge_index.numpy()
    mesh_path = _resolve_anchor_mesh(data)

    gn = np.linalg.norm(g, axis=1)
    row["gt_speed_p50"] = float(np.median(gn))
    # How much of the field the hard BC pins to the prior outright.
    row["frac_sdf0"] = float(np.mean(sdf <= 1e-6))

    for arm in arms:
        try:
            if arm == "analytic":
                u, v, _, _ = _analytic(data)
            else:
                u, v = _solve_fem(data, mesh_path, legal=(arm == "fem_legal"))
        except Exception as exc:
            row[arm + "_error"] = type(exc).__name__ + ": " + str(exc)
            continue
        p = np.stack([u, v], 1)
        d = p - g

        den = np.linalg.norm(g)
        row[arm + "_rel_l2"] = float(np.linalg.norm(d) / den) if den > 0 else NAN
        db = np.linalg.norm(g[band])
        row[arm + "_rel_l2_wall"] = float(np.linalg.norm(d[band]) / db) if db > 0 else NAN
        # A prior that does not vanish on the wall breaks the hard BC it is the base point of.
        at0 = sdf <= 1e-6
        row[arm + "_wall_leak"] = (float(np.abs(p[at0]).max() / max(np.abs(g).max(), 1e-30))
                                   if at0.any() else NAN)
        err = np.linalg.norm(d, axis=1)
        row[arm + "_err_top10pct_share"] = _concentration(err, 0.10)
        row[arm + "_err_smooth_1hop"] = _smoothness(d, ei)
        row.update({arm + "_" + k: v_ for k, v_ in _residual_stats(p, g, sdf, band).items()})
        for k, v_ in _gate(data, bio, u, v, wall, hops, gain).items():
            row[arm + "_" + k] = v_
    return row


def _med(rows, key):
    vs = [r[key] for r in rows if isinstance(r.get(key), (int, float))
          and r[key] == r[key] and np.isfinite(r[key])]
    return float(np.median(vs)) if vs else NAN


def main(argv=None):
    bootstrap()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", nargs="*", default=None)
    ap.add_argument("--cohort", action="store_true")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--hops", type=int, default=3, help="dsrx stencil for the candidate field")
    ap.add_argument("--gain", type=float, default=1.0, help="dsrx amplitude gain (1.0 = converged-field treatment)")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    stems = args.stems or (_cohort_stems() if args.cohort
                           else ["comsol001", "comsol020", "comsol041"])
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit("unknown arm " + repr(a) + "; known: " + ", ".join(ARMS))
    print("[i] n=%d arms=%s hops=%d gain=%g" % (len(stems), arms, args.hops, args.gain), flush=True)

    rows = []
    for stem in stems:
        r = _score_one(stem, arms, args.hops, args.gain)
        rows.append(r)
        parts = ["%-22s" % r["stem"]]
        for a in arms:
            if a + "_error" in r:
                parts.append(a + "=ERR")
            else:
                parts.append("%s L2=%.4f sdfTail=%.1f rawTail=%.1f gJ=%.3f"
                             % (a, r[a + "_rel_l2"], r.get(a + "_resid_tail", NAN),
                                r.get(a + "_residraw_tail", NAN),
                                r.get(a + "_gate_jaccard", NAN)))
        print("  ".join(parts), flush=True)
        for a in arms:
            if a + "_error" in r:
                print("    [ERR] %s: %s" % (a, r[a + "_error"]), flush=True)

    print("\n=== medians over %d vessels ===" % len(rows), flush=True)
    cols = ([("rel_l2", "relL2"), ("rel_l2_wall", "relL2wall"), ("wall_leak", "wallLeak"),
             ("resid_p50", "r_p50"), ("resid_tail", "sdf_tail"), ("residraw_tail", "raw_tail")]
            + [("residenv%g_tail" % L, "env%g_tail" % L) for L in ENVELOPE_LAMBDAS]
            + [("err_top10pct_share", "top10%err"), ("err_smooth_1hop", "smooth1hop"),
               ("dsrx_corr", "dsrxCorr"), ("gate_jaccard", "gateJ")])
    print("%-11s" % "arm" + "".join("%11s" % c[1] for c in cols), flush=True)
    for a in arms:
        print("%-11s" % a + "".join("%11.4g" % _med(rows, a + "_" + c[0]) for c in cols), flush=True)
    print("\nDecoder can emit |r| p99/p50 ~ 9 (ginodeq.py s17). An arm whose r_p99/p50 is at or "
          "under that is expressible by the architecture as it stands.", flush=True)

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(arms=arms, hops=args.hops, gain=args.gain, rows=rows), indent=2),
                     encoding="utf-8")
        print("[OK] wrote " + str(p), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
