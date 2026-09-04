#!/usr/bin/env python
"""Is the local FEM's own error PREDICTABLE from quantities a deploy run can compute?

The residual head's job under ``prior_source="fem"`` is to emit ``e = y - fem``.  Its inputs are
geometry and the prior -- but ``e`` is not a property of the geometry, it is a property of the
DISCRETISATION: where the mesh was too coarse for the velocity gradient, and where the solver's
artificial viscosity (``art_visc``, a fitted stand-in for COMSOL's own stabilisation) did the
most work.  A head that cannot see those cannot localise the error it is asked to correct, which
is the leading explanation for why the ReZero head buys wall-band gate agreement and still loses
global rel-L2.

This measures whether that is fixable by adding input channels, BEFORE any are added.  Every
candidate is computable from the mesh and the FEM solution alone -- no COMSOL -- so any of them
is deploy-legal:

    art_frac    art_visc*0.5*rho*|u|*h / (mu_carreau + that)  -- the fraction of this node's
                viscosity that is stabilisation rather than rheology.  The solver's own
                docstring says this term is what reattached the post-stenotic bubble early.
    cell_re     rho*|u|*h/mu -- local cell Reynolds, the classic where-is-the-scheme-strained
    absdiv      |div u| of the FEM field on the graph operator -- discretisation residual
    shear       the FEM shear rate
    gradspeed   |grad |u|| -- where the field is hardest to resolve
    sdf         distance to the wall, as a control: geometry the head ALREADY has

`r2_lovo` is the leave-one-vessel-out ridge R^2 for |e| from all indicators together.  It is the
number that decides the arm: a head cannot beat the information in its inputs, so an R^2 near
zero means new channels will not help and the ceiling is elsewhere.

    python -m src.tools.diagnostics fem-error-indicators --cohort
"""
from __future__ import annotations

from src.tools.diagnostics._common import bootstrap, biochem_packs_dir

import argparse
import json
from pathlib import Path

import numpy as np
import torch

PACKS = biochem_packs_dir()
NAN = float("nan")

INDICATORS = ("art_frac", "cell_re", "absdiv", "shear", "gradspeed", "sdf")


def _cohort_stems():
    from src.core_physics.wall_cohort_splits import CLOT_FREE, DEV, FIT, SEALED

    skip = set(SEALED) | set(CLOT_FREE)
    return [a for a in list(FIT) + list(DEV) if a not in skip and (PACKS / f"{a}.pt").exists()]


def _corr(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    if a.size < 3 or a.std() < 1e-14 or b.std() < 1e-14:
        return NAN
    return float(np.corrcoef(a, b)[0, 1])


def _wall_band(data, hops=3):
    n = int(data.num_nodes)
    row, col = data.edge_index
    band = data.mask_wall.reshape(-1).bool().clone()
    for _ in range(hops):
        acc = torch.zeros(n, dtype=torch.bool)
        acc.index_put_((row,), band[col], accumulate=False)
        band = band | acc
    return band.numpy()


def _features(data, uv_fem):
    """The candidate indicators at every node, in the pack's own non-dimensional units."""
    from src.config import PhysicsConfig
    from src.core_physics.mls_gradient import build_mls_gradient, node_positions, shear_rate_2d

    ph = PhysicsConfig()
    pos = node_positions(data)
    ei = data.edge_index.detach().cpu().numpy()
    Dx, Dy = build_mls_gradient(pos, ei)
    u, v = uv_fem[:, 0], uv_fem[:, 1]
    ux, uy, vx, vy = Dx @ u, Dy @ u, Dx @ v, Dy @ v
    gamma = shear_rate_2d(ux, uy, vx, vy)                       # non-dim shear rate

    u_ref = float(data.u_ref.reshape(-1)[0])
    d_bar = float(data.d_bar.reshape(-1)[0])
    lam_nd = ph.lam * (u_ref / max(d_bar, 1e-12))
    ref = float(ph.mu_viscosity_nd_scale)
    mu_car = (ph.mu_inf / ref) + ((ph.mu_0 / ref) - (ph.mu_inf / ref)) * (
        1.0 + (lam_nd * gamma) ** ph.a) ** ((ph.n - 1.0) / ph.a)

    speed = np.sqrt(u * u + v * v)
    # Local element size, non-dim: median edge length at each node.
    row, col = ei
    elen = np.linalg.norm(pos[row] - pos[col], axis=1)
    h = np.zeros(pos.shape[0])
    np.add.at(h, row, elen)
    cnt = np.bincount(row, minlength=pos.shape[0]).astype(np.float64)
    h[cnt > 0] /= cnt[cnt > 0]
    h[cnt == 0] = np.median(elen)

    # `art_visc * 0.5 * rho * |u| * h`, in the same non-dim viscosity unit as `mu_car`.
    art = 0.70 * 0.5 * ph.rho * (speed * u_ref) * (h * d_bar) / ref
    sdf = data.x[:, 2].detach().cpu().numpy().astype(np.float64)
    gs = np.sqrt((Dx @ speed) ** 2 + (Dy @ speed) ** 2)

    return {
        "art_frac": art / np.maximum(mu_car + art, 1e-30),
        "cell_re": ph.rho * (speed * u_ref) * (h * d_bar) / np.maximum(mu_car * ref, 1e-30),
        "absdiv": np.abs(ux + vy),
        "shear": gamma,
        "gradspeed": gs,
        "sdf": sdf,
    }


def _ridge_lovo(X_by_v, y_by_v, alpha=1.0):
    """Leave-one-vessel-out ridge R^2 -- the honest predictability of |e| from the indicators."""
    stems = list(X_by_v)
    if len(stems) < 3:
        return NAN
    num, den = 0.0, 0.0
    for held in stems:
        tr = [s for s in stems if s != held]
        Xtr = np.concatenate([X_by_v[s] for s in tr])
        ytr = np.concatenate([y_by_v[s] for s in tr])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
        Xs = (Xtr - mu) / sd
        Xs = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
        w = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(Xs.shape[1]), Xs.T @ ytr)
        Xh = (X_by_v[held] - mu) / sd
        Xh = np.hstack([Xh, np.ones((Xh.shape[0], 1))])
        pred = Xh @ w
        yh = y_by_v[held]
        num += float(((yh - pred) ** 2).sum())
        den += float(((yh - ytr.mean()) ** 2).sum())
    return float(1.0 - num / den) if den > 0 else NAN


def main(argv=None):
    bootstrap()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", nargs="*", default=None)
    ap.add_argument("--cohort", action="store_true")
    ap.add_argument("--band-only", action="store_true",
                    help="restrict the fit to the wall band the gate metric reads")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    from src.data_gen.lib.legal_priors import build_fem_priors

    stems = args.stems or (_cohort_stems() if args.cohort
                           else ["comsol003", "comsol008", "comsol015", "comsol021"])
    print("[i] %d vessels, band_only=%s" % (len(stems), args.band_only), flush=True)

    per_corr = {k: [] for k in INDICATORS}
    X_by_v, y_by_v = {}, {}
    for stem in stems:
        f = PACKS / f"{stem}.pt"
        if not f.is_file():
            continue
        data = torch.load(f, map_location="cpu", weights_only=False)
        data.graph_stem = stem
        try:
            u, v, _, _ = build_fem_priors(data)
        except Exception as exc:
            print("%-14s FEM prior failed: %s" % (stem, exc), flush=True)
            continue
        uv = np.stack([u.cpu().numpy(), v.cpu().numpy()], 1).astype(np.float64)
        y = data.y[0, :, 0:2].numpy().astype(np.float64)
        e = np.linalg.norm(y - uv, axis=1)
        feats = _features(data, uv)
        sel = _wall_band(data) if args.band_only else np.ones(len(e), dtype=bool)
        for k in INDICATORS:
            per_corr[k].append(_corr(feats[k][sel], e[sel]))
        X_by_v[stem] = np.stack([feats[k][sel] for k in INDICATORS], 1)
        y_by_v[stem] = e[sel]
        print("%-14s n=%6d  |e| p50=%.5f  " % (stem, int(sel.sum()), float(np.median(e[sel])))
              + " ".join("%s=%+.2f" % (k, per_corr[k][-1]) for k in INDICATORS), flush=True)

    print("\n=== median corr(indicator, |FEM error|) over %d vessels ===" % len(X_by_v), flush=True)
    for k in INDICATORS:
        v = [c for c in per_corr[k] if c == c and np.isfinite(c)]
        print("  %-10s %+.3f" % (k, np.median(v) if v else NAN), flush=True)

    r2 = _ridge_lovo(X_by_v, y_by_v)
    print("\nleave-one-vessel-out ridge R^2 for |e| from all indicators: %.4f" % r2, flush=True)
    print("  (a head cannot beat the information in its inputs; R^2 near 0 means new channels "
          "will not help)", flush=True)

    if args.out:
        o = Path(args.out)
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text(json.dumps(dict(
            band_only=bool(args.band_only), r2_lovo=r2,
            median_corr={k: (float(np.median([c for c in per_corr[k] if c == c]))
                             if any(c == c for c in per_corr[k]) else None) for k in INDICATORS},
        ), indent=2), encoding="utf-8")
        print("[OK] wrote " + str(o), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
