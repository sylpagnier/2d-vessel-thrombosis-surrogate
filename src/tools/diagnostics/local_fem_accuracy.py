#!/usr/bin/env python
"""How accurate is the local scikit-fem Carreau solver against COMSOL's t=0 field?

Scores `src/core_physics/local_fem_solver.py` on the synthetic biochem anchor packs, in the
currencies that were measured to predict the clot outcome -- wall `dsrx` correlation and gate
union Jaccard first, velocity rel-L2 second, because rel-L2 has repeatedly ranked flow arms
the wrong way round.

Two arms per vessel, all against COMSOL `y[0, :, 0:2]`:

    fem     the local solve, inlet Dirichlet taken from the pack's own inlet BC
    deq     the RGP-DEQ field already cached on the pack as `u0_pred` (when present)

Gate columns are reported across (hops, gain) combinations.  The shipped path for FEM inherited
the pred treatment (hops=6 in features.py, gain=PRED_DSRX_GAIN=3.00) -- both factors were
fitted for the surrogate's under-resolution and are not obviously right for a converged field.
The table below shows what each combination tests:

    hops=3  gain=1.00  -- GT treatment (converged field; hypothesis: this is correct for FEM)
    hops=4  gain=1.00  -- temporal.py treatment, no amplitude correction
    hops=6  gain=1.00  -- features.py stencil, no amplitude correction
    hops=6  gain=2.18  -- features.py stencil + stencil attenuation only (DSRX_STENCIL_GAIN)
    hops=6  gain=3.00  -- today's shipped path (PRED_DSRX_GAIN = DSRX_STENCIL_GAIN * 1.38)
    deq h6  gain=3.00  -- surrogate at shipped settings (reference)

DSRX_STENCIL_GAIN=2.18 is the h3->h6 attenuation measured on the GT field alone (2026-08-23);
it is a property of the operator, not the flow model.  PRED_DSRX_GAIN=3.00 bundles that with
an additional 1.38x surrogate deficit.  A converged FEM field should not carry the deficit term.

    python scripts/diag_local_fem_accuracy.py --cohort --out outputs/diag_local_fem_accuracy.json
"""
from __future__ import annotations

from src.tools.diagnostics._common import bootstrap, biochem_packs_dir

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


from src.config import BiochemConfig  # noqa: E402
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402
from src.core_physics.wall_cohort_splits import CLOT_FREE, DEV, FIT, SEALED  # noqa: E402

PACKS = biochem_packs_dir()


def _cohort_stems() -> list[str]:
    skip = set(SEALED) | set(CLOT_FREE)
    out = [a for a in list(FIT) + list(DEV) if a not in skip and (PACKS / f"{a}.pt").exists()]
    for s in ("wound_patient001", "wound_patient002", "wound_patient003"):
        if (PACKS / f"{s}.pt").exists() and s not in out:
            out.append(s)
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    u = int((a | b).sum())
    return float((a & b).sum() / u) if u else float("nan")


def _wall_band(data, hops: int = 3) -> np.ndarray:
    n = int(data.num_nodes)
    row, col = data.edge_index
    band = data.mask_wall.reshape(-1).bool().clone()
    for _ in range(hops):
        acc = torch.zeros(n, dtype=torch.bool)
        acc.index_put_((row,), band[col], accumulate=False)
        band = band | acc
    return band.numpy()


def _field_metrics(u: np.ndarray, g: np.ndarray, wall: np.ndarray) -> dict:
    """Non-dimensional velocity error against COMSOL, globally and in the wall band."""
    def rel(mask):
        d = u[mask] - g[mask]
        den = np.linalg.norm(g[mask])
        return float(np.linalg.norm(d) / den) if den > 0 else float("nan")

    su, sg = np.linalg.norm(u, axis=1), np.linalg.norm(g, axis=1)
    ok = (su > 1e-9) & (sg > 1e-9)
    cos = float(np.mean((u[ok] * g[ok]).sum(1) / (su[ok] * sg[ok]))) if ok.any() else float("nan")
    return dict(
        rel_l2=rel(np.ones(len(g), dtype=bool)),
        rel_l2_wall=rel(wall),
        dir_cos=cos,
        speed_corr=_corr(su, sg),
        speed_ratio=float(np.median(su[ok] / sg[ok])) if ok.any() else float("nan"),
        zero_frac=float(np.mean(su <= 1e-12)),
    )


def _gate_metrics(data, bio, wall: np.ndarray, hops_pred: int, gain: float) -> dict:
    """Gate agreement of the pack's current `u0_pred` against COMSOL's own gate.

    GT is differentiated at the consumer's hops=3 -- the labels' own stencil.
    """
    import os

    gt = t0_flow_fields(data, bio, hops=3, flow_source="gt")
    old = os.environ.get("CLOT_PRED_DSRX_GAIN", "")
    os.environ["CLOT_PRED_DSRX_GAIN"] = str(float(gain))
    try:
        pr = t0_flow_fields(data, bio, hops=int(hops_pred), flow_source="pred")
    finally:
        if old:
            os.environ["CLOT_PRED_DSRX_GAIN"] = old
        else:
            os.environ.pop("CLOT_PRED_DSRX_GAIN", None)

    w = np.asarray(wall, dtype=bool)
    g_gt, g_pr = np.asarray(gt.gate)[w] > 0, np.asarray(pr.gate)[w] > 0
    den = np.median(np.abs(gt.dsrx[w]))
    return dict(
        sr_corr=_corr(pr.sr[w], gt.sr[w]),
        sr_ratio=float(np.median(np.abs(pr.sr[w])) / max(np.median(np.abs(gt.sr[w])), 1e-30)),
        dsrx_corr=_corr(pr.dsrx[w], gt.dsrx[w]),
        dsrx_ratio=float(np.median(np.abs(pr.dsrx[w])) / max(den, 1e-30)),
        gate_jaccard=_jaccard(g_pr, g_gt),
        gate_low_jaccard=_jaccard(np.asarray(pr.gate_low)[w] > 0, np.asarray(gt.gate_low)[w] > 0),
        gate_sep_jaccard=_jaccard(np.asarray(pr.gate_sep)[w] > 0, np.asarray(gt.gate_sep)[w] > 0),
        fire_ratio=float(g_pr.mean() / max(g_gt.mean(), 1e-12)),
        gt_fire=float(g_gt.mean()),
    )


def _score_one(stem: str, hops_list: list[int], gains: list[float], quiet: bool) -> dict:
    from src.clot_ml.v0 import solve_fem_into_pack

    data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
    if getattr(data, "graph_stem", None) is None:
        data.graph_stem = stem
    bio = BiochemConfig(phase="biochem")
    g = data.y[0, :, 0:2].numpy().astype(np.float64)
    wall = np.asarray(data.mask_wall.reshape(-1).bool().numpy())
    band = _wall_band(data, hops=3)
    row = dict(stem=stem, n_nodes=int(data.num_nodes), gt_speed_max=float(np.linalg.norm(g, axis=1).max()))

    deq = None
    if getattr(data, "u0_pred", None) is not None:
        deq = np.stack([data.u0_pred.reshape(-1).numpy(), data.v0_pred.reshape(-1).numpy()], 1).astype(np.float64)

    arms = {}
    if deq is not None:
        arms["deq"] = deq
    t0 = time.time()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf if quiet else sys.stdout):
            solve_fem_into_pack(data)
    except Exception as exc:  # a missing mesh or a singular solve must not kill the sweep
        row["fem_error"] = f"{type(exc).__name__}: {exc}"
        return row
    row["fem_secs"] = round(time.time() - t0, 1)
    row["fem_iters"] = buf.getvalue().count("Iter ")
    fem = np.stack([data.u0_pred.reshape(-1).numpy(), data.v0_pred.reshape(-1).numpy()], 1).astype(np.float64)
    arms["fem"] = fem

    for name, u in arms.items():
        for k, v in _field_metrics(u, g, band).items():
            row[f"{name}_{k}"] = v

    # Gate metrics read `u0_pred` off the pack, so each arm is written in turn.
    for name, u in arms.items():
        data.u0_pred = torch.tensor(u[:, 0], dtype=torch.float32)
        data.v0_pred = torch.tensor(u[:, 1], dtype=torch.float32)
        for h in hops_list:
            for gn in gains:
                tag = f"{name}_h{h}_g{gn:g}"
                for k, v in _gate_metrics(data, bio, wall, h, gn).items():
                    row[f"{tag}_{k}"] = v
    return row


def _med(rows: list[dict], key: str) -> float:
    vs = [r[key] for r in rows if key in r and r[key] == r[key] and np.isfinite(r[key])]
    return float(np.median(vs)) if vs else float("nan")


def main(argv: list[str] | None = None) -> int:
    bootstrap()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", nargs="*", default=None)
    ap.add_argument("--cohort", action="store_true", help="FIT+DEV clot-carrying packs plus the wounds")
    ap.add_argument("--hops", default="3,4,6")
    ap.add_argument("--gains", default="1,2.18,3")
    ap.add_argument("--quiet", action="store_true", default=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    stems = args.stems or (_cohort_stems() if args.cohort else ["patient001", "patient020", "patient041"])
    hops = [int(h) for h in args.hops.split(",") if h.strip()]
    gains = [float(g) for g in args.gains.split(",") if g.strip()]
    print(f"[i] n={len(stems)} hops={hops} gains={gains}", flush=True)

    rows = []
    for stem in stems:
        r = _score_one(stem, hops, gains, args.quiet)
        rows.append(r)
        if "fem_error" in r:
            print(f"{r['stem']:22s} ERROR {r['fem_error']}", flush=True)
        else:
            print(f"{r['stem']:22s} {r['fem_secs']:6.1f}s relL2 {r['fem_rel_l2']:.4f} "
                  f"(wall {r['fem_rel_l2_wall']:.4f}) cos {r['fem_dir_cos']:.4f} "
                  f"dsrx_corr(h3,g1) {r.get('fem_h3_g1_dsrx_corr', float('nan')):.3f} "
                  f"gateJ {r.get('fem_h3_g1_gate_jaccard', float('nan')):.3f}", flush=True)

    ok = [r for r in rows if "fem_error" not in r]
    print("\nMEDIANS (n=%d)" % len(ok))
    for arm in ("fem", "deq"):
        if not any(f"{arm}_rel_l2" in r for r in ok):
            continue
        print(f"  {arm:4s} relL2 {_med(ok, f'{arm}_rel_l2'):.4f}  wall {_med(ok, f'{arm}_rel_l2_wall'):.4f}  "
              f"cos {_med(ok, f'{arm}_dir_cos'):.4f}  speed_corr {_med(ok, f'{arm}_speed_corr'):.4f}")
        for h in hops:
            for gn in gains:
                t = f"{arm}_h{h}_g{gn:g}"
                print(f"       h{h} g{gn:g}: dsrx_corr {_med(ok, t+'_dsrx_corr'):+.3f}  "
                      f"dsrx_ratio {_med(ok, t+'_dsrx_ratio'):.2f}  sr_corr {_med(ok, t+'_sr_corr'):+.3f}  "
                      f"gateJ {_med(ok, t+'_gate_jaccard'):.3f}  sepJ {_med(ok, t+'_gate_sep_jaccard'):.3f}  "
                      f"lowJ {_med(ok, t+'_gate_low_jaccard'):.3f}  fire x{_med(ok, t+'_fire_ratio'):.2f}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"[save] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
