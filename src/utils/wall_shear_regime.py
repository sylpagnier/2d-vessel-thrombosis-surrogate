"""The wall-shear statistics the clot gate is actually decided by — one implementation.

**Why this module exists.**  `preflight_kine_cohort.py` checked `h_nd`, stenosis ratio, `u_ref`,
node count and solve rate — every one of them a property of the MESH or the SAMPLER.  It passed
a cohort (0 FAIL) whose labels do not contain the thing the consumer reads: the deposition gate
is `(sr < lss) | (dsrx < sgt)`, and measured over 60 synthetic against 47 deploy packs
(RGP_DEQ_REPAIR_PLAN.md §16.5), wall nodes:

```
                    synth p10/med/p90        deploy p10/med/p90
wall sr median      23.5 / 52.1 / 116.7      64.7 /  99.4 / 203.0      1.9x
wall dsrx sd        21.5 / 55.5 / 2242      222.5 / 592.4 / 2075      10.7x
`dsrx < sgt` fires  0 / 0.000 / 0.078         0 / 0.056 / 0.296
sep-branch share    0 / 0.000 / 0.330         0 / 0.508 / 1.000
```

At deployment the median vessel has **50.8% of its firing wall nodes firing through the `dsrx`
branch alone; in the corpus that is 0.0%**.  A producer-side check cannot see that.  So the
acceptance criterion for a cohort belongs in CONSUMER units, and this is where it is computed.

Everything here is measured through the consumer's own convention -- MLS `hops=3` on the ground
truth, exactly what `clot_ml.features.build_features` does for `flow="gt"` -- so the numbers are
directly comparable to `wall_shear_selection_metrics` and to the deploy reference band.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

#: Statistics a cohort is accepted on, and how to read each one.
REGIME_KEYS = ("wall_sr_med", "wall_dsrx_sd", "wall_fire", "wall_sep", "wall_sep_only")

#: Where the deploy reference band lives.  Derived from FIT only -- see
#: `scripts/derive_deploy_wall_shear_band.py`.
REFERENCE_PATH = "data/reference/deploy_wall_shear_band.json"


def wall_shear_regime(data, *, hops: int = 3) -> dict[str, float] | None:
    """Wall-node gate statistics for one graph, from its own labels.  ``None`` if unusable.

    ``wall_sep_only`` is the share of *firing* wall nodes that fire through the `dsrx` branch
    alone -- the number that separates the corpus from deployment most sharply, and the one a
    model trained on this cohort can learn nothing about when it is zero.
    """
    from src.clot_ml.features import M_TO_CM
    from src.config import BiochemConfig
    from src.core_physics.mls_gradient import build_mls_gradient, node_positions, shear_rate_2d

    y = getattr(data, "y", None)
    if y is None or not hasattr(data, "mask_wall"):
        return None
    yv = y[0] if y.dim() == 3 else y
    if yv.shape[1] < 2 or float(yv[:, 0:2].abs().max()) == 0.0:
        return None
    wall = data.mask_wall.reshape(-1).bool().detach().cpu().numpy()
    if wall.sum() < 5:
        return None

    bio = BiochemConfig(phase="biochem")
    lss, sgt = float(bio.lss), float(bio.sgt) / M_TO_CM
    u_ref = float(data.u_ref.reshape(-1)[0])
    d_bar = float(data.d_bar.reshape(-1)[0])

    Dx, Dy = build_mls_gradient(
        node_positions(data), data.edge_index.detach().cpu().numpy(), hops=hops
    )
    u = yv[:, 0].double().detach().cpu().numpy()
    v = yv[:, 1].double().detach().cpu().numpy()
    sr = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * (u_ref / d_bar)
    dsrx = (Dx @ sr) / (d_bar * M_TO_CM)

    s, dx = sr[wall], dsrx[wall]
    lo, sep = s < lss, dx < sgt
    fire = lo | sep
    return {
        "wall_sr_med": float(np.median(s)),
        "wall_dsrx_sd": float(np.std(dx)),
        "wall_fire": float(fire.mean()),
        "wall_sep": float(sep.mean()),
        "wall_sep_only": float((sep & ~lo).sum() / max(fire.sum(), 1)),
        "n_wall": int(wall.sum()),
    }


def summarise(rows: list[dict]) -> dict[str, dict[str, float]]:
    """``{key: {p10, p50, p90}}`` over per-vessel regimes."""
    out: dict[str, dict[str, float]] = {}
    for k in REGIME_KEYS:
        v = np.array([r[k] for r in rows if k in r and np.isfinite(r[k])], dtype=float)
        if v.size:
            out[k] = {q: float(np.percentile(v, p))
                      for q, p in (("p10", 10), ("p50", 50), ("p90", 90))}
    return out


def load_reference(root: Path | None = None) -> dict | None:
    """The stored deploy band, or ``None`` when it has not been derived on this machine."""
    import json

    from src.utils.paths import get_project_root

    p = (root or get_project_root()) / REFERENCE_PATH
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def compare_to_reference(cohort: dict, reference: dict) -> list[tuple[str, str, str]]:
    """``[(key, status, detail)]`` -- how a cohort's regime sits against deployment's.

    Judged on the MEDIAN against deployment's p10-p90, because the failure this exists to catch
    is a whole cohort sitting in the wrong regime, not a few odd vessels.  ``wall_sep_only`` also
    fails on an outright zero median regardless of the band: a cohort whose `dsrx` branch never
    fires teaches nothing about the branch that decides half of deployment.
    """
    ref = reference.get("summary", reference)
    out = []
    for k in REGIME_KEYS:
        if k not in cohort or k not in ref:
            continue
        med, lo, hi = cohort[k]["p50"], ref[k]["p10"], ref[k]["p90"]
        ratio = med / ref[k]["p50"] if ref[k]["p50"] else float("inf")
        detail = (f"median {med:.4g} vs deploy {ref[k]['p50']:.4g} "
                  f"({ratio:.2f}x); deploy p10-p90 [{lo:.4g}, {hi:.4g}]")
        if k == "wall_sep_only" and med <= 0.0 and ref[k]["p50"] > 0.0:
            out.append((k, "FAIL", detail + " -- the `dsrx` gate branch NEVER fires here"))
        elif lo <= med <= hi:
            out.append((k, "PASS", detail))
        elif 0.5 <= ratio <= 2.0:
            out.append((k, "WARN", detail))
        else:
            out.append((k, "FAIL", detail))
    return out


__all__ = ["REFERENCE_PATH", "REGIME_KEYS", "compare_to_reference", "load_reference",
           "summarise", "wall_shear_regime"]
