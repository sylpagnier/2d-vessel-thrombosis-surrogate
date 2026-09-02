"""Why a constant flux exponent on geometric h is wrong (pi wall-shear).

Regresses log(sr/sr0) on log(h) inside delta_mu terciles of the FEM pi corpus.
The Poiseuille slope -2 appears only where occlusion is stiff; soft gel barely redirects flux.
This measurement motivated hydraulic_h in src/core_physics/pi_wall_shear.py.

    python -m src.tools.diagnostics pi-flux-interaction
    python -m src.tools.diagnostics pi-flux-interaction --corpus outputs/pi_corpus
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.core_physics.wall_shear_attenuation import DELTA_MU_HALF_SI
from src.core_physics.pi_wall_shear import hydraulic_h
from src.tools.diagnostics._common import bootstrap


def _load_corpus(corpus_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    keys = ("sr0", "sr_fem", "delta_mu", "h_over_h0", "in_clot")
    for f in sorted(corpus_dir.glob("*.npz")):
        z = np.load(f)
        d = {k: z[k].astype(np.float64) for k in keys if k in z}
        keep = (d["sr0"] > 1e-2) & (d["sr_fem"] > 0.0) & (d["in_clot"] > 0.5)
        d = {k: v[keep] for k, v in d.items()}
        if len(d["sr0"]) >= 50:
            out[f.stem] = d
    return out


def _tercile_slopes(data: dict[str, dict[str, np.ndarray]], *, hydraulic: bool) -> list[float]:
    dmu = np.concatenate([d["delta_mu"] for d in data.values()])
    lo, hi = np.quantile(dmu, [1.0 / 3.0, 2.0 / 3.0])
    slopes: list[float] = []
    for label, mask_fn in (
        ("low", lambda x: x <= lo),
        ("mid", lambda x: (x > lo) & (x <= hi)),
        ("high", lambda x: x > hi),
    ):
        y_parts, b_parts = [], []
        for d in data.values():
            m = mask_fn(d["delta_mu"])
            if not np.any(m):
                continue
            h = (
                hydraulic_h(d["delta_mu"][m], d["h_over_h0"][m], delta_mu_half=DELTA_MU_HALF_SI)
                if hydraulic
                else np.clip(d["h_over_h0"][m], 1e-3, 1.0)
            )
            y_parts.append(np.log(d["sr_fem"][m] / d["sr0"][m]) + np.log1p(d["delta_mu"][m] / DELTA_MU_HALF_SI))
            b_parts.append(np.log(np.clip(h, 1e-9, 1.0)))
        if not y_parts:
            slopes.append(float("nan"))
            continue
        y = np.concatenate(y_parts)
        b = np.concatenate(b_parts)
        denom = float(b @ b)
        slopes.append(float(-(y @ b) / denom) if denom > 1e-12 else float("nan"))
    return slopes


def main(argv: list[str] | None = None) -> int:
    bootstrap()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("outputs/pi_corpus"))
    args = ap.parse_args(argv)
    if not args.corpus.is_dir():
        print(f"[ERR] corpus not found: {args.corpus}", flush=True)
        return 2
    data = _load_corpus(args.corpus)
    if not data:
        print(f"[ERR] no usable rows under {args.corpus}", flush=True)
        return 2
    geom = _tercile_slopes(data, hydraulic=False)
    hyd = _tercile_slopes(data, hydraulic=True)
    print("[i] Flux exponent p from log(sr/sr0) ~ -p * log(h) inside dmu terciles", flush=True)
    print("      low dmu    mid dmu    high dmu", flush=True)
    print(
        "geom  %9.3f  %9.3f  %9.3f   (constant exponent on geometric h)"
        % tuple(geom),
        flush=True,
    )
    print(
        "hyd   %9.3f  %9.3f  %9.3f   (hydraulic_h effective lumen)"
        % tuple(hyd),
        flush=True,
    )
    print("[i] Poiseuille reference p=2.0; see docs/LOCAL_KINEMATIC_CORRECTOR.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
