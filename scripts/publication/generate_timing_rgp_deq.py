"""Controlled wall-clock timing of the RGP-DEQ t=0 solve, for the same table Table 3 reports FEM in.

WHY THIS EXISTS.  `generate_timing_data.py` times the shipped deploy path's local-FEM t=0 stage
(median 4.66-6.02 s depending on cohort/subset -- see `timing.json`).  Nobody had ever timed the
RGP-DEQ forward solve itself on the same cohort, under the same wall-clock protocol, so "is the
surrogate at least fast" was unmeasured.  It reuses the shipped kinematics checkpoint --
`resolve_kinematics_checkpoint()`'s default, i.e. exactly what a customer deploy loads -- not a
cross-fit evaluation arm (those are "no checkpoint promoted; evaluation device", never deployed).

WHAT IS TIMED.  Per vessel: pack load, then one `predict_kinematics_and_latent` Anderson solve.
No disk or in-memory cache is reused across vessels; the per-vessel graph key differs anyway, but
the model's cache is on a fresh model instance per run to avoid a lucky first-call skew.

Usage:
    python scripts/publication/generate_timing_rgp_deq.py
    python scripts/publication/generate_timing_rgp_deq.py --stems comsol020 comsol005
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from scripts.publication.config import DATA_DIR  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.wall_cohort_splits import CLOT_FREE, DEV, FIT, SEALED  # noqa: E402
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    predict_kinematics_and_latent,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import anchor_packs_dir  # noqa: E402

PACKS = anchor_packs_dir()


def _cohort_stems() -> list[str]:
    skip = set(SEALED) | set(CLOT_FREE)
    return [a for a in list(FIT) + list(DEV)
            if a not in skip and (PACKS / f"{a}.pt").exists()]


def _env() -> dict:
    env = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": "cpu",
    }
    if torch.cuda.is_available():
        env["device"] = torch.cuda.get_device_name(0)
        env["cuda"] = torch.version.cuda
    return env


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_one(model, stem: str) -> dict:
    t0 = time.perf_counter()
    data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
    if getattr(data, "graph_stem", None) is None:
        data.graph_stem = stem
    _sync()
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    predict_kinematics_and_latent(model, data)
    _sync()
    t_solve = time.perf_counter() - t0

    return {"stem": stem, "n_nodes": int(data.num_nodes), "load_s": t_load, "deq_s": t_solve}


def _stats(xs: list[float]) -> dict:
    xs = sorted(float(x) for x in xs)
    if not xs:
        return {}
    out = {"n": len(xs), "median": statistics.median(xs), "mean": statistics.fmean(xs),
           "min": xs[0], "max": xs[-1]}
    if len(xs) >= 4:
        out["q1"] = float(np.percentile(xs, 25))
        out["q3"] = float(np.percentile(xs, 75))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", nargs="*", default=None,
                    help="vessels to time (default: FIT+DEV, SEALED and clot-free excluded)")
    ap.add_argument("--checkpoint", default=None,
                    help="kinematics checkpoint; defaults to the resolved shipped one")
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args()

    stems = args.stems or _cohort_stems()
    if not stems:
        print("no packs found under", PACKS)
        return 1

    device = require_cuda_device() if torch.cuda.is_available() else torch.device("cpu")
    ckpt = resolve_kinematics_checkpoint(args.checkpoint)
    model = load_kinematics_predictor(ckpt, device, cache=False)
    env = _env()
    print(f"[env] {env['device']}  torch {env['torch']}  ckpt={ckpt}")
    print(f"[cohort] {len(stems)} vessels\n")

    # Warm up on a vessel OUTSIDE the reported cohort: the model joint-caches (pred, z_kin) per
    # graph key, so warming up on stems[0] would make its real timed call a free cache hit
    # (observed: 0.00s) instead of a solve.
    warmup_candidates = [s for s in (list(SEALED) + list(CLOT_FREE))
                         if (PACKS / f"{s}.pt").exists() and s not in stems]
    warmup_stem = warmup_candidates[0] if warmup_candidates else None
    if warmup_stem:
        print(f"[warmup] {warmup_stem} (discarded) ...", flush=True)
        try:
            _time_one(model, warmup_stem)
        except Exception as exc:
            print(f"[warmup] failed: {exc}")
    else:
        print("[warmup] no out-of-cohort vessel available; skipping (first row may be skewed)")

    rows, failed = [], []
    for stem in stems:
        try:
            row = _time_one(model, stem)
            rows.append(row)
            print(f"  {stem:<20} nodes={row['n_nodes']:>6}  deq={row['deq_s']:6.2f}s", flush=True)
        except Exception as exc:
            failed.append({"stem": stem, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  {stem:<20} FAILED  {type(exc).__name__}: {exc}", flush=True)

    if not rows:
        print("\nno vessel timed successfully")
        return 1

    summary = {k: _stats([r[k] for r in rows]) for k in ("deq_s", "load_s")}
    payload = {
        "env": env,
        "checkpoint": str(ckpt),
        "boundary": "pack (graph, priors already baked) -> one RGP-DEQ Anderson solve; "
                    "comparable to timing.json's fem_s stage, not to its end-to-end deploy_s",
        "per_vessel": rows,
        "failed": failed,
        "summary": summary,
    }
    out = Path(args.out) if args.out else (DATA_DIR / "timing_rgp_deq.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(out, "w"), indent=2)

    csv = out.with_suffix(".csv")
    with open(csv, "w", encoding="utf-8") as fh:
        fh.write("stem,n_nodes,load_s,deq_s\n")
        for r in rows:
            fh.write(f"{r['stem']},{r['n_nodes']},{r['load_s']:.4f},{r['deq_s']:.4f}\n")

    print(f"\n=== RGP-DEQ t=0 solve wall-clock, n={len(rows)} vessels ===")
    s = summary["deq_s"]
    iqr = f"  IQR [{s['q1']:.2f}, {s['q3']:.2f}]" if "q1" in s else ""
    print(f"  deq_s        median {s['median']:8.2f} s   min {s['min']:7.2f}   "
          f"max {s['max']:7.2f}{iqr}")
    if failed:
        print(f"\n  {len(failed)} vessel(s) failed -- see {out.name}")
    print(f"\nwrote {out}\n     {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
