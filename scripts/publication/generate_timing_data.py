"""Controlled wall-clock timing of the shipped deploy path, for the paper's speedup claim.

WHY THIS EXISTS.  The repo carried two order-of-magnitude-different numbers for "a rollout":
~25-30 min/anchor (`docs/WALL_MODEL_PLAN.md` s0 -- the RETIRED `WC_v7` + compound stack, on a
4 GB GPU, and including graded scoring) and ~1.5 min (the UI, shipped `clot_ml_0`, inference
only).  The speedup against COMSOL's ~48 h/vessel is a headline claim, so it gets measured
rather than recalled.

WHAT IS TIMED.  The deploy path as `research_sweep_runner` / `CustomerDeployPipeline` run it:

    pack load -> FEM t=0 solve -> build_sample -> clot_ml_0 rollout

STATE THE BOUNDARY WHEN QUOTING.  Geometry construction and meshing happen upstream of a pack
and are NOT included here; the run always prints that reminder.  The COMSOL 48 h figure
covers geometry -> mesh -> solve, so a like-for-like end-to-end comparison must either add the
meshing cost to this side or say plainly that it is excluded.

Rollout cost scales with the number of scored timesteps, so `per_step_s` is reported alongside
the total: any other time grid can be derived from it without re-running.

Usage:
    python scripts/publication/generate_timing_data.py                  # default cohort
    python scripts/publication/generate_timing_data.py --stems comsol020 comsol005
    python scripts/publication/generate_timing_data.py --every 4 --repeats 3
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.publication.config import DATA_DIR  # noqa: E402
from src.clot_ml.locked import build_sample  # noqa: E402
from src.clot_ml.v0 import load_v0_bundle, predict_clot_ml_0, solve_fem_into_pack  # noqa: E402
from src.config import BiochemConfig  # noqa: E402
from src.core_physics.wall_cohort_splits import CLOT_FREE, DEV, FIT, SEALED  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"


def _times(data, every: int) -> list[int]:
    T = int(data.y.shape[0])
    grid = list(range(0, T, max(int(every), 1)))
    if grid[-1] != T - 1:
        grid.append(T - 1)
    return grid


def _cohort_stems() -> list[str]:
    """Scored cohort: FIT + DEV, minus SEALED and clot-free.  SEALED stays closed."""
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
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            env["device_memory_gb"] = round(total / (1024 ** 3), 2)
        except Exception:
            pass
    return env


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_one(bundle, stem: str, every: int, flow: str) -> dict:
    """One end-to-end deploy run, stage by stage.  Wall-clock, CUDA-synced."""
    bio = BiochemConfig(phase="biochem")

    t0 = time.perf_counter()
    data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
    if getattr(data, "graph_stem", None) is None:
        data.graph_stem = stem
    _sync()
    t_load = time.perf_counter() - t0

    times = _times(data, every)

    t_fem = 0.0
    if flow == "fem":
        t0 = time.perf_counter()
        solve_fem_into_pack(data)
        _sync()
        t_fem = time.perf_counter() - t0

    t0 = time.perf_counter()
    S = build_sample(data, bio, flow=flow, variant="v4")
    _sync()
    t_sample = time.perf_counter() - t0

    t0 = time.perf_counter()
    predict_clot_ml_0(bundle, data, times, flow=flow, sample=S)
    _sync()
    t_roll = time.perf_counter() - t0

    deploy = t_fem + t_sample + t_roll
    return {
        "stem": stem,
        "n_nodes": int(data.num_nodes),
        "T": int(data.y.shape[0]),
        "n_times": len(times),
        "load_s": t_load,
        "fem_s": t_fem,
        "sample_s": t_sample,
        "rollout_s": t_roll,
        # The quotable unit: everything from a graph to a clot timeline.  Pack load is I/O of a
        # cached artifact and is reported separately rather than folded in.
        "deploy_s": deploy,
        "per_step_s": t_roll / max(len(times), 1),
    }


def _stats(xs: list[float]) -> dict:
    xs = sorted(float(x) for x in xs)
    if not xs:
        return {}
    out = {
        "n": len(xs),
        "median": statistics.median(xs),
        "mean": statistics.fmean(xs),
        "min": xs[0],
        "max": xs[-1],
    }
    if len(xs) >= 4:
        out["q1"] = float(np.percentile(xs, 25))
        out["q3"] = float(np.percentile(xs, 75))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", nargs="*", default=None,
                    help="vessels to time (default: FIT+DEV, SEALED and clot-free excluded)")
    ap.add_argument("--every", type=int, default=1,
                    help="timestep stride; 1 = full horizon (default)")
    ap.add_argument("--flow", default="fem", choices=("fem", "gt", "pred"),
                    help="'fem' is the shipped deploy path (default)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="timed repeats per vessel; the median is reported")
    ap.add_argument("--comsol-hours", type=float, default=48.0,
                    help="COMSOL reference solve time per vessel, for the speedup line")
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args()

    stems = args.stems or _cohort_stems()
    if not stems:
        print("no packs found under", PACKS)
        return 1

    bundle = load_v0_bundle()
    env = _env()
    print(f"[env] {env['device']}  torch {env['torch']}  flow={args.flow}  every={args.every}")
    print(f"[cohort] {len(stems)} vessels\n")

    # Warm-up: first call pays CUDA context, lazy imports and kernel autotune.  Discarded.
    print(f"[warmup] {stems[0]} (discarded) ...", flush=True)
    try:
        _time_one(bundle, stems[0], args.every, args.flow)
    except Exception as exc:  # a warm-up failure is not fatal; the real run will report it
        print(f"[warmup] failed: {exc}")

    rows, failed = [], []
    for stem in stems:
        try:
            reps = [_time_one(bundle, stem, args.every, args.flow)
                    for _ in range(max(args.repeats, 1))]
            row = min(reps, key=lambda r: r["deploy_s"]) if args.repeats > 1 else reps[0]
            if args.repeats > 1:
                row = dict(row)
                row["deploy_s"] = statistics.median(r["deploy_s"] for r in reps)
                row["repeats"] = args.repeats
            rows.append(row)
            print(f"  {stem:<20} nodes={row['n_nodes']:>6}  steps={row['n_times']:>4}  "
                  f"fem={row['fem_s']:6.1f}s  sample={row['sample_s']:6.1f}s  "
                  f"roll={row['rollout_s']:7.1f}s  TOTAL={row['deploy_s']:7.1f}s", flush=True)
        except Exception as exc:
            failed.append({"stem": stem, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  {stem:<20} FAILED  {type(exc).__name__}: {exc}", flush=True)

    if not rows:
        print("\nno vessel timed successfully")
        json.dump({"env": env, "failed": failed}, open(DATA_DIR / "timing.json", "w"), indent=2)
        return 1

    summary = {k: _stats([r[k] for r in rows])
               for k in ("deploy_s", "fem_s", "sample_s", "rollout_s", "per_step_s")}
    med = summary["deploy_s"]["median"]
    comsol_s = args.comsol_hours * 3600.0

    payload = {
        "env": env,
        "flow": args.flow,
        "every": args.every,
        "repeats": args.repeats,
        "boundary": "pack (graph) -> FEM t=0 -> build_sample -> clot_ml_0 rollout; "
                    "geometry construction and meshing are UPSTREAM and excluded",
        "comsol_reference_hours": args.comsol_hours,
        "speedup_vs_comsol_median": comsol_s / med if med > 0 else None,
        "per_vessel": rows,
        "failed": failed,
        "summary": summary,
    }
    out = Path(args.out) if args.out else (DATA_DIR / "timing.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(out, "w"), indent=2)

    csv = out.with_suffix(".csv")
    cols = ["stem", "n_nodes", "T", "n_times", "fem_s", "sample_s", "rollout_s",
            "deploy_s", "per_step_s"]
    with open(csv, "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                              for c in cols) + "\n")

    print(f"\n=== deploy wall-clock, n={len(rows)} vessels ===")
    for k in ("fem_s", "sample_s", "rollout_s", "deploy_s"):
        s = summary[k]
        iqr = f"  IQR [{s['q1']:.1f}, {s['q3']:.1f}]" if "q1" in s else ""
        print(f"  {k:<12} median {s['median']:8.2f} s   min {s['min']:7.2f}   "
              f"max {s['max']:7.2f}{iqr}")
    print(f"\n  median end-to-end : {med:.1f} s  ({med / 60.0:.2f} min)")
    print(f"  COMSOL reference  : {args.comsol_hours:.0f} h")
    print(f"  SPEEDUP           : {comsol_s / med:,.0f}x")
    if failed:
        print(f"\n  {len(failed)} vessel(s) failed -- see {out.name}")
    print(f"\nwrote {out}\n     {csv}")
    print("\nNOTE when quoting: meshing/geometry construction is upstream of a pack and is NOT "
          "in these numbers; COMSOL's 48 h covers geometry -> mesh -> solve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
