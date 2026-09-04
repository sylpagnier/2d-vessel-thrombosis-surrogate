"""Run geometry-sensitivity research sweeps (default: clot_ml_0 + FEM t=0).

Usage:
  python scripts/run_research_sweep.py --sweep 01_stenosis_strength
  python scripts/run_research_sweep.py --all
  python scripts/run_research_sweep.py --list
"""

from __future__ import annotations

import argparse
from pathlib import Path


from src.evaluation.research_sweep_config import (  # noqa: E402
    DEFAULT_RESEARCH_MODEL,
    SWEEPS_DIR,
    list_sweep_configs,
    load_sweep_config,
    resolve_sweep_path,
)
from src.evaluation.research_sweep_runner import run_sweep  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Geometry-sensitivity research sweeps (clot_ml_0 + FEM default)"
    )
    ap.add_argument(
        "--sweep",
        type=str,
        default="",
        help="Sweep id or path under configs/research_sweeps/",
    )
    ap.add_argument("--all", action="store_true", help="Run all configs in configs/research_sweeps/")
    ap.add_argument("--list", action="store_true", help="List available sweep configs and exit")
    ap.add_argument("--arm", type=str, default="", help="Optional single arm name filter")
    ap.add_argument("--force-rebuild-mesh", action="store_true", help="Ignore mesh cache")
    ap.add_argument("--cpu", action="store_true", help="Allow CPU (slow; CUDA recommended)")
    args = ap.parse_args(argv)


    if args.list:
        cfgs = list_sweep_configs()
        if not cfgs:
            print(f"[WARN] No configs under {SWEEPS_DIR}", flush=True)
            return 1
        for p in cfgs:
            try:
                c = load_sweep_config(p)
                print(
                    f"  {c.get('id', p.stem):28s}  {c.get('title', '')}",
                    flush=True,
                )
            except Exception as exc:
                print(f"  {p.name}: [ERR] {exc}", flush=True)
        return 0

    paths: list[Path] = []
    if args.all:
        paths = list_sweep_configs()
        if not paths:
            print(f"[ERR] No configs under {SWEEPS_DIR}", flush=True)
            return 1
    elif args.sweep.strip():
        paths = [resolve_sweep_path(args.sweep)]
    else:
        ap.print_help()
        print("\n[ERR] Pass --sweep <id> or --all (or --list)", flush=True)
        return 2

    failures = 0
    for path in paths:
        try:
            cfg = load_sweep_config(path)
            pipeline = None
            if str(cfg.get("model")) == DEFAULT_RESEARCH_MODEL:
                print(f"[i] Using {DEFAULT_RESEARCH_MODEL} + flow={cfg['control'].get('flow')}", flush=True)
            run_sweep(
                cfg,
                pipeline=pipeline,
                force_rebuild=bool(args.force_rebuild_mesh),
                arm_filter=args.arm.strip() or None,
                progress=lambda msg: print(msg, flush=True),
            )
        except Exception as exc:
            import traceback

            traceback.print_exc()
            failures += 1
            print(f"[ERR] Sweep {path.name} failed: {exc}", flush=True)
            if not args.all:
                raise
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
