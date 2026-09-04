"""Run geometry-sensitivity research sweeps (default: clot_ml_0 + FEM t=0).

Usage:
  python scripts/run_research_sweep.py --sweep 01_stenosis_strength
  python scripts/run_research_sweep.py --all
  python scripts/run_research_sweep.py --list
  python scripts/run_research_sweep.py --sweep legacy/15_stack_coupling --legacy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.evaluation.research_sweep_config import (  # noqa: E402
    DEFAULT_RESEARCH_MODEL,
    LEGACY_BIOCHEM_MODEL,
    SWEEPS_DIR,
    LEGACY_SWEEPS_DIR,
    list_sweep_configs,
    load_sweep_config,
    resolve_sweep_path,
)
from src.evaluation.research_sweep_runner import run_sweep  # noqa: E402
from src.inference.customer_pipeline import (  # noqa: E402
    DEFAULT_MAT_LEG,
    DEFAULT_WALL_CKPT,
    CustomerDeployPipeline,
)


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
    ap.add_argument("--legacy", action="store_true", help="Include configs/research_sweeps/legacy/")
    ap.add_argument("--arm", type=str, default="", help="Optional single arm name filter")
    ap.add_argument("--force-rebuild-mesh", action="store_true", help="Ignore mesh cache")
    ap.add_argument("--cpu", action="store_true", help="Allow CPU (slow; CUDA recommended)")
    ap.add_argument(
        "--wall-ckpt",
        type=str,
        default="",
        help="Legacy biochem only: override wall ckpt",
    )
    ap.add_argument(
        "--mat-leg",
        type=str,
        default="",
        help=f"Legacy biochem only: mat-growth leg (default: {DEFAULT_MAT_LEG})",
    )
    args = ap.parse_args(argv)

    include_legacy = bool(args.legacy)

    if args.list:
        cfgs = list_sweep_configs(include_legacy=include_legacy)
        if not cfgs:
            print(f"[WARN] No configs under {SWEEPS_DIR}", flush=True)
            return 1
        for p in cfgs:
            try:
                c = load_sweep_config(p)
                tag = "legacy" if LEGACY_SWEEPS_DIR.name in p.parts else "active"
                print(
                    f"  {c.get('id', p.stem):28s}  [{tag}]  {c.get('title', '')}",
                    flush=True,
                )
            except Exception as exc:
                print(f"  {p.name}: [ERR] {exc}", flush=True)
        return 0

    paths: list[Path] = []
    if args.all:
        paths = list_sweep_configs(include_legacy=include_legacy)
        if not paths:
            print(f"[ERR] No configs under {SWEEPS_DIR}", flush=True)
            return 1
    elif args.sweep.strip():
        paths = [resolve_sweep_path(args.sweep, include_legacy=include_legacy)]
    else:
        ap.print_help()
        print("\n[ERR] Pass --sweep <id> or --all (or --list)", flush=True)
        return 2

    legacy_pipeline: CustomerDeployPipeline | None = None
    needs_legacy = False
    for path in paths:
        cfg = load_sweep_config(path)
        if str(cfg.get("model")) == LEGACY_BIOCHEM_MODEL:
            needs_legacy = True
            break

    if needs_legacy:
        wall = Path(args.wall_ckpt) if args.wall_ckpt.strip() else None
        mat_leg = args.mat_leg.strip() or DEFAULT_MAT_LEG
        if wall is None:
            print(f"[i] Legacy biochem ckpt: {DEFAULT_WALL_CKPT}", flush=True)
        legacy_pipeline = CustomerDeployPipeline(
            model_name="legacy_species",
            wall_ckpt=wall,
            mat_leg=mat_leg,
            require_cuda=not bool(args.cpu),
        )

    failures = 0
    for path in paths:
        try:
            cfg = load_sweep_config(path)
            pipeline = legacy_pipeline if str(cfg.get("model")) == LEGACY_BIOCHEM_MODEL else None
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
