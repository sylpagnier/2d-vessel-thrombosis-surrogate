"""Stage-A production launcher (foundation -> polish -> comsol -> promote).

Usage:
  python scripts/run_kinematics_production.py
  python scripts/run_kinematics_production.py --foundation-only --fresh
  python scripts/run_kinematics_production.py ladder --skip-foundation
  python scripts/run_kinematics_production.py finetune --continuity-focus
  python scripts/run_kinematics_production.py comsol --holdout comsol007
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.training.kinematics_production_config import (  # noqa: E402
    ComsolFinetuneConfig,
    FoundationConfig,
    LadderConfig,
    ProductionRunConfig,
    SyntheticPolishConfig,
)
from src.training.kinematics_production_runner import (  # noqa: E402
    run_comsol_finetune,
    run_foundation,
    run_ladder,
    run_production,
    run_synthetic_polish,
)

_SUBCOMMANDS = frozenset({"ladder", "foundation", "finetune", "comsol"})


def _add_foundation_flags(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--adam-epochs", type=int, default=85)
    ap.add_argument("--stage1-end", type=int, default=40)
    ap.add_argument("--stage2-end", type=int, default=60)
    ap.add_argument("--graph-cap", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quiet", action="store_true")


def _foundation_from_ns(ns: argparse.Namespace) -> FoundationConfig:
    return FoundationConfig(
        fresh=bool(getattr(ns, "fresh", False)),
        epochs=int(ns.epochs),
        adam_epochs=int(ns.adam_epochs),
        stage1_end=int(ns.stage1_end),
        stage2_end=int(ns.stage2_end),
        graph_cap=int(ns.graph_cap),
        seed=int(ns.seed),
        quiet=bool(ns.quiet),
    )


def _ladder_from_ns(ns: argparse.Namespace) -> LadderConfig:
    return LadderConfig(
        foundation=_foundation_from_ns(ns),
        polish=SyntheticPolishConfig(
            finetune_epochs=int(getattr(ns, "synthetic_finetune_epochs", 40)),
            continuity_focus=not bool(getattr(ns, "no_continuity_focus", False)),
            quiet=bool(ns.quiet),
        ),
        comsol=ComsolFinetuneConfig(
            holdout=str(ns.holdout),
            finetune_epochs=int(getattr(ns, "comsol_finetune_epochs", 25)),
        ),
        skip_foundation=bool(getattr(ns, "skip_foundation", False)),
        skip_synthetic_polish=bool(getattr(ns, "skip_synthetic_polish", False)),
        skip_comsol_anchors=bool(getattr(ns, "skip_comsol_anchors", False)),
        skip_promote=bool(getattr(ns, "skip_promote", False)),
        require_comsol=bool(getattr(ns, "require_comsol", False)),
    )


def _build_production_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--foundation-only", action="store_true")
    ap.add_argument("--skip-synthetic-polish", action="store_true")
    ap.add_argument("--skip-comsol-anchors", action="store_true")
    ap.add_argument("--skip-promote", action="store_true")
    ap.add_argument("--require-comsol", action="store_true")
    ap.add_argument("--holdout", type=str, default="comsol007")
    ap.add_argument("--no-continuity-focus", action="store_true")
    ap.add_argument("--synthetic-finetune-epochs", type=int, default=40)
    ap.add_argument("--comsol-finetune-epochs", type=int, default=25)
    ap.add_argument("--skip-foundation", action="store_true", help="Ladder-only: skip phase 1")
    _add_foundation_flags(ap)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv_in = list(argv if argv is not None else sys.argv[1:])
    if argv_in and argv_in[0] in _SUBCOMMANDS:
        cmd = argv_in[0]
        rest = argv_in[1:]
        if cmd == "foundation":
            ap = argparse.ArgumentParser(description="Stage-A foundation only")
            _add_foundation_flags(ap)
            return run_foundation(_foundation_from_ns(ap.parse_args(rest)))
        if cmd == "finetune":
            ap = argparse.ArgumentParser(description="Synthetic polish finetune")
            ap.add_argument("--resume", type=str, default="best")
            ap.add_argument("--finetune-epochs", type=int, default=40)
            ap.add_argument("--continuity-focus", action="store_true")
            ap.add_argument("--try-lbfgs", action="store_true")
            ap.add_argument("--quiet", action="store_true")
            ns = ap.parse_args(rest)
            return run_synthetic_polish(
                SyntheticPolishConfig(
                    resume=str(ns.resume),
                    finetune_epochs=int(ns.finetune_epochs),
                    continuity_focus=bool(ns.continuity_focus),
                    try_lbfgs=bool(ns.try_lbfgs),
                    quiet=bool(ns.quiet),
                )
            )
        if cmd == "comsol":
            ap = argparse.ArgumentParser(description="COMSOL anchor finetune")
            ap.add_argument(
                "--resume",
                type=str,
                default="outputs/kinematics/production_allfix/kinematics_best.pth",
            )
            ap.add_argument("--holdout", type=str, default="comsol007")
            ap.add_argument("--finetune-epochs", type=int, default=25)
            ap.add_argument("--synthetic-cap", type=int, default=120)
            ns = ap.parse_args(rest)
            return run_comsol_finetune(
                ComsolFinetuneConfig(
                    resume=Path(ns.resume),
                    holdout=str(ns.holdout),
                    finetune_epochs=int(ns.finetune_epochs),
                    synthetic_cap=int(ns.synthetic_cap),
                )
            )
        ap = _build_production_parser()
        ns = ap.parse_args(rest)
        return run_ladder(_ladder_from_ns(ns))

    ap = _build_production_parser()
    ns = ap.parse_args(argv_in)
    if ns.skip_foundation:
        return run_ladder(_ladder_from_ns(ns))
    cfg = ProductionRunConfig(ladder=_ladder_from_ns(ns), foundation_only=bool(ns.foundation_only))
    return run_production(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
