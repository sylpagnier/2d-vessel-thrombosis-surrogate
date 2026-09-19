"""Stage-A production orchestration (foundation, polish, comsol, promote)."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import torch

from src.training.kinematics_production_config import (
    BEST_CKPT,
    CKPT_LATEST,
    COMSOL_OUTPUT_DIR,
    FoundationConfig,
    LadderConfig,
    PRODUCTION_OUTPUT_DIR,
    PROMOTED_BEST_PATH,
    ProductionRunConfig,
    SKIP_LBFGS_FLAG,
    STATE_LATEST,
    SyntheticPolishConfig,
    bind_env,
    has_comsol_anchor_packs,
)
from src.utils.paths import get_project_root


def _log(msg: str, *, progress: Callable[[str], None] | None = None) -> None:
    fn = progress or print
    fn(msg)


def checkpoint_next_epoch(out_dir: Path) -> int:
    """Return the next training epoch index from latest state/ckpt in ``out_dir``."""
    out_dir = Path(out_dir)
    for name in (STATE_LATEST, CKPT_LATEST):
        path = out_dir / name
        if not path.is_file():
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "epoch" in ckpt:
            return int(ckpt["epoch"]) + 1
        match = re.search(r"kinematics_ckpt_(\d+)\.pth$", name)
        if match:
            return int(match.group(1))
    return 0


def resume_epoch_from_checkpoint(path: Path) -> int:
    if not path.is_file():
        return 0
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("epoch", "best_epoch"):
            if key in ckpt:
                return int(ckpt[key])
    return 0


def clear_foundation_checkpoints(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    flag = out_dir / SKIP_LBFGS_FLAG
    if flag.is_file():
        flag.unlink()
    for name in (STATE_LATEST, CKPT_LATEST):
        p = out_dir / name
        if p.is_file():
            p.unlink()
    for pattern in ("kinematics_ckpt_*.pth", "kinematics_state_*.pth"):
        for p in out_dir.glob(pattern):
            p.unlink(missing_ok=True)


def _has_resume_checkpoint(out_dir: Path) -> bool:
    out_dir = Path(out_dir)
    return (out_dir / STATE_LATEST).is_file() or (out_dir / CKPT_LATEST).is_file()


def _run_train(argv: list[str]) -> int:
    cmd = [sys.executable, "-m", "src.training.train_kinematics_predictor", *argv]
    return int(subprocess.call(cmd))


def _resolve_resume_path(cfg: SyntheticPolishConfig) -> Path:
    out_dir = Path(cfg.output_dir)
    best = out_dir / BEST_CKPT
    resume = str(cfg.resume).strip()
    if resume == "best":
        if not best.is_file():
            raise FileNotFoundError(
                f"missing {best}; run foundation phase first"
            )
        return best
    if resume == "latest":
        for name in (STATE_LATEST, CKPT_LATEST):
            path = out_dir / name
            if path.is_file():
                return path
        raise FileNotFoundError(f"no latest checkpoint in {out_dir}")
    path = Path(resume)
    if not path.is_absolute():
        path = get_project_root() / path
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    return path


def run_foundation(
    cfg: FoundationConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> int:
    """Phase 1 with crash-resume loop and Adam-only LBFGS guard."""
    cfg.bind_process_env()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skip_flag = out_dir / SKIP_LBFGS_FLAG

    if cfg.fresh:
        clear_foundation_checkpoints(out_dir)
        _log("[kin-prod] -Fresh: cleared production_allfix checkpoints.", progress=progress)

    cap_label = str(cfg.graph_cap) if cfg.graph_cap > 0 else "all"
    _log(
        f"[kin-prod] epochs={cfg.epochs} adam={cfg.adam_epochs} "
        f"stage1={cfg.stage1_end} stage2={cfg.stage2_end} graphs={cap_label} seed={cfg.seed}",
        progress=progress,
    )

    fresh = bool(cfg.fresh)
    for attempt in range(1, int(cfg.max_attempts) + 1):
        if skip_flag.is_file():
            bind_env({"KINEMATICS_SKIP_LBFGS": "1"})
            _log(
                "[kin-prod] LBFGS skip flag set (prior crash at Adam handoff); "
                "Adam-only for remaining epochs.",
                progress=progress,
            )
        else:
            bind_env({"KINEMATICS_SKIP_LBFGS": "1"})

        has_ckpt = _has_resume_checkpoint(out_dir)
        train_args = [
            "--no-prompt",
            "--epochs",
            str(cfg.epochs),
            "--adam-epochs",
            str(cfg.adam_epochs),
            "--stage1-end-epoch",
            str(cfg.stage1_end),
            "--stage2-end-epoch",
            str(cfg.stage2_end),
            "--l0l1-only-epochs",
            "0",
            "--hard-mining-start-epoch",
            str(cfg.hard_mining_start),
            "--accum-steps",
            str(cfg.accum_steps),
            "--shuffle-graphs",
            "--graph-load-seed",
            str(cfg.seed),
        ]
        if cfg.quiet:
            train_args.append("--quiet")
        if fresh and attempt == 1 and not has_ckpt:
            train_args.append("--fresh")
            _log(f"[kin-prod] attempt {attempt}: fresh start.", progress=progress)
        elif has_ckpt:
            train_args.extend(["--resume", "latest"])
            nxt = checkpoint_next_epoch(out_dir)
            _log(f"[kin-prod] attempt {attempt}: resume latest (next epoch {nxt}).", progress=progress)
        else:
            train_args.append("--fresh")
            _log(f"[kin-prod] attempt {attempt}: no checkpoint; fresh start.", progress=progress)

        rc = _run_train(train_args)
        if rc == 0:
            _log(
                f"[kin-prod] phase 1 finished OK -> {out_dir / BEST_CKPT}",
                progress=progress,
            )
            return 0

        if not _has_resume_checkpoint(out_dir):
            raise RuntimeError(f"[kin-prod] training failed (exit {rc}) and no checkpoint to resume.")

        nxt = checkpoint_next_epoch(out_dir)
        if nxt >= cfg.adam_epochs:
            skip_flag.touch()
            _log(
                f"[kin-prod] failed near/after Adam epoch {cfg.adam_epochs}; "
                "next retry skips LBFGS (best.pth kept from Adam phase).",
                progress=progress,
            )
        _log(f"[kin-prod] failed (exit {rc}); retrying in {cfg.retry_sleep_s:.0f}s...", progress=progress)
        time.sleep(float(cfg.retry_sleep_s))
        fresh = False

    raise RuntimeError(f"[kin-prod] exceeded {cfg.max_attempts} resume attempts.")


def run_synthetic_polish(
    cfg: SyntheticPolishConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> int:
    cfg.bind_process_env()
    resume_path = _resolve_resume_path(cfg)
    start_ep = resume_epoch_from_checkpoint(resume_path)
    total_epochs = start_ep + int(cfg.finetune_epochs)
    adam_epochs = total_epochs - 5 if cfg.try_lbfgs else total_epochs

    if cfg.try_lbfgs:
        _log("[kin-prod-ft] WARN TryLbfgs: last 5 epochs use LBFGS (experimental; may NaN).", progress=progress)

    focus = "continuity-mild" if cfg.continuity_focus else "balanced"
    _log(
        f"[kin-prod-ft] resume={resume_path} start_ep={start_ep + 1} "
        f"total_epochs={total_epochs} lr={cfg.finetune_lr} focus={focus}",
        progress=progress,
    )

    train_args = [
        "--no-prompt",
        "--resume",
        str(resume_path),
        "--epochs",
        str(total_epochs),
        "--adam-epochs",
        str(adam_epochs),
        "--stage1-end-epoch",
        "40",
        "--stage2-end-epoch",
        "60",
        "--geometry-phase",
        "l2_heavy",
        "--hard-mining-start-epoch",
        str(cfg.hard_mining_start),
        "--finetune-lr",
        str(cfg.finetune_lr),
        "--weight-data",
        "500.0",
        "--shuffle-graphs",
        "--graph-load-seed",
        "42",
    ]
    if cfg.try_lbfgs:
        train_args.extend(["--max-lbfgs-graphs", "2"])
    if cfg.quiet:
        train_args.append("--quiet")

    rc = _run_train(train_args)
    if rc != 0:
        raise RuntimeError(f"[kin-prod-ft] training failed (exit {rc}).")
    _log("[kin-prod-ft] done. best -> outputs/kinematics/production_allfix/kinematics_best.pth", progress=progress)
    return 0


def run_comsol_finetune(
    cfg,
    *,
    progress: Callable[[str], None] | None = None,
) -> int:
    from src.training.kinematics_production_config import ComsolFinetuneConfig

    if not isinstance(cfg, ComsolFinetuneConfig):
        raise TypeError(type(cfg))
    cfg.bind_process_env()
    resume = Path(cfg.resume)
    if not resume.is_absolute():
        resume = get_project_root() / resume
    if not resume.is_file():
        raise FileNotFoundError(f"[kin-comsol-ft] resume checkpoint missing: {resume}")

    _log(
        f"[kin-comsol-ft] resume={resume} holdout={cfg.holdout} epochs={cfg.finetune_epochs} "
        f"lr={cfg.finetune_lr} synth_cap={cfg.synthetic_cap} out={cfg.output_dir}",
        progress=progress,
    )

    argv = [
        sys.executable,
        "scripts/finetune_kine_comsol_anchors.py",
        "--epochs",
        str(cfg.finetune_epochs),
        "--lr",
        str(cfg.finetune_lr),
        "--synthetic-cap",
        str(cfg.synthetic_cap),
        "--resume",
        str(resume),
        "--out-dir",
        str(cfg.output_dir),
    ]
    rc = int(subprocess.call(argv))
    if rc != 0:
        raise RuntimeError(f"[kin-comsol-ft] training failed (exit {rc}).")
    _log(
        "[kin-comsol-ft] done. Run promotion gates before copying to global kinematics_best.pth",
        progress=progress,
    )
    return 0


def promote_checkpoint(
    checkpoint: Path | str,
    *,
    holdout: str = "comsol007",
    progress: Callable[[str], None] | None = None,
) -> int:
    ckpt = Path(checkpoint)
    if not ckpt.is_absolute():
        ckpt = get_project_root() / ckpt
    gate_argv = [
        sys.executable,
        "scripts/check_kinematics_promotion_gates.py",
        "--checkpoint",
        str(ckpt),
        "--holdout",
        holdout,
    ]
    rc = int(subprocess.call(gate_argv))
    if rc != 0:
        raise RuntimeError("[ladder] promotion gates failed (use skip_promote to inspect).")
    dest = get_project_root() / PROMOTED_BEST_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt, dest)
    _log(f"[ladder] promoted -> {dest.as_posix()}", progress=progress)
    return 0


def run_ladder(
    cfg: LadderConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> int:
    _log("[ladder] Stage-A: foundation -> synthetic polish -> comsol anchors -> promote", progress=progress)
    root = get_project_root()
    ckpt = (
        Path(cfg.resume_after_foundation)
        if cfg.resume_after_foundation is not None
        else root / PRODUCTION_OUTPUT_DIR / BEST_CKPT
    )

    if not cfg.skip_foundation:
        _log("[ladder] === phase 1/3: synthetic foundation ===", progress=progress)
        rc = run_foundation(cfg.foundation, progress=progress)
        if rc != 0:
            return rc
        ckpt = root / PRODUCTION_OUTPUT_DIR / BEST_CKPT

    if not ckpt.is_file():
        raise FileNotFoundError(
            f"[ladder] resume checkpoint missing: {ckpt} (run phase 1 first or pass resume_after_foundation)."
        )

    if not cfg.skip_synthetic_polish:
        _log("[ladder] === phase 2/3: synthetic polish (ContinuityFocus finetune) ===", progress=progress)
        polish = cfg.polish
        polish.resume = str(ckpt)
        if not cfg.foundation.quiet:
            polish.quiet = cfg.foundation.quiet
        run_synthetic_polish(polish, progress=progress)
        ckpt = root / PRODUCTION_OUTPUT_DIR / BEST_CKPT

    if cfg.skip_comsol_anchors:
        _log("[ladder] phase 3 comsol anchors skipped (-SkipComsolAnchors).", progress=progress)
    elif not has_comsol_anchor_packs():
        msg = (
            "[ladder] phase 3 skipped: no comsol*.pt under "
            "data/processed/graphs_kinematics_anchors/carreau/"
        )
        if cfg.require_comsol:
            raise RuntimeError(msg)
        _log(f"[ladder] WARN {msg}", progress=progress)
        _log("[ladder] Add COMSOL anchor kine graphs and re-run with skip_foundation + skip_synthetic_polish", progress=progress)
    else:
        _log(
            f"[ladder] === phase 3/3: comsol geometry finetune "
            f"(holdout={cfg.comsol.holdout}, epochs={cfg.comsol.finetune_epochs}) ===",
            progress=progress,
        )
        comsol = cfg.comsol
        comsol.resume = ckpt
        run_comsol_finetune(comsol, progress=progress)
        comsol_best = root / COMSOL_OUTPUT_DIR / BEST_CKPT
        if not cfg.skip_promote:
            _log("[ladder] === promotion gates (COMSOL anchor + synthetic + synthetic L2) ===", progress=progress)
            promote_checkpoint(comsol_best, holdout=cfg.comsol.holdout, progress=progress)
        else:
            _log(f"[ladder] comsol best (not promoted): {comsol_best}", progress=progress)
            _log(
                f"  python scripts/check_kinematics_promotion_gates.py --checkpoint {comsol_best}",
                progress=progress,
            )
        return 0

    if not cfg.skip_promote:
        _log("[ladder] no comsol phase; promoting synthetic best -> outputs/kinematics/kinematics_best.pth", progress=progress)
        dest = root / PROMOTED_BEST_PATH
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt, dest)
        _log(f"[ladder] copied {ckpt} -> {dest}", progress=progress)
    return 0


def run_production(
    cfg: ProductionRunConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> int:
    rc = run_foundation(cfg.ladder.foundation, progress=progress)
    if rc != 0:
        return rc
    if cfg.foundation_only:
        _log("[kin-prod] foundation_only: skipping phases 2-3.", progress=progress)
        return 0
    _log("[kin-prod] chaining phases 2-3 (synthetic polish + comsol geometry finetune)...", progress=progress)
    ladder = cfg.ladder
    ladder.skip_foundation = True
    ladder.resume_after_foundation = get_project_root() / PRODUCTION_OUTPUT_DIR / BEST_CKPT
    return run_ladder(ladder, progress=progress)


__all__ = [
    "checkpoint_next_epoch",
    "clear_foundation_checkpoints",
    "promote_checkpoint",
    "run_comsol_finetune",
    "run_foundation",
    "run_ladder",
    "run_production",
    "run_synthetic_polish",
    "resume_epoch_from_checkpoint",
]
