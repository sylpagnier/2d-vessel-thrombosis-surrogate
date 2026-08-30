from pathlib import Path


def get_project_root() -> Path:
    """Returns the project root folder."""
    current_path = Path(__file__).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return current_path.parent


def outputs_root() -> Path:
    """Unified generated-artifact root (checkpoints, reports, logs)."""
    return get_project_root() / "outputs"


def data_root() -> Path:
    """Canonical dataset tree: ``data/raw``, ``data/processed``, ``data/benchmark`` (scratch), etc."""
    return get_project_root() / "data"




def comsol_models_dir() -> Path:
    """Reference COMSOL projects and templates (versioned assets, not training checkpoints)."""
    return get_project_root() / "comsol_models"


def reports_dir() -> Path:
    """Generated reports: CSVs, figures, validation PNGs, training diaries, debug logs → ``outputs/reports``."""
    p = outputs_root() / "reports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def reports_subdir(*parts: str) -> Path:
    """Create and return a scoped reports subdirectory under ``outputs/reports``."""
    p = reports_dir().joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def reports_training_dir(*parts: str) -> Path:
    """Training-origin artifacts under ``outputs/reports/training``."""
    return reports_subdir("training", *parts)


def reports_evaluation_dir(*parts: str) -> Path:
    """Evaluation / benchmark artifacts under ``outputs/reports/evaluation``."""
    return reports_subdir("evaluation", *parts)


def reports_inspection_dir(*parts: str) -> Path:
    """Inspection-tool artifacts under ``outputs/reports/inspection``."""
    return reports_subdir("inspection", *parts)


def kinematics_dir() -> Path:
    """Kinematics checkpoints and validation artifacts under ``outputs/kinematics``.

    Override with env ``KINEMATICS_OUTPUT_DIR`` (absolute or project-relative) for
    per-leg recovery sweeps without clobbering the global best checkpoint.
    """
    import os

    override = os.environ.get("KINEMATICS_OUTPUT_DIR", "").strip()
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = get_project_root() / p
    else:
        p = outputs_root() / "kinematics"
    p.mkdir(parents=True, exist_ok=True)
    return p


def biochem_dir() -> Path:
    """Biochem / phase-B checkpoints under ``outputs/biochem``."""
    p = outputs_root() / "biochem"
    p.mkdir(parents=True, exist_ok=True)
    return p


def stage_a_dir() -> Path:
    """Backward-compatible alias for ``kinematics_dir()``."""
    return kinematics_dir()


def stage_b_dir() -> Path:
    """Backward-compatible alias for ``biochem_dir()``."""
    return biochem_dir()


def resolve_checkpoint(stage: str, filename: str) -> Path:
    """Return the canonical checkpoint path, falling back to legacy stage dirs when reading old runs."""
    key = str(stage).strip().lower()
    if key in {"a", "kinematics", "t1", "t2"}:
        canonical = kinematics_dir() / filename
        legacy = outputs_root() / "stage_a" / filename
    elif key in {"b", "biochem", "phase_b", "t3"}:
        canonical = biochem_dir() / filename
        legacy = outputs_root() / "stage_b" / filename
    else:
        raise ValueError(f"Unknown checkpoint stage: {stage!r}")
    return canonical if canonical.exists() or not legacy.exists() else legacy
