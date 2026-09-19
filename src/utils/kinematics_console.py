"""Console / tqdm policy for kinematics training (Windows-friendly logs)."""

from __future__ import annotations

import os

# Set by the Stage-A arm scripts (`scripts/stage_a/run_*.sh`), the only setter and one
# outside the tree the knob sweep grepped -- so sweeping these to plain constants made
# every E-series arm a silent no-op for them.  Read from the environment with the swept
# value as the default: unset behaves exactly as the constant did.
KINEMATICS_VAL_EVERY = os.environ.get("KINEMATICS_VAL_EVERY", "")



def kinematics_quiet_logs() -> bool:
    """Epoch + validation summary lines only (no tqdm bars)."""
    return os.environ.get("KINEMATICS_QUIET", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def kinematics_tqdm_enabled() -> bool:
    if kinematics_quiet_logs():
        return False
    raw = os.environ.get("KINEMATICS_TQDM", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def kinematics_val_progress_enabled() -> bool:
    if kinematics_quiet_logs():
        return False
    raw = os.environ.get("KINEMATICS_VAL_PROGRESS", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def kinematics_val_every(total_epochs: int) -> int:
    """Validate every N epochs (every epoch when total_epochs <= 12 or VAL_EVERY=1)."""
    raw = KINEMATICS_VAL_EVERY.strip()
    if raw:
        return max(1, int(raw))
    if total_epochs <= 12:
        return 1
    return 2


def kinematics_skip_lbfgs() -> bool:
    """Recovery sweeps: Adam-only (Apr best was pre-L-BFGS; LBFGS often NaNs on short runs)."""
    return os.environ.get("KINEMATICS_SKIP_LBFGS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
