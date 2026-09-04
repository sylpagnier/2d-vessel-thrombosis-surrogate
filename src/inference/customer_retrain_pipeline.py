"""Orchestrates customer data ingest and clot_ml_0 candidate retraining.

Delegates the actual work to ``scripts/customer_retrain_run.py`` in a subprocess: file
recognition (.pt graphs vs .mph COMSOL solves), feature-cache building, a real train/val/test
split, training, and scoring. This process never imports torch/torch-geometric itself, so a
long or crashing training run cannot take the UI server down with it.

The result is a CANDIDATE checkpoint under ``outputs/customer/retrain_candidates/<timestamp>/``
-- it is never auto-promoted to the live ``clot_ml_0`` checkpoint. See
``scripts/customer_retrain_run.py`` for why, and ``scripts/promote_clot_gnn_v4.py`` for the
manual promotion step.
"""

import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

from src.utils.paths import get_project_root


class CustomerRetrainPipeline:
    def __init__(self, require_cuda: bool = True):
        self.require_cuda = require_cuda
        self._process = None
        self._stop_event = threading.Event()

    def run(self, data_dir: Path, progress_cb: Callable[[str], None], log_cb: Callable[[str], None]) -> bool:
        """Run the retrain pipeline on a directory of .pt graphs / .mph COMSOL solves."""
        self._stop_event.clear()
        progress_cb(f"Scanning {data_dir.name} for graphs and COMSOL files...")

        cmd = [
            sys.executable, "-u", str(get_project_root() / "scripts" / "customer_retrain_run.py"),
            "--data-dir", str(data_dir),
        ]
        log_cb(f"> {' '.join(cmd)}")

        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=str(get_project_root()),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in self._process.stdout:
                if self._stop_event.is_set():
                    self._process.terminate()
                    break
                log_cb(line.rstrip())
                progress_cb(line.rstrip())

            self._process.wait()
            rc = self._process.returncode

            if rc == 0:
                progress_cb("Retrain complete. Candidate checkpoint saved for review.")
                return True
            progress_cb(f"Retrain failed with exit code {rc}.")
            return False

        except Exception as e:
            log_cb(f"[ERR] Subprocess failed: {e}")
            progress_cb("Retrain failed.")
            return False

    def cancel(self):
        """Cancel the running training process."""
        self._stop_event.set()
        if self._process:
            self._process.terminate()
