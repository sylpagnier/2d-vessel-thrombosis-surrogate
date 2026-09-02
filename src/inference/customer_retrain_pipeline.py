"""Orchestrates customer data ingest and Biochem GNN retraining."""

import json
import subprocess
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
        """Run the retrain pipeline on a directory of geometry/solution files."""
        self._stop_event.clear()
        
        # 1. Discover files
        progress_cb(f"Scanning {data_dir.name} for graphs...")
        files = [p for p in data_dir.iterdir() if p.suffix.lower() == ".pt"]
        if not files:
            log_cb("[ERR] No .pt files found in the selected directory.")
            return False
            
        log_cb(f"[i] Found {len(files)} files for retraining.")
        
        # 2. Build custom split manifest
        # (For now, we use a simple 80/20 train/val split or just pass them as anchors)
        manifest_path = get_project_root() / "data" / "reference" / "customer_retrain_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        train_names = [p.stem for p in files]
        val_name = train_names[0] if train_names else "patient007" # Fallback validation anchor
        
        # In a real scenario we'd copy/link these to the cache, but train_biochem_gnn 
        # usually reads from KINEMATICS_PREPARED_CACHE or accepts anchors.
        
        # 3. Launch the subprocess
        progress_cb("Launching Biochem GNN training...")
        
        # Using the standard biochem gnn training entry point
        cmd = [
            "python", "-u", "-m", "src.training.train_biochem_gnn",
            "--step", "species",
            "--val-anchor", val_name,
            "--epochs", "20",  # shorter for customer UI default
            "--early-stop", "10",
        ]
        
        # Pass the custom anchors if we have them
        if train_names:
            cmd.extend(["--anchors", ",".join(train_names)])
            
        log_cb(f"> {' '.join(cmd)}")
        
        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=str(get_project_root()),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output
            for line in self._process.stdout:
                if self._stop_event.is_set():
                    self._process.terminate()
                    break
                log_cb(line.rstrip())
                
            self._process.wait()
            rc = self._process.returncode
            
            if rc == 0:
                progress_cb("Training complete. Model weights saved.")
                return True
            else:
                progress_cb(f"Training failed with exit code {rc}.")
                return False
                
        except Exception as e:
            log_cb(f"[ERR] Subprocess failed: {e}")
            progress_cb("Training failed.")
            return False
            
    def cancel(self):
        """Cancel the running training process."""
        self._stop_event.set()
        if self._process:
            self._process.terminate()
