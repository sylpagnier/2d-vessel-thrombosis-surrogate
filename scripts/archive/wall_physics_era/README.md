# Wall physics era (Phase 3-6)

Zero-parameter wall readout, FIT/DEV/SEALED protocol tables, and calibration diagnostics
from the pre-`clot_ml_v0` wall-cohort physics ladder.

| Script | Purpose |
|--------|---------|
| `predict_wall_clot.py` | Shipped Phase-3 gate + graph-growth wall mask (`--flow pred` deployable) |
| `eval_wall_protocol.py` | FIT / DEV / SEALED comparison table for the physics wall model |
| `fit_gelation_wake_kernel.py` | Gelation wake kernel fit (mat-growth coupling research) |
| `diag_*.py`, `diagnose_*.py` | One-off calibration probes (gate support, wound composition, FEM accuracy, etc.) |

**Active deploy clot** is `clot_ml_v0` (`scripts/promote_clot_ml_v0.py`), not this stack.

**Viz template (optional):** `scripts/gen_offwall_temporal_data.py` uses the kept
`scripts/predict_wall_clot.py` zero-param reference build.
