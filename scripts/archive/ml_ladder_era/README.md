# ML-ladder / differentiable-wall era (retired 2026-09-01)

Pre-`clot_ml_v0` experimental stack: learned corrections on top of a differentiable
COMSOL-style wall ODE (`src/archive/differentiable_wall_model/`), Optuna sweeps, and
temporal-head ladders (`sweep_ml_v2`, `sweep_temporal_only`, `train_ml_ladder`).

**Superseded by:** `clot_ml_v0` (unified wall + wound readout) and deploy-faithful
`biochem_gnn` / RGP-DEQ kinematics. Findings that motivated the parity gate and
clean-protocol eval live in `docs/PHASE6_HANDOFF.md` and `docs/PHASE7_FINDINGS.md`.

Do not run these scripts on the active GPU queue. Recover via git if a number must
be reproduced.
