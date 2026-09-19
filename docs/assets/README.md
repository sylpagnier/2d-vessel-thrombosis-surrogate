# README assets

Figures used by the project README. Keep them small — prefer PNG under ~500 KB.

| File | Description |
|------|-------------|
| `customer_predict_demo.png` | The Predict app mid-run: wall clot forming across a stenosis, with the wall-coverage / occlusion / span readout |
| `flow_fem_vs_deq_vs_comsol.png` | Flow field on `comsol020` — RGP-DEQ surrogate, in-house FEM, and COMSOL ground truth, with matching error maps. Regenerate with `python scripts/publication/plot_fig1_flow.py` |
| `architecture.png` | The shipped `clot_ml_0` pipeline, wound branch included. Copy of `plot_biochem_architecture.py`'s output |
| `applications.png` | Stenosis and aneurysm sweeps (total clot mass against strength); surrogate predictions, no COMSOL ground truth. Copy of `plot_applications.py`'s output |
| `oof_example.png` | One out-of-fold vessel (`comsol021`, near the cohort median) at final time, model vs COMSOL, with wall/off-wall BATC₀ and strict F1. Regenerate with `python -m scripts.publication.plot_readme_example --stem comsol021` (`--list` scores every OOF vessel) |
| `timing_cost.png` | Per-vessel runtime by stage and the speed-up against COMSOL. Copy of `plot_timing.py`'s output |
