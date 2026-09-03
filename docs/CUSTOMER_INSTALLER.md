# Customer Predict UX -- Windows installer bundle

`src/tools/customer_predict_web.py` (the "Local FEM Solver" clot-prediction browser tool) can
be packaged into a self-contained Windows bundle: embeddable Python + the CPU-only deploy
dependencies + the ~11 MB of `clot_ml_0` checkpoints it actually loads + a double-click
launcher. No research Python env, no CUDA, no separate checkpoint hunt on the other end.

## Build

```
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_customer_bundle.ps1
```

Optional `-Version` (default `1.0`) and `-PythonVersion` (default `3.13.7`, matched to the dev
`.venv`) parameters. Output: `dist\LocalFEMSolver-Predict-win64-<Version>.zip`.

The script downloads an official embeddable Python distribution and `get-pip.py` from
python.org / pypa.io (needs internet access to build; the resulting bundle needs none to run).
It reinstalls CPU-only `torch` from `https://download.pytorch.org/whl/cpu`, then the rest of
`requirements-customer.txt` -- a deliberately smaller set than the full research
`requirements.txt` (only `pytest` is actually dropped; `pandas` and `mph`, the COMSOL/MATLAB
bridge, both stay in even though the deploy path never calls their functions, because
`src/data_gen/__init__.py` imports `AnchorGenerator`/`PatientDataExtractor` eagerly at package
level and pulls them in transitively -- see the comment in `requirements-customer.txt`).

## What's in the bundle, and why that's everything it needs

- `python/` -- embeddable Python + pip-installed CPU deps, self-contained (gmsh's pip wheel
  bundles its own native SDK on Windows -- no separate Gmsh install needed).
- `src/` -- full app source tree (cheap to include wholesale; only the checkpoints are large).
- `pyproject.toml` -- a marker file. `src/utils/paths.py::get_project_root()` looks for either
  this file or a `.git` folder to find the app root; the bundle has neither by default, so
  this file has to travel with it for checkpoint/inbox paths to resolve correctly outside a
  git checkout.
- `outputs/clot_ml/locked/{clot_ml_v0,clot_gnn_v6}/` -- the only checkpoints
  `CustomerDeployPipeline.run()` ever loads for this tool (traced end to end: Flow Simulator
  and Scientific modes call the identical pipeline, no extra footprint; no kinematics or
  wall-model checkpoint is on this path at all).
- `customer_geometries/README.txt` -- the empty geometry inbox, matching what the app already
  documents for `.msh`/`.nas`/`.pt` uploads.
- `run.bat` -- launches `python -m src.tools.customer_predict_web --cpu`. The app itself opens
  the user's browser automatically a moment after the server starts.

## Publish

```
gh release create v<Version> dist\LocalFEMSolver-Predict-win64-<Version>.zip \
    --title "Predict UX v<Version>" \
    --notes "Local FEM Solver Predict, Windows, CPU-only. Unzip and run run.bat."
```

Publishing is a deliberate manual step when the tool changes meaningfully, same as the repo's
existing manual checkpoint-promotion scripts -- not wired into CI.

## Known limitations

- **Windows only.** No embeddable-Python equivalent bundling story for macOS exists here yet;
  a Mac build would need a different approach (e.g. a relocatable venv + shell launcher).
- **Unsigned.** First launch triggers a Windows SmartScreen "unrecognized app" prompt (this is
  expected, not a bug -- click "More info" -> "Run anyway"). Code-signing would remove this
  but is out of scope for now.
- **CPU-only, by design.** Verified end-to-end (~13s for a small vessel at draft mesh
  resolution) -- most laptops this bundle targets have no NVIDIA GPU. A researcher with CUDA
  should keep using the normal `.venv` + `scripts/go_customer_predict_web.ps1` path instead,
  which is faster.
