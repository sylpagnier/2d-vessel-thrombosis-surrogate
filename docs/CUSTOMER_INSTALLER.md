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
`scikit-learn` is version-pinned (`==1.8.0`): the readout heads in
`outputs/clot_ml/locked/clot_gnn_v6/temporal.pkl` are pickled sklearn objects, and sklearn's
pickle format is not stable across versions.

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
  wall-model checkpoint is on this path at all). **Pinned by name, not resolved dynamically**:
  `clot_ml_0` is being retrained as of 2026-09-03 (see the in-progress
  `outputs/clot_ml/locked/DeployClot*` folders, no manifest yet). Copying these two specific,
  already-validated directories means a bundle built today ships the last known-good artifact
  regardless of what's mid-training alongside it -- bump the two names in
  `build_customer_bundle.ps1` deliberately once a retrained artifact is promoted and validated.
- `data/reference/clot_gnn_locked.json` -- `src/clot_ml/locked.py::load_ensemble()` reads this
  pointer file unconditionally at the top of the function, even on the branch that ignores its
  contents (an explicit `name=` is passed). Tiny reference JSON, not a checkpoint.
- `scripts/` -- despite the directory name, this is deploy-reachable: `locked.py`'s readout
  selection (`expected_tuned` / `resid_adapt`) lazily imports helper functions from
  `scripts/eval_*.py` at call time. Copied wholesale (7 MB) rather than hand-picking which of
  the several lazy imports a given run happens to hit.
- `customer_geometries/README.txt` -- the empty geometry inbox, matching what the app already
  documents for `.msh`/`.nas`/`.pt` uploads.
- `run.bat` -- launches `python -m src.tools.customer_predict_web --cpu`. The app itself opens
  the user's browser automatically a moment after the server starts.

## Publish

```
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\release_bundle.ps1 -Version <Version>
```

`release_bundle.ps1` runs the build above, then tags, pushes, and runs `gh release create` to
attach the zip to a GitHub Release, with a confirm prompt and rollback if any step fails (see
[`docs/PUBLISHING.md`](PUBLISHING.md#distributing-the-customer-app)). It requires the `gh` CLI
to be installed and authenticated (`gh auth login`), and a clean working tree so the bundle
matches the tag it ships under.

Publishing is a deliberate manual step when the tool changes meaningfully, same as the repo's
existing manual checkpoint-promotion scripts -- not wired into CI, since the checkpoints and
demo geometry it packages are gitignored and never reach a CI runner.

## Known limitations

- **Unzip to a short path.** Extracting to a deeply nested path (over roughly 240 characters
  total, e.g. a long chain of subfolders under Downloads, or certain synced-folder setups) hits
  Windows' legacy `MAX_PATH` (260 char) limit, which breaks loading
  `sklearn.metrics._pairwise_distances_reduction._datasets_pair.pyd` specifically -- scikit-learn's
  package nesting is unusually deep, and this is the one file most likely to cross the limit.
  The symptom is a `ModuleNotFoundError` mentioning `sklearn` on the very first prediction,
  even though the app itself starts and serves the page fine. `run.bat` checks the path length
  and warns before this can happen; README.txt tells users to unzip near a drive root (e.g.
  `C:\LocalFEMSolver\`). Verified: the identical bundle at a short path (~180 chars) runs a
  full prediction end to end; the same bytes at a long path (~260+ chars) fail on that one
  import. Not something the build can fix from its side -- it's how Windows resolves paths for
  the caller, not a property of the files themselves.
- **Windows only.** No embeddable-Python equivalent bundling story for macOS exists here yet;
  a Mac build would need a different approach (e.g. a relocatable venv + shell launcher).
- **Unsigned.** First launch triggers a Windows SmartScreen "unrecognized app" prompt (this is
  expected, not a bug -- click "More info" -> "Run anyway"). Code-signing would remove this
  but is out of scope for now.
- **CPU-only, by design.** Verified end-to-end (~13s for a small vessel at draft mesh
  resolution) -- most laptops this bundle targets have no NVIDIA GPU. A researcher with CUDA
  should keep using the normal `.venv` + `scripts/go_customer_predict_web.ps1` path instead,
  which is faster.
