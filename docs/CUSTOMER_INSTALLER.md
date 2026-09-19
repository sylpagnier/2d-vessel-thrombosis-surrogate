# Customer Predict UX -- release bundles (Windows, macOS, Linux)

`src/tools/customer_predict_web.py` (the "2D Thrombosis Surrogate" clot-prediction browser tool) is
packaged into two release zips, both carrying the ~11 MB of `clot_ml_0` checkpoints it actually
loads, the demo vessel and a double-click launcher:

| Zip | For | How it gets Python |
|-----|-----|--------------------|
| `…-win64-<Version>.zip` | Windows x64 | Ships an embeddable Python with every dependency installed. Nothing to install. |
| `…-macos-linux-<Version>.zip` | Apple Silicon Macs, x86-64 Linux | `run.command` / `run.sh` (`scripts/bundle/run_unix.sh`) builds a `.venv` next to itself on first launch from `requirements-lock.txt`, using the user's Python 3.12–3.14. |

No research Python env, no CUDA and no separate checkpoint hunt on the other end.

## Build

```
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_customer_bundle.ps1
```

Optional `-Version` (default `1.0`) and `-PythonVersion` (default `3.13.7`, matched to the dev
`.venv`) parameters. Output: `dist\2DThrombosisSurrogate-Predict-win64-<Version>.zip` and
`dist\2DThrombosisSurrogate-Predict-macos-linux-<Version>.zip`. The build runs on Windows; the
macOS / Linux zip is assembled from the Windows bundle minus its interpreter.

The script downloads an official embeddable Python distribution and `get-pip.py` from
python.org / pypa.io (needs internet access to build; the resulting bundle needs none to run).
It reinstalls CPU-only `torch` from `https://download.pytorch.org/whl/cpu`, then the rest of
`requirements-customer.txt` -- a deliberately smaller set than the full research
`requirements.txt` (only `pytest` is actually dropped; `pandas` and `mph`, the COMSOL/MATLAB
bridge, both stay in even though the deploy path never calls their functions, because
`src/data_gen/__init__.py` imports `AnchorGenerator`/`ComsolAnchorDataExtractor` eagerly at package
level and pulls them in transitively -- see the comment in `requirements-customer.txt`).
`scikit-learn` is version-pinned (`==1.8.0`): the readout heads in
`outputs/clot_ml/locked/clot_ml_final/temporal.pkl` are pickled sklearn objects, and sklearn's
pickle format is not stable across versions.

## What's in the bundle, and why that's everything it needs

- `python/` -- embeddable Python + pip-installed CPU deps, self-contained (gmsh's pip wheel
  bundles its own native SDK on Windows -- no separate Gmsh install needed).
- `src/`, `scripts/` -- the git-tracked files only (`git ls-files`), so nothing `.gitignore`
  keeps local reaches the public zip.
- `pyproject.toml` -- a marker file. `src/utils/paths.py::get_project_root()` looks for either
  this file or a `.git` folder to find the app root; the bundle has neither by default, so
  this file has to travel with it for checkpoint/inbox paths to resolve correctly outside a
  git checkout.
- `outputs/clot_ml/locked/{clot_ml_final_0,clot_ml_final_w,clot_ml_final}/` -- the
  `clot_ml_0` chain (unified wound/no-wound model -> wound base -> 9-member GNN ensemble and
  temporal heads), and the only checkpoints `CustomerDeployPipeline.run()` loads for this tool:
  Flow Simulator and clot modes call the identical pipeline, and no kinematics or wall-model
  checkpoint is on this path. **Pinned by name, not resolved at build time**, so a bundle ships a
  validated chain regardless of what else sits in `outputs/clot_ml/locked/`. Bump the names in
  `build_customer_bundle.ps1` deliberately once a new artifact is promoted and validated; the
  script's comment has the one-liner that walks the current chain.
- `data/reference/clot_gnn_locked.json` -- `src/clot_ml/locked.py::load_ensemble()` reads this
  pointer file unconditionally at the top of the function, even on the branch that ignores its
  contents (an explicit `name=` is passed). Tiny reference JSON, not a checkpoint.
- `scripts/` is deploy-reachable despite the name: `locked.py`'s readout selection
  (`expected_tuned` / `resid_adapt`) lazily imports helpers from `scripts/eval_*.py` at call
  time, so the tracked tree ships whole rather than hand-picking imports.
- `customer_geometries/` -- the geometry inbox: its `README.txt` plus `demo_stenosis_vessel.pt`
  (`comsol041`, a stenosis anchor with a COMSOL solve behind it), so first launch shows a
  prediction instead of an empty list; `data/raw/biochem_anchors/comsol041.msh` travels with it.
- `run.bat` -- launches `python -m src.tools.customer_predict_web --cpu`. The app itself opens
  the user's browser automatically a moment after the server starts.

The macOS / Linux zip holds the same files except `python/` and `run.bat`, plus:

- `run.command` and `run.sh` -- the same script, `scripts/bundle/run_unix.sh`. It refuses
  unsupported platforms with the reason, finds a Python 3.12–3.14 (on a Mac, a native arm64
  one), creates `.venv/` on first launch, installs `requirements-lock.txt` (on Linux, torch from
  the CPU wheel index first, because PyPI's Linux torch is the multi-GB CUDA build), then starts
  the app. A hash of the lock file is stamped into `.venv/`, so a changed lock rebuilds it.
- `requirements-lock.txt` -- `pip freeze` of the Windows bundle's interpreter with `+cpu`
  dropped: the exact versions the release was tested with. Every pinned package ships macOS
  arm64 and Linux x86-64 wheels for Python 3.12–3.14 (checked against PyPI for v1.2.1).
- Zipped by `scripts/bundle/make_zip.py`, which records Unix mode bits. `Compress-Archive` does
  not, and macOS then unzips `run.command` without its executable bit, so Finder cannot open it.

## Publish

```
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\release_bundle.ps1 -Version <Version>
```

`release_bundle.ps1` runs the build above, then tags, pushes, and runs `gh release create` to
attach both zips to a GitHub Release, with a confirm prompt and rollback if any step fails (see
[`docs/PUBLISHING.md`](PUBLISHING.md#distributing-the-customer-app)). It requires the `gh` CLI
to be installed and authenticated (`gh auth login`), and a clean working tree so the bundle
matches the tag it ships under.

Publishing is a deliberate manual step when the tool changes meaningfully, same as the repo's
existing manual checkpoint-promotion scripts -- not wired into CI, since the checkpoints and
demo geometry it packages are gitignored and never reach a CI runner.

## Security model

- **Local only, no authentication.** The server binds `127.0.0.1` and refuses requests whose
  `Host` or `Origin` is not that address, and POSTs that are not `application/json`. That
  blocks a web page open in the same browser (cross-site POST, DNS rebinding) from starting
  jobs. Passing `--host` to expose it on a network gives everyone who can reach it full use
  of the app; the launcher warns when you do.
- **Uploaded `.pt` graphs cannot run code.** Customer graphs (web upload, inbox, retrain
  folder) load through `src/utils/safe_load.py`: tensors and PyG graph classes only, never a
  full unpickle.
- **The shipped checkpoints are trusted.** `outputs/clot_ml/locked/` is loaded with full
  unpickling (`torch.load(weights_only=False)`, `pickle`). Only use a bundle downloaded from
  this repository's Releases, and never drop a checkpoint from elsewhere into that folder.
  Each release lists the SHA-256 of both zips in its notes and in `SHA256SUMS.txt`
  (written by `scripts/release_bundle.ps1`); check a copy obtained any other way against it.
- **Request values are bounded on the server too.** Every numeric field is rejected if it is
  not a finite number and clamped to its slider's range, so a raw API call cannot hand the
  mesher a geometry the UI would never produce.
- **Build downloads are pinned.** `build_customer_bundle.ps1` checks the embeddable Python zip
  and the pip wheel against digests in the script; a new `-PythonVersion` needs its digest added.
- **Cancel stops the whole retrain.** The training subprocess watches the process that started
  it and exits when it does, and Cancel kills its process tree, so no orphaned run keeps the GPU.
- **COMSOL `.mph` files are not sandboxed.** Retrain opens them through COMSOL itself; only
  retrain on solves you or a collaborator produced.

## Known limitations

- **Unzip to a short path.** Extracting to a deeply nested path (over roughly 240 characters
  total, e.g. a long chain of subfolders under Downloads, or certain synced-folder setups) hits
  Windows' legacy `MAX_PATH` (260 char) limit, which breaks loading
  `sklearn.metrics._pairwise_distances_reduction._datasets_pair.pyd` specifically -- scikit-learn's
  package nesting is unusually deep, and this is the one file most likely to cross the limit.
  The symptom is a `ModuleNotFoundError` mentioning `sklearn` on the very first prediction,
  even though the app itself starts and serves the page fine. `run.bat` checks the path length
  and warns before this can happen; README.txt tells users to unzip near a drive root (e.g.
  `C:\ThrombosisSurrogate\`). Verified: the identical bundle at a short path (~180 chars) runs a
  full prediction end to end; the same bytes at a long path (~260+ chars) fail on that one
  import. Not something the build can fix from its side -- it's how Windows resolves paths for
  the caller, not a property of the files themselves.
- **Horizon capped at 30,000 s (8.33 h).** The longest COMSOL runs; nothing beyond has ground
  truth. Both apps cap the simulation-time control there, and the server clamps raw API
  requests too (`VALIDATED_HORIZON_S` in `src/tools/customer_predict_metrics.py`). Hours are
  real 3,600 s hours; before v1.2.2 the apps used 3,750 s so that "8 h" meant 30,000 s.
- **Platforms.** Windows x64, Apple Silicon Macs and x86-64 Linux. **Intel Macs** are out:
  PyTorch stopped publishing Intel-Mac wheels. **ARM Linux** is out: gmsh publishes no wheel
  for it. The macOS / Linux zip is tested end to end on Ubuntu 24.04 (WSL): first-launch setup, both
  demo predictions, results identical to the Windows bundle. The macOS path shares the script and
  every pinned package has a macOS arm64 wheel, but it has not been run on a Mac by us.
- **macOS / Linux need Python and a one-time download.** Python 3.12+ must be installed (numpy
  and scipy require it), and the first launch downloads ~1 GB of packages. On Linux, gmsh also needs system libraries minimal installs lack (Ubuntu 24.04: `libglu1-mesa libopengl0 libxft2 libgomp1`); the launcher checks and names them.
- **Unsigned.** Windows shows a SmartScreen "unrecognized app" prompt ("More info" -> "Run
  anyway"); macOS Gatekeeper may block `run.command` as from an unidentified developer
  (right-click -> Open). Code-signing would remove both but is out of scope for now.
- **CPU-only, by design.** Most laptops this targets have no NVIDIA GPU. Measured on the v1.2
  Windows bundle: 157 s for a 4 h parametric injured vessel, 283 s for the 8 h demo vessel. A
  researcher with CUDA should use the normal `.venv` + `scripts/go_customer_predict_web.ps1`
  path instead, which is much faster.
