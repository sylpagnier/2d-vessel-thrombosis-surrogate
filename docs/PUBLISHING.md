# Publishing policy (Local FEM Solver)

This repository is meant to be **publicly pushable**: source, docs, and small reference manifests are versioned. Heavy data, checkpoints, and COMSOL models stay on the machine that trains.

## Track in git

| Path | Why |
|------|-----|
| `src/` | Library, training, tools, tests |
| `scripts/` (active) + `scripts/README.md` | Supported launchers |
| `scripts/archive/MANIFEST.md` | Deleted retired launchers (recover via git) |
| `docs/` (active) + `docs/archive/` | Design docs + historical notebooks |
| `docs/assets/` | Small README / paper figures (tracked) |
| `data/reference/` | Small baseline / architecture JSON + README |
| `customer_geometries/README.txt` | Inbox instructions only |
| `README.md`, `AGENTS.md`, `requirements.txt`, `pytest.ini` | Project entry |

## Keep local (never push)

| Path | Why |
|------|-----|
| `data/raw/`, `data/processed/`, `data/benchmark/` | Large meshes / graphs / CFD extracts |
| `data/reference_local/` | Sweep leftovers (gitignored) |
| `outputs/` | Checkpoints, logs, viz |
| `comsol_models/` | `.mph` sources |
| `customer_geometries/*` (except README) | User uploads |
| `*.pth`, `*.pt`, `*.ckpt` | Weights |
| `.venv/`, `__pycache__/`, `.pytest_cache/`, `.idea/` | Environment / IDE |

## Do not re-add

- Root dumps (`test_legend.png`, `check_nodes_out.txt`, probe logs)
- One-off census / compare JSON under `outputs/`
- Personal notes under `notes/`

## After clone

1. `pip install -r requirements.txt` (venv recommended).
2. Place COMSOL / graph data under `data/` and `comsol_models/` as needed.
3. Optional: copy promoted checkpoints into `outputs/biochem/biochem_gnn/locked/` and `outputs/kinematics/` from your private artifact store.
4. Use `data/reference/*.json` to see which runs are canonical.

## Distributing the customer app

Researchers who just want to *run* the app should never `git clone` -- the checkpoints and
demo geometry it needs are gitignored on purpose (see above) and cloning the source repo
buys nothing without them. Instead they should get the self-contained bundle from the
repo's [GitHub Releases](https://github.com/sylpagnier/2d-vessel-thrombosis-surrogate/releases)
page (see the root README's Quickstart).

To cut a new release from a machine that has the checkpoints locally:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\release_bundle.ps1 -Version 1.1
```

This builds the zip (`scripts/build_customer_bundle.ps1`), then tags, pushes, and publishes
it as a GitHub Release via the `gh` CLI -- one command, and none of the private artifacts it
packages ever leave this machine or pass through CI. Full detail on what's in the bundle and
why: [`CUSTOMER_INSTALLER.md`](CUSTOMER_INSTALLER.md).

## Script surface

- **Supported:** only what [`scripts/README.md`](../scripts/README.md) lists.
- **Deleted archives:** inventory in [`scripts/archive/MANIFEST.md`](../scripts/archive/MANIFEST.md) (git history for recovery).
- Prefer not adding one-off `analyze_*.py` / `_print_*.py` to the active `scripts/` root unless documented in that README.
