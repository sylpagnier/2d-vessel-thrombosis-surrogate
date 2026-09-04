"""Customer-facing retrain driver: .pt graphs (+ best-effort .mph) -> a candidate clot_ml_0 GNN.

Invoked by ``src.inference.customer_retrain_pipeline.CustomerRetrainPipeline`` as a subprocess
so the web/desktop UI process never blocks on torch training or a COMSOL LiveLink session.

What this does NOT do: touch ``outputs/clot_ml/locked/`` or ``data/reference/clot_gnn_locked.json``.
It trains one lightweight ``ClotGNN`` configuration (not the shipped 9-member ensemble) on
whatever valid anchors it finds, and writes a candidate checkpoint + scores for a human to
review.  Promoting a candidate to the live model is a separate, deliberate step -- see
``scripts/promote_clot_gnn_v4.py``.

    python scripts/customer_retrain_run.py --data-dir "C:\\path\\to\\graphs"
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.clot_ml.data import attach_physics  # noqa: E402
from src.clot_ml.features import build_features, feature_matrix  # noqa: E402
from src.clot_ml.gnn import ClotGNN  # noqa: E402
from src.clot_ml.protocol import Bench  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.wall_cohort_splits import (  # noqa: E402
    CLOT_FREE, DEV, FIT, MIN_T, SEALED,
)
from train_clot_gnn import GRID, apply_readout, pick_readout, train_one  # noqa: E402

PROC_DIR = REPO / "data/processed/graphs_biochem_anchors"
CANDIDATE_ROOT = REPO / "outputs/customer/retrain_candidates"
CANONICAL_NAMES = set(FIT) | set(DEV) | set(SEALED) | set(CLOT_FREE)

# Modest, CPU-friendly single-config training -- not the shipped 9-member ensemble.  Speed
# matters more than squeezing out the last point here; a promising candidate gets a real
# ensemble run through the standard promote scripts afterward.
DEFAULT_CFG = dict(
    epochs=40, dim=64, layers=4, drop=0.1, lr=3e-3, wd=1e-4, pos_weight=30.0,
    reg_w=1.0, metric_w=2.0, metric_start=0.3, rounds=1, metric="legacy",
    off_mult=1.0, empty_gt_loss="none", burden_w=0.0, shape_w=0.0, adv_fb=False,
    off_only=False, clot_free_w=1.0,
)


@dataclass
class FileVerdict:
    path: Path
    kind: str          # "pt" | "mph"
    accepted: bool
    stem: str | None = None
    reason: str = ""


@dataclass
class RunReport:
    verdicts: list = field(default_factory=list)

    def log_ok(self, v: FileVerdict) -> None:
        self.verdicts.append(v)
        print(f"[ok  ] {v.path.name} -> anchor '{v.stem}'", flush=True)

    def log_skip(self, v: FileVerdict) -> None:
        self.verdicts.append(v)
        print(f"[skip] {v.path.name}: {v.reason}", flush=True)


def _safe_stem(name: str) -> str:
    import re
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", Path(name).stem).strip("_") or "anchor"
    if stem in CANONICAL_NAMES or not stem.startswith("customer_"):
        stem = f"customer_{stem}"
    return stem


def _validate_pt_graph(path: Path) -> tuple[object | None, str]:
    """Load a .pt and check it is a real (non-synthetic) biochem-anchor-schema graph."""
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        return None, f"could not load ({exc})"
    required = ("mask_wall", "edge_index", "x", "x_channel_names", "y", "y_channel_names",
                "u_ref", "d_bar", "num_nodes")
    missing = [f for f in required if not hasattr(data, f) or getattr(data, f) is None]
    if missing:
        return None, f"not a recognized biochem graph (missing {', '.join(missing)})"
    if bool(getattr(data, "research_synthetic", False)):
        return None, ("carries placeholder/synthetic labels, not real simulation ground "
                       "truth -- cannot be used to retrain")
    t = int(data.y.shape[0])
    if t < MIN_T:
        return None, f"only {t} of {MIN_T} required timesteps"
    return data, ""


def discover_and_classify(data_dir: Path, report: RunReport) -> dict[str, object]:
    """Returns ``{stem: torch_geometric.Data}`` for every accepted anchor."""
    pt_files = sorted(p for p in data_dir.iterdir() if p.suffix.lower() == ".pt")
    mph_files = sorted(p for p in data_dir.iterdir() if p.suffix.lower() == ".mph")
    print(f"[i] Found {len(pt_files)} .pt file(s) and {len(mph_files)} .mph file(s) in "
          f"{data_dir}", flush=True)

    accepted: dict[str, object] = {}

    for p in pt_files:
        data, reason = _validate_pt_graph(p)
        if data is None:
            report.log_skip(FileVerdict(p, "pt", False, reason=reason))
            continue
        stem = _safe_stem(p.stem)
        accepted[stem] = data
        report.log_ok(FileVerdict(p, "pt", True, stem=stem))

    if mph_files:
        _classify_mph(mph_files, data_dir, accepted, report)

    return accepted


def _classify_mph(mph_files: list[Path], data_dir: Path, accepted: dict[str, object],
                   report: RunReport) -> None:
    try:
        import mph  # noqa: F401
    except Exception:
        for p in mph_files:
            report.log_skip(FileVerdict(
                p, "mph", False,
                reason=("COMSOL LiveLink ('mph' package/COMSOL install) is not available in "
                        "this environment -- .mph conversion needs a machine with COMSOL. "
                        "Run 'python -m src.tools.extract_biochem_comsol' there to produce a "
                        ".pt graph, or supply pre-built .pt graphs directly.")))
        return

    for p in mph_files:
        mesh = next((data_dir / f"{p.stem}{ext}" for ext in (".msh", ".nas")
                     if (data_dir / f"{p.stem}{ext}").exists()), None)
        if mesh is None:
            report.log_skip(FileVerdict(
                p, "mph", False,
                reason=("no matching .msh/.nas mesh alongside it (COMSOL sampling needs the "
                        "anchor mesh, same stem) -- skipping")))
            continue
        stem = _safe_stem(p.stem)
        try:
            _convert_mph(p, mesh, stem, data_dir)
        except Exception as exc:  # noqa: BLE001
            report.log_skip(FileVerdict(p, "mph", False, reason=f"conversion failed ({exc})"))
            continue
        pt_out = PROC_DIR / f"{stem}.pt"
        if not pt_out.exists():
            report.log_skip(FileVerdict(p, "mph", False, reason="conversion did not produce a graph"))
            continue
        data, reason = _validate_pt_graph(pt_out)
        if data is None:
            report.log_skip(FileVerdict(p, "mph", False, reason=f"converted but invalid ({reason})"))
            continue
        accepted[stem] = data
        report.log_ok(FileVerdict(p, "mph", True, stem=stem))


def _convert_mph(mph_path: Path, mesh_path: Path, stem: str, data_dir: Path) -> None:
    """Best-effort real extraction, reusing the researcher .mph -> graph pipeline."""
    from src.data_gen.lib.extract_biochem_comsol_data import ComsolAnchorDataExtractor
    from src.tools.prepare_biochem_anchors import enrich_anchor_meshes

    raw_dir = REPO / "data/raw/biochem_anchors"
    label_dir = REPO / "data/processed/cfd_results_biochem"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mesh_path, raw_dir / f"{stem}{mesh_path.suffix.lower()}")
    sidecar = data_dir / f"{mesh_path.stem}.json"
    if sidecar.exists():
        shutil.copy2(sidecar, raw_dir / f"{stem}.json")

    extractor = ComsolAnchorDataExtractor(phase="biochem_anchors", raw_dir=raw_dir,
                                      label_dir=label_dir, proc_dir=PROC_DIR)
    extractor.pull_comsol_exports(stem, model_path=mph_path, force=True)
    enrich_anchor_meshes(raw_dir, overwrite=False, dry_run=False, stems=[stem], quiet=True)
    extractor.process_comsol_anchor(stem)


def build_cache(anchors: dict[str, object], report: RunReport) -> dict[str, dict]:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    cache: dict[str, dict] = {}
    for stem, data in anchors.items():
        try:
            S = build_features(data, bio, phys, flow="gt")
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {stem}: feature extraction failed ({exc})", flush=True)
            continue
        if S["y"].sum() == 0:
            print(f"[skip] {stem}: empty ground truth (no clot anywhere) -- cannot supervise "
                  f"training on this vessel", flush=True)
            continue
        X, cols = feature_matrix(S["F"])
        cache[stem] = dict(
            X=X, cols=np.array(cols), y=S["y"], mat_gt=S["mat_gt"], wall=S["wall"],
            solid=S["solid"], shell=S["shell"], owner=S["owner"], edge_index=S["edge_index"],
            pos=S["pos"], mat_phys=S["mat_phys"], gate=S["gate"], sr=S["sr"], spd=S["spd"],
            u=S["u"], v=S["v"])
        # Persist the validated anchor into the canonical pack directory too, so a researcher
        # who wants a full ensemble retrain can point the standard promote scripts at it.
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(anchors[stem], PROC_DIR / f"{stem}.pt")
    return attach_physics(cache)


def split_anchors(stems: list[str], seed: int = 42) -> tuple[list[str], list[str], list[str]]:
    stems = sorted(stems)
    random.Random(seed).shuffle(stems)
    n = len(stems)
    if n < 5:
        return stems[:-1], stems[-1:], []
    n_val = max(1, round(n * 0.15))
    n_test = max(1, round(n * 0.15))
    n_train = n - n_val - n_test
    if n_train < 2:
        n_val = n_test = 1
        n_train = n - 2
    return stems[:n_train], stems[n_train:n_train + n_val], stems[n_train + n_val:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=DEFAULT_CFG["epochs"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_dir = args.data_dir
    if not data_dir.is_dir():
        print(f"[ERR ] Directory not found: {data_dir}", flush=True)
        return 1

    report = RunReport()
    print("[i] Scanning and recognizing input files...", flush=True)
    anchors = discover_and_classify(data_dir, report)
    if len(anchors) < 3:
        print(f"[ERR ] Only {len(anchors)} usable graph(s) found; need at least 3 to form a "
              f"train/validation split. Add more .pt graphs (or COMSOL .mph + mesh pairs) and "
              f"try again.", flush=True)
        return 1

    print(f"[i] Building feature cache for {len(anchors)} accepted anchor(s)...", flush=True)
    cache = build_cache(anchors, report)
    if len(cache) < 3:
        print(f"[ERR ] Only {len(cache)} anchor(s) survived feature extraction; need at least "
              f"3.", flush=True)
        return 1

    train_a, val_a, test_a = split_anchors(list(cache.keys()), seed=args.seed)
    print(f"[i] Split: {len(train_a)} train / {len(val_a)} val / {len(test_a)} test", flush=True)
    print(f"    train: {', '.join(train_a)}", flush=True)
    print(f"    val:   {', '.join(val_a)}", flush=True)
    if test_a:
        print(f"    test:  {', '.join(test_a)}", flush=True)

    cfg = dict(DEFAULT_CFG, epochs=int(args.epochs))
    dev_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Training on {dev_t} for {cfg['epochs']} epochs "
          f"(this can take a while on CPU)...", flush=True)
    t0 = time.time()
    predict = train_one(train_a, cache, SimpleNamespace(**cfg, seeds=1), dev_t, seed=args.seed)
    print(f"[i] Training done in {time.time() - t0:.0f}s", flush=True)

    bench = Bench(cache, train_a, val_a + test_a)
    held = val_a + test_a
    sc = {a: predict(a) for a in train_a + held}
    th = pick_readout(bench, sc, train_a, GRID)
    scores: dict[str, dict] = {}
    for a in held:
        row = bench.row(a, apply_readout(cache[a], sc[a], th))
        scores[a] = row
        which = "val" if a in val_a else "test"
        off = ("%.4f" % row["off"]) if row["off"] == row["off"] else "n/a"
        print(f"[i] [{which}] {a:<24} wall {row['wall']:.4f} off {off}", flush=True)

    def _mean(key: str, anchors: list[str]) -> float | None:
        vals = [scores[a][key] for a in anchors if a in scores and scores[a][key] == scores[a][key]]
        return float(np.mean(vals)) if vals else None

    val_summary = dict(n=len(val_a), wall=_mean("wall", val_a), off=_mean("off", val_a))
    test_summary = dict(n=len(test_a), wall=_mean("wall", test_a), off=_mean("off", test_a))
    print(f"[i] VAL  mean wall={val_summary['wall']} off={val_summary['off']} (n={val_summary['n']})",
          flush=True)
    if test_a:
        print(f"[i] TEST mean wall={test_summary['wall']} off={test_summary['off']} "
              f"(n={test_summary['n']})", flush=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = CANDIDATE_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    model = predict.model
    mu, sd = predict.norm
    cols = [str(c) for c in cache[train_a[0]]["cols"]]
    torch.save(dict(
        state_dict=model.state_dict(), cfg=cfg, seed=args.seed,
        in_dim=model.enc[0].in_features - model.extra_dim, extra_dim=model.extra_dim,
    ), out_dir / "candidate.pth")
    np.savez_compressed(out_dir / "feature_norm.npz", mu=mu, sd=sd, cols=np.array(cols))

    manifest = dict(
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        source_dir=str(data_dir), cfg=cfg,
        files=[dict(name=v.path.name, kind=v.kind, accepted=v.accepted, stem=v.stem,
                     reason=v.reason) for v in report.verdicts],
        train_anchors=train_a, val_anchors=val_a, test_anchors=test_a,
        val_scores=val_summary, test_scores=test_summary,
        status="candidate_not_promoted",
        note=("This is a quick single-configuration candidate, not the shipped 9-member "
              "ensemble. It does NOT replace the live model. A researcher should review these "
              "scores and, if promising, run scripts/promote_clot_gnn_v4.py (or the matching "
              "promote script) with --repoint to ship a new locked checkpoint. The validated "
              "input graphs were also copied into data/processed/graphs_biochem_anchors/ so a "
              "full ensemble run can include them."),
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ok  ] Candidate saved to {out_dir}", flush=True)
    print("[done] This is a CANDIDATE, not a live model update. Ask a researcher to review the "
          "scores above and run the promotion step before it ships.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
