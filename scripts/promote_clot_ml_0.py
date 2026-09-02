"""Lock ``clot_ml_0`` -- one artifact for wounded and non-wounded vessels.

This is a composition, not a retrain (docs/WOUND_PROGRESS.md 18.3):

    wall              C0 GNN ensemble (the ``--base`` wound-capable artifact)
    wound             two-regime constants already on that base; no-op without a mask
    off-wall (wound)  chemistry ODE + solid-anchored replace+depth (REPLACE, not union)
    off-wall (else)   the GNN's own readout
    003-like AP       upwind renewal + da_scale_auto=123; optional residual GNN hook

On a pack with no wound it returns the base GNN unchanged -- asserted here, not argued.

    python scripts/promote_clot_ml_0.py
    python scripts/promote_clot_ml_0.py --base clot_gnn_v5w --repoint
    python scripts/eval_clot_ml_0.py          # compare against --baseline (v5w)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.locked import predict_temporal_v4_wound  # noqa: E402
from src.clot_ml.v0 import (  # noqa: E402
    DEFAULT_NAME, KIND, REPLACE_SCOPES, ClotMlV0Config, load_v0_bundle, predict_clot_ml_0,
)
from src.clot_ml.wound import has_wound, wound_mask  # noqa: E402

LOCKED = REPO / "outputs/clot_ml/locked"
POINTER = REPO / "data/reference/clot_gnn_locked.json"
GRAPHS = REPO / "data/processed/graphs_biochem_anchors"
WOUND_STEMS = ("wound_patient001", "wound_patient002", "wound_patient003")
NOWOUND_CHECK = ("patient012", "patient020", "patient044")
DEFAULT_BASE = "clot_gnn_v5w"


def _assert_noop_without_wound(bundle: dict, base_name: str) -> list[str]:
    checked = []
    for stem in NOWOUND_CHECK:
        p = GRAPHS / f"{stem}.pt"
        if not p.exists():
            print(f"    [skip] {stem} not on disk")
            continue
        data = torch.load(p, map_location="cpu", weights_only=False)
        if has_wound(data):
            raise SystemExit(f"[ERR] {stem} unexpectedly carries a wound mask")
        T = int(data.y.shape[0])
        times = sorted({0, T // 3, 2 * T // 3, T - 1})
        a = predict_temporal_v4_wound(bundle["base"], data, times, flow="gt")
        b = predict_clot_ml_0(bundle, data, times, flow="gt")
        if not np.array_equal(a["mask"], b["mask"]):
            raise SystemExit(f"[ERR] {stem}: v0 changed the no-wound MASK")
        if not np.array_equal(np.asarray(a["onset"]), np.asarray(b["onset"])):
            raise SystemExit(f"[ERR] {stem}: v0 changed the no-wound ONSET")
        for ti in times:
            if not np.array_equal(a["series"][int(ti)], b["series"][int(ti)]):
                raise SystemExit(f"[ERR] {stem}: v0 changed the no-wound SERIES at t={ti}")
        print(f"    [OK] {stem}: bit-identical to {base_name}")
        checked.append(stem)
    if not checked:
        raise SystemExit("[ERR] no no-wound pack available to verify the no-op property")
    return checked


def _assert_wound_is_covered(bundle: dict) -> dict:
    out = {}
    for stem in WOUND_STEMS:
        p = GRAPHS / f"{stem}.pt"
        if not p.exists():
            print(f"    [skip] {stem} not on disk")
            continue
        data = torch.load(p, map_location="cpu", weights_only=False)
        T = int(data.y.shape[0])
        w = wound_mask(data)
        pred = predict_clot_ml_0(bundle, data, [T - 1], flow="gt")
        cov = float(pred["mask"][w].mean()) if w.any() else float("nan")
        if cov < 0.99:
            raise SystemExit(f"[ERR] {stem}: v0 commits only {cov:.1%} of the wound")
        print(f"    [OK] {stem}: wound coverage {cov:.1%}")
        out[stem] = dict(wound_coverage=cov)
    if not out:
        raise SystemExit("[ERR] no wound pack available to verify coverage")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE,
                    help="wound-capable GNN this wraps (default: %(default)s)")
    ap.add_argument("--name", default=DEFAULT_NAME)
    ap.add_argument("--ap-residual", default="",
                    help="optional v7 AP residual checkpoint (relative to repo root)")
    ap.add_argument("--replace-scope", choices=REPLACE_SCOPES, default="all_lumen",
                    help="chemistry replacement extent on wound packs (default: %(default)s)")
    ap.add_argument("--repoint", action="store_true",
                    help="point data/reference/clot_gnn_locked.json at this artifact")
    args = ap.parse_args()

    base_root = LOCKED / args.base
    if not (base_root / "manifest.json").exists():
        raise SystemExit(f"[ERR] base artifact {args.base} not found under {LOCKED}")

    cfg = ClotMlV0Config(
        base_model=args.base,
        ap_residual=(args.ap_residual or None),
        replace_scope=args.replace_scope,
    )
    if cfg.ap_residual:
        rp = REPO / cfg.ap_residual
        if not rp.exists():
            raise SystemExit(f"[ERR] AP residual checkpoint not found: {rp}")

    dst = LOCKED / args.name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    manifest = dict(
        name=args.name,
        kind=KIND,
        base_model=args.base,
        promoted_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        description=(
            "Unified clot-ML stack for wounded and non-wounded vessels. Wall SET and "
            "non-wound off-wall stay the C0 GNN; on a wound pack the configured lumen scope is REPLACED "
            "by chemistry-ODE Mat through solid-anchored replace+depth (att=0.23, "
            "depth=3, da_scale_auto=123, upwind AP renewal). Bit-identical to the base "
            "on any pack without a wound mask. Optional AP residual GNN is a hook for "
            "future 003-like vessels (docs/WOUND_PROGRESS.md 18.3)."
        ),
        docs="docs/WOUND_PROGRESS.md",
        supersedes=args.base,
        v0=cfg.to_manifest_block(),
    )
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[i] wrote {dst / 'manifest.json'}")
    print(f"    base={cfg.base_model}  att={cfg.replace_att}  depth={cfg.replace_depth}  "
          f"scope={cfg.replace_scope}  da_auto={cfg.da_scale_auto}  "
          f"ap_renewal={cfg.ap_renewal_scale}")

    print("[i] loading bundle for promotion gates ...")
    bundle = load_v0_bundle(args.name)
    print("[i] no-wound no-op gate")
    _assert_noop_without_wound(bundle, args.base)
    print("[i] wound coverage gate")
    _assert_wound_is_covered(bundle)

    if args.repoint:
        ptr = json.loads(POINTER.read_text()) if POINTER.exists() else {}
        ptr.update(
            name=args.name,
            kind=KIND,
            path=f"outputs/clot_ml/locked/{args.name}",
            manifest=f"outputs/clot_ml/locked/{args.name}/manifest.json",
            promoted_at=manifest["promoted_at"],
            docs="docs/WOUND_PROGRESS.md",
            supersedes=args.base,
        )
        POINTER.write_text(json.dumps(ptr, indent=2), encoding="utf-8")
        print(f"[OK] repointed {POINTER} -> {args.name}")
    else:
        print("[i] pointer left unchanged (pass --repoint to switch the default)")
    print(f"[OK] Promoted {args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
