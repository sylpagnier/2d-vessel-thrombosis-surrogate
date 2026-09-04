"""Precache RGP-DEQ t=0 flow onto biochem graph packs.

    python scripts/precache_rgp_deq.py --only comsol040,comsol041,comsol044,comsol012
    python scripts/precache_rgp_deq.py --only comsol040 --force

**Prior source (RGP_DEQ_REPAIR_PLAN.md B1).**  This script used to call the DEQ on whatever
``x[:, 11:14]`` the pack happened to carry.  Every biochem pack carries a CFD-derived prior
block (rel-L2 0.012-0.049 against COMSOL's own ``t=0``; ``comsol002`` is bit-identical), so
the "deployable" ``u0_pred`` consumed by ``clot_ml.features`` and ``physics_wall_model`` was
produced by handing the flow surrogate the field it exists to predict.  The default is now
``analytic`` -- the only source legal under the s17 Z2 contract -- and the choice is stamped
onto the pack so a stale cache cannot be mistaken for a fresh one (B11).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Repo root by marker, not by depth: this file may move between
# scripts/ and scripts/<subdir>/ without silently resolving one level off.
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import PredChannels  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.data_gen.lib.legal_priors import (  # noqa: E402
    PRIOR_SOURCES,
    apply_prior_source,
    assert_train_deploy_prior_parity,
    resolve_prior_source,
)
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    predict_kinematics_and_latent,
    resolve_kinematics_checkpoint,
)


def _ckpt_fingerprint(path: Path) -> dict:
    """Identity of the weights that produced a cache, so a stale one is detectable."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    meta = {}
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict):
            meta = {
                k: raw.get(k)
                for k in ("checkpoint_role", "best_epoch", "rel_l2", "composite", "run_id",
                          "prior_source")
                if k in raw
            }
    except Exception:
        pass
    return {"path": str(path), "md5": h.hexdigest(), **meta}


def _cache_is_stale(data, provenance: dict) -> str | None:
    """``None`` when the pack's cached ``u0_pred`` matches this run; else why it does not."""
    raw = getattr(data, "u0_pred_provenance", None)
    if not raw:
        return "no provenance stamp (predates B11)"
    try:
        old = json.loads(raw)
    except (TypeError, ValueError):
        return "unreadable provenance stamp"
    if old.get("prior_source") != provenance["prior_source"]:
        return "prior_source %r != %r" % (old.get("prior_source"), provenance["prior_source"])
    old_md5 = (old.get("checkpoint") or {}).get("md5")
    if old_md5 != provenance["checkpoint"]["md5"]:
        return "checkpoint %s != %s" % (str(old_md5)[:12], provenance["checkpoint"]["md5"][:12])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attach RGP-DEQ u0_pred/v0_pred/z_kin_pred onto graph packs.")
    parser.add_argument("--graph-dir", default="data/processed/graphs_biochem_anchors")
    parser.add_argument("--only", default="",
                        help="comma-separated anchors. Default: every *.pt in graph-dir.")
    parser.add_argument("--cohort", default="", choices=["", "fitdev"],
                        help="fitdev = FIT+DEV wall-cohort anchors (still skips missing files)")
    parser.add_argument("--force", action="store_true",
                        help="recompute even when u0_pred is already attached")
    parser.add_argument(
        "--prior-source", default="", choices=("",) + PRIOR_SOURCES,
        help="prior block handed to the DEQ. Default: runtime config, else 'analytic'. "
             "'stored' is the leaked CFD field and is ILLEGAL under the s17 Z2 deploy contract.")
    parser.add_argument("--checkpoint", default="",
                        help="RGP-DEQ checkpoint; defaults to the resolved promoted one.")
    args = parser.parse_args()

    prior_source = (args.prior_source or resolve_prior_source(default="analytic")).strip().lower()

    graph_dir = Path(args.graph_dir)
    if not graph_dir.exists():
        print("[ERR] no graph dir at %s" % graph_dir)
        return 1

    if args.only.strip():
        pt_files = [graph_dir / f"{a.strip()}.pt" for a in args.only.split(",") if a.strip()]
    elif args.cohort == "fitdev":
        from src.core_physics.wall_cohort_splits import DEV, FIT
        pt_files = [graph_dir / f"{a}.pt" for a in list(FIT) + list(DEV)]
    else:
        pt_files = sorted(graph_dir.glob("*.pt"))
    if not pt_files:
        print("[WARN] no packs to cache")
        return 1

    device = require_cuda_device()
    ckpt_path = resolve_kinematics_checkpoint(args.checkpoint or None)
    fingerprint = _ckpt_fingerprint(ckpt_path)
    print("[i] kinematics ckpt %s (md5 %s)" % (ckpt_path, fingerprint["md5"][:12]))
    print("[i] prior_source = %s%s" % (
        prior_source,
        "   *** LEAKED CFD PRIORS -- NOT DEPLOYABLE (s17 Z2) ***" if prior_source == "stored" else "",
    ))
    # If the checkpoint records what it was trained with, refuse a silent train/deploy mismatch.
    trained_with = fingerprint.get("prior_source")
    if trained_with:
        assert_train_deploy_prior_parity(str(trained_with), prior_source)
    else:
        print("[warn] checkpoint records no prior_source; train/deploy parity NOT verified")

    kine = load_kinematics_predictor(ckpt_path, device)
    kine.eval()
    provenance = {
        "prior_source": prior_source,
        "checkpoint": fingerprint,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    n_ok = n_skip = 0
    for file_path in pt_files:
        if not file_path.exists():
            print("[ERR] missing %s" % file_path.name)
            continue
        anchor = file_path.stem
        data = torch.load(file_path, map_location="cpu", weights_only=False)
        if (not args.force) and getattr(data, "u0_pred", None) is not None:
            # "Has a cache" is not the same as "has the RIGHT cache".  The default prior source
            # is now `analytic` while every pre-existing cache was built with `stored`, so a
            # presence-only check would skip all 52 packs and silently keep the leaked field.
            # Compare the provenance stamp instead (B11); a pack with no stamp predates it and
            # cannot be trusted to match.
            stale = _cache_is_stale(data, provenance)
            if stale is None:
                print("[skip] %s already has u0_pred (provenance matches)" % anchor)
                n_skip += 1
                continue
            print("[stale] %s: %s -- recomputing" % (anchor, stale))
        print("[%s] RGP-DEQ t=0 (priors=%s) ..." % (anchor, prior_source))
        # Rewrite the prior block BEFORE the solve: the DEQ consumes UV_PRIOR/MU_PRIOR both in
        # its encoder and in the hard BC `u = uv_prior + sdf * uvp`, so applying this after the
        # solve would leave every output conditioned on the leaked field.
        data_cuda = apply_prior_source(data, prior_source).to(device)
        with torch.no_grad():
            pred, z_kin = predict_kinematics_and_latent(kine, data_cuda)
        u0 = pred[:, PredChannels.U].contiguous()
        v0 = pred[:, PredChannels.V].contiguous()
        data.u0_pred = u0.detach().cpu()
        data.v0_pred = v0.detach().cpu()
        data.z_kin_pred = z_kin.detach().cpu()
        # Direct shear head (nd), converted to 1/s.  Cached for a later head; t0_flow_fields
        # still MLS-differentiates u0_pred (PHASE7_FINDINGS 10.7).
        if pred.shape[1] > PredChannels.SHEAR_RATE:
            u_ref = float(data.u_ref.reshape(-1)[0])
            d_bar = float(data.d_bar.reshape(-1)[0])
            sr_nd = pred[:, PredChannels.SHEAR_RATE].reshape(-1).clamp(min=0)
            data.sr0_pred = (sr_nd * (u_ref / max(d_bar, 1e-12))).detach().cpu()
        rel = ""
        if getattr(data, "y", None) is not None:
            u = data.y[0, :, 0].detach().cpu().numpy()
            v = data.y[0, :, 1].detach().cpu().numpy()
            up = data.u0_pred.reshape(-1).numpy()
            vp = data.v0_pred.reshape(-1).numpy()
            den = float((u * u + v * v).mean() ** 0.5) + 1e-12
            rel_l2 = float((((up - u) ** 2 + (vp - v) ** 2).mean() ** 0.5) / den)
            rel = "  RelL2=%.3f" % rel_l2
            data.u0_pred_rel_l2 = float(rel_l2)
        # B11: without this, a cache built from the wrong priors, the wrong checkpoint, or
        # before a pack repair is indistinguishable from a fresh one.  DEPLOY_FLOW_PLAN s1b
        # records exactly that failure (a cache that predated `repair_pack_wall_normals`).
        data.u0_pred_provenance = json.dumps(provenance)
        # Atomic: a pack is ~335 MB and holds the COMSOL solution, which cannot be
        # regenerated from this repo.  Writing in place means an interrupted --force run
        # (Ctrl-C, OOM, power) leaves a truncated file where the ground truth used to be.
        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        torch.save(data, tmp_path)
        tmp_path.replace(file_path)
        print("[OK] %s  n=%d%s  (saved u,v,sr0_pred)" % (anchor, int(data.num_nodes), rel))
        n_ok += 1
    print("[i] wrote %d  skipped %d" % (n_ok, n_skip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
