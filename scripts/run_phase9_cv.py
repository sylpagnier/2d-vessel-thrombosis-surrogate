"""Geometry-stratified K-fold over the whole eligible non-SEALED pool.

Replaces the confounded FIT/DEV cut (see `src/clot_ml/geometry_splits.py`).  Saves, for
every fold, that fold's model's score on every vessel, so readouts and metrics can be
re-evaluated later without retraining.

    python scripts/run_phase9_cv.py --tag cv5 --folds 5 --seeds 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.clot_ml.recipe import CUSTOMER_RETRAIN_EPOCHS, PROMOTION_EPOCHS, recipe
from src.utils.paths import anchor_packs_dir, get_project_root

REPO = get_project_root()

from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.geometry_splits import classes_for, describe, stratified_folds  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.wall_cohort_splits import CLOT_FREE, SEALED  # noqa: E402
from scripts.train_clot_gnn import train_one  # noqa: E402

OUT = REPO / "outputs/phase9_scores"
PACKS = anchor_packs_dir()
# Objective shared with promotion lives in src/clot_ml/recipe.py -- including
# `clot_free_w=1.0` (a clot-free vessel's whole-loss gradient weight) and
# `empty_gt_loss="none"` (what a clot-free vessel contributes to the metric term;
# "score" has no measurable effect either way, MODEL_REVIEW 8f.4, so "none" wins on
# parsimony).  Only the CV-specific deltas are spelled out below.
BASE = recipe(
    epochs=PROMOTION_EPOCHS, rounds=3, adv_fb=0, off_only=0, loss_shape_w=0.5,
    burden_t=0.89, burden_tau=0.02,
    # C0 / MODEL_REVIEW 3.4.  `burden_agg` weights the TAIL of the per-vessel burden
    # error (l1 = the measured-null 2026-08-22 form, sq/cvar are the retry);
    # `shape_w` constrains the field's SPREAD toward a running cohort reference,
    # leaving burden free.  Both 0/"l1" reproduce the shipped objective exactly.
    burden_agg="l1", burden_cvar_q=0.5, burden_t_off=0.0,
)


def _save_fold_member(root: Path, *, tag: str, fold: int, held: list[str],
                      train: list[str], pool: list[str], cols: list[str], cfg: dict,
                      seed: int, predict) -> None:
    """Persist a CV member solely for reproducible OOF visualization.

    The existing ``phase9_scores`` files preserve OOF score fields but not the model that
    generated them.  This sidecar records the member, fold normalizer, and exact exclusion
    set.  It is intentionally not a promoted deployment artifact.
    """
    from src.clot_ml.gnn import ClotGNN

    fold_dir = root / tag / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    mu, sd = predict.norm
    norm_path = fold_dir / "feature_norm.npz"
    if norm_path.exists():
        old = np.load(norm_path, allow_pickle=True)
        if (not np.array_equal(old["mu"], mu)
                or not np.array_equal(old["sd"], sd)
                or [str(x) for x in old["cols"]] != cols):
            raise RuntimeError(
                f"{fold_dir}: fold normalizer disagrees with the saved member; "
                "choose a new --save-fold-models directory")
    else:
        np.savez_compressed(norm_path, mu=mu, sd=sd, cols=np.asarray(cols))

    model = predict.model
    assert isinstance(model, ClotGNN), type(model)
    member_path = fold_dir / f"member_s{seed}.pth"
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(dict(
        state_dict=state,
        cfg=dict(cfg),
        seed=int(seed),
        in_dim=int(model.enc[0].in_features - model.extra_dim),
        extra_dim=int(model.extra_dim),
        held_out=list(held),
        train_anchors=list(train),
    ), member_path)

    manifest_path = fold_dir / "manifest.json"
    manifest = dict(
        schema_version=1,
        purpose="outer-fold checkpoint for out-of-fold visualization; never deploy",
        tag=tag,
        fold=int(fold),
        held_out=list(held),
        train_anchors=list(train),
        cv_pool=list(pool),
        feature_norm="feature_norm.npz",
        members=[],
    )
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for key, expected in (("tag", tag), ("fold", int(fold)), ("held_out", list(held)),
                              ("train_anchors", list(train)), ("cv_pool", list(pool))):
            if manifest.get(key) != expected:
                raise RuntimeError(
                    f"{fold_dir}: existing OOF manifest has incompatible {key}; "
                    "choose a new --save-fold-models directory")
    members = [m for m in manifest.get("members", []) if int(m["seed"]) != int(seed)]
    members.append(dict(file=member_path.name, seed=int(seed), cfg=dict(cfg)))
    manifest["members"] = sorted(members, key=lambda member: int(member["seed"]))
    manifest_path.write_text(json.dumps(manifest, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    for k, v in BASE.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="first ensemble seed (default 0, i.e. seeds 0..seeds-1).  Set it to "
                         "`--seeds` to produce an INDEPENDENT replicate of the same arm: same "
                         "cache, same folds, disjoint seeds.  Pairing two such runs measures "
                         "the pipeline's own noise, which is the null every flow-source "
                         "comparison is read against and which nothing had ever quantified.")
    # which feature cache to read: "gt" is the v3 55-channel one, "v4" adds the advective
    # transport + indicator-gate channels (scripts/build_clot_ml_cache_v4.py)
    ap.add_argument("--cache", default="gt")
    ap.add_argument("--pool", default="all", choices=["all", "carrying"],
                    help="'all' (default) trains on every vessel in the cache, clot-free "
                         "included; 'carrying' drops the empty-GT vessels, which is the "
                         "A/B arm for whether they help -- see "
                         "python -m src.tools.diagnostics clot-free-headroom")
    ap.add_argument("--save-fold-models", default="",
                    help=("directory for outer-fold member checkpoints and normalizers. "
                          "Use a shared root across the separate v5a/v5b/v5c C0 runs; "
                          "each tag receives its own subdirectory. These are OOF-viz "
                          "assets, never a promoted deployment artifact."))
    args = ap.parse_args()
    cfg = SimpleNamespace(**{k: getattr(args, k) for k in BASE}, seeds=1)
    # advective recurrence (src/clot_ml/recurrent.feedback_channels_advective)
    cfg.adv_fb = bool(cfg.adv_fb)
    cfg.off_only = bool(cfg.off_only)   # off-wall specialist (train_clot_gnn.train_one)

    dev_t = require_cuda_device()
    cache = attach_physics(load_cache(args.cache))
    # SEALED IS NEVER IN THE POOL, whatever the cache holds.  The cache builders gained an
    # `--include-sealed` flag on 2026-09-02 so the final read has features to run on; caching
    # a sealed vessel is not spending it, but training or selecting on one is, and the pool
    # used to be "whatever is in the directory".  Dropped here rather than at the call site so
    # no launcher can forget (docs/SEALED_SPLIT.md).
    dropped = sorted(a for a in cache if a in SEALED)
    if dropped:
        print("[i] SEALED excluded from the CV pool: %s" % ", ".join(dropped), flush=True)
    pool = [a for a in cache if a not in SEALED]
    if args.pool == "carrying":
        pool = [a for a in pool if a not in CLOT_FREE]
    classes = classes_for(pool, PACKS)
    pool = [a for a in pool if a in classes]
    folds = stratified_folds({a: classes[a] for a in pool}, k=args.folds)
    # Clot-free vessels are IN the pool as of 2026-08-22: they train (their metric term is
    # the false-positive branch, `softmetric.soft_empty_gt_score`) and they are held out once
    # each, so their false-positive number is out-of-fold like everybody else's.  They are
    # excluded from recall-bearing means at REPORTING time, in `eval_strict.py`, not here.
    n_free = len([a for a in pool if a in CLOT_FREE])
    cols = [str(c) for c in cache[pool[0]]["cols"]]
    print("[i] pool n=%d (%d clot-carrying, %d clot-free), %d folds\n%s"
          % (len(pool), len(pool) - n_free, n_free, len(folds), describe(classes, folds)),
          flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    store, t0 = {}, time.time()
    for k, held in enumerate(folds):
        tr = [a for a in pool if a not in held]
        acc, accr = {}, {}
        for s in range(args.seed_offset, args.seed_offset + args.seeds):
            predict = train_one(tr, cache, cfg, dev_t, seed=s)
            if args.save_fold_models:
                _save_fold_member(
                    Path(args.save_fold_models), tag=args.tag, fold=k, held=held, train=tr,
                    pool=pool, cols=cols, cfg=vars(cfg), seed=s, predict=predict)
            for a in pool:
                acc[a] = acc.get(a, 0.0) + predict(a) / args.seeds
                accr[a] = accr.get(a, 0.0) + predict.reg(a) / args.seeds
        for a in pool:
            store["%d|%s" % (k, a)] = acc[a].astype(np.float32)
            # the regression head's log1p(Mat/crit); see train_clot_gnn.predict_reg
            store["reg|%d|%s" % (k, a)] = accr[a].astype(np.float32)
        store["held|%d" % k] = np.array(held)
        print("   fold %d/%d held=%s (%.0fs)"
              % (k + 1, len(folds), ",".join(x[-3:] for x in held), time.time() - t0),
              flush=True)
    store["pool"] = np.array(pool)
    store["classes"] = np.array([classes[a] for a in pool])
    store["cfg"] = np.array([repr(vars(cfg))])
    np.savez_compressed(OUT / f"{args.tag}.npz", **store)
    print("wrote %s (%.0fs)" % (OUT / f"{args.tag}.npz", time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
