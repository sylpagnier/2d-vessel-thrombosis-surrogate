"""Train and lock `clot_gnn_v4` -- the advective-transport ensemble, strict-protocol validated.

v4 differs from `clot_gnn_v2`/`v3` in exactly one thing: the feature block.  It adds the 13
channels of `src/clot_ml/features_v4.py` -- COMSOL's own advection operator solved on the
mesh (`src/clot_ml/transport.py`), plus the indicator-gate physics variant.  Architecture,
configurations, seeds and readout family are unchanged, so the comparison in
`docs/PHASE10_V4.md` is a clean feature ablation.

Validated strictly-nested (every readout scalar selected on out-of-fold scores of vessels
outside the held-out fold -- `scripts/eval_strict.py`, `scripts/eval_strict_temporal.py`):

                  mean wall   mean off   FIN wall   FIN off
    v3 (cv5a,b,c)    0.8687     0.6389     0.9014     0.7011
    v4 (v5a,b,c)     0.8750     0.7188     0.9176     0.7366

Read `docs/PHASE10_V4.md` 2 before quoting any of it: the cohort noise floor is +-0.024 wall
and +-0.091 off-wall, so what supports v4 is the direction being consistent on all four
metrics and both domains, not the size of any one of them.

As with v2/v3 the SHIPPED weights train on the whole 19-vessel eligible pool, so there is no
held-out vessel left to score them against; the manifest carries the out-of-fold CV numbers
that SELECTED this design, which is the honest generalisation estimate.  SEALED never seen.

    python scripts/promote_clot_gnn_v4.py --name clot_gnn_v4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.clot_ml.recipe import CUSTOMER_RETRAIN_EPOCHS, PROMOTION_EPOCHS, recipe
from src.utils.paths import anchor_packs_dir, clot_ml_locked_dir, get_project_root

REPO = get_project_root()
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.features_v4 import V4_CHANNELS  # noqa: E402
from src.clot_ml.geometry_splits import classes_for, eligible_pool, is_priority  # noqa: E402
from src.core_physics.wall_cohort_splits import CLOT_FREE, SEALED  # noqa: E402
from src.clot_ml.gnn import ClotGNN  # noqa: E402
from train_clot_gnn import train_one  # noqa: E402

PACKS = anchor_packs_dir()

# The three configurations behind the v5a / v5b / v5c CV tags, 3 seeds each.
# `empty_gt_loss` and `shape_w` are C0 and must match run_phase9_cv; they now do
# so structurally, from src/clot_ml/recipe.py, instead of by comment.
#
# `clot_free_w` is deliberately NOT carried into BASE here even though the shared
# recipe defines it.  BASE is hashed into the artifact manifest fingerprint (see
# the `fingerprint=` argument below), so adding a key -- even at its existing
# effective value -- would change the identity of every newly promoted artifact
# and stop it comparing against the ones already locked.  The per-member cfg sets
# `clot_free_w` explicitly from the CLI, so training is unaffected either way.
# NOTE: that the manifest fingerprint omits a value the members were trained with
# is a pre-existing inconsistency, left alone here rather than changed in passing.
BASE = {k: v for k, v in recipe(
    epochs=PROMOTION_EPOCHS,
    burden_t=0.89, burden_tau=0.02, burden_agg="l1",
    burden_cvar_q=0.5, burden_t_off=0.0,
).items() if k != "clot_free_w"}
MEMBERS = {
    "v5a": dict(rounds=3, seeds=3),
    "v5b": dict(rounds=5, seeds=3),
    "v5c": dict(rounds=3, seeds=3, off_mult=2.5),
}

#: Strictly-nested, out-of-fold, severity metric (docs/PHASE10_V4.md).  NOT from these
#: weights -- these train on the whole pool.
STRICT_CV = dict(
    protocol=("geometry-stratified 5-fold; every readout scalar selected on the OUT-OF-FOLD "
              "scores of vessels outside the held-out fold (scripts/eval_strict.py)"),
    cohort=("23 clot-carrying vessels (VIZ_HALF released 2026-08-22) + 8 clot-free scored "
            "on the false-positive branch only; SEALED 007/013/031/043 never seen"),
    v4=dict(final=dict(all=dict(wall=0.9203, off=0.7078),
                       baseline=dict(wall=0.9246, off=0.6631),
                       priority=dict(wall=0.8997, off=0.8419),
                       clot_free_fp_only=dict(wall=1.0000, off=1.0000, nodes_committed=0)),
            mean_over_time=dict(all=dict(wall=0.8694, off=0.5792),
                                baseline=dict(wall=0.8695, off=0.5029),
                                priority=dict(wall=0.8690, off=0.8081)),
            frozen_same_set=dict(all=dict(wall=0.7885, off=0.4124)),
            oracle_timing_same_set=dict(all=dict(wall=0.9673, off=0.8219)),
            per_vessel_oracle_cut=dict(all=dict(wall=0.9377, off=0.7526))),
    without_c0=dict(
        final=dict(all=dict(wall=0.9008, off=0.5812)),
        mean_over_time=dict(all=dict(wall=0.8716, off=0.5713)),
        per_vessel_oracle_cut=dict(all=dict(wall=0.9312, off=0.7746)),
        note=("the same cohort and features WITHOUT the C0 distributional constraint "
              "(shape_w=0; tags v5a/v5b/v5c).  Paired per configuration, off-wall "
              "+0.0854 P=0.031, +0.1059 P=0.002, +0.1649 P=0.000 -- three of three "
              "positive.  The WALL gain (+0.0195 at ensemble) is INSIDE the +/-0.024 "
              "floor and is NOT claimed.  Mean-over-time is unchanged, so C0 is a "
              "FINAL-TIME result.  MODEL_REVIEW_2026-08-22 9b.")),
    physics_backbone=dict(final=dict(all=dict(wall=0.8832, off=0.3999))),
    superseded_pre_repair_n19=dict(
        final=dict(all=dict(wall=0.9176, off=0.7366)),
        mean_over_time=dict(all=dict(wall=0.8750, off=0.7188)),
        note=("docs/PHASE10_V4.md.  Measured on 19 vessels before the pack repair; NOT "
              "comparable.  The off-wall difference is a READOUT gap, not a model one -- "
              "out-of-fold AUC is unchanged at 0.989 and the per-vessel oracle cut is within "
              "the noise floor.  See MODEL_REVIEW_2026-08-22 8f.")),
    noise_floor=dict(wall=0.024, off=0.074,
                     note=("config spread of one arm, RE-MEASURED on this cohort "
                           "(scripts/eval_significance.py --cache v5). Per-vessel spread is "
                           "far larger: median 0.042 wall / 0.112 off, max 0.216 / 0.628. A "
                           "cohort-mean difference below the spread is not a result.")),
    readout_gap=dict(
        off_cohort_cut=0.7078, off_per_vessel_oracle=0.7526, gap=0.0448,
        gap_without_c0=0.1934, previous_gap=0.120,
        note=("Was the largest quantified loss in the project (0.193).  Closed to 0.045 by "
              "the C0 training-time distributional constraint (shape_w), NOT by a new "
              "readout: the per-vessel oracle did NOT rise (0.7746 -> 0.7526), the cohort "
              "cut did (0.5812 -> 0.7078).  The field became cuttable by one constant.  "
              "MODEL_REVIEW_2026-08-22 8f.2 and 9b.")))

#: The readout these scores are produced with.  It is NOT a plain threshold, and the choice
#: between the two arms is made per domain inside each fold
#: (scripts/eval_expected_score_readout.py).  In every fold it selects the same pair.
READOUT = dict(
    selection="per-domain, in-fold, over {cohort_cut, resid, resid_adapt, expected_tuned}",
    wall=("resid_adapt -- the physics-conditioned keep/add readout, with its four cuts "
          "PERTURBED by a fitted slope on the vessel's mean score.  b=0 reproduces the "
          "cohort readout exactly, so it can only move if the statistic pays."),
    off=("expected_tuned -- rank the nodes and commit the prefix that maximises the "
         "EXPECTED severity score, using the model's own p in place of GT, with two "
         "in-fold scalars (gamma sharpening, prefix scale) correcting for miscalibration. "
         "This is the fix for the low-burden precision problem: the budget adapts per "
         "vessel with no label.  Off-wall 0.7136 -> 0.7359 against a cohort cut."),
    temporal=("time-conditioned head, 4 seeds averaged, monotone in time, every node in the "
              "committed set clot at the last timestep (commit_final), off-wall never before "
              "its owner, and OFF-WALL ONSET from a learned per-node LAG behind the owner "
              "(--owner-lag --learn-lag --lag-anchor ode): the lag regression is fitted "
              "OUT-OF-FOLD on the inner split, gated on predicted off-wall burden, and "
              "anchored on the ODE's OWN owner crossing rather than the head's predicted "
              "one -- target and anchor must be the same object at train and apply time, "
              "and wall-onset error is worth 2.3x the lag error (docs/PHASE10_V4.md 15). "
              "Together worth +0.038 mean-over-time off-wall."),
    scripts=["scripts/eval_expected_score_readout.py", "scripts/eval_strict_temporal.py"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="clot_gnn_v4")
    ap.add_argument("--supersedes", default="",
                    help="the generation this replaces.  Was hardcoded to 'clot_gnn_v3', so "
                         "every later generation claimed to supersede v3.  Default: the "
                         "highest-numbered existing artifact below --name.")
    ap.add_argument("--cache", default="v5")
    ap.add_argument("--seeds", type=int, default=0,
                    help="override every member configuration's seed count.  0 (default) "
                         "keeps MEMBERS' own 3.  Recorded in the manifest, because a "
                         "different ensemble width is a different artifact.")
    ap.add_argument("--repoint", action="store_true",
                    help="move data/reference/clot_gnn_locked.json to this artifact")
    ap.add_argument("--empty-gt-loss", default=BASE["empty_gt_loss"],
                    choices=["none", "score"],
                    help="MUST match the objective the promoted CV tags were run under. The "
                         "strict-CV scores in the manifest describe a training procedure; "
                         "shipping weights fitted by a different one makes them a "
                         "description of some other model. Recorded in the manifest.")
    ap.add_argument("--shape-w", type=float, default=BASE["shape_w"],
                    help="C0 distributional constraint (MODEL_REVIEW 9b). MUST match the "
                         "promoted CV tags, for the same reason as --empty-gt-loss.")
    ap.add_argument("--include-sealed", action="store_true",
                    help="PRODUCTION ONLY. Train on the SEALED vessels too. This permanently "
                         "forfeits the held-out test for the resulting artifact: it can never "
                         "be scored on anything, and the manifest is stamped so. Use it for "
                         "the deployed product, never for a number that will be published.")
    ap.add_argument("--clot-free-w", type=float, default=BASE.get("clot_free_w", 1.0),
                    help="gradient weight on a clot-free vessel's node loss; 1.0 = shipped")
    ap.add_argument("--pool", default="all", choices=["all", "carrying"],
                    help="which vessels the SHIPPED weights train on.  Must match what "
                         "`run_phase9_cv.py --pool` used, or the artifact is a different "
                         "model from the one the strict CV measured.")
    args = ap.parse_args()

    dev_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = attach_physics(load_cache(args.cache))
    # THE TRAINING POOL IS THE CACHE, not `eligible_pool()`.  `run_phase9_cv.py` trains on
    # every vessel in the cache -- clot-free included since 2026-08-22 -- so promoting on
    # `eligible_pool()` (clot-carrying only) would ship weights fitted on a DIFFERENT set
    # from the one the strict CV measured, and `STRICT_CV` below would be describing another
    # model.  `eligible_pool()` is still the right answer for REPORTING, which is what it is
    # used for downstream.
    # SEALED never trains.  Same guard as `run_phase9_cv.py`: the cache may legitimately hold
    # those four vessels so the final read has features, and the shipped weights must still
    # never have seen them (docs/SEALED_SPLIT.md).
    # `--include-sealed` is the ONE way SEALED enters a training pool, and it is deliberate:
    # the deployed product should use every vessel that exists, and the SEALED read has
    # already been taken, so there is nothing left to protect on THAT artifact.  What must be
    # protected is the boundary between the two families -- see `metrics_invalid` below.
    pool = [a for a in sorted(cache)
            if (args.include_sealed or a not in SEALED)
            and (a not in CLOT_FREE or args.pool == "all")]
    if args.include_sealed:
        got = sorted(a for a in pool if a in SEALED)
        print("[!] PRODUCTION BUILD: SEALED vessels are IN the training pool: %s"
              % ", ".join(got) if got else "[!] --include-sealed set but none are cached",
              flush=True)
        print("    This artifact has NO valid held-out metric and must never be scored.",
              flush=True)
    classes = classes_for(pool, PACKS)
    pool = [a for a in pool if a in classes]
    carrying = [a for a in pool if a not in CLOT_FREE]
    free = [a for a in pool if a in CLOT_FREE]
    prio = [a for a in carrying if is_priority(classes[a])]
    cols = [str(c) for c in cache[pool[0]]["cols"]]
    missing = [c for c in V4_CHANNELS if c not in cols]
    if missing:
        raise SystemExit("cache %r is not a v4 cache; missing %s" % (args.cache, missing))
    print("[i] pool n=%d (%d clot-carrying, %d clot-free) priority=%d (%s), %d features"
          % (len(pool), len(carrying), len(free), len(prio), ", ".join(prio), len(cols)),
          flush=True)
    missing_cv = sorted(set(eligible_pool()) & set(cache) - set(pool))
    if missing_cv:
        print("[WARN] cached but not promoted on: %s" % missing_cv, flush=True)

    out = clot_ml_locked_dir() / args.name
    if not args.supersedes:
        sibs = sorted(q.name for q in out.parent.glob("clot_gnn_v*")
                      if q.is_dir() and q.name < args.name and not q.name.endswith("w"))
        args.supersedes = sibs[-1] if sibs else ""
    out.mkdir(parents=True, exist_ok=True)
    Xall = np.concatenate([cache[a]["X"] for a in pool])
    mu, sd = Xall.mean(0), Xall.std(0)
    sd[sd < 1e-6] = 1.0

    # WHAT THE RESUME IS ALLOWED TO REUSE.  Training nine members takes ~25 min and has to be
    # interruptible, so an existing file is reused -- but `os.path.exists` is not enough, and
    # was actively dangerous: `--name` defaults to `clot_gnn_v4`, so a re-promotion after the
    # 2026-08-22 pack repair found the PREVIOUS generation's weights, kept all nine, and
    # rewrote the manifest -- new scores, superseded weights, no warning.  `in_dim` does not
    # catch it either, because the channel count did not change.
    #
    # The fingerprint covers everything a member's weights depend on: which vessels, which
    # feature columns, the training config, and `mu`/`sd` -- which are computed from the whole
    # pool's `X`, so they move whenever any feature value moves.  A member without one predates
    # this check and is never reused.
    def _fingerprint(cfg: dict) -> str:
        h = hashlib.sha1()
        h.update(repr(sorted(pool)).encode())
        h.update(repr(list(cols)).encode())
        h.update(repr(sorted(cfg.items())).encode())
        h.update(np.ascontiguousarray(mu, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(sd, dtype=np.float64).tobytes())
        return h.hexdigest()

    members, t0 = [], time.time()
    for cname, over in MEMBERS.items():
        cfg = dict(BASE)
        cfg["empty_gt_loss"] = args.empty_gt_loss
        cfg["shape_w"] = float(args.shape_w)
        # MUST be threaded explicitly: `cfg` starts from BASE, so a CLI flag that is parsed
        # but not copied here is silently ignored -- the run trains at the BASE value while
        # the manifest records the flag.  That would have shipped a clot_free_w=1.0 model
        # labelled 0.25 (DEPLOYCLOT.md 25).
        cfg["clot_free_w"] = float(args.clot_free_w)
        cfg.update({k: v for k, v in over.items() if k != "seeds"})
        fp = _fingerprint(cfg)
        n_seeds = int(args.seeds) if int(args.seeds) > 0 else int(over.get("seeds", 3))
        for s in range(n_seeds):
            fn = "member_%s_s%d.pth" % (cname, s)
            if (out / fn).exists():
                got = torch.load(out / fn, map_location="cpu",
                                 weights_only=False).get("fingerprint")
                if got == fp:
                    members.append(dict(file=fn, config=cname, seed=s, **cfg))
                    print("   kept  %-22s" % fn, flush=True)
                    continue
                print("   STALE %-22s (%s) -- retraining"
                      % (fn, "no fingerprint" if got is None else got[:8]), flush=True)
            predict = train_one(pool, cache, SimpleNamespace(**cfg, seeds=1), dev_t, seed=s)
            model = predict.model
            assert isinstance(model, ClotGNN), type(model)
            torch.save(dict(state_dict=model.state_dict(), cfg=cfg, seed=s,
                            in_dim=model.enc[0].in_features - model.extra_dim,
                            extra_dim=model.extra_dim, fingerprint=fp), out / fn)
            members.append(dict(file=fn, config=cname, seed=s, **cfg))
            print("   saved %-22s (%.0fs)" % (fn, time.time() - t0), flush=True)

    np.savez_compressed(out / "feature_norm.npz", mu=mu, sd=sd, cols=np.array(cols))
    manifest = dict(
        name=args.name, kind="gnn_ensemble",
        promoted_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        description=(
            "Physics-informed recurrent clot GNN, PHASE10. Same architecture and "
            "configurations as clot_gnn_v2, plus the 13 advective-transport / indicator-gate "
            "channels of src/clot_ml/features_v4.py -- COMSOL's own operator "
            "(dMat/dt + u.grad(Mat) = 0, D=0, wall flux BC) solved on the mesh, which is the "
            "first off-wall feature family that transports along the flow rather than along "
            "the mesh normal. Fitted on the 2026-08-22 cohort -- 23 clot-carrying vessels "
            "(VIZ_HALF released into TRAIN) plus 8 clot-free vessels that carry no recall "
            "and contribute the metric's false-positive branch; SEALED (007/013/031/043) "
            "never seen. Features are post-repair: `wall_normal` and `node_type_*` are "
            "populated and the clot-ML geometry takes the wall/wound union "
            "(MODEL_REVIEW_2026-08-22 6.5, 5b.5). "
            "Scores are STRICTLY NESTED out-of-fold from the CV that selected this design, "
            "not from these weights."),
        docs="docs/PHASE10_V4.md", supersedes=args.supersedes,
        feature_cache=args.cache, v4_channels=list(V4_CHANNELS),
        empty_gt_loss=args.empty_gt_loss, shape_w=float(args.shape_w),
        training_objective_note=(
            "empty_gt_loss records what a CLOT-FREE vessel contributes to the metric term. "
            "It is 'score' for the 2026-08-22 v5a/v5b/v5c tags because those ran before the "
            "flag existed; the repository default is now 'none' (per-node BCE only). "
            "Measured difference: none (MODEL_REVIEW_2026-08-22 8f.4)."),
        requires=("src.clot_ml.features_v4.augment_sample must be applied to a v3 sample "
                  "before predict_scores -- these members expect %d features" % len(cols)),
        training_pool=list(pool), training_pool_carrying=list(carrying),
        training_pool_clot_free=list(free), priority_anchors=list(prio),
        geometry_classes={a: classes[a] for a in pool},
        # A PRODUCTION artifact trained on SEALED carries no valid metric of any kind: its
        # training pool is the whole corpus, so nothing is held out and `STRICT_CV` -- which
        # describes the 36-vessel procedure -- would be a number from a different model.
        # Stamping it here is what lets `eval_clot_ml_0.py` refuse to score it.
        **(dict(metrics_invalid=True, metrics_invalid_reason=(
            "trained with --include-sealed: the whole corpus is in the training pool, so no "
            "held-out estimate exists for THIS artifact. Quote the validated sibling "
            "(docs/PUBLICATION_PLAN.md 12)."), training_includes_sealed=True)
           if args.include_sealed else
           dict(scores_strict_cv=STRICT_CV, metrics_invalid=False)),
        readout=READOUT,
        n_members=len(members), members=members, seeds_per_config=int(args.seeds) or 3,
        feature_norm="feature_norm.npz", n_features=len(cols),
        fingerprint=_fingerprint(dict(BASE, empty_gt_loss=args.empty_gt_loss,
                                      shape_w=float(args.shape_w))))
    # THE TEMPORAL HEAD REGISTERS ITSELF IN THIS SAME FILE.  `promote_clot_gnn_v4_temporal.py`
    # adds `temporal_file` (and its own scores) to this manifest, so writing it wholesale here
    # silently de-registers the head -- and the next `promote_clot_gnn_v4_wound.py` dies with a
    # bare `KeyError: 'temporal_file'` that names neither cause nor cure.  Carry those keys
    # across when the ensemble is byte-identical (fingerprint unchanged, every member kept);
    # drop them loudly when it is not, because a temporal head fitted on different weights is
    # not valid for these.
    mpath = out / "manifest.json"
    if mpath.exists():
        prev = json.loads(mpath.read_text())
        # `kind` too, not just the `temporal*` keys: the temporal promotion PROMOTES the
        # manifest from "gnn_ensemble" to "temporal_v4", and carrying the head's files while
        # resetting its kind leaves a manifest that owns a temporal head and denies it --
        # which `test_temporal_v4_manifest_is_consistent_and_excludes_sealed` catches, and
        # which `locked.py`'s loader dispatches on.
        carry = {k: v for k, v in prev.items() if k.startswith("temporal")}
        if carry and prev.get("kind"):
            carry["kind"] = prev["kind"]
        if carry:
            if prev.get("fingerprint") == manifest["fingerprint"]:
                manifest.update(carry)
                print("   kept temporal head registration (ensemble unchanged)", flush=True)
            else:
                print("[!] ensemble CHANGED -- the temporal head is now stale and "
                      "has been de-registered.  Re-run "
                      "scripts/promote_clot_gnn_v4_temporal.py before promoting "
                      "the wound complement.", flush=True)
    mpath.write_text(json.dumps(manifest, indent=2))
    print("locked %d members -> %s  (%.0fs)" % (len(members), out, time.time() - t0))

    if args.repoint:
        ptr = REPO / "data/reference/clot_gnn_locked.json"
        prev = json.loads(ptr.read_text()) if ptr.exists() else {}
        ptr.write_text(json.dumps(dict(
            name=args.name, kind="gnn_ensemble",
            path=str(out.relative_to(REPO)).replace("\\", "/"),
            manifest=str((out / "manifest.json").relative_to(REPO)).replace("\\", "/"),
            promoted_at=manifest["promoted_at"], docs="docs/PHASE10_V4.md",
            supersedes=prev.get("name", "clot_gnn_v3"),
            scores_strict_cv=STRICT_CV, readout=READOUT), indent=2))
        print("pointer -> %s (now %s)" % (ptr, args.name))
    else:
        print("[i] pointer NOT moved; rerun with --repoint to ship this generation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
