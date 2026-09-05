"""The graphs Stage-A is *selected* on: real deployment packs, never trained on.

**Why this exists.**  Selection used to run only when ``KINEMATICS_VAL_HOLDOUT_COMSOL_STEMS``
and ``KINEMATICS_INCLUDE_COMSOL_ANCHORS`` were both set *and* comsol steady-kine sidecars
existed under ``graphs_kinematics_anchors/``.  That directory is empty on this cohort, so the
whole T7 selection block was skipped and the run fell back to promoting on
``rel_l2 + 100 * continuity`` -- the metric RGP_DEQ_REPAIR_PLAN.md §10.3 measured as **not**
predicting the clot outcome.  Observed in a smoke run: epoch 1 was saved as "best" with rel-L2
0.578 -> 2.229, because continuity happened to fall.

The steady sidecars are not needed.  ``graphs_biochem_anchors/*.pt`` are the deployment meshes
themselves and carry COMSOL's ``t=0`` velocity in ``y[0]``, which is exactly what the clot stack
consumes and what ``eval_deploy_flow_acceptance.py`` scores against.  Selecting on them closes
the loop between the training run and the acceptance test.

**Seal policy.**  These packs are used to *choose a checkpoint*, which is tuning.  So the pool
excludes both halves of the old SEALED set -- FINAL_HALF (``007/013/031/043``) because it is
reserved for the project's one final read, and VIZ_HALF (``001/010/014/042``) because
``docs/SEALED_SPLIT.md`` allows showing those vessels and not selecting on them.  Everything
else in FIT + DEV is fair game: Stage-A trains on synthetic vessels only.
"""

from __future__ import annotations

import os
from pathlib import Path

# Set by the Stage-A arm scripts (`scripts/stage_a/run_*.sh`), the only setter and one
# outside the tree the knob sweep grepped -- so sweeping these to plain constants made
# every E-series arm a silent no-op for them.  Read from the environment with the swept
# value as the default: unset behaves exactly as the constant did.
KINEMATICS_SELECT_MAX_GRAPHS = os.environ.get("KINEMATICS_SELECT_MAX_GRAPHS", "6")
KINEMATICS_SELECT_PACK_STEMS = ""

#: Comma list restricting the deploy TRAINING pool to a subset of the legal one, for
#: cross-fitting the flow model against the deploy score.  19 of the 27 vessels the biochem
#: deploy score is computed on are in that pool, so an arm trained on all of them and then
#: asked to supply their `t=0` flow is scoring itself on its own training data.  Splitting the
#: pool in halves and precaching each half with the arm that never saw it removes the leak
#: without touching the biochem CV.  It INTERSECTS the legal pool -- it can only ever remove
#: vessels, never add a sealed or selection one -- so a typo cannot widen the pool.
KINEMATICS_DEPLOY_TRAIN_STEMS = os.environ.get("KINEMATICS_DEPLOY_TRAIN_STEMS", "")


#: Deploy packs from an older extractor revision -- dead ``node_type`` and anomalous prior
#: blocks (``comsol002`` is the bit-identical s17 Z2 leak; the ``*_mirror_y`` copies read
#: 0.06-0.45).  Excluded from selection so a stale pack cannot move a checkpoint choice.
#: See PILOT_COHORT_RUNBOOK.md §6.
STALE_EXTRACTOR_STEMS = ("comsol002",)

#: Scores 0.000 in every predicted-flow arm and is its own problem (`DEPLOY_FLOW_PLAN.md` §2).
KNOWN_BAD_STEMS = ("comsol018",)


def selection_pack_dir(root: Path | None = None) -> Path:
    from src.utils.paths import get_project_root

    base = root if root is not None else get_project_root()
    return base / "data/processed/graphs_biochem_anchors"


def selection_pack_stems() -> list[str]:
    """Deploy stems legal to select a Stage-A checkpoint on, deterministically ordered.

    ``KINEMATICS_SELECT_PACK_STEMS`` overrides with an explicit comma list -- and is honoured
    verbatim, seal policy included, because an override is a deliberate act.
    """
    raw = KINEMATICS_SELECT_PACK_STEMS.strip()
    if raw:
        return [s.strip() for s in raw.split(",") if s.strip()]
    from src.core_physics.wall_cohort_splits import DEV, FIT, SEALED, VIZ_RELEASED

    banned = set(SEALED) | set(VIZ_RELEASED) | set(STALE_EXTRACTOR_STEMS) | set(KNOWN_BAD_STEMS)
    return sorted((set(FIT) | set(DEV)) - banned)


def load_selection_packs(*, limit: int = 0, prior_source: str | None = None, verbose=True):
    """Load the deploy packs used for selection, priors rewritten to the deploy-legal source.

    Returns ``[]`` (never raises) when the directory is absent, so a machine without the deploy
    packs still trains -- it just falls back to selecting on synthetic val, and says so.
    """
    import torch

    from src.data_gen.lib.legal_priors import apply_prior_source, resolve_prior_source

    d = selection_pack_dir()
    if not d.is_dir():
        if verbose:
            print(f"[kin] WARN no deploy selection packs under {d}")
        return []
    # `resolve_prior_source` defaults to "stored", and on these packs the stored prior block is
    # bit-identical to COMSOL's t=0 velocity on 43 of 43 vessels (the s17 Z2 leak).  A selection
    # metric read off a leaked prior measures nothing -- the analytic-prior arm posts rel-L2
    # 0.02 that way against its true 0.14 -- so the DEFAULT here is deploy-legal and "stored"
    # has to be asked for by name.
    source = (prior_source or resolve_prior_source(default="analytic") or "analytic").strip()
    if source == "stored":
        print("[kin] WARN selection packs are using the STORED prior block -- on these vessels "
              "that is COMSOL's own t=0 velocity (s17 Z2). Any gate Jaccard read from it is "
              "meaningless. Set SPECIES_PRIOR_SOURCE=analytic.")
    # The STRIDED subset the run actually scores, not the whole legal pool.  One definition:
    # loading 25 and then capping to 8 inside the metric made "the selection set" mean two
    # different things, and a training pool built as "everything except the selection set" then
    # legitimately overlapped the 25 while being disjoint from the 8 -- which is what the
    # train/select leak assert tripped on.
    stems = selection_subset_stems()
    if limit and limit > 0:
        stems = stems[:limit]
    out, missing = [], []
    for stem in stems:
        f = d / f"{stem}.pt"
        if not f.is_file():
            missing.append(stem)
            continue
        try:
            g = torch.load(f, map_location="cpu", weights_only=False)
            y = getattr(g, "y", None)
            if y is None or float(y[..., 0:2].abs().max()) == 0.0:
                missing.append(stem)
                continue
            # Stem BEFORE the prior rewrite: these packs carry no `graph_stem` of their own,
            # and `prior_source="fem"` resolves the vessel's mesh (and its solve cache) by
            # stem.  Setting it afterwards left the FEM prior with nothing to look up.
            g.graph_stem = stem
            g = apply_prior_source(g, source)
            out.append(g)
        except Exception as exc:
            print(f"[kin] WARN selection pack {stem}: {type(exc).__name__}: {exc}")
    if verbose:
        print(f"[kin] Selection set: {len(out)} DEPLOY packs from {d.name} "
              f"(prior_source={source}) -- {', '.join(g.graph_stem for g in out)}")
        if missing:
            print(f"[kin]   missing/unusable: {', '.join(missing)}")
    return out



def use_stems(stems: list[str] | str) -> list[str]:
    """Point ``load_selection_packs`` at an explicit vessel list, for the rest of the process.

    The Stage-A selection subset and the vessels the biochem deploy score is decided on barely
    overlap -- four of the six selection packs are short-timeline vessels the clot cache drops
    entirely -- so "how does this arm score" and "does it help where deployment is lost" are
    different questions over different stems.  Every diagnostic that wants the second one needs
    the same override, so it lives here rather than being re-implemented per probe (and one
    probe re-implementing it WRONG is how an audit silently scored the default six).
    """
    names = ([x.strip() for x in stems.split(",")] if isinstance(stems, str) else list(stems))
    names = [n for n in names if n]
    if not names:
        raise ValueError("use_stems() needs at least one stem")
    globals()["KINEMATICS_SELECT_PACK_STEMS"] = ",".join(names)
    globals()["KINEMATICS_SELECT_MAX_GRAPHS"] = str(len(names))
    os.environ["KINEMATICS_SELECT_PACK_STEMS"] = ",".join(names)
    os.environ["KINEMATICS_SELECT_MAX_GRAPHS"] = str(len(names))
    return names

def selection_subset_stems(cap: int = 0) -> list[str]:
    """The stems a run actually selects on, i.e. what ``KINEMATICS_SELECT_MAX_GRAPHS`` keeps.

    Strided, not the alphabetical prefix -- the first 8 deploy stems are the easy end of the
    cohort (the analytic prior scores 32.5% of ceiling on the strided 8 against 36.7% on the
    prefix 8 and 16.6% over all 25).  Mirrors ``_selection_metrics_on_graphs`` exactly so a
    training pool can be held disjoint from it without the two definitions drifting apart.
    """
    import os as _os

    ordered = selection_pack_stems()
    if cap <= 0:
        cap = int(KINEMATICS_SELECT_MAX_GRAPHS or 0)
    if cap <= 0 or len(ordered) <= cap:
        return ordered
    step = max(1, len(ordered) // cap)
    return ordered[::step][:cap]


def load_deploy_training_packs(*, prior_source: str | None = None, verbose: bool = True):
    """Deploy packs usable as TRAINING graphs -- the regime the synthetic corpus is missing.

    The synthetic corpus's wall `dsrx` sits 10.7x below deployment and its `dsrx < sgt` gate
    branch fires on 0.0% of wall nodes at the median, against 50.8% of firing nodes at
    deployment (RGP_DEQ_REPAIR_PLAN.md §16.5).  These packs ARE the deployment regime, and
    `y[0]` is COMSOL's own `t=0` velocity -- the same field the clot stack consumes.

    The steady `graphs_kinematics_anchors/` sidecars that `KINEMATICS_INCLUDE_COMSOL_ANCHORS`
    expects do not exist on this machine; this reads the deploy packs directly instead.

    **Disjoint from selection by construction.**  Every stem the run selects on is excluded
    here, along with both halves of the old SEALED set, so a training pool can never quietly
    contain a vessel the checkpoint is chosen on.
    """
    import torch

    from src.data_gen.lib.legal_priors import apply_prior_source, resolve_prior_source

    d = selection_pack_dir()
    if not d.is_dir():
        return []
    source = (prior_source or resolve_prior_source(default="analytic") or "analytic").strip()
    from src.core_physics.wall_cohort_splits import DEV, FIT, SEALED, VIZ_RELEASED

    banned = (set(SEALED) | set(VIZ_RELEASED) | set(STALE_EXTRACTOR_STEMS)
              | set(KNOWN_BAD_STEMS) | set(selection_subset_stems()))
    stems = sorted((set(FIT) | set(DEV)) - banned)
    restrict = [x.strip() for x in KINEMATICS_DEPLOY_TRAIN_STEMS.split(",") if x.strip()]
    if restrict:
        keep = set(restrict) & set(stems)
        dropped = sorted(set(restrict) - keep)
        if dropped:
            # Naming a sealed / selection / nonexistent stem is a mistake worth hearing about,
            # but it must not be able to widen the pool, so it is reported and discarded.
            print("[kin] WARN KINEMATICS_DEPLOY_TRAIN_STEMS names %d stem(s) outside the legal "
                  "deploy training pool; ignored: %s" % (len(dropped), ", ".join(dropped)))
        if not keep:
            raise RuntimeError(
                "KINEMATICS_DEPLOY_TRAIN_STEMS was set but selects no legal deploy pack; "
                "the run would train on nothing.  Legal pool: " + ", ".join(stems))
        stems = sorted(keep)
        if verbose:
            print("[kin] deploy training pool RESTRICTED to %d/%d packs (cross-fit): %s"
                  % (len(stems), len(restrict), ", ".join(stems)))
    out = []
    for stem in stems:
        f = d / f"{stem}.pt"
        if not f.is_file():
            continue
        try:
            g = torch.load(f, map_location="cpu", weights_only=False)
            y = getattr(g, "y", None)
            if y is None:
                continue
            if y.dim() == 3:            # [T, N, C] biochem timeline -> the t=0 kinematics
                g.y = y[0].contiguous()
            if float(g.y[:, 0:2].abs().max()) == 0.0:
                continue
            # TRUNCATE to the four kinematics channels.  These packs carry
            # `biochem_v1_16ch` -- `u_nd, v_nd, p_nd, mu_eff_nd` first, which is exactly the
            # kine order, and then TWELVE CHEMISTRY SPECIES.  `PredChannels.WSS` is 4, so
            # `wall_shear_stress_loss` would have supervised the WSS head against
            # `RP_log1p_nd` at a weight of 5.35.  It already disables itself on
            # `y.shape[1] <= 4`, so truncating is both the fix and the signal.
            g.y = g.y[:, :4].contiguous()
            ym = getattr(g, "y_valid_mask", None)
            if torch.is_tensor(ym):
                g.y_valid_mask = (ym[0] if ym.dim() == 3 else ym)[:, :4].contiguous()
            # A graph-level `is_anchor` broadcasts to every node; these packs are fully
            # labelled by COMSOL, so that is the truth rather than a fabrication.
            g.is_anchor = torch.ones(int(g.num_nodes), dtype=torch.bool)
            # Stem BEFORE the prior rewrite -- see `load_selection_packs`.
            g.graph_stem = stem
            g = apply_prior_source(g, source)
            g.is_comsol_anchor = True
            out.append(g)
        except Exception as exc:
            print(f"[kin] WARN deploy training pack {stem}: {type(exc).__name__}: {exc}")
    if verbose:
        print(f"[kin] Deploy TRAINING packs: {len(out)} (prior_source={source}); "
              f"held out of training: {', '.join(sorted(banned & (set(FIT) | set(DEV))))}")
    return out


__all__ = ["load_deploy_training_packs", "load_selection_packs", "selection_pack_dir",
           "selection_pack_stems", "selection_subset_stems"]
