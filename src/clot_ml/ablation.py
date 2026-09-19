"""The frozen physics/architecture ablation ladder (docs/PHYSICS_ABLATION_PLAN.md).

WHY THIS EXISTS.  The paper wants to claim that the surrogate works because COMSOL's own
governing equations are discretised and the learned parts sit where those equations were
*measured* to fail.  Nothing in the repo has ever stripped the physics and retrained, so that
is a hypothesis (PHYSICS_ABLATION_PLAN 0).  This module freezes the ladder -- the feature
partition, the arms, and the architecture switches -- in ONE place, so an arm cannot drift
between the launcher, the run and the report.

TWO LADDERS, and they answer different questions.

  * **A0-A4, physics as CONDITIONING.**  Columns are held at a constant AFTER standardization
    (`apply_arm` writes the constant; the fold normalizer then maps it to exactly 0.0 because
    a constant column has zero variance and `train_one` clamps `sd` to 1).  The matrix width,
    the `feature_fingerprint`, the normalization layout and the ensemble shapes are unchanged,
    so no cache is rebuilt and no promotion path is touched.  This is the plan's own
    low-risk recipe, and it bounds the value of physics as INFORMATION.
  * **B*, physics and inductive bias as ARCHITECTURE.**  Column zeroing cannot reach the
    anisotropic messages, the residual physics base, the recurrent occlusion seed or the
    metric-shaped loss -- those are code paths.  Each B arm flips exactly one of them at FULL
    features, so its delta is attributable.

THE ONE CONFOUND, stated rather than hidden.  Under the A ladder the physics backbone still
reaches every arm through two architectural doors: `mat_phys` is the additive base of the
regression head, and `phys_mask` seeds round 0 of the recurrent rollout.  So A0/A1 are NOT
generic mesh GNNs, and A4 - A1 UNDERSTATES the value of physics.  That is the conservative
direction for the claim, and `A1_pure` closes it directly: same features as A1 with both
doors shut, which is the genuinely generic mesh-GNN baseline the reviewer is imagining.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "FEATURE_GROUPS", "ARMS", "ARM_NOTES", "ARCH_ARMS", "GROUP_ORDER",
    "columns_for", "zeroed_columns", "apply_arm", "describe_arm",
]


# ---------------------------------------------------------------------------
# the feature partition.  Every column of the v5 cache (68) plus the `phys_mask`
# column `data.attach_physics` appends (69) belongs to EXACTLY ONE group.
# ---------------------------------------------------------------------------
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    # shape and mesh topology only -- nothing that has seen a velocity
    "geom": (
        "is_wall", "is_shell", "is_midside", "dist_wall_edges", "dist_wall_dbar",
        "hop_wall", "degree", "sdf_nd", "width_nd", "width_d1", "width_d2",
    ),
    # the raw t=0 velocity field and its first derivatives.  A mesh GNN that is handed the
    # flow gets exactly this and nothing else.
    "flow": (
        "speed_nd", "u_n", "u_t", "p_nd", "log_sr", "log_absdsrx", "log_absdsry",
        "vort", "div",
    ),
    # neighbourhood and upstream/downstream aggregates OF THE RAW FLOW.  No knowledge of the
    # deposition law is in here -- a message-passing network with 4 layers can build these
    # itself, which is why they belong on the BASELINE side of the ladder (see `A1p`).
    # `sr_owner` is raw shear read at the nearest-wall node: the owner map is mesh topology.
    "flow_agg": (
        "up_spd", "dn_spd", "up_sr", "dn_sr", "sr_owner",
        "sr_mean_h1", "sr_mean_h2", "sr_mean_h4", "sr_mean_h8", "sr_mean_h16",
        "spd_mean_h1", "spd_mean_h2", "spd_mean_h4", "spd_mean_h8", "spd_mean_h16",
    ),
    # PLAN 1b: the dimensionless groups of `G_wall` itself.  These require knowing the
    # deposition law's arguments (`lss`, `sgt`) but integrate nothing.
    "gate": (
        "sr_over_lss", "dsrx_over_sgt", "gate_low", "gate_sep", "gate_sum", "gate_A",
        "dist_to_gate", "gate_owner", "up_gate", "dn_gate", "gate_ind",
        "gate_frac_h1", "gate_frac_h2", "gate_frac_h4", "gate_frac_h8", "gate_frac_h16",
    ),
    # PLAN 1c: the integrated surface-deposition ODE (`integrate_mat_trajectory`, COMSOL's
    # own `J0_Mat` law, with the fitted AP closure inside it) entering as an input, plus the
    # backbone occlusion mask it implies.
    "phys": (
        "log_mat_phys", "onset_phys", "log_mat_owner", "phys_mask",
        "log_mat_phys_ind", "onset_phys_ind", "log_mat_ind_owner",
    ),
    # PLAN 1a: the OFF-WALL operator -- COMSOL's zero-diffusion hyperbolic transport solved
    # as upwind FV with a residence-time cap (`src/clot_ml/transport.py`).  Separated from
    # `phys` on purpose: the ODE and the transport solve are two different equations, and
    # "which solved equation pays" is a question the plan's A3-A2 alone cannot answer.
    "transport": (
        "log_mat_adv", "log_mat_adv_ind", "log_mat_adv_n", "log_src_reach", "log_tau",
        "att_adv", "att_reach", "tau_rel_owner", "log_mat_off_est",
    ),
    # the species initial condition read off the pack
    "ic": ("rp0", "ap0"),
}

#: nesting order, for reporting
GROUP_ORDER: tuple[str, ...] = (
    "geom", "flow", "flow_agg", "gate", "phys", "transport", "ic")


# ---------------------------------------------------------------------------
# the arms.  Values are the groups KEPT; everything else is held at a constant.
# ---------------------------------------------------------------------------
ARMS: dict[str, tuple[str, ...]] = {
    "A0":      ("geom",),
    "A0_pure": ("geom",),                          # + architecture doors shut, see ARCH_ARMS
    "A0_fresh": ("geom",),                         # + doors shut AND direction-free messages
    "A_naive": ("geom",),                          # + everything else stripped, see ARCH_ARMS
    "A_mgn":   ("geom",),                          # + MeshGraphNet-style depth, see ARCH_ARMS
    "A1":      ("geom", "flow"),
    "A1p":     ("geom", "flow", "flow_agg"),
    "A1_pure": ("geom", "flow", "flow_agg"),      # + architecture doors shut, see ARCH_ARMS
    "A2":      ("geom", "flow", "flow_agg", "gate"),
    "A3":      ("geom", "flow", "flow_agg", "gate", "phys"),
    "A4":      GROUP_ORDER,                        # full, the shipped conditioning
}

ARM_NOTES: dict[str, str] = {
    "A0": "geometry and mesh topology only -- but the physics base and rollout seed are still "
          "ON, so this is NOT a physics-free floor; use A0_pure for that",
    "A0_pure": "geometry alone with the physics base and rollout seed removed too -- but the "
               "message passing still carries the velocity field, so NOT physics-free",
    "A_naive": "the FROM-SCRATCH control: geometry only, isotropic messages, one binary head "
               "on plain BCE, no Mat regression, no metric-shaped loss, no C0, one global cut. "
               "What a fresh attempt looks like before anyone knows the physics.",
    "A_mgn": "the PUBLISHED-ARCHITECTURE control: `A_naive` at MeshGraphNet depth (15 "
             "message-passing layers). Answers the reviewer question `A_naive` cannot -- "
             "whether our margin is over a competent published mesh surrogate or only over a "
             "shallow control. Read the scope note in ARCH_ARMS before quoting it as `MGN`.",
    "A0_fresh": "the problem solved fresh: geometry alone, no physics base, no physics seed, "
                "and direction-free messages so the velocity cannot enter through the edges "
                "either.  Read under the `plain` readout, which also keeps `phys_mask` out "
                "of the cut.  This is the true floor.",
    "A1": "+ the raw t=0 velocity field: the plan's generic mesh-GNN baseline",
    "A1p": "+ neighbourhood/directional aggregates of the raw flow: a STRONGER generic "
           "baseline, so A4-A1 cannot be dismissed as a handicapped control",
    "A1_pure": "A1p features with the physics base and the physics rollout seed removed too "
               "-- the genuinely generic mesh GNN",
    "A2": "+ the deposition law's own dimensionless groups (PLAN 1b): knowing the law's "
          "ARGUMENTS, integrating nothing",
    "A3": "+ the integrated deposition ODE and its occlusion mask (PLAN 1c)",
    "A4": "+ the upwind transport solve and the species IC: the shipped conditioning",
}

#: architecture switches an arm may set, mapped to the `train_one` config keys they drive.
#: `A1_pure` is the only FEATURE arm that touches these; every `B*` arm keeps full features.
ARCH_ARMS: dict[str, dict] = {
    # THE STRICTLY-PLAIN FLOOR.  `A0` is *not* one: it zeroes every physics column but leaves
    # both architectural doors open, so it still receives the backbone's `log(Mat/crit)` as the
    # regression base and its occlusion mask as the rollout seed.  Quoting `A0` as "geometry
    # only" therefore overstates what a physics-free model can do -- caught 2026-09-08.  This
    # arm is geometry alone with nothing physics-derived anywhere in the model; read it under
    # the `plain` readout, which is the only one that also keeps `phys_mask` out of the cut.
    "A0_pure":  dict(phys_base=False, phys_seed=False),
    # THE ONE THE OTHERS MISS.  `apply_arm` zeroes only `S["X"]`, but `gnn.to_device` builds the
    # edge features and BOTH aggregation weights from `S["u"]`/`S["v"]` directly -- so the flow
    # field reaches every arm through the message passing no matter which columns are held
    # constant.  `iso=True` is the only switch that closes that door.  Without it, "geometry
    # only" is really "geometry node features + flow-aware message passing" (found 2026-09-08).
    "A0_fresh": dict(phys_base=False, phys_seed=False, iso=True),
    # THE FROM-SCRATCH CONTROL.  Every arm above is still *our* architecture with pieces
    # removed, and that architecture was designed by people who already knew the physics:
    # anisotropic messages exist because PHASE6_RESULTS 3.4 measured the non-locality to be
    # advective rather than diffusive; the auxiliary head regresses the simulator's own `Mat`
    # field; the loss is shaped like the evaluation metric; C0 is a constraint discovered on
    # this problem; the hyperparameters were tuned with the full physics conditioning on.
    # Calling any of them "a plain GNN" overstates the baseline.
    #
    # This is what somebody starting fresh would actually write: geometry and mesh topology,
    # direction-free message passing, ONE binary head on plain BCE, no auxiliary regression
    # onto `Mat`, no metric-shaped loss, no C0, no physics base or seed, and a single global
    # threshold at readout.  It is expected to score WORSE than `A1_pure`, which is the point:
    # it bounds how much of the reported physics contribution is really "our architecture".
    "A_naive":  dict(phys_base=False, phys_seed=False, iso=True,
                     reg_w=0.0, metric_w=0.0, shape_w=0.0, clot_free_w=1.0, rounds=1),
    # THE PUBLISHED-ARCHITECTURE CONTROL, added 2026-09-09.
    #
    # WHY IT EXISTS.  `A_naive` is a from-scratch control at the SHIPPED depth (4 layers), so
    # it bounds the from-scratch floor and nothing else.  It does not answer "is your margin
    # over a competent published mesh surrogate?", which is the question a reviewer asks once
    # they have read Pelissier et al. 2026 (Comput Biol Med 208:111649), who benchmark against
    # MeshGraphNet, BSMS-GNN and Transolver++ at matched compute.  Without an arm like this the
    # manuscript has to say, in the methods, that no published architecture was benchmarked.
    #
    # WHAT IT IS.  `A_naive` at MeshGraphNet's depth: 15 message-passing layers, `dim` left at
    # the shipped 64, which is exactly the configuration Pelissier et al. report for their own
    # MGN replication ("15 message-passing layers and a hidden dimension of 64 for each MLP
    # block").  The per-layer block in `gnn.py` is already MeshGraphNet-shaped -- an edge MLP
    # over [x_src, x_dst, edge_attr], a node update MLP, LayerNorm and a residual -- and
    # `iso=True` makes the edge features direction-free geometric offsets, which is MGN's
    # relative-displacement encoding.  Physics is absent everywhere: no base, no seed, no
    # physics columns, no metric-shaped loss, no C0, one binary head on plain BCE, one global
    # cut, and it must be read under the `plain` readout so `phys_mask` cannot enter the
    # threshold either (docs/PAPER.md 5, the three-door result).
    #
    # WHAT IT IS NOT -- AND THE MANUSCRIPT MUST SAY SO.  This is NOT a MeshGraphNet
    # reimplementation.  MGN carries edge features that are themselves updated at every
    # processor step; our block recomputes messages from static edge attributes and updates
    # nodes only.  There is no separate edge encoder/decoder, and the training loop is ours
    # (one full-graph step per vessel per epoch), not Pfaff et al.'s noise-injected rollout.
    # Call it "a MeshGraphNet-style depth-matched control", never "MeshGraphNet".  Claiming
    # the latter is the same class of overclaim as the priority sentences retracted on
    # 2026-09-09 (docs/PUBLICATION_NOTES.md 6).
    # ATTEMPT 1 COLLAPSED -- kept in the comment because the failure is the useful part.
    # `layers=15` at the shipped `lr=3e-3` produced a CONSTANT field: std 1.2e-06 over 9,490
    # nodes, 0.1691 everywhere, i.e. the base rate.  It scored 0.1639 wall / 0.1116 off against
    # `A_naive`'s 0.6833 / 0.4849.  That is not a result about MeshGraphNet and must never be
    # quoted as one -- the arm did not train.  Cause: the per-layer block is POST-norm residual
    # (`self.norm(x + self.drop(h))`, gnn.py:88), and a 15-deep post-norm stack driven by
    # OneCycleLR at max_lr=3e-3 -- a recipe tuned at 4 layers -- diverges into the prior.
    # Attempt 2 lowers the peak lr, which is the standard fix and the only thing changed.
    #
    # IF THIS ALSO COLLAPSES, STOP AND SAY SO.  The fallback is already written into
    # docs/PAPER.md 9: state in the methods that no published architecture was benchmarked and
    # that `A_naive` bounds the from-scratch floor rather than the state of the art.  Reporting
    # a collapsed arm as a baseline would be a worse overclaim than reporting none.
    "A_mgn":    dict(phys_base=False, phys_seed=False, iso=True,
                     reg_w=0.0, metric_w=0.0, shape_w=0.0, clot_free_w=1.0, rounds=1,
                     layers=15, lr=5e-4),
    "A1_pure":  dict(phys_base=False, phys_seed=False),
    # --- the B ladder: one code path each, at full A4 conditioning -------------------
    "B_iso":    dict(iso=True),
    "B_nomp":   dict(layers=0),
    # `layers=0` alone does NOT remove the graph: the K=3 refinement rounds still feed each
    # node its neighbours' and its owner's occupancy through `recurrent.feedback_channels`,
    # which is graph communication by another route.  `B_mlp` shuts that down too and is the
    # honest "no graph at all" control; the pair prices the two channels separately.
    "B_mlp":    dict(layers=0, rounds=1),
    "B_nobase": dict(phys_base=False),
    "B_noseed": dict(phys_seed=False),
    "B_r1":     dict(rounds=1),
    "B_bce":    dict(metric_w=0.0),
}

ARM_NOTES.update({
    "B_iso": "isotropic messages: the t=0 velocity is removed from the edge features and "
             "up/downstream aggregation is made direction-free (PHASE6_RESULTS 3.4 claim)",
    "B_nomp": "no message-passing LAYERS (layers=0); the recurrent rounds still carry "
              "neighbour and owner occupancy, so the graph is not fully gone",
    "B_mlp": "no graph at all: layers=0 AND one round, a purely node-wise MLP on the same "
             "features -- the true non-graph floor",
    "B_nobase": "the regression head no longer sits on the physics `log(Mat/crit)` base",
    "B_noseed": "the recurrent rollout starts from zero occlusion instead of `phys_mask`",
    "B_r1": "one pass instead of 3 shared-weight refinement rounds",
    "B_bce": "metric-shaped loss off (metric_w=0): plain BCE + regression",
})

#: every B arm keeps the full conditioning
for _b in ARCH_ARMS:
    ARMS.setdefault(_b, GROUP_ORDER)


# ---------------------------------------------------------------------------
def columns_for(arm: str) -> tuple[str, ...]:
    """The column names an arm KEEPS."""
    if arm not in ARMS:
        raise KeyError("unknown ablation arm %r; known: %s" % (arm, ", ".join(sorted(ARMS))))
    keep: list[str] = []
    for g in ARMS[arm]:
        keep.extend(FEATURE_GROUPS[g])
    return tuple(keep)


def zeroed_columns(arm: str, cols: list[str]) -> list[str]:
    """The columns of `cols` this arm holds at a constant, in cache order.

    Raises if `cols` holds a name the partition does not cover: an unpartitioned column
    would be silently KEPT in every arm, which is the one failure mode that would make the
    whole ladder wrong without anything looking wrong.
    """
    known = {c for g in FEATURE_GROUPS.values() for c in g}
    unknown = [c for c in cols if c not in known]
    if unknown:
        raise ValueError(
            "feature columns missing from src/clot_ml/ablation.FEATURE_GROUPS: %s. "
            "Every column must be partitioned or the ablation silently keeps it."
            % ", ".join(unknown))
    keep = set(columns_for(arm))
    return [c for c in cols if c not in keep]


def apply_arm(cache: dict, arm: str, *, verbose: bool = True) -> list[str]:
    """Hold this arm's ablated columns at a constant, IN PLACE, on an attached cache.

    Call AFTER `data.attach_physics` -- the `phys_mask` column it appends is part of the
    partition, and `attach_physics` also reads `sdf_nd` out of `X` to build `S["sdf"]`,
    which must see the real values.

    Writing the constant 0.0 rather than the training mean is deliberate and equivalent:
    `train_one` standardizes with the FOLD's own `mu`/`sd`, and a constant column has
    `sd == 0`, which that function clamps to 1.0 -- so the standardized column is exactly
    0.0 in every fold, which is the neutral value the plan asks for.  Zeroing the RAW column
    instead would inject an out-of-distribution value; this cannot, because the value is
    constant and therefore carries no information whatever it is.

    Returns the ablated column names.
    """
    if not cache:
        return []
    cols = [str(c) for c in next(iter(cache.values()))["cols"]]
    drop = zeroed_columns(arm, cols)
    idx = [cols.index(c) for c in drop]
    for a, S in cache.items():
        if [str(c) for c in S["cols"]] != cols:
            raise ValueError("%s: column layout differs from the rest of the cache" % a)
        X = np.array(S["X"], copy=True)
        if idx:
            X[:, idx] = 0.0
        S["X"] = X
    if verbose:
        print("[ablation] arm=%s keeps %s | %d/%d columns held constant%s"
              % (arm, "+".join(ARMS[arm]), len(drop), len(cols),
                 (": " + ", ".join(drop)) if drop else ""), flush=True)
    return drop


def describe_arm(arm: str) -> str:
    arch = ARCH_ARMS.get(arm, {})
    return "%s: %s%s" % (
        arm, ARM_NOTES.get(arm, ""),
        ("  [arch: %s]" % ", ".join("%s=%s" % kv for kv in sorted(arch.items())))
        if arch else "")
