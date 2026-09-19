"""One table of the t=0 velocity fields the clot stack can be built on.

Three properties travel together and were previously written out three times
(``physics_wall_model.dsrx_gain``, ``clot_ml.temporal._FLOW_HOPS``,
``clot_ml.features_v4.HOPS_FOR_FLOW``), so adding a source meant finding all of them and an
arm could silently differentiate its v3 and v4 blocks with different stencils:

    READS_U0_PRED   the field lives in ``data.u0_pred`` rather than ``data.y[0]``
    HOPS            MLS stencil width used to differentiate it
    DSRX_GAIN       amplitude correction on ``d(sr,x)`` before it meets ``sgt``

    gt      COMSOL's own t=0 field.  The ceiling, not deployable.
    fem     the in-house local FEM solve (``clot_ml.v0.solve_fem_into_pack``).  Converged,
            so it takes the GT stencil and no gain.  This is what ships.
    pred    the pre-2026-09 RGP-DEQ surrogate on an analytic prior.  Noisy: hops=6 and a
            fitted 3.0 gain compensate for a second derivative that flips its own sign.
    split   the OFF-WALL feature block computed on `fem` and everything else on `rgp` --
            see `OFFWALL_BLOCK` below and `clot_ml.locked.build_sample`.  Every direct
            consumer of `t0_flow_fields` reads SHEAR, which is a wall quantity, so this
            source behaves as `rgp` there; the split lives in the feature builder.
    rgp     RGP-DEQ residual head on top of the FEM prior (the E-series arms).  Its base
            point IS the ``fem`` field and its residual is band-localised, so it inherits
            the ``fem`` treatment -- hops=3, gain 1.0.  Giving it `pred`'s constants would
            re-apply a correction fitted to a field that is 20x further from the truth.
"""

from __future__ import annotations

#: every legal ``flow`` / ``flow_source`` string
FLOW_SOURCES: tuple[str, ...] = ("gt", "pred", "fem", "rgp", "split")

#: sources whose velocity is read from ``data.u0_pred`` / ``data.v0_pred``
RECONSTRUCTED: frozenset[str] = frozenset({"pred", "fem", "rgp", "split"})

#: MLS stencil width, per source
HOPS: dict[str, int] = {"gt": 3, "pred": 6, "fem": 3, "rgp": 3, "split": 3}

#: ``d(sr,x)`` amplitude correction, per source.  `pred`'s 3.00 is
#: ``DSRX_STENCIL_GAIN * DSRX_SURROGATE_GAIN`` (2.18 * 1.38, rounded), fitted on FIT+DEV;
#: `physics_wall_model.PRED_DSRX_GAIN` re-exports it so ablations have something to patch.
DSRX_GAIN: dict[str, float] = {"gt": 1.0, "pred": 3.00, "fem": 1.0, "rgp": 1.0,
                               "split": 1.0}


#: The feature channels the OFF-WALL scoring domain leans on: advective transport, its
#: attention and reach, and the nearest-wall-node "owner" maps that carry wall quantities to
#: off-wall nodes.
#:
#: They are named here, rather than in the one script that first spliced them, because the
#: measurement that motivates the split is a property of the PROBLEM and not of that script:
#: the wall score is flow-limited (COMSOL's own t=0 field beats the FEM solve by +0.036, seven
#: times the seed-noise null) and the off-wall score is not (that same perfect field scores
#: -0.014, inside its own null), so the best off-wall policy is to leave the field these
#: channels are built from alone and spend a corrected field only on the wall band.
#: `RGP_DEQ_REPAIR_PLAN.md` s18.13-18.15.
OFFWALL_BLOCK: tuple[str, ...] = (
    "att_adv", "att_reach", "log_mat_adv", "log_mat_adv_ind", "log_mat_adv_n",
    "log_mat_ind_owner", "log_mat_off_est", "log_src_reach", "log_tau", "tau_rel_owner",
    "log_mat_owner", "log_mat_phys", "log_mat_phys_ind", "gate_owner", "sr_owner",
    "onset_phys", "onset_phys_ind", "is_shell",
)


def check(flow_source: str) -> str:
    """Return ``flow_source`` if it names a known field, else raise.

    An unrecognised source used to fall through to the ground-truth branch, so a
    ``flow="fem"`` run silently scored GT and looked like a perfect solver.
    """
    src = str(flow_source)
    if src not in FLOW_SOURCES:
        raise ValueError(
            f"unknown flow_source {flow_source!r}; expected one of {', '.join(FLOW_SOURCES)}")
    return src


__all__ = ["FLOW_SOURCES", "RECONSTRUCTED", "HOPS", "DSRX_GAIN", "OFFWALL_BLOCK",
           "check"]
