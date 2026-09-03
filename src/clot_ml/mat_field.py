"""v6: a LEARNED surface ``Mat`` field, because the ODE's cannot drive the off-wall rule.

WHY THIS MODULE EXISTS.  ``docs/WOUND_PROGRESS.md`` §16 localised the entire off-wall gap to
one quantity and proved the rest of the architecture is sound.  Holding the shell, the
topological owner map and the 0.16 attenuation FIXED and changing only which ``Mat`` feeds
them, the shipped rule ``shell & off_att*Mat_owner >= crit`` scores:

    Mat source                      001      002      003
    ODE (shipped)                0.0000   0.0000   0.0000      <- fires on NOTHING
    ClotGNN reg head             0.0000   0.0000   0.0000      <- zero-init residual, inert
    GT                           0.9755   0.9755   0.7897

and on ``wound_patient003``'s far-field candidates GT ``Mat_owner`` separates clot from lumen
at **AUC 0.9961** while the ODE's is at **chance (0.5048)**, its two medians equal to three
significant figures.  A total flow stall -- the strongest form of the flow hypothesis, bounded
exactly by ``gate == 1`` -- moves the ODE's wall p90 from 1.73x to 2.31x crit where the rule
needs 6.25x.  So the field has to be learned; there is no flow model, blockage or
recalibration that recovers it.

WHAT IS PREDICTED, AND WHY IT IS NOT A MAGNITUDE REGRESSION.  ``wound_patient003``'s wall
``Mat`` p90 is **27.78x crit, the largest in the dataset** -- no non-wound vessel exceeds
11.13x (``scripts/diag_mat_magnitude_cohort.py``).  Regressing the magnitude and thresholding
it would therefore ask the model to extrapolate past its training support on exactly the
vessel that matters.  But the rule is a THRESHOLD, not a magnitude: it only asks whether
``Mat_owner`` clears ``crit / off_att``.  That boundary is deep inside the training support --
nine legal vessels carry 9-21% of their solid nodes above it -- so the primary head is a
CLASSIFIER of the crossing and the magnitude regression rides along as an auxiliary.

The model is :class:`~src.clot_ml.gnn.ClotGNN` unchanged, with the time-varying channels
supplied through its existing ``extra`` port and the ODE's own ``Mat(t)`` as the regression
head's residual base -- so at initialisation v6 IS the physics, and training only adds to it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

#: Time samples per vessel used for training.  The target is a monotone crossing, so the
#: curve is nearly a step and does not need dense sampling; 16 keeps a 48-pack cache under
#: ~400 MB and every vessel contributes its whole horizon.
N_TIME_SAMPLES: int = 16

#: The off-wall attenuation the shipped rule uses.  The classifier head predicts
#: ``Mat >= crit / OFF_ATT`` -- i.e. exactly the question the readout asks, so no threshold
#: has to be re-fitted downstream.
OFF_ATT: float = 0.16

#: Extra (time-varying) channels handed to ``ClotGNN.extra``:
#: ``t/T``, the ODE's own ``log1p(Mat/crit)``, its owner's, and whether it has ignited.
EXTRA_CHANNELS: tuple[str, ...] = ("t_frac", "ode_self", "ode_owner", "ode_fired")


@dataclass
class MatFieldConfig:
    """Typed knobs for the v6 field.  No env toggles -- see AGENTS.md's guardrail."""

    dim: int = 96
    layers: int = 6
    drop: float = 0.1
    lr: float = 2e-3
    wd: float = 1e-4
    epochs: int = 60
    cls_w: float = 1.0
    reg_w: float = 0.3
    #: Positives (``Mat >= crit/0.16``) are ~5-10% of solid nodes and MOST vessels have none
    #: at all -- 30 of 47 have wall p90 at or below 1.01x crit.  At ``pos_weight = 3`` the
    #: first trained field came back with ``Mat`` p90 = 2.00x on ``wound_patient003`` against
    #: a GT 27.78x, i.e. the residual had collapsed onto the ODE's own 1.96x and learned
    #: nothing there.  Shrinkage toward an overwhelmingly empty prior is the thing to beat.
    pos_weight: float = 12.0
    #: Weight on the regression term GROWS with the target, so the nodes that decide the
    #: off-wall rule are not averaged away by the structural zeros (PHASE7 8: ~45% of
    #: mid-side wall nodes carry a structural zero ``Mat``).
    reg_mag_w: float = 1.0
    #: Down-weight vessels with no clot anywhere.  They are real negatives and worth keeping
    #: -- false positives on clot-free tubes are a scored failure -- but at full weight they
    #: are most of the gradient.
    clot_free_w: float = 0.25
    seed: int = 0
    #: vessels whose packs may be read during training.  SEALED and the target wound vessel
    #: are excluded by the caller, not here, so the exclusion is visible at the call site.
    train_stems: tuple[str, ...] = field(default_factory=tuple)


def _log1p_crit(a: np.ndarray, crit: float) -> np.ndarray:
    return np.log1p(np.maximum(np.asarray(a, dtype=np.float64), 0.0) / float(crit))


def sample_time_indices(T: int, k: int = N_TIME_SAMPLES) -> np.ndarray:
    """``k`` indices spanning the horizon, always including the final frame.

    The final frame is the one the deploy score reads, so it must never be dropped by a
    rounding accident on a short vessel.
    """
    T = int(T)
    if T <= k:
        return np.arange(T, dtype=np.int64)
    idx = np.unique(np.linspace(0, T - 1, int(k)).round().astype(np.int64))
    if idx[-1] != T - 1:
        idx = np.append(idx, T - 1)
    return idx


#: Which physics field the learned residual sits on top of.  ``"chem"`` is the field the
#: SHIPPED wound off-wall readout actually integrates (`v0.chemistry_mat_trajectory`: upwind
#: AP renewal, `da_scale_auto`, washout, wound-rate blockage, no AP closure).  ``"ode"`` is
#: the plain surface ODE v6 was first written against in 2026-08.  The base matters more than
#: any hyper-parameter here: because `head_reg` is zero-init, an UNTRAINED field reproduces
#: its base exactly, so choosing "chem" makes the untrained arm equal to what ships and every
#: measured move attributable to the learned residual alone.
MAT_BASES: tuple[str, ...] = ("chem", "ode")


def build_mat_field_entry(data, bio_cfg, *, flow: str = "gt",
                          n_times: int = N_TIME_SAMPLES,
                          wound_rate: tuple[float, float] | None = None,
                          mat_base: str = "chem",
                          v0_cfg=None) -> dict:
    """One vessel's v6 training entry: the static v4 sample plus per-time base and GT ``Mat``.

    ``wound_rate`` must be the SHIPPED artifact's fitted ``(G_pre, G_post)``, so the residual
    base the model corrects is exactly the field deploy would have used on its own; passing
    ``None`` on a wound pack would train against a quieter wound than deploy integrates
    (WOUND_PROGRESS 15).  Ignored on packs with no wound mask, where it is a structural no-op.

    ``mat_base`` selects that field -- see :data:`MAT_BASES`.  ``v0_cfg`` is the shipped
    :class:`~src.clot_ml.v0.ClotMlV0Config`; it is what carries ``da_scale_auto`` and the
    washout flag, and it is REQUIRED for ``mat_base="chem"`` so the base cannot silently
    drift from the artifact's own settings.
    """
    from src.clot_ml.locked import build_sample
    from src.clot_ml.wound import has_wound
    from src.core_physics.clot_phi_simple import mat_si_for_gelation_from_log1p
    from src.core_physics.physics_lumen_model import first_corner_shell, topological_owner

    if mat_base not in MAT_BASES:
        raise ValueError(f"mat_base must be one of {MAT_BASES}, got {mat_base!r}")
    crit = float(bio_cfg.viscosity_mat_crit)
    S = build_sample(data, bio_cfg, flow=flow, variant="v4")
    T = int(data.y.shape[0])
    ti = sample_time_indices(T, n_times)

    wr = tuple(wound_rate) if (wound_rate is not None and has_wound(data)) else None
    if mat_base == "chem":
        from src.clot_ml.v0 import ClotMlV0Config, chemistry_mat_trajectory

        traj = chemistry_mat_trajectory(data, bio_cfg, v0_cfg or ClotMlV0Config(),
                                        flow=flow, sample=S, wound_rate=wr)
    else:
        from src.clot_ml.temporal import ode_trajectory

        traj, _ = ode_trajectory(data, bio_cfg, flow=flow, wound_rate=wr)
    traj = np.asarray(traj, dtype=np.float64)

    mi = data.y_channel_names.split(",").index("Mat_log1p_nd")
    gmat = mat_si_for_gelation_from_log1p(data.y[:, :, mi], bio_cfg).reshape(T, -1).numpy()

    pos = np.asarray(S["pos"], dtype=np.float64)
    ei = np.asarray(S["edge_index"])
    solid = np.asarray(S.get("solid", S["wall"]), dtype=bool)
    town = topological_owner(pos, solid, ei)
    shell = first_corner_shell(pos, solid, ei)

    return dict(
        X=np.asarray(S["X"], dtype=np.float32),
        edge_index=ei.astype(np.int64),
        pos=pos.astype(np.float32),
        u=np.asarray(S["u"], dtype=np.float32),
        v=np.asarray(S["v"], dtype=np.float32),
        wall=np.asarray(S["wall"], dtype=bool),
        solid=solid,
        shell=shell,
        owner=np.asarray(S["owner"], dtype=np.int64),
        town=town.astype(np.int64),
        t_idx=ti.astype(np.int64),
        T=np.int64(T),
        mat_base=np.str_(mat_base),
        # per-time fields, log1p(Mat/crit); [K, N]
        ode_t=_log1p_crit(traj[ti], crit).astype(np.float32),
        gt_t=_log1p_crit(gmat[ti], crit).astype(np.float32),
    )


def extra_channels(entry: dict, k: int, dev) -> torch.Tensor:
    """``[N, 4]`` time-varying input for ``ClotGNN.extra`` at time sample ``k``."""
    ode = np.asarray(entry["ode_t"][k], dtype=np.float32)
    town = np.asarray(entry["town"], dtype=np.int64)
    own = np.where(town >= 0, ode[np.maximum(town, 0)], 0.0).astype(np.float32)
    T = int(entry["T"])
    tf = float(entry["t_idx"][k]) / max(T - 1, 1)
    cols = np.stack([
        np.full(ode.shape, tf, dtype=np.float32),
        ode,
        own,
        (ode >= np.log1p(1.0)).astype(np.float32),
    ], axis=1)
    return torch.tensor(cols, dtype=torch.float32, device=dev)


def crossing_target(entry: dict, k: int, off_att: float = OFF_ATT) -> np.ndarray:
    """``Mat_gt(t) >= crit / off_att`` -- the question the off-wall readout actually asks."""
    return (np.asarray(entry["gt_t"][k]) >= np.log1p(1.0 / float(off_att))).astype(np.float32)


def build_static_graph(entry: dict, mu: np.ndarray, sd: np.ndarray, dev) -> dict:
    """Everything that does not depend on time, moved to device once per vessel."""
    from src.clot_ml.gnn import edge_features

    ei = np.asarray(entry["edge_index"])
    pos = np.asarray(entry["pos"], dtype=np.float64)
    u, v = np.asarray(entry["u"]), np.asarray(entry["v"])
    h_edge = float(np.median(np.linalg.norm(pos[ei[0]] - pos[ei[1]], axis=1)))
    ea = edge_features(pos, ei, u, v, h_edge)
    cos_s = ea[:, 4:5]
    t = lambda a, d=torch.float32: torch.tensor(np.ascontiguousarray(a), dtype=d, device=dev)
    X = (np.asarray(entry["X"], dtype=np.float32) - mu) / sd
    return dict(
        x=t(X), ei=t(ei, torch.long), ea=t(ea),
        w_up=t(np.clip(cos_s, 0.0, None)), w_dn=t(np.clip(-cos_s, 0.0, None)),
        solid=t(np.asarray(entry["solid"], dtype=np.float32)),
        n=int(len(entry["solid"])),
    )


def make_model(in_dim: int, edim: int, cfg: MatFieldConfig):
    from src.clot_ml.gnn import ClotGNN

    return ClotGNN(in_dim, edim, dim=cfg.dim, layers=cfg.layers, drop=cfg.drop,
                   extra_dim=len(EXTRA_CHANNELS))


@torch.no_grad()
def predict_entry(model, entry: dict, mu, sd, dev, k: int) -> tuple[np.ndarray, np.ndarray]:
    """``(p_cross, mat_log1p)`` at every node for time sample ``k``."""
    model.eval()
    g = build_static_graph(entry, mu, sd, dev)
    ex = extra_channels(entry, k, dev)
    base = torch.tensor(np.asarray(entry["ode_t"][k], dtype=np.float32), device=dev)
    logit, reg = model(g["x"], g["ei"], g["ea"], g["w_up"], g["w_dn"], base, extra=ex)
    return torch.sigmoid(logit).cpu().numpy(), reg.cpu().numpy()
