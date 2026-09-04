"""Species GNN -> full-timeline species -> physics clot trigger rollout.

Builds a ``(T, N, C)`` y-shaped species series from the species GNN pushforward
(continuous log-delta). Non-FI/Mat channels use resting plasma IC; only FI/Mat
are predicted. Clot phi uses ``clot_trigger_physics`` (gelation + nucleation), not ML.

See ``scripts/viz_species_gnn_clot_ladder.py`` and ``docs/MODEL_NOMENCLATURE.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.utils import species_channels as sc
from src.core_physics.clot_phi_simple import sdf_nd_from_data
from src.core_physics.clot_nucleation_mask import resolve_nucleation_hops
from src.core_physics.species_deploy_rollout import (
    alloc_species_y_series,
    band_speed_for_rollout,
    band_uv_for_model,
    deploy_fimat_log_init,
    pin_species_block,
    reset_species_rollout_flow_cache,
    species_rollout_pin_other,
)
from src.architecture.pushforward_config import PushforwardConfig
from src.core_physics.species_pushforward_continuous import (
    SpeciesContinuousBundle,
    SpeciesDualHeadContinuousGNN,
    bind_band_geometry,
    continuous_max_sat_log,
    continuous_vel_decay_enabled,
    predict_continuous_step_delta,
    load_continuous_bundle,
    log_series_on_band,
    model_vel_decay_alphas,
    normalize_log_state,
    pushforward_log_state_step,
)
from src.core_physics.species_pushforward_gnn import (
    SpeciesPushforwardBundle,
    build_band_base_features,
    load_pushforward_bundle,
    pushforward_state_step,
)
from src.training.biochem_species_scope import (
    pushforward_state_dim,
    scatter_log_state_to_species_block,
)
from src.core_physics.species_snapshot_gnn import (
    build_snapshot_features,
    fi_mat_active_labels,
    induced_subgraph,
    snapshot_wall_hops,
    wall_band_mask,
)
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env
from src.training.biochem_species_scope import FI_CHANNEL, MAT_CHANNEL
from src.utils.kinematics_inference import (
    load_kinematics_predictor,
    predict_kinematics_latent,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root

RolloutKind = Literal["continuous", "binary"]

# Session cache for closed-loop kine/corrector handles (eval/viz multi-vessel).
_CLOSED_LOOP_MODELS_CACHE: dict[tuple, object] = {}


def clear_species_gnn_closed_loop_cache() -> None:
    _CLOSED_LOOP_MODELS_CACHE.clear()


def species_gnn_rollout_ckpt() -> Path:
    raw = (
        os.environ.get("T0_R4_SPECIES_GNN_CKPT")
        or os.environ.get("SPECIES_GNN_CLOUT_CKPT")
        or os.environ.get("SPECIES_CONTINUOUS_CKPT")
        or os.environ.get("SPECIES_PUSHFORWARD_CKPT")
        or "outputs/biochem/biochem_gnn/species/best.pth"
    ).strip()
    p = Path(raw)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


@dataclass(frozen=True)
class SpeciesGnnRolloutStatic:
    base_feats: torch.Tensor
    edge_index: torch.Tensor
    node_idx: torch.Tensor
    band: torch.Tensor
    device: torch.device
    pos_band: torch.Tensor | None = None
    flow_series: torch.Tensor | None = None  # [n_t, n_band, flow_dim] for dynamic flow (Trap C)
    flow_cols: tuple[int, int] | None = None  # (start, width) of the flow block in base_feats
    wall_normals_band: torch.Tensor | None = None
    sdf_band: torch.Tensor | None = None
    edge_attr_band: torch.Tensor | None = None


@dataclass(frozen=True)
class SpeciesGnnRolloutBundle:
    kind: RolloutKind
    label: str
    continuous: SpeciesContinuousBundle | None = None
    binary: SpeciesPushforwardBundle | None = None
    config: PushforwardConfig | None = None

    @property
    def device(self) -> torch.device:
        if self.continuous is not None:
            return self.continuous.device
        if self.binary is not None:
            return self.binary.device
        raise RuntimeError("empty SpeciesGnnRolloutBundle")


def _bundle_label_from_path(path: Path, phase: str) -> str:
    path_s = str(path).replace("\\", "/")
    if (
        "biochem_gnn" in phase
        or "biochem_deploy" in phase
        or "clot_deploy_gnn" in phase
        or "biochem_gnn" in path_s
        or "biochem_deploy" in path_s
        or "clot_deploy_gnn" in path_s
        or "continuous" in phase
    ):
        return "biochem_gnn"
    if "s2" in phase or "pushforward" in phase:
        return "s2"
    stem = path.parent.name
    if stem.startswith("species_snapshot_"):
        return stem.replace("species_snapshot_", "")
    return stem or "gnn"


def load_species_gnn_rollout_bundle(
    ckpt_path: Path | str | None = None,
    *,
    device: torch.device | None = None,
    quiet: bool = False,
) -> SpeciesGnnRolloutBundle | None:
    path = Path(ckpt_path) if ckpt_path is not None else species_gnn_rollout_ckpt()
    if not path.is_file():
        if not quiet:
            print(f"[WARN] species GNN rollout checkpoint missing: {path}")
        return None
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=dev, weights_only=False)
    meta = dict(payload.get("meta") or {})
    phase = str(meta.get("phase") or payload.get("phase") or "").lower()
    config = PushforwardConfig.from_meta(meta)

    # Note: Closed-loop models and flow features are resolved based on the new config rather than global env variables,
    # though some environment overrides might be required during transition.
    
    label = _bundle_label_from_path(path, phase)
    if (
        "continuous" in phase
        or "biochem_gnn" in phase
        or "clot_deploy_gnn" in phase
        or bool(meta.get("dual_head"))
        or bool(meta.get("saturation_gate"))
        or bool(meta.get("vel_decay"))
    ):
        cont = load_continuous_bundle(path, device=dev, quiet=True)
        if cont is None:
            return None
        return SpeciesGnnRolloutBundle(kind="continuous", label=label, continuous=cont, config=config)
    binary = load_pushforward_bundle(path, device=dev, quiet=True)
    if binary is None:
        return None
    return SpeciesGnnRolloutBundle(kind="binary", label=label, binary=binary, config=config)


@torch.no_grad()
def prepare_species_gnn_rollout_static(
    data,
    *,
    device: torch.device,
    wall_hops: int | None = None,
    z_kin_override: torch.Tensor | None = None,
    kine_model=None,
) -> SpeciesGnnRolloutStatic:
    """Static band features for the species rollout.

    ``z_kin_override`` injects a clot-aware DEQ latent (full re-solve) so the GraphSAGE teacher's
    primary flow input tracks the rerouted flow once the clot is large enough to change it.

    When ``u0_pred``/``v0_pred`` are missing, runs one joint GINO-DEQ solve and stores them on
    ``data`` so closed-loop coupling does not re-solve the baseline later.
    """
    hops = int(wall_hops if wall_hops is not None else snapshot_wall_hops())
    kine = kine_model
    if kine is None:
        kine = load_kinematics_predictor(resolve_kinematics_checkpoint(), device)
    z_use = z_kin_override
    if z_use is None and (
        getattr(data, "u0_pred", None) is None or getattr(data, "v0_pred", None) is None
    ):
        from src.utils.kinematics_inference import predict_kinematics_and_latent

        pred_uv, z_use = predict_kinematics_and_latent(kine, data.to(device))
        data.u0_pred = pred_uv[:, 0].detach().to(device="cpu").clone()
        data.v0_pred = pred_uv[:, 1].detach().to(device="cpu").clone()
    elif z_use is None:
        z_use = predict_kinematics_latent(kine, data.to(device))
    stat = build_band_base_features(
        data, kine, device, wall_hops=hops, z_kin_override=z_use
    )
    return species_gnn_static_from_band_dict(stat, data, device=device, wall_hops=hops)


def species_gnn_static_from_band_dict(
    stat: dict,
    data,
    *,
    device: torch.device,
    wall_hops: int | None = None,
) -> SpeciesGnnRolloutStatic:
    """Wrap a ``build_band_base_features`` dict without reloading kinematics / re-solving DEQ."""
    hops = int(wall_hops if wall_hops is not None else snapshot_wall_hops())
    band = wall_band_mask(data, device, wall_hops=hops).reshape(-1).bool()
    return SpeciesGnnRolloutStatic(
        base_feats=stat["base_feats"],
        edge_index=stat["edge_index"],
        node_idx=stat["node_idx"],
        band=band,
        device=device,
        pos_band=stat.get("pos_band"),
        flow_series=stat.get("flow_series"),
        flow_cols=stat.get("flow_cols"),
        wall_normals_band=stat.get("wall_normals_band"),
        sdf_band=stat.get("sdf_band"),
        edge_attr_band=stat.get("edge_attr_band"),
    )


def _write_fimat_log_to_species(
    species: torch.Tensor,
    log_state: torch.Tensor,
    node_idx: torch.Tensor,
) -> torch.Tensor:
    return scatter_log_state_to_species_block(species, log_state, node_idx)


def _binary_state_to_log(state: torch.Tensor) -> torch.Tensor:
    fi_sat, mat_sat = continuous_max_sat_log()
    sd = pushforward_state_dim()
    st = state.reshape(-1, sd)
    out = torch.zeros_like(st)
    if sd > 0:
        out[:, 0] = torch.where(st[:, 0] > 0.5, torch.tensor(fi_sat, device=st.device, dtype=st.dtype), out[:, 0])
    if sd > 1:
        out[:, 1] = torch.where(st[:, 1] > 0.5, torch.tensor(mat_sat, device=st.device, dtype=st.dtype), out[:, 1])
    return out


@torch.no_grad()
def rollout_species_gnn_species_series(
    data,
    bundle: SpeciesGnnRolloutBundle,
    static: SpeciesGnnRolloutStatic | None = None,
    *,
    phys_cfg: PhysicsConfig | None = None,
    bio_cfg: BiochemConfig | None = None,
    device: torch.device | None = None,
    pin_other_species: str | None = None,
) -> torch.Tensor:
    """Full-timeline species series ``(T, N, C)`` with FI/Mat from GNN rollout.

    Non-FI/Mat channels use resting plasma IC (deploy default), not GT.
    """
    phys = phys_cfg or PhysicsConfig(phase="biochem")
    bio = bio_cfg or BiochemConfig(phase="biochem")
    dev = device or bundle.device
    stat = static or prepare_species_gnn_rollout_static(data, device=dev)
    pin_mode = pin_other_species if pin_other_species is not None else species_rollout_pin_other()
    reset_species_rollout_flow_cache()
    n_steps = int(data.y.shape[0])
    out = alloc_species_y_series(data, dev)

    if bundle.kind == "continuous":
        assert bundle.continuous is not None
        model = bundle.continuous.model
        wmask = data.mask_wall[stat.node_idx] if hasattr(data, "mask_wall") and data.mask_wall is not None else None
        bind_band_geometry(
            model,
            {
                "pos_band": stat.pos_band,
                "edge_index": stat.edge_index,
                "wall_mask_band": wmask,
                "wall_normals_band": stat.wall_normals_band,
                "sdf_band": stat.sdf_band,
                "edge_attr_band": stat.edge_attr_band,
            },
        )
        log_state = deploy_fimat_log_init(data, dev, stat.node_idx)
        vel_alphas = model_vel_decay_alphas(model) if continuous_vel_decay_enabled(config=bundle.config) else None

        coupler = None
        mu_bulk_si = None

        for t in range(n_steps):
            sp = pin_species_block(data, t, dev, pin_other=pin_mode)  # type: ignore[arg-type]
            sp = _write_fimat_log_to_species(sp, log_state, stat.node_idx)
            out[t, :, sc.SPECIES_BLOCK] = sp
            if t >= n_steps - 1:
                break
            # Deploy-faithful UV only (coupled / u0_pred / RGP-DEQ). Never COMSOL data.y.
            vel_val = band_uv_for_model(
                data, t, dev, stat.node_idx, for_training=False
            )
            pred_delta = predict_continuous_step_delta(
                model,
                stat.base_feats,
                stat.edge_index,
                log_state,
                training=False,
                pos_band=stat.pos_band,
                # only thread time when dynamic flow is active (preserves prior temporal-gate behavior)
                time_index=(t if stat.flow_series is not None else None),
                flow_series=stat.flow_series,
                flow_cols=stat.flow_cols,
                wall_mask_band=wmask,
                species_block=sp,
                velocity=vel_val,
            )
            spd = (
                band_speed_for_rollout(data, t + 1, dev, stat.node_idx)
                if vel_alphas is not None
                else None
            )
            log_state = pushforward_log_state_step(
                log_state,
                pred_delta,
                straight_through=False,
                wall_speed=spd,
                vel_decay_alphas=vel_alphas,
                wall_mask=wmask,
            )
        from src.core_physics.species_viscosity_calibration import (
            apply_mat_beta_to_species_series,
            load_viscosity_calibration,
            resolve_deploy_gelation_beta,
            viscosity_calibration_dir,
        )

        gel_beta = resolve_deploy_gelation_beta(dev)
        if gel_beta is not None:
            cal_path = os.environ.get("SPECIES_VISCOSITY_CALIB_PATH") or str(
                viscosity_calibration_dir() / "beta.pth"
            )
            t_boost = max(int(out.shape[0]) - 1, 0)
            if Path(cal_path).is_file():
                _, calib_bundle = load_viscosity_calibration(cal_path, device=dev)
                t_boost = int(calib_bundle.time_index)
            out = apply_mat_beta_to_species_series(
                out, gel_beta, bio, time_index=min(t_boost, int(out.shape[0]) - 1)
            )
        return out

    assert bundle.binary is not None
    model = bundle.binary.model
    state = fi_mat_active_labels(deploy_fimat_log_init(data, dev, stat.node_idx))
    for t in range(n_steps):
        sp = pin_species_block(data, t, dev, pin_other=pin_mode)  # type: ignore[arg-type]
        log_state = _binary_state_to_log(state)
        sp = _write_fimat_log_to_species(sp, log_state, stat.node_idx)
        out[t, :, sc.SPECIES_BLOCK] = sp
        if t >= n_steps - 1:
            break
        feats = torch.cat([stat.base_feats, state], dim=-1)
        logits = model(feats, stat.edge_index)
        state = pushforward_state_step(state, logits, straight_through=False)
    return out


def _resolve_flow_source(flow_source: str | None) -> str:
    if flow_source is None:
        try:
            from src.core_physics.species_deploy_rollout import species_rollout_vel_source

            src = species_rollout_vel_source()
            if src in ("kinematics", "coupled"):
                return "kinematics"
            if src == "gt":
                return "gt"
        except Exception:
            pass
    raw = (flow_source or os.environ.get("T0_R4_FLOW_SOURCE") or "gt").strip().lower()
    if raw in ("pred", "kine", "kinematics", "deq", "gino", "auto"):
        return "kinematics"
    return "gt"


@torch.no_grad()
def rollout_species_gnn_phi_trajectory(
    data,
    bundle: SpeciesGnnRolloutBundle,
    static: SpeciesGnnRolloutStatic | None = None,
    *,
    phys_cfg: PhysicsConfig | None = None,
    bio_cfg: BiochemConfig | None = None,
    device: torch.device | None = None,
    flow_source: str | None = None,
) -> dict[int, torch.Tensor]:
    from src.core_physics.t0_mu_physics import rollout_t0_clot_phi

    phys = phys_cfg or PhysicsConfig(phase="biochem")
    bio = bio_cfg or BiochemConfig(phase="biochem")
    dev = device or bundle.device
    pred = rollout_species_gnn_species_series(
        data, bundle, static, phys_cfg=phys, bio_cfg=bio, device=dev,
    )
    from src.core_physics.species_viscosity_calibration import resolve_clot_readout_beta

    # Explicit override only -- keeps timeline metrics on the same grading as deploy_clot_f1
    # and stops the stale on-disk t=53 beta from silently re-grading phi.
    gel_beta = resolve_clot_readout_beta()
    flow = _resolve_flow_source(flow_source)
    # Precedence (typed runtime > env > default) lives in one place; see
    # clot_nucleation_mask.resolve_nucleation_hops.
    nuc_hops = resolve_nucleation_hops()
    with t0_rung2_env():
        traj = rollout_t0_clot_phi(
            data, phys, bio, dev,
            gamma_mode=RUNG2_GAMMA_MODE, flow_source=flow,
            pred_species_series=pred, nucleation=True, nucleation_hops=nuc_hops,
            gelation_beta=gel_beta,
        )
    return {int(t): v["phi"] for t, v in traj.items()}
