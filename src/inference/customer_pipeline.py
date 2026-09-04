"""Customer-facing deploy pipeline: local FEM t=0 flow + clot_ml_0 rollout.

The only path: in-house Carreau FEM at t=0, then the locked unified ``clot_ml_0``
artifact -- the same stack the research sweeps run.  The species / mat-growth
deploy pipeline that used to sit beside it has been removed; recover it from git
history if a comparison against that retired stack is ever needed.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.t0_device import require_cuda_device


@dataclass
class CustomerTrajectory:
    """Cached per-step fields for the time slider."""

    t_sec: np.ndarray
    pos: np.ndarray
    vel_mag: dict[int, np.ndarray]
    mu_eff_si: dict[int, np.ndarray]
    phi: dict[int, np.ndarray]
    elapsed_s: float = 0.0
    n_steps: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    shear_mag: dict[int, np.ndarray] = field(default_factory=dict)
    mask_wall: np.ndarray | None = None
    mask_wound: np.ndarray | None = None
    mask_inlet: np.ndarray | None = None
    mask_outlet: np.ndarray | None = None
    hop_from_wall: np.ndarray | None = None

    def frame(self, index: int) -> dict[str, np.ndarray | float]:
        i = int(max(0, min(index, self.n_steps - 1)))
        return {
            "index": i,
            "t_sec": float(self.t_sec[i]),
            "vel_mag": self.vel_mag[i],
            "mu_eff_si": self.mu_eff_si[i],
            "phi": self.phi[i],
        }

    def interior_mask(self) -> np.ndarray:
        """Nodes that are neither inlet nor outlet (wall + lumen)."""
        n = int(self.pos.shape[0])
        interior = np.ones(n, dtype=bool)
        if self.mask_inlet is not None:
            interior &= ~np.asarray(self.mask_inlet, dtype=bool).reshape(-1)
        if self.mask_outlet is not None:
            interior &= ~np.asarray(self.mask_outlet, dtype=bool).reshape(-1)
        return interior

    def has_velocity_at(self, index: int) -> bool:
        idxs = (self.meta or {}).get("velocity_indices")
        if idxs is None:
            return bool((self.meta or {}).get("include_velocity", False))
        return int(index) in {int(i) for i in idxs}


DEFAULT_CUSTOMER_CLOT_MODEL = "clot_ml_0"
DEFAULT_CUSTOMER_FLOW = "fem"
_LOCKED_CUSTOMER_CLOT_MODEL = "clot_ml_0"   # alias; resolved via clot_ml.artifacts
#: The alias set lives in `clot_ml.artifacts`; imported rather than restated.
from src.clot_ml.artifacts import LEGACY_NAMES as _CLOT_ML_ALIASES  # noqa: E402


class CustomerDeployPipeline:
    """Customer inference with the locked ``clot_ml_0`` baseline and wound complement.

    A single local Carreau FEM solve produces deployable t=0 flow (``solve_fem_into_pack``).
    The unified ``clot_ml_0`` stack then rolls the clot set forward.  This intentionally does
    not use COMSOL velocity labels.  The retired species pipeline is gone; see git
    for explicit comparisons.
    """

    def __init__(
        self,
        *,
        device: torch.device | None = None,
        model_name: str = DEFAULT_CUSTOMER_CLOT_MODEL,
        require_cuda: bool = True,
        **_legacy_kwargs: Any,
    ) -> None:
        self.device = device or (require_cuda_device() if require_cuda else torch.device("cpu"))
        self.model_name = str(model_name)
        if self.model_name == "legacy_species" or _legacy_kwargs:
            raise ValueError(
                "the legacy species deploy pipeline was removed: the shipped path is "
                "local FEM t=0 + clot_ml_0. Recover it from git history if a "
                "comparison against the retired species stack is really needed."
            )
        # This read `{"clot_ml_0", "clot_ml_0"}` -- a set with the same element twice, so
        # the `clot_ml_v0` alias fell through and was reported verbatim as the locked
        # artifact. `clot_ml.artifacts` already owns the alias set; defer to it rather
        # than re-spelling the pair here.
        self.locked_model_name = (
            _LOCKED_CUSTOMER_CLOT_MODEL
            if self.model_name in _CLOT_ML_ALIASES
            else self.model_name
        )
        self._bundle: dict | None = None

    def _ensure_loaded(self) -> None:
        if self._bundle is not None:
            return
        from src.clot_ml.v0 import load_v0_bundle

        self._bundle = load_v0_bundle(name=self.locked_model_name)

    def run(
        self,
        data,
        *,
        t_final_s: float | None = None,
        progress: Callable[[str], None] | None = None,
        include_velocity: bool = True,
        extra_env: dict[str, str] | None = None,
    ) -> CustomerTrajectory:
        """Generate a deploy-flow C0-tail clot trajectory for one new vessel."""
        del extra_env  # Was a legacy stack knob; the locked baseline is immutable.
        self._ensure_loaded()
        assert self._bundle is not None
        log = progress or (lambda _msg: None)
        out = data.clone() if hasattr(data, "clone") else data
        if t_final_s is not None and hasattr(out, "t") and out.t is not None:
            n_steps = int(out.y.shape[0])
            out.t = torch.linspace(0.0, float(t_final_s), steps=n_steps, dtype=torch.float32)

        # Local FEM at t=0 (same path as research sweeps).  Writes u0_pred/v0_pred; the
        # clot stack consumes them through flow="fem" with the wider MLS stencil.
        log("[i] Solving local FEM flow at t=0...")
        from src.clot_ml.locked import build_sample
        from src.clot_ml.v0 import predict_clot_ml_0, solve_fem_into_pack

        solve_fem_into_pack(out)
        out = out.cpu()
        n_steps = int(out.t.reshape(-1).numel())
        indices = list(range(n_steps))
        bio = BiochemConfig(phase="biochem")
        sample = build_sample(out, bio, flow=DEFAULT_CUSTOMER_FLOW, variant="v4")
        log("[i] Rolling out clot_ml_0...")
        result = predict_clot_ml_0(
            self._bundle,
            out,
            indices,
            flow=DEFAULT_CUSTOMER_FLOW,
            sample=sample,
        )
        phi_all = {
            int(i): np.asarray(result["series"][int(i)], dtype=np.float32)
            for i in indices
        }
        phys = PhysicsConfig(phase="biochem")
        baseline_mu = float(phys.mu_inf)
        clot_mu = float(phys.mu_0)
        mu_all = {
            int(i): baseline_mu + (clot_mu - baseline_mu) * phi_all[int(i)]
            for i in indices
        }
        vel_all: dict[int, np.ndarray] = {
            int(i): np.zeros_like(phi_all[int(i)], dtype=np.float32) for i in indices
        }
        shear_all: dict[int, np.ndarray] = {
            int(i): np.zeros_like(phi_all[int(i)], dtype=np.float32) for i in indices
        }
        velocity_indices: list[int] = []
        if include_velocity and indices and hasattr(out, "u0_pred") and out.u0_pred is not None:
            # The flow field comes from a single local FEM solve at t=0 and is never
            # recomputed over the rollout -- there is exactly one flow snapshot to report,
            # not an "initial vs final" pair (those would be identical and misleading).
            velocity_indices = [indices[0]]
            log("[i] Using t=0 flow field...")
            # u0_pred/v0_pred are non-dimensional (u/u_ref, see solve_fem_into_pack) -- scale
            # back to physical m/s so the UI can show a real, labelled unit.
            u_ref_mps = float(out.u_ref.reshape(-1)[0])
            u0 = out.u0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64) * u_ref_mps
            v0 = out.v0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64) * u_ref_mps
            vel_all[int(indices[0])] = np.sqrt(u0 * u0 + v0 * v0).astype(np.float32)
            try:
                from src.clot_ml.temporal import _flow_hops
                from src.core_physics.mls_gradient import build_mls_gradient, node_positions, shear_rate_2d

                pos_nd = node_positions(out)
                ei_np = out.edge_index.detach().cpu().numpy()
                Dx, Dy = build_mls_gradient(pos_nd, ei_np, hops=_flow_hops(DEFAULT_CUSTOMER_FLOW))
                # u0/v0 are already physical (m/s); the gradient operators differentiate
                # w.r.t. non-dimensional position, so dividing by d_bar (metres) converts the
                # result to a physical d(m/s)/d(m) = 1/s shear rate directly.
                d_bar = float(out.d_bar.reshape(-1)[0])
                ux, uy, vx, vy = Dx @ u0, Dy @ u0, Dx @ v0, Dy @ v0
                sr = shear_rate_2d(ux, uy, vx, vy) / d_bar
                shear_all[int(indices[0])] = sr.astype(np.float32)
            except Exception as exc:
                log(f"[WARN] shear rate unavailable: {exc}")

        def _mask_np(name: str) -> np.ndarray | None:
            value = getattr(out, name, None)
            return None if value is None else value.reshape(-1).bool().detach().cpu().numpy()

        hop_from_wall: np.ndarray | None = None
        try:
            from src.core_physics.species_pushforward_continuous import compute_hop_distances
            from src.clot_ml.wound import solid_mask

            hop_from_wall = compute_hop_distances(
                out.edge_index, torch.as_tensor(solid_mask(out), device=out.edge_index.device), int(out.num_nodes)
            ).detach().cpu().numpy().astype(np.int32)
        except Exception as exc:
            log(f"[WARN] wall-hop distances unavailable: {exc}")

        log(f"[OK] clot_ml_0 rollout done ({n_steps} steps)")
        return CustomerTrajectory(
            t_sec=out.t.detach().cpu().numpy().astype(np.float64),
            pos=out.x[:, :2].detach().cpu().numpy(),
            vel_mag=vel_all,
            mu_eff_si=mu_all,
            phi=phi_all,
            shear_mag=shear_all,
            n_steps=n_steps,
            meta={
                "model": DEFAULT_CUSTOMER_CLOT_MODEL,
                "locked_artifact": self.locked_model_name,
                "flow_source": DEFAULT_CUSTOMER_FLOW,
                "include_velocity": bool(include_velocity),
                "velocity_indices": velocity_indices,
                "velocity_mode": "single" if include_velocity else "none",
                "wound_enabled": bool(getattr(out, "customer_wound_enabled", False)),
                "wound_position_frac": getattr(out, "customer_wound_position_frac", None),
                "wound_width_frac": getattr(out, "customer_wound_width_frac", None),
                "geometry": dict(getattr(out, "customer_parametric_spec", {}) or {}),
            },
            mask_wall=_mask_np("mask_wall"),
            mask_wound=_mask_np("mask_wound"),
            mask_inlet=_mask_np("mask_inlet"),
            mask_outlet=_mask_np("mask_outlet"),
            hop_from_wall=hop_from_wall,
        )
