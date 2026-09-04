"""Research sweep rollout backend: clot_ml_0 on the in-house FEM t=0 flow."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from src.clot_ml.locked import build_sample
from src.clot_ml.v0 import load_v0_bundle, predict_clot_ml_0, solve_fem_into_pack
from src.clot_ml.wound import has_wound, solid_mask, wound_region_masks
from src.config import BiochemConfig
from src.evaluation.research_parameters import (
    research_parameters_from_trajectory,
    write_scientific_csv,
)
from src.evaluation.research_sweep_geometry import load_or_build_research_graph
from src.inference.customer_pipeline import (
    CustomerDeployPipeline,
)


@dataclass
class ClotMlResearchTrajectory:
    """Adapter: clot_ml boolean masks -> research_parameters timeseries schema."""

    t_sec: np.ndarray
    pos: np.ndarray
    phi: dict[int, np.ndarray]
    vel_mag: dict[int, np.ndarray]
    n_steps: int
    elapsed_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)
    mask_wall: np.ndarray | None = None
    mask_inlet: np.ndarray | None = None
    mask_outlet: np.ndarray | None = None
    hop_from_wall: np.ndarray | None = None

    def frame(self, index: int) -> dict[str, np.ndarray | float]:
        i = int(max(0, min(index, self.n_steps - 1)))
        return {
            "index": i,
            "t_sec": float(self.t_sec[i]),
            "phi": self.phi[i],
            "vel_mag": self.vel_mag.get(i, np.zeros_like(self.phi[i])),
        }

    def has_velocity_at(self, _index: int) -> bool:
        return False


def _bool_series_to_phi(series: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for k, mask in series.items():
        m = np.asarray(mask, dtype=bool).reshape(-1)
        out[int(k)] = m.astype(np.float64)
    return out


def _wound_domain_summary(data, series: dict[int, np.ndarray], *, final_idx: int) -> dict[str, float]:
    """Wound-region coverage at final frame (no GT required)."""
    out: dict[str, float] = {
        "wound_enabled": 1.0 if has_wound(data) else 0.0,
        "wound_area_pct": float("nan"),
        "wound_region_clot_pct_final": float("nan"),
        "wound_lumen_clot_pct_final": float("nan"),
        "far_lumen_clot_pct_final": float("nan"),
    }
    if not has_wound(data):
        return out
    solid = solid_mask(data)
    wound = getattr(data, "mask_wound", None)
    if wound is not None:
        w = wound.reshape(-1).bool().cpu().numpy()
        out["wound_area_pct"] = 100.0 * float(w.sum()) / max(int(solid.sum()), 1)
    keys = sorted(series.keys())
    if not keys:
        return out
    ti = keys[min(final_idx, len(keys) - 1)]
    pred = np.asarray(series[int(ti)], dtype=bool)
    region, lumen, far = wound_region_masks(data)
    for name, dom in (
        ("wound_region_clot_pct_final", region),
        ("wound_lumen_clot_pct_final", lumen),
        ("far_lumen_clot_pct_final", far),
    ):
        n = int(dom.sum())
        if n <= 0:
            out[name] = 0.0
        else:
            out[name] = 100.0 * float((pred & dom).sum()) / float(n)
    return out


def run_clot_ml_0_arm(
    *,
    data,
    geom_spec: dict[str, Any],
    flow: str = "fem",
    clot_model: str = "clot_ml_0",
    include_velocity: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[ClotMlResearchTrajectory, dict[str, Any]]:
    """FEM t=0 flow + clot_ml_0 temporal rollout on a research graph."""
    log = progress or (lambda _msg: None)
    t0 = time.perf_counter()

    bio = BiochemConfig(phase="biochem")
    bundle = load_v0_bundle(clot_model)
    times = list(range(int(data.y.shape[0])))

    flow_norm = str(flow).lower().strip()
    if flow_norm == "fem":
        log("[i] Solving in-house FEM flow at t=0...")
        solve_fem_into_pack(data)
        sample_flow = "fem"   # keep label; features.py/temporal.py give FEM the GT treatment
    elif flow_norm in ("pred", "gt"):
        sample_flow = flow_norm
    else:
        raise ValueError(f"Unsupported flow={flow!r}; use fem, pred, or gt")

    log("[i] Building clot_ml sample features...")
    sample = build_sample(data, bio, flow=sample_flow, variant="v4")
    log("[i] Running clot_ml_0 rollout...")
    result = predict_clot_ml_0(
        bundle,
        data,
        times,
        flow=sample_flow,
        sample=sample,
    )

    series = result.get("series") or {}
    if not series:
        raise RuntimeError("clot_ml_0 returned empty series")

    t_sec = data.t.detach().cpu().numpy().astype(np.float64)
    keys = sorted(int(k) for k in series.keys())
    phi = _bool_series_to_phi(series)
    vel: dict[int, np.ndarray] = {i: np.zeros(int(data.num_nodes), dtype=np.float32) for i in range(len(keys))}

    traj = ClotMlResearchTrajectory(
        t_sec=t_sec[: len(keys)] if t_sec.shape[0] >= len(keys) else np.linspace(0, float(t_sec[-1]), len(keys)),
        pos=data.x[:, :2].detach().cpu().numpy(),
        phi={i: phi[keys[i]] for i in range(len(keys))},
        vel_mag=vel,
        n_steps=len(keys),
        elapsed_s=time.perf_counter() - t0,
        meta={
            "backend": "clot_ml_0",
            "clot_model": clot_model,
            "flow": flow_norm,
            "sample_flow": sample_flow,
            "include_velocity": bool(include_velocity),
        },
        mask_wall=_mask_np(data, "mask_wall"),
        mask_inlet=_mask_np(data, "mask_inlet"),
        mask_outlet=_mask_np(data, "mask_outlet"),
        hop_from_wall=_hop_from_wall(data),
    )

    pack = research_parameters_from_trajectory(traj)
    pack["summary"].update(_wound_domain_summary(data, series, final_idx=-1))
    pack["summary"]["clot_ml_onset"] = result.get("onset")
    return traj, pack


def _mask_np(data, name: str) -> np.ndarray | None:
    m = getattr(data, name, None)
    if m is None:
        return None
    return m.reshape(-1).bool().detach().cpu().numpy()


def _hop_from_wall(data) -> np.ndarray | None:
    try:
        from src.core_physics.species_pushforward_continuous import compute_hop_distances

        wall_t = getattr(data, "mask_wall", None)
        ei = getattr(data, "edge_index", None)
        if wall_t is None or ei is None:
            return None
        n = int(data.num_nodes)
        return (
            compute_hop_distances(ei, wall_t.reshape(-1).bool(), n)
            .detach()
            .cpu()
            .numpy()
            .astype(np.int32)
        )
    except Exception:
        return None


def run_research_arm(
    *,
    arm: dict[str, Any],
    control: dict[str, Any],
    out_dir,
    cache_dir,
    force_rebuild: bool,
    model: str,
    flow: str,
    clot_model: str,
    pipeline: CustomerDeployPipeline | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build geometry, run selected backend, write per-arm artifacts."""
    from pathlib import Path

    name = str(arm.get("name", "arm"))
    log = progress or (lambda _msg: None)
    out_dir = Path(out_dir)

    log(f"[i] Arm {name}: building / loading geometry...")
    t0 = time.perf_counter()
    data, geom_spec, mesh_pt = load_or_build_research_graph(
        arm,
        control,
        cache_dir=cache_dir,
        force_rebuild=force_rebuild,
    )
    log(
        f"[OK] Geometry ready ({time.perf_counter() - t0:.1f}s) "
        f"nodes={int(data.x.shape[0])} wound={bool(getattr(data, 'customer_wound_enabled', False))}"
    )

    include_velocity = bool(
        arm.get("include_velocity", control.get("include_velocity", False))
    )
    env_overrides: dict[str, str] = {}
    for src in (control.get("env_overrides"), arm.get("env_overrides")):
        if isinstance(src, dict):
            env_overrides.update({str(k): str(v) for k, v in src.items()})

    model_norm = str(model).lower().strip()
    flow_use = str(arm.get("flow", control.get("flow", flow))).lower().strip()

    if model_norm in ("clot_ml_0", "clot_ml_v0"):
        traj, pack = run_clot_ml_0_arm(
            data=data,
            geom_spec=geom_spec,
            flow=flow_use,
            clot_model=clot_model,
            include_velocity=include_velocity,
            progress=log,
        )
        model_meta = {
            "resolver": "clot_ml_0",
            "clot_model": clot_model,
            "flow": flow_use,
        }
    else:
        raise ValueError(
            f"Unsupported model={model!r}. The legacy biochem / "
            "locked_canonical arm was removed; sweeps run clot_ml_0 on FEM t=0 flow."
        )

    arm_out = {
        "name": name,
        "axis_value": arm.get("axis_value"),
        "labels": arm.get("labels") or {},
        "geometry_spec": geom_spec,
        "env_overrides": env_overrides,
        "mesh_cache": str(mesh_pt),
        "model": model_meta,
        "rollout": {
            "elapsed_s": float(getattr(traj, "elapsed_s", 0.0)),
            "n_steps": int(traj.n_steps),
            "include_velocity": include_velocity,
            "t_final_s": float(geom_spec.get("t_final_s", 30000.0)),
        },
        "research_parameters": pack,
    }

    arm_json = out_dir / f"arm_{name}.json"
    arm_csv = out_dir / f"arm_{name}.csv"
    import json

    def _json_safe(obj: Any) -> Any:
        import numpy as np

        if isinstance(obj, float):
            if obj != obj:
                return None
            return obj
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            import torch

            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
        except ImportError:
            pass
        if isinstance(obj, dict):
            return {str(k): _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        return obj

    arm_json.write_text(
        json.dumps(_json_safe(arm_out), indent=2) + "\n", encoding="utf-8"
    )
    write_scientific_csv(arm_csv, pack["timeseries"])
    log(f"[save] {arm_json}")
    return arm_out


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float):
        if obj != obj:
            return None
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def run_sweep(
    cfg: dict[str, Any],
    *,
    pipeline: CustomerDeployPipeline | None = None,
    force_rebuild: bool = False,
    arm_filter: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run all arms in a normalized sweep config and write summary.json."""
    from pathlib import Path

    from src.evaluation.research_sweep_config import DEFAULT_RESEARCH_MODEL
    from src.evaluation.research_sweep_geometry import default_mesh_cache_dir
    from src.utils.paths import get_project_root

    log = progress or (lambda _msg: None)
    root = get_project_root()
    out_dir = Path(cfg.get("output_dir") or f"outputs/research_sweeps/{cfg['id']}")
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = default_mesh_cache_dir(root)

    model = str(cfg.get("model", DEFAULT_RESEARCH_MODEL))
    flow = str(cfg.get("control", {}).get("flow", "fem"))
    clot_model = str(cfg.get("clot_model", DEFAULT_RESEARCH_MODEL))
    control = dict(cfg.get("control") or {})

    arms = list(cfg["arms"])
    if arm_filter:
        arms = [a for a in arms if str(a.get("name")) == arm_filter]
        if not arms:
            raise ValueError(f"No arm named {arm_filter!r} in sweep {cfg.get('id')}")

    log(f"[i] Sweep {cfg.get('id')}: {len(arms)} arm(s) -> {out_dir}")
    log(f"[i] Model: {model}  flow={flow}  clot_model={clot_model}")

    # One bad arm used to raise straight out of the sweep, so an unattended run lost every
    # later arm and wrote no summary at all.  Keep going, but record what failed and re-raise
    # at the end -- a partial sweep must never read as a complete one.
    arm_results: list[dict[str, Any]] = []
    failed_arms: list[dict[str, str]] = []
    for arm in arms:
        try:
            arm_results.append(
                run_research_arm(
                    arm=arm,
                    control=control,
                    out_dir=out_dir,
                    cache_dir=cache_dir,
                    force_rebuild=force_rebuild,
                    model=model,
                    flow=flow,
                    clot_model=clot_model,
                    pipeline=pipeline,
                    progress=log,
                )
            )
        except Exception as exc:  # noqa: BLE001 -- reported below and re-raised
            import traceback

            traceback.print_exc()
            failed_arms.append({"name": str(arm.get("name")), "error": f"{type(exc).__name__}: {exc}"})
            log(f"[ERR] Arm {arm.get('name')} failed: {exc}")

    summary_rows = []
    for ar in arm_results:
        row = {
            "name": ar["name"],
            "axis_value": ar.get("axis_value"),
            **(ar.get("labels") or {}),
            **(ar.get("research_parameters", {}).get("summary") or {}),
        }
        summary_rows.append(row)

    summary = {
        "id": cfg.get("id"),
        "title": cfg.get("title"),
        "axis": cfg.get("axis"),
        "model": model,
        "flow": flow,
        "clot_model": clot_model if model == DEFAULT_RESEARCH_MODEL else None,
        "output_dir": str(out_dir.as_posix()),
        "n_arms": len(arm_results),
        "n_arms_requested": len(arms),
        "failed_arms": failed_arms,
        "arms": summary_rows,
    }

    summary_path = out_dir / "summary.json"
    import json

    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    if failed_arms:
        names = ", ".join(f["name"] for f in failed_arms)
        log(f"[PARTIAL] Sweep {cfg.get('id')}: {len(failed_arms)}/{len(arms)} arm(s) failed -> {summary_path}")
        raise RuntimeError(f"sweep {cfg.get('id')}: arm(s) failed: {names}")
    log(f"[OK] Sweep complete -> {summary_path}")
    return summary


__all__ = [
    "ClotMlResearchTrajectory",
    "run_clot_ml_0_arm",
    "run_research_arm",
    "run_sweep",
]
