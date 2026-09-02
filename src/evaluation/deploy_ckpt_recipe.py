"""Deploy-faithful checkpoint recipe binding for biochem / mat-growth eval.

Extracted from retired ``scripts/eval_mat_growth_simple.py`` so diagnostics and
unit tests can bind ``PushforwardConfig`` / ``BiochemRuntimeConfig`` without the
full mat-growth eval script.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

_EVAL_PF_CM = None
_EVAL_RT_CM = None
_EVAL_GELATION_BETA = ""


def set_eval_gelation_beta(beta: str) -> None:
    global _EVAL_GELATION_BETA
    _EVAL_GELATION_BETA = str(beta or "")


def bind_eval_typed_configs(pf, rt) -> None:
    """Keep PushforwardConfig / BiochemRuntimeConfig active for the eval process."""
    global _EVAL_PF_CM, _EVAL_RT_CM
    from src.architecture.pushforward_config import use_pushforward_config
    from src.architecture.runtime_config import use_biochem_runtime

    if _EVAL_PF_CM is not None:
        try:
            _EVAL_PF_CM.__exit__(None, None, None)
        except Exception:
            pass
        _EVAL_PF_CM = None
    if _EVAL_RT_CM is not None:
        try:
            _EVAL_RT_CM.__exit__(None, None, None)
        except Exception:
            pass
        _EVAL_RT_CM = None
    _EVAL_PF_CM = use_pushforward_config(pf)
    _EVAL_PF_CM.__enter__()
    _EVAL_RT_CM = use_biochem_runtime(rt)
    _EVAL_RT_CM.__enter__()


def load_eval_static(data, device, kine_model, wall_hops: int, anchor: str) -> dict:
    """One joint RGP-DEQ solve per vessel; bake u0_pred + z_kin into pack features."""
    from src.core_physics.species_pushforward_gnn import build_band_base_features
    from src.utils.kinematics_inference import predict_kinematics_and_latent
    from src.utils.paths import get_project_root

    kine_stem = (
        Path(kine_model.config.ckpt_path).stem
        if getattr(kine_model, "config", None) and hasattr(kine_model.config, "ckpt_path")
        else "deploy"
    )
    cache_dir = get_project_root() / ".cache" / "kinematics_t0" / kine_stem

    with torch.no_grad():
        pred_uv, z_kin = predict_kinematics_and_latent(
            kine_model,
            data,
            disk_cache_dir=cache_dir,
            disk_cache_key=anchor.strip(),
        )
    data.u0_pred = pred_uv[:, 0].detach().to(device="cpu").clone()
    data.v0_pred = pred_uv[:, 1].detach().to(device="cpu").clone()
    return build_band_base_features(
        data, kine_model, device, wall_hops=wall_hops, z_kin_override=z_kin
    )


def apply_deploy_ckpt_recipe(
    meta: dict,
    *,
    label: str,
    ckpt_path: Path | str | None = None,
    pf_overrides: dict[str, object] | None = None,
) -> None:
    """Bind typed train/deploy configs from checkpoint meta (architecture + runtime)."""
    from dataclasses import replace

    from src.architecture.pushforward_config import (
        PushforwardConfig,
        split_legacy_env_overrides,
    )
    from src.architecture.runtime_config import (
        BiochemRuntimeConfig,
        split_legacy_runtime_env,
    )
    from src.biochem_gnn.config import GLOBAL_TRAIN_RECIPE

    scope = meta.get("pushforward_species_scope") or meta.get("species_scope")
    recipe_env: dict[str, str] = dict(GLOBAL_TRAIN_RECIPE)
    if label == "mat_growth_simple" or scope == "mat":
        from src.biochem_gnn.mat_growth_simple import MAT_GROWTH_SIMPLE_RECIPE

        recipe_env.update({k: str(v) for k, v in MAT_GROWTH_SIMPLE_RECIPE.items()})
    recipe_env.pop("SPECIES_FLOW_FEATS_SOURCE", None)

    residual_env: dict[str, str] = {}
    overrides = meta.get("env_overrides")
    if isinstance(overrides, dict) and overrides:
        recipe_env = {**recipe_env, **{k: str(v) for k, v in overrides.items()}}
        recipe_env.pop("SPECIES_FLOW_FEATS_SOURCE", None)
    elif ckpt_path is not None:
        path_s = str(ckpt_path).replace("\\", "/")
        if "mat_growth_ladder/" in path_s:
            parts = path_s.split("mat_growth_ladder/")
            if len(parts) > 1:
                leg = parts[1].split("/")[0]
                if leg:
                    try:
                        from src.biochem_gnn.mat_growth_simple import mat_growth_leg_spec

                        spec = mat_growth_leg_spec(leg)
                        recipe_env.update({k: str(v) for k, v in spec.env_overrides.items()})
                        meta = {
                            **meta,
                            "config_kwargs": {
                                **dict(spec.config_kwargs),
                                **dict(meta.get("config_kwargs") or {}),
                            },
                            "runtime_kwargs": {
                                **dict(spec.runtime_kwargs),
                                **dict(meta.get("runtime_kwargs") or {}),
                            },
                        }
                    except Exception as e:
                        print(f"[WARN] Failed to apply leg typed config for {leg} from path: {e}")

    merged_meta = {**meta, "env_overrides": recipe_env if recipe_env else meta.get("env_overrides")}
    pf = PushforwardConfig.from_meta(merged_meta)
    pf = replace(pf, flow_feats_source="auto")
    if pf_overrides:
        pf = replace(pf, **pf_overrides)

    rt = BiochemRuntimeConfig.from_meta(merged_meta).with_overrides(
        deploy_faithful=True,
        rollout_vel_source="kinematics",
        rollout_pin_other="rest",
        rollout_ic_source="resting",
        closed_loop_coupling=True,
        train_deploy_eval_flow="auto",
    )
    if _EVAL_GELATION_BETA:
        rt = rt.with_overrides(beta_override=_EVAL_GELATION_BETA)

    prev_rt = None
    try:
        from src.architecture.runtime_config import get_active_runtime

        prev_rt = get_active_runtime()
    except Exception:
        pass
    if prev_rt is not None and prev_rt.offwall.two_model_mode:
        rt = rt.with_overrides(
            two_model_mode=True,
            offwall_model_ckpt=prev_rt.offwall.offwall_model_ckpt,
            two_model_route=prev_rt.offwall.two_model_route,
            two_model_frontier_hops=prev_rt.offwall.two_model_frontier_hops,
            frontier_hops_map=prev_rt.offwall.frontier_hops_map,
            frontier_hops_anchor=prev_rt.offwall.frontier_hops_anchor,
        )
    bind_eval_typed_configs(pf, rt)

    if recipe_env:
        _cfg, rem1 = split_legacy_env_overrides(recipe_env)
        _rt, rem2 = split_legacy_runtime_env(rem1)
        residual_env.update(rem2)
    for k, v in residual_env.items():
        os.environ[k] = str(v)
    os.environ.pop("SPECIES_FLOW_FEATS_SOURCE", None)


# Back-compat aliases for archived scripts/tests.
_apply_ckpt_recipe = apply_deploy_ckpt_recipe
_load_static = load_eval_static
