"""Persist and resolve ``RGP_DEQ`` constructor kwargs from checkpoints and reference manifests.

Checkpoints written by ``train_kinematics_predictor`` embed ``model_config`` so
``train_biochem_gnn``, visualization, and agents can rebuild the same
architecture without guessing from env flags.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from src.architecture.ginodeq import GINO_DEQ, RGP_DEQ
from src.config import WIDTH_D1_MAX, WIDTH_D2_MAX
from src.utils.paths import data_root, get_project_root, kinematics_dir

# Set by the Stage-A arm scripts (`scripts/stage_a/run_*.sh`), the only setter and one
# outside the tree the knob sweep grepped -- so sweeping these to plain constants made
# every E-series arm a silent no-op for them.  Read from the environment with the swept
# value as the default: unset behaves exactly as the constant did.
KINEMATICS_BC_ENVELOPE = os.environ.get("KINEMATICS_BC_ENVELOPE", "0")
KINEMATICS_BC_ENVELOPE_DECAY = os.environ.get("KINEMATICS_BC_ENVELOPE_DECAY", "0.0")
KINEMATICS_BC_ENVELOPE_FLOOR = os.environ.get("KINEMATICS_BC_ENVELOPE_FLOOR", "0.0")
KINEMATICS_DECODER_SKIP = os.environ.get("KINEMATICS_DECODER_SKIP", "0")
KINEMATICS_FOURIER_LEARNABLE = "0"
KINEMATICS_RESIDUAL_GAIN = os.environ.get("KINEMATICS_RESIDUAL_GAIN", "0")
KINEMATICS_RESIDUAL_REZERO = os.environ.get("KINEMATICS_RESIDUAL_REZERO", "0")
KINEMATICS_USE_WIDTH_PRIORS = "1"
KINEMATICS_WSS_FUSE = "0"


KINEMATICS_MODEL_CONFIG_SCHEMA = 1
KINEMATICS_REFERENCE_REL = Path("reference") / "kinematics_best_20260426T184600Z.json"
KINEMATICS_ARCH_MANIFEST_NAME = "kinematics_architecture.json"

# Accepted model_class strings in checkpoint / reference JSON
_RGP_MODEL_CLASS_NAMES = frozenset({"RGP_DEQ", "GINO_DEQ", "PMGP_DEQ"})


def kinematics_reference_path() -> Path:
    override = os.environ.get("KINEMATICS_MODEL_CONFIG_REF", "").strip()
    if override:
        p = Path(override)
        return p if p.is_absolute() else get_project_root() / p
    return data_root() / KINEMATICS_REFERENCE_REL


def load_kinematics_reference_record() -> dict[str, Any] | None:
    path = kinematics_reference_path()
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw if isinstance(raw, dict) else None


def snapshot_rgp_deq_model_config(model: RGP_DEQ) -> dict[str, Any]:
    """Serializable constructor recipe for ``RGP_DEQ`` (schema v1)."""
    return {
        "schema": KINEMATICS_MODEL_CONFIG_SCHEMA,
        "model_class": "RGP_DEQ",
        "in_channels": 15,
        "out_channels": 5,
        "latent_dim": int(model.encoder[0].out_features),
        "max_iters": int(model.max_iters),
        "outer_iters": int(model.outer_iters),
        "num_fourier_freqs": int(model.num_fourier_freqs),
        "activation_fn": str(model.activation_fn),
        "fourier_base": float(model.fourier_base),
        "use_hard_bcs": bool(model.use_hard_bcs),
        "use_siren_decoder": bool(model.use_siren_decoder),
        "use_width_priors": bool(model.use_width_priors),
        "wss_fuse": bool(getattr(model, "wss_fuse", False)),
        "bc_envelope": bool(getattr(model, "bc_envelope", False)),
        "bc_lambda": float(getattr(model, "bc_lambda", 10.0)),
        "bc_envelope_decay": float(getattr(model, "bc_envelope_decay", 0.0)),
        "bc_envelope_floor": float(getattr(model, "bc_envelope_floor", 0.0)),
        "fourier_learnable": bool(getattr(model, "fourier_learnable", False)),
        "shear_head": bool(getattr(model, "shear_head", False)),
        "decoder_skip": bool(getattr(model, "decoder_skip", False)),
        "residual_gain": bool(getattr(model, "residual_gain", False)),
        "residual_rezero": bool(getattr(model, "residual_rezero", False)),
        "num_global_tokens": 16,
        "phase": "kinematics",
    }


# Backward-compatible alias
snapshot_gino_deq_model_config = snapshot_rgp_deq_model_config


def infer_use_siren_decoder_from_state_dict(state_dict: Mapping[str, Any]) -> bool | None:
    has_siren = any(str(k).startswith("siren_decoder.") for k in state_dict)
    has_linear = any(str(k).startswith("kinematics_decoder.") for k in state_dict)
    if has_siren:
        return True
    if has_linear:
        return False
    return None


def infer_latent_dim_from_state_dict(state_dict: Mapping[str, Any]) -> int | None:
    for key in ("encoder.0.weight", "kin_encoder.0.weight"):
        w = state_dict.get(key)
        if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
            return int(w.shape[0])
    return None


def infer_wss_fuse_from_state_dict(
    state_dict: Mapping[str, Any],
    *,
    latent_dim: int,
) -> bool | None:
    """True when WSS head was trained with ``z + (u,v,p,mu)`` (``latent_dim + 4`` inputs)."""
    w = state_dict.get("wss_decoder.0.linear.parametrizations.weight.original")
    if w is None or not hasattr(w, "shape") or len(w.shape) != 2:
        w = state_dict.get("wss_decoder.0.weight")
    if w is None or not hasattr(w, "shape") or len(w.shape) != 2:
        return None
    in_features = int(w.shape[1])
    if in_features == int(latent_dim) + 4:
        return True
    if in_features == int(latent_dim):
        return False
    return None


def infer_decoder_skip_from_state_dict(
    state_dict: Mapping[str, Any],
    *,
    latent_dim: int,
) -> bool | None:
    """True when the kinematics decoder was trained on ``[z*, x_enc]`` (``2 * latent_dim`` in).

    Read from the weight rather than trusted from the manifest: the decoder width is the one
    thing that makes a skip checkpoint physically incompatible with a non-skip rebuild, so it
    has to be recoverable from the tensors themselves.
    """
    for key in ("kinematics_decoder.weight", "siren_decoder.net.0.weight"):
        w = state_dict.get(key)
        if w is None or not hasattr(w, "shape") or len(w.shape) != 2:
            continue
        # The SIREN takes the latent concatenated with 2 coordinates.
        in_features = int(w.shape[1]) - (2 if key.startswith("siren_decoder") else 0)
        if in_features == 2 * int(latent_dim):
            return True
        if in_features == int(latent_dim):
            return False
    return None


def infer_residual_gain_from_state_dict(state_dict: Mapping[str, Any]) -> bool | None:
    """True/False from the tensors; ``None`` only when there are no tensors to read.

    The False case matters as much as the True one: a stale ``KINEMATICS_RESIDUAL_GAIN=1`` left
    exported in a shell must not add a head that the checkpoint being loaded cannot fill.  An
    EMPTY state dict is the fresh-model path, and there env is the right answer.
    """
    if not state_dict:
        return None
    return any(str(k).startswith("residual_gain_head.") for k in state_dict)


def infer_fourier_learnable_from_state_dict(state_dict: Mapping[str, Any]) -> bool | None:
    return True if "fourier_freqs" in state_dict else None


def infer_num_fourier_freqs_from_state_dict(
    state_dict: Mapping[str, Any],
    *,
    in_channels: int = 15,
    use_width_priors: bool | None = None,
) -> int | None:
    w = state_dict.get("encoder.0.weight") or state_dict.get("kin_encoder.0.weight")
    if w is None or not hasattr(w, "shape") or len(w.shape) != 2:
        return None
    if use_width_priors is None:
        use_width_priors = bool(int(KINEMATICS_USE_WIDTH_PRIORS))
    width_extra = 3 if use_width_priors else 0
    encoded_channels = int(w.shape[1])
    bands = (encoded_channels - int(in_channels) - width_extra) // 10
    return bands if bands >= 1 else None


def resolve_rgp_deq_ctor_kwargs(
    meta: Mapping[str, Any] | None,
    state_dict: Mapping[str, Any],
    *,
    latent_dim_default: int = 256,
    max_iters_default: int = 25,
    num_fourier_freqs_default: int = 16,
    use_siren_default: bool = True,
    use_hard_bcs_default: bool = True,
    use_width_priors_default: bool = True,
    fourier_base_default: float = 2.0,
    activation_fn_default: str = "silu",
) -> dict[str, Any]:
    """Build ``RGP_DEQ`` kwargs from checkpoint metadata, reference JSON, or tensor shapes."""
    saved = dict((meta or {}).get("model_config") or {})
    if int(saved.get("schema", 0)) != KINEMATICS_MODEL_CONFIG_SCHEMA:
        ref = load_kinematics_reference_record()
        if ref:
            saved = dict(ref.get("model_config") or saved)

    def _merge_kinematics_toggle_flags(ctor: dict[str, Any]) -> dict[str, Any]:
        latent = int(ctor["latent_dim"])
        if "wss_fuse" not in ctor or ctor.get("wss_fuse") is None:
            inferred_wss = infer_wss_fuse_from_state_dict(state_dict, latent_dim=latent)
            ctor["wss_fuse"] = (
                inferred_wss
                if inferred_wss is not None
                else bool(int(KINEMATICS_WSS_FUSE))
            )
        if "fourier_learnable" not in ctor or ctor.get("fourier_learnable") is None:
            inferred_fl = infer_fourier_learnable_from_state_dict(state_dict)
            ctor["fourier_learnable"] = (
                inferred_fl
                if inferred_fl is not None
                else bool(int(KINEMATICS_FOURIER_LEARNABLE))
            )
        if "bc_envelope" not in ctor or ctor.get("bc_envelope") is None:
            ctor["bc_envelope"] = bool(int(KINEMATICS_BC_ENVELOPE))
        if ctor.get("bc_lambda") is None:
            ctor["bc_lambda"] = float(os.environ.get("KINEMATICS_BC_LAMBDA", "10.0"))
        if ctor.get("bc_envelope_decay") is None:
            ctor["bc_envelope_decay"] = float(
                KINEMATICS_BC_ENVELOPE_DECAY)
        if ctor.get("bc_envelope_floor") is None:
            ctor["bc_envelope_floor"] = float(KINEMATICS_BC_ENVELOPE_FLOOR)
        if "shear_head" not in ctor or ctor.get("shear_head") is None:
            ctor["shear_head"] = any(str(k).startswith("shear_decoder.") for k in state_dict)
        # The tensors outrank the manifest for both of these: the decoder width and the presence
        # of the gain head are what make a rebuild load or fail, so a stale manifest must not be
        # able to describe a model the state dict cannot fill.
        inferred_skip = infer_decoder_skip_from_state_dict(state_dict, latent_dim=latent)
        if inferred_skip is not None:
            ctor["decoder_skip"] = inferred_skip
        elif "decoder_skip" not in ctor or ctor.get("decoder_skip") is None:
            ctor["decoder_skip"] = bool(int(KINEMATICS_DECODER_SKIP))
        # The tensor is the truth here too: `residual_scale` exists in the state dict iff the
        # model was built with the flag, so a manifest cannot describe a model that will not load.
        has_rezero = any(str(k) == "residual_scale" for k in state_dict)
        if has_rezero or ctor.get("residual_rezero") is None:
            ctor["residual_rezero"] = bool(has_rezero) if has_rezero else bool(
                int(KINEMATICS_RESIDUAL_REZERO)
            )
        inferred_gain = infer_residual_gain_from_state_dict(state_dict)
        ctor["residual_gain"] = bool(inferred_gain) if inferred_gain is not None else bool(
            int(KINEMATICS_RESIDUAL_GAIN)
        )
        return ctor

    if int(saved.get("schema", 0)) == KINEMATICS_MODEL_CONFIG_SCHEMA:
        ctor = {
            "in_channels": int(saved.get("in_channels", 15)),
            "out_channels": int(saved.get("out_channels", 5)),
            "latent_dim": max(8, int(saved.get("latent_dim", latent_dim_default))),
            "max_iters": max(3, int(saved.get("max_iters", max_iters_default))),
            "outer_iters": max(1, int(saved.get("outer_iters", 3))),
            "num_fourier_freqs": max(1, int(saved.get("num_fourier_freqs", num_fourier_freqs_default))),
            "activation_fn": str(saved.get("activation_fn", activation_fn_default)),
            "fourier_base": float(saved.get("fourier_base", fourier_base_default)),
            "use_hard_bcs": bool(saved.get("use_hard_bcs", use_hard_bcs_default)),
            "use_siren_decoder": bool(saved.get("use_siren_decoder", use_siren_default)),
            "use_width_priors": bool(saved.get("use_width_priors", use_width_priors_default)),
            "num_global_tokens": max(1, int(saved.get("num_global_tokens", 16))),
        }
        if "wss_fuse" in saved:
            ctor["wss_fuse"] = bool(saved["wss_fuse"])
        if "bc_envelope" in saved:
            ctor["bc_envelope"] = bool(saved["bc_envelope"])
        if "bc_lambda" in saved:
            ctor["bc_lambda"] = float(saved["bc_lambda"])
        if "bc_envelope_decay" in saved:
            ctor["bc_envelope_decay"] = float(saved["bc_envelope_decay"])
        if "bc_envelope_floor" in saved:
            ctor["bc_envelope_floor"] = float(saved["bc_envelope_floor"])
        if "fourier_learnable" in saved:
            ctor["fourier_learnable"] = bool(saved["fourier_learnable"])
        if "decoder_skip" in saved:
            ctor["decoder_skip"] = bool(saved["decoder_skip"])
        if "residual_gain" in saved:
            ctor["residual_gain"] = bool(saved["residual_gain"])
        if "residual_rezero" in saved:
            ctor["residual_rezero"] = bool(saved["residual_rezero"])
        return _merge_kinematics_toggle_flags(ctor)

    inferred_siren = infer_use_siren_decoder_from_state_dict(state_dict)
    inferred_latent = infer_latent_dim_from_state_dict(state_dict)
    inferred_fourier = infer_num_fourier_freqs_from_state_dict(state_dict)
    return _merge_kinematics_toggle_flags(
        {
            "in_channels": 15,
            "out_channels": 5,
            "latent_dim": inferred_latent if inferred_latent is not None else latent_dim_default,
            "max_iters": max_iters_default,
            "outer_iters": 3,
            "num_fourier_freqs": inferred_fourier if inferred_fourier is not None else num_fourier_freqs_default,
            "activation_fn": activation_fn_default,
            "fourier_base": fourier_base_default,
            "use_hard_bcs": use_hard_bcs_default,
            "use_siren_decoder": inferred_siren if inferred_siren is not None else use_siren_default,
            "use_width_priors": use_width_priors_default,
            "num_global_tokens": 16,
        }
    )


def kinematics_checkpoint_tensors(raw: Any) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Unpack nested kinematics ``.pth`` (``model_state_dict`` + metadata)."""
    if isinstance(raw, dict) and "model_state_dict" in raw:
        meta = {k: v for k, v in raw.items() if k != "model_state_dict"}
        state = raw["model_state_dict"]
        if isinstance(state, dict):
            return meta, state
    if isinstance(raw, dict):
        return {}, raw
    raise TypeError(f"Unsupported kinematics checkpoint type: {type(raw)!r}")


def build_rgp_deq_from_ctor(phys_cfg: Any, ctor: Mapping[str, Any]) -> RGP_DEQ:
    return RGP_DEQ(
        in_channels=int(ctor["in_channels"]),
        out_channels=int(ctor["out_channels"]),
        latent_dim=int(ctor["latent_dim"]),
        max_iters=int(ctor["max_iters"]),
        num_fourier_freqs=int(ctor["num_fourier_freqs"]),
        outer_iters=int(ctor.get("outer_iters", 3)),
        phys_cfg=phys_cfg,
        activation_fn=str(ctor.get("activation_fn", "silu")),
        fourier_base=float(ctor.get("fourier_base", 2.0)),
        use_hard_bcs=bool(ctor.get("use_hard_bcs", True)),
        num_global_tokens=int(ctor.get("num_global_tokens", 16)),
        use_siren_decoder=bool(ctor.get("use_siren_decoder", True)),
        use_width_priors=bool(ctor.get("use_width_priors", True)),
        wss_fuse=bool(ctor["wss_fuse"]) if "wss_fuse" in ctor else None,
        bc_envelope=bool(ctor["bc_envelope"]) if "bc_envelope" in ctor else None,
        bc_lambda=float(ctor["bc_lambda"]) if ctor.get("bc_lambda") is not None else None,
        bc_envelope_decay=(float(ctor["bc_envelope_decay"])
                           if ctor.get("bc_envelope_decay") is not None else None),
        fourier_learnable=bool(ctor["fourier_learnable"]) if "fourier_learnable" in ctor else None,
        shear_head=bool(ctor["shear_head"]) if "shear_head" in ctor else True,
        decoder_skip=bool(ctor["decoder_skip"]) if "decoder_skip" in ctor else None,
        residual_gain=bool(ctor["residual_gain"]) if "residual_gain" in ctor else None,
        residual_rezero=bool(ctor["residual_rezero"]) if "residual_rezero" in ctor else None,
    )


# Backward-compatible aliases
resolve_gino_deq_ctor_kwargs = resolve_rgp_deq_ctor_kwargs
build_gino_deq_from_ctor = build_rgp_deq_from_ctor


def save_kinematics_checkpoint_file(
    path: Path | str,
    model: nn.Module,
    *,
    checkpoint_role: str,
    best_epoch: int = -1,
    rel_l2: float = float("nan"),
    continuity: float = float("nan"),
    composite: float = float("nan"),
    run_id: str = "",
    run_note: str = "",
    training_manifest: Mapping[str, Any] | None = None,
    prior_source: str = "",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_config = snapshot_rgp_deq_model_config(model) if isinstance(model, RGP_DEQ) else None
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "checkpoint_role": checkpoint_role,
        "best_epoch": int(best_epoch),
        "epoch": int(best_epoch),
        "rel_l2": float(rel_l2),
        "continuity": float(continuity),
        "composite": float(composite),
        "best_val_composite_loss": float(composite),
        "run_id": (run_id or "").strip(),
        "run_note": (run_note or "").strip(),
        # RGP_DEQ_REPAIR_PLAN.md B2/D6: the prior block a checkpoint was TRAINED with, so
        # `precache_rgp_deq` can refuse a train/deploy mismatch instead of discovering it
        # months later as an unexplained score collapse.
        "prior_source": (prior_source or "").strip().lower(),
        # The width-derivative clamp is a property of the checkpoint's TRAINING CORPUS, not a
        # global constant: `clamped_width_priors` uses it to keep deploy inputs inside the range
        # the weights were fitted on.  Moving the corpus moved it 4.14/73.8 -> 15.56/282.3, so a
        # checkpoint that does not record its own bounds gets silently re-interpreted the next
        # time the constant changes.  Recorded here so it travels with the weights.
        "width_clamp": [float(WIDTH_D1_MAX), float(WIDTH_D2_MAX)],
    }
    if model_config:
        payload["model_config"] = model_config
    if training_manifest:
        payload["training_manifest"] = dict(training_manifest)
    torch.save(payload, path)


def write_kinematics_architecture_manifest(
    ctor: Mapping[str, Any],
    *,
    best_epoch: int = -1,
    rel_l2: float = float("nan"),
    continuity: float = float("nan"),
    composite: float = float("nan"),
    run_id: str = "",
    checkpoint_role: str = "kinematics_best",
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Sidecar JSON next to checkpoints for agents (no torch.load required)."""
    out_dir = kinematics_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / KINEMATICS_ARCH_MANIFEST_NAME
    body: dict[str, Any] = {
        "schema": KINEMATICS_MODEL_CONFIG_SCHEMA,
        "checkpoint_role": checkpoint_role,
        "best_epoch": int(best_epoch),
        "rel_l2": float(rel_l2),
        "continuity": float(continuity),
        "composite": float(composite),
        "run_id": (run_id or "").strip(),
        "model_config": dict(ctor),
    }
    if extra:
        body.update(dict(extra))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2)
        f.write("\n")
    return path
