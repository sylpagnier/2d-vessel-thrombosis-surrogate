"""Shared RGP-DEQ (Stage-A) checkpoint resolution and inference.

Flow model = ``RGP_DEQ`` class (canonical id ``rgp_deq_kine`` / RGP-DEQ).
Prefer ``load_rgp_deq_kine`` / ``resolve_rgp_deq_kine_ckpt``; legacy ``load_gino_deq_kine`` aliases retained.

Inference helpers share one DEQ solve per graph: UV predictions and ``z_kin`` are cached together
so pack-build / coupling paths never pay for a second Anderson solve on the same vessel.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

from src.architecture.ginodeq import GINO_DEQ, RGP_DEQ
from src.architecture.kinematics_model_config import (
    build_rgp_deq_from_ctor,
    kinematics_checkpoint_tensors,
    resolve_rgp_deq_ctor_kwargs,
)
from src.config import NodeFeat, PhysicsConfig
from src.utils.paths import resolve_checkpoint

KINEMATICS_CKPT_CANDIDATES = (
    "kinematics_best.pth",
    "kinematics_ckpt_latest.pth",
    "kinematics_ckpt_100.pth",
)


def resolve_kinematics_checkpoint(explicit: Path | str | None = None) -> Path:
    """Return an existing kinematics checkpoint path (explicit or search candidates).

    Training runs write into ``KINEMATICS_OUTPUT_DIR`` subdirectories, but this resolver only
    ever looked at the stage-A root -- so a freshly trained model is invisible to every deploy
    path unless it is passed explicitly.  On 2026-08-27 two runs finished into
    ``outputs/kinematics/{production_allfix,comsol_anchor_finetune}/`` while every consumer
    kept loading the 2026-08-12 root checkpoint (RGP_DEQ_REPAIR_PLAN.md B7).

    The resolution order is unchanged -- silently switching to a subdirectory checkpoint would
    be its own surprise -- but newer candidates are now *named* so the mismatch is visible at
    the point of use rather than months later.
    """
    if explicit is not None:
        path = Path(explicit)
        if path.is_file():
            return path.resolve()
        candidate = resolve_checkpoint("a", path.name)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Kinematics checkpoint not found: {explicit}")

    for ckpt_name in KINEMATICS_CKPT_CANDIDATES:
        candidate = resolve_checkpoint("a", ckpt_name)
        if candidate.exists():
            _warn_newer_run_checkpoints(candidate)
            return candidate

    expected_dir = resolve_checkpoint("a", KINEMATICS_CKPT_CANDIDATES[0]).parent
    raise FileNotFoundError(
        "No kinematics checkpoint found. Tried: "
        + ", ".join(str(expected_dir / name) for name in KINEMATICS_CKPT_CANDIDATES)
    )


def _warn_newer_run_checkpoints(chosen: Path) -> None:
    """Name any run-directory checkpoint newer than the one actually being loaded."""
    try:
        root = chosen.parent
        chosen_mtime = chosen.stat().st_mtime
        newer = [
            p
            for sub in root.iterdir() if sub.is_dir()
            for p in sub.glob("kinematics_*.pth")
            if p.stat().st_mtime > chosen_mtime
        ]
    except OSError:
        return
    if not newer:
        return
    print(
        f"[kine] NOTE loading {chosen.name} from {chosen.parent}, but "
        f"{len(newer)} newer checkpoint(s) exist in run subdirectories: "
        + ", ".join(str(p.relative_to(chosen.parent)) for p in sorted(newer)[:5])
        + ". Pass one explicitly if that is the model you meant."
    )


def assert_promotable_checkpoint(path: Path | str) -> dict:
    """Reject a checkpoint that carries no evidence it was ever selected.

    RGP_DEQ_REPAIR_PLAN.md D7.  A periodic/latest checkpoint records the weights but not a
    passed promotion gate; copying one over ``kinematics_best.pth`` produces an artifact that
    looks promoted and is not.  Both 2026-08-27 checkpoints are exactly that:
    ``checkpoint_role='kinematics_ckpt_latest'`` with ``rel_l2/continuity/composite = nan``.
    """
    import math

    p = Path(path)
    raw = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: model-only checkpoint carries no promotion metadata")
    role = str(raw.get("checkpoint_role", ""))
    comp = float(raw.get("composite", float("nan")))
    problems = []
    if role != "kinematics_best":
        problems.append(f"checkpoint_role={role!r} (expected 'kinematics_best')")
    if not math.isfinite(comp):
        problems.append("composite is NaN -- no validation backed this checkpoint")
    if not str(raw.get("run_id", "")).strip():
        problems.append("run_id is empty -- no run provenance")
    if problems:
        raise ValueError(f"{p} is not promotable: " + "; ".join(problems))
    return {k: raw.get(k) for k in
            ("checkpoint_role", "best_epoch", "rel_l2", "continuity", "composite", "run_id",
             "prior_source")}


def _load_torch_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


# Session cache: eval/viz reload the same Stage-A ckpt many times; training should clear after pack build.
_KINE_MODEL_CACHE: dict[tuple[str, str, int], RGP_DEQ] = {}


def clear_kinematics_predictor_cache() -> None:
    """Drop cached RGP-DEQ handles (frees VRAM once callers drop their refs)."""
    _KINE_MODEL_CACHE.clear()


def load_kinematics_predictor(
    checkpoint: Path | str,
    device: torch.device | str,
    *,
    phys_cfg: PhysicsConfig | None = None,
    max_iters: int = 25,
    cache: bool = True,
) -> RGP_DEQ:
    """Load RGP-DEQ from a kinematics checkpoint with training-default architecture.

    When ``cache=True`` (default), the same ``(ckpt, device, max_iters)`` returns the same
    eval-mode module so multi-vessel eval/viz does not reload weights from disk each time.
    Training pack-build should pass ``cache=False`` or call :func:`clear_kinematics_predictor_cache`
    after features are baked, so VRAM can be released.
    """
    ckpt_path = Path(checkpoint).resolve()
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    cache_key = (str(ckpt_path), str(dev), max(3, int(max_iters)))
    if cache and cache_key in _KINE_MODEL_CACHE:
        return _KINE_MODEL_CACHE[cache_key]

    raw = _load_torch_checkpoint(ckpt_path)
    meta, state = kinematics_checkpoint_tensors(raw)
    ctor = resolve_rgp_deq_ctor_kwargs(meta, state)
    ctor["max_iters"] = max(3, int(max_iters))
    cfg = phys_cfg or PhysicsConfig(phase="kinematics")
    model = build_rgp_deq_from_ctor(cfg, ctor).to(dev)
    model.load_state_dict(state, strict=False)
    model.eval()
    if cache:
        _KINE_MODEL_CACHE[cache_key] = model
    return model


def _kin_solver_kwargs() -> dict[str, object]:
    solver = VIZ_KIN_SOLVER.strip().lower() or "anderson"
    beta = float(VIZ_KIN_ANDERSON_BETA)
    warmup = int(VIZ_KIN_ANDERSON_WARMUP)
    return {
        "solver": solver,
        "anderson_beta": beta,
        "anderson_warmup_iters": max(0, warmup),
    }


def _graph_key(data) -> tuple[int, int, int]:
    n = int(data.num_nodes)
    e = int(data.edge_index.shape[1])
    ptr = 0
    if hasattr(data, "x") and torch.is_tensor(data.x) and data.x.numel() > 0:
        ptr = int(data.x.untyped_storage().data_ptr())
    return (n, e, ptr)


def _cache_hit(model: RGP_DEQ, key: tuple[int, int, int]) -> bool:
    return getattr(model, "_cache_key", None) == key


def _store_joint_cache(model: RGP_DEQ, key: tuple[int, int, int], pred: torch.Tensor, z: torch.Tensor) -> None:
    model._cache_key = key
    model._cache_pred = pred
    model._cache_latent = z


#: Largest ``|width_d1|`` / ``|width_d2|`` the Stage-A checkpoint was trained against --
#: the 95th percentile of the per-vessel maximum over ``graphs_kinematics/carreau`` (n=40).
#: Re-exported from `src.config` -- one definition, shared with the encoder's own clamp.
from src.config import WIDTH_D1_MAX, WIDTH_D2_MAX  # noqa: E402,F401

# Former environment overrides that nothing in the tree ever set and no doc
# named, so each always resolved to the value below.  Kept as named constants
# rather than inlined literals so the value stays greppable and explainable.
VIZ_KIN_ANDERSON_BETA = "0.8"
VIZ_KIN_ANDERSON_WARMUP = "5"
VIZ_KIN_SOLVER = "anderson"



@contextlib.contextmanager
def clamped_width_priors(data: Data, *, d1_max: float = WIDTH_D1_MAX, d2_max: float = WIDTH_D2_MAX):
    """Hold ``width_d1``/``width_d2`` inside the range the checkpoint was trained on.

    **Why this is not cosmetic.**  ``width_d1 = G @ width_nd`` and ``width_d2 = G @ width_d1``
    apply the stored WLS gradient operator twice.  COMSOL exports ``triangle6``, so 74.5% of
    biochem graph nodes are P2 mid-side nodes of degree 2, and a 2nd-order WLS fitted from two
    COLLINEAR neighbours is rank-deficient -- ``precompute_wls_operators`` regularises with
    ``M + 1e-6*I`` before ``pinv(rcond=1e-5)``, which lifts the null directions just above the
    truncation cut so they get inverted instead of dropped.  On 34 of 52 packs that leaves
    mid-side rows with three near-cancelling coefficients (row norms to 3296 against the
    training set's max of 83) and ``|width_d2|`` reaching 1.8e5.  The 18 packs built from a
    corner-only edge list are unaffected: their mid-side rows are a single zero.

    Those values are the ONLY unnormalised inputs the model has -- ``_apply_fourier_encoding``
    appends the three width channels raw to ``Linear(178, 256)``.  Measured 2026-08-23:
    encoder latents reach ``|z| = 64,387`` against a median of 27, those nodes then take
    **76-90%** of the Perceiver global-token read mass (10% clamped), and the poisoned tokens
    are broadcast back to every node with ``x_enc`` re-injected at each of the 25 Anderson
    iterations.  So 0.01-0.6% of the mesh destroys the whole field: rel L2 vs COMSOL t=0
    **0.375 -> 0.138** with this clamp (affected vessels 0.448 -> 0.150), against the
    checkpoint's own recorded benchmark of 0.1007.  Cross-section flux CV returns to GT's
    0.147 exactly.

    An input already inside the range is passed through untouched.  The 18 corner-edge packs
    still trip the ``d1`` bound (comsol001 reads 6.90 against 4.14) and clamping them is
    measurably inert -- rel L2 0.130 -> 0.131 -- so the operation only ever bites where there
    is something to fix.  Restores ``data.x`` on exit: callers, and ``_graph_key``, keep the
    tensor they passed in.
    """
    x = getattr(data, "x", None)
    if not torch.is_tensor(x) or x.dim() != 2 or x.shape[1] < NodeFeat.WIDTH_D2.stop:
        yield data
        return
    d1, d2 = x[:, NodeFeat.WIDTH_D1], x[:, NodeFeat.WIDTH_D2]
    if float(d1.abs().max()) <= d1_max and float(d2.abs().max()) <= d2_max:
        yield data
        return
    clamped = x.clone()
    clamped[:, NodeFeat.WIDTH_D1] = d1.clamp(-d1_max, d1_max)
    clamped[:, NodeFeat.WIDTH_D2] = d2.clamp(-d2_max, d2_max)
    data.x = clamped
    try:
        yield data
    finally:
        data.x = x


def _run_joint_solve(model: RGP_DEQ, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """One Anderson solve on ``data``; returns ``(pred, z_kin)``."""
    orig_device = next(model.parameters()).device
    kwargs = _kin_solver_kwargs()
    if orig_device.type == "cuda":
        moved = data.to(orig_device)
        try:
            with clamped_width_priors(moved) as g:
                return model.predict_uv_and_latent(g, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            try:
                moved = data.to(orig_device)
                with clamped_width_priors(moved) as g:
                    return model.predict_uv_and_latent(g, **kwargs)
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError(
                    "predict_kinematics_and_latent OOM on CUDA. Silent fallbacks to CPU are "
                    "disabled by Hardware Execution Policy to prevent hangs."
                ) from e
    with clamped_width_priors(data) as g:
        return model.predict_uv_and_latent(g, **kwargs)


@torch.no_grad()
def predict_kinematics_and_latent(
    model: RGP_DEQ,
    data: Data,
    *,
    disk_cache_dir: Path | None = None,
    disk_cache_key: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One GINO-DEQ solve; returns ``(pred [N, C], z_kin [N, latent_dim])`` and fills both caches."""
    key = _graph_key(data)
    if (
        _cache_hit(model, key)
        and getattr(model, "_cache_pred", None) is not None
        and getattr(model, "_cache_latent", None) is not None
    ):
        return model._cache_pred, model._cache_latent

    cache_path = None
    if disk_cache_dir is not None and disk_cache_key is not None:
        cache_path = disk_cache_dir / f"{disk_cache_key}.pt"
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            pred = payload["pred_uv"]
            z = payload["z_kin"]
            device = next(model.parameters()).device
            pred, z = pred.to(device), z.to(device)
            _store_joint_cache(model, key, pred, z)
            return pred, z

    pred, z = _run_joint_solve(model, data)
    _store_joint_cache(model, key, pred, z)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "pred_uv": pred.detach().cpu(),
            "z_kin": z.detach().cpu(),
        }, cache_path)

    return pred, z


def predict_kinematics(model: RGP_DEQ, data: Data) -> torch.Tensor:
    """Run one RGP-DEQ forward pass; returns ``(N, C)`` predictions (joint-caches ``z_kin``)."""
    key = _graph_key(data)
    if _cache_hit(model, key) and getattr(model, "_cache_pred", None) is not None:
        return model._cache_pred
    pred, _ = predict_kinematics_and_latent(model, data)
    return pred


@torch.no_grad()
def predict_kinematics_latent(model: RGP_DEQ, data: Data) -> torch.Tensor:
    """Frozen DEQ latent ``z_kin`` per node, shape ``[N, latent_dim]`` (joint-caches UV pred)."""
    key = _graph_key(data)
    if _cache_hit(model, key) and getattr(model, "_cache_latent", None) is not None:
        return model._cache_latent
    _, z = predict_kinematics_and_latent(model, data)
    return z


# Canonical names (RGP-DEQ Stage-A flow)
resolve_rgp_deq_kine_ckpt = resolve_kinematics_checkpoint
load_rgp_deq_kine = load_kinematics_predictor
predict_rgp_deq_flow = predict_kinematics
predict_rgp_deq_latent = predict_kinematics_latent
predict_rgp_deq_flow_and_latent = predict_kinematics_and_latent
# Legacy PMGP / GINO aliases
resolve_pmgp_deq_kine_ckpt = resolve_rgp_deq_kine_ckpt
load_pmgp_deq_kine = load_rgp_deq_kine
predict_pmgp_deq_flow = predict_rgp_deq_flow
predict_pmgp_deq_latent = predict_rgp_deq_latent
predict_pmgp_deq_flow_and_latent = predict_rgp_deq_flow_and_latent
resolve_gino_deq_kine_ckpt = resolve_rgp_deq_kine_ckpt
load_gino_deq_kine = load_rgp_deq_kine
predict_gino_deq_flow = predict_rgp_deq_flow
predict_gino_deq_latent = predict_rgp_deq_latent
predict_gino_deq_flow_and_latent = predict_rgp_deq_flow_and_latent
