"""Unified clot-ML deploy stack: one artifact for wounded and non-wounded vessels.

WHY THIS MODULE.  ``clot_gnn_v5`` / ``clot_gnn_v5w`` were two classes of model -- a
wound-free ensemble and a wound complement wrapping it.  The measurements in
docs/WOUND_PROGRESS.md 17-18 say the right deploy object is a *composition*, not a
retrain:

    wall SET + timing     the C0 GNN ensemble (already 0.92 / 0.71 on the cohort)
    wound boundary        two-regime ``(G_pre, G_post)``; structural no-op without a mask
    wound-local lumen     optionally REPLACE the GNN verdict with chemistry-ODE Mat through
                          solid-anchored replace+depth (att=0.23, depth=3), while keeping
                          the GNN verdict in the far lumen
    off-wall otherwise    keep the GNN -- ODE Mat through the same rule scores ~0.40
    003-like chemistry    upwind AP renewal + ``da_scale_auto=123`` feeding that ODE;
                          optional ClotGNN residual on AP when a checkpoint exists

On a pack with no wound this returns the base GNN's own output unchanged, which is the
property that lets one artifact supersede both classes.  The GNN's temporal ODE is NOT
replaced: the chemistry integration is a second, off-wall-only field.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
LOCKED = REPO / "outputs" / "clot_ml" / "locked"
DEFAULT_NAME = "clot_ml_0"
LEGACY_NAMES = frozenset({"clot_ml_v0"})
KIND = "unified_v0"


#: The locked pointer.  Same file `src.clot_ml.locked` follows for the GNN generations.
POINTER = REPO / "data/reference/clot_gnn_locked.json"


def pointer_v0_name() -> str | None:
    """The ``unified_v0`` artifact the locked pointer currently names, or ``None``.

    Returns ``None`` rather than raising when the pointer is missing, unreadable, or names a
    generation of a different kind, so an ordinary checkout without a promoted artifact keeps
    working on the compiled-in default.
    """
    import json

    try:
        ptr = json.loads(POINTER.read_text())
    except (OSError, ValueError):
        return None
    if ptr.get("kind") != KIND:
        return None
    n = str(ptr.get("name") or "").strip()
    return n or None


def resolve_clot_ml_name(name: str | None = None) -> str:
    """Resolve an artifact id, following the locked POINTER for the canonical name.

    ``clot_ml_0`` is not a directory on disk -- it is the NAME OF WHATEVER IS SHIPPED, and
    every generation so far has lived under its own id (`clot_ml_v0`, `DeployClot_0`,
    `DeployClot_1`).  Until 2026-09-03 this function ignored the pointer and returned the
    compiled-in default, whose directory does not exist, so `_locked_root` fell through to
    the `clot_ml_v0` legacy fallback.  The consequence was silent and total: **promoting and
    repointing changed nothing for any caller that did not pass an explicit name** -- which
    includes `locked.load_default` (it read the pointer for the KIND and then dropped the
    name) and `CustomerDeployPipeline`, i.e. the shipped product, which asks for
    ``clot_ml_0`` by that constant and was still being served a two-generation-old artifact
    built on `clot_gnn_v6`.

    An explicit id is always honoured verbatim, so pinned comparisons against a named past
    generation are unaffected.
    """
    n = (name or DEFAULT_NAME).strip()
    if n in LEGACY_NAMES or n == DEFAULT_NAME:
        pointed = pointer_v0_name()
        if pointed and (LOCKED / pointed).is_dir():
            return pointed
        return DEFAULT_NAME
    return n


def _locked_root(name: str | None = None) -> Path:
    """Resolve on-disk locked artifact directory (canonical name, then legacy fallback)."""
    canonical = resolve_clot_ml_name(name)
    root = LOCKED / canonical
    if root.is_dir():
        return root
    if canonical == DEFAULT_NAME and (legacy := LOCKED / "clot_ml_v0").is_dir():
        return legacy
    return root

#: COMSOL's own Damkohler split (docs/WOUND_PROGRESS.md 18.2).  Not an 003 fit.
DA_SCALE_AUTO = 123.0
#: Shipped-family replace+depth (same section).  Selected on the training cohort,
#: never on n=3 wounds.
REPLACE_ATT = 0.23
REPLACE_DEPTH = 3
REPLACE_SCOPE_ALL_LUMEN = "all_lumen"
REPLACE_SCOPE_WOUND_REGION = "wound_region"
REPLACE_SCOPES = (REPLACE_SCOPE_ALL_LUMEN, REPLACE_SCOPE_WOUND_REGION)

#: An attenuation is a survival fraction per hop; outside this the depth rule is degenerate
#: (nothing ever commits, or a whole shell does regardless of `Mat`).
ATT_CLIP = (0.05, 0.95)


def shear_attenuation(sr: np.ndarray, wall: np.ndarray, att0: float, beta: float) -> np.ndarray:
    """Per-node attenuation from local shear.  ``beta == 0`` is the constant, bit-for-bit."""
    if not beta:
        return np.full(np.shape(sr), float(att0), dtype=np.float64)
    w = np.asarray(wall, dtype=bool)
    ref = float(np.median(np.asarray(sr)[w])) if w.any() else float(np.median(sr))
    ratio = max(ref, 1e-12) / np.maximum(np.asarray(sr, dtype=np.float64), 1e-12)
    return np.clip(float(att0) * ratio ** float(beta), *ATT_CLIP)


#: MLS stencil width per flow source.  GT is differentiated at the consumer's own hops=3;
#: any RECONSTRUCTED field needs a wider stencil to keep its second derivative from being its
#: own sign flip.  `fem` is a solved field like `pred`, not ground truth, so it takes the same
#: treatment -- and it must appear here or a `fem` run dies on a KeyError deep in the rollout.
from src.clot_ml.temporal import _flow_hops as _flow_hops  # single source of truth


@dataclass
class ClotMlV0Config:
    """Typed knobs for the unified stack.  No env toggles."""

    base_model: str = "clot_gnn_v5w"
    ap_renewal_scale: float = 1.0
    da_scale_auto: float = DA_SCALE_AUTO
    washout: bool = True
    replace_att: float = REPLACE_ATT
    replace_depth: int = REPLACE_DEPTH
    #: Shear exponent on the per-hop attenuation:
    #: ``att_node = clip(replace_att * (sr_ref/sr_node)**att_beta, ATT_CLIP)`` with ``sr_ref``
    #: the vessel's own median wall shear.  ``0.0`` is the cohort constant, bit-for-bit.
    #:
    #: The constant is standing in for TRANSPORT -- `Mat` is made at the surface and has to
    #: survive convection to reach depth, so how far it gets is local to the flow.  Measured
    #: leave-one-vessel-out over the six wounds (`scripts/diag_wound_offwall_attenuation.py`,
    #: DEPLOYCLOT 18): the family `att_beta` in {0, .25, .5, 1} x `replace_depth` in {1,2,3}
    #: returns held-out wound-lumen 0.8611 against the shipped 0.8375 and wound-region 0.9270
    #: against 0.9044, with ALL SIX folds picking depth 1 and five of six picking 0.5.
    #: Left at 0.0 here: the default is what unpromoted callers get, and moving it is an
    #: artifact decision, made in a manifest.
    att_beta: float = 0.0
    #: ``all_lumen`` reproduces the original v0 replacement.  ``wound_region`` gives
    #: chemistry the local wound lumen only and leaves the GNN's far-lumen verdict intact.
    #:
    #: DEFAULT CHANGED 2026-09-03, on the six-vessel wound cohort under deploy flow
    #: (`scripts/eval_replace_scope.py`).  The two scopes are IDENTICAL on wall, wound region
    #: and wound lumen -- 0.8866 / 0.8642 / 0.8285 either way, to four decimals -- and differ
    #: only in the far field, where `all_lumen` reads 0.0817 against `wound_region`'s 0.2448.
    #: `wound_patient004/005/006` go from exactly 0.0000 to 0.356 / 0.179 / 0.164; only
    #: `wound_patient003` gives back 0.047.  All six leave-one-vessel-out folds pick
    #: `wound_region`.  At n=3 there was no basis to choose and WOUND_PROGRESS 19 shipped
    #: `all_lumen`; there is now.
    #:
    #: Artifacts promoted before this date carry `replace_scope` in their own manifest and are
    #: unaffected -- `clot_ml_v0_chem_legacy`, which never recorded one, was pinned to
    #: `all_lumen` explicitly on the same day so this default could move without changing it.
    replace_scope: str = REPLACE_SCOPE_WOUND_REGION
    #: Path relative to repo root, or None.  Missing file = physics AP (the hook).
    ap_residual: str | None = None

    @classmethod
    def from_manifest(cls, manifest: dict) -> "ClotMlV0Config":
        block = dict(manifest.get("v0") or {})
        if "base_model" not in block and "base_model" in manifest:
            block["base_model"] = manifest["base_model"]
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in block.items() if k in allowed})

    def to_manifest_block(self) -> dict:
        return asdict(self)


def _replace_target(data, off: np.ndarray, scope: str) -> np.ndarray:
    """Return the off-wall nodes owned by the chemistry replacement policy.

    The wound-region scope is a spatial stitch, not a learned selector: it makes the
    physics and GNN responsible for disjoint parts of the lumen.  That preserves the
    GNN's far-field decision, which the all-lumen chemistry replacement can otherwise
    erase on wound_patient003.
    """
    scope = str(scope)
    if scope == REPLACE_SCOPE_ALL_LUMEN:
        return np.asarray(off, dtype=bool)
    if scope == REPLACE_SCOPE_WOUND_REGION:
        from src.clot_ml.wound import wound_region_masks

        _region, lumen, _far = wound_region_masks(data)
        return np.asarray(off, dtype=bool) & np.asarray(lumen, dtype=bool)
    raise ValueError(f"Unknown chemistry replacement scope {scope!r}; expected one of {REPLACE_SCOPES}")


def metrics_invalid_reason(bundle: dict) -> str | None:
    """Why this artifact must not be scored, or ``None`` if it may be.

    A PRODUCTION artifact is trained with `--include-sealed`, i.e. on the whole corpus, so
    nothing is held out and no number computed on any vessel is an estimate of anything.  The
    stamp lands on the temporal_v4 manifest at the bottom of the chain, so this walks the
    chain rather than looking at one level -- a unified_v0 wrapping a production base is just
    as unscoreable as the base itself.

    The two families are otherwise identical in shape, which is exactly why the boundary needs
    a machine-checkable marker instead of a naming convention (docs/PUBLICATION_PLAN.md 12).
    """
    seen = []
    node = bundle
    for _ in range(4):
        if not isinstance(node, dict):
            break
        man = node.get("manifest")
        if isinstance(man, dict):
            seen.append(man)
        node = node.get("base")
    for man in seen:
        if man.get("metrics_invalid"):
            return str(man.get("metrics_invalid_reason")
                       or f"{man.get('name', '?')} is stamped metrics_invalid")
    return None


def load_v0_bundle(name: str | None = None, device=None) -> dict:
    """Load a ``unified_v0`` artifact: base GNN (wound-capable) + v0 knobs + optional AP net."""
    import json

    root = _locked_root(name)
    manifest = json.loads((root / "manifest.json").read_text())
    cfg = ClotMlV0Config.from_manifest(manifest)
    base = _load_wound_capable_base(cfg.base_model, manifest)
    residual = _load_ap_residual(cfg.ap_residual, device=device)
    return dict(base=base, cfg=cfg, manifest=manifest, ap_residual=residual)


def _default_wound_block() -> dict:
    from src.clot_ml.wound import G_POST0, G_PRE0, OFFWALL_LAG_FRAC, TRIGGER_HOPS
    return dict(g_pre=G_PRE0, g_post=G_POST0, off_att=0.16,
                lag_frac=OFFWALL_LAG_FRAC, trigger="self", k_hops=TRIGGER_HOPS)


def _load_wound_capable_base(name: str, v0_manifest: dict) -> dict:
    """Accept a ``temporal_v4_wound`` artifact, or wrap a ``temporal_v4`` GNN with wound constants."""
    import json

    from src.clot_ml.locked import load_temporal_v4, load_temporal_v4_wound

    root = LOCKED / name
    man = json.loads((root / "manifest.json").read_text())
    kind = man.get("kind")
    if kind == "temporal_v4_wound":
        return load_temporal_v4_wound(name=name)
    if kind == "temporal_v4":
        wound = dict(v0_manifest.get("wound") or man.get("wound") or _default_wound_block())
        gnn = load_temporal_v4(name=name)
        return dict(base=gnn, wound=wound, manifest=man)
    raise ValueError(
        f"clot_ml_0 base '{name}' has kind {kind!r}; need temporal_v4 or temporal_v4_wound")


def _load_ap_residual(path: str | None, device=None):
    """v7 ClotGNN residual on AP, or None.  Untrained residual is the physics."""
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    if not p.exists():
        return None
    from src.clot_ml.ap_field import ApFieldConfig, make_model

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(p, map_location=dev, weights_only=False)
    cfg = ApFieldConfig(**{k: v for k, v in (ck.get("cfg") or {}).items()
                           if k in ApFieldConfig.__dataclass_fields__})
    model = make_model(int(ck["in_dim"]), int(ck["edim"]), cfg).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()
    return dict(model=model, mu=ck["mu"], sd=ck["sd"], device=dev)


def chemistry_mat_trajectory(
    data,
    bio,
    cfg: ClotMlV0Config,
    *,
    flow: str = "gt",
    sample: dict | None = None,
    ap_residual=None,
    wound_rate: tuple[float, float] | None = None,
) -> np.ndarray:
    """``[T, N]`` Mat from the chemistry ODE used by replace+depth -- not the GNN clock.

    Upwind AP renewal (optional residual) + ``da_scale_auto`` + washout + wound-rate
    blockage.  ``ap_closure`` is cleared: the dynamic field already subsumes it.
    """
    from src.clot_ml.wound import wound_rate_blockage
    from src.config import BiochemConfig
    from src.core_physics.ap_closure import SHIPPED_DA_SCALE
    from src.core_physics.physics_wall_model import (
        PER_M3_TO_PER_CM3, WASHOUT_LAMBDA, deposition_gate, integrate_mat_trajectory,
        t0_flow_fields,
    )
    from src.core_physics.wall_ap_renewal import WallApRenewal, make_species_from_renewal

    bio = bio or BiochemConfig(phase="biochem")
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    f = t0_flow_fields(data, bio, hops=_flow_hops(flow), flow_source=flow)
    gate = deposition_gate(data, f, wall=wall, wound_source=True)
    blk = None
    if wound_rate is not None:
        blk = wound_rate_blockage(data, bio, g_pre=float(wound_rate[0]),
                                  g_post=float(wound_rate[1]), inner=None)
    renewal = WallApRenewal(renewal_scale=float(cfg.ap_renewal_scale))
    rp0, ap_cgs = make_species_from_renewal(data, bio, f, renewal=renewal)

    if ap_residual is not None:
        from src.clot_ml.ap_field import correct_ap_cgs_trajectory
        from src.clot_ml.locked import build_sample

        S = sample if sample is not None else build_sample(data, bio, flow=flow, variant="v4")
        scales = bio.get_species_scales(device="cpu")
        ap_scale_cgs = float(scales[1]) * PER_M3_TO_PER_CM3
        ap_cgs = correct_ap_cgs_trajectory(
            ap_residual["model"], S, gate, ap_cgs, ap_scale_cgs,
            ap_residual["mu"], ap_residual["sd"], ap_residual["device"])

    T = int(data.y.shape[0])
    wash = float(WASHOUT_LAMBDA) if cfg.washout else 0.0
    wash_sr = np.broadcast_to(f.sr, (T, int(data.num_nodes))) if cfg.washout else None
    traj, _ = integrate_mat_trajectory(
        data, bio, gate,
        da_scale=SHIPPED_DA_SCALE,
        da_scale_auto=float(cfg.da_scale_auto),
        ap_closure=None,
        species=(rp0, ap_cgs),
        blockage=blk,
        washout=wash,
        washout_sr=wash_sr,
    )
    return np.asarray(traj, dtype=np.float64)


def replace_depth_mask(
    mat: np.ndarray,
    shells: list[np.ndarray],
    owner: np.ndarray,
    *,
    crit: float,
    att,
    depth: int,
) -> np.ndarray:
    """Off-wall mask from owner ``Mat`` through ``att**d`` bars.  Solid is the caller's.

    ``att`` is a scalar attenuation per hop, or a PER-NODE array of them.  The per-node form
    exists because the constant is standing in for transport: material made at the surface
    reaches depth ``d`` against washout, so how far it gets is a local property of the flow,
    not a cohort constant.  A scalar reproduces the shipped rule exactly.
    """
    n = int(mat.shape[0])
    out = np.zeros(n, dtype=bool)
    crit = float(crit)
    own = np.asarray(owner)
    a = np.asarray(att, dtype=np.float64)
    per_node = a.ndim > 0
    if per_node and a.shape != (n,):
        raise ValueError(f"per-node att must be [N={n}], got {a.shape}")
    for d in range(1, int(depth) + 1):
        if d - 1 >= len(shells):
            break
        valid = np.asarray(shells[d - 1], dtype=bool) & (own >= 0)
        if not valid.any():
            continue
        if per_node:
            bar = crit / np.maximum(a[valid] ** d, 1e-30)
        else:
            bar = crit / max(float(a) ** d, 1e-30)
        out[valid] = mat[own[valid]] >= bar
    return out


def _series_from_masks(masks: dict[int, np.ndarray], grid) -> tuple[dict, np.ndarray, np.ndarray]:
    onset = np.full(len(next(iter(masks.values()))), -1, dtype=np.int32)
    seen = np.zeros_like(onset, dtype=bool)
    for ti in grid:
        newly = masks[int(ti)] & ~seen
        onset[newly] = int(ti)
        seen |= masks[int(ti)]
    last = int(grid[-1])
    return masks, masks[last], onset


def _resolve_anchor_mesh(data) -> str:
    """Locate the `.nas`/`.msh` a pack was meshed from, for the local FEM solve.

    Deploy packs carry no `mesh_path`, and the old rewrite mapped
    `data/processed/graphs_biochem_anchors/x.pt` to `data/processed/meshes/x.nas`, which does
    not exist -- every `flow="fem"` run died on the `data.mesh_path` fallback.  The meshes live
    beside the raw anchors.  Raises with the stem named, because a silent wrong mesh would be
    solved happily and score as if it were the vessel.
    """
    import os

    explicit = getattr(data, "mesh_path", None)
    if explicit and os.path.exists(str(explicit)):
        return str(explicit)

    stem = str(getattr(data, "graph_stem", "") or "")
    if not stem:
        pt = str(getattr(data, "path", "") or "")
        stem = os.path.splitext(os.path.basename(pt))[0] if pt else ""
    if not stem:
        raise FileNotFoundError("flow='fem': pack carries neither mesh_path nor a stem")

    from src.utils.paths import get_project_root

    import skfem as fem

    root = get_project_root()
    tried = []
    unreadable = []
    for sub in (
        "data/raw/biochem_anchors",
        "data/raw/biochem",
        "data/raw/kinematics/meshes",
        "outputs/research_sweeps/_meshes",
    ):
        for ext in (".nas", ".msh"):
            cand = root / sub / f"{stem}{ext}"
            tried.append(str(cand))
            if not cand.is_file():
                continue
            # Existing is not the same as loadable.  `.nas` is tried before `.msh`, and the
            # 60 synthetic vessels carry a gmsh-written Nastran file with no `BEGIN BULK`
            # statement, which meshio refuses -- so the whole resolver died on a candidate
            # sitting next to a perfectly good `.msh` of the same mesh.  Fall through to the
            # next candidate instead, and only report the failures if none of them works.
            try:
                fem.Mesh.load(str(cand))
            except Exception as exc:
                unreadable.append(f"{cand} ({type(exc).__name__}: {exc})")
                continue
            return str(cand)
    msg = "flow='fem': no mesh for " + repr(stem) + "; tried: " + ", ".join(tried)
    if unreadable:
        msg += "; present but unreadable: " + "; ".join(unreadable)
    raise FileNotFoundError(msg)


def solve_fem_into_pack(data, *, warm_start: bool = True) -> None:
    """Solve the local Carreau FEM at t=0 and write it into the pack's `u0_pred`/`v0_pred`.

    Separate from `predict_clot_ml_0` because callers build the feature SAMPLE before they
    call it -- doing the solve inside meant the sample, and the baseline scored beside it,
    were still ground truth while the run was labelled `fem`.

    ``warm_start`` seeds the Picard iterate with the analytic Poiseuille prior and releases the
    under-relaxation that exists only to survive the opening transient a cold start pays.  The
    iterate is the wind the operator is linearised about and nothing else, so the converged
    field does not move: measured over the 37-vessel FIT+DEV cohort, the largest deviation from
    the cold solve is 9.5e-08 of the field's own max-norm.  What moves is the cost -- 267 s to
    148 s over the cohort (1.81x), 4.9x on the worst vessel, and **not one of the 37 solved
    slower** (`src.tools.diagnostics.fem_warm_start`).  The gain is concentrated in the
    stenoses, where the cold Picard contracts slowly and the transient is most of the solve.
    Pass ``False`` to reproduce the pre-2026-09 cold path exactly.
    """
    from src.config import PhysicsConfig
    from src.core_physics.local_fem_solver import solve_local_t0_flow

    nas_path = _resolve_anchor_mesh(data)
    # Use the COMSOL t=0 field as the inlet condition only when the pack actually HAS one.
    # `research_synthetic` was added after some research caches were written, so those packs
    # report the flag absent while carrying an all-zero `y[0]` -- and an all-zero inlet is not
    # a ground truth, it is the absence of one.  Imposing it as Dirichlet drove the whole
    # solve to exactly zero, which then "converged" at iteration 0.  Test the field itself.
    u_gt_nd = None
    if not bool(getattr(data, "research_synthetic", False)):
        y = getattr(data, "y", None)
        if y is not None and torch.is_tensor(y) and y.numel() > 0 and y.shape[1] > 0:
            cand = y[0, :, 0:2].detach().cpu().numpy()
            if np.isfinite(cand).all() and float(np.abs(cand).max()) > 0.0:
                u_gt_nd = cand
    phys = PhysicsConfig()
    seed, damping = None, 0.5
    if warm_start:
        from src.data_gen.lib.legal_priors import build_analytic_priors

        try:
            au, av, _, _ = build_analytic_priors(data, phys_cfg=phys)
            seed = torch.stack([au.reshape(-1), av.reshape(-1)], dim=1).cpu().numpy().astype(
                np.float64)
            damping = 1.0
        except Exception as exc:
            # A pack the analytic prior cannot be built on (no width channel, degenerate
            # centreline) is still perfectly solvable cold.  Losing the speedup is not a
            # reason to lose the solve.
            print(f"[i] local FEM: no warm start ({type(exc).__name__}: {exc}); solving cold",
                  flush=True)
            seed, damping = None, 0.5
    u_dim = solve_local_t0_flow(nas_path, data, phys, max_iters=300, tol=1e-9,
                                u_gt_inlet_nd=u_gt_nd, u_init_nd=seed, damping=damping)
    if isinstance(u_dim, torch.Tensor):
        u_dim = u_dim.numpy()
    u_ref = float(data.u_ref.reshape(-1)[0])
    nd = u_dim / u_ref
    data.u0_pred = torch.tensor(nd[:, 0], dtype=torch.float32)
    data.v0_pred = torch.tensor(nd[:, 1], dtype=torch.float32)


def predict_clot_ml_0(bundle, data, times, *, flow: str = "gt", sample=None,
                      preflight: str = "warn", mat_field=None) -> dict:
    """Unified inference.  No-wound packs are bit-identical to the base GNN.

    Wound packs keep the GNN on the healthy wall (and the wound complement on the injured
    segment).  They replace either the full true lumen (the legacy v0 policy) or only the
    wound-local lumen with chemistry-ODE replace+depth.  Unioning the two off-wall verdicts
    is a measured loss (docs/WOUND_PROGRESS.md 17.1).

    ``preflight`` checks the supplied t=0 flow before spending a rollout on it
    (`src/clot_ml/preflight.py`).  An empty wall gate makes the readout's seed empty and every
    downstream channel identically zero, so the result is vacuous rather than merely worse.

      * ``"warn"``  (default) -- print the verdict, run anyway.  Non-breaking: every existing
                    caller keeps its current behaviour and gains the diagnosis.
      * ``"raise"`` -- raise on FAIL.  Use in deployment, where a vacuous prediction is worse
                    than a refusal.
      * ``"off"``   -- skip the check.

    The verdict is returned under the ``"preflight"`` key regardless of mode.
    """
    from src.clot_ml.locked import build_sample, predict_temporal_v4_wound
    from src.clot_ml.wound import has_wound, solid_mask
    from src.config import BiochemConfig
    from src.core_physics.physics_lumen_model import (
        first_corner_shell, solid_boundary_shells, topological_owner,
    )

    cfg: ClotMlV0Config = bundle["cfg"]
    bio = BiochemConfig(phase="biochem")
    # Note: flow="fem" must NOT be rewritten here.  The caller (eval_clot_ml_0.py,
    # research_sweep_runner.py) runs solve_fem_into_pack() before calling this function so that
    # the sample, features, and baseline are all built on the FEM field -- not on GT.  Aliasing
    # "fem" -> "pred" here would give the right u0_pred slot but the wrong stencil/gain treatment
    # downstream (D1+D2+D3 as documented in the commit message).

    # Check the flow BEFORE the rollout: an empty wall gate costs a full rollout and returns
    # identically-zero channels.  Cheap relative to what it can save.
    pf = None
    if preflight != "off":
        from src.clot_ml.preflight import preflight_check

        pf = preflight_check(data, flow, bio)
        if not pf.ok:
            if preflight == "raise":
                raise ValueError(f"preflight FAILED for flow={flow!r}: {pf}")
            print(pf, flush=True)
        elif pf.reasons:
            print(pf, flush=True)

    S = sample if sample is not None else build_sample(data, bio, flow=flow, variant="v4")
    base = predict_temporal_v4_wound(bundle["base"], data, times, flow=flow, sample=S)
    if not has_wound(data):
        if pf is not None:
            base = dict(base, preflight=pf)
        return base

    w = bundle["base"].get("wound") or {}
    wr = (float(w["g_pre"]), float(w["g_post"])) if "g_pre" in w else None
    if mat_field is not None:
        # `mat_field` substitutes a [T, N] SI `Mat` field for the chemistry integration and
        # changes nothing else -- same shells, same owner map, same `att`/`depth`/`scope`,
        # same monotone union.  That makes the learned-field arm an exact ablation of the one
        # component WOUND_PROGRESS 16 localised the off-wall gap to, rather than a second
        # pipeline that has to be argued to be comparable.
        traj = np.asarray(mat_field, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[1] != int(data.num_nodes):
            raise ValueError(f"mat_field must be [T, N={int(data.num_nodes)}], got {traj.shape}")
    else:
        traj = chemistry_mat_trajectory(
            data, bio, cfg, flow=flow, sample=S,
            ap_residual=bundle.get("ap_residual"),
            wound_rate=wr,
        )
    crit = float(bio.viscosity_mat_crit)
    solid = solid_mask(data)
    pos = np.asarray(S["pos"], dtype=np.float64)
    ei = np.asarray(S["edge_index"])
    shell1 = first_corner_shell(pos, solid, ei)
    town = topological_owner(pos, solid, ei)
    shells, owner = solid_boundary_shells(
        pos, solid, ei, shell1=shell1, town=town, max_depth=int(cfg.replace_depth))

    grid = sorted({int(t) for t in times})
    T_raw = int(traj.shape[0])
    off = ~solid
    att_field = float(cfg.replace_att)
    if cfg.att_beta:
        from src.core_physics.physics_wall_model import t0_flow_fields

        _f = t0_flow_fields(data, bio, hops=_flow_hops(flow), flow_source=flow)
        att_field = shear_attenuation(_f.sr, data.mask_wall.reshape(-1).bool().cpu().numpy(),
                                      float(cfg.replace_att), float(cfg.att_beta))
    replace_target = _replace_target(data, off, cfg.replace_scope)
    prev = np.zeros(int(data.num_nodes), dtype=bool)
    series: dict[int, np.ndarray] = {}
    for ti in grid:
        m = np.asarray(base["series"][int(ti)], dtype=bool).copy()
        fld = traj[int(np.clip(ti, 0, T_raw - 1))]
        chemistry_mask = replace_depth_mask(
            fld, shells, owner,
            crit=crit, att=att_field, depth=int(cfg.replace_depth),
        )
        m[replace_target] = chemistry_mask[replace_target]
        m = m | prev
        series[int(ti)] = m
        prev = m
    series, mask, onset = _series_from_masks(series, grid)
    return dict(
        score=base.get("score"),
        mask=mask,
        onset=onset,
        series=series,
        base_comp=base,
        preflight=pf,
    )


# Legacy alias (pre-2026-09 rename).
predict_clot_ml_v0 = predict_clot_ml_0
