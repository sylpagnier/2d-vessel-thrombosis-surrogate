"""Load and run the locked clot-GNN ensemble by name.

    from src.clot_ml.locked import load_ensemble, predict_scores
    ens = load_ensemble()                      # reads data/reference/clot_gnn_locked.json
    score = predict_scores(ens, sample)        # [N] per-node probability

For the temporal model (v3 and on), use the dispatcher instead of ``load_ensemble``
directly -- it reads the pointer's ``kind`` and returns whatever is currently shipped:

    from src.clot_ml.locked import load_default, predict_default_series
    bundle = load_default()
    out = predict_default_series(bundle, data, times)   # {score, mask, onset, series}
"""
from __future__ import annotations

import json
import pickle
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from src.utils.paths import clot_ml_locked_dir, get_project_root

REPO = get_project_root()
POINTER = REPO / "data/reference/clot_gnn_locked.json"


def load_ensemble(name: str | None = None, device=None) -> dict:
    ptr = json.loads(POINTER.read_text())
    root = REPO / (ptr["path"] if name is None else f"outputs/clot_ml/locked/{name}")
    manifest = json.loads((root / "manifest.json").read_text())
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = np.load(root / "feature_norm.npz", allow_pickle=True)

    from src.clot_ml.gnn import ClotGNN

    members = []
    for m in manifest["members"]:
        blob = torch.load(root / m["file"], map_location=dev, weights_only=False)
        edim = 7  # edge_features() width; asserted at call time
        net = ClotGNN(blob["in_dim"], edim, dim=m["dim"], layers=m["layers"], drop=0.0,
                      extra_dim=blob["extra_dim"]).to(dev)
        net.load_state_dict(blob["state_dict"])
        net.eval()
        members.append(dict(net=net, rounds=int(m["rounds"]), file=m["file"]))
    return dict(members=members, mu=norm["mu"], sd=norm["sd"],
                cols=[str(c) for c in norm["cols"]], manifest=manifest, device=dev)


def ensemble_variant(ens: dict) -> str:
    """Which feature block this artifact was trained on -- read from its own manifest."""
    return "v4" if ens.get("manifest", {}).get("v4_channels") else "v3"


def sample_for_ensemble(ens: dict, data, bio_cfg=None, phys_cfg=None, *,
                        flow: str = "gt") -> dict:
    """Build the sample the loaded ensemble actually expects, and check the width.

    A `clot_gnn_v4` member takes 69 columns and a v2/v3 member takes 56; feeding one the
    other's block fails deep inside the first linear layer with an unhelpful shape error.
    Both counts include ``phys_mask``.
    """
    S = build_sample(data, bio_cfg, phys_cfg, flow=flow, variant=ensemble_variant(ens))
    # `n_features` in the manifest is the FULL width, phys_mask included -- it is written
    # from the cache's own `cols` after `attach_physics` has appended it.
    want = int(ens["manifest"].get("n_features", S["X"].shape[1]))
    got = S["X"].shape[1]
    if got != want:
        raise ValueError("%s expects %d features, sample has %d"
                         % (ens["manifest"].get("name", "ensemble"), want, got))
    return S


@torch.no_grad()
def predict_scores(ens: dict, sample: dict) -> np.ndarray:
    """Mean per-node probability over the ensemble.  ``sample`` is a clot-ml cache entry."""
    from src.clot_ml.gnn import build_graph, rollout  # noqa: PLC0415

    out = None
    for m in ens["members"]:
        g = build_graph(sample, ens["mu"], ens["sd"], ens["device"], need_fb=m["rounds"] > 1)
        logit, _ = rollout(m["net"], g, m["rounds"])
        p = torch.sigmoid(logit).cpu().numpy()
        out = p if out is None else out + p
    return out / max(len(ens["members"]), 1)


# Thresholds the locked readout was tuned at (docs/PHASE9_ML.md 8, per domain).
THRESH_WALL, THRESH_OFF = 0.73, 0.92

# Off-wall onset = the time the node's OWNER wall trajectory reaches ``crit / ONSET_OFF_ATT``.
# 0.80 means "just after its owner commits".  Chosen for a PHYSICAL reason, not a scored
# one: an off-wall node cannot clot before the wall node feeding it, and freezing off-wall
# at the final mask (the alternative) puts off-wall clot on screen at t=0 with an empty
# wall, which is nonsense.  On score it is a wash -- the att sweep reads
# 0.490 / 0.494 / 0.459 / 0.510 at 0.16 / 0.30 / 0.50 / 0.80 against frozen's 0.5015, all
# inside noise -- so the constraint is doing the work, not a fit.
#: Attenuation on the off-wall ONSET-TIME constraint.  Distinct from
#: ``wound.OFF_ATT_WOUND`` (0.16), the wound commit rule -- both were ``OFF_ATT``.
ONSET_OFF_ATT = 0.80


def build_sample(data, bio_cfg=None, phys_cfg=None, *, flow: str = "gt",
                 variant: str = "v3") -> dict:
    """Feature dict for one raw pack, matching the locked ensemble's training layout.

    ``variant="v4"`` additionally applies :func:`src.clot_ml.features_v4.augment_sample`,
    which appends the 13 advective-transport / indicator-gate channels a `clot_gnn_v4`
    member expects.  Order matters and is pinned by the cache builder: the v4 block goes
    **after** the 55 base channels and **before** ``phys_mask``.  Use
    :func:`sample_for_ensemble` rather than choosing the variant by hand.
    """
    from src.clot_ml.data import physics_mask
    from src.clot_ml.features import build_features, feature_matrix
    from src.config import BiochemConfig, PhysicsConfig

    bio = bio_cfg or BiochemConfig(phase="biochem")
    phys = phys_cfg or PhysicsConfig(phase="biochem")
    S = build_features(data, bio, phys, flow=flow)
    X, cols = feature_matrix(S["F"])
    out = dict(X=X, cols=np.array(cols), y=S["y"], mat_gt=S["mat_gt"], wall=S["wall"],
               solid=S["solid"], shell=S["shell"], owner=S["owner"],
               edge_index=S["edge_index"],
               pos=S["pos"], mat_phys=S["mat_phys"], gate=S["gate"], sr=S["sr"],
               spd=S["spd"], u=S["u"], v=S["v"])
    if variant == "v4":
        from src.clot_ml.features_v4 import augment_sample
        Xv, colsv = augment_sample(data, out, bio, flow=flow)
        out["X"], out["cols"] = Xv, np.array(colsv)
    m = physics_mask(out)
    out["phys_mask"] = m
    out["X"] = np.concatenate([out["X"], m.astype(np.float32).reshape(-1, 1)], axis=1)
    out["cols"] = np.array([str(c) for c in out["cols"]] + ["phys_mask"])
    return out


def predict_clot_series(ens: dict, data, times, *, flow: str = "gt",
                        sample: dict | None = None) -> dict:
    """Clot mask at each requested time index.

    The SET is the locked ensemble's; the WALL timing is the zero-parameter surface ODE's
    first crossing of ``viscosity_mat_crit``.  Off-wall stays frozen at the final mask --
    every off-wall timing rule measured so far scores below frozen because it depends on the
    ``Mat`` magnitude field (docs/PHASE9_ML.md 12.4).

    Returns ``{score, mask, onset, series}`` where ``series`` maps time index -> bool mask.
    """
    from src.clot_ml.temporal import mask_series, ode_trajectory, onset_from_ode
    from src.config import BiochemConfig

    bio = BiochemConfig(phase="biochem")
    S = sample if sample is not None else build_sample(data, bio, flow=flow)
    score = predict_scores(ens, S)
    wall = S["wall"]
    mask = ((score >= THRESH_WALL) & wall) | ((score >= THRESH_OFF) & ~wall)

    traj, _ = ode_trajectory(data, bio, flow=flow)
    crit = float(bio.viscosity_mat_crit)
    onset = onset_from_ode(traj, mask, wall, S["pos"].astype(np.float64), crit,
                           attenuation=ONSET_OFF_ATT)
    return dict(score=score, mask=mask, onset=onset,
                series=mask_series(onset, mask, times))


@torch.no_grad()
def predict_mat(ens: dict, sample: dict) -> np.ndarray:
    """Mean predicted ``log1p(Mat/crit)`` over the ensemble's REGRESSION head.

    That head exists in every locked member (physics-based, zero-init residual on the
    backbone's own ``Mat``) but has never been the readout -- the deploy score uses the
    classifier.  It is the natural place to read the magnitude field from.
    """
    from src.clot_ml.gnn import build_graph, rollout  # noqa: PLC0415

    out = None
    for m in ens["members"]:
        g = build_graph(sample, ens["mu"], ens["sd"], ens["device"], need_fb=m["rounds"] > 1)
        _, reg = rollout(m["net"], g, m["rounds"])
        r = reg.cpu().numpy()
        out = r if out is None else out + r
    return out / max(len(ens["members"]), 1)


# ---------------------------------------------------------------------------
# v3: time-conditioned model (docs/PHASE9_ML.md 13.9)
# ---------------------------------------------------------------------------
def load_temporal_v3(name: str | None = None) -> dict:
    """Load a v3-kind artifact: the base GNN (the SET) plus the time-conditioned head."""
    ptr = json.loads(POINTER.read_text())
    root = REPO / (ptr["path"] if name is None else f"outputs/clot_ml/locked/{name}")
    manifest = json.loads((root / "manifest.json").read_text())
    ens = load_ensemble(name=manifest["base_set_model"])
    with (root / manifest["clf_file"]).open("rb") as fh:
        clf = pickle.load(fh)
    return dict(ens=ens, clf=clf, manifest=manifest,
               thresh_wall=float(manifest["thresh_wall"]),
               thresh_off=float(manifest["thresh_off"]),
               n_times_trained=int(manifest["n_times"]))


def enforce_owner_and_monotone(series: dict[int, np.ndarray], wall: np.ndarray,
                               owner: np.ndarray, times) -> dict[int, np.ndarray]:
    """Two physical constraints applied to a raw per-time mask series, in place order:

    1. MONOTONE in time -- the production law has no sink, so a node once clot stays clot.
    2. An off-wall node cannot be clot before its OWNER wall node is (it is fed by it).

    Pure and model-free, so it is unit-testable without loading any weights.
    """
    out: dict[int, np.ndarray] = {}
    prev = np.zeros_like(wall)
    for ti in times:
        m = series[ti] | prev
        m = m & (wall | m[owner])
        out[int(ti)] = m
        prev = m
    return out


def load_temporal_v4(name: str | None = None) -> dict:
    """Load a v4-kind artifact: the v4 GNN ensemble plus the temporal readout.

    Unlike v3's fixed thresholds, the committed SET is the readout family
    `scripts/promote_clot_gnn_v4_temporal.py` selected honestly on the whole pool (an
    adaptive keep/add cut on the wall, an expected-score budget off it -- 10), and the
    off-wall SCHEDULE is a learned per-node lag anchored on the ODE's own owner crossing
    (15) rather than a threshold rule.  The pickle holds only plain sklearn estimators, not
    a wrapper class, so it does not depend on any script module at unpickle time.
    """
    # `name=None` means the BASE role, resolved from the pointer's manifest chain.  It used
    # to mean `ptr["path"]`, which is the UNIFIED artifact -- a different kind entirely.
    from src.clot_ml.artifacts import BASE, root as artifact_root

    root = artifact_root(name, BASE)
    manifest = json.loads((root / "manifest.json").read_text())
    ens = load_ensemble(name=manifest["name"])
    with (root / manifest["temporal_file"]).open("rb") as fh:
        temporal = pickle.load(fh)
    return dict(ens=ens, temporal=temporal, manifest=manifest)


def _committed_set_v4(S: dict, sc: np.ndarray, temporal: dict) -> np.ndarray:
    """Apply the shipped wall + off-wall committed-set specs to one vessel's scores."""
    from src.clot_ml.readouts import expected_curve  # noqa: PLC0415
    from src.clot_ml.strict_readout import apply_adapt, readout_resid  # noqa: PLC0415
    from src.clot_ml.softmetric import dilation_operator, to_torch_sparse  # noqa: PLC0415

    def apply_spec(spec, dom_of):
        d = dom_of(S)
        if spec["kind"] == "cohort_cut":
            return d & (sc >= spec["t"])
        if spec["kind"] == "resid":
            return d & readout_resid(S, sc, tuple(spec["th"]))
        if spec["kind"] == "resid_adapt":
            # `lo`/`hi` bound the vessel statistic to the support the slope was fitted on.
            # Specs promoted before that was recorded simply lack the keys and behave
            # exactly as they always did (docs/PHASE10_V4.md; scripts/eval_adapt_clamp.py).
            return d & apply_adapt(S, sc, "resid", tuple(spec["th"]), dom_of,
                                   spec["b"], spec["med"],
                                   spec.get("lo"), spec.get("hi"))
        if spec["kind"] == "expected_tuned":
            dev = torch.device("cpu")
            Dt = to_torch_sparse(dilation_operator(S["edge_index"], len(S["wall"]), 2), dev)
            ks, vals = expected_curve(sc, d, Dt, dev, spec["gamma"])
            if len(ks) < 2:
                return np.zeros(len(sc), bool)
            k = int(np.clip(round(ks[int(np.argmax(vals))] * spec["kscale"]), 1, ks[-1]))
            order = np.flatnonzero(d)[np.argsort(-sc[d])]
            m = np.zeros(len(sc), bool)
            m[order[:k]] = True
            return m
        raise ValueError(spec["kind"])

    # The committed set now uses the SAME domains the score is computed on (A3,
    # `src/clot_ml/data.eval_domains`): off-wall is TRUE LUMEN, `~solid`.  This was deferred
    # at A3 because the artifact was stale and could not be re-verified against its promotion
    # gates; closed after the Phase B and C0 re-promotions, whose gates pass.
    #
    # Measured before changing it: the two conventions differ on exactly the wound nodes
    # (80 / 80 / 26) and on ZERO cohort nodes, so no published figure moves.  On a wound pack
    # the base no longer commits into a region the wound module owns and would override.
    from src.clot_ml.data import off_domain, wall_domain  # noqa: PLC0415

    wall_of, off_of = wall_domain, off_domain

    # REGIME-CONDITIONED READOUT.  The shipped cuts are ABSOLUTE (0.95/0.53/0.98/0.92) and
    # they are only valid in the regime they were fitted in.  On a wound pack the score
    # field's calibration collapses while its RANKING survives: off-wall p99 reads
    # 0.001-0.53 against a 0.92 cut, and yet w_lum AUC is 0.95-0.97 -- so the committed set
    # is empty for a field that ranks the thrombus almost perfectly
    # (docs/WOUND_PROGRESS.md 14.7).  `wound_spec` swaps in a RANK-based readout there.
    #
    # It cannot touch any published cohort figure: the branch is taken only when the pack
    # carries wound nodes, and no cohort, clot-free or SEALED vessel does.  A global swap was
    # measured and rejected -- it costs the 23-vessel cohort -0.0222 wall / -0.1616 off-wall
    # to buy +0.0982 / +0.0922 on the three wound vessels.
    spec = temporal.get("wound_spec")
    if spec is not None and bool((np.asarray(S.get("solid", S["wall"]), dtype=bool)
                                  & ~np.asarray(S["wall"], dtype=bool)).any()):
        return apply_spec(spec, wall_of) | apply_spec(spec, off_of)
    return (apply_spec(temporal["wall_spec"], wall_of)
            | apply_spec(temporal["off_spec"], off_of))


def predict_temporal_v4(bundle: dict, data, times, *, flow: str = "gt",
                        sample: dict | None = None) -> dict:
    """Time-conditioned v4 prediction.  Returns ``{score, mask, onset, series}``, the same
    shape as :func:`predict_clot_series`.

    ``times`` is used directly as the evaluation grid (sorted, deduplicated) -- the
    time-resolved transport field (mat_adv_t) is solved fresh for exactly these times, so
    unlike a precomputed cache this is not restricted to any fixed grid density.
    """
    from src.clot_ml.temporal import (  # noqa: PLC0415
        lag_features, node_features, ode_wall_series, offwall_by_learned_lag, series_masks,
        time_block,
    )
    from src.clot_ml.features_v4 import horizon_for  # noqa: PLC0415
    from src.clot_ml.temporal import ode_trajectory  # noqa: PLC0415
    from src.clot_ml.transport import (  # noqa: PLC0415
        _node_volume, _solve_upwind, upwind_operator,
    )
    from src.config import BiochemConfig  # noqa: PLC0415

    temporal = bundle["temporal"]
    # The committed-set spec, the ODE clock and the transport channels were all fitted
    # against ONE t=0 flow.  Artifacts promoted before the field was recorded carry no
    # `flow` key and are read as the historical `gt`; a mismatch is a warning rather than an
    # error because the GT-fitted head is still the right comparison arm for a FEM run.
    _fit_flow = str(temporal.get("flow", "gt"))
    if _fit_flow != str(flow):
        print("[WARN] temporal head was fitted on flow=%r and is being applied at flow=%r; "
              "its cuts and clock describe the other field" % (_fit_flow, flow), flush=True)
    bio = BiochemConfig(phase="biochem")
    S = sample if sample is not None else build_sample(data, bio, flow=flow, variant="v4")
    wall, owner = S["wall"], S["owner"]
    crit = float(bio.viscosity_mat_crit)

    sc = predict_scores(bundle["ens"], S)
    gm = _committed_set_v4(S, sc, temporal)

    grid = sorted({int(t) for t in times})
    # The ODE clock must be the SAME OBJECT the head was fitted against, so the flag travels
    # on the artifact rather than being a call-site choice.  Absent (every shipped artifact
    # to date) means the wake-free trajectory, bit-for-bit.
    wr = temporal.get("wound_rate")
    traj, t_grid = ode_trajectory(data, bio, flow=flow,
                                  wake=bool(temporal.get("wake_ode", False)),
                                  stall=bool(temporal.get("stall_ode", False)),
                                  wound_source=bool(temporal.get("wound_source", True)),
                                  wound_rate=None if wr is None else tuple(wr))
    T_raw = traj.shape[0]
    r0 = traj[1] / max(float(t_grid[1] - t_grid[0]), 1e-9)
    hot = traj >= crit
    oon = np.where(hot.any(0), hot.argmax(0), T_raw)

    # time-resolved transport for exactly the requested grid -- build_temporal_transport.py's
    # construction, run live (t=0 flow only, deploy-legal): the operator is linear and
    # time-independent, only the wall source `traj[ti]` changes per query time.
    pos = S["pos"].astype(np.float64)
    u, v = S["u"].astype(np.float64), S["v"].astype(np.float64)
    # ONE definition of the transport horizon (`features_v4.horizon_for`).  It excludes
    # the SOLID boundary from the bulk-speed median, not just the healthy wall -- an
    # inline `~wall` copy here would compute a different horizon than the cache builder
    # on a wound pack, i.e. a silent train/deploy skew in the v4 transport channels.
    H = horizon_for(pos, u, v, np.asarray(S.get("solid", wall), dtype=bool))
    F, out = upwind_operator(pos, S["edge_index"], u, v)
    vol = _node_volume(pos, S["edge_index"])
    n_grid = len(grid)
    adv = np.zeros((n_grid, len(wall)), dtype=np.float32)
    own = np.zeros_like(adv)
    slf = np.zeros_like(adv)
    # The advective source is every SOLID boundary node, not just the healthy wall.  On a
    # wound pack the two differ and the wound is the larger source (WOUND_PROGRESS 14.6);
    # off a wound pack `solid` IS `wall`, so this is a no-op on every cohort vessel.
    src_mask = np.asarray(S.get("solid", wall), dtype=bool)
    for j, ti in enumerate(grid):
        ti_c = int(np.clip(ti, 0, T_raw - 1))
        src = np.zeros(len(wall))
        src[src_mask] = np.maximum(traj[ti_c][src_mask], 0.0)
        adv[j] = _solve_upwind(F, out, src * vol, vol, H).astype(np.float32)
        own[j] = traj[ti_c][owner].astype(np.float32)
        slf[j] = traj[ti_c].astype(np.float32)
    tt = dict(mat_adv_t=np.log1p(np.maximum(adv, 0) / crit).astype(np.float32),
              mat_owner_t=np.log1p(np.maximum(own, 0) / crit).astype(np.float32),
              mat_self_t=np.log1p(np.maximum(slf, 0) / crit).astype(np.float32))

    Vd = {"q": dict(S=S, T=T_raw, times=grid, r0=r0, oon=oon, oon_c={1.0: oon}, tt=tt,
                    clock=[])}
    oofs = {"v4": {"q": sc}}

    P = np.zeros((n_grid, len(wall)), dtype=np.float32)
    for j in range(n_grid):
        row = np.concatenate([node_features(Vd, "q", oofs), time_block(Vd, "q", j)], axis=1)
        P[j] = np.mean([m.predict_proba(row)[:, 1] for m in temporal["head"]], axis=0)
    P = np.maximum.accumulate(P, axis=0)

    th_w, cf_w = temporal["time_th_wall"]
    th_o, cf_o = temporal["time_th_off"]
    M_wall = series_masks(gm, P, th_w, bool(cf_w), owner, wall)

    burden_gate = temporal["burden_gate"]
    lag_models = temporal["lag_models"]
    off_burden = int((gm & ~wall).sum())
    if burden_gate is not None and lag_models and off_burden >= burden_gate:
        lag_pred = np.mean([m.predict(lag_features(Vd, "q", oofs)) for m in lag_models],
                           axis=0)
        Mw_ode = ode_wall_series(Vd, "q", gm, n_grid)
        M_off = offwall_by_learned_lag(Mw_ode, gm, owner, wall, lag_pred, bool(cf_o))
    else:
        M_off = series_masks(gm, P, th_o, bool(cf_o), owner, wall)

    raw = {grid[j]: (M_wall[j] & wall) | (M_off[j] & ~wall) for j in range(n_grid)}
    series = enforce_owner_and_monotone(raw, wall, owner, grid)

    onset = np.full(len(wall), -1, dtype=int)
    seen = np.zeros(len(wall), dtype=bool)
    for ti in grid:
        newly = series[int(ti)] & ~seen
        onset[newly] = int(ti)
        seen |= series[int(ti)]
    score = P[-1] if n_grid else sc
    return dict(score=score, mask=series[grid[-1]], onset=onset, series=series)


# ---------------------------------------------------------------------------
# v4w: v4 plus the wound complement (docs/WOUND_PROGRESS.md 10)
# ---------------------------------------------------------------------------
def load_temporal_v4_wound(name: str | None = None) -> dict:
    """Load a ``temporal_v4_wound`` artifact: an unmodified v4 plus the wound module.

    The base ensemble and readout are byte-identical to ``clot_gnn_v4`` -- this artifact adds
    a boundary-condition branch, not a retrained model. It exists because COMSOL's wound law
    is the wall law with the shear gates deleted, which v4 has no channel for: 100% of wound
    nodes clot and the t=0 gate fires on 0% of them.
    """
    # `name=None` means the WOUND role.  It used to resolve to `ptr["path"]`, i.e. the
    # unified_v0 artifact, whose manifest has no "wound" key -- so the default path raised
    # KeyError and every caller was forced to name a baseline explicitly, which is how
    # `eval_clot_ml_0.py --baseline` came to sit two generations stale.
    from src.clot_ml.artifacts import WOUND, root as artifact_root

    root = artifact_root(name, WOUND)
    manifest = json.loads((root / "manifest.json").read_text())
    base = load_temporal_v4(name=manifest["base_model"])
    _assert_wound_alias_integrity(manifest, base)
    return dict(base=base, wound=dict(manifest["wound"]), manifest=manifest)


def _assert_wound_alias_integrity(manifest: dict, base: dict) -> None:
    """Fail closed if a named wound-baseline alias drifts from its locked source.

    An alias is deliberately lightweight—it must not duplicate nine checkpoints—but that
    makes silent source drift especially dangerous.  The optional ``source`` block binds an
    alias to the exact base-manifest bytes and to the exact wound constants it was reviewed
    with.  Ordinary wound artifacts have no block and preserve their historical behaviour.
    """
    source = manifest.get("source")
    if not source:
        return
    base_name = str(source.get("base_artifact", ""))
    if base_name and base_name != str(manifest.get("base_model", "")):
        raise ValueError(
            f"{manifest.get('name', 'wound artifact')} aliases {base_name!r} but resolves "
            f"base_model={manifest.get('base_model')!r}")
    base_path = clot_ml_locked_dir() / str(manifest["base_model"]) / "manifest.json"
    expected_hash = source.get("base_manifest_sha256")
    if expected_hash:
        got_hash = sha256(base_path.read_bytes()).hexdigest().upper()
        if got_hash != str(expected_hash).upper():
            raise ValueError(
                f"{manifest.get('name', 'wound artifact')} base manifest changed: "
                f"expected {expected_hash}, got {got_hash}")
    expected_fp = source.get("base_fingerprint")
    got_fp = base.get("manifest", {}).get("fingerprint")
    if expected_fp and got_fp != expected_fp:
        raise ValueError(
            f"{manifest.get('name', 'wound artifact')} base fingerprint changed: "
            f"expected {expected_fp!r}, got {got_fp!r}")
    wound_source = source.get("wound_artifact")
    if wound_source:
        wound_path = clot_ml_locked_dir() / str(wound_source) / "manifest.json"
        source_manifest = json.loads(wound_path.read_text())
        lhs = json.dumps(manifest["wound"], sort_keys=True, separators=(",", ":"))
        rhs = json.dumps(source_manifest["wound"], sort_keys=True, separators=(",", ":"))
        if lhs != rhs:
            raise ValueError(
                f"{manifest.get('name', 'wound artifact')} wound constants differ from "
                f"its declared source {wound_source!r}")


def predict_temporal_v4_wound(bundle: dict, data, times, *, flow: str = "gt",
                              sample: dict | None = None) -> dict:
    """v4 everywhere, the wound module on the nodes it owns.

    **On a pack with no wound this returns v4's own output unchanged**, which is the property
    that lets the artifact supersede v4 outright rather than sit beside it;
    ``src/tests/test_wound_complement.py`` pins it at the dispatcher level.

    On a wound pack, after compose, OR hop-2 stall-opened t=0-ungated wall into the series.
    That is the only 0-FP SET gain measured on ``wound_comsol003`` (blinds the GNN never
    ranks); it is inert on 001/002 and does not add owner-basin lumen.
    """
    from src.clot_ml.temporal import union_ungated_stall_series
    from src.clot_ml.wound import compose_with_v4, has_wound, predict_wound_series
    from src.config import BiochemConfig

    if not has_wound(data):
        return predict_temporal_v4(bundle["base"], data, times, flow=flow, sample=sample)
    w = bundle["wound"]
    # The wound-regime readout AND the wound's deposition rate travel on the WOUND artifact,
    # not on the base one: v5's temporal.pkl stays byte-identical and neither branch can fire
    # without a wound mask.  `wound_rate` makes the shared ODE integrate the same two-regime
    # law the complement does, instead of the static `srf2` prefactor of 1 -- without it the
    # injured patch reads 1.35x crit against GT's 9.04x and no wound-owned lumen node can ever
    # clear the `crit / off_att` magnitude bar (docs/WOUND_PROGRESS.md 15).
    over = {}
    if w.get("readout"):
        over["wound_spec"] = dict(w["readout"])
    if bool(w.get("rate_in_ode", True)):
        over["wound_rate"] = (float(w["g_pre"]), float(w["g_post"]))
    if over:
        bundle = dict(bundle)
        bundle["base"] = dict(bundle["base"],
                              temporal=dict(bundle["base"]["temporal"], **over))
    bio = BiochemConfig(phase="biochem")
    wr = bundle["base"]["temporal"].get("wound_rate")
    base = predict_temporal_v4(bundle["base"], data, times, flow=flow, sample=sample)
    out = predict_wound_series(
        data, bio, times,
        g_pre=float(w["g_pre"]), g_post=float(w["g_post"]),
        flow=flow, off_att=float(w["off_att"]), lag_frac=float(w["lag_frac"]),
        trigger=str(w.get("trigger", "self")), k_hops=int(w.get("k_hops", 25)),
        # The off-boundary DEPTH rule travels on the artifact like every other wound scalar.
        # It defaults to `shell` -- one corner shell -- so an artifact without the key is
        # bit-identical, which is every artifact promoted before 2026-08-24 because this
        # dispatcher never passed the argument at all.  `recursive` is safe but currently
        # buys nothing: shell 2 needs `Mat >= crit / off_att**2` = 39x crit and the ODE's
        # wound reaches 17x, so no deeper ring clears its bar on any of the three wound
        # vessels (docs/WOUND_PROGRESS.md 16.4).
        lumen=str(w.get("lumen", "shell")),
        # Resting-platelet renewal.  Travels on the WOUND artifact like every other wound
        # scalar; absent (every artifact before 2026-09-03) means 0.0, which is the
        # frozen-`rp` model bit-for-bit.  See docs/DEPLOYCLOT.md 5b for why it exists.
        rp_C=float(w.get("rp_C", 0.0) or 0.0),
        # Whether the wall-AP CONSUMPTION closure applies at the wound.  Absent (every
        # artifact before 2026-09-03) means True, the shipped behaviour.
        wound_ap_closure=bool(w.get("wound_ap_closure", True)),
    )
    comp = compose_with_v4(base, out, times)
    series = union_ungated_stall_series(data, bio, comp["series"], times, flow=flow,
                                        wound_rate=wr)
    last = int(max(times))
    mask = series[last]
    onset = np.asarray(comp["onset"]).copy()
    newly = mask & ~np.asarray(comp["mask"], dtype=bool)
    if newly.any():
        for ti in sorted(int(t) for t in times):
            hit = newly & series[int(ti)] & (onset < 0)
            onset[hit] = int(ti)
    comp = dict(comp)
    comp["series"] = series
    comp["mask"] = mask
    comp["onset"] = onset
    return comp



def load_default(device=None) -> tuple[dict, str]:
    """Follow the pointer and load whatever generation is currently shipped.

    Returns ``(bundle, kind)``; ``kind`` is ``"gnn_ensemble"`` (v1/v2, ODE-only timing),
    ``"temporal_v3"``, ``"temporal_v4"``, ``"temporal_v4_wound"``, or ``"unified_v0"``.
    Use with :func:`predict_default_series`.
    """
    ptr = json.loads(POINTER.read_text())
    kind = ptr.get("kind", "gnn_ensemble")
    if kind == "unified_v0":
        from src.clot_ml.v0 import load_v0_bundle
        return load_v0_bundle(), kind
    if kind == "temporal_v4_wound":
        return load_temporal_v4_wound(), kind
    if kind == "temporal_v4":
        return load_temporal_v4(), kind
    if kind == "temporal_v3":
        return load_temporal_v3(), kind
    return load_ensemble(device=device), kind


def predict_default_series(bundle: dict, kind: str, data, times, *, flow: str = "gt",
                           sample: dict | None = None) -> dict:
    if kind == "unified_v0":
        from src.clot_ml.v0 import load_v0_bundle, predict_clot_ml_0
        return predict_clot_ml_0(bundle, data, times, flow=flow, sample=sample)
    if kind == "temporal_v4_wound":
        return predict_temporal_v4_wound(bundle, data, times, flow=flow, sample=sample)
    if kind == "temporal_v4":
        return predict_temporal_v4(bundle, data, times, flow=flow, sample=sample)
    if kind == "temporal_v3":
        # `clot_gnn_v3` is a retired generation. Its promotion script
        # (`scripts/promote_clot_gnn_v3.py`) was deleted, so the reader that went
        # with it could not be imported either -- loading such an artifact raised a
        # bare ImportError about a missing module, which named neither the cause nor
        # the cure. Say so instead.
        raise NotImplementedError(
            "clot_ml artifact kind 'temporal_v3' (clot_gnn_v3) is retired: its "
            "promotion script and reader were removed. Re-promote with "
            "scripts/promote_clot_ml_0.py, or recover the v3 reader from git history."
        )
    return predict_clot_series(bundle, data, times, flow=flow, sample=sample)
