"""Score a Stage-A model on the ACTUAL clot_ml_0 deploy metric, during training.

Every diagnostic in the Stage-A log is a proxy, and the proxies were measured against the
real thing on 33 vessels with a trained checkpoint:

    gate Jaccard fraction   +0.613      <- best, but only R^2 0.38
    wall-gate fire ratio    -0.395
    empty-gate flag         -0.350
    dsrx correlation        +0.131
    rel-L2 on velocity      -0.030      <- essentially unrelated

So a run can improve every number it prints and still lose the thing it is for.  This runs
the deployed readout instead: predict t=0 flow, write it to the pack as `u0_pred`, and score
`clot_ml_v0` exactly as `eval_clot_ml_v0.py` does.

It costs about 33 s per vessel, and -- measured -- the score does NOT depend on the temporal
grid (`every` 4, 12 and 25 return identical F1), so the grid is coarsened for speed.  Five
vessels every few epochs is roughly a 10% overhead.

The default vessel set spans the failure modes found on the full cohort rather than being the
easy end: patient010 loses everything (its wall gate empties), patient005 over-fires,
patient020 degrades gradually, patient003 can IMPROVE on predicted flow, patient011 is robust.
"""
from __future__ import annotations

import os

#: Spans the measured failure modes; override with `KINEMATICS_DEPLOY_PROBE_STEMS`.
DEFAULT_STEMS = ("patient010", "patient005", "patient020", "patient003", "patient011")

#: GT wall F1 for those vessels, so the probe reports a DROP rather than a bare number.
_GT_WALL = {"patient010": 0.969, "patient005": 0.986, "patient020": 0.988,
            "patient003": 0.335, "patient011": 0.737}
#: Ground-truth OFF-WALL F1 for the same vessels.  Reported separately because the two domains
#: fail differently: on patient005 off-wall collapses at 0.7% velocity error while wall holds
#: to ~5%, so a single averaged drop hides which one moved.
_GT_OFF = {"patient010": 0.895, "patient005": 0.415, "patient020": 0.477,
           "patient003": float("nan"), "patient011": float("nan")}


def probe_stems() -> list[str]:
    raw = os.environ.get("KINEMATICS_DEPLOY_PROBE_STEMS", "").strip()
    return [s.strip() for s in raw.split(",") if s.strip()] or list(DEFAULT_STEMS)


def deploy_f1_probe(model, device, *, stems=None, every: int = 25, verbose: bool = True) -> dict:
    """``{stem: wall_f1}`` plus ``mean_drop`` against ground-truth flow.  ``{}`` on any failure.

    Never raises: a probe that kills a training run is worse than no probe.
    """
    import numpy as np
    import torch

    try:
        from pathlib import Path

        from scripts.eval_clot_ml_v0 import PACKS, _times
        from scripts.eval_wound_complement import gt_series, score_domains
        from src.clot_ml.data import eval_domains
        from src.clot_ml.locked import build_sample, load_temporal_v4_wound
        from src.clot_ml.v0 import load_v0_bundle, predict_clot_ml_v0
        from src.config import BiochemConfig, PhysicsConfig
        from src.data_gen.lib.legal_priors import apply_prior_source
    except Exception as exc:                       # clot stack absent on this machine
        if verbose:
            print(f"[kin] deploy probe unavailable: {type(exc).__name__}: {exc}")
        return {}

    stems = list(stems or probe_stems())
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    try:
        bundle_v0 = load_v0_bundle("clot_ml_v0")
        _ = load_temporal_v4_wound("clot_gnn_v5w")
    except Exception as exc:
        if verbose:
            print(f"[kin] deploy probe: no clot bundle ({type(exc).__name__})")
        return {}

    was_training = model.training
    model.eval()
    out: dict[str, float] = {}
    for stem in stems:
        p = Path(PACKS) / f"{stem}.pt"
        if not p.is_file():
            continue
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
            d.graph_stem = stem
            d = apply_prior_source(d, "analytic")
            with torch.no_grad():
                pred = model(d.to(device), solver="anderson")
            pred = (pred[0] if isinstance(pred, tuple) else pred).detach().cpu()
            d = d.to("cpu")
            d.u0_pred = pred[:, 0].contiguous()
            d.v0_pred = pred[:, 1].contiguous()
            times = _times(d, every)
            S = build_sample(d, bio, flow="pred", variant="v4")
            v0 = predict_clot_ml_v0(bundle_v0, d, times, flow="pred", sample=S)
            wall, off = eval_domains(S)
            wall = np.asarray(wall, dtype=bool); off = np.asarray(off, dtype=bool)
            sc = score_domains(v0["series"][times[-1]], gt_series(d, phys, times)[times[-1]],
                               torch.tensor(np.asarray(S["edge_index"])), wall,
                               dict(wall=wall, off=off))
            out[stem] = float(sc.get("wall", float("nan")))
            out[stem + "/off"] = float(sc.get("off", float("nan")))
        except Exception as exc:
            if verbose:
                print(f"[kin] deploy probe {stem}: {type(exc).__name__}: {exc}")
        finally:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    if was_training:
        model.train()

    wal = {k: v for k, v in out.items() if not k.endswith("/off") and v == v}
    offs = {k[:-4]: v for k, v in out.items() if k.endswith("/off") and v == v}
    if wal:
        dw = [wal[k] - _GT_WALL[k] for k in wal if k in _GT_WALL]
        out["mean_wall"] = float(np.mean(list(wal.values())))
        out["wall_drop"] = float(np.mean(dw)) if dw else float("nan")
    do = [offs[k] - _GT_OFF[k] for k in offs if _GT_OFF.get(k, float("nan")) == _GT_OFF.get(k)]
    if do:
        out["mean_off"] = float(np.mean([v for k, v in offs.items()
                                         if _GT_OFF.get(k, float("nan")) == _GT_OFF.get(k)]))
        out["off_drop"] = float(np.mean(do))
    return out


__all__ = ["DEFAULT_STEMS", "deploy_f1_probe", "probe_stems"]
