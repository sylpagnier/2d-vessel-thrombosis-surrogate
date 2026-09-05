"""Fit the ONE scalar the training objective leaves unconstrained: the residual's amplitude.

Under the hard BC the model's whole contribution is ``delta = envelope(sdf) * gain * s * r``,
where ``s`` is the ReZero scalar ``residual_rezero`` introduced.  ReZero fixed an OVERSHOOT --
a randomly-initialised decoder emits an O(1) field against a FEM prior whose error is O(0.01),
so a fresh model started 19-24x too large and spent the run suppressing itself.  Starting at
zero fixed that, and then left the opposite problem: nothing in the objective pushes ``s`` back
up.  The data term is ~92% of the loss and is FLAT in the residual scale to first order around
an already-accurate prior, so the run converges with the gate still asking for more.  Measured
on arm E5:

    residual_scale = 0.0067

and re-scoring ``prior + alpha*delta`` on the seven vessels the biochem deploy score is
decided on gives

    alpha       0.0    1.0    2.0    3.0    4.0    6.0    8.0   12.0
    gateJ%     79.0   80.7   83.3   85.8   91.4   88.8   85.0   76.7
    rel-L2    .1909  .1909  .1910  .1912  .1916  .1927  .1940  .1973

i.e. the shipped model sits at 80.7 when its own head, unchanged in direction and merely
scaled, reaches 91.4 -- and rel-L2 moves in the fourth decimal, so this is not a trade.  The
head knew where the prior was wrong; it was never told how wrong.

**This is a checkpoint transform, not an architecture change.**  ``delta`` is linear in
``residual_scale``, so multiplying that one tensor by ``alpha`` reproduces the sweep above --
no new flag, no new inference path, and the calibrated checkpoint loads through the existing
loader.  Linear to 3e-5 of ``max|delta|`` rather than bit-exactly, because ``outer_iters=3``
feeds the decoded field back into the equilibrium solve; that is four decades below anything
that could move a choice of alpha, and ``src/tests/test_residual_scale_calibration.py`` fails
if it stops being true.

**Where alpha may be fitted.**  On vessels the flow model has already trained on.  Those are
"seen" under the cross-fit accounting (``scripts/stage_a/crossfit_halves.py``), so calibrating
there adds no leak that is not already declared, and it leaves the flow-holdout panel clean for
the generalisation read.  Fitting alpha on the deploy cohort would be tuning on the evaluation.

    python scripts/calibrate_residual_scale.py \
        --ckpt outputs/runs/E5_band_gateup/kinematics_best.pth \
        --out  outputs/runs/E5_band_gateup/kinematics_best_calibrated.pth
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

#: The sweep is flat below 1 and turns over between 4 and 6, so a uniform ladder over [0, 8]
#: brackets the optimum without wasting forward passes on a region that cannot win.
DEFAULT_ALPHAS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)


def _state_dict(raw: dict) -> dict:
    for key in ("model_state_dict", "state_dict"):
        if key in raw:
            return raw[key]
    return raw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--stems", default="",
                    help="tuning vessels; default is the flow model's own deploy training pool, "
                         "which is where alpha may legally be fitted")
    ap.add_argument("--alphas", default="")
    ap.add_argument("--decays", default="",
                    help="envelope decays to try; default sweeps the trained one plus "
                         "24/36/48/72, which confine the residual to the wall band")
    ap.add_argument("--core-tol", type=float, default=0.0,
                    help="fraction by which the calibrated field may exceed the prior's core "
                         "rel-L2.  Default 0 admits only points that tie the prior to floating "
                         "point; raise it (1e-3 is a reasonable start) when that is too strict "
                         "to admit any real head.")
    ap.add_argument("--report", default="", help="write the sweep as JSON")
    args = ap.parse_args(argv)

    from src.config import NodeFeat
    from src.training.train_kinematics_predictor import _selection_gain_mode
    from src.utils.kinematics_inference import clamped_width_priors, load_kinematics_predictor
    from src.utils.kinematics_select_packs import load_selection_packs, use_stems
    from src.utils.kinematics_selection import wall_shear_selection_metrics

    alphas = ([float(x) for x in args.alphas.split(",") if x.strip()]
              if args.alphas.strip() else list(DEFAULT_ALPHAS))

    if args.stems.strip():
        stems = use_stems(args.stems)
    else:
        from scripts.stage_a.crossfit_halves import legal_pool
        stems = use_stems(legal_pool())
    print("[i] tuning alpha on %d vessel(s): %s" % (len(stems), ", ".join(stems)), flush=True)

    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = _state_dict(raw)
    if "residual_scale" not in sd:
        raise SystemExit(
            "%s has no `residual_scale` -- it was trained without `residual_rezero`, so there "
            "is no single tensor the residual amplitude is linear in and this calibration does "
            "not apply." % args.ckpt)
    base = float(sd["residual_scale"].reshape(-1)[0])
    print("[i] trained residual_scale = %.6g" % base, flush=True)

    prior_source = str(raw.get("prior_source") or "fem")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_kinematics_predictor(checkpoint=Path(args.ckpt), device=device)
    model.eval()
    gain = _selection_gain_mode()
    graphs = load_selection_packs(prior_source=prior_source, verbose=False)
    if not graphs:
        raise SystemExit("no tuning packs loaded")

    # One forward pass per vessel: `delta` is linear in `residual_scale`, so the whole sweep is
    # arithmetic on a single prediction.  Re-running the DEQ per alpha would cost 12x and would
    # also let solver noise into a curve that is a straight line by construction.
    deltas, priors, keep = [], [], []
    with torch.no_grad():
        for g in graphs:
            gg = g.clone().to(device)
            with clamped_width_priors(gg) as gc:
                out = model(gc, solver="anderson")
            pred = (out[0] if isinstance(out, tuple) else out)[:, :2].detach().cpu().double()
            prior = g.x[:, NodeFeat.UV_PRIOR].double()
            deltas.append(pred - prior)
            priors.append(prior)
            keep.append(g)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- the second knob -------------------------------------------------------------------
    # Scaling the head UP helps the wall gate and hurts everything off it: measured end to end,
    # alpha=4 moved the held-out wall deploy score +0.079 (p=0.034) and the off-wall score
    # -0.162 (p=0.011).  That is not a surprise -- `corr(delta, e)` is +0.25 in the wall band
    # and ~0 globally -- so a uniform alpha buys signal in the band and 4x the noise in the core.
    # The envelope decay is what confines the residual to the band:
    #
    #     env(sdf) = (1 - exp(-bc_lambda*sdf)) * exp(-decay*sdf)
    #
    # and `delta` is exactly proportional to it, so re-decaying a prediction already made at
    # `decay_trained` costs no forward pass:
    #
    #     delta(decay) = delta_trained * exp(-(decay - decay_trained) * sdf)
    #
    # Both knobs are already in the checkpoint (`residual_scale`, `model_config.
    # bc_envelope_decay`), so the calibrated artifact is still a plain checkpoint.
    trained_decay = float((raw.get("model_config") or {}).get("bc_envelope_decay", 0.0))
    decays = ([float(x) for x in args.decays.split(",") if x.strip()]
              if args.decays.strip() else [trained_decay, 24.0, 36.0, 48.0, 72.0])

    sdfs = [g.x[:, NodeFeat.SDF].reshape(-1).double() for g in keep]
    cores = [sdf > float(torch.quantile(sdf, 0.4)) for sdf in sdfs]

    def core_rel_l2(pred_list):
        """Velocity rel-L2 over the outer 60% of the lumen by wall distance.

        The off-wall half of the deploy score reads the field the wall gate does not, so the
        band metric alone cannot see the damage a large alpha does there.  This is the guard.
        """
        num = den = 0.0
        for uv, g, m in zip(pred_list, keep, cores):
            gt = (g.y[0] if g.y.dim() == 3 else g.y)[:, 0:2].double()
            num += float(((uv - gt)[m] ** 2).sum())
            den += float((gt[m] ** 2).sum())
        return (num / den) ** 0.5 if den > 0 else float("nan")

    base_core = core_rel_l2(list(priors))
    print("", flush=True)
    print("[i] FEM prior alone: core rel-L2 %.4f  (the guard -- no arm may exceed it)"
          % base_core, flush=True)
    print("%8s %8s %9s %9s %11s" % ("alpha", "decay", "gateJ%", "gateJ", "coreRelL2"),
          flush=True)

    rows = []
    for d in decays:
        shape = [torch.exp(-(d - trained_decay) * sdf).unsqueeze(1) for sdf in sdfs]
        for a in alphas:
            preds = [prior + a * sh * delta
                     for prior, sh, delta in zip(priors, shape, deltas)]
            ms = [wall_shear_selection_metrics(uv.float(), g, gain=gain)
                  for uv, g in zip(preds, keep)]

            def agg(key, _ms=ms):
                vs = [m[key] for m in _ms
                      if key in m and m[key] == m[key] and np.isfinite(m[key])]
                return float(np.mean(vs)) if vs else float("nan")

            row = dict(alpha=a, decay=d, gate_jaccard_frac=agg("gate_jaccard_frac"),
                       gate_jaccard=agg("gate_jaccard"), dsrx_scale=agg("dsrx_scale"),
                       core_rel_l2=core_rel_l2(preds))
            rows.append(row)
            print("%8.2f %8.1f %9.1f %9.3f %11.4f"
                  % (a, d, 100.0 * row["gate_jaccard_frac"], row["gate_jaccard"],
                     row["core_rel_l2"]), flush=True)

    def frac(r):
        v = r["gate_jaccard_frac"]
        return v if v == v else -1.0

    # Maximise the wall gate SUBJECT TO not degrading the core against the prior.
    #
    # Two things this guard is NOT.  It is not a proxy for the off-wall deploy score: measured
    # end to end, alpha=4 cost 0.162 of off-wall (p=0.011) while moving core rel-L2 in the
    # SIXTH decimal, because the off-wall features are built by a nearest-WALL-node owner rule
    # and therefore inherit the band, not the core (RGP_DEQ_REPAIR_PLAN.md s18.7).  And it is
    # not a substitute for measuring: it bounds how far the field may drift from the FEM solve
    # in the region the head has no signal in, nothing more.
    #
    # `alpha = 0` is the prior itself, so it satisfies any guard exactly and would win every
    # tie -- which is how half A of the cross-fit selected "no head at all".  A calibration
    # that returns the null model is a result, not a setting, so it is excluded here and the
    # caller is told when nothing beat it.
    tol = 1.0 + max(float(args.core_tol), 1e-9)
    feasible = [r for r in rows if r["core_rel_l2"] <= base_core * tol and r["alpha"] > 0.0]
    if not feasible:
        raise SystemExit("no (alpha, decay) pair with alpha > 0 keeps the core within %.3f%% of "
                         "the prior; widen --decays or raise --core-tol"
                         % (100 * float(args.core_tol)))
    best = max(feasible, key=frac)
    alpha, decay = float(best["alpha"]), float(best["decay"])
    at_one = next(r for r in rows if r["alpha"] == 1.0 and r["decay"] == trained_decay)
    print("", flush=True)
    print("[i] best feasible: alpha %.2f decay %.1f -> gateJ%% %.1f, core rel-L2 %.4f "
          "(prior %.4f)  |  as trained: gateJ%% %.1f"
          % (alpha, decay, 100 * best["gate_jaccard_frac"], best["core_rel_l2"], base_core,
             100 * at_one["gate_jaccard_frac"]), flush=True)
    null = next(r for r in rows if r["alpha"] == 0.0 and r["decay"] == trained_decay)
    if frac(best) <= frac(null):
        print("[i] WARNING no positive alpha beats the prior alone (%.1f vs %.1f gateJ%%) -- "
              "this head earns nothing on these vessels and the arm should be reconsidered "
              "rather than shipped at alpha %.2f."
              % (100 * frac(best), 100 * frac(null), alpha), flush=True)
    loose = max(rows, key=frac)
    if (loose["alpha"], loose["decay"]) != (alpha, decay):
        print("[i] the UNCONSTRAINED optimum is alpha %.2f decay %.1f (gateJ%% %.1f) but its "
              "core rel-L2 is %.4f against the prior's %.4f -- rejected"
              % (loose["alpha"], loose["decay"], 100 * loose["gate_jaccard_frac"],
                 loose["core_rel_l2"], base_core), flush=True)

    if args.report:
        p = Path(args.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(ckpt=args.ckpt, stems=stems,
                                     trained_residual_scale=base, alpha=alpha, decay=decay,
                                     trained_decay=trained_decay,
                                     core_rel_l2_prior=base_core, sweep=rows),
                                indent=2), encoding="utf-8")
        print("[OK] wrote %s" % p, flush=True)

    if not args.out:
        print("[i] no --out: nothing written", flush=True)
        return 0

    new = copy.deepcopy(raw)
    _state_dict(new)["residual_scale"] = sd["residual_scale"] * alpha
    # The decay lives in the manifest and the loader reads it from there, so re-decaying is
    # a metadata edit: the weights are untouched and the architecture is unchanged.
    mc = dict(new.get("model_config") or {})
    if mc:
        mc["bc_envelope_decay"] = decay
        new["model_config"] = mc
    # Provenance, so a calibrated checkpoint can never be mistaken for a run's own output.
    new["residual_scale_calibration"] = dict(
        source_checkpoint=str(args.ckpt), alpha=alpha, trained_residual_scale=base,
        calibrated_residual_scale=base * alpha, tuned_on=stems,
        trained_bc_envelope_decay=trained_decay, calibrated_bc_envelope_decay=decay,
        core_rel_l2_prior=base_core, core_rel_l2_calibrated=best["core_rel_l2"],
        gate_frac_at_1=at_one["gate_jaccard_frac"], gate_frac_at_alpha=best["gate_jaccard_frac"],
    )
    new["run_note"] = (str(new.get("run_note") or "")
                       + " | residual_scale x%.2f, bc_envelope_decay %.1f->%.1f, %d tuning vessels"
                       % (alpha, trained_decay, decay, len(stems))).strip(" |")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new, out)
    print("[OK] wrote %s  (residual_scale %.6g -> %.6g)" % (out, base, base * alpha), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
