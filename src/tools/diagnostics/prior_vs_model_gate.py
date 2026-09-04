#!/usr/bin/env python
"""How much of a Stage-A arm's gate score is the model, and how much is the prior it was handed?

A run's ``ep0`` line is NOT the prior's score: the residual head is randomly initialised, so
``u = prior + envelope * r`` at epoch 0 is the prior plus noise, and it reads BELOW what the
prior alone does.  Under a FEM prior that gap is large enough to invert the conclusion -- the
prior alone can beat the model that was built on top of it -- so the honest baseline has to be
measured by feeding the prior in as if it were the prediction.

Every arm below is scored by ``wall_shear_selection_metrics`` on the SAME strided selection
subset with the SAME gain the trainer selects on, so the numbers sit beside a run's ``gateJ%``
line without translation:

    gt          COMSOL's own t=0 velocity as the prediction -- the metric's ceiling
    analytic    the Poiseuille prior alone, no model
    fem         the local FEM prior alone, no model
    <ckpt>      any checkpoint passed with --ckpt NAME=path

    python -m src.tools.diagnostics prior-vs-model-gate \
        --ckpt E0=outputs/runs/E0_prior_analytic/kinematics_best.pth \
        --ckpt E1=outputs/runs/E1_prior_fem/kinematics_best.pth
"""
from __future__ import annotations


import argparse
import json
from pathlib import Path

import numpy as np
import torch

NAN = float("nan")


def _score(pred_uv, graph, gain):
    from src.utils.kinematics_selection import wall_shear_selection_metrics

    m = wall_shear_selection_metrics(pred_uv, graph, gain=gain)
    y = graph.y[0] if graph.y.dim() == 3 else graph.y
    yv = y[:, :2].double()
    m["rel_l2"] = float((pred_uv.double() - yv).norm() / yv.norm().clamp(min=1e-30))
    return m


def _prior_uv(graph):
    from src.data_gen.lib.legal_priors import COL_U_PRIOR, COL_V_PRIOR

    return graph.x[:, [COL_U_PRIOR, COL_V_PRIOR]].detach().cpu().float()


def _gt_uv(graph):
    y = graph.y[0] if graph.y.dim() == 3 else graph.y
    return y[:, :2].detach().cpu().float()


def _agg(rows, key):
    vs = [r[key] for r in rows if key in r and r[key] == r[key] and np.isfinite(r[key])]
    return float(np.mean(vs)) if vs else NAN


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", default=[],
                    help="NAME=path; repeatable. Each is scored with its own stored prior_source.")
    ap.add_argument("--priors", default="analytic,fem",
                    help="prior-only arms to score (comma list)")
    ap.add_argument("--gain", default="", help="override KINEMATICS_SELECT_GAIN")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    import os

    if args.gain:
        os.environ["KINEMATICS_SELECT_GAIN"] = args.gain
    from src.training.train_kinematics_predictor import _selection_gain_mode
    from src.utils.kinematics_select_packs import load_selection_packs, selection_subset_stems

    gain = _selection_gain_mode()
    stems = selection_subset_stems()
    print("[i] selection subset (n=%d): %s" % (len(stems), ", ".join(stems)), flush=True)
    print("[i] gain=%r" % (gain,), flush=True)

    arms: dict[str, list[dict]] = {}

    # Prior-only arms.  Each needs the packs loaded under ITS OWN prior source, because the
    # prior block is what is being scored.
    for src in [s.strip() for s in args.priors.split(",") if s.strip()]:
        graphs = load_selection_packs(prior_source=src, verbose=False)
        arms[src] = [_score(_prior_uv(g), g, gain) for g in graphs]
        if src == [s.strip() for s in args.priors.split(",") if s.strip()][0]:
            arms["gt"] = [_score(_gt_uv(g), g, gain) for g in graphs]

    # Checkpoints.
    for spec in args.ckpt:
        if "=" not in spec:
            raise SystemExit("--ckpt takes NAME=path, got %r" % spec)
        name, path = spec.split("=", 1)
        p = Path(path)
        if not p.is_file():
            print("[WARN] %s: no such checkpoint %s" % (name, p), flush=True)
            continue
        raw = torch.load(p, map_location="cpu", weights_only=False)
        # Score the checkpoint on the prior block it was TRAINED with -- the hard BC reads the
        # prior as its base point, so a model evaluated against a different one is a different
        # function.  The trainer records it; fall back to the arm name only if it does not.
        src = str((raw.get("prior_source") or "") if isinstance(raw, dict) else "").strip()
        if not src:
            # Checkpoints written before the `kinematics_best.pth` save recorded `prior_source`
            # carry an empty string.  Guess from the arm PATH as well as the name -- `--ckpt`
            # splits on "=", so the name is often just "E1" and says nothing.
            hay = (name + " " + str(p)).lower()
            src = "fem" if "fem" in hay else "analytic"
            print("[WARN] %s: checkpoint records no prior_source; assuming %r from its path"
                  % (name, src), flush=True)
        from src.utils.kinematics_inference import load_kinematics_predictor

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_kinematics_predictor(checkpoint=p, device=device)
        model.eval()
        graphs = load_selection_packs(prior_source=src, verbose=False)
        from src.utils.kinematics_inference import clamped_width_priors

        got = []
        with torch.no_grad():
            for g in graphs:
                gg = g.clone().to(device)
                with clamped_width_priors(gg) as gc:
                    out = model(gc, solver="anderson")
                pred = out[0] if isinstance(out, tuple) else out
                got.append(_score(pred[:, :2].detach().cpu(), g, gain))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        arms["%s (prior=%s)" % (name, src)] = got

    order = ["gt"] + [k for k in arms if k != "gt"]
    print("\n%-26s %9s %9s %9s %9s %9s" % ("arm", "gateJ%", "gateJ", "dsrxCorr", "dsrxScale", "relL2"),
          flush=True)
    summary = {}
    for k in order:
        rows = arms.get(k) or []
        if not rows:
            continue
        frac = _agg(rows, "gate_jaccard_frac")
        summary[k] = dict(
            gate_jaccard_frac=frac, gate_jaccard=_agg(rows, "gate_jaccard"),
            dsrx_corr=_agg(rows, "dsrx_corr"), dsrx_scale=_agg(rows, "dsrx_scale"),
            rel_l2=_agg(rows, "rel_l2"), n=len(rows),
        )
        print("%-26s %9.1f %9.3f %9.3f %9.3f %9.4f"
              % (k, 100.0 * frac, summary[k]["gate_jaccard"], summary[k]["dsrx_corr"],
                 summary[k]["dsrx_scale"], summary[k]["rel_l2"]), flush=True)

    if args.out:
        o = Path(args.out)
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text(json.dumps(dict(stems=stems, gain=str(gain), arms=summary), indent=2),
                     encoding="utf-8")
        print("[OK] wrote " + str(o), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
