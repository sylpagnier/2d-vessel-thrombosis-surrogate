#!/usr/bin/env python
"""Does the RGP-DEQ residual head carry signal about the prior's error, or just noise?

Under the hard BC the model's whole contribution is one additive field:

    pred = prior + delta,     delta = envelope(sdf) * gain * r

and the thing it *should* be is ``e = y - prior``, the prior's own error.  On the FEM-prior arm
the model scores BELOW the prior it was handed (89.1 vs 95.4 gateJ%), so ``delta`` is net
harmful -- but "harmful" has two very different causes and they call for opposite responses:

  * ``delta`` is uncorrelated with ``e``       -- the head learned nothing; damping it just
                                                  converges to the prior and there is no arm.
  * ``delta`` is correlated but MIS-SCALED     -- the head knows where the prior is wrong and
                                                  overshoots; a shrinkage recovers the signal.

So this reports, per vessel and pooled:

  ``corr``      corr(delta, e) over both velocity components -- is there any signal at all
  ``ratio``     |delta| / |e| in median -- how far the amplitude is off
  ``alpha*``    the least-squares scalar <delta,e>/<delta,delta> that best uses this delta
  ``gain*``     the rel-L2 improvement over the prior AT alpha*, as a fraction of the error the
                prior leaves.  This is the ceiling on what any rescaling of THIS head can buy.

and then re-scores the metric the project selects on (``gateJ%``) along an alpha sweep, because
a rel-L2 improvement that does not move the gate is not worth having.

    python -m src.tools.diagnostics residual-head-audit \
        --ckpt outputs/runs/E1_prior_fem/kinematics_best.pth --prior fem
"""
from __future__ import annotations

from src.tools.diagnostics._common import bootstrap

import argparse
import json
from pathlib import Path

import numpy as np
import torch

NAN = float("nan")

#: Shrinkage factors swept on the model's own residual field.  0.0 IS the prior alone and 1.0 is
#: the shipped model, so the sweep contains both endpoints of the comparison it exists to make.
ALPHAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
#: Past 1.0 the sweep EXTRAPOLATES the head's own field.  It answers a question the trained scale
#: cannot: if the metric is still improving at 1.0, the head is under-scaled and the limit is the
#: optimisation, not the signal -- if it turns over, 1.0 is already at or past the useful point.


def _corr(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 3 or a.std() < 1e-14 or b.std() < 1e-14:
        return NAN
    return float(np.corrcoef(a, b)[0, 1])


def _wall_band(data, hops=3):
    n = int(data.num_nodes)
    row, col = data.edge_index
    band = data.mask_wall.reshape(-1).bool().clone()
    for _ in range(hops):
        acc = torch.zeros(n, dtype=torch.bool)
        acc.index_put_((row,), band[col], accumulate=False)
        band = band | acc
    return band.numpy()


def _score_gate(uv, graph, gain):
    from src.utils.kinematics_selection import wall_shear_selection_metrics

    return wall_shear_selection_metrics(torch.as_tensor(uv, dtype=torch.float32), graph, gain=gain)


def main(argv=None):
    bootstrap()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prior", default="fem")
    ap.add_argument("--gain", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    import os

    if args.gain:
        os.environ["KINEMATICS_SELECT_GAIN"] = args.gain
    from src.data_gen.lib.legal_priors import COL_U_PRIOR, COL_V_PRIOR
    from src.training.train_kinematics_predictor import _selection_gain_mode
    from src.utils.kinematics_inference import clamped_width_priors, load_kinematics_predictor
    from src.utils.kinematics_select_packs import load_selection_packs

    gain = _selection_gain_mode()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_kinematics_predictor(checkpoint=Path(args.ckpt), device=device)
    model.eval()
    graphs = load_selection_packs(prior_source=args.prior, verbose=False)
    print("[i] %d packs, prior=%s, gain=%r" % (len(graphs), args.prior, gain), flush=True)

    rows = []
    sweep = {a: [] for a in ALPHAS}
    for g in graphs:
        stem = str(getattr(g, "graph_stem", "?"))
        with torch.no_grad():
            gg = g.clone().to(device)
            with clamped_width_priors(gg) as gc:
                out = model(gc, solver="anderson")
            pred = (out[0] if isinstance(out, tuple) else out)[:, :2].detach().cpu().double().numpy()
        prior = g.x[:, [COL_U_PRIOR, COL_V_PRIOR]].detach().cpu().double().numpy()
        y = (g.y[0] if g.y.dim() == 3 else g.y)[:, :2].detach().cpu().double().numpy()

        delta = pred - prior          # everything the model contributed
        e = y - prior                 # everything it should have contributed
        band = _wall_band(g, hops=3)

        dd = float((delta * delta).sum())
        alpha = float((delta * e).sum() / dd) if dd > 0 else NAN
        n_e = float(np.linalg.norm(e))
        # Residual error at the best possible scaling of THIS delta, as a fraction of the
        # prior's own error.  1.0 = the head is useless; 0.0 = it explains the prior's error.
        resid_at_alpha = (float(np.linalg.norm(e - alpha * delta)) / n_e) if n_e > 0 else NAN
        md, me = np.linalg.norm(delta, axis=1), np.linalg.norm(e, axis=1)
        row = dict(
            stem=stem,
            corr=_corr(delta, e),
            corr_band=_corr(delta[band], e[band]),
            ratio=float(np.median(md[me > 1e-12]) / max(np.median(me[me > 1e-12]), 1e-30)),
            alpha_star=alpha,
            frac_err_left_at_alpha=resid_at_alpha,
            prior_rel_l2=float(n_e / max(np.linalg.norm(y), 1e-30)),
        )
        for a in ALPHAS:
            uv = prior + a * delta
            m = _score_gate(uv, g, gain)
            m["rel_l2"] = float(np.linalg.norm(uv - y) / max(np.linalg.norm(y), 1e-30))
            sweep[a].append(m)
        rows.append(row)
        print("%-14s corr=%+.3f (band %+.3f)  |delta|/|e|=%6.2f  alpha*=%+.3f  "
              "err left at alpha*=%.3f" % (stem, row["corr"], row["corr_band"], row["ratio"],
                                           row["alpha_star"], row["frac_err_left_at_alpha"]),
              flush=True)

    def med(key):
        v = [r[key] for r in rows if r[key] == r[key] and np.isfinite(r[key])]
        return float(np.median(v)) if v else NAN

    def agg(ms, key):
        v = [m[key] for m in ms if key in m and m[key] == m[key] and np.isfinite(m[key])]
        return float(np.mean(v)) if v else NAN

    print("\n=== residual head, pooled over %d vessels ===" % len(rows), flush=True)
    print("corr(delta, e)            %+.3f   (wall band %+.3f)" % (med("corr"), med("corr_band")),
          flush=True)
    print("|delta| / |e|  (median)   %.2f" % med("ratio"), flush=True)
    print("alpha*                    %+.3f" % med("alpha_star"), flush=True)
    print("error left at alpha*      %.3f  of the prior's own error "
          "(1.00 = head is useless)" % med("frac_err_left_at_alpha"), flush=True)

    print("\n=== alpha sweep: pred = prior + alpha * delta ===", flush=True)
    print("%8s %9s %9s %10s %10s" % ("alpha", "gateJ%", "gateJ", "relL2", "dsrxScale"), flush=True)
    for a in ALPHAS:
        ms = sweep[a]
        print("%8.2f %9.1f %9.3f %10.4f %10.3f"
              % (a, 100.0 * agg(ms, "gate_jaccard_frac"), agg(ms, "gate_jaccard"),
                 agg(ms, "rel_l2"), agg(ms, "dsrx_scale")), flush=True)

    if args.out:
        o = Path(args.out)
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text(json.dumps(dict(
            ckpt=args.ckpt, prior=args.prior, per_vessel=rows,
            sweep={str(a): dict(gate_jaccard_frac=agg(sweep[a], "gate_jaccard_frac"),
                                gate_jaccard=agg(sweep[a], "gate_jaccard"),
                                rel_l2=agg(sweep[a], "rel_l2"),
                                dsrx_scale=agg(sweep[a], "dsrx_scale")) for a in ALPHAS},
        ), indent=2), encoding="utf-8")
        print("[OK] wrote " + str(o), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
