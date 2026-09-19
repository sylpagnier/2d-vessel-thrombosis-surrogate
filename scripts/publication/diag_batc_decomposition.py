"""Measure the BATC vs BATC_0 gap KNOB BY KNOB, on real predictions.

`docs/publication/STORY.md` 0.3 states the two settings compound to +0.192 off-wall, decomposed as
tolerance +0.066, shape weight +0.066, the graces +0.041 and beta +0.002. That decomposition is
prose in `docs/DEPLOYCLOT.md` 0.4 with no artifact behind it -- the last unbacked number in the
story (EXPERIMENTS.md E2b). A paper that ARGUES for its own metric cannot attach unverifiable
numbers to it.

**One knob at a time, from BATC_0 toward BATC**, on one fixed set of predictions and masks, so
each contribution is measured rather than asserted. The knobs, in the order DEPLOYCLOT lists them:

    tolerance     relax_hops  2 -> 4
    shape weight  shape_w   0.5 -> 0.2
    graces        tau_abs/rho and tau_fp_abs/rho_fp  0 -> the shipped values
    beta          0.5 -> 1.0

Each row reports the gain from turning ON that knob alone (BATC_0 + knob), which is what
"worth +0.066" means. The compounded total is BATC minus BATC_0 on the same predictions, and it
is NOT the sum of the singles -- the knobs interact. Both are reported, and the artifact says
which is which.

> **SCOPE, and it is not the scope DEPLOYCLOT measured.** That +0.192 was quoted off-wall on the
> deploy cohort. This runs on whatever `--src` supplies -- by default the wound series archive,
> the one place a set of real predictions, their GT and their domains are saved together. The
> numbers here are therefore a MEASUREMENT OF THE SAME EFFECT ON A DIFFERENT SET, not a
> confirmation of DEPLOYCLOT's figure. Do not present one as reproducing the other.

    python scripts/publication/diag_batc_decomposition.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from scripts.eval_clot_ml_0 import PACKS
from src.clot_ml.severity_metric import (BATC, BATC_0, SeverityConfig, dilation_operator,
                                         severity_components)
from src.utils.paths import get_project_root

REPO = get_project_root()


def _cfg(**over) -> SeverityConfig:
    base = dict(relax_hops=BATC_0.relax_hops, beta=BATC_0.beta, shape_w=BATC_0.shape_w,
                tau_abs=BATC_0.tau_abs, rho=BATC_0.rho,
                tau_fp_abs=BATC_0.tau_fp_abs, rho_fp=BATC_0.rho_fp)
    base.update(over)
    return SeverityConfig(**base)


KNOBS = {
    "tolerance (hops 2->4)": _cfg(relax_hops=BATC.relax_hops),
    "shape weight (0.5->0.2)": _cfg(shape_w=BATC.shape_w),
    "graces (0 -> shipped)": _cfg(tau_abs=BATC.tau_abs, rho=BATC.rho,
                                  tau_fp_abs=BATC.tau_fp_abs, rho_fp=BATC.rho_fp),
    "beta (0.5->1.0)": _cfg(beta=BATC.beta),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/publication/data/wound_series_fem.npz")
    ap.add_argument("--domains", nargs="*", default=["w_reg", "w_lum", "wall"])
    ap.add_argument("--out", default="outputs/publication/data/batc_decomposition.json")
    args = ap.parse_args()

    z = np.load(REPO / args.src, allow_pickle=True)
    meta = json.loads(str(z["meta"][0]))
    stems = meta["vessels"]

    # cache the dilation operators: one per (vessel, hops)
    D_cache: dict[tuple[str, int], object] = {}
    acc: dict[str, dict[str, list]] = {d: {k: [] for k in list(KNOBS) + ["BATC_0", "BATC"]}
                                       for d in args.domains}

    for stem in stems:
        data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
        ei = np.asarray(data.edge_index)
        pred = z[f"masks|{stem}"][-1].astype(bool)
        gt = z[f"gt|{stem}"][-1].astype(bool)
        n = len(pred)
        for dom_name in args.domains:
            key = f"domain|{stem}|{dom_name}"
            if key not in z.files:
                continue
            dom = z[key].astype(bool)
            g, p = gt & dom, pred & dom
            if g.sum() == 0:
                continue
            for label, cfg in list(KNOBS.items()) + [("BATC_0", BATC_0), ("BATC", BATC)]:
                ck = (stem, int(cfg.relax_hops))
                if ck not in D_cache:
                    D_cache[ck] = dilation_operator(ei, n, hops=cfg.relax_hops)
                c = severity_components(p, g, D_cache[ck], cfg=cfg)
                v = float(c.get("score", float("nan")))
                if v == v:
                    acc[dom_name][label].append(v)

    out = {
        "scope": (f"{len(stems)} vessels from {args.src}; predictions are the SHIPPED wound arm "
                  f"at final time. NOT the deploy cohort off-wall set DEPLOYCLOT 0.4's +0.192 "
                  f"was measured on -- same effect, different set."),
        "note": ("each knob row is BATC_0 PLUS that knob alone. The compounded total is "
                 "BATC - BATC_0 and is NOT the sum of the singles; the knobs interact."),
        "vessels": stems, "domains": args.domains, "per_domain": {},
    }
    print(f"\n{'domain':<8}{'knob':<26}{'mean':>9}{'gain over BATC_0':>19}")
    for d in args.domains:
        if not acc[d]["BATC_0"]:
            continue
        b0 = float(np.mean(acc[d]["BATC_0"]))
        b1 = float(np.mean(acc[d]["BATC"]))
        row = {"BATC_0": round(b0, 4), "BATC": round(b1, 4),
               "compounded": round(b1 - b0, 4), "n": len(acc[d]["BATC_0"]), "knobs": {}}
        print(f"{d:<8}{'BATC_0 (baseline)':<26}{b0:>9.4f}{'--':>19}")
        for label in KNOBS:
            if not acc[d][label]:
                continue
            m = float(np.mean(acc[d][label]))
            row["knobs"][label] = {"mean": round(m, 4), "gain": round(m - b0, 4)}
            print(f"{'':<8}{label:<26}{m:>9.4f}{m - b0:>+19.4f}")
        print(f"{'':<8}{'BATC (all knobs)':<26}{b1:>9.4f}{b1 - b0:>+19.4f}   <- compounded")
        out["per_domain"][d] = row

    (REPO / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[i] wrote {REPO / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
