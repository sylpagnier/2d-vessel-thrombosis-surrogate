"""Are the held-out vessels INSIDE the feature distribution the ensemble was normalised on?

`clot_gnn_v4` standardises every input channel by `feature_norm.npz` -- mu/sd computed over
the 19-vessel training pool.  A vessel whose features sit outside that range is fed inputs
the network never saw at any depth, and a GNN's confidence is not reliable there.

Cross-validation cannot detect this.  In LOO/CV the held-out vessel is still one of the 19
that DEFINED mu/sd (the normaliser is fitted on the whole cache, not per fold), and it is
drawn from the same generation batch as the other 18.  A genuinely sealed vessel is the
first time either assumption is tested -- which is exactly why a CV estimate can be honest
about the weights and still not transfer.

Reports, per vessel, how far its standardised features sit from the pool, on the WALL nodes
where the readout gates live.

    python scripts/diag_feature_ood.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.clot_ml.locked import build_sample, load_ensemble, predict_scores  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
LOCKED = REPO / "outputs/clot_ml/locked/clot_gnn_v4"
# docs/SEALED_SPLIT.md -- VIZ_HALF only.  FINAL_HALF must never be opened.
FINAL_HALF = {"patient007", "patient013", "patient031", "patient043"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", default="patient042,patient001",
                    help="held-out vessels to compare against the pool (VIZ_HALF only)")
    ap.add_argument("--zcut", type=float, default=5.0)
    ap.add_argument("--save", default="outputs/diag_feature_ood.json")
    args = ap.parse_args()

    man = json.loads((LOCKED / "manifest.json").read_text())
    pool = man["training_pool"]
    extra = [a for a in args.extra.split(",") if a]
    for a in extra:
        assert a not in FINAL_HALF, "FINAL_HALF is SEALED -- docs/SEALED_SPLIT.md"

    norm = np.load(LOCKED / "feature_norm.npz", allow_pickle=True)
    mu, sd, cols = norm["mu"], norm["sd"], [str(c) for c in norm["cols"]]
    ens = load_ensemble(name="clot_gnn_v4")
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")

    rows = []
    for a in pool + extra:
        d = torch.load(PACKS / f"{a}.pt", map_location="cpu", weights_only=False)
        S = build_sample(d, bio, phys, flow="gt", variant="v4")
        w = S["wall"].astype(bool)
        Z = np.abs((S["X"][w] - mu) / np.maximum(sd, 1e-9))
        sc = predict_scores(ens, S)
        rows.append(dict(v=a, held=a in extra, med_z=float(np.median(Z)),
                         p99_z=float(np.percentile(Z, 99)), max_z=float(Z.max()),
                         frac_out=float((Z > args.zcut).mean()),
                         # per-column mean |z|, for naming the channels that drift
                         col_z={c: float(Z[:, j].mean()) for j, c in enumerate(cols)},
                         stat=float(sc[w].mean())))
        r = rows[-1]
        print("%-12s%s med|z|=%.2f  p99|z|=%6.2f  max|z|=%8.1f  frac|z|>%g = %.4f  stat=%.3f"
              % (a, "*HELD" if r["held"] else "     ", r["med_z"], r["p99_z"], r["max_z"],
                 args.zcut, r["frac_out"], r["stat"]), flush=True)

    P = [r for r in rows if not r["held"]]
    H = [r for r in rows if r["held"]]
    print("\n%-20s %8s %8s %10s" % ("", "med|z|", "p99|z|", "frac_out"))
    for nm, rs in (("POOL (n=%d)" % len(P), P), ("HELD OUT (n=%d)" % len(H), H)):
        print("%-20s %8.2f %8.2f %10.4f" % (nm, np.mean([r["med_z"] for r in rs]),
              np.mean([r["p99_z"] for r in rs]), np.mean([r["frac_out"] for r in rs])))

    # which channels drift most on the held-out vessels, relative to the pool's own spread
    pool_mean = {c: np.mean([r["col_z"][c] for r in P]) for c in cols}
    pool_sd = {c: np.std([r["col_z"][c] for r in P], ddof=1) + 1e-9 for c in cols}
    for r in H:
        drift = sorted(((r["col_z"][c] - pool_mean[c]) / pool_sd[c], c) for c in cols)[::-1]
        print("\n%s -- channels furthest outside the pool's own spread of mean|z|:" % r["v"])
        for z, c in drift[:8]:
            print("    %-22s  mean|z| %6.2f   pool %6.2f +- %5.2f   -> %+6.1f sd"
                  % (c, r["col_z"][c], pool_mean[c], pool_sd[c], z))

    out = REPO / args.save
    out.write_text(json.dumps(rows, default=float))
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
