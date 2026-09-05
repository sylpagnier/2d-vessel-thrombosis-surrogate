"""Take each feature channel from the flow field that computes it better.

`RGP_DEQ_REPAIR_PLAN.md` §18.11 measured, channel by channel, what the RGP-DEQ residual does to
the off-wall feature block against plain FEM.  It is not uniform:

    log_mat_owner   0.5190 -> 0.5248     gate_owner      0.5191 -> 0.5248
    log_mat_adv     0.5099 -> 0.5311     log_mat_adv_n   0.5157 -> 0.5347
    log_mat_off_est 0.5037 -> 0.5108
    att_adv         0.1724 -> 0.1643     <- degraded
    sr_owner        0.0451 -> 0.0239     <- degraded

Every transport MAGNITUDE channel improves; the two that degrade are the two derived from the
flow's DIRECTION.  That is the signature of a residual that carries real information about how
much material moves and adds noise to which way it goes -- unsurprising, since `att_adv` is an
upstream/downstream cosine and `sr_owner` a differentiated field, and both amplify the
high-frequency part of a correction whose smoothness nothing in the objective constrains.

There is no reason a single velocity field has to serve every channel.  This writes a cache
that is the surrogate's everywhere except the named columns, which are copied from the FEM
cache.  Both caches are built from the same packs in the same node order with the same feature
fingerprint, so the splice is column-for-column and everything not named is bit-identical to
the arm it came from.

    python scripts/build_hybrid_flow_cache.py \
        --base outputs/clot_ml_cache_v5_dc_rgpcal \
        --from-fem outputs/clot_ml_cache_v5_fem \
        --out outputs/clot_ml_cache_v5_hybrid
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

#: The channels measured as degraded by the residual (§18.11).  Both are direction-derived.
DIRECTION_CHANNELS = ("att_adv", "sr_owner")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="cache supplying every channel by default")
    ap.add_argument("--from-fem", required=True, help="cache supplying --channels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--channels", default=",".join(DIRECTION_CHANNELS),
                    help="comma list of feature columns to take from --from-fem")
    args = ap.parse_args()

    base, fem, out = Path(args.base), Path(args.from_fem), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want = [c.strip() for c in args.channels.split(",") if c.strip()]

    n_ok = n_skip = 0
    for src in sorted(base.glob("*.npz")):
        other = fem / src.name
        if not other.is_file():
            print("[skip] %s: no FEM counterpart" % src.stem, flush=True)
            n_skip += 1
            continue
        zb = np.load(src, allow_pickle=True)
        zf = np.load(other, allow_pickle=True)
        cb = [str(c) for c in zb["cols"]]
        cf = [str(c) for c in zf["cols"]]
        if cb != cf:
            raise SystemExit(
                "%s: the two caches do not share a column layout, so a positional splice would "
                "silently mix channels.  Rebuild both from the same features.py." % src.stem)
        Xb, Xf = zb["X"], zf["X"]
        if Xb.shape != Xf.shape:
            raise SystemExit("%s: node counts differ (%s vs %s)" % (src.stem, Xb.shape, Xf.shape))
        X = Xb.copy()
        taken = []
        for name in want:
            if name not in cb:
                continue
            j = cb.index(name)
            X[:, j] = Xf[:, j]
            taken.append(name)
        data = {k: zb[k] for k in zb.files}
        data["X"] = X
        np.savez_compressed(out / src.name, **data)
        n_ok += 1
        print("[ok  ] %-12s spliced %s" % (src.stem, ", ".join(taken) or "nothing"), flush=True)

    print("done -> %s  (%d written, %d skipped)" % (out, n_ok, n_skip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
