"""Sweep ``predict_wound_series(lumen=...)`` end-to-end through the v5w deploy path.

``predict_wound_series`` implements four off-boundary rules and the shipped dispatcher
(:func:`~src.clot_ml.locked.predict_temporal_v4_wound`) never passes the argument, so deploy
has always run ``lumen="shell"`` -- ONE corner shell.  ``diag_wound_offwall_depth.py`` measured
why that matters: ``wound_patient003``'s off-wall GT clot is three layers deep (161 / 59 / 11)
while 001/002 are exactly one, so a one-shell rule caps 003's off-wall recall at 0.663.

``recursive`` is the mode built for this and its depth is emergent, not chosen: shell ``k``
needs ``Mat_wound >= crit / off_att**k``, so a wound reaching 9x crit (001/002) admits one
shell and 104x (003) admits more.  It is also strictly additive on top of the shipped shell 1,
so "no regression on 001/002" is a property of the construction rather than a hope -- this
script is what checks the construction actually behaves that way.

    python scripts/diag_wound_lumen_modes.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.evaluate import domain_score  # noqa: E402
from src.clot_ml.locked import (  # noqa: E402
    build_sample, load_temporal_v4_wound, predict_temporal_v4,
)
from src.clot_ml.temporal import union_ungated_stall_series  # noqa: E402
from src.clot_ml.wound import (  # noqa: E402
    compose_with_v4, predict_wound_series, prepare_vessel, solid_mask, wound_mask,
    wound_region_masks,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
STEMS = ("wound_patient001", "wound_patient002", "wound_patient003")
MODES = ("shell", "recursive", "transport", "union")
DOMS = ("wall", "off", "w_lum", "far", "full")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=list(STEMS))
    ap.add_argument("--modes", nargs="*", default=list(MODES))
    ap.add_argument("--name", default="clot_gnn_v5w")
    args = ap.parse_args()

    phys, bio = PhysicsConfig(phase="biochem"), BiochemConfig(phase="biochem")
    bundle = load_temporal_v4_wound(name=args.name)
    w = bundle["wound"]
    acc: dict[str, dict[str, list[float]]] = {}

    for stem in args.stems:
        data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
        T = int(data.y.shape[0])
        times = [0, T - 1]
        ei = torch.tensor(data.edge_index.detach().cpu().numpy())
        wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
        solid = solid_mask(data)
        off = ~solid
        _, w_lum, far = wound_region_masks(data)
        gt = gt_clot_phi_at_time(data, T - 1, phys).numpy() > 0.5
        doms = {"wall": wall, "off": off, "w_lum": w_lum, "far": far,
                "full": np.ones_like(wall)}

        over = {"wound_spec": dict(w["readout"])} if w.get("readout") else {}
        if bool(w.get("rate_in_ode", True)):
            over["wound_rate"] = (float(w["g_pre"]), float(w["g_post"]))
        b = dict(bundle)
        b["base"] = dict(bundle["base"],
                         temporal=dict(bundle["base"]["temporal"], **over))
        wr = b["base"]["temporal"].get("wound_rate")

        S = build_sample(data, bio, flow="gt", variant="v4")
        base = predict_temporal_v4(b["base"], data, times, flow="gt", sample=S)
        V = prepare_vessel(data, bio, flow="gt")

        print("=" * 104)
        print(f"{stem}  T={T}  off GT+={int((gt & off).sum())}  "
              f"w_lum GT+={int((gt & w_lum).sum())}  far GT+={int((gt & far).sum())}")
        print("  " + f"{'lumen mode':12s}" + "".join(f"{d:>9s}" for d in DOMS)
              + f"{'offTP':>7s}{'offFP':>7s}")
        for mode in args.modes:
            out = predict_wound_series(
                data, bio, times, g_pre=float(w["g_pre"]), g_post=float(w["g_post"]),
                flow="gt", off_att=float(w["off_att"]), lag_frac=float(w["lag_frac"]),
                trigger=str(w.get("trigger", "self")), k_hops=int(w.get("k_hops", 25)),
                lumen=mode, prepared=V)
            comp = compose_with_v4(base, out, times)
            series = union_ungated_stall_series(data, bio, comp["series"], times,
                                                flow="gt", wound_rate=wr)
            pred = series[T - 1]
            line = f"  {mode:12s}"
            for d in DOMS:
                sc = domain_score(pred, gt, ei, doms[d], solid)
                acc.setdefault(mode, {}).setdefault(d, []).append(sc)
                line += f"{sc:9.4f}"
            tp = int((pred & gt & off).sum())
            fp = int((pred & ~gt & off).sum())
            print(line + f"{tp:7d}{fp:7d}")

    print("=" * 104)
    print("  " + f"{'MEAN':12s}" + "".join(f"{d:>9s}" for d in DOMS))
    for mode in args.modes:
        print(f"  {mode:12s}" + "".join(
            f"{np.nanmean(acc[mode][d]):9.4f}" for d in DOMS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
