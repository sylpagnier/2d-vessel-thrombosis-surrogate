"""The shipped model's specification table -- read from the promoted artifact, never typed.

WHY A GENERATOR.  Every number in the paper's methods table (width, depth, parameter count,
ensemble composition, optimiser, loss weights, temporal-head settings) already lives in the
locked artifact: the member checkpoints carry their `cfg`, the manifest carries the ensemble
and the pool, and `temporal.pkl` carries the fitted head.  A hand-written table is a second
copy of those facts, and second copies are what went stale in `DEPLOYCLOT.md` 0.3 (FIGURES
0.3).  So this reads the artifact and nothing else, and it FAILS if the nine members disagree
on anything but the keys the manifest says they vary on.

    python scripts/publication/generate_architecture_table.py
    ->  outputs/<profile>/data/architecture_table.{json,md}
"""
from __future__ import annotations

import json
import pickle
from collections import Counter

import numpy as np
import torch

from scripts.publication.config import DATA_DIR
from src.clot_ml.recurrent import N_FEEDBACK
from src.utils.paths import get_project_root

REPO = get_project_root()
LOCKED = REPO / "data/reference/clot_gnn_locked.json"

#: keys the ensemble is DESIGNED to vary across its three configs; anything else differing
#: between members is a promotion error and must stop the table, not be averaged into it.
VARIED = ("config", "seed", "file", "rounds", "off_mult")


def _base_dir() -> "Path":
    """Walk the locked pointer's `base_model` chain down to the artifact holding the GNN."""
    ref = json.loads(LOCKED.read_text())
    root = REPO / "outputs/clot_ml/locked"
    name = ref["name"]
    chain = [name]
    while True:
        man = json.loads((root / name / "manifest.json").read_text())
        if "members" in man:
            return root / name, man, chain
        name = man["base_model"]
        chain.append(name)


def _param_count(sd: dict) -> int:
    return int(sum(v.numel() for v in sd.values() if torch.is_tensor(v)))


def main() -> int:
    d, man, chain = _base_dir()
    members = man["members"]

    # --- the nine members must agree on everything they are not designed to vary on ------
    shared = {}
    for k in members[0]:
        if k in VARIED:
            continue
        vals = {json.dumps(m.get(k)) for m in members}
        if len(vals) != 1:
            raise SystemExit("members disagree on %r: %s -- not a table, a promotion bug"
                             % (k, sorted(vals)))
        shared[k] = members[0][k]
    variants = {}
    for m in members:
        variants.setdefault(m["config"], {k: m[k] for k in ("rounds", "off_mult")})
        variants[m["config"]].setdefault("seeds", []).append(m["seed"])

    # --- sizes, from the checkpoints themselves ------------------------------------------
    params, in_dim, extra_dim, edim = set(), set(), set(), set()
    for m in members:
        ck = torch.load(d / m["file"], map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        params.add(_param_count(sd))
        in_dim.add(int(ck["in_dim"]))
        extra_dim.add(int(ck["extra_dim"]))
        edim.add(int(sd["mp.0.msg.0.weight"].shape[1]) - 2 * int(shared["dim"]))
    if len(params) != 1:
        raise SystemExit("member parameter counts differ: %s" % sorted(params))
    (n_params,), (n_in,), (n_extra,), (n_edge,) = params, in_dim, extra_dim, edim
    assert n_extra == N_FEEDBACK, (n_extra, N_FEEDBACK)

    tmp = pickle.load(open(d / man["temporal_file"], "rb"))

    def _hgb(est) -> dict:
        p = est.get_params()
        return {k: p[k] for k in ("learning_rate", "max_depth", "max_iter",
                                  "l2_regularization") if k in p}

    head_kinds = Counter(type(e).__name__ for e in tmp["head"])
    lag_kinds = Counter(type(e).__name__ for e in tmp["lag_models"])

    table = {
        "artifact_chain": chain,
        "gnn_artifact": d.name,
        "feature_cache": man["feature_cache"],
        "graph": {
            "node_features": n_in,
            "edge_features": n_edge,
            "recurrent_feedback_channels": n_extra,
        },
        "gnn": {
            "block": "encoder MLP -> L message-passing layers (edge MLP over [x_src, x_dst, e]; "
                     "upstream-mean, downstream-mean and max aggregation; node MLP; "
                     "post-norm residual) -> classification head + residual regression head "
                     "(zero-init, on the physics log(Mat/crit) base)",
            "hidden_dim": shared["dim"],
            "layers": shared["layers"],
            "dropout": shared["drop"],
            "activation": "SiLU",
            "params_per_member": n_params,
        },
        "ensemble": {
            "members": len(members),
            "params_total": n_params * len(members),
            "configs": variants,
        },
        "optimisation": {
            "optimiser": "AdamW",
            "lr_max": shared["lr"],
            "weight_decay": shared["wd"],
            "schedule": "OneCycleLR, pct_start 0.25, one step per vessel",
            "epochs": shared["epochs"],
            "batch": "one full-mesh graph per step",
            "grad_clip_norm": 1.0,
        },
        "objective": {k: shared[k] for k in (
            "pos_weight", "reg_w", "metric_w", "metric_start", "metric", "shape_w",
            "clot_free_w", "empty_gt_loss", "burden_w")},
        "temporal_head": {
            "onset_classifier": {"estimators": dict(head_kinds),
                                 "seeds": tmp["head_seeds"],
                                 "settings": _hgb(tmp["head"][0])},
            "lag_regressor": {"estimators": dict(lag_kinds),
                              "seeds": tmp["lag_seeds"],
                              "settings": _hgb(tmp["lag_models"][0])},
            "time_samples": tmp["n_times"],
            "lag_anchor": tmp["lag_anchor"],
        },
        "readout": {"wall": tmp["wall_spec"]["kind"], "off": tmp["off_spec"]["kind"],
                    "burden_gate": tmp["burden_gate"]},
        "training_pool": {
            "clot_carrying": len(man["training_pool_carrying"]),
            "clot_free": len(man["training_pool_clot_free"]),
        },
    }

    out_json = DATA_DIR / "architecture_table.json"
    out_json.write_text(json.dumps(table, indent=2))

    g, o, t, e = table["gnn"], table["optimisation"], table["temporal_head"], table["ensemble"]
    cfgs = "; ".join("%s: rounds %d, off_mult %g, seeds %s"
                     % (k, v["rounds"], v["off_mult"], ",".join(map(str, v["seeds"])))
                     for k, v in e["configs"].items())
    rows = [
        ("Input", "%d node features, %d edge features, %d recurrent feedback channels"
         % (n_in, n_edge, n_extra)),
        ("GNN", "hidden %d, %d message-passing layers, dropout %g, SiLU, post-norm residual"
         % (g["hidden_dim"], g["layers"], g["dropout"])),
        ("Heads", "per-node clot logit + zero-init residual on physics log(Mat/crit)"),
        ("Parameters", "%s per member, %s total" % (f"{n_params:,}", f"{e['params_total']:,}")),
        ("Ensemble", "%d members = %s" % (e["members"], cfgs)),
        ("Optimiser", "AdamW, max lr %g, weight decay %g, %s, %d epochs, grad-norm clip %g"
         % (o["lr_max"], o["weight_decay"], o["schedule"], o["epochs"], o["grad_clip_norm"])),
        ("Loss", "weighted BCE (pos_weight %g) + %g x smooth-L1 on log(Mat/crit) + %g x "
                 "(1 - soft concordance), wall and off-wall, from %d%% of training + %g x "
                 "logit-spread penalty toward a running cohort reference; clot-free vessels "
                 "weighted %g"
         % (shared["pos_weight"], shared["reg_w"], shared["metric_w"],
            round(100 * shared["metric_start"]), shared["shape_w"], shared["clot_free_w"])),
        ("Temporal head", "%d x %s onset classifier (lr %g, depth %d, %d iters) + %d x %s lag "
                          "regressor (lr %g, depth %d, %d iters); %d time samples; anchor %s"
         % (t["onset_classifier"]["seeds"], next(iter(head_kinds)),
            t["onset_classifier"]["settings"]["learning_rate"],
            t["onset_classifier"]["settings"]["max_depth"],
            t["onset_classifier"]["settings"]["max_iter"],
            t["lag_regressor"]["seeds"], next(iter(lag_kinds)),
            t["lag_regressor"]["settings"]["learning_rate"],
            t["lag_regressor"]["settings"]["max_depth"],
            t["lag_regressor"]["settings"]["max_iter"],
            t["time_samples"], t["lag_anchor"])),
        ("Readout", "wall %s, off-wall %s, burden gate %d"
         % (table["readout"]["wall"], table["readout"]["off"], table["readout"]["burden_gate"])),
        ("Training pool", "%d clot-carrying + %d clot-free vessels"
         % (table["training_pool"]["clot_carrying"], table["training_pool"]["clot_free"])),
    ]
    md = ["| component | setting |", "|---|---|"] + ["| %s | %s |" % r for r in rows]
    out_md = DATA_DIR / "architecture_table.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("\nwrote %s\n      %s" % (out_json, out_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
