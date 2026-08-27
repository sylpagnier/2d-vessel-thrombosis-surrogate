"""Reproduce docs/WOUND_PROGRESS.md §14 -- why `wound_patient003` misses its wound-region clot.

Four blocks, in the order the argument runs:

  set       precision/recall on the wound region at FINAL time, where timing cannot be the
            cause, plus where the misses sit relative to the wound.
  owners    who owns the missed lumen nodes -- the wound, or healthy wall the ODE never
            ignites.
  gate      the near-wound wall gate under the t=0 field AND under the GT flow oracle at
            every step.  This is what rules out the WOUND_PROGRESS §11 neighbour trigger:
            a gate already at 1.0 has nothing left to open.
  species   GT AP amplification across the whole cohort, and the falsification of the one
            vessel-level selector the code nominates (`wall_gate_frac_vessel`).

    python scripts/diag_wound_p003_causes.py
    python scripts/diag_wound_p003_causes.py --blocks set owners
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.evaluate import domain_score  # noqa: E402
from src.clot_ml.features import adjacency, hop_distance  # noqa: E402
from src.clot_ml.locked import load_temporal_v4, predict_temporal_v4  # noqa: E402
from src.clot_ml.wound import (  # noqa: E402
    compose_with_v4, predict_wound_series, prepare_vessel, solid_mask, wound_mask,
    wound_region_masks,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.ap_closure import (  # noqa: E402
    SHIPPED, SHIPPED_DA_SCALE, make_rollout_hook,
)
from src.core_physics.clot_phi_simple import mat_si_for_gelation_from_log1p  # noqa: E402
from src.core_physics.physics_wall_model import (  # noqa: E402
    PER_M3_TO_PER_CM3, gt_flow_gate_series, integrate_mat_trajectory, t0_flow_fields,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
LOVO = REPO / "outputs/clot_ml/wound_rate/lovo.json"
STEM = "wound_patient003"


def gt_mat(data, bio):
    T = int(data.y.shape[0])
    mi = data.y_channel_names.split(",").index("Mat_log1p_nd")
    return mat_si_for_gelation_from_log1p(data.y[:, :, mi], bio).reshape(T, -1).numpy()


def first_crossing(traj, crit, T):
    hot = traj >= crit
    return np.where(hot.any(0), hot.argmax(0), T)


def blind_owners(data, gtm, crit):
    """The healthy-wall nodes that own the wound region's GT lumen clot."""
    solid, wnd = solid_mask(data), wound_mask(data)
    pos = data.x[:, :2].numpy().astype(np.float64)
    _, j = cKDTree(pos[solid]).query(pos)
    owner = np.flatnonzero(solid)[j]
    _, lumen, _ = wound_region_masks(data)
    sel = (gtm[-1] >= crit) & lumen
    return np.unique(owner[sel][~wnd[owner[sel]]]), lumen, owner


def block_set(data, T, V, gtm, bio, phys, base_name):
    times = sorted(set(list(range(0, T, 2)) + [T - 1]))
    ei = torch.tensor(data.edge_index.detach().cpu().numpy())
    wnd, solid = wound_mask(data), solid_mask(data)
    region, lumen, _ = wound_region_masks(data)
    h = hop_distance(wnd, adjacency(data.edge_index.numpy(), int(data.num_nodes)), max_h=40)

    bundle = load_temporal_v4(name=base_name)
    base = predict_temporal_v4(bundle, data, times, flow="gt")
    lo = json.loads(LOVO.read_text())["folds"][STEM]
    w = predict_wound_series(data, bio, times, g_pre=float(lo["g_pre"]),
                             g_post=float(lo["g_post"]), prepared=V)
    pred = compose_with_v4(base, w, times)["series"][times[-1]]
    gt = gt_clot_phi_at_time(data, times[-1], phys).numpy() > 0.5

    print("\n--- SET (final time -- timing cannot be the cause here) ---")
    for nm, dom in (("w_reg", region), ("w_lum", lumen)):
        tp = int((pred & gt & dom).sum())
        fp = int((pred & ~gt & dom).sum())
        fn = int((~pred & gt & dom).sum())
        print(f"   {nm}: score {domain_score(pred, gt, ei, dom, solid):.4f}"
              f"   precision {tp / max(tp + fp, 1):.3f}  recall {tp / max(tp + fn, 1):.3f}"
              f"   TP {tp} FP {fp} FN {fn}")
    miss = (~pred) & gt & lumen
    print("   missed region-lumen GT clot, by hops from the wound:")
    for a, b in ((0, 2), (3, 4), (5, 6), (7, 8)):
        band = (h >= a) & (h <= b)
        tot = int((band & gt & lumen).sum())
        if tot:
            print(f"      hops {a}-{b}: missed {int((miss & band).sum()):3d} of {tot:3d}")
    return pred, gt


def block_owners(data, T, V, gtm, bio, pred, gt):
    crit = V["C"].crit
    wnd, wall = wound_mask(data), V["wall"]
    u, lumen, owner = blind_owners(data, gtm, crit)
    miss = (~pred) & gt & lumen
    ow = owner[miss]
    g_on = first_crossing(gtm, crit, T)
    hook = make_rollout_hook(SHIPPED, bio, V["f0"].sr)
    ode, _ = integrate_mat_trajectory(data, bio, V["f0"].gate * wall,
                                      da_scale=SHIPPED_DA_SCALE, ap_closure=hook)
    o_on = first_crossing(ode, crit, T)

    print("\n--- OWNERS of the missed lumen clot ---")
    wnd_owned = miss & wnd[owner]
    print(f"   owner is the WOUND        : {int(wnd[ow].sum()):3d}"
          f"   -- geometry only: their GT onset is step "
          f"{np.median(g_on[wnd_owned]):.0f}, against the wound's "
          f"{np.median(g_on[wnd]):.0f}")
    print(f"   owner is HEALTHY WALL     : {int((~wnd[ow]).sum()):3d}"
          f"   on {u.size} distinct nodes")
    print(f"   those owners: t=0 gate {np.median(V['f0'].gate[u]):.3f}"
          f"   sr {np.median(V['f0'].sr[u]):.1f} /s (lss {float(bio.lss)})"
          f"   GT onset {np.median(g_on[u]):.0f}"
          f"   ODE ignites {int((o_on[u] < T).sum())}/{u.size}")


def block_gate(data, T, V, gtm, bio):
    crit = V["C"].crit
    wall = V["wall"]
    u, _, _ = blind_owners(data, gtm, crit)
    g_on = first_crossing(gtm, crit, T)
    early = wall & (g_on <= 5)

    print("\n--- GATE (this is what rules out the WOUND_PROGRESS 11 trigger) ---")
    print(f"   t=0 gate on the {int(early.sum())} wall nodes GT gels by step 5: "
          f"median {np.median(V['f0'].gate[early]):.3f} -- already saturated")
    G = gt_flow_gate_series(data, bio, hops=3)
    print("   the SAME nodes under the GT flow oracle:  "
          + "  ".join(f"s{s}:{np.median(G[min(s, T - 1)][early]):.2f}"
                      for s in (0, 2, 10, 40, T - 1)))
    print("   -> nothing left to open there; the deficit is a RATE, not a gate.")
    print(f"   the {u.size} blind owners instead start at gate 0, and the oracle DOES open "
          f"them (frac > 0):  "
          + "  ".join(f"s{s}:{float((G[min(s, T - 1)][u] > 0).mean()):.2f}"
                      for s in (0, 10, 20, 40, T - 1)))


def block_species(bio):
    print("\n--- SPECIES (and the falsification of the only vessel-level selector) ---")
    pool = json.loads(
        (REPO / "outputs/clot_ml/locked/clot_gnn_v5/manifest.json").read_text())
    stems = ["wound_patient001", "wound_patient002", "wound_patient003"] + [
        s for s in pool["training_pool"] if (PACKS / f"{s}.pt").exists()]
    rows = []
    for stem in stems:
        data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
        names = data.y_channel_names.split(",")
        sc = bio.get_species_scales(device="cpu")
        wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
        ap = (torch.expm1(data.y[:, :, names.index("AP_log1p_nd")].clamp(-10, 8)).numpy()
              * float(sc[1]) * PER_M3_TO_PER_CM3)
        f0 = t0_flow_fields(data, bio, hops=3, flow_source="gt")
        amp = ap.max(0)[wall] / np.maximum(ap[0][wall], 1e-30)
        rows.append((stem, float((f0.gate[wall] > 0).mean()),
                     float(np.median(amp)), float(np.percentile(amp, 90))))
    rows.sort(key=lambda r: -r[1])
    print(f"   {'vessel':22s} {'gate frac':>10s} {'AP amp med':>11s} {'AP amp p90':>11s}")
    for stem, gf, md, p9 in rows[:8]:
        tag = "   <-- WOUND" if stem.startswith("wound") else ""
        print(f"   {stem:22s} {gf:10.3f} {md:11.3f} {p9:11.3f}{tag}")
    hi = [r for r in rows if not r[0].startswith("wound") and r[1] >= 0.30]
    print("   cohort vessels as gated as wound_patient003 (0.352): "
          + ", ".join(f"{r[0]} ({r[1]:.3f}, AP amp {r[2]:.3f})" for r in hi))
    print("   -> `wall_gate_frac_vessel` does NOT predict activation.  Selector falsified.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="clot_gnn_v5",
                    help="baseline artifact whose temporal head supplies the v4/v5 arm")
    ap.add_argument("--blocks", nargs="*", default=["set", "owners", "gate", "species"],
                    choices=["set", "owners", "gate", "species"])
    args = ap.parse_args()

    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    data = torch.load(PACKS / f"{STEM}.pt", map_location="cpu", weights_only=False)
    T = int(data.y.shape[0])
    V = prepare_vessel(data, bio, flow="gt")
    gtm = gt_mat(data, bio)
    print(f"{STEM}   T={T}   wound={int(wound_mask(data).sum())}"
          f"   healthy wall={int(V['wall'].sum())}   base={args.base}")

    pred = gt = None
    if "set" in args.blocks or "owners" in args.blocks:
        pred, gt = block_set(data, T, V, gtm, bio, phys, args.base)
    if "owners" in args.blocks:
        block_owners(data, T, V, gtm, bio, pred, gt)
    if "gate" in args.blocks:
        block_gate(data, T, V, gtm, bio)
    if "species" in args.blocks:
        block_species(bio)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
