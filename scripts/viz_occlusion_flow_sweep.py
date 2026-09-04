"""How does the flow profile change as a clot grows?  An FEM occlusion sweep, as a figure.

WHAT THIS IS FOR.  The local kinematic corrector was built to answer this question and cannot
(`docs/LOCAL_KINEMATIC_CORRECTOR.md`): its diversion has `cos = -0.14` against the true one and
a magnitude ratio of ~0, i.e. it does not reroute flow around a clot in any measurable sense.
The QUESTION is still worth answering, and the FEM oracle answers it exactly -- so this produces
the picture the corrector was supposed to produce, from the solver instead of from a model.

Three panels, all from converged Carreau Navier-Stokes on the real vessel geometry under its
real fixed-flux inlet BC:

  TOP     speed field at each occlusion fraction, clot outlined.  Shows the jet forming through
          the residual lumen as the clot grows.
  MIDDLE  wall shear rate along the wall, against signed arclength from the clot centre.  This
          is the quantity the deposition gate reads, and the panel shows the two competing
          effects directly: SHIELDING under the clot (shear collapses) and ACCELERATION at its
          shoulders (shear overshoots the clot-free profile).
  BOTTOM  the same as a ratio to the clot-free field, which is where the non-monotonicity in
          `clot-shear-map-is-non-monotone` becomes visible: deeper occlusion does not mean
          lower wall shear.

    python scripts/viz_occlusion_flow_sweep.py --stem comsol001
    python scripts/viz_occlusion_flow_sweep.py --stem comsol012 --fracs 0.2 0.4 0.6 0.8
"""
from __future__ import annotations
from src.utils.paths import anchor_packs_dir, get_project_root

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO = get_project_root()

from src.clot_ml.v0 import _resolve_anchor_mesh  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.local_fem_solver import solve_local_t0_flow  # noqa: E402
from src.core_physics.mls_gradient import (  # noqa: E402
    build_mls_gradient, node_positions, shear_rate_2d)
from src.data_gen.lib.mesh_wls import solid_boundary_nodes  # noqa: E402

PACKS = anchor_packs_dir()
FIGDIR = REPO / "outputs/reports/figures/kinematics"


def pick_mid_vessel_seed(data, pos, solid: np.ndarray) -> int:
    """Wall node near the vessel midline (used to place a synthetic occlusion)."""
    solid_idx = np.flatnonzero(solid)
    if solid_idx.size == 0:
        return int(np.argmin(pos[:, 0]))
    x_med = float(np.median(pos[solid_idx, 0]))
    return int(solid_idx[np.argmin(np.abs(pos[solid_idx, 0] - x_med))])


def _clot_mask(
    data,
    pos: np.ndarray,
    solid: np.ndarray,
    width: np.ndarray,
    sdf: np.ndarray,
    seed: int,
    arc_nodes: int,
    frac: float,
) -> np.ndarray:
    """Synthetic clot occupancy mask for FEM occlusion sweeps."""
    n = len(pos)
    ei = data.edge_index.numpy()
    src, dst = ei[0], ei[1]
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in zip(src, dst):
        adj[a].append(b)
        adj[b].append(a)

    lumen = (~solid) & (sdf < width * 0.55)
    if not lumen.any():
        lumen = ~solid

    wall_band = np.zeros(n, dtype=bool)
    if solid[seed]:
        wall_band[seed] = True
        frontier = [seed]
        for _ in range(max(int(arc_nodes), 0)):
            nxt = []
            for a in frontier:
                for b in adj[a]:
                    if solid[b] and not wall_band[b]:
                        wall_band[b] = True
                        nxt.append(b)
            frontier = nxt

    center = pos[wall_band].mean(axis=0) if wall_band.any() else pos[seed]
    dist = np.linalg.norm(pos - center, axis=1)
    candidates = np.flatnonzero(lumen)
    if candidates.size == 0:
        return wall_band
    order = candidates[np.argsort(dist[candidates])]
    n_take = max(1, int(round(float(frac) * candidates.size)))
    out = np.zeros(n, dtype=bool)
    out[order[:n_take]] = True
    out |= wall_band
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="comsol001")
    ap.add_argument("--fracs", nargs="*", type=float, default=[0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--clot-mu", type=float, default=0.68)
    ap.add_argument("--arc-nodes", type=int, default=40)
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phys, bio = PhysicsConfig(), BiochemConfig(phase="biochem")
    lss = float(bio.lss)
    data = torch.load(PACKS / f"{args.stem}.pt", map_location="cpu", weights_only=False)
    if getattr(data, "graph_stem", None) is None:
        data.graph_stem = args.stem
    mesh_path = _resolve_anchor_mesh(data)

    pos = node_positions(data)
    ei = data.edge_index.numpy()
    Dx, Dy = build_mls_gradient(pos, ei, hops=args.hops)
    solid = solid_boundary_nodes(data)
    wall = data.mask_wall.reshape(-1).bool().numpy()
    ch = {c: i for i, c in enumerate(data.x_channel_names.split(","))}
    xs = data.x.numpy()
    width = np.maximum(xs[:, ch["width_nd"]].astype(np.float64), 1e-9)
    sdf = np.abs(xs[:, ch["sdf_nd"]].astype(np.float64))
    u_ref = float(data.u_ref.reshape(-1)[0])
    d_bar = float(data.d_bar.reshape(-1)[0])
    u_gt = data.y[0, :, 0:2].numpy()
    seed = pick_mid_vessel_seed(data, pos, solid)

    def sr_of(U):
        return shear_rate_2d(Dx @ U[:, 0], Dy @ U[:, 0],
                             Dx @ U[:, 1], Dy @ U[:, 1]) * (u_ref / d_bar)

    print(f"[{args.stem}] clot-free solve ...", flush=True)
    u0 = solve_local_t0_flow(mesh_path, data, phys, max_iters=200, tol=1e-9,
                             u_gt_inlet_nd=u_gt, verbose=False)
    u0 = (u0.numpy() if torch.is_tensor(u0) else u0).astype(np.float64)
    sr0 = sr_of(u0)

    # Signed arclength ALONG THE WALL, as a graph geodesic from the clot's centre wall node.
    #
    # A straight-line projection onto the mean flow direction -- the obvious first choice, and
    # what this script did originally -- collapses on a curved vessel: the wall wraps through
    # the bend so distant nodes project to the same coordinate, and it mixes the near and far
    # wall together.  The profile panels came out as a single vertical stripe at s = 0.
    # Hop distance within the WALL subgraph is the right metric: it follows the boundary.
    occ_ref = _clot_mask(data, pos, solid, width, sdf, seed, args.arc_nodes, max(args.fracs))
    cw = occ_ref & wall
    com = pos[cw].mean(axis=0)
    tv = u0[cw].mean(axis=0)
    t_hat = tv / max(float(np.linalg.norm(tv)), 1e-12)

    src, dst = ei[0], ei[1]
    keep_e = wall[src] & wall[dst]
    n_nodes = len(wall)
    adj = [[] for _ in range(n_nodes)]
    for a_, b_ in zip(src[keep_e], dst[keep_e]):
        adj[a_].append(b_)
        adj[b_].append(a_)
    centre = int(np.flatnonzero(cw)[np.argmin(np.linalg.norm(pos[cw] - com, axis=1))])
    hop = np.full(n_nodes, -1, dtype=np.int64)
    hop[centre] = 0
    frontier = [centre]
    while frontier:
        nxt = []
        for a_ in frontier:
            for b_ in adj[a_]:
                if hop[b_] < 0:
                    hop[b_] = hop[a_] + 1
                    nxt.append(b_)
        frontier = nxt
    edge_len = float(np.median(np.linalg.norm(pos[src[keep_e]] - pos[dst[keep_e]], axis=1)))
    # Sign from the WALL TANGENT at the clot centre, oriented downstream -- not from the flow
    # direction directly.  On a bend the mean flow vector can sit almost perpendicular to the
    # local wall, which collapses every band node onto one side and produced a one-sided
    # profile with no upstream half at all.
    nbrs = adj[centre]
    if len(nbrs) >= 2:
        d0 = pos[nbrs[0]] - pos[centre]
        d1 = pos[nbrs[-1]] - pos[centre]
        t_wall = d1 - d0
    else:
        t_wall = t_hat
    nrm_w = float(np.linalg.norm(t_wall))
    t_wall = t_wall / nrm_w if nrm_w > 1e-12 else t_hat
    if float(t_wall @ t_hat) < 0:
        t_wall = -t_wall                      # orient downstream
    sign = np.sign((pos - pos[centre]) @ t_wall)
    sign[sign == 0] = 1.0
    s = sign * hop * edge_len            # signed distance along the wall, ND units
    clot_radius = float(np.linalg.norm(pos[cw] - com, axis=1).max())
    reach = hop[cw].max() if (hop[cw] >= 0).any() else 0
    band = wall & (hop >= 0) & (hop <= 4 * max(int(reach), 1))
    order = np.argsort(s[band])
    sb = s[band][order]

    n = len(args.fracs)
    fig = plt.figure(figsize=(4.6 * n, 11.0), constrained_layout=True)
    gs = fig.add_gridspec(3, n, height_ratios=[1.5, 1.0, 1.0])
    ax_sr = fig.add_subplot(gs[1, :])
    ax_rt = fig.add_subplot(gs[2, :])
    def smooth(v, k=7):
        """Rolling median along the wall.

        The anchor meshes are P2, so vertices and mid-side nodes ALTERNATE along the boundary
        and carry visibly different shear -- the raw profile is a sawtooth between two
        interleaved curves rather than noise.  A short rolling median collapses the two without
        moving the envelope.
        """
        if len(v) < k:
            return v
        pad = k // 2
        vv = np.pad(v, pad, mode="edge")
        return np.array([np.median(vv[i:i + k]) for i in range(len(v))])

    ax_sr.plot(sb, smooth(sr0[band][order]), "k-", lw=2.0, label="clot-free", zorder=5)
    ax_rt.axhline(1.0, color="k", lw=1.2, ls="-", zorder=5)

    rows = []
    cmap = plt.get_cmap("plasma")
    for i, frac in enumerate(args.fracs):
        occ = _clot_mask(data, pos, solid, width, sdf, seed, args.arc_nodes, frac)
        dmu = occ.astype(np.float64) * float(args.clot_mu)
        print(f"  frac={frac:.2f}  n_clot={int(occ.sum())}  solving ...", flush=True)
        uf = solve_local_t0_flow(mesh_path, data, phys, max_iters=200, tol=1e-9,
                                 u_gt_inlet_nd=u_gt, delta_mu_nodal_si=dmu, verbose=False)
        uf = (uf.numpy() if torch.is_tensor(uf) else uf).astype(np.float64)
        sr = sr_of(uf)
        col = cmap(0.15 + 0.7 * i / max(n - 1, 1))

        a = fig.add_subplot(gs[0, i])
        spd = np.linalg.norm(uf, axis=1)
        # Euclidean for the SPATIAL panels: `s` is a wall geodesic and is undefined off the
        # boundary, so it cannot select the lumen nodes these panels are made of.
        sel = np.linalg.norm(pos - com, axis=1) <= 5.0 * clot_radius
        sc = a.scatter(pos[sel, 0], pos[sel, 1], c=spd[sel], s=5, cmap="viridis",
                       vmin=0, vmax=float(np.linalg.norm(u0, axis=1).max()))
        a.scatter(pos[occ & sel, 0], pos[occ & sel, 1], s=7, color="crimson", alpha=0.85)
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(f"occlusion {frac:.0%}   |u| max {spd.max():.3f} m/s", fontsize=10)
        if i == n - 1:
            fig.colorbar(sc, ax=a, fraction=0.04, label="speed [m/s]")

        ax_sr.plot(sb, smooth(sr[band][order]), color=col, lw=1.6, label=f"{frac:.0%}")
        with np.errstate(divide="ignore", invalid="ignore"):
            rt = np.where(sr0[band][order] > 1e-9, sr[band][order] / sr0[band][order], np.nan)
        ax_rt.plot(sb, smooth(rt), color=col, lw=1.6, label=f"{frac:.0%}")
        m = occ & wall
        rows.append(dict(frac=float(frac), n_clot=int(occ.sum()),
                         sr_med_clot=float(np.median(sr[m])),
                         sr0_med_clot=float(np.median(sr0[m])),
                         ratio_med=float(np.median(sr[m]) / max(np.median(sr0[m]), 1e-12)),
                         sr_max_band=float(sr[band].max()),
                         sr0_max_band=float(sr0[band].max()),
                         speed_max=float(spd.max())))

    half = float(np.abs(s[cw]).max()) or edge_len
    for a, ttl, yl in ((ax_sr, "wall shear rate along the wall", "sr [1/s]"),
                       (ax_rt, "wall shear RATIO to the clot-free field", "sr / sr0")):
        a.axvspan(-half, half, color="crimson", alpha=0.10, label="clot extent")
        a.set_xlabel("signed arclength from clot centre, along the flow  [nd]")
        a.set_ylabel(yl)
        a.set_title(ttl, fontsize=11)
        a.grid(alpha=0.25)
        a.legend(fontsize=8, ncol=max(2, n // 2 + 1))
    ax_sr.axhline(lss, color="tab:red", ls="--", lw=1.2)
    ax_sr.annotate(f"lss = {lss:.0f} 1/s (gate fires below)", (sb[0], lss),
                   fontsize=8, color="tab:red", va="bottom")
    ax_sr.set_yscale("log")
    fig.suptitle(f"{args.stem} -- how the flow profile changes as a clot occludes the lumen "
                 f"(FEM, clot mu = {args.clot_mu} Pa.s)", fontsize=13)

    out = Path(REPO / (args.out or f"outputs/reports/figures/kinematics/"
                                   f"occlusion_sweep_{args.stem}.png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)

    print(f"\n{'frac':>6} {'n_clot':>7} {'sr med clot':>12} {'ratio':>8} "
          f"{'sr max band':>12} {'vs clot-free':>13}")
    for r in rows:
        print(f"{r['frac']:>6.2f} {r['n_clot']:>7d} {r['sr_med_clot']:>12.2f} "
              f"{r['ratio_med']:>8.3f} {r['sr_max_band']:>12.2f} "
              f"{r['sr_max_band']/max(r['sr0_max_band'],1e-9):>13.3f}")
    js = Path(REPO / f"outputs/viz_occlusion_sweep_{args.stem}.json")
    js.write_text(json.dumps(rows, indent=2))
    print(f"\n[fig]  {out}")
    print(f"[save] {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
