"""Does COMSOL's velocity element order change the wall shear the clot gate reads?

The synthetic kinematics corpus is solved with `phase1_template.mph`, which sets
`order_fluid = P1+P1`.  EVERY deployment vessel is solved P2+P1
(`comsol_models/phase2_nowound_*.mph`).  With linear velocity elements the profile inside the
first cell off the wall is linear by construction, so the wall shear rate is an element average
and its along-wall derivative -- `dsrx`, the gate's dominant branch at deployment -- is
whatever the piecewise-linear field happens to leave behind.

Same vessel, same mesh, same coordinates, only `order_fluid` differs.

**Runs on the generation machine.**  It needs a COMSOL *server*; a client-only install fails at
`mph.start()` with "Could not find a supported Comsol installation".

    python scripts/exp_comsol_element_order.py --stems vessel_0,vessel_5,vessel_7

Read the `dsrx_sd` ratio column.  Deployment sits 10.7x above the corpus on that number; if the
element order carries a large part of it, set `PhysicsConfig.comsol_order_fluid = 2` and
regenerate rather than trying to reweight the loss around it.
"""
import argparse, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ap = argparse.ArgumentParser()
ap.add_argument("--stems", default="vessel_0,vessel_5,vessel_7")
ap.add_argument("--out", default="outputs/exp_element_order.json")
ap.add_argument("--p2-nodes", action="store_true",
                help="also evaluate COMSOL at the MID-SIDE coordinates and compare against the "
                     "corner-mean labels the pipeline fabricates today")
a = ap.parse_args()

import meshio, numpy as np

from src.clot_ml.features import M_TO_CM
from src.config import BiochemConfig, VesselConfig
from src.core_physics.mls_gradient import build_mls_gradient, shear_rate_2d
from src.data_gen.lib.anchor_generator import AnchorGenerator
from src.data_gen.lib.mesh_triangle6_edges import edge_index_from_mesh

bio = BiochemConfig(phase="biochem")
lss, sgt = float(bio.lss), float(bio.sgt) / M_TO_CM
vcfg = VesselConfig(phase="kinematics")
mesh_dir = vcfg.mesh_input_dir





def p2_node_set(pos, ei):
    """``(pos_p2, ei_p2, mid_ends)`` -- corners then one mid-side per undirected edge.

    Same convention as `src/data_gen/lib/p2_elevation.elevate_to_p2`: corner<->mid-side
    half-edges only, no corner-corner edge, original corner indices preserved.
    """
    pairs = np.unique(np.sort(ei.T, axis=1), axis=0)
    a, b = pairs[:, 0], pairs[:, 1]
    n, m = pos.shape[0], pairs.shape[0]
    mid = np.arange(n, n + m)
    pos_p2 = np.vstack([pos, 0.5 * (pos[a] + pos[b])])
    src = np.concatenate([a, mid, b, mid])
    dst = np.concatenate([mid, a, mid, b])
    return pos_p2, np.stack([src, dst]), pairs


def wall_metrics(pos, ei, u, v, u_ref, d_bar, wall):
    Dx, Dy = build_mls_gradient(pos, ei, hops=3)
    sr = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * (u_ref / d_bar)
    dsrx = (Dx @ sr) / (d_bar * M_TO_CM)
    lo, sep = sr[wall] < lss, dsrx[wall] < sgt
    # A zero shear field is not a result.  COMSOL's own wall shear on these vessels runs
    # 20-200 1/s; anything near zero means the operator or the evaluation is broken, not that
    # the vessel is stagnant -- and it would otherwise print as `fire=100%` and look like data.
    if float(np.median(sr[wall])) < 1e-3:
        raise RuntimeError(
            f"wall shear rate is ~0 (median {float(np.median(sr[wall])):.3e} 1/s). "
            "The MLS operator or the COMSOL evaluation is broken; these numbers are not usable."
        )
    return dict(sr_med=float(np.median(sr[wall])), dsrx_sd=float(np.std(dsrx[wall])),
                fire=float((lo | sep).mean()), low=float(lo.mean()), sep=float(sep.mean()),
                sep_only=float((sep & ~lo).sum() / max((lo | sep).sum(), 1)))


stems = [s.strip() for s in a.stems.split(",") if s.strip()]
rows = []
with AnchorGenerator(phase="kinematics", rheology="carreau") as gen:
    model = gen.model
    mesh_j, import_tag = gen._ensure_mesh_handles()
    for stem in stems:
        nas = mesh_dir / f"{stem}.nas"
        msh = mesh_dir / f"{stem}.msh"
        js = mesh_dir / f"{stem}.json"
        if not (nas.exists() and msh.exists() and js.exists()):
            print(f"[skip] {stem}: missing mesh files"); continue
        meta = json.loads(js.read_text(encoding="utf-8"))
        d_bar = float(meta["d_bar"])
        u_ref = gen.phys_cfg.get_u_ref(d_bar)
        model.parameter("D_eff", f"{d_bar:.8f} [m]")
        model.parameter("U_inlet", f"{u_ref:.8f} [m/s]")
        mesh_j.feature(import_tag).set("filename", str(nas).replace("\\", "/"))
        mesh_j.run()

        m = meshio.read(msh)
        pos = np.ascontiguousarray(m.points[:, :2], dtype=float)
        # ALREADY (2, 2E) -- a PyG edge_index, not edge pairs.  Transposing it made
        # `_khop_neighbors` read the first two EDGES as the row/col arrays, every stencil
        # came out empty, and the operator was all zeros: `sr_med` 0 and `fire` 100%.
        ei = edge_index_from_mesh(m).numpy()
        wall = np.zeros(pos.shape[0], dtype=bool)
        try:
            for j, tag in enumerate(m.get_cell_data("gmsh:physical", "line")):
                if int(tag) == vcfg.TAGS["Walls"]:
                    wall[np.asarray(m.get_cells_type("line")[j], dtype=np.int64)] = True
        except Exception as e:
            print(f"[skip] {stem}: no wall tags ({e})"); continue
        if wall.sum() < 10:
            print(f"[skip] {stem}: {wall.sum()} wall nodes"); continue

        row = {"stem": stem, "n": int(pos.shape[0]), "wall_n": int(wall.sum())}
        for order in (1, 2):
            grp = gen._set_element_order(order)
            gen._clear_all_solution_data()
            try:
                model.solve()
            except Exception as e:
                print(f"[fail] {stem} order={order}: {type(e).__name__}: {e}")
                row[f"order{order}"] = None
                continue
            u, v, p, mu = gen._evaluate_at_coords(pos)
            # COMSOL's `Interp.getData()` comes back 2-D (1, N); the MLS operators are (N, N).
            u_nd = np.asarray(u, dtype=float).reshape(-1) / u_ref
            v_nd = np.asarray(v, dtype=float).reshape(-1) / u_ref
            if u_nd.size != pos.shape[0]:
                print(f"[skip] {stem} order={order}: got {u_nd.size} values for "
                      f"{pos.shape[0]} nodes")
                row[f"order{order}"] = None
                continue
            row[f"order{order}"] = wall_metrics(pos / d_bar, ei, u_nd, v_nd, u_ref, d_bar, wall)
            row[f"order{order}"]["prop_group"] = grp

            if a.p2_nodes:
                # THE QUESTION THIS SCRIPT EXISTS FOR.  `KINEMATICS_ELEVATE_P2` sets every
                # mid-side label to the mean of its two corners, which makes the field
                # piecewise-linear along each half-edge BY CONSTRUCTION -- and `dsrx`, the gate
                # branch that decides deployment, is a second derivative of it.  Ask COMSOL for
                # the mid-side values instead and compare, on the SAME node set and operator.
                pos2, ei2, pairs = p2_node_set(pos, ei)
                wall2 = np.concatenate([wall, wall[pairs[:, 0]] & wall[pairs[:, 1]]])
                u2, v2, _, _ = gen._evaluate_at_coords(pos2)
                u2 = np.asarray(u2, float).reshape(-1) / u_ref
                v2 = np.asarray(v2, float).reshape(-1) / u_ref
                nan = ~np.isfinite(u2) | ~np.isfinite(v2)
                if nan.any():   # a point that fell outside the domain keeps the corner mean
                    u2[nan] = np.concatenate([u_nd, 0.5 * (u_nd[pairs[:, 0]] + u_nd[pairs[:, 1]])])[nan]
                    v2[nan] = np.concatenate([v_nd, 0.5 * (v_nd[pairs[:, 0]] + v_nd[pairs[:, 1]])])[nan]
                u_lin = np.concatenate([u_nd, 0.5 * (u_nd[pairs[:, 0]] + u_nd[pairs[:, 1]])])
                v_lin = np.concatenate([v_nd, 0.5 * (v_nd[pairs[:, 0]] + v_nd[pairs[:, 1]])])
                row[f"order{order}_p2_true"] = wall_metrics(
                    pos2 / d_bar, ei2, u2, v2, u_ref, d_bar, wall2)
                row[f"order{order}_p2_interp"] = wall_metrics(
                    pos2 / d_bar, ei2, u_lin, v_lin, u_ref, d_bar, wall2)
                row[f"order{order}_p2_midside_rel"] = float(
                    np.abs(u2[len(u_nd):] - u_lin[len(u_nd):]).max()
                    / max(np.abs(u_lin[len(u_nd):]).max(), 1e-30))
                t = row[f"order{order}_p2_true"]; i = row[f"order{order}_p2_interp"]
                print(f"  {stem} order={order} P2 nodes: TRUE dsrx_sd={t['dsrx_sd']:.4g} "
                      f"sep={100*t['sep']:.2f}%   INTERP dsrx_sd={i['dsrx_sd']:.4g} "
                      f"sep={100*i['sep']:.2f}%   ratio={t['dsrx_sd']/max(i['dsrx_sd'],1e-30):.2f}"
                      f"   midside |du|max={row[f'order{order}_p2_midside_rel']:.3f}")
            print(f"  {stem} order={order}: " + "  ".join(
                f"{k}={row[f'order{order}'][k]:.4g}" for k in
                ("sr_med", "dsrx_sd", "fire", "low", "sep", "sep_only")))
        rows.append(row)

print(f"\n{'stem':<14}{'sr_med P1':>11}{'sr_med P2':>11}{'ratio':>8}"
      f"{'dsrx_sd P1':>12}{'dsrx_sd P2':>12}{'ratio':>8}{'sep% P1':>9}{'sep% P2':>9}")
for r in rows:
    o1, o2 = r.get("order1"), r.get("order2")
    if not (o1 and o2):
        continue
    print(f"{r['stem']:<14}{o1['sr_med']:>11.1f}{o2['sr_med']:>11.1f}{o2['sr_med']/max(o1['sr_med'],1e-9):>8.2f}"
          f"{o1['dsrx_sd']:>12.1f}{o2['dsrx_sd']:>12.1f}{o2['dsrx_sd']/max(o1['dsrx_sd'],1e-9):>8.2f}"
          f"{100*o1['sep']:>9.2f}{100*o2['sep']:>9.2f}")
Path(a.out).write_text(json.dumps(rows, indent=2, default=float))
print(f"\n[save] {a.out}")
