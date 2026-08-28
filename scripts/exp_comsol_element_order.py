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




def wall_metrics(pos, ei, u, v, u_ref, d_bar, wall):
    Dx, Dy = build_mls_gradient(pos, ei, hops=3)
    sr = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * (u_ref / d_bar)
    dsrx = (Dx @ sr) / (d_bar * M_TO_CM)
    lo, sep = sr[wall] < lss, dsrx[wall] < sgt
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
        ei = edge_index_from_mesh(m).T
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
