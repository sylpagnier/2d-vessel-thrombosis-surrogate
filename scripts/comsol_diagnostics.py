"""One-shot COMSOL audit: what the solve can tell us that we are not currently asking for.

    python scripts/comsol_diagnostics.py --stems vessel_0,vessel_5,vessel_7

**Why.**  ``AnchorGenerator._evaluate_at_coords`` asks COMSOL for exactly four things --
``u, v, p, mu_final`` -- and EVERY shear quantity in this project is then reconstructed by
finite-differencing that sampled velocity: ``sr`` once, ``dsrx`` twice.  COMSOL carries
``spf.sr`` and ``d(spf.sr,x)`` natively at FEM accuracy and has never been asked for either.

The measured problem is that wall ``dsrx`` AMPLITUDE in the corpus is 0.13x deployment's, and
the model converges correctly to what it is shown (RGP_DEQ_REPAIR_PLAN.md §16.10).  Two very
different causes remain open, and only COMSOL can separate them:

    A. the flow really is smoother in these vessels  -> fix the SAMPLER (wall_noise_*)
    B. our reconstruction destroys it                -> fix the EXPORT, and supervise on
                                                        COMSOL's own shear instead

This prints, per vessel, COMSOL's own wall ``sr`` / ``dsrx`` beside our MLS reconstruction from
the same sampled velocity.  If COMSOL's spread is much larger, it is B -- the solve had the
signal and the pipeline threw it away afterwards -- and the corpus was never the binding
constraint.

Also dumps discretisation, stabilisation and mesh statistics so the template can be judged
rather than assumed, and re-measures the true-vs-interpolated mid-side gap from the same solve.

Runs on the generation box: needs a COMSOL *server*, so a client-only install fails at
``mph.start()`` with "Could not find a supported Comsol installation".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: Everything worth asking a Laminar Flow solution for.  Each is probed independently and a
#: failure is recorded rather than fatal -- expression names vary by COMSOL version and by what
#: the template's physics node actually defines.
PROBE_EXPRESSIONS = [
    ("u", "velocity x -- exported today"),
    ("v", "velocity y -- exported today"),
    ("p", "pressure -- exported today"),
    ("mu_final", "viscosity, the template's own variable -- exported today"),
    ("spf.sr", "COMSOL's OWN shear rate; we reconstruct this by differencing u,v"),
    ("d(spf.sr,x)", "COMSOL's OWN d(sr)/dx -- the gate's dominant branch, never exported"),
    ("d(spf.sr,y)", "COMSOL's OWN d(sr)/dy"),
    ("ux", "du/dx at FEM accuracy"),
    ("uy", "du/dy -- the dominant wall-shear term"),
    ("vx", "dv/dx"),
    ("vy", "dv/dy"),
    ("spf.U", "velocity magnitude"),
    ("spf.cellRe", "cell Reynolds; >2 means the cell is convection-dominated"),
    ("spf.mu", "the physics node's viscosity (vs the template's mu_final)"),
    ("spf.divU", "divergence -- how incompressible the discrete solution actually is"),
]

DEPLOY_FIT_DSRX_SD = 717.7   # FIT median, data/reference/deploy_wall_shear_band.json
DEPLOY_FIT_SEP = 0.0921


def _evaluate(gen, exprs, coords):
    """``({expr: array}, {expr: why_it_failed})`` at arbitrary coordinates."""
    import numpy as np

    results, failed = {}, {}
    model_j = gen.model.java
    tag = "py_diag_interp"
    for expr in exprs:
        try:
            try:
                model_j.result().numerical().remove(tag)
            except Exception:
                pass
            model_j.result().numerical().create(tag, "Interp")
            it = model_j.result().numerical(tag)
            it.set("data", "dset1")
            it.set("expr", [expr])
            it.setInterpolationCoordinates(coords.T.tolist())
            got = np.asarray(it.getData()[0], dtype=float).reshape(-1)
            if got.size != coords.shape[0]:
                failed[expr] = f"returned {got.size} values for {coords.shape[0]} points"
            else:
                results[expr] = got
        except Exception as exc:
            failed[expr] = f"{type(exc).__name__}: {str(exc)[:120]}"
        finally:
            try:
                model_j.result().numerical().remove(tag)
            except Exception:
                pass
    return results, failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", default="vessel_0,vessel_5,vessel_7")
    ap.add_argument("--orders", default="1,2", help="order_fluid values to compare")
    ap.add_argument("--out", default="outputs/comsol_diagnostics.json")
    a = ap.parse_args()

    import meshio
    import numpy as np

    from src.clot_ml.features import M_TO_CM
    from src.config import BiochemConfig, VesselConfig
    from src.core_physics.mls_gradient import build_mls_gradient, shear_rate_2d
    from src.data_gen.lib.anchor_generator import AnchorGenerator
    from src.data_gen.lib.mesh_triangle6_edges import (
        edge_index_from_mesh, mesh_undirected_edge_pairs,
    )

    bio = BiochemConfig(phase="biochem")
    lss, sgt = float(bio.lss), float(bio.sgt) / M_TO_CM
    vcfg = VesselConfig(phase="kinematics")
    mesh_dir = vcfg.mesh_input_dir
    orders = [int(o) for o in a.orders.split(",") if o.strip()]
    report: dict = {"expressions": {}, "model": {}, "vessels": {}}

    with AnchorGenerator(phase="kinematics", rheology="carreau") as gen:
        model = gen.model
        mesh_j, import_tag = gen._ensure_mesh_handles()

        # ---- 1. what the template actually IS -----------------------------------------------
        print("=" * 78 + "\n1. TEMPLATE / PHYSICS\n" + "=" * 78)
        print(f"  template          {gen.template_path.name}")
        probes = [("ShapeProperty", "order_fluid"), ("PhysicsShapeProperty", "order_fluid"),
                  ("EquationForm", "form"), ("Stabilization", "StreamlineDiffusion"),
                  ("Stabilization", "CrosswindDiffusion"),
                  ("Compressibility", "Compressibility")]
        for grp, name in probes:
            try:
                v = model.java.physics("spf").prop(grp).getString(name)
                report["model"][f"{grp}.{name}"] = str(v)
                print(f"  {grp}.{name} = {v}")
            except Exception:
                pass
        try:
            tags = [str(t) for t in model.java.physics("spf").feature().tags()]
            report["model"]["physics_features"] = tags
            print(f"  physics features  {tags}")
        except Exception:
            pass

        # ---- 2. per vessel, per element order ------------------------------------------------
        for stem in [s.strip() for s in a.stems.split(",") if s.strip()]:
            nas, msh, js = (mesh_dir / f"{stem}.{e}" for e in ("nas", "msh", "json"))
            if not (nas.exists() and msh.exists() and js.exists()):
                print(f"[skip] {stem}: missing mesh files")
                continue
            meta = json.loads(js.read_text(encoding="utf-8"))
            d_bar = float(meta["d_bar"])
            u_ref = gen.phys_cfg.get_u_ref(d_bar)
            model.parameter("D_eff", f"{d_bar:.8f} [m]")
            model.parameter("U_inlet", f"{u_ref:.8f} [m/s]")
            mesh_j.feature(import_tag).set("filename", str(nas).replace("\\", "/"))
            mesh_j.run()

            m = meshio.read(msh)
            pos = np.ascontiguousarray(m.points[:, :2], dtype=float)
            ei = edge_index_from_mesh(m).numpy()
            pairs = mesh_undirected_edge_pairs(m)
            mid_xy = 0.5 * (pos[pairs[:, 0]] + pos[pairs[:, 1]])
            wall = np.zeros(pos.shape[0], dtype=bool)
            try:
                for j, tag in enumerate(m.get_cell_data("gmsh:physical", "line")):
                    if int(tag) == vcfg.TAGS["Walls"]:
                        wall[np.asarray(m.get_cells_type("line")[j], dtype=np.int64)] = True
            except Exception as exc:
                print(f"[skip] {stem}: no wall tags ({exc})")
                continue
            if wall.sum() < 10:
                print(f"[skip] {stem}: {int(wall.sum())} wall nodes")
                continue

            print("\n" + "=" * 78)
            print(f"2. {stem}   {pos.shape[0]} P1 nodes, {int(wall.sum())} wall, "
                  f"{pairs.shape[0]} edges (P2 would be {pos.shape[0] + pairs.shape[0]})")
            print("=" * 78)
            vrec = report["vessels"].setdefault(
                stem, {"n_nodes": int(pos.shape[0]), "n_wall": int(wall.sum())})

            for order in orders:
                gen._set_element_order(order)
                gen._clear_all_solution_data()
                try:
                    model.solve()
                except Exception as exc:
                    print(f"  order={order}: SOLVE FAILED {type(exc).__name__}: {exc}")
                    continue

                res, failed = _evaluate(gen, [e for e, _ in PROBE_EXPRESSIONS], pos)
                for expr, why in failed.items():
                    report["expressions"].setdefault(expr, {"ok": False, "why": why})
                for expr in res:
                    report["expressions"][expr] = {"ok": True}
                if not {"u", "v"} <= res.keys():
                    print(f"  order={order}: u/v unavailable; skipping comparison")
                    continue

                # OURS: MLS on the sampled velocity, exactly as the packs build it
                Dx, Dy = build_mls_gradient(pos / d_bar, ei, hops=3)
                un, vn = res["u"] / u_ref, res["v"] / u_ref
                sr_rec = shear_rate_2d(Dx @ un, Dy @ un, Dx @ vn, Dy @ vn) * (u_ref / d_bar)
                dsrx_rec = (Dx @ sr_rec) / (d_bar * M_TO_CM)

                row = {"reconstructed": {
                    "wall_sr_med": float(np.median(sr_rec[wall])),
                    "wall_dsrx_sd": float(np.std(dsrx_rec[wall])),
                    "wall_sep": float((dsrx_rec[wall] < sgt).mean()),
                    "wall_fire": float(((sr_rec[wall] < lss) | (dsrx_rec[wall] < sgt)).mean()),
                }}
                r = row["reconstructed"]
                print(f"\n  order={order}  OURS (MLS on sampled u,v)     sr_med={r['wall_sr_med']:8.2f}"
                      f"  dsrx_sd={r['wall_dsrx_sd']:9.2f}  sep={100 * r['wall_sep']:5.2f}%")

                if "spf.sr" in res:
                    st = res["spf.sr"]
                    ok = np.std(st[wall]) > 0
                    row["comsol_sr"] = {
                        "wall_sr_med": float(np.median(st[wall])),
                        "corr_vs_reconstructed": float(np.corrcoef(st[wall], sr_rec[wall])[0, 1])
                        if ok else float("nan"),
                        "scale_ours_over_comsol": float(
                            np.std(sr_rec[wall]) / (np.std(st[wall]) + 1e-30)),
                    }
                    c = row["comsol_sr"]
                    print(f"           COMSOL spf.sr                sr_med={c['wall_sr_med']:8.2f}"
                          f"  corr={c['corr_vs_reconstructed']:+.3f}"
                          f"  ours/its={c['scale_ours_over_comsol']:.3f}")

                if "d(spf.sr,x)" in res:
                    dt = res["d(spf.sr,x)"] / M_TO_CM
                    ok = np.std(dt[wall]) > 0
                    row["comsol_dsrx"] = {
                        "wall_dsrx_sd": float(np.std(dt[wall])),
                        "wall_sep": float((dt[wall] < sgt).mean()),
                        "corr_vs_reconstructed": float(np.corrcoef(dt[wall], dsrx_rec[wall])[0, 1])
                        if ok else float("nan"),
                        "scale_ours_over_comsol": float(
                            np.std(dsrx_rec[wall]) / (np.std(dt[wall]) + 1e-30)),
                    }
                    c = row["comsol_dsrx"]
                    print(f"           COMSOL d(spf.sr,x) <- THE ONE"
                          f"  dsrx_sd={c['wall_dsrx_sd']:9.2f}  sep={100 * c['wall_sep']:5.2f}%"
                          f"  corr={c['corr_vs_reconstructed']:+.3f}"
                          f"  ours/its={c['scale_ours_over_comsol']:.3f}")
                    print(f"           DEPLOY reference (FIT median) dsrx_sd={DEPLOY_FIT_DSRX_SD:9.2f}"
                          f"  sep={100 * DEPLOY_FIT_SEP:5.2f}%")

                res_mid, _ = _evaluate(gen, ["u", "v"], mid_xy)
                if {"u", "v"} <= res_mid.keys():
                    um = res_mid["u"] / u_ref
                    lin_u = 0.5 * (un[pairs[:, 0]] + un[pairs[:, 1]])
                    row["midside_max_abs_dev_vs_corner_mean"] = float(
                        np.abs(um - lin_u).max() / max(np.abs(lin_u).max(), 1e-30))
                    print(f"           mid-side vs corner mean      "
                          f"|du|max={row['midside_max_abs_dev_vs_corner_mean']:.4f}"
                          f"   (order 1 -> ~0 by construction)")
                vrec[f"order{order}"] = row

        print("\n" + "=" * 78 + "\n3. EXPRESSION AVAILABILITY\n" + "=" * 78)
        for expr, why in PROBE_EXPRESSIONS:
            st = report["expressions"].get(expr, {"ok": False, "why": "never probed"})
            print(f"  [{'OK  ' if st.get('ok') else 'FAIL'}] {expr:<16} {why}")
            if not st.get("ok"):
                print(f"         -> {st.get('why', '')}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\n[save] {a.out}")
    print("""
HOW TO READ THIS
  `ours/its` on d(spf.sr,x) is the whole question.
    ~1.0   our MLS reconstruction is faithful, the corpus's low wall dsrx is REAL PHYSICS, and
           the fix is the sampler (VesselConfig.wall_noise_*) as currently planned.
    <<1    the solve HAS the signal and the pipeline destroys it after the fact.  Then the fix
           is to EXPORT `spf.sr` / `d(spf.sr,x)` and supervise on them, and the corpus was never
           the binding constraint.
  Compare COMSOL's own dsrx_sd against the deploy FIT median printed beside it (717.7).""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
