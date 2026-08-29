"""A/B the two COMSOL models on ONE geometry: is the gap the vessels, or the template?

Every measurement so far compares synthetic vessels solved in `phase1_template.mph` against
patient vessels solved in `phase2_*.mph`, so geometry and template are perfectly confounded.
COMSOL's own ground truth says the corpus's wall `dsrx` is 27-188 against deployment's 718
(`scripts/comsol_diagnostics.py`), and that has been read as "the vessel designs are smoother".
It is equally consistent with "the kinematics template solves a different problem".

The two models were built at different times and they are NOT obviously the same physics:

    phase1 (kinematics)   dynamicviscosity = userdef, mu = mu_final
                          mu_final = mu_inf + (mu_0-mu_inf)*(1+(lambda_cy*spf.sr)^a_yasuda)
                                              ^((n_index-1)/a_yasuda)
    phase2 (biochem)      nonNewtonianModels = 'Carreau'  (COMSOL's built-in)
                          mu_inf = mu_b*(mu2(FI)+mu1(Mat))   <- STATE-DEPENDENT
    length unit           phase1 = m;  phase2 meshes are in cm

So: import a DEPLOY mesh into the kinematics template, solve it there, and compare against what
the biochem model produced for the same vessel.  Same geometry, same mesh, different template --
the one control that separates the two.

TWO STEPS, because the deploy packs live on the training box and COMSOL lives on the other one.

1. ON THE COMSOL BOX (the template already has the deploy mesh imported):

     python scripts/exp_template_ab.py export --template comsol_models/phase1_ab_test.mph \\
         --d-bar 0.0159418 --u-ref 0.0893280 --mesh-unit cm --out outputs/ab_p041_phase1.npz

   `--d-bar` / `--u-ref` are patient041's own, so the inlet BC matches what the biochem model
   was driven with; without that the comparison measures the boundary condition, not the physics.

2. BACK ON THE TRAINING BOX, with that .npz copied over:

     python scripts/exp_template_ab.py compare --npz outputs/ab_p041_phase1.npz --stem patient041

HOW TO READ IT
  wall `dsrx` spread similar    -> the template is not the difference; the vessel DESIGNS are,
                                   and the sampler (VesselConfig.wall_noise_*) is the lever.
  phase1 much smoother          -> the kinematics template is solving a smoother problem, the
                                   corpus was never the constraint, and the fix is the TEMPLATE.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: What to pull out of the A/B solve.  `spf.sr` and `d(spf.sr,x)` are COMSOL's own -- audited
#: available on the generation box -- so the comparison does not depend on our reconstruction.
EXPORT_EXPRS = ["u", "v", "p", "spf.sr", "d(spf.sr,x)", "d(spf.sr,y)", "spf.mu"]


def cmd_export(a) -> int:
    import numpy as np

    from src.data_gen.lib.anchor_generator import AnchorGenerator

    with AnchorGenerator(phase="kinematics", rheology="carreau",
                         template_path=str(REPO / a.template)) as gen:
        model = gen.model
        print(f"[i] template {Path(a.template).name}")
        for grp, name in (("ShapeProperty", "order_fluid"),
                          ("Stabilization", "StreamlineDiffusion"),
                          ("Stabilization", "CrosswindDiffusion")):
            try:
                print(f"    {grp}.{name} = {model.java.physics('spf').prop(grp).getString(name)}")
            except Exception:
                pass
        gen._set_element_order(a.order)

        # The inlet BC must be the one the biochem model was driven with, or the A/B measures
        # the boundary condition rather than the physics.
        model.parameter("D_eff", f"{a.d_bar:.10f} [m]")
        model.parameter("U_inlet", f"{a.u_ref:.10f} [m/s]")
        print(f"[i] D_eff={a.d_bar:.6g} m   U_inlet={a.u_ref:.6g} m/s   order_fluid={a.order}")

        if a.mesh_file:
            mesh_j, import_tag = gen._ensure_mesh_handles()
            mesh_j.feature(import_tag).set("filename", str(Path(a.mesh_file)).replace("\\", "/"))
            mesh_j.run()
            print(f"[i] re-imported mesh {a.mesh_file}")

        if a.coords_npz:
            zc = np.load(a.coords_npz, allow_pickle=True)
            coords = np.ascontiguousarray(
                np.stack([zc["x"], zc["y"]], axis=-1).astype(float))
            print(f"[i] evaluating at {coords.shape[0]} SUPPLIED coordinates, bbox "
                  f"{coords.min(0).round(5).tolist()} .. {coords.max(0).round(5).tolist()}")
            gen._clear_all_solution_data()
            print("[i] solving ...")
            model.solve()
            got = gen._evaluate_exprs(EXPORT_EXPRS, coords)
            fields = {e.replace("(", "_").replace(")", "").replace(",", "_").replace(".", "_"): v
                      for e, v in zip(EXPORT_EXPRS, got)}
            out = Path(a.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, x=coords[:, 0], y=coords[:, 1], d_bar=a.d_bar, u_ref=a.u_ref,
                     mesh_unit="m", coords_supplied=True, order_fluid=a.order,
                     template=Path(a.template).name, **fields)
            sr = fields["spf_sr"]
            nan = int(np.sum(~np.isfinite(sr)))
            print(f"[i] spf.sr: median {np.nanmedian(sr):.2f}  p90 {np.nanpercentile(sr, 90):.2f}"
                  f"  non-finite {nan} (points outside the domain)")
            print(f"[save] {out}   ({coords.shape[0]} points)")
            return 0

        try:
            n_v = int(model.java.mesh("mesh1").getNumVertex())
            coords = np.asarray(model.java.mesh("mesh1").getVertex(), dtype=float)
            coords = coords.T if coords.shape[0] in (2, 3) else coords
            coords = np.ascontiguousarray(coords[:, :2])
            print(f"[i] mesh vertices {n_v}, coord bbox "
                  f"{coords.min(0).round(5).tolist()} .. {coords.max(0).round(5).tolist()}")
        except Exception as exc:
            print(f"[ERR] could not read mesh vertices: {type(exc).__name__}: {exc}")
            return 1

        gen._clear_all_solution_data()
        print("[i] solving ...")
        model.solve()

        got = gen._evaluate_exprs(EXPORT_EXPRS, coords)
        fields = {e.replace("(", "_").replace(")", "").replace(",", "_").replace(".", "_"): v
                  for e, v in zip(EXPORT_EXPRS, got)}
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, x=coords[:, 0], y=coords[:, 1], d_bar=a.d_bar, u_ref=a.u_ref,
                 mesh_unit=a.mesh_unit, order_fluid=a.order,
                 template=Path(a.template).name, **fields)
        sr = fields["spf_sr"]
        print(f"[i] spf.sr over ALL nodes: median {np.median(sr):.2f}  p90 {np.percentile(sr, 90):.2f}")
        print(f"[save] {out}   ({coords.shape[0]} points, {len(EXPORT_EXPRS)} fields)")
        print("\nCopy this .npz to the training box and run:\n"
              f"  python scripts/exp_template_ab.py compare --npz {out} --stem patient041")
    return 0


def cmd_compare(a) -> int:
    import numpy as np
    import torch
    from scipy.spatial import cKDTree

    from src.clot_ml.features import M_TO_CM
    from src.config import BiochemConfig
    from src.core_physics.mls_gradient import build_mls_gradient, node_positions, shear_rate_2d

    z = np.load(a.npz, allow_pickle=True)
    d_bar = float(z["d_bar"])
    u_ref = float(z["u_ref"])
    # COMSOL reports coordinates in the MODEL's length unit, which is not necessarily the unit
    # the mesh file was written in -- reading `--mesh-unit` as gospel scaled an already-metre
    # bbox by 0.01 and matched 2 of 14295 nodes.  Detect it from the geometry instead.
    ab_xy = np.stack([z["x"], z["y"]], axis=-1).astype(float)
    unit_scale = 1.0
    print(f"[i] {a.npz}: template={z['template']} order={z['order_fluid']} "
          f"{ab_xy.shape[0]} points, mesh_unit={z['mesh_unit']}")

    pack = torch.load(REPO / f"data/processed/graphs_biochem_anchors/{a.stem}.pt",
                      map_location="cpu", weights_only=False)
    y = pack.y[0] if pack.y.dim() == 3 else pack.y
    wall = pack.mask_wall.reshape(-1).bool().numpy()
    pk_xy = pack.x[:, 0:2].numpy() * float(pack.d_bar.reshape(-1)[0])

    if not bool(z.get("coords_supplied", False)):
        span_ab = float(np.ptp(ab_xy[:, 0]))
        span_pk = float(np.ptp(pk_xy[:, 0]))
        unit_scale = span_pk / span_ab if span_ab > 0 else 1.0
        if not (0.9 < unit_scale < 1.1):
            print(f"[i] auto-scaling A/B coordinates by {unit_scale:.4g} "
                  f"(x-span {span_ab:.4g} -> {span_pk:.4g} m)")
        ab_xy = ab_xy * unit_scale

    # Match by position; both sides are in metres now.
    dist, idx = cKDTree(ab_xy).query(pk_xy)
    if bool(z.get("coords_supplied", False)):
        matched = np.isfinite(z["spf_sr"][idx]) & (dist <= 1e-9)
        print("[i] coordinates were supplied to COMSOL -- no matching error by construction")
    else:
        tol = 0.5 * float(np.median(np.linalg.norm(np.diff(pk_xy, axis=0), axis=1)) + 1e-12)
        matched = (dist <= max(tol, 1e-6)) & np.isfinite(z["spf_sr"][idx])
    print(f"[i] matched {int(matched.sum())}/{len(pk_xy)} pack nodes to A/B points "
          f"(median dist {np.median(dist):.3e} m); wall matched "
          f"{int((matched & wall).sum())}/{int(wall.sum())}")
    if (matched & wall).sum() < 20:
        print("[WARN] few wall matches -- check --mesh-unit and that it is the same geometry")

    bio = BiochemConfig(phase="biochem")
    lss, sgt = float(bio.lss), float(bio.sgt) / M_TO_CM
    w = matched & wall

    # --- arm 1: the BIOCHEM solve, as the packs store it (reconstructed from u,v) -----------
    Dx, Dy = build_mls_gradient(node_positions(pack), pack.edge_index.numpy(), hops=3)
    dbp = float(pack.d_bar.reshape(-1)[0]); urp = float(pack.u_ref.reshape(-1)[0])
    up, vp = y[:, 0].double().numpy(), y[:, 1].double().numpy()
    sr_bio = shear_rate_2d(Dx @ up, Dy @ up, Dx @ vp, Dy @ vp) * (urp / dbp)
    dsrx_bio = (Dx @ sr_bio) / (dbp * M_TO_CM)

    # --- arm 2: the KINEMATICS template on the same geometry, COMSOL's own shear ------------
    sr_ab = z["spf_sr"][idx]
    dsrx_ab = z["d_spf_sr_x"][idx] / (unit_scale * M_TO_CM)   # per model-length -> per cm
    # and reconstructed the same way the packs are, for a like-for-like third column
    u_ab = z["u"][idx] / u_ref
    v_ab = z["v"][idx] / u_ref
    sr_ab_rec = shear_rate_2d(Dx @ u_ab, Dy @ u_ab, Dx @ v_ab, Dy @ v_ab) * (urp / dbp)
    dsrx_ab_rec = (Dx @ sr_ab_rec) / (dbp * M_TO_CM)

    def row(name, sr, dsrx):
        return dict(arm=name, sr_med=float(np.median(sr[w])), dsrx_sd=float(np.std(dsrx[w])),
                    sep=float((dsrx[w] < sgt).mean()),
                    fire=float(((sr[w] < lss) | (dsrx[w] < sgt)).mean()))

    rows = [row("biochem template (the pack)", sr_bio, dsrx_bio),
            row("kinematics template, COMSOL sr", sr_ab, dsrx_ab),
            row("kinematics template, reconstructed", sr_ab_rec, dsrx_ab_rec)]
    print(f"\n{'arm':<38}{'wall sr_med':>13}{'wall dsrx_sd':>14}{'sep%':>8}{'fire%':>8}")
    for r in rows:
        print(f"{r['arm']:<38}{r['sr_med']:>13.2f}{r['dsrx_sd']:>14.2f}"
              f"{100 * r['sep']:>8.2f}{100 * r['fire']:>8.2f}")
    ratio = rows[1]["dsrx_sd"] / max(rows[0]["dsrx_sd"], 1e-30)
    print(f"\n  kinematics / biochem wall dsrx spread = {ratio:.3f}")
    print("""
  ~1.0   the template is NOT the difference.  The vessel DESIGNS are, and the lever is
         VesselConfig.wall_noise_* tuned against scripts/comsol_diagnostics.py.
  <<1    the kinematics template solves a smoother problem on the SAME geometry.  The corpus
         was never the binding constraint and the fix is the TEMPLATE -- compare viscosity
         model, stabilisation and inlet condition against phase2 before regenerating anything.""")
    Path(a.out).write_text(json.dumps({"npz": str(a.npz), "stem": a.stem,
                                       "ratio_kine_over_biochem": ratio, "rows": rows},
                                      indent=2), encoding="utf-8")
    print(f"\n[save] {a.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="run on the COMSOL box")
    e.add_argument("--template", default="comsol_models/phase1_ab_test.mph")
    e.add_argument("--mesh-file", default="", help="re-import this NAS/mesh first (optional)")
    e.add_argument("--coords-npz", default="",
                   help="evaluate at THESE coordinates (metres) instead of COMSOL's mesh "
                        "vertices -- pass the deploy pack's own nodes so the comparison needs "
                        "no nearest-neighbour matching at all")
    e.add_argument("--d-bar", type=float, required=True, help="the deploy vessel's own d_bar, in metres")
    e.add_argument("--u-ref", type=float, required=True, help="the deploy vessel's own u_ref, in m/s")
    e.add_argument("--mesh-unit", default="cm", choices=("m", "cm"))
    e.add_argument("--order", type=int, default=2)
    e.add_argument("--out", default="outputs/ab_phase1_on_deploy.npz")
    e.set_defaults(func=cmd_export)

    c = sub.add_parser("compare", help="run on the training box")
    c.add_argument("--npz", required=True)
    c.add_argument("--stem", default="patient041")
    c.add_argument("--out", default="outputs/exp_template_ab.json")
    c.set_defaults(func=cmd_compare)

    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
