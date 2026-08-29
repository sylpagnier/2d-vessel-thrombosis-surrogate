"""A/B the two COMSOL models on ONE geometry: is the gap the vessels, or the template?

Every measurement so far compares synthetic vessels solved in `phase1_template.mph` against
patient vessels solved in `phase2_*.mph`, so geometry and template are perfectly confounded.

The two models are NOT obviously the same physics:

    phase1 (kinematics)   std2 fluid-only transient; length unit m; built-in Carreau gel=1
                          Gmsh NAS import only (no COMSOL remesh); order_fluid=P2
    phase2 (biochem)      Carreau built-in; mu0/mu_inf gel-scaled; order_fluid=P2
                          length unit cm; same Gmsh->NAS->Import path; std2 then std1

Same geometry, different template.  At t=0, gelation is identity (mu1=1, mu2=0), so a match
here is what RGP-DEQ is asked to learn for clot_ml_0.

Use the venv interpreter (Windows Store `python` is a stub and fails as a native app):

    .venv\\Scripts\\python.exe scripts/exp_template_ab.py ...

STEPS

1. Dump the deploy pack's own nodes (metres) -- no COMSOL needed:

     python scripts/exp_template_ab.py dump-coords --stem patient041 \\
         --out outputs/ab_p041_coords.npz

2. ON THE COMSOL BOX, with the deploy mesh already imported into the kinematics template:

     python scripts/exp_template_ab.py export --template comsol_models/phase1_ab_test.mph \\
         --coords-npz outputs/ab_p041_coords.npz --d-bar 0.0159418 --u-ref 0.0893280 \\
         --out outputs/ab_p041_phase1.npz

   `--d-bar` / `--u-ref` must be the vessel's own.  `--order` is forced and READ BACK;
   the .mph file still says P1 even when the live session is P2.

3. Compare (no COMSOL):

     python scripts/exp_template_ab.py compare --npz outputs/ab_p041_phase1.npz --stem patient041

HOW TO READ IT
  wall dsrx spread ~1 and rms(v) ~1   -> template is not the gap; fix the corpus (wall_noise,
                                        true P2 labels, order_fluid=2 at generate time).
  wall dsrx ~1 but rms(v) << 1        -> bulk shear looks like phase2, secondary flow does not.
                                        RGP-DEQ trained on phase1 will miss recirculation.
  wall dsrx << 1                      -> kinematics template is the smoother problem.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXPORT_EXPRS = ["u", "v", "p", "spf.sr", "d(spf.sr,x)", "d(spf.sr,y)", "spf.mu"]


def expr_to_key(expr: str) -> str:
    return expr.replace("(", "_").replace(")", "").replace(",", "_").replace(".", "_")


def npz_scalar(z, key, default=None):
    """NpzFile has .get on numpy 2; this stays safe on 1.x and 0-d arrays."""
    files = getattr(z, "files", None)
    if files is not None:
        if key not in files:
            return default
        val = z[key]
    else:
        val = z.get(key, default) if hasattr(z, "get") else z[key] if key in z else default
        if val is default:
            return default
    if isinstance(val, np.ndarray) and val.shape == ():
        return val.item()
    return val


def detect_xy_scale(ab_xy: np.ndarray, pk_xy: np.ndarray) -> float:
    """Scale A/B coordinates onto the pack bbox.  ~1, ~100, or ~0.01 are the live cases."""
    span_ab = float(np.ptp(ab_xy[:, 0]))
    span_pk = float(np.ptp(pk_xy[:, 0]))
    if span_ab <= 0:
        return 1.0
    return span_pk / span_ab


def pack_xy_metres(pack) -> tuple[np.ndarray, float, float]:
    d_bar = float(np.asarray(pack.d_bar).reshape(-1)[0])
    u_ref = float(np.asarray(pack.u_ref).reshape(-1)[0])
    xy = np.ascontiguousarray(pack.x[:, 0:2].detach().cpu().numpy().astype(float) * d_bar)
    return xy, d_bar, u_ref


def rel_l2(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        pred, gt = pred[mask], gt[mask]
    denom = float(np.linalg.norm(gt))
    if denom < 1e-30:
        return float("nan")
    return float(np.linalg.norm(pred - gt) / denom)


def _read_order_fluid(model) -> str:
    ph = model.java.physics("spf")
    for grp in ("ShapeProperty", "PhysicsShapeProperty"):
        try:
            return str(ph.prop(grp).getString("order_fluid"))
        except Exception:
            continue
    return ""


def _model_vertex_xy(model) -> np.ndarray | None:
    try:
        coords = np.asarray(model.java.mesh("mesh1").getVertex(), dtype=float)
        coords = coords.T if coords.shape[0] in (2, 3) else coords
        return np.ascontiguousarray(coords[:, :2])
    except Exception:
        return None


def cmd_dump_coords(a) -> int:
    import torch

    pack_path = REPO / f"data/processed/graphs_biochem_anchors/{a.stem}.pt"
    pack = torch.load(pack_path, map_location="cpu", weights_only=False)
    xy, d_bar, u_ref = pack_xy_metres(pack)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, x=xy[:, 0], y=xy[:, 1], d_bar=d_bar, u_ref=u_ref,
             stem=a.stem, unit="m", n=xy.shape[0])
    print(f"[i] {a.stem}: {xy.shape[0]} nodes  d_bar={d_bar:.6g} m  u_ref={u_ref:.6g} m/s")
    print(f"    bbox m {xy.min(0).round(6).tolist()} .. {xy.max(0).round(6).tolist()}")
    print(f"[save] {out}")
    print("Copy this file to the COMSOL box and pass it as --coords-npz to export.")
    return 0


def cmd_export(a) -> int:
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
        got_order = _read_order_fluid(model)
        print(f"[i] requested order_fluid={a.order}  live={got_order or 'UNREADABLE'}")
        if got_order and str(got_order).strip() not in (str(a.order), f"P{a.order}"):
            print("[WARN] live order_fluid does not match --order; the .mph file stores P1 and "
                  "the Java set may have been ignored.  Do not trust the npz 'order_fluid' field.")

        model.parameter("D_eff", f"{a.d_bar:.10f} [m]")
        model.parameter("U_inlet", f"{a.u_ref:.10f} [m/s]")
        print(f"[i] D_eff={a.d_bar:.6g} m   U_inlet={a.u_ref:.6g} m/s")

        if a.mesh_file:
            mesh_j, import_tag = gen._ensure_mesh_handles()
            mesh_j.feature(import_tag).set("filename", str(Path(a.mesh_file)).replace("\\", "/"))
            mesh_j.run()
            print(f"[i] re-imported mesh {a.mesh_file}")

        mesh_xy = _model_vertex_xy(model)
        if mesh_xy is not None:
            print(f"[i] mesh vertices {mesh_xy.shape[0]}, bbox "
                  f"{mesh_xy.min(0).round(5).tolist()} .. {mesh_xy.max(0).round(5).tolist()}")

        coords_supplied = bool(a.coords_npz)
        if coords_supplied:
            zc = np.load(a.coords_npz, allow_pickle=True)
            coords = np.ascontiguousarray(
                np.stack([zc["x"], zc["y"]], axis=-1).astype(float))
            print(f"[i] supplied {coords.shape[0]} coords, bbox "
                  f"{coords.min(0).round(5).tolist()} .. {coords.max(0).round(5).tolist()}")
            if mesh_xy is not None:
                scale = detect_xy_scale(coords, mesh_xy)
                if not (0.5 < scale < 2.0):
                    print(f"[i] scaling supplied coords by {scale:.6g} to sit on the mesh bbox "
                          f"(phase1 length unit is metres; a cm dump is 100x too large)")
                    coords = coords * scale
        else:
            if mesh_xy is None:
                print("[ERR] could not read mesh vertices")
                return 1
            coords = mesh_xy

        gen._clear_all_solution_data()
        print("[i] solving std2 fluid-only (phase2 t=0 equivalent) ...")
        gen._run_comsol_solve()
        got = gen._evaluate_exprs(EXPORT_EXPRS, coords)
        fields = {expr_to_key(e): v for e, v in zip(EXPORT_EXPRS, got)}
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, x=coords[:, 0], y=coords[:, 1], d_bar=a.d_bar, u_ref=a.u_ref,
                 mesh_unit="m", coords_supplied=coords_supplied, order_fluid=a.order,
                 order_fluid_live=got_order or "",
                 template=Path(a.template).name, **fields)
        sr = fields["spf_sr"]
        nan = int(np.sum(~np.isfinite(sr)))
        print(f"[i] spf.sr: median {np.nanmedian(sr):.2f}  p90 {np.nanpercentile(sr, 90):.2f}"
              f"  non-finite {nan}")
        print(f"[save] {out}   ({coords.shape[0]} points)")
        if not coords_supplied:
            print("Vertex export is not the pack's P2 nodes.  Prefer dump-coords + --coords-npz.")
        print("\nThen:\n"
              f"  python scripts/exp_template_ab.py compare --npz {out} --stem patient041")
    return 0


def cmd_compare(a) -> int:
    import torch
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from scipy.spatial import cKDTree

    from src.clot_ml.features import M_TO_CM
    from src.config import BiochemConfig, PhysicsConfig
    from src.core_physics.mls_gradient import build_mls_gradient, node_positions, shear_rate_2d

    z = np.load(a.npz, allow_pickle=True)
    d_bar_ab = float(npz_scalar(z, "d_bar"))
    u_ref_ab = float(npz_scalar(z, "u_ref"))
    ab_xy = np.stack([np.asarray(z["x"], dtype=float), np.asarray(z["y"], dtype=float)], axis=-1)
    coords_supplied = bool(npz_scalar(z, "coords_supplied", False))
    print(f"[i] {a.npz}: template={npz_scalar(z, 'template')} "
          f"order={npz_scalar(z, 'order_fluid')} live={npz_scalar(z, 'order_fluid_live', '')} "
          f"{ab_xy.shape[0]} points, mesh_unit={npz_scalar(z, 'mesh_unit')} "
          f"coords_supplied={coords_supplied}")
    print(f"    A/B bbox {ab_xy.min(0).round(6).tolist()} .. {ab_xy.max(0).round(6).tolist()}")

    pack = torch.load(REPO / f"data/processed/graphs_biochem_anchors/{a.stem}.pt",
                      map_location="cpu", weights_only=False)
    y = pack.y[0] if pack.y.dim() == 3 else pack.y
    wall = pack.mask_wall.reshape(-1).bool().numpy()
    pk_xy, d_bar, u_ref = pack_xy_metres(pack)
    print(f"    pack bbox {pk_xy.min(0).round(6).tolist()} .. {pk_xy.max(0).round(6).tolist()} "
          f"N={pk_xy.shape[0]}  d_bar={d_bar:.6g} u_ref={u_ref:.6g}")
    if abs(d_bar - d_bar_ab) / max(d_bar, 1e-12) > 0.01 or abs(u_ref - u_ref_ab) / max(u_ref, 1e-12) > 0.01:
        print(f"[WARN] npz d_bar/u_ref ({d_bar_ab:.6g}, {u_ref_ab:.6g}) != pack "
              f"({d_bar:.6g}, {u_ref:.6g}) -- the A/B solve used the wrong inlet")

    unit_scale = 1.0
    if not coords_supplied:
        unit_scale = detect_xy_scale(ab_xy, pk_xy)
        if not (0.9 < unit_scale < 1.1):
            print(f"[i] auto-scaling A/B coordinates by {unit_scale:.4g} "
                  f"(ignore mesh_unit={npz_scalar(z, 'mesh_unit')}; detect from bbox)")
        ab_xy = ab_xy * unit_scale

    dist, idx = cKDTree(ab_xy).query(pk_xy)
    print(f"[i] NN dist p10/med/p90 = {np.percentile(dist, [10, 50, 90]).round(6).tolist()} m")

    # Interpolate A/B -> pack nodes.  Vertex exports are a different node set; NN alone
    # mixed interpolation error into the field comparison.
    def _onto(field: np.ndarray) -> np.ndarray:
        lin = LinearNDInterpolator(ab_xy, field)
        out = lin(pk_xy)
        miss = ~np.isfinite(out)
        if miss.any():
            out[miss] = NearestNDInterpolator(ab_xy, field)(pk_xy[miss])
        return out

    u_ab = _onto(np.asarray(z["u"], dtype=float))
    v_ab = _onto(np.asarray(z["v"], dtype=float))
    sr_ab = _onto(np.asarray(z["spf_sr"], dtype=float))
    dsrx_ab_comsol = _onto(np.asarray(z["d_spf_sr_x"], dtype=float)) / (unit_scale * M_TO_CM)
    mu_ab = _onto(np.asarray(z["spf_mu"], dtype=float)) if "spf_mu" in z.files else None

    u_gt = y[:, 0].numpy() * u_ref
    v_gt = y[:, 1].numpy() * u_ref
    interior = ~wall
    print(f"\n  velocity (SI, linear interp of A/B onto pack nodes)")
    print(f"    relL2 uv={rel_l2(np.c_[u_ab, v_ab], np.c_[u_gt, v_gt]):.4f}  "
          f"u={rel_l2(u_ab, u_gt):.4f}  v={rel_l2(v_ab, v_gt):.4f}")
    print(f"    rms u  gt={np.sqrt(np.mean(u_gt ** 2)):.4f}  ab={np.sqrt(np.mean(u_ab ** 2)):.4f}")
    print(f"    rms v  gt={np.sqrt(np.mean(v_gt ** 2)):.4f}  ab={np.sqrt(np.mean(v_ab ** 2)):.4f}  "
          f"ratio={np.sqrt(np.mean(v_ab ** 2)) / max(np.sqrt(np.mean(v_gt ** 2)), 1e-30):.3f}")
    print(f"    mean u gt={float(u_gt.mean()):.4f}  ab={float(u_ab.mean()):.4f}  "
          f"(bulk flux; should match if Re/inlet match)")
    num = float(np.dot(np.c_[u_ab, v_ab][interior].ravel(), np.c_[u_gt, v_gt][interior].ravel()))
    den = (np.linalg.norm(np.c_[u_ab, v_ab][interior]) * np.linalg.norm(np.c_[u_gt, v_gt][interior]))
    print(f"    cosine interior uv={num / max(den, 1e-30):.4f}")
    if mu_ab is not None:
        mu_gt = y[:, 3].numpy() * float(PhysicsConfig(phase="biochem").mu_viscosity_nd_reference)
        print(f"    mu SI relL2={rel_l2(mu_ab, mu_gt):.4f}  med gt/ab="
              f"{np.median(mu_gt):.5f}/{np.median(mu_ab):.5f}")

    bio = BiochemConfig(phase="biochem")
    lss, sgt = float(bio.lss), float(bio.sgt) / M_TO_CM
    w = wall

    Dx, Dy = build_mls_gradient(node_positions(pack), pack.edge_index.numpy(), hops=3)
    dbp = d_bar
    urp = u_ref
    up, vp = y[:, 0].double().numpy(), y[:, 1].double().numpy()
    sr_bio = shear_rate_2d(Dx @ up, Dy @ up, Dx @ vp, Dy @ vp) * (urp / dbp)
    dsrx_bio = (Dx @ sr_bio) / (dbp * M_TO_CM)

    u_nd, v_nd = u_ab / u_ref, v_ab / u_ref
    sr_ab_rec = shear_rate_2d(Dx @ u_nd, Dy @ u_nd, Dx @ v_nd, Dy @ v_nd) * (urp / dbp)
    dsrx_ab_rec = (Dx @ sr_ab_rec) / (dbp * M_TO_CM)

    def row(name, sr, dsrx):
        return dict(arm=name, sr_med=float(np.median(sr[w])), dsrx_sd=float(np.std(dsrx[w])),
                    sep=float((dsrx[w] < sgt).mean()),
                    fire=float(((sr[w] < lss) | (dsrx[w] < sgt)).mean()))

    rows = [row("biochem template (the pack)", sr_bio, dsrx_bio),
            row("kinematics template, COMSOL sr", sr_ab, dsrx_ab_comsol),
            row("kinematics template, reconstructed", sr_ab_rec, dsrx_ab_rec)]
    print(f"\n{'arm':<38}{'wall sr_med':>13}{'wall dsrx_sd':>14}{'sep%':>8}{'fire%':>8}")
    for r in rows:
        print(f"{r['arm']:<38}{r['sr_med']:>13.2f}{r['dsrx_sd']:>14.2f}"
              f"{100 * r['sep']:>8.2f}{100 * r['fire']:>8.2f}")
    ratio = rows[2]["dsrx_sd"] / max(rows[0]["dsrx_sd"], 1e-30)
    v_ratio = float(np.sqrt(np.mean(v_ab ** 2)) / max(np.sqrt(np.mean(v_gt ** 2)), 1e-30))
    print(f"\n  kinematics / biochem wall dsrx spread (reconstructed) = {ratio:.3f}")
    print(f"  kinematics / biochem rms(v) = {v_ratio:.3f}")
    print("""
  wall dsrx ~1 AND rms(v) ~1
         template is not the gap.  Corpus lever: VesselConfig.wall_noise_* and true P2 labels.
  wall dsrx ~1 AND rms(v) << 1
         same wall-shear amplitude, weaker secondary flow.  RGP-DEQ will not see phase2
         recirculation.  Check order_fluid live=2, CrosswindDiffusion, Stationary vs the
         phase2 study that produced t=0.
  wall dsrx << 1
         kinematics template solves a smoother problem on the SAME geometry.""")
    Path(a.out).write_text(json.dumps({
        "npz": str(a.npz), "stem": a.stem,
        "ratio_kine_over_biochem": ratio, "rms_v_ratio": v_ratio,
        "relL2_uv": rel_l2(np.c_[u_ab, v_ab], np.c_[u_gt, v_gt]),
        "relL2_u": rel_l2(u_ab, u_gt), "relL2_v": rel_l2(v_ab, v_gt),
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\n[save] {a.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump-coords", help="pack nodes in metres (training box, no COMSOL)")
    d.add_argument("--stem", default="patient041")
    d.add_argument("--out", default="outputs/ab_p041_coords.npz")
    d.set_defaults(func=cmd_dump_coords)

    e = sub.add_parser("export", help="run on the COMSOL box")
    e.add_argument("--template", default="comsol_models/phase1_ab_test.mph")
    e.add_argument("--mesh-file", default="", help="re-import this NAS/mesh first (optional)")
    e.add_argument("--coords-npz", default="",
                   help="evaluate at THESE coordinates (metres) instead of COMSOL's mesh "
                        "vertices -- dump-coords output so the comparison needs no matching")
    e.add_argument("--d-bar", type=float, required=True, help="the deploy vessel's own d_bar, in metres")
    e.add_argument("--u-ref", type=float, required=True, help="the deploy vessel's own u_ref, in m/s")
    e.add_argument("--mesh-unit", default="m", choices=("m", "cm"),
                   help="ignored; bbox detection is used. kept so old commands still parse")
    e.add_argument("--order", type=int, default=2)
    e.add_argument("--out", default="outputs/ab_phase1_on_deploy.npz")
    e.set_defaults(func=cmd_export)

    c = sub.add_parser("compare", help="run on the training box")
    c.add_argument("--npz", required=True)
    c.add_argument("--stem", default="patient041")
    c.add_argument("--out", default="outputs/exp_template_ab.json")
    c.set_defaults(func=cmd_compare)

    parsed = ap.parse_args()
    return parsed.func(parsed)


if __name__ == "__main__":
    raise SystemExit(main())
