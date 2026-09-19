"""Make kinematics COMSOL the same problem as phase2 Study 2 (fluid only).

Phase2 ``phase2_template_nowound.mph`` has two studies:

* ``std1`` -- ``Study 1 (fluid + biochemistry)``, ``range(0,150,30000)``, spf+tds+tds2.
  Its initial condition is ``sol2`` (``initmethod=sol``, ``initstudy`` -> std2).
* ``std2`` -- ``Study 2 (only fluid)``, transient ``range(0,0.1,15)``, **spf on, tds off**.

Biochem pack ``y[0]`` is therefore the end of std2, not a stationary Navier-Stokes solve.
The old kinematics template ran ``std1/Stationary`` with user-defined ``mu_final`` and
``order_fluid=P1``.  That is a different problem, even on the same synthetic mesh.

This module patches a loaded model (and can persist ``phase1_template.mph``) so generation
runs std2-equivalent physics: P2, built-in Carreau at gel=1, 0..15 s fluid-only.
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.config import PhysicsConfig
from src.utils.paths import comsol_models_dir

logger = logging.getLogger(__name__)

FLUID_ONLY_STUDY = "std2"
FLUID_ONLY_STUDY_LABEL = "Study 2 (only fluid)"
#: Copied from phase2_template_nowound.mph ``std2`` / Transitorio ``p:tlist``.
FLUID_ONLY_TLIST = "range(0,0.1,15)"
FLUID_ONLY_T_END = 15.0
FLUID_ONLY_RTOL = "0.005"
STATIONARY_BACKUP = "phase1_template_stationary.mph"
#: phase2 ``rho_b`` in SI (``1.106 g/cm^3``).
PHASE2_FLUID_RHO_KG_M3 = 1106.0
#: Max corner-to-corner offset after a faithful Gmsh NAS import (metres).
#: NAS/COMSOL formatting is typically O(1e-5 m); remeshing after unit scaling was O(1e-4 m).
MESH_IMPORT_MAX_NN_M = 2.0e-5


def gel_identity_carreau_si(phys: PhysicsConfig) -> dict[str, str]:
    """Phase2 Carreau at t=0 (mu1=1, mu2=0), in the kinematics template's SI units."""
    return {
        "mu0": f"{float(phys.mu_0)}[Pa*s]",
        "mu_inf": f"{float(phys.mu_inf)}[Pa*s]",
        "lam_car": f"{float(phys.lam)}[s]",
        "n_car": str(float(phys.n)),
    }


def apply_t0_fluid_physics(model, phys: PhysicsConfig | None = None) -> None:
    """P2 + built-in fluid model matching phase2 std2 at gel=1 (Carreau) or Newtonian primer."""
    phys = phys or PhysicsConfig(phase="kinematics", rheology="carreau")
    j = model.java
    spf = j.physics("spf")
    want = str(int(phys.comsol_order_fluid))
    for grp in ("ShapeProperty", "PhysicsShapeProperty"):
        try:
            spf.prop(grp).set("order_fluid", want)
            logger.debug("order_fluid=%s via %s", want, grp)
            break
        except Exception:
            continue
    fp = spf.feature("fp1")
    if phys.viscosity_model == "carreau":
        fp.set("Constitutiverelation", "InelasticNonNewtonian")
        fp.set("nonNewtonianModels", "Carreau")
        for mat_key in ("mu0_mat", "mu_inf_mat", "lam_car_mat", "n_car_mat"):
            try:
                fp.set(mat_key, "userdef")
            except Exception:
                pass
        for key, val in gel_identity_carreau_si(phys).items():
            fp.set(key, val)
        logger.info(
            "spf Carreau gel=1: mu0=%s mu_inf=%s lam=%s n=%s",
            fp.getString("mu0"), fp.getString("mu_inf"),
            fp.getString("lam_car"), fp.getString("n_car"),
        )
    else:
        fp.set("Constitutiverelation", "Newtonian")
        fp.set("dynamicviscosity", f"{float(phys.mu_ref)}[Pa*s]")
        logger.info("spf Newtonian: mu=%s", fp.getString("dynamicviscosity"))


def apply_phase2_inlet_parameters(
    model,
    d_bar_si: float,
    u_ref: float,
    phys: PhysicsConfig | None = None,
) -> None:
    """Per-vessel inlet scales shared with phase2 biochem (SI length / velocity)."""
    phys = phys or PhysicsConfig(phase="kinematics", rheology="carreau")
    model.parameter("D_eff", f"{float(d_bar_si):.10f} [m]")
    model.parameter("U_inlet", f"{float(u_ref):.10f} [m/s]")
    model.parameter("rho_fluid", f"{float(phys.rho)} [kg/m^3]")
    model.parameter("Re_target", str(int(phys.re_target)))


def mesh_vertex_xy_metres(model_java) -> "np.ndarray":
    """COMSOL mesh vertices as ``(N, 2)`` in metres (handles cm templates)."""
    import numpy as np

    v = np.asarray(model_java.mesh("mesh1").getVertex(), dtype=float)
    if v.shape[0] == 2:
        v = v.T
    xy = np.ascontiguousarray(v[:, :2], dtype=float)
    if float(np.ptp(xy[:, 0])) > 1.0:
        xy = xy * 0.01
    return xy


def import_gmsh_nas_mesh(mesh_j, import_tag: str, nas_path: Path | str) -> int:
    """Import a Gmsh NAS mesh only -- no geometry scaling or remeshing in COMSOL."""
    safe = str(Path(nas_path)).replace("\\", "/")
    feat = mesh_j.feature(str(import_tag))
    feat.set("filename", safe)
    mesh_j.run()
    return int(mesh_j.getNumVertex())


def validate_mesh_import(
    gmsh_xy_m: "np.ndarray",
    comsol_xy_m: "np.ndarray",
    *,
    max_nn_m: float = MESH_IMPORT_MAX_NN_M,
    count_tol: int = 0,
) -> dict[str, float]:
    """Raise if COMSOL changed the imported discretization (remesh / unit scaling).

    Returns alignment stats when the import is faithful.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    gmsh_xy_m = np.asarray(gmsh_xy_m, dtype=float)
    comsol_xy_m = np.asarray(comsol_xy_m, dtype=float)
    if gmsh_xy_m.ndim != 2 or comsol_xy_m.ndim != 2:
        raise ValueError("gmsh_xy_m and comsol_xy_m must be (N, 2)")
    n_g, n_c = gmsh_xy_m.shape[0], comsol_xy_m.shape[0]
    if abs(n_g - n_c) > int(count_tol):
        raise ValueError(
            f"mesh import changed vertex count: gmsh={n_g} comsol={n_c}. "
            "Do not scale geometry or remesh in COMSOL; import the Gmsh NAS as-is."
        )
    dist, _ = cKDTree(comsol_xy_m).query(gmsh_xy_m)
    stats = {
        "gmsh_vertices": float(n_g),
        "comsol_vertices": float(n_c),
        "nn_median_m": float(np.median(dist)),
        "nn_p90_m": float(np.percentile(dist, 90)),
        "nn_max_m": float(np.max(dist)),
        "exact_matches": float(np.sum(dist < 1.0e-9)),
    }
    if stats["nn_p90_m"] > float(max_nn_m):
        raise ValueError(
            f"mesh import offset too large (p90={stats['nn_p90_m']:.3e} m > {max_nn_m:.3e} m). "
            "Likely COMSOL remeshed or scaled the geometry after import."
        )
    return stats


def ensure_fluid_only_study(model, *, tlist: str | None = None) -> str:
    """Create ``std2`` Transient like phase2 if missing.  Returns the study tag."""
    j = model.java
    tags = [str(t) for t in j.study().tags()]
    tlist = tlist or FLUID_ONLY_TLIST
    if FLUID_ONLY_STUDY not in tags:
        j.study().create(FLUID_ONLY_STUDY)
        j.study(FLUID_ONLY_STUDY).label(FLUID_ONLY_STUDY_LABEL)
        j.study(FLUID_ONLY_STUDY).create("time", "Transient")
        logger.info("created study %s (%s)", FLUID_ONLY_STUDY, FLUID_ONLY_STUDY_LABEL)
    time = j.study(FLUID_ONLY_STUDY).feature("time")
    time.set("tlist", tlist)
    try:
        time.set("rtol", FLUID_ONLY_RTOL)
    except Exception:
        pass
    if "std1" in tags:
        try:
            j.study("std1").active(False)
        except Exception:
            try:
                j.study("std1").set("activate", False)
            except Exception:
                logger.warning("could not deactivate std1; generator must run std2 explicitly")
    return FLUID_ONLY_STUDY


def configure_kinematics_model(model, phys: PhysicsConfig | None = None) -> str:
    """Physics + std2 on a loaded kinematics model.  Returns the study tag to run."""
    phys = phys or PhysicsConfig(phase="kinematics", rheology="carreau")
    apply_t0_fluid_physics(model, phys)
    return ensure_fluid_only_study(model, tlist=str(phys.comsol_fluid_tlist))


def set_carreau_n(model, n_val: float) -> None:
    """Ramp ``n_car`` (built-in Carreau) together with leftover ``n_index``."""
    model.parameter("n_index", str(n_val))
    try:
        model.java.physics("spf").feature("fp1").set("n_car", str(float(n_val)))
    except Exception as exc:
        logger.warning("could not set n_car=%s: %s", n_val, exc)


def solve_fluid_only(model, study_tag: str | None = None) -> None:
    tag = study_tag or FLUID_ONLY_STUDY
    logger.debug("running COMSOL study %s (phase2 fluid-only equivalent)", tag)
    model.java.study(tag).run()


def last_time_slice(values, n_pts: int):
    """Interp on a transient dataset may return ``n_times * n_pts`` (all levels stacked).

    Phase2 std2 is 151 times; the clot stack wants the last one (t=15 s, biochem IC).
    """
    import numpy as np

    a = np.asarray(values, dtype=float).reshape(-1)
    if a.size == n_pts:
        return a
    if a.size > n_pts and a.size % n_pts == 0:
        return a.reshape(-1, n_pts)[-1]
    raise ValueError(f"expected {n_pts} values or a multiple, got {a.size}")


def pin_interp_last_time(interp, t_end: float = FLUID_ONLY_T_END) -> None:
    for attempt in (
        lambda: interp.set("t", float(t_end)),
        lambda: interp.set("outersolnum", "last"),
        lambda: interp.set("solnum", "last"),
        lambda: interp.set("looplevelinput", [["t", str(t_end)]]),
    ):
        try:
            attempt()
            return
        except Exception:
            continue
    logger.warning("could not pin Interp to t=%s; will slice the last time block", t_end)


def interpolation_dataset_tag(model_java) -> str:
    """Results dataset for the fluid-only solution (not biochem std1)."""
    from src.data_gen.lib.biochem_comsol_datasets import list_comsol_datasets, resolve_solution_dataset

    for sol in ("sol2", "sol1"):
        try:
            tag = resolve_solution_dataset(model_java, sol)
        except Exception:
            continue
        rows = {r["tag"]: r for r in list_comsol_datasets(model_java)}
        label = str(rows.get(tag, {}).get("label") or "")
        ll = label.lower()
        if sol == "sol2" or "study 2" in ll or "only fluid" in ll:
            return tag
        if sol == "sol1" and "biochem" not in ll:
            return tag
    return "dset1"


def persist_phase1_template(
    *,
    src: Path | None = None,
    dest: Path | None = None,
    backup: bool = True,
) -> Path:
    """Load, configure, save ``phase1_template.mph``.  Needs a COMSOL server."""
    import shutil
    import mph

    root = comsol_models_dir()
    src = Path(src) if src is not None else root / "phase1_template.mph"
    dest = Path(dest) if dest is not None else src
    if backup:
        bak = root / STATIONARY_BACKUP
        if not bak.exists():
            shutil.copy2(src, bak)
            logger.info("backed up original to %s", bak)
    client = mph.start(cores=1)
    try:
        model = client.load(str(src.resolve()))
        configure_kinematics_model(model)
        dest.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(dest.resolve()))
        logger.info("saved %s (%s bytes)", dest, dest.stat().st_size)
        return dest
    finally:
        try:
            client.clear()
        except Exception:
            pass


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    persist_phase1_template()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
