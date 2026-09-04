import warnings

import numpy as np
import skfem as fem
from skfem.helpers import grad
from scipy.sparse.linalg import splu
from scipy.spatial import KDTree
import torch
from src.config import PhysicsConfig
from src.core_physics.inlet_profile import get_inlet_profile


def _asm_threads() -> int:
    """Threads for the form kernel loop.  numpy releases the GIL inside the ufuncs the kernel
    is made of, so the 225 basis pairs parallelise; capped because the arrays are small enough
    that the scheduling overhead wins beyond a handful of workers.
    """
    import os
    env = os.environ.get("FEM_ASM_THREADS", "")
    if env.strip():
        return max(1, int(env))
    return max(1, min(4, (os.cpu_count() or 1) - 1))


def solve_local_t0_flow(mesh_path, data, phys_cfg: PhysicsConfig, max_iters=300, tol=1e-8,
                        u_gt_inlet_nd: np.ndarray | None = None, damping: float = 0.5,
                        lu_refresh: float = 1.0, lu_max_age: int = 25, anderson_m: int = 5,
                        art_visc: float = 0.70, stab: str = "iso",
                        delta_mu_nodal_si: np.ndarray | None = None,
                        u_init_nd: np.ndarray | None = None,
                        verbose: bool = True):
    """Solve steady-state Carreau Navier-Stokes on the .nas/.msh mesh.

    The nonlinearity is handled by a damped Picard iteration accelerated three ways, none of
    which changes the converged answer -- the operator is reassembled from the current iterate
    every step, so the fixed point is the same one textbook Picard reaches (verified to 1e-8
    relative, and exactly on the wall band, on patients 001/008/041/042 and the wounds):

    ``damping``    initial under-relaxation, released to 1.0 as soon as the increment falls
                   monotonically.  Holding it at 0.5 for the whole solve, as this did, caps the
                   asymptotic rate at 0.5 per iteration however strongly the map contracts.
    ``lu_refresh`` the step is a defect correction against a FROZEN LU of the operator, since
                   factorising costs ~36x a triangular solve.  A new factorisation is taken
                   whenever the increment grows by more than this ratio, or after
                   ``lu_max_age`` reuses.  Pass 0.0 to refactorise every iteration.
    ``anderson_m`` history depth for Anderson extrapolation over the iterates, engaged once
                   damping is released.  Picard alone contracts at ~0.98 per iteration on the
                   stenoses -- 500+ iterations, which is why those vessels were previously read
                   off unconverged solves.  Pass 0 to disable.

    ``tol`` is RELATIVE, on the undamped increment measured over the velocity DoFs only.

    ``u_init_nd`` optional per-TARGET-NODE non-dimensional velocity (u/u_ref, the same units
                   `u0_pred`/`v0_pred` are stored in) used as the INITIAL ITERATE.  Picard's
                   first step is `x <- A(x_0)^-1 b`, so this only sets the wind the first
                   operator is linearised about -- the fixed point, and therefore the returned
                   field, is unchanged.  Starting from a surrogate field instead of zero skips
                   the opening transient where the wind is furthest from the answer.

    ``delta_mu_nodal_si`` optional per-TARGET-NODE clot viscosity elevation [Pa.s], added to
                   the Carreau viscosity at the quadrature points.  This is what makes the
                   solver usable as a severe-occlusion oracle: a clot is a high-viscosity
                   region, never a hole, exactly as the COMSOL Patch Factory models it -- but
                   here on the real vessel geometry, under its real fixed-flux inlet BC, at
                   any occlusion fraction.  ``None`` reproduces the clot-free solve
                   bit-for-bit.
    """
    mesh = fem.Mesh.load(mesh_path)

    pos_target_nd = data.x[:, 0:2].cpu().numpy()
    d_bar = float(data.d_bar.item()) if hasattr(data.d_bar, 'item') else float(data.d_bar)
    u_ref = float(data.u_ref.item()) if hasattr(data.u_ref, 'item') else float(data.u_ref)

    # Mesh units are not uniform across the two mesh families: the patient `.nas` anchors
    # are in cm, the research `.msh` vessels gmsh writes are already in metres.  A
    # hardcoded 0.01 was right for the anchors and 100x wrong for every research vessel,
    # which collapsed the whole mesh onto ~3 pack nodes -- so the wall and outlet tagged
    # zero facets, "inlet" tagged interior facets, and the saddle point came out exactly
    # singular.  Register the mesh onto the pack instead of assuming a unit, and refuse to
    # solve when it does not register: a silently mis-scaled mesh solves to garbage that
    # still scores.
    pos_phys = pos_target_nd * d_bar
    kd = KDTree(pos_phys)
    n_mesh = mesh.p.shape[1]
    best = None
    for scale in (1.0, 0.01, 0.001):
        cand = mesh if scale == 1.0 else mesh.scaled(scale)
        dist, idx = kd.query(cand.p.T)
        n_unique = int(np.unique(idx).size)
        key = (n_unique, -float(np.median(dist)))
        if best is None or key > best[0]:
            best = (key, scale, cand, dist, idx, n_unique)

    _, mesh_scale, mesh, nn_dist, skfem_to_target, n_unique = best
    span = float(np.linalg.norm(pos_phys.max(axis=0) - pos_phys.min(axis=0)))
    nn_med = float(np.median(nn_dist))
    if n_unique != n_mesh or nn_med > 1e-6 * span:
        raise ValueError(
            f"local FEM: mesh {mesh_path!r} does not register onto the pack at any known unit "
            f"scale (best scale={mesh_scale}, injective={n_unique}/{n_mesh}, "
            f"median nn={nn_med:.3e}, pack span={span:.3e})"
        )

    mask_inlet = data.mask_inlet.cpu().numpy().astype(bool)
    mask_outlet = data.mask_outlet.cpu().numpy().astype(bool)

    is_inlet = mask_inlet[skfem_to_target]
    is_outlet = mask_outlet[skfem_to_target]

    bfacets = mesh.boundary_facets()
    n_boundary = int(bfacets.size)
    P = mesh.p.T
    fa, fb = P[mesh.facets[0, bfacets]], P[mesh.facets[1, bfacets]]
    fmid = 0.5 * (fa + fb)
    fh = np.linalg.norm(fb - fa, axis=1)

    def corner_facets(node_mask):
        """Boundary facets whose BOTH corner vertices carry the tag."""
        return bfacets[np.all(node_mask[mesh.facets], axis=0)[bfacets]]

    def planar_facets(node_mask, name):
        """Boundary facets lying on the plane the tagged nodes span, inside their extent.

        The node tags come off COMSOL's own selections and are not always complete on a
        quadratic mesh: `patient038` tags no two adjacent corners of its inlet at all (0
        facets under the corner rule) and `patient048` tags 4 of its 21 outlet facets with
        only one corner each, which silently handed those 4 to the no-slip wall.  An inlet
        or outlet is a straight cut through the lumen, so the tagged nodes determine it
        completely: fit the line through them and take every boundary facet whose midpoint
        sits on it, within the tagged nodes' own along-line extent.

        Returns ``None`` when the tag cannot support the fit -- fewer than two nodes, or
        nodes that are not collinear -- so the caller falls back to the corner rule rather
        than inventing a boundary.
        """
        pts = P[node_mask]
        if pts.shape[0] < 2:
            return None
        c = pts.mean(axis=0)
        _, sv, vt = np.linalg.svd(pts - c, full_matrices=False)
        along, normal = vt[0], vt[1]
        extent = float(np.abs((pts - c) @ along).max())
        if extent <= 0.0:
            return None
        # collinearity guard: a curved or smeared selection is not a cut plane
        if float(np.abs((pts - c) @ normal).max()) > 0.05 * extent:
            return None
        sel = (np.abs((fmid - c) @ normal) < 0.05 * fh) & (np.abs((fmid - c) @ along) <= extent)
        got = bfacets[sel]
        return got if got.size else None

    f_inlet, f_outlet = corner_facets(is_inlet), corner_facets(is_outlet)
    p_inlet, p_outlet = planar_facets(is_inlet, "inlet"), planar_facets(is_outlet, "outlet")
    # The planar completion is only accepted when it CONTAINS what the node tags already
    # agreed on -- it may add the facets a partial tag missed, never move one.
    if p_inlet is not None and p_outlet is not None             and np.isin(f_inlet, p_inlet).all() and np.isin(f_outlet, p_outlet).all()             and not np.intersect1d(p_inlet, p_outlet).size:
        if len(p_inlet) != len(f_inlet) or len(p_outlet) != len(f_outlet):
            print("[i] local FEM: planar completion of the boundary tags: "
                  "inlet %d->%d, outlet %d->%d facets"
                  % (len(f_inlet), len(p_inlet), len(f_outlet), len(p_outlet)), flush=True)
        f_inlet, f_outlet = p_inlet, p_outlet
    if min(len(f_inlet), len(f_outlet)) == 0:
        raise ValueError(
            f"local FEM: mesh {mesh_path!r} tagged inlet={len(f_inlet)} outlet={len(f_outlet)} "
            f"facets; an untagged inlet or outlet leaves no boundary condition there and a "
            f"singular solve"
        )
    if len(f_inlet) + len(f_outlet) > n_boundary:
        raise ValueError(
            f"local FEM: mesh {mesh_path!r} tagged {len(f_inlet) + len(f_outlet)} inlet/outlet "
            f"facets but the mesh only has {n_boundary} boundary facets; the node masks are "
            f"smeared across the interior"
        )

    # Every EXTERIOR facet that is not inlet or outlet is a no-slip wall.  Taking the wall from
    # the pack's `mask_wall` instead misses the wound cavity boundary: the wound packs carry
    # ~80 nodes (against 1-12 on a plain vessel) whose COMSOL velocity is exactly zero but
    # which are tagged neither wall nor inlet nor outlet, so the solve imposed NO condition on
    # the injury at all and the flow leaked through it.  Topology cannot miss a boundary the
    # way a node tag can, and it stays deploy-legal -- this reads the mesh, never the labels.
    _named = (np.concatenate([f_inlet, f_outlet]) if (f_inlet.size or f_outlet.size)
              else np.array([], dtype=int))
    f_wall = np.setdiff1d(mesh.boundary_facets(), _named)
    if f_wall.size == 0:
        raise ValueError(
            f"local FEM: mesh {mesh_path!r} has no boundary left for the wall after removing "
            f"inlet/outlet; an untagged wall means no no-slip and a singular solve"
        )

    mesh = mesh.with_boundaries({"inlet": f_inlet, "outlet": f_outlet, "wall": f_wall})

    element_v = fem.ElementVector(fem.ElementTriP2())
    element_p = fem.ElementTriP1()
    element = element_v * element_p
    basis = fem.Basis(mesh, element)

    # Clot viscosity elevation, mapped graph-node -> mesh-vertex -> quadrature point once.
    # A scalar P1 basis sharing `basis`'s quadrature guarantees the array lines up with the
    # Carreau `mu` computed inside `effective_viscosity` (shape [n_elems, n_qp]).
    dmu_qp = None
    if delta_mu_nodal_si is not None:
        dmu_nodal = np.asarray(delta_mu_nodal_si, dtype=np.float64).reshape(-1)
        if dmu_nodal.shape[0] != pos_target_nd.shape[0]:
            raise ValueError(
                f"delta_mu_nodal_si has {dmu_nodal.shape[0]} entries, expected "
                f"{pos_target_nd.shape[0]} (one per graph node)")
        dmu_mesh = np.clip(dmu_nodal[skfem_to_target], 0.0, None)
        # The anchor meshes are QUADRATIC (`MeshTri2`): `mesh.p` holds the vertices followed
        # by the mid-side nodes (patient001: 9490 = 2447 + 7043).  A scalar P2 basis sharing
        # `basis`'s quadrature therefore takes the field exactly as stored, using the same
        # nodal-then-facet convention the velocity is written back out with below -- and it
        # keeps the clot resolved on the near-wall mid-side ring rather than averaging it away.
        _sb = fem.Basis(mesh, fem.ElementTriP2(), quadrature=basis.quadrature)
        n_vert = _sb.nodal_dofs.shape[1]
        vec = np.zeros(_sb.N)
        vec[_sb.nodal_dofs[0, :]] = dmu_mesh[:n_vert]
        if dmu_mesh.shape[0] > n_vert:
            vec[_sb.facet_dofs[0, :]] = dmu_mesh[n_vert:]
        dmu_qp = np.asarray(_sb.interpolate(vec))

    rho = phys_cfg.rho
    mu_inf = phys_cfg.mu_inf
    mu_0 = phys_cfg.mu_0
    lam = phys_cfg.lam
    n_car = phys_cfg.n
    a = phys_cfg.a

    h = 0.0005

    # Element size at the quadrature points, for the SUPG parameter.  The isotropic branch
    # keeps the historical hardcoded 0.5 mm; `h_elem` is the real thing (median 0.71 mm).
    _v = mesh.p[:, mesh.t[:3]]
    h_elem = np.max([np.linalg.norm(_v[:, i] - _v[:, j], axis=0)
                     for i, j in ((0, 1), (1, 2), (2, 0))], axis=0)

    def supg_tau(wind, mu):
        """Doubly-asymptotic SUPG parameter, tau = ((2|u|/h)^2 + (4 mu / rho h^2)^2)^-1/2.

        `tau` is built on the KINEMATIC viscosity and carries units of seconds, so the term it
        multiplies is `tau * (u.grad v) . R` with no further `rho` -- R already carries it.

        Streamline-directed, and multiplied by the momentum residual below, so it is
        *consistent*: the term vanishes on the exact solution instead of biasing it, unlike the
        isotropic `mu_art` it replaces -- which added 0.045 Pa.s of cross-stream diffusion on
        top of a physical viscosity measured at 0.0065, and reattached the post-stenotic
        separation bubble two deciles early.
        """
        he = h_elem[:, None]
        wind = np.asarray(wind)
        u_mag = np.sqrt(wind[0]**2 + wind[1]**2 + 1e-12)
        return 1.0 / np.sqrt((2.0 * u_mag / he)**2 + (4.0 * mu / (rho * he**2))**2 + 1e-30)

    def _tau_kw(wind, mu):
        """`tau=` for the form kernels, built only when one of them actually reads it.

        `stab` defaults to "iso", where neither kernel touches `w["tau"]` -- but the SUPG
        parameter was still being evaluated over every quadrature point on every iteration
        and thrown away.
        """
        if stab not in ("supg", "su"):
            return {}
        return {"tau": supg_tau(wind, mu)}

    def effective_viscosity(wind):
        """Carreau viscosity plus artificial diffusion at the quadrature points.

        Hoisted out of the bilinear form deliberately: skfem evaluates the form kernel
        `Nbfun**2` times per assembly (225 for this P2/P1 pair) and `mu` depends only on the
        previous iterate, so leaving it inside recomputed the whole Carreau law -- two powers
        and two square roots over every quadrature point -- 225 times instead of once.
        """
        g = np.asarray(wind.grad)
        wind = np.asarray(wind)
        eps01 = 0.5 * (g[0][1] + g[1][0])
        gamma_dot = np.sqrt(2.0 * (g[0][0]**2 + g[1][1]**2 + 2 * eps01**2) + 1e-12)
        mu_car = mu_inf + (mu_0 - mu_inf) * (1.0 + (lam * gamma_dot)**a)**((n_car - 1) / a)
        # Isotropic artificial viscosity, standing in for the streamline+crosswind diffusion
        # COMSOL's Laminar Flow interface applies by default and which is therefore baked into
        # the labels.  It is NOT physics: the Carreau law above was verified against COMSOL's
        # own `spf.mu` to within 1.5% (p25/med/p75 0.00551/0.00651/0.00855 against
        # 0.00559/0.00655/0.00867 on patient001).  `art_visc` was 1.0, which put 0.045 Pa.s of
        # cross-stream diffusion on top of that 0.0065 and reattached the post-stenotic
        # separation bubble two deciles early; 0.70 is the fitted value -- see the sweep in the
        # solver notes.  Streamline-only stabilisation (SUPG/SU) does NOT substitute: the jet
        # spreading the labels show is cross-stream, so those arms behave like art_visc=0.
        if dmu_qp is not None:
            # The clot is a high-viscosity porous zone, not a hole: added to the Carreau
            # viscosity BEFORE the stabilisation term, so artificial diffusion is not scaled
            # by the occlusion.
            mu_car = mu_car + dmu_qp
        u_mag = np.sqrt(wind[0]**2 + wind[1]**2 + 1e-12)
        if stab in ("supg", "su"):
            return mu_car
        return mu_car + art_visc * 0.5 * rho * u_mag * h

    @fem.BilinearForm(nthreads=_asm_threads())
    def navier_stokes_picard(u, p, v, q, w):
        # `w[...]`, `v[...]` and `grad(u)[...]` all return skfem `DiscreteField`s, whose
        # `__getitem__` is `np.array(self)[key]` -- a full COPY of the whole field on every
        # single index.  `grad(u)` is (2, 2, n_elems, n_qp), so each `gu[0][1]` below copied
        # the entire gradient, and this kernel runs Nbfun**2 = 225 times per assembly.
        # Taking one plain-ndarray view of each field up front makes every index after it a
        # view instead, at no cost to the arithmetic.
        gu = np.asarray(grad(u))
        gv = np.asarray(grad(v))
        gp = np.asarray(grad(p)) if stab in ("supg", "su") else None
        u_prev = np.asarray(w["wind"])
        v = np.asarray(v)
        mu = w["mu"]

        # `2 * eps(u) : eps(v)` written on the raw gradients.  `sym_grad` builds a fresh 2x2
        # object array on every call and the kernel runs Nbfun**2 = 225 times per assembly for
        # only 15 distinct basis functions, so calling it here cost ~27% of the assembly to
        # recompute 15 values 225 times.  Algebraically identical:
        #   eps_ij = .5 (g_ij + g_ji)  =>  2 eps(u):eps(v)
        #          = 2 [ gu00 gv00 + gu11 gv11 + .5 (gu01 + gu10)(gv01 + gv10) ]
        shear = (gu[0][1] + gu[1][0]) * (gv[0][1] + gv[1][0])
        viscous = 2.0 * mu * (gu[0][0]*gv[0][0] + gu[1][1]*gv[1][1] + 0.5 * shear)

        convective = rho * (v[0] * (u_prev[0] * gu[0][0] + u_prev[1] * gu[0][1]) + v[1] * (u_prev[0] * gu[1][0] + u_prev[1] * gu[1][1]))
        pressure = -p * (gv[0][0] + gv[1][1])
        continuity = -q * (gu[0][0] + gu[1][1])
        out = convective + viscous + pressure + continuity

        if stab in ("supg", "su"):
            for k in (0, 1):
                wgv = u_prev[0] * gv[k][0] + u_prev[1] * gv[k][1]
                res = rho * (u_prev[0] * gu[k][0] + u_prev[1] * gu[k][1])
                if stab == "supg":
                    res = res + gp[k]
                out = out + w["tau"] * wgv * res
        return out

    @fem.LinearForm(nthreads=_asm_threads())
    def navier_stokes_residual(v, q, w):
        """`A(x) @ x` for the operator above, assembled as a vector.

        Identical arithmetic, but skfem evaluates a LinearForm kernel Nbfun times instead of
        Nbfun**2 -- 15 evaluations against 225 -- so the defect-correction iterations below
        cost a fifteenth of a matrix assembly.  Exact: the Picard wind IS the current iterate,
        so this is the true nonlinear residual and its zero is the same solution.
        """
        # `w[...]`, `v[...]` and `grad(u)[...]` all return skfem `DiscreteField`s, whose
        # `__getitem__` is `np.array(self)[key]` -- a full COPY of the whole field on every
        # single index.  `grad(u)` is (2, 2, n_elems, n_qp), so each `gu[0][1]` below copied
        # the entire gradient, and this kernel runs Nbfun**2 = 225 times per assembly.
        # Taking one plain-ndarray view of each field up front makes every index after it a
        # view instead, at no cost to the arithmetic.
        us_f = w["sol"]
        gu = np.asarray(us_f.grad)
        gv = np.asarray(grad(v))
        ps = w["pres"]
        gp = np.asarray(ps.grad) if stab in ("supg", "su") else None
        us = np.asarray(us_f)
        v = np.asarray(v)
        mu = w["mu"]
        convective = rho * (v[0] * (us[0] * gu[0][0] + us[1] * gu[0][1]) + v[1] * (us[0] * gu[1][0] + us[1] * gu[1][1]))
        shear = (gu[0][1] + gu[1][0]) * (gv[0][1] + gv[1][0])
        viscous = 2.0 * mu * (gu[0][0]*gv[0][0] + gu[1][1]*gv[1][1] + 0.5 * shear)
        pressure = -ps * (gv[0][0] + gv[1][1])
        continuity = -q * (gu[0][0] + gu[1][1])
        out = convective + viscous + pressure + continuity

        if stab in ("supg", "su"):
            for k in (0, 1):
                wgv = us[0] * gv[k][0] + us[1] * gv[k][1]
                res = rho * (us[0] * gu[k][0] + us[1] * gu[k][1])
                if stab == "supg":
                    res = res + gp[k]
                out = out + w["tau"] * wgv * res
        return out

    x = np.zeros(basis.N)

    inlet_facets = mesh.boundaries["inlet"]
    inlet_nodes = np.unique(mesh.facets[:, inlet_facets])
    inlet_pts = mesh.p[:, inlet_nodes].T

    if len(inlet_pts) > 0:
        inlet_center = np.mean(inlet_pts, axis=0)
        inlet_pts_centered = inlet_pts - inlet_center
        cov = np.cov(inlet_pts_centered.T)
        evals, evecs = np.linalg.eigh(cov)
        tangent = evecs[:, 1]
        normal = evecs[:, 0]

        # The pack carries its own Reynolds number in `u_ref` (= Re*mu_inf/(rho*d_bar), the
        # mean inlet velocity `get_inlet_profile` wants).  `phys_cfg.re_target` is a global
        # default of 450, and `solve_fem_into_pack` builds a default PhysicsConfig -- so every
        # research arm was solved at Re=450 no matter what Reynolds it swept.
        U_inlet = u_ref
        H_inlet = np.max(np.dot(inlet_pts, tangent)) - np.min(np.dot(inlet_pts, tangent))

        mass_center = mesh.p.mean(axis=1)
        if np.dot(mass_center - inlet_center, normal) > 0:
            flow_dir = normal
        else:
            flow_dir = -normal
    else:
        U_inlet = 0.0
        H_inlet = 1.0
        tangent = np.array([0, 1])
        flow_dir = np.array([1, 0])
        inlet_center = np.array([0, 0])

    # Only the INLET Dirichlet rows of `x_bc` are ever read: `fem.enforce` touches the rows in
    # `D_all` and nothing else, and the wall half of `D_all` is overwritten with zero below.
    # So the DoF sets are taken first and the profile is evaluated at the inlet DoF locations
    # only -- this used to build the boundary values over EVERY DoF in the mesh (5173 vertices
    # plus ~15k edge midpoints on a typical synthetic vessel, against ~60 that are actually on
    # the inlet), and to do it four separate times, once per (block, component), each of which
    # rebuilt the whole 1D profile from scratch.
    D_inlet = basis.get_dofs("inlet").drop('u^2')
    D_wall = basis.get_dofs("wall").drop('u^2')

    x_bc = np.zeros(basis.N)

    # P2 velocity facet DoFs live on edge midpoints; skfem stores them in `doflocs`,
    # not as extra columns of `mesh.p` (which only carries geometric nodes).
    facet_dof_locs = basis.doflocs[:, basis.facet_dofs[0, :]]
    nodal_pts = mesh.p[:, :basis.nodal_dofs.shape[1]]

    on_inlet = np.zeros(basis.N, dtype=bool)
    on_inlet[D_inlet.all()] = True
    sel_nodal = on_inlet[basis.nodal_dofs[0, :]] | on_inlet[basis.nodal_dofs[1, :]]
    sel_facet = on_inlet[basis.facet_dofs[0, :]] | on_inlet[basis.facet_dofs[1, :]]

    def u_in_func(x_pts):
        """Inlet velocity at `x_pts` (2, n), returned as a (2, n) stack."""
        # If we have exact GT inlet, map points directly to GT inlet nodes
        if u_gt_inlet_nd is not None:
            _, nearest = kd_exact.query(x_pts.T)
            return np.stack([u_gt_inlet_nd[nearest, 0] * u_ref, u_gt_inlet_nd[nearest, 1] * u_ref])

        y = x_pts[0]*tangent[0] + x_pts[1]*tangent[1] - (inlet_center[0]*tangent[0] + inlet_center[1]*tangent[1])
        u_mag = get_inlet_profile(U_inlet, H_inlet, y, phys_cfg)
        return np.stack([u_mag * flow_dir[0], u_mag * flow_dir[1]])

    # Built only for the ground-truth branch, which is the only thing that queries it.
    kd_exact = KDTree(pos_target_nd * d_bar) if u_gt_inlet_nd is not None else None

    if sel_nodal.any():
        u_n = u_in_func(nodal_pts[:, sel_nodal])
        x_bc[basis.nodal_dofs[0, sel_nodal]] = u_n[0]
        x_bc[basis.nodal_dofs[1, sel_nodal]] = u_n[1]
    if sel_facet.any():
        u_f = u_in_func(facet_dof_locs[:, sel_facet])
        x_bc[basis.facet_dofs[0, sel_facet]] = u_f[0]
        x_bc[basis.facet_dofs[1, sel_facet]] = u_f[1]

    D_all = np.concatenate([D_inlet.all(), D_wall.all()]) if len(D_inlet.all()) > 0 else D_wall.all()

    basis_outlet = fem.FacetBasis(mesh, element, facets=mesh.boundaries["outlet"])

    @fem.BilinearForm
    def outlet_penalty(u, p, v, q, w):
        u_t = -u[0]*w.n[1] + u[1]*w.n[0]
        v_t = -v[0]*w.n[1] + v[1]*w.n[0]
        return 1e4 * u_t * v_t

    x_bc[D_wall.all()] = 0.0

    # Warm start.  Assigned AFTER `x_bc` is built so the boundary rows can be overwritten with
    # the conditions the solve actually imposes: a surrogate is only approximately no-slip and
    # only approximately the inlet profile, and seeding the wind with a wall that slips is the
    # one way a warm start can cost iterations instead of saving them.  Everything else --
    # pressure included -- keeps its zero, which is what the cold solve starts from.
    if u_init_nd is not None:
        u_init = np.asarray(u_init_nd, dtype=np.float64)
        if u_init.shape != (pos_target_nd.shape[0], 2):
            raise ValueError(
                f"u_init_nd has shape {u_init.shape}, expected "
                f"({pos_target_nd.shape[0]}, 2) (one row per graph node, [u, v] / u_ref)")
        if not np.isfinite(u_init).all():
            raise ValueError("u_init_nd contains non-finite entries")
        # Nearest graph node per velocity DoF.  On a P2 mesh file every DoF location IS a graph
        # node (vertices then mid-sides), so this is exact; on a P1 file the facet DoFs sit at
        # edge midpoints the graph may not carry and the nearest node is the right stand-in for
        # an initial guess.
        for blk in (basis.nodal_dofs, basis.facet_dofs):
            if blk.shape[1] == 0:
                continue
            _, near = kd.query(basis.doflocs[:, blk[0, :]].T)
            x[blk[0, :]] = u_init[near, 0] * u_ref
            x[blk[1, :]] = u_init[near, 1] * u_ref
        if len(D_all) > 0:
            x[D_all] = x_bc[D_all]

    # Outlet penalty is iterate-independent, so it is assembled once.
    A_outlet = fem.asm(outlet_penalty, basis_outlet) if basis_outlet is not None else None

    # `tol` is a RELATIVE tolerance on the undamped Picard increment, measured on the velocity
    # DoFs only.  The old criterion took an absolute norm over the whole solution vector, which
    # mixed velocity (m/s) with pressure (Pa) and therefore moved with the units, the mesh size
    # and the Reynolds number.
    vel_dofs = np.concatenate([basis.nodal_dofs[0], basis.nodal_dofs[1],
                               basis.facet_dofs[0], basis.facet_dofs[1]])

    damp = float(damping)
    prev_step = np.inf
    lu = None
    need_factor = True
    n_factor = 0
    lu_age = 0
    converged = False
    hist_x: list[np.ndarray] = []
    hist_f: list[np.ndarray] = []
    for it in range(max_iters):
        sol, pres = basis.interpolate(x)
        mu_eff = effective_viscosity(sol)

        # The Picard step is `x <- A(x)^-1 b`.  Factorising A costs ~36x a triangular solve
        # (0.94 s against 0.026 s on the largest anchor), and A changes only through the
        # viscosity and the wind, which barely move once the transient is over.  So the step is
        # taken as one defect correction against a FROZEN factorisation,
        #     x_full = x - M^-1 (A(x) x - b),
        # which has the same fixed point as the exact step for any invertible M -- the operator
        # is still reassembled every iteration, so nothing is approximated in the answer, only
        # in the rate at which it is reached.  M is refreshed whenever the increment stops
        # falling fast enough to pay for the reuse.
        if need_factor:
            A = fem.asm(navier_stokes_picard, basis, wind=sol, mu=mu_eff, **_tau_kw(sol, mu_eff))
            if A_outlet is not None:
                A = A + A_outlet
            b = np.zeros(A.shape[0])
            if len(D_all) > 0:
                A_enc, b_enc = fem.enforce(A, b, x=x_bc, D=D_all)
            else:
                A_enc, b_enc = A, b
            lu = splu(A_enc.tocsc())
            n_factor += 1
            need_factor = False
            x_full = lu.solve(b_enc)
        else:
            # `enforce` only zeroes the rows in `D_all` and puts 1 on their diagonal, so the
            # enforced residual is the raw one everywhere else and `x - x_bc` on those rows.
            r = fem.asm(navier_stokes_residual, basis, sol=sol, pres=pres, mu=mu_eff,
                        **_tau_kw(sol, mu_eff))
            if A_outlet is not None:
                r = r + A_outlet @ x
            if len(D_all) > 0:
                r[D_all] = x[D_all] - x_bc[D_all]
            x_full = x - lu.solve(r)

        step = x_full - x
        step_norm = float(np.linalg.norm(step[vel_dofs]))
        scale = max(float(np.linalg.norm(x[vel_dofs])), 1e-30)
        rel_step = step_norm / scale if it else np.inf

        # Anderson acceleration.  Plain Picard is a linear contraction and on the stenoses its
        # rate is ~0.98 per iteration -- 500+ iterations to a converged wall field, which is
        # why those vessels were being read off unconverged solves.  Extrapolating over the
        # last `anderson_m` iterates costs a least-squares solve of size m and does not touch
        # the fixed point: the correction is built from differences of iterates, so it vanishes
        # exactly when the residual does.  Disabled while damping is active, i.e. through the
        # transient, where the iterates are not yet in the asymptotic regime.
        if anderson_m > 0 and damp >= 1.0 and it >= 2:
            hist_x.append(x)
            hist_f.append(step)
            if len(hist_x) > anderson_m + 1:
                hist_x.pop(0)
                hist_f.pop(0)
        else:
            hist_x.clear()
            hist_f.clear()

        if len(hist_f) >= 2:
            dX = np.diff(np.asarray(hist_x), axis=0).T
            dF = np.diff(np.asarray(hist_f), axis=0).T
            normal = dF.T @ dF
            normal.flat[:: normal.shape[0] + 1] += 1e-10 * max(np.trace(normal), 1e-30)
            try:
                gamma = np.linalg.solve(normal, dF.T @ step)
                x = x + damp * step - (dX + damp * dF) @ gamma
            except np.linalg.LinAlgError:
                hist_x.clear()
                hist_f.clear()
                x = x + damp * step
        else:
            x = x + damp * step

        if verbose:
            print(f"Iter {it}: diff={damp * step_norm:.6e} rel={rel_step:.3e} damp={damp:.2f} lu={n_factor}")
        if rel_step < tol:
            converged = True
            break

        # Damping exists only to survive the opening transient, where the wind field is still
        # far from the fixed point.  Holding it at 0.5 for the whole solve caps the asymptotic
        # rate at 0.5 per iteration no matter how strongly the true Picard map contracts; here
        # it does contract (measured rate ~0.05), so releasing the damping once the increment
        # is falling monotonically is worth roughly 4x the iteration count.
        if step_norm < prev_step:
            damp = min(1.0, damp * 1.6)
        else:
            damp = max(0.25, 0.5 * damp)

        # Refresh the factorisation through the transient, and afterwards only when the
        # increment stops falling -- the sign that the frozen operator, not the nonlinearity,
        # is now what limits the rate.  Refreshing on a merely SLOW but monotone decrease is
        # waste: on the stenoses that rule refactorised on 113 of 120 iterations and bought
        # nothing, because there the rate is set by Picard itself.
        lu_age += 1
        if it < 2 or step_norm > lu_refresh * prev_step or lu_age >= lu_max_age:
            need_factor = True
            lu_age = 0
        prev_step = step_norm

    if not converged:
        # This has silently produced wrong stenosis fields twice.  A solve that runs out of
        # iterations must say so, because every downstream metric reads it as if it were the
        # converged answer.
        warnings.warn(
            f"local FEM did not converge in {max_iters} iterations "
            f"(last relative step {rel_step:.2e} against tol {tol:.1e}) for {mesh_path}; "
            "the returned field is NOT converged",
            RuntimeWarning, stacklevel=2,
        )

    # Interpolate results to target nodes (velocity lives on geometric mesh nodes).
    # P2 velocity lives on BOTH the vertices and the mid-side nodes, and the anchor meshes
    # carry both (patient001: 9490 points = 2447 vertices + 7043 mid-sides).  Writing only the
    # vertex block leaves every mid-side node at zero -- and the mid-sides are exactly the
    # near-wall ring the shear stencil differentiates across.  Restored after a 2026-08 edit
    # dropped the facet half and started raising a broadcast error on every P2 anchor mesh.
    u_skfem = np.zeros((mesh.p.shape[1], 2))
    u_skfem[:basis.nodal_dofs.shape[1], 0] = x[basis.nodal_dofs[0, :]]
    u_skfem[:basis.nodal_dofs.shape[1], 1] = x[basis.nodal_dofs[1, :]]
    if u_skfem.shape[0] > basis.nodal_dofs.shape[1]:
        u_skfem[basis.nodal_dofs.shape[1]:, 0] = x[basis.facet_dofs[0, :]]
        u_skfem[basis.nodal_dofs.shape[1]:, 1] = x[basis.facet_dofs[1, :]]

    u_pred = np.zeros((len(pos_target_nd), 2))
    u_pred[skfem_to_target] = u_skfem

    # A P1 mesh FILE (`mesh.p` covers only vertices) still solves full P2 velocity --
    # `element_v` is `ElementTriP2` regardless of the file's own storage order, skfem places
    # every facet DOF at its edge's geometric midpoint independent of whether that midpoint
    # was ever written to disk. The block above only ever registered `mesh.p.T` (corners) as
    # `skfem_to_target`, so on a P1 file the facet DOFs were solved and then discarded, and
    # any TARGET mid-side node (a customer graph elevated to P2 topology by
    # `src/data_gen/lib/p2_elevation.py`, which places its mid-side positions at the exact
    # same corner-mean the FEM edge DOF sits at) was left at the zero `u_pred` was initialised
    # to. Register those facet-DOF locations the same way and write them in -- tight enough
    # a tolerance that on any pack with no matching mid-side node (every pre-existing P1
    # caller) this is a no-op, exactly reproducing today's output.
    if mesh.p.shape[1] == basis.nodal_dofs.shape[1]:
        facet_xy = basis.doflocs[:, basis.facet_dofs[0, :]].T
        fdist, fidx = kd.query(facet_xy)
        fkeep = fdist <= 1e-6 * span
        if fkeep.any():
            u_facet = np.stack([x[basis.facet_dofs[0, :]], x[basis.facet_dofs[1, :]]], axis=1)
            u_pred[fidx[fkeep]] = u_facet[fkeep]

    # An identically zero field satisfies the Picard test at iteration 0 -- the increment from
    # the zero initial guess is zero, so the solve reports convergence and returns nothing.
    # That is how a zero-valued inlet Dirichlet (an absent ground truth mistaken for a real
    # one) produced "converged" arms with no flow at all.  A t=0 vessel always flows.
    if not np.isfinite(u_pred).all():
        raise ValueError(f"local FEM: non-finite velocity from {mesh_path!r}")
    u_max = float(np.abs(u_pred).max())
    if u_max <= 1e-6 * abs(u_ref):
        raise ValueError(
            f"local FEM: solve on {mesh_path!r} converged to an identically zero velocity "
            f"field (|u|max={u_max:.3e}, u_ref={u_ref:.3e}); the inlet condition carried no flow"
        )

    u_pred = torch.tensor(u_pred, dtype=torch.float32)

    return u_pred
