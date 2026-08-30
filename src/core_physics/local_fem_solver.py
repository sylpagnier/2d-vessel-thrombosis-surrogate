import numpy as np
import skfem as fem
from skfem.helpers import grad, dot, sym_grad
from scipy.sparse.linalg import spsolve
from scipy.spatial import KDTree
import torch
from src.config import PhysicsConfig
from src.core_physics.inlet_profile import get_inlet_profile

def solve_local_t0_flow(mesh_path, data, phys_cfg: PhysicsConfig, max_iters=20, tol=1e-5, u_gt_inlet_nd: np.ndarray | None = None):
    """
    Solves steady-state Carreau Navier-Stokes on the .nas/.msh mesh.
    """
    mesh = fem.Mesh.load(mesh_path)
    
    pos_target_nd = data.x[:, 0:2].cpu().numpy()
    d_bar = float(data.d_bar.item()) if hasattr(data.d_bar, 'item') else float(data.d_bar)
    u_ref = float(data.u_ref.item()) if hasattr(data.u_ref, 'item') else float(data.u_ref)
    
    # Scale mesh to target
    mesh = mesh.scaled(0.01)
    
    kd = KDTree(pos_target_nd * d_bar)
    _, skfem_to_target = kd.query(mesh.p.T)
    
    mask_inlet = data.mask_inlet.cpu().numpy().astype(bool)
    mask_outlet = data.mask_outlet.cpu().numpy().astype(bool)
    mask_wall = data.mask_wall.cpu().numpy().astype(bool)
    
    is_inlet = mask_inlet[skfem_to_target]
    is_outlet = mask_outlet[skfem_to_target]
    is_wall = mask_wall[skfem_to_target]
    
    def get_facets(node_mask):
        return np.nonzero(np.all(node_mask[mesh.facets], axis=0))[0]
        
    mesh = mesh.with_boundaries({
        "inlet": get_facets(is_inlet),
        "outlet": get_facets(is_outlet),
        "wall": get_facets(is_wall)
    })
    
    element_v = fem.ElementVector(fem.ElementTriP2())
    element_p = fem.ElementTriP1()
    element = element_v * element_p
    basis = fem.Basis(mesh, element)
    
    rho = phys_cfg.rho
    mu_inf = phys_cfg.mu_inf
    mu_0 = phys_cfg.mu_0
    lam = phys_cfg.lam
    n_car = phys_cfg.n
    a = phys_cfg.a
    
    @fem.BilinearForm
    def navier_stokes_picard(u, p, v, q, w):
        u_prev = w.w[0]
        eps_u = sym_grad(u)
        eps_v = sym_grad(v)
        eps_prev = sym_grad(u_prev)
        
        gamma_dot = np.sqrt(2.0 * (eps_prev[0][0]**2 + eps_prev[1][1]**2 + 2*eps_prev[0][1]**2) + 1e-12)
        mu_car = mu_inf + (mu_0 - mu_inf) * (1.0 + (lam * gamma_dot)**a)**((n_car - 1) / a)
        
        # Simple isotropic artificial viscosity (approximate SUPG/Crosswind)
        u_mag = np.sqrt(u_prev[0]**2 + u_prev[1]**2 + 1e-12)
        
        h = 0.0005
        mu_art = 0.5 * rho * u_mag * h
        mu = mu_car + mu_art
        
        gu = grad(u)
        gv = grad(v)
        convective = rho * (v[0] * (u_prev[0] * gu[0][0] + u_prev[1] * gu[0][1]) + v[1] * (u_prev[0] * gu[1][0] + u_prev[1] * gu[1][1]))
        viscous = 2.0 * mu * (eps_u[0][0]*eps_v[0][0] + eps_u[1][1]*eps_v[1][1] + 2*eps_u[0][1]*eps_v[0][1])
        pressure = -p * (gv[0][0] + gv[1][1])
        continuity = -q * (gu[0][0] + gu[1][1])
        
        return convective + viscous + pressure + continuity
    
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
        
        U_inlet = (phys_cfg.re_target * mu_inf) / (rho * d_bar)
        H_inlet = np.max(np.dot(inlet_pts, tangent)) - np.min(np.dot(inlet_pts, tangent))
        
        mass_center = mesh.p.mean(axis=1)
        if np.dot(mass_center - inlet_center, normal) > 0:
            flow_dir = normal
        else:
            flow_dir = -normal
            
        @fem.LinearForm
        def inlet_rhs(v, w):
            y = w.x[0]*tangent[0] + w.x[1]*tangent[1]
            # get_inlet_profile only uses U_inlet, H_inlet, y, phys_cfg
            # Since w.x is an array (integration points), we pass it to python safely
            return 0.0 * v[0] # we will enforce Dirichlet explicitly
    else:
        U_inlet = 0.0
        H_inlet = 1.0
        tangent = np.array([0, 1])
        flow_dir = np.array([1, 0])
        inlet_center = np.array([0, 0])
        
    D_inlet = basis.get_dofs("inlet").drop('p')
    D_wall = basis.get_dofs("wall").drop('p')
    D_all = np.concatenate([D_inlet.all(), D_wall.all()]) if len(D_inlet.all()) > 0 else D_wall.all()
    
    # We evaluate get_inlet_profile at the exact node locations to set Dirichlet values.
    x_bc = np.zeros(basis.N)
    if len(inlet_facets) > 0:
        # basis.get_dofs returns DoF indices. 
        # For each DOF, we can find its node coordinate and set the value.
        basis_v = basis.split_bases()[0]
        # nodal_dofs has shape (2, N_nodes). 
        for comp in range(2):
            for i, node_idx in enumerate(inlet_nodes):
                # get dof for this node
                # Wait, this requires knowing the dof for the specific node.
                pass
                
        # A simpler way to enforce inlet:
        # Use skfem.project on the inlet boundary
        pass
    
    # Let's simply set the inlet velocity to 0 for now in this test.
    # To properly set it, I will define a function and interpolate it over basis_v
    
    kd_exact = KDTree(pos_target_nd * d_bar)

    def u_in_func(x_pts):
        # If we have exact GT inlet, map points directly to GT inlet nodes
        if u_gt_inlet_nd is not None:
            _, nearest = kd_exact.query(x_pts.T)
            return np.stack([u_gt_inlet_nd[nearest, 0] * u_ref, u_gt_inlet_nd[nearest, 1] * u_ref])
            
        y = x_pts[0]*tangent[0] + x_pts[1]*tangent[1] - (inlet_center[0]*tangent[0] + inlet_center[1]*tangent[1])
        u_mag = get_inlet_profile(U_inlet, H_inlet, y, phys_cfg)
        return np.stack([u_mag * flow_dir[0], u_mag * flow_dir[1]])
        
    x_bc = np.zeros(basis.N)
    
    if u_gt_inlet_nd is not None:
        _, nearest_corners = kd_exact.query(mesh.p[:, :basis.nodal_dofs.shape[1]].T)
        x_bc[basis.nodal_dofs[0, :]] = u_gt_inlet_nd[nearest_corners, 0] * u_ref
        x_bc[basis.nodal_dofs[1, :]] = u_gt_inlet_nd[nearest_corners, 1] * u_ref
        
        _, nearest_facets = kd_exact.query(mesh.p[:, basis.nodal_dofs.shape[1]:].T)
        x_bc[basis.facet_dofs[0, :]] = u_gt_inlet_nd[nearest_facets, 0] * u_ref
        x_bc[basis.facet_dofs[1, :]] = u_gt_inlet_nd[nearest_facets, 1] * u_ref
    else:
        x_bc[basis.nodal_dofs[0, :]] = u_in_func(mesh.p[:, :basis.nodal_dofs.shape[1]])[0]
        x_bc[basis.nodal_dofs[1, :]] = u_in_func(mesh.p[:, :basis.nodal_dofs.shape[1]])[1]
        facet_pts = mesh.p[:, basis.nodal_dofs.shape[1]:]
        x_bc[basis.facet_dofs[0, :]] = u_in_func(facet_pts)[0]
        x_bc[basis.facet_dofs[1, :]] = u_in_func(facet_pts)[1]
    D_inlet = basis.get_dofs("inlet").drop('u^2')
    D_wall = basis.get_dofs("wall").drop('u^2')
    D_all = np.concatenate([D_inlet.all(), D_wall.all()]) if len(D_inlet.all()) > 0 else D_wall.all()
    
    basis_outlet = fem.FacetBasis(mesh, element, facets=mesh.boundaries["outlet"])
    
    @fem.BilinearForm
    def outlet_penalty(u, p, v, q, w):
        u_t = -u[0]*w.n[1] + u[1]*w.n[0]
        v_t = -v[0]*w.n[1] + v[1]*w.n[0]
        return 1e4 * u_t * v_t

    x_bc[D_wall.all()] = 0.0

    for it in range(max_iters):
        A = fem.asm(navier_stokes_picard, basis, w=basis.interpolate(x))
        if basis_outlet is not None:
            A = A + fem.asm(outlet_penalty, basis_outlet)
        b = np.zeros(A.shape[0])
        
        if len(D_all) > 0:
            A_enc, b_enc = fem.enforce(A, b, x=x_bc, D=D_all)
            x_new = fem.solve(A_enc, b_enc)
        else:
            x_new = fem.solve(A, b)
            
        x_new = 0.5 * x_new + 0.5 * x
            
        diff = np.linalg.norm(x_new - x)
        print(f"Iter {it}: diff={diff:.6e}")
        x = x_new
        if diff < tol:
            break
            
    # Interpolate results to target nodes
    u_skfem = np.zeros((mesh.p.shape[1], 2))
    u_skfem[:basis.nodal_dofs.shape[1], 0] = x[basis.nodal_dofs[0, :]]
    u_skfem[:basis.nodal_dofs.shape[1], 1] = x[basis.nodal_dofs[1, :]]
    u_skfem[basis.nodal_dofs.shape[1]:, 0] = x[basis.facet_dofs[0, :]]
    u_skfem[basis.nodal_dofs.shape[1]:, 1] = x[basis.facet_dofs[1, :]]
    
    u_pred = np.zeros((len(pos_target_nd), 2))
    u_pred[skfem_to_target] = u_skfem
    u_pred = torch.tensor(u_pred, dtype=torch.float32)
    
    return u_pred
