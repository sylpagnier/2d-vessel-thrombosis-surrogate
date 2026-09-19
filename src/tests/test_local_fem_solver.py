
import numpy as np
import pytest
import torch

from src.config import PhysicsConfig
from src.core_physics.local_fem_solver import solve_local_t0_flow
from src.utils.paths import anchor_meshes_dir, anchor_packs_dir, get_project_root

ROOT = get_project_root()


def _research_case():
    """First cached research vessel that has both a graph and its `.msh`."""
    cache = ROOT / "outputs" / "research_sweeps" / "_meshes"
    for msh in sorted(cache.glob("research_*.msh")):
        pt = cache / (msh.name[len("research_"):-len(".msh")] + ".pt")
        if pt.is_file():
            return pt, msh
    return None, None


def test_local_fem_solver():
    pt_path = anchor_packs_dir() / "comsol001.pt"
    nas_path = anchor_meshes_dir() / "comsol001.nas"

    if not pt_path.is_file() or not nas_path.is_file():
        pytest.skip("Data not found")

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    u_pred = solve_local_t0_flow(str(nas_path), data, PhysicsConfig(), max_iters=2)

    assert u_pred.shape == (data.x.shape[0], 2)
    assert np.max(np.linalg.norm(u_pred, axis=1)) > 0.0


def test_research_mesh_registers_and_respects_no_slip():
    """The research `.msh` vessels are in metres; the COMSOL anchor `.nas` anchors are in cm.

    A hardcoded cm->m scale collapsed every research mesh onto ~3 pack nodes, so the wall
    tagged zero facets, the solve was exactly singular, and the sweeps still scored.  Pin
    the two properties that were violated: the mesh registers onto the pack node-for-node,
    and the converged field is no-slip at the wall.
    """
    pt_path, msh_path = _research_case()
    if pt_path is None:
        pytest.skip("no cached research mesh")

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    u = solve_local_t0_flow(str(msh_path), data, PhysicsConfig(), max_iters=40, tol=1e-9)
    u = u.numpy() if hasattr(u, "numpy") else np.asarray(u)
    mag = np.linalg.norm(u, axis=1)

    assert not np.isnan(u).any()

    wall = data.mask_wall.cpu().numpy().astype(bool).ravel()
    u_ref = float(np.asarray(data.u_ref).reshape(-1)[0])
    assert mag[wall].max() < 1e-6 * u_ref, "no-slip violated: wall was not tagged"
    assert mag[~wall].mean() > 0.1 * u_ref, "interior flow is degenerate"


def test_mis_scaled_mesh_is_refused_not_solved():
    """A mesh that does not register must raise, never solve to garbage that still scores."""
    pt_path, msh_path = _research_case()
    if pt_path is None:
        pytest.skip("no cached research mesh")

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    data = data.clone()
    data.x = data.x.clone()
    data.x[:, 0:2] *= 7.3  # a scale no candidate factor can recover

    with pytest.raises(ValueError, match="does not register"):
        solve_local_t0_flow(str(msh_path), data, PhysicsConfig(), max_iters=1)


def test_zero_inlet_is_refused_not_reported_as_converged():
    """A zero inlet Dirichlet drives the solve to exactly zero, which passes the Picard test.

    The increment from the zero initial guess is zero, so the loop reports convergence at
    iteration 0 and returns no flow at all.  That is how research arms whose packs carry an
    all-zero `y[0]` produced "converged" sweeps with no velocity.
    """
    pt_path, msh_path = _research_case()
    if pt_path is None:
        pytest.skip("no cached research mesh")

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    zeros = np.zeros((data.x.shape[0], 2), dtype=np.float64)

    with pytest.raises(ValueError, match="identically zero"):
        solve_local_t0_flow(str(msh_path), data, PhysicsConfig(), max_iters=3,
                            u_gt_inlet_nd=zeros)


def test_absent_ground_truth_falls_back_to_the_analytic_inlet():
    """`research_synthetic` post-dates some caches, so absence of the flag is not evidence of
    a ground truth.  `solve_fem_into_pack` must test `y[0]` itself and fall back."""
    from src.clot_ml.v0 import solve_fem_into_pack

    pt_path, msh_path = _research_case()
    if pt_path is None:
        pytest.skip("no cached research mesh")

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    data.mesh_path = str(msh_path)
    if hasattr(data, "research_synthetic"):
        del data.research_synthetic          # the pre-flag cache state
    assert float(data.y[0, :, 0:2].abs().max()) == 0.0, "fixture must have no GT flow"

    solve_fem_into_pack(data)

    u_ref = float(np.asarray(data.u_ref).reshape(-1)[0])
    speed = torch.hypot(data.u0_pred, data.v0_pred)
    assert float(speed.max()) > 0.5 * u_ref, "fell back to a zero field instead of the profile"
