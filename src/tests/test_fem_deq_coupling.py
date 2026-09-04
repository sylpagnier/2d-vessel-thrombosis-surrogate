"""The two RGP-DEQ <-> local-FEM couplings: warm-starting the solve, and the FEM prior block.

Both rest on one claim each that is cheap to state and expensive to discover violated:

  * a warm start moves the ITERATION COUNT and not the ANSWER (Picard's iterate is only the
    wind the operator is linearised about);
  * ``prior_source="fem"`` is deploy-legal -- it must never read COMSOL's velocity, and it must
    leave the hard BC ``u = uv_prior + sdf * r`` exact at the wall.
"""
import numpy as np
import pytest
import torch

from src.config import NodeFeat, PhysicsConfig
from src.core_physics.local_fem_solver import solve_local_t0_flow
from src.utils.paths import get_project_root

ROOT = get_project_root()

PACK = ROOT / "data" / "processed" / "graphs_biochem_anchors" / "patient001.pt"
MESH = ROOT / "data" / "raw" / "biochem_anchors" / "patient001.nas"


def _pack():
    if not PACK.is_file() or not MESH.is_file():
        pytest.skip("patient001 pack or mesh not found")
    data = torch.load(PACK, map_location="cpu", weights_only=False)
    data.graph_stem = "patient001"
    return data


def test_warm_start_reaches_the_same_fixed_point():
    """A warm start that changes the converged field is a bug, not a speedup."""
    data = _pack()
    kw = dict(max_iters=300, tol=1e-9, verbose=False)

    cold = solve_local_t0_flow(str(MESH), data, PhysicsConfig(), **kw).numpy()

    from src.data_gen.lib.legal_priors import build_analytic_priors

    au, av, _, _ = build_analytic_priors(data)
    seed = torch.stack([au.reshape(-1), av.reshape(-1)], dim=1).numpy().astype(np.float64)
    warm = solve_local_t0_flow(str(MESH), data, PhysicsConfig(), u_init_nd=seed, **kw).numpy()

    scale = max(float(np.abs(cold).max()), 1e-30)
    assert float(np.abs(warm - cold).max()) / scale < 1e-5


def test_warm_start_rejects_a_mis_shaped_seed():
    """Silently broadcasting a wrong-length seed would seed the wind with another vessel."""
    data = _pack()
    with pytest.raises(ValueError, match="u_init_nd"):
        solve_local_t0_flow(str(MESH), data, PhysicsConfig(), max_iters=1, verbose=False,
                            u_init_nd=np.zeros((int(data.num_nodes) + 1, 2)))


def test_fem_prior_is_a_closer_base_point_than_the_analytic_one():
    from src.data_gen.lib.legal_priors import COL_U_PRIOR, COL_V_PRIOR, apply_prior_source

    data = _pack()
    g = data.y[0, :, 0:2].numpy().astype(np.float64)
    den = np.linalg.norm(g)

    def rel(src):
        p = apply_prior_source(data, src).x[:, [COL_U_PRIOR, COL_V_PRIOR]].numpy().astype(np.float64)
        return float(np.linalg.norm(p - g) / den)

    fem, analytic = rel("fem"), rel("analytic")
    assert fem < analytic, f"fem prior {fem:.4f} is not closer than analytic {analytic:.4f}"


def test_fem_prior_keeps_the_hard_bc_exact_at_the_wall():
    """``u = uv_prior + sdf * r`` is only a hard BC if the prior itself vanishes at sdf=0."""
    from src.data_gen.lib.legal_priors import COL_U_PRIOR, COL_V_PRIOR, apply_prior_source

    data = _pack()
    out = apply_prior_source(data, "fem")
    p = out.x[:, [COL_U_PRIOR, COL_V_PRIOR]].numpy().astype(np.float64)
    sdf = out.x[:, NodeFeat.SDF].reshape(-1).numpy().astype(np.float64)
    at_wall = sdf <= 1e-6
    if not at_wall.any():
        pytest.skip("pack carries no sdf=0 nodes")
    assert float(np.abs(p[at_wall]).max()) == 0.0


def test_fem_prior_never_reads_the_comsol_inlet(monkeypatch):
    """s17 Z2: the prior block is a model INPUT, so a GT inlet Dirichlet would be a leak."""
    import src.core_physics.local_fem_solver as solver_mod
    from src.data_gen.lib import legal_priors

    data = _pack()
    seen = {}
    real = solver_mod.solve_local_t0_flow

    def spy(*args, **kwargs):
        seen["u_gt_inlet_nd"] = kwargs.get("u_gt_inlet_nd", "MISSING")
        return real(*args, **kwargs)

    monkeypatch.setattr(solver_mod, "solve_local_t0_flow", spy)
    # `cache=False` so a warm cache cannot skip the solve this test exists to inspect.
    legal_priors.build_fem_priors(data, cache=False)
    assert seen.get("u_gt_inlet_nd") is None


def test_fem_prior_does_not_mutate_the_callers_pack():
    from src.data_gen.lib.legal_priors import apply_prior_source

    data = _pack()
    before = data.x.clone()
    apply_prior_source(data, "fem")
    assert torch.equal(data.x, before)
