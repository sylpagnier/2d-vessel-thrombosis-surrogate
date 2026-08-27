"""Regression tests for docs/RGP_DEQ_REPAIR_PLAN.md.

Each test pins one bug from the plan's table so it cannot come back silently.  Everything here
builds its own mesh: the real packs are ~335 MB and are not a test dependency.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from src.data_gen.lib.legal_priors import (
    _lsq_gradient,
    apply_prior_source,
    assert_train_deploy_prior_parity,
)
from src.data_gen.lib.mesh_wls import rank_aware_pinv_sym
from src.utils.kinematics_physics_terms import wall_band_mask
from src.utils.math_operators import wls_derivatives

REPO = Path(__file__).resolve().parents[2]


def _p2_strip(nx: int = 12, ny: int = 4):
    """A triangulated strip with P2 mid-side nodes: the biochem mesh topology in miniature.

    Corner nodes on a grid, plus one mid-side node on every edge, wired corner-mid-corner --
    so every mid-side node has degree 2 with exactly antiparallel edge vectors, which is the
    condition that breaks a 2x2 / 5x5 least-squares stencil.
    """
    pos = []
    idx = {}
    for i in range(nx):
        for j in range(ny):
            idx[(i, j)] = len(pos)
            pos.append((float(i), float(j)))
    edges = []
    corner_pairs = []
    for i in range(nx):
        for j in range(ny):
            for di, dj in ((1, 0), (0, 1), (1, 1)):
                b = (i + di, j + dj)
                if b in idx:
                    corner_pairs.append((idx[(i, j)], idx[b]))
    for a, b in corner_pairs:
        m = len(pos)
        pos.append(((pos[a][0] + pos[b][0]) / 2.0, (pos[a][1] + pos[b][1]) / 2.0))
        edges += [(a, m), (m, a), (m, b), (b, m)]
    # Corner-to-corner edges too, so corners keep a full-rank 2nd-order stencil.
    for a, b in corner_pairs:
        edges += [(a, b), (b, a)]
    pos_t = torch.tensor(pos, dtype=torch.float64)
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    n = pos_t.shape[0]
    deg = torch.zeros(n)
    deg.index_add_(0, ei[0], torch.ones(ei.shape[1]))
    return pos_t, ei, n, deg


def _wls_parts(pos, ei, n):
    row, col = ei
    d = pos[col] - pos[row]
    dx, dy = d[:, 0], d[:, 1]
    V = torch.stack([dx, dy, 0.5 * dx**2, dx * dy, 0.5 * dy**2], dim=1)
    W = 1.0 / (dx**2 + dy**2 + 1e-8)
    Me = W.view(-1, 1, 1) * torch.bmm(V.unsqueeze(2), V.unsqueeze(1))
    M = torch.zeros(n, 25, dtype=Me.dtype).scatter_add_(
        0, row.view(-1, 1).expand(-1, 25), Me.view(-1, 25)
    ).view(n, 5, 5)
    return V, W, M


# --- B3 / D5: rank-deficient stencils -------------------------------------------------------

def test_p2_midside_nodes_really_are_collinear():
    """The premise of B3/D5: this is a property of the mesh, not a numerical accident."""
    pos, ei, n, deg = _p2_strip()
    row, col = ei
    mids = (deg == 2).nonzero().reshape(-1)
    assert mids.numel() > 0
    worst = 0.0
    for i in mids[:200].tolist():
        nb = col[row == i]
        v1, v2 = pos[nb[0]] - pos[i], pos[nb[1]] - pos[i]
        cos = torch.nn.functional.cosine_similarity(v1[None], v2[None]).item()
        worst = max(worst, cos + 1.0)  # exactly antiparallel => cos == -1
    assert worst < 1e-9, f"mid-side stencils are not collinear (worst cos+1 = {worst})"


def test_rank_aware_pinv_never_claims_rank_the_stencil_lacks():
    """What the truncated operator guarantees, and nothing more.

    Note what this does NOT assert.  On these meshes `pinv(M + 1e-6*I, rcond=1e-5)` produces
    the *same* per-node rank as honest truncation at every scale tested (1.0 down to 1e-3), and
    on `patient020` the two operators agree to within 3% in norm.  So the ridge is not the
    active defect here -- the rank deficiency itself is, and the repair that matters is the
    neighbour fill in `wls_derivatives`.  Truncation is kept because it is unconditionally
    correct, not because it was measured to fix something.
    """
    for scale in (1.0, 1e-2):
        pos, ei, n, deg = _p2_strip()
        pos = pos * scale
        _, _, M = _wls_parts(pos, ei, n)
        new = rank_aware_pinv_sym(M)
        mids = deg == 2

        ev_M = torch.linalg.eigvalsh(M).abs()
        rank_M = ((ev_M / ev_M.amax(dim=1, keepdim=True).clamp(min=1e-300)) > 1e-5).sum(dim=1)
        ev_i = torch.linalg.eigvalsh(new).abs()
        rank_i = ((ev_i / ev_i.amax(dim=1, keepdim=True).clamp(min=1e-300)) > 1e-8).sum(dim=1)

        assert torch.all(rank_i <= rank_M), (
            f"scale={scale}: the inverse claims more resolved directions than the stencil has"
        )
        assert torch.isfinite(new).all()
        assert int(rank_i[mids].max()) <= 2, "a collinear 2-neighbour stencil is rank <= 2"


def test_wls_derivatives_recover_a_quadratic_on_a_p2_mesh():
    """The fill is what makes a P2 mesh usable: without it 3/4 of the nodes carry no gradient."""
    pos, ei, n, deg = _p2_strip()
    V, W, M = _wls_parts(pos, ei, n)
    M_inv = rank_aware_pinv_sym(M)

    f = 0.5 * pos[:, 0] ** 2 + pos[:, 0] * pos[:, 1]
    grad_true = torch.stack([pos[:, 0] + pos[:, 1], pos[:, 0]], dim=1)

    got = wls_derivatives(f.unsqueeze(1), ei, n, V, W, M_inv)[:, 0:2, 0]
    err = (got - grad_true).norm(dim=1) / grad_true.norm(dim=1).clamp(min=1e-9)

    interior = (pos[:, 0] > 1.5) & (pos[:, 0] < 9.5) & (pos[:, 1] > 0.5) & (pos[:, 1] < 2.5)
    mids = (deg == 2) & interior
    assert int(mids.sum()) > 10
    assert float(err[mids].median()) < 1e-6, (
        f"mid-side rows still carry no usable gradient (median rel err {float(err[mids].median()):.2e})"
    )


def test_lsq_gradient_is_not_degree_dependent():
    """B3: the 2x2 gradient used by `potential_flow_direction`."""
    pos, ei, n, deg = _p2_strip()
    row, col = ei
    f = 3.0 * pos[:, 0] - 2.0 * pos[:, 1]
    gx, gy = _lsq_gradient(f, pos, row, col, n, torch.device("cpu"))
    got = torch.stack([gx, gy], dim=1)
    truth = torch.tensor([3.0, -2.0], dtype=got.dtype)

    interior = (pos[:, 0] > 1.5) & (pos[:, 0] < 9.5) & (pos[:, 1] > 0.5) & (pos[:, 1] < 2.5)
    err = (got - truth).norm(dim=1) / truth.norm()
    for name, m in (("mid-side", (deg == 2) & interior), ("corner", (deg > 2) & interior)):
        assert int(m.sum()) > 5, name
        assert float(err[m].median()) < 1e-6, (
            f"{name} nodes: median rel err {float(err[m].median()):.2e}; a rank-1 stencil must "
            "be filled from neighbours, not inverted through a ridge"
        )


# --- B10: the wall band must be a real dilation ---------------------------------------------

def test_wall_band_mask_is_a_true_dilation():
    """`acc[row] = band[col]` is last-write-wins and silently subsamples the band."""
    n = 6
    ei = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]])

    class _D:
        num_nodes = n
        edge_index = ei
        mask_wall = torch.tensor([True, False, False, False, False, False])
        x = torch.zeros(n, 2)

    d = _D()
    assert wall_band_mask(d, 0).tolist() == [True, False, False, False, False, False]
    d2 = _D()
    assert wall_band_mask(d2, 2).tolist() == [True, True, True, False, False, False]
    d3 = _D()
    got = wall_band_mask(d3, 3)
    assert got.tolist() == [True, True, True, True, False, False]
    # The buggy form would drop node 1 here: its LAST incident edge points at node 2.
    buggy = d3.mask_wall.clone()
    for _ in range(3):
        grown = torch.zeros_like(buggy)
        grown[ei[0]] = buggy[ei[1]]
        buggy = buggy | grown
    assert buggy.sum() < got.sum(), "premise changed: the buggy dilation no longer undercounts"


# --- B12: prior source must not alias the caller --------------------------------------------

def test_apply_prior_source_does_not_alias_the_callers_pack():
    from torch_geometric.data import Data

    n = 8
    d = Data(x=torch.zeros(n, 18), edge_index=torch.tensor([[0, 1], [1, 0]]))
    d.y = torch.ones(n, 5)
    out = apply_prior_source(d, "zero")
    out.x[:, 11] = 7.0
    assert float(d.x[:, 11].abs().max()) == 0.0, "writing to the result reached the caller's pack"
    assert out.y.data_ptr() == d.y.data_ptr(), "y was deep-copied; that is a 335 MB regression"


def test_prior_parity_assert_rejects_a_train_deploy_mismatch():
    assert_train_deploy_prior_parity("analytic", "analytic")
    with pytest.raises(ValueError):
        assert_train_deploy_prior_parity("stored", "analytic")


# --- B1 / B2 / B6 / B8: wiring that can only be checked at the source ------------------------

def test_precache_applies_a_prior_source_before_the_solve():
    src = (REPO / "scripts" / "precache_rgp_deq.py").read_text(encoding="utf-8")
    i_apply = src.find("apply_prior_source(")
    i_solve = src.find("predict_kinematics_and_latent(")
    assert i_apply > 0, "precache never rewrites the prior block (B1)"
    assert i_apply < i_solve, (
        "priors are rewritten AFTER the DEQ solve; the solve consumes UV_PRIOR/MU_PRIOR, so "
        "this would be a silent no-op"
    )
    assert 'default="analytic"' in src or "default=\"analytic\"" in src


def test_trainer_never_mints_a_best_checkpoint_it_did_not_promote():
    src = (REPO / "src" / "training" / "train_kinematics_predictor.py").read_text(encoding="utf-8")
    # Comments are allowed to mention the removed call; executable lines are not.
    code = [
        ln for ln in src.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    assert not any("shutil.copy2" in ln for ln in code), (
        "the trainer copies kinematics_ckpt_latest over kinematics_best; that is how a "
        "NaN-metric 'best' gets minted (B6)"
    )
    assert "_apply_prior_source_to_dataset" in src, "trainer never applies the prior source (B2)"


def test_finetune_holds_out_every_sealed_vessel():
    from src.core_physics.wall_cohort_splits import SEALED

    src = (REPO / "scripts" / "finetune_kine_patient_anchors.py").read_text(encoding="utf-8")
    assert "_default_holdout_stems" in src
    assert "wall_cohort_splits" in src, "holdout is hard-coded rather than derived (B8)"
    assert len(set(SEALED)) >= 4


# --- B5 / B7 / D7: promotion evidence -------------------------------------------------------

def test_assert_promotable_rejects_a_latest_checkpoint(tmp_path):
    from src.utils.kinematics_inference import assert_promotable_checkpoint

    bad = tmp_path / "kinematics_best.pth"
    torch.save(
        {
            "model_state_dict": {},
            "checkpoint_role": "kinematics_ckpt_latest",
            "composite": float("nan"),
            "run_id": "",
        },
        bad,
    )
    with pytest.raises(ValueError) as e:
        assert_promotable_checkpoint(bad)
    msg = str(e.value)
    assert "kinematics_ckpt_latest" in msg and "NaN" in msg

    good = tmp_path / "ok.pth"
    torch.save(
        {
            "model_state_dict": {},
            "checkpoint_role": "kinematics_best",
            "composite": 0.31,
            "rel_l2": 0.1,
            "continuity": 0.002,
            "run_id": "20260827T120000Z",
            "prior_source": "analytic",
        },
        good,
    )
    meta = assert_promotable_checkpoint(good)
    assert meta["prior_source"] == "analytic"
    assert math.isfinite(float(meta["composite"]))


# --- D2 / D3: the new objective terms -------------------------------------------------------

_PACK = REPO / "data" / "processed" / "graphs_kinematics_anchors" / "carreau" / "patient020.pt"


@pytest.mark.skipif(not _PACK.is_file(), reason="needs a local kinematics anchor pack")
def test_wall_band_shear_terms_are_zero_at_truth_and_punish_under_scaling():
    """The failure mode these exist to catch: right structure, wrong amplitude.

    Every measured `sr`/`dsrx` scale in §1j is < 1 against COMSOL, and an unnormalised MSE
    rewards shrinking further.  Normalising by the GT's own spread on the band removes that.
    """
    from src.config import PhysicsConfig
    from src.core_physics.physics_kernels import PhysicsKernels
    from src.utils.kinematics_physics_terms import compute_kinematics_physics_terms

    d = torch.load(_PACK, map_location="cpu", weights_only=False)
    k = PhysicsKernels(phys_cfg=PhysicsConfig(phase="kinematics"))

    exact = torch.cat([d.y[:, :4], torch.zeros(int(d.num_nodes), 2)], dim=1)
    t = compute_kinematics_physics_terms(exact, d, k, wall_band_hops=3)
    assert float(t["l_band_sr"]) < 1e-6
    assert float(t["l_band_dsrx"]) < 1e-6

    shrunk = exact.clone()
    shrunk[:, 0:2] *= 0.4
    t2 = compute_kinematics_physics_terms(shrunk, d, k, wall_band_hops=3)
    assert float(t2["l_band_sr"]) > 1e-2, "an under-scaled field must not be free"
    assert float(t2["l_band_dsrx"]) > 1e-3


def test_relative_data_loss_is_scale_invariant():
    """D3: so a 2x spread in `u_ref` across the cohort stops re-weighting the vessels."""
    from src.utils.kinematics_physics_terms import boundary_weighted_mse

    n = 64
    torch.manual_seed(0)
    y = torch.randn(n, 5)
    pred = y.clone()
    pred[:, 0:2] *= 0.4
    mask = torch.ones(n, dtype=torch.bool)

    a = float(boundary_weighted_mse(pred, y, mask, relative=True))
    b = float(boundary_weighted_mse(pred * 3.0, y * 3.0, mask, relative=True))
    assert a == pytest.approx(b, rel=1e-6), "relative loss still depends on the field's scale"

    absolute_a = float(boundary_weighted_mse(pred, y, mask, relative=False))
    absolute_b = float(boundary_weighted_mse(pred * 3.0, y * 3.0, mask, relative=False))
    assert absolute_b > absolute_a * 5, "premise changed: the absolute loss was already scale-free"


def test_checkpoint_writer_records_the_prior_source():
    import inspect

    from src.architecture.kinematics_model_config import save_kinematics_checkpoint_file

    sig = inspect.signature(save_kinematics_checkpoint_file)
    assert "prior_source" in sig.parameters, (
        "a checkpoint that does not say what priors it was trained with cannot be parity-checked"
    )
