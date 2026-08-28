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


# --- T1 / T2: the training-vs-deploy geometry gap and the P1 corner graph --------------------

def test_geometry_sync_covers_the_mesh_channels_and_nothing_else():
    """B14: the sync must rewrite mesh channels and leave labels + the prior block alone."""
    from torch_geometric.data import Data

    from src.utils.kinematics_paths import (
        GEOMETRY_SYNC_CHANNELS,
        sync_geometry_from_deploy_pack,
    )

    # The prior block (11-14) must NOT be synced: `apply_prior_source` owns it, and the stored
    # values are the s17 Z2 leak.  Positions and sdf are identical across copies by definition.
    assert set(GEOMETRY_SYNC_CHANNELS).isdisjoint({0, 1, 2, 3, 10, 11, 12, 13, 14})
    assert set(GEOMETRY_SYNC_CHANNELS) == {4, 5, 6, 7, 8, 9, 15, 16, 17}

    n = 6
    train = Data(x=torch.zeros(n, 18), edge_index=torch.tensor([[0, 1], [1, 0]]))
    train.graph_stem = "patient001"
    ref = Data(x=torch.arange(n * 18, dtype=torch.float32).reshape(n, 18))
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        torch.save(ref, Path(td) / "patient001.pt")
        assert sync_geometry_from_deploy_pack(train, deploy_dir=td) is True
    for c in range(18):
        if c in GEOMETRY_SYNC_CHANNELS:
            assert torch.equal(train.x[:, c], ref.x[:, c]), f"channel {c} not synced"
        else:
            assert float(train.x[:, c].abs().max()) == 0.0, f"channel {c} was touched"


def test_geometry_sync_refuses_a_node_count_mismatch():
    """Silently using a mismatched mesh is the original bug; skipping is the safe outcome."""
    from torch_geometric.data import Data

    from src.utils.kinematics_paths import sync_geometry_from_deploy_pack

    import tempfile

    train = Data(x=torch.zeros(6, 18), edge_index=torch.tensor([[0, 1], [1, 0]]))
    train.graph_stem = "patient001"
    with tempfile.TemporaryDirectory() as td:
        torch.save(Data(x=torch.ones(9, 18)), Path(td) / "patient001.pt")
        assert sync_geometry_from_deploy_pack(train, deploy_dir=td) is False
    assert float(train.x.abs().max()) == 0.0


def _p2_pack():
    """A tiny triangle6-style pack: corners on a grid, one mid-side per edge."""
    from torch_geometric.data import Data

    from src.tests.test_rgp_deq_repair import _p2_strip  # self-import keeps the mesh in one place

    pos, ei, n, deg = _p2_strip(nx=6, ny=3)
    # Drop the corner-corner shortcut edges so the topology is genuinely P2.
    row, col = ei
    keep = (deg[row] == 2) | (deg[col] == 2)
    ei = ei[:, keep]
    d = Data(x=torch.zeros(n, 18, dtype=torch.float32), edge_index=ei)
    d.x[:, 0:2] = pos.float()
    d.x[:, 15] = 1.0
    d.mask_wall = torch.zeros(n, dtype=torch.bool)
    d.num_nodes = n
    return d, pos


def test_corner_graph_removes_every_degree_two_node():
    """T2: the whole point is that deployment stops containing a node type training lacks."""
    from src.data_gen.lib.p1_corner_graph import build_corner_graph

    d, _ = _p2_pack()
    g, cmap = build_corner_graph(d, recompute_width_derivs=False)

    n = int(g.num_nodes)
    deg = torch.zeros(n)
    deg.index_add_(0, g.edge_index[0], torch.ones(g.edge_index.shape[1]))

    # A degree-2 node is only pathological when its two edges are ANTI-PARALLEL -- that is what
    # makes the least-squares stencil rank-deficient.  A small test grid legitimately has a
    # couple of geometric corners whose two edges are perpendicular; those are fine, and on the
    # real packs the corner graph measures 0.00% degree-2 regardless.
    row, col = g.edge_index
    pos = g.x[:, 0:2]
    collinear = 0
    for i in (deg == 2).nonzero().reshape(-1).tolist():
        nb = col[row == i]
        v1, v2 = pos[nb[0]] - pos[i], pos[nb[1]] - pos[i]
        if torch.nn.functional.cosine_similarity(v1[None], v2[None]).item() < -0.9:
            collinear += 1
    assert collinear == 0, f"{collinear} collinear (rank-deficient) stencils survived the collapse"
    assert n + int(cmap.midside_ids.numel()) == int(d.num_nodes)
    assert n < int(d.num_nodes)
    # Edges must be symmetric and free of self-loops.
    r, c = g.edge_index
    assert int((r == c).sum()) == 0
    und = {(int(a), int(b)) for a, b in zip(r.tolist(), c.tolist())}
    assert all((b, a) in und for a, b in und), "corner edge set is not symmetric"


def test_lift_is_exact_for_a_linear_field():
    """Lifting a PREDICTION by averaging its two corners is 2nd-order exact at a midpoint."""
    from src.data_gen.lib.p1_corner_graph import build_corner_graph, lift_to_full_mesh

    d, pos = _p2_pack()
    g, cmap = build_corner_graph(d, recompute_width_derivs=False)
    f = 2.0 * pos[:, 0].float() - 3.0 * pos[:, 1].float()
    back = lift_to_full_mesh(f[cmap.corner_ids], cmap)
    assert float((back - f).abs().max()) < 1e-4


def test_midside_detection_survives_a_curved_boundary():
    """A strict midpoint test misclassifies every mid-side node on a curved wall (§7.2)."""
    from src.data_gen.lib.p1_corner_graph import identify_midside_nodes

    d, pos = _p2_pack()
    base, _ = identify_midside_nodes(d)
    n_base = int(base.sum())
    assert n_base > 0

    # Push every mid-side node off the chord, the way COMSOL places it on the true geometry.
    moved = d.clone()
    off = base.nonzero().reshape(-1)
    moved.x[off, 1] = moved.x[off, 1] + 0.01
    after, _ = identify_midside_nodes(moved)
    assert int(after.sum()) == n_base, (
        "curved-boundary mid-side nodes were misclassified as corners; the test must be "
        "anti-parallelism, not an exact-midpoint tolerance"
    )


# --- A1: the training corpus must carry the deployment mesh order ---------------------------

def test_p2_elevation_reproduces_the_comsol_edge_convention():
    """COMSOL emits corner-midside half-edges ONLY -- measured 100.00% on patient020, with
    0.00% corner-corner and 0.00% midside-midside.  Elevation that keeps the original P1 edges
    would produce a different degree distribution and re-open the gap it exists to close."""
    from src.data_gen.lib.p1_corner_graph import identify_midside_nodes
    from src.data_gen.lib.p2_elevation import elevate_to_p2, undirected_edges

    d, _ = _p2_pack()
    # Start from the P1 corner graph so the input is genuinely P1.
    from src.data_gen.lib.p1_corner_graph import build_corner_graph

    p1, _ = build_corner_graph(d, recompute_width_derivs=False)
    n_c = int(p1.num_nodes)
    n_e = int(undirected_edges(p1.edge_index).shape[0])

    up = elevate_to_p2(p1)
    assert int(up.num_nodes) == n_c + n_e

    mask, _ = identify_midside_nodes(up)
    row, col = up.edge_index
    cm = int((mask[row] ^ mask[col]).sum())
    cc = int(((~mask[row]) & (~mask[col])).sum())
    mm = int((mask[row] & mask[col]).sum())
    assert cc == 0, f"{cc} corner-corner edges; COMSOL has none"
    assert mm == 0, f"{mm} midside-midside edges; COMSOL has none"
    assert cm == row.shape[0], "not every edge is a corner-midside half-edge"
    assert int(mask.sum()) == n_e


def test_p2_elevation_leaves_the_original_nodes_untouched():
    """Elevation adds nodes.  It does not get to quietly rewrite the mesh it was handed."""
    from src.data_gen.lib.p1_corner_graph import build_corner_graph
    from src.data_gen.lib.p2_elevation import elevate_to_p2

    d, _ = _p2_pack()
    p1, _ = build_corner_graph(d, recompute_width_derivs=False)
    p1.y = torch.randn(int(p1.num_nodes), 5)
    n = int(p1.num_nodes)
    up = elevate_to_p2(p1)

    # width_d1/d2 (16,17) are re-derived on the new connectivity by design; everything else
    # about the original rows must survive bit-for-bit.
    assert torch.equal(up.x[:n, :16], p1.x[:n, :16])
    assert torch.equal(up.y[:n], p1.y[:n])
    assert torch.equal(up.mask_wall[:n], p1.mask_wall.reshape(-1))


def test_p2_elevation_midside_boundary_labels_require_both_parents():
    """A node bridging a wall corner and an interior corner is interior, not wall."""
    from src.data_gen.lib.p1_corner_graph import build_corner_graph
    from src.data_gen.lib.p2_elevation import elevate_to_p2, undirected_edges

    d, _ = _p2_pack()
    p1, _ = build_corner_graph(d, recompute_width_derivs=False)
    n = int(p1.num_nodes)
    wall = torch.zeros(n, dtype=torch.bool)
    wall[: max(2, n // 3)] = True
    p1.mask_wall = wall

    up = elevate_to_p2(p1)
    pairs = undirected_edges(p1.edge_index)
    expected = wall[pairs[:, 0]] & wall[pairs[:, 1]]
    assert torch.equal(up.mask_wall[n:], expected)
    assert int(up.mask_wall[n:].sum()) < int(expected.numel()), "test would be vacuous"


# --- A2: absolute coordinates are a memorisation shortcut ------------------------------------

def test_coord_mode_centering_is_exactly_translation_invariant():
    """§8 A2.  Translating a vessel changes nothing physical, so the coordinates the network
    sees must not move.  Measured on the shipped checkpoint, a full-span translation moved the
    prediction by 0.55 (patient020) under 'absolute' and 0.0019 under 'centered'."""
    import os

    from src.architecture.ginodeq import KINEMATICS_COORD_MODE_ENV, _canonical_coords

    pos = torch.randn(50, 2)
    shift = torch.tensor([[3.7, -1.2]])
    prev = os.environ.get(KINEMATICS_COORD_MODE_ENV)
    try:
        os.environ[KINEMATICS_COORD_MODE_ENV] = "absolute"
        assert torch.equal(_canonical_coords(pos), pos)
        assert not torch.allclose(_canonical_coords(pos + shift), _canonical_coords(pos))

        os.environ[KINEMATICS_COORD_MODE_ENV] = "centered"
        a = _canonical_coords(pos)
        b = _canonical_coords(pos + shift)
        assert torch.allclose(a, b, atol=1e-5), "centred coordinates are not translation-invariant"
        assert float(a.mean(dim=0).abs().max()) < 1e-6

        os.environ[KINEMATICS_COORD_MODE_ENV] = "nonsense"
        with pytest.raises(ValueError):
            _canonical_coords(pos)
    finally:
        if prev is None:
            os.environ.pop(KINEMATICS_COORD_MODE_ENV, None)
        else:
            os.environ[KINEMATICS_COORD_MODE_ENV] = prev


def test_siren_coordinates_go_through_the_same_canonicaliser():
    """Centring the encoder alone left patient041 WORSE (0.284 -> 0.715): the SIREN decoder is
    a coordinate network and was still being handed the absolute frame."""
    src = (REPO / "src" / "architecture" / "ginodeq.py").read_text(encoding="utf-8")
    i_siren = src.find("pos_nd = data.x[:, NodeFeat.XY]")
    assert i_siren > 0
    tail = src[i_siren : i_siren + 900]
    assert "_canonical_coords(pos_nd)" in tail, (
        "the SIREN decoder's coordinates bypass the canonicaliser; encoder-only centring does "
        "not make the model translation-invariant"
    )


# --- T6 / T7: the prior floor and selection on what the consumer reads ----------------------

def test_prior_floor_is_zero_when_the_model_beats_the_prior():
    """T6: a one-sided hinge, not shrinkage.  It must not fight a model that is already right."""
    from torch_geometric.data import Data

    from src.utils.kinematics_physics_terms import prior_floor_loss

    n = 32
    torch.manual_seed(0)
    truth = torch.randn(n, 2)
    d = Data(x=torch.zeros(n, 18), edge_index=torch.tensor([[0, 1], [1, 0]]))
    d.num_nodes = n
    d.y = torch.zeros(n, 5)
    d.y[:, 0:2] = truth
    d.x[:, 11:13] = truth + 0.5 * torch.randn(n, 2)     # a mediocre prior

    exact = torch.zeros(n, 6)
    exact[:, 0:2] = truth
    assert float(prior_floor_loss(exact, d)) == 0.0, "a perfect model must pay nothing"

    better = torch.zeros(n, 6)
    better[:, 0:2] = truth + 0.05 * torch.randn(n, 2)
    assert float(prior_floor_loss(better, d)) < 1e-3

    worse = torch.zeros(n, 6)
    worse[:, 0:2] = truth + 3.0 * torch.randn(n, 2)
    assert float(prior_floor_loss(worse, d)) > 0.1, "a model worse than its prior must pay"


def test_selection_score_ranks_dsrx_correlation_above_rel_l2():
    """T7: rel-L2 is a tie-break, never a driver -- the width fix halved it and made the clot
    model worse, and the current surrogate ties the prior on rel-L2 while halving dsrx corr."""
    from src.utils.kinematics_selection import selection_score

    good_struct_bad_l2 = selection_score(dsrx_corr=0.90, gate_jaccard=0.45, rel_l2=0.40)
    bad_struct_good_l2 = selection_score(dsrx_corr=0.30, gate_jaccard=0.20, rel_l2=0.10)
    assert good_struct_bad_l2 < bad_struct_good_l2, (
        "selection prefers the lower rel-L2 model despite far worse shear structure"
    )
    # Gate Jaccard must OUTRANK dsrx correlation: measured against the locked clot ensemble,
    # gate J tracks oracle-F1 at +0.918 while dsrx corr reads -0.073 within a single flow arm.
    high_gate_low_corr = selection_score(dsrx_corr=0.30, gate_jaccard=0.60, rel_l2=0.2)
    low_gate_high_corr = selection_score(dsrx_corr=0.95, gate_jaccard=0.20, rel_l2=0.2)
    assert high_gate_low_corr < low_gate_high_corr, (
        "selection still ranks dsrx correlation above gate agreement; that ordering was "
        "measured to be backwards"
    )
    # NaNs must never look like a good score.
    assert selection_score(float("nan"), float("nan"), float("nan")) > selection_score(0.9, 0.9, 0.1)


def test_selection_metrics_catch_an_under_scaled_field():
    """Correlation is scale-blind; the gate term is what catches amplitude.  Measured on
    patient020: a 0.4x field keeps dsrx_corr at 0.996 but gate Jaccard falls 0.948 -> 0.114."""
    pack = REPO / "data" / "processed" / "graphs_kinematics_anchors" / "carreau" / "patient020.pt"
    if not pack.is_file():
        pytest.skip("needs a local kinematics anchor pack")
    from src.utils.kinematics_selection import wall_shear_selection_metrics

    d = torch.load(pack, map_location="cpu", weights_only=False)
    gt = d.y[:, 0:2]
    full = wall_shear_selection_metrics(gt, d)
    small = wall_shear_selection_metrics(gt * 0.4, d)
    assert full["dsrx_corr"] > 0.9 and small["dsrx_corr"] > 0.9, "correlation should be scale-blind"
    assert small["gate_jaccard"] < 0.5 * full["gate_jaccard"], (
        "the gate metric failed to notice a 2.5x amplitude error"
    )


def test_promotion_gates_fail_closed_on_an_uncomputable_metric():
    """A threshold that cannot be evaluated must block promotion, never wave it through."""
    import os

    from src.training.train_kinematics_predictor import _kinematics_promotion_gates_pass

    base = dict(patient_rel=0.1, patient_n=3, synthetic_rel=0.1, synthetic_n=5,
                synthetic_l2_rel=0.1, synthetic_l2_n=5)
    prev = os.environ.get("KINEMATICS_MIN_DSRX_CORR")
    try:
        os.environ.pop("KINEMATICS_MIN_DSRX_CORR", None)
        assert _kinematics_promotion_gates_pass(**base)[0] is True, "unset must be historical"
        os.environ["KINEMATICS_MIN_DSRX_CORR"] = "0.6"
        assert _kinematics_promotion_gates_pass(**base, dsrx_corr=0.8)[0] is True
        assert _kinematics_promotion_gates_pass(**base, dsrx_corr=0.3)[0] is False
        assert _kinematics_promotion_gates_pass(**base, dsrx_corr=float("nan"))[0] is False
    finally:
        if prev is None:
            os.environ.pop("KINEMATICS_MIN_DSRX_CORR", None)
        else:
            os.environ["KINEMATICS_MIN_DSRX_CORR"] = prev


# --- B16: no single loss term may own the objective ------------------------------------------

def _smooth_field(v, edge_index, n, iters=40):
    row, col = edge_index
    deg = torch.zeros(n).index_add_(0, row, torch.ones(row.shape[1] if v.dim() > 2 else row.shape[0]))
    deg = deg.clamp(min=1.0)
    for _ in range(iters):
        v = 0.5 * v + 0.5 * torch.zeros_like(v).index_add_(0, row, v[col]) / deg.unsqueeze(1)
    return v


@pytest.mark.skipif(not _PACK.is_file(), reason="needs a local kinematics anchor pack")
def test_no_single_loss_term_owns_the_objective():
    """B16.  `l_shear_grad` is an ABSOLUTE MSE on d(shear)/dx -- units (1/s)/m, so raw values
    ~1e4 where every other term is O(1).  At its weight of 50 it was **99.99% of the total
    loss** even for a spatially smooth 15% error, leaving the supervised data term (weight 500)
    at 0.00%.  That is the mechanism behind the surrogate being worse than its own input prior
    on 45 of 52 packs: it was never meaningfully trained to match velocity at all.
    """
    import os

    from src.config import PhysicsConfig
    from src.core_physics.physics_kernels import PhysicsKernels
    from src.utils.kinematics_physics_terms import compute_kinematics_physics_terms

    d = torch.load(_PACK, map_location="cpu", weights_only=False)
    n = int(d.num_nodes)
    kern = PhysicsKernels(phys_cfg=PhysicsConfig(phase="kinematics"))

    torch.manual_seed(0)
    noise = _smooth_field(torch.randn(n, 2), d.edge_index, n)
    noise = noise / noise.norm() * d.y[:, 0:2].norm() * 0.15
    pred = torch.cat([d.y[:, :5], torch.zeros(n, 1)], dim=1)
    pred[:, 0:2] = pred[:, 0:2] + noise
    pred[:, 4] = pred[:, 4] * 1.1

    weights = {"l_data_kine": 500.0, "l_data_mu": 10.0, "l_wss": 10.0, "l_shear_grad": 50.0,
               "l_cont": 50.0, "l_mom": 1.0, "l_bc": 5.0, "l_io": 5.0}

    def share(flag):
        prev = os.environ.get("KINEMATICS_NORMALIZE_SHEAR_GRAD")
        os.environ["KINEMATICS_NORMALIZE_SHEAR_GRAD"] = flag
        try:
            t = compute_kinematics_physics_terms(pred, d, kern)
        finally:
            if prev is None:
                os.environ.pop("KINEMATICS_NORMALIZE_SHEAR_GRAD", None)
            else:
                os.environ["KINEMATICS_NORMALIZE_SHEAR_GRAD"] = prev
        c = {k: float(t[k]) * w for k, w in weights.items() if k in t}
        tot = sum(c.values())
        return {k: v / tot for k, v in c.items()}

    unnormalised = share("0")
    assert unnormalised["l_shear_grad"] > 0.9, (
        "premise changed: l_shear_grad no longer dominates the unnormalised objective"
    )

    assert unnormalised["l_data_kine"] < 1e-3, (
        "premise changed: the supervised data term already contributes under the raw objective"
    )

    normalised = share("1")
    # The substantive property: the SUPERVISED term must actually reach the optimiser.  How
    # many other terms clear 1% is graph-dependent (measured 5 of 11 on an elevated synthetic
    # graph, 3 of 8 on patient020), so asserting a count here would be fitting the test to one
    # mesh -- assert the invariant instead.
    assert normalised["l_data_kine"] > 0.01, (
        f"data term is still inert at {100 * normalised['l_data_kine']:.3f}% of the objective"
    )
    assert normalised["l_shear_grad"] < 0.5, (
        f"l_shear_grad still dominates at {100 * normalised['l_shear_grad']:.1f}%"
    )


# --- gate-agreement loss: optimise the metric that actually predicts the outcome -------------

@pytest.mark.skipif(not _PACK.is_file(), reason="needs a local kinematics anchor pack")
def test_gate_loss_moves_opposite_to_the_gate_metric():
    """Gate union Jaccard is the only Stage-A metric measured to predict the clot model's
    oracle-F1 (+0.918), and nothing in the objective optimised it.  The soft-gate term must be
    monotone against the hard metric, or it is optimising something else."""
    from src.config import PhysicsConfig
    from src.core_physics.physics_kernels import PhysicsKernels
    from src.utils.kinematics_physics_terms import compute_kinematics_physics_terms
    from src.utils.kinematics_selection import wall_shear_selection_metrics

    d = torch.load(_PACK, map_location="cpu", weights_only=False)
    n = int(d.num_nodes)
    kern = PhysicsKernels(phys_cfg=PhysicsConfig(phase="kinematics"))

    exact = torch.cat([d.y[:, :5], torch.zeros(n, 1)], dim=1)
    scaled = exact.clone()
    scaled[:, 0:2] = scaled[:, 0:2] * 0.4

    losses, jaccs = [], []
    for pred in (exact, scaled):
        losses.append(float(compute_kinematics_physics_terms(pred, d, kern)["l_band_gate"]))
        jaccs.append(wall_shear_selection_metrics(pred[:, 0:2], d)["gate_jaccard"])

    assert jaccs[0] > jaccs[1], "test premise: the scaled field must score worse on the metric"
    assert losses[0] < losses[1], (
        f"gate loss did not follow the metric: loss {losses} against Jaccard {jaccs}"
    )


def test_gate_loss_is_differentiable():
    """A term the optimiser cannot move is decoration."""
    if not _PACK.is_file():
        pytest.skip("needs a local kinematics anchor pack")
    from src.config import PhysicsConfig
    from src.core_physics.physics_kernels import PhysicsKernels
    from src.utils.kinematics_physics_terms import compute_kinematics_physics_terms

    d = torch.load(_PACK, map_location="cpu", weights_only=False)
    n = int(d.num_nodes)
    kern = PhysicsKernels(phys_cfg=PhysicsConfig(phase="kinematics"))
    pred = torch.cat([d.y[:, :5], torch.zeros(n, 1)], dim=1)
    pred[:, 0:2] = pred[:, 0:2] * 0.7
    pred = pred.detach().requires_grad_(True)
    compute_kinematics_physics_terms(pred, d, kern)["l_band_gate"].backward()
    assert pred.grad is not None and float(pred.grad.norm()) > 0
    assert torch.isfinite(pred.grad).all()


def test_sealed_holdout_default_does_not_depend_on_the_launcher():
    """B21.  The holdout default lived in `finetune_kine_patient_anchors.py`, so invoking
    `train_kinematics_predictor` directly fell back to the single stem "patient007" and trained
    on the other three FINAL_HALF vessels.  The default must live at the point of use."""
    import os

    from src.core_physics.wall_cohort_splits import DEV, SEALED

    src = (REPO / "src" / "utils" / "kinematics_geometry.py").read_text(encoding="utf-8")
    assert "wall_cohort_splits" in src, "the split default is not derived from the canonical sets"

    prev = os.environ.pop("KINEMATICS_VAL_HOLDOUT_PATIENT_STEMS", None)
    try:
        from src.utils.kinematics_geometry import split_clinical_anchor_train_val

        class _G:
            def __init__(self, stem, clinical=True):
                self.graph_stem = stem
                self.is_clinical_anchor = clinical

        stems = sorted(set(SEALED) | set(DEV) | {"patient002", "patient005"})
        out = split_clinical_anchor_train_val([_G(s) for s in stems])
        train = {g.graph_stem for g in out["train"]}
        assert not (set(SEALED) & train), f"SEALED vessels in TRAIN: {sorted(set(SEALED) & train)}"
        assert not (set(DEV) & train), f"DEV vessels in TRAIN: {sorted(set(DEV) & train)}"
        assert "patient002" in train, "the split held out everything; test would be vacuous"
    finally:
        if prev is not None:
            os.environ["KINEMATICS_VAL_HOLDOUT_PATIENT_STEMS"] = prev


# --- s12: loss weights by measured gradient share --------------------------------------------

def test_default_relative_weights_reproduce_the_shipped_recipe():
    """Unset must be bit-identical to the historical numbers, or the refactor is a silent
    behaviour change dressed up as a cleanup."""
    import os

    from src.training.train_kinematics_predictor import _resolve_loss_weights

    prev = os.environ.pop("KINEMATICS_LOSS_WEIGHTS", None)
    try:
        rel = _resolve_loss_weights()
    finally:
        if prev is not None:
            os.environ["KINEMATICS_LOSS_WEIGHTS"] = prev
    s = 500.0
    assert rel["l_cont"] * s == pytest.approx(50.0)
    assert rel["l_mom"] * s == pytest.approx(1.0)
    assert rel["l_bc"] * s == pytest.approx(5.0)
    assert rel["l_io"] * s == pytest.approx(5.0)
    assert rel["l_wss"] * s == pytest.approx(10.0)
    assert rel["l_shear_grad"] * s == pytest.approx(50.0)
    # The new terms default OFF.
    for k in ("l_band_sr", "l_band_dsrx", "l_band_gate", "l_prior_floor"):
        assert rel[k] == 0.0


def test_calibration_drops_an_inert_term_instead_of_amplifying_it(tmp_path):
    """`l_bc` measures 3.6e-09 because the hard BC satisfies the boundary condition by
    construction.  Solving `w = share / g` for it asks for ~1e+07, which would make numerical
    noise the dominant gradient."""
    from src.utils.loss_calibration import weights_from_gradient_norms

    norms = {"l_data_kine": 0.58, "l_cont": 13.6, "l_bc": 3.6e-9, "l_band_gate": 84.7}
    w = weights_from_gradient_norms(norms, {"l_data_kine": 0.5, "l_cont": 0.2,
                                            "l_bc": 0.2, "l_band_gate": 0.1})
    assert "l_bc" not in w, "an inert term was given a weight"
    assert w["l_data_kine"] == pytest.approx(1.0)
    # A term with a LARGER gradient must receive a SMALLER weight for the same share.
    assert w["l_band_gate"] < w["l_cont"] < w["l_data_kine"]


def test_calibration_cutoff_is_not_anchored_to_the_largest_term():
    """Anchoring "inert" to the max lets one heavy-tailed term drop every other loss -- observed,
    and it produced a recipe containing exactly one term."""
    from src.utils.loss_calibration import weights_from_gradient_norms

    norms = {"l_data_kine": 0.58, "l_cont": 13.6, "l_shear_grad": 3.0e9}
    w = weights_from_gradient_norms(norms, {"l_data_kine": 0.5, "l_cont": 0.3,
                                            "l_shear_grad": 0.2})
    assert set(w) == {"l_data_kine", "l_cont", "l_shear_grad"}, (
        f"a heavy-tailed term suppressed the others: kept {sorted(w)}"
    )


def test_calibration_uses_a_median_not_a_mean_across_graphs():
    """One vessel measured `l_shear_grad` at 3.0e+09 against 7.0e+03 elsewhere; a mean would let
    that single graph set every weight in the recipe."""
    src = (REPO / "src" / "utils" / "loss_calibration.py").read_text(encoding="utf-8")
    assert "statistics.median" in src, "gradient norms are aggregated with a non-robust statistic"


# --- pilot workflow: transfer, preflight, training debug -------------------------------------

def test_slimming_only_drops_tensors_nothing_reads_by_default():
    """`G_x`/`G_y` are 98.4% of a pack and are read only under BIOCHEM_GRAD_OPERATOR=legacy.
    `V`/`W`/`M_inv` are rebuildable but dropping them CHANGES numerics on packs whose stored
    operator does not match their graph (B13), so that must stay opt-in."""
    src = (REPO / "scripts" / "slim_kine_packs.py").read_text(encoding="utf-8")
    assert 'drop = REBUILDABLE if args.drop_wls else ("G_x", "G_y")' in src, (
        "slimming drops the WLS operators by default; that is a numerics change in a copy tool"
    )
    assert "legacy" in src, "the tool does not warn about BIOCHEM_GRAD_OPERATOR=legacy"


def test_preflight_fails_a_cohort_with_a_leaked_prior_block():
    """The preflight's whole job is to be the last gate before GPU time."""
    src = (REPO / "scripts" / "preflight_kine_cohort.py").read_text(encoding="utf-8")
    for needle in ("prior block is NOT the CFD solution", "node_type populated",
                   "wall_normal populated", "width_d2 within training range",
                   "severe-stenosis coverage"):
        assert needle in src, f"preflight does not check: {needle}"
    assert "return 1 if n_fail else 0" in src, "preflight does not fail the exit code"


def test_selection_metric_subset_is_capped_and_deterministic():
    """Each selection graph costs a full Anderson solve, and validation runs every other epoch.
    A cap keeps that affordable; sorting keeps it comparable across epochs."""
    src = (REPO / "src" / "training" / "train_kinematics_predictor.py").read_text(encoding="utf-8")
    i = src.find("KINEMATICS_SELECT_MAX_GRAPHS")
    assert i > 0, "selection metrics run over the full holdout every validation"
    assert "sorted(subset" in src[i : i + 400], "the capped subset is not deterministic"


def test_early_abort_is_scored_on_the_selection_metric_not_rel_l2():
    """rel-L2 is a tie-break (s10.3); aborting on it would stop good runs and continue bad ones."""
    src = (REPO / "src" / "training" / "train_kinematics_predictor.py").read_text(encoding="utf-8")
    i = src.find("KINEMATICS_SELECT_PATIENCE")
    assert i > 0, "no early-abort path"
    window = src[max(0, i - 900) : i + 300]
    assert "selection_score" in window, "early abort is not scored on the selection metric"


# --- generation pipeline: guard, mix, and the channels it writes ------------------------------

def test_generation_refuses_a_populated_cohort_without_declared_intent():
    """A 12-vessel smoke test replaced a 370-graph corpus and its meshes.  `data/` is gitignored,
    so there was nothing to restore from.  Intent must be explicit."""
    import argparse

    from src.data_gen.pipeline_kinematics import _assert_write_intent_declared

    args = argparse.Namespace(overwrite=False, append=False)
    try:
        _assert_write_intent_declared(args, "carreau")
    except SystemExit as exc:
        assert "REFUSING TO GENERATE" in str(exc)
        assert "--overwrite" in str(exc) and "--append" in str(exc)
    else:
        graph_dir = REPO / "data" / "processed" / "graphs_kinematics" / "carreau"
        mesh_dir = REPO / "data" / "raw" / "kinematics" / "meshes"
        populated = (graph_dir.is_dir() and any(graph_dir.glob("*.pt"))) or (
            mesh_dir.is_dir() and any(mesh_dir.glob("vessel_*.msh")))
        assert not populated, "guard permitted generation into a populated cohort"

    for flag in ("overwrite", "append"):
        ok = argparse.Namespace(overwrite=False, append=False)
        setattr(ok, flag, True)
        _assert_write_intent_declared(ok, "carreau")   # must not raise


def test_pathology_mix_expands_per_vessel_and_covers_the_tail():
    """One command instead of one run per mode.  Random sampling alone under-represents the
    severe-stenosis regime that deployment actually fails in."""
    import numpy as np

    from src.data_gen.lib.vessel_generator import parse_pathology_mix

    modes = parse_pathology_mix("random:0.72,max_stenosis:0.18,max_aneurysm:0.10", 250,
                                np.random.default_rng(0))
    assert len(modes) == 250
    from collections import Counter

    c = Counter(modes)
    assert c["max_stenosis"] == 45 and c["max_aneurysm"] == 25 and c["random"] == 180

    # exact counts summing to n are honoured verbatim
    c2 = Counter(parse_pathology_mix("random:18,max_stenosis:4,max_aneurysm:2", 24,
                                     np.random.default_rng(0)))
    assert c2["random"] == 18 and c2["max_stenosis"] == 4 and c2["max_aneurysm"] == 2

    # a single mode still works
    assert set(parse_pathology_mix("random", 10, np.random.default_rng(0))) == {"random"}


def test_builder_no_longer_writes_a_node_type_placeholder():
    """B22.  `mesh_to_graph` had a literal `torch.zeros((len(nodes), 4))  # Node Type
    (Placeholder)`.  Regenerating the corpus would NOT have fixed the dead channel -- this line
    is why.  It is live at deploy and was identically zero across all training data."""
    src = (REPO / "src" / "data_gen" / "lib" / "mesh_to_graph.py").read_text(encoding="utf-8")
    assert "Node Type (Placeholder)" not in src, "the node_type placeholder is back"
    assert "node_type_one_hot(" in src, "the builder does not populate node_type"


def test_builder_does_not_store_the_dead_sparse_operators_by_default():
    """`G_x`/`G_y` are 98.4% of a pack and are read only under BIOCHEM_GRAD_OPERATOR=legacy.
    At 250 vessels that is a 33 GB transfer instead of 500 MB."""
    src = (REPO / "src" / "data_gen" / "lib" / "mesh_to_graph.py").read_text(encoding="utf-8")
    assert "KINEMATICS_STORE_G_OPERATORS" in src, "storing G_x/G_y is not opt-in"
    i = src.find("data.G_x, data.G_y = G_x, G_y")
    assert i > 0, "the opt-in path is missing"
    assert "KINEMATICS_STORE_G_OPERATORS" in src[max(0, i - 300):i]


def test_sampled_params_record_a_resolved_mode_so_retries_can_reuse_it():
    """B26.  The retry path re-samples rejected geometries and used to pass the caller's
    `pathology_mode` verbatim.  With `--pathology-mix` that argument is a spec
    ("random:0.72,max_stenosis:0.18,...") which `_sample_params` cannot parse -- it expects a
    single resolved mode, because the main loop expands the spec per vessel before calling it.

    So a 250-vessel run crashed the moment any geometry was rejected and had to be resampled
    (3 of 250: "outlet curled back past L/3") -- after the meshes were already generated.
    """
    import numpy as np

    from src.config import VesselConfig
    from src.data_gen.lib.vessel_generator import (
        _sample_params, normalize_pathology_mode, parse_pathology_mix,
    )

    cfg = VesselConfig(phase="kinematics")
    rng = np.random.default_rng(3)
    modes = parse_pathology_mix("random:0.5,max_stenosis:0.3,max_aneurysm:0.2", 10, rng)

    for i, mode in enumerate(modes):
        params = _sample_params(i, i % 3, cfg, rng, pathology_mode=mode)
        stored = params.get("pathology_mode")
        assert stored is not None, "params do not record their mode; a retry cannot reuse it"
        # The retry feeds this straight back in, so it must be a single resolved mode.
        assert normalize_pathology_mode(stored) is not None or stored == "random"
        _sample_params(i, i % 3, cfg, rng, pathology_mode=stored)   # must not raise


def test_retry_path_reuses_the_stored_mode_not_the_raw_argument():
    src = (REPO / "src" / "data_gen" / "lib" / "vessel_generator.py").read_text(encoding="utf-8")
    i = src.find("# ---- Retry failed samples ----")
    assert i > 0
    block = src[i : i + 2000]
    assert 'pathology_mode=failed_p.get("pathology_mode")' in block, (
        "the retry path passes the caller's pathology_mode, which is a mix SPEC under "
        "--pathology-mix and will raise on the first rejected geometry"
    )
    assert "pathology_mode=pathology_mode," not in block
