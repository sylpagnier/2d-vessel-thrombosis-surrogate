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
                   "wall_normal populated", "width_d2 operator is sane",
                   "clamp bounds match this cohort", "inlet BC present",
                   "geometry_level present", "severe-stenosis coverage"):
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


# --- B27: the width clamp is a corpus property, defined once -----------------------------

def test_width_clamp_has_exactly_one_definition():
    """The bounds were hardcoded in TWO places -- `kinematics_inference` (deploy-time clamp) and
    `ginodeq._apply_fourier_encoding` (the encoder clamping its own inputs).  Both carried
    4.14 / 73.8, derived from a 40-vessel corpus containing no severe stenosis.  On the
    250-vessel cohort those bounds clamp **44% / 34%** of vessels, which would have silently
    truncated exactly the sharp-throat signal the cohort was generated to provide.
    """
    from src.architecture.ginodeq import WIDTH_D1_MAX as ENC_D1, WIDTH_D2_MAX as ENC_D2
    from src.config import WIDTH_D1_MAX, WIDTH_D2_MAX
    from src.utils.kinematics_inference import WIDTH_D1_MAX as INF_D1, WIDTH_D2_MAX as INF_D2

    assert (WIDTH_D1_MAX, WIDTH_D2_MAX) == (ENC_D1, ENC_D2) == (INF_D1, INF_D2)

    enc = (REPO / "src" / "architecture" / "ginodeq.py").read_text(encoding="utf-8")
    assert "-4.14, 4.14" not in enc and "-73.8, 73.8" not in enc, (
        "the encoder still hardcodes the old corpus bounds"
    )


def test_checkpoints_record_the_clamp_they_were_trained_under(tmp_path):
    """The clamp is a property of the checkpoint's training corpus, so it has to travel with the
    weights -- otherwise changing the constant silently re-interprets every older checkpoint."""
    import torch.nn as nn

    from src.architecture.kinematics_model_config import save_kinematics_checkpoint_file
    from src.config import WIDTH_D1_MAX, WIDTH_D2_MAX

    f = tmp_path / "c.pth"
    save_kinematics_checkpoint_file(f, nn.Linear(2, 2), checkpoint_role="kinematics_best",
                                    rel_l2=0.1, continuity=0.001, composite=0.2,
                                    run_id="X", prior_source="analytic")
    raw = torch.load(f, map_location="cpu", weights_only=False)
    assert raw["width_clamp"] == [float(WIDTH_D1_MAX), float(WIDTH_D2_MAX)]


def test_preflight_reports_the_clamp_bounds_the_cohort_implies():
    """A stale clamp is invisible unless something computes the cohort's own p95 and compares."""
    src = (REPO / "scripts" / "preflight_kine_cohort.py").read_text(encoding="utf-8")
    assert "clamp bounds match this cohort" in src
    assert "WIDTH_D1_MAX" in src and "percentile(d1a, 95)" in src


# --- B26: the PDE label floor --------------------------------------------------------------

def test_pde_floor_zeroes_the_terms_on_the_labels_themselves():
    """B26: `l_cont` / `l_mom` must cost nothing when the model reproduces COMSOL exactly.

    Un-floored they do not: on the 250-vessel cohort the labels' own discrete continuity
    residual reaches 0.22 -- 22 at the training weight of 100 -- concentrated in the first ring
    off the wall on the severe-stenosis vessels the cohort exists to teach.  The hinge makes the
    labels the floor rather than a target to beat.
    """
    import numpy as np
    from torch_geometric.data import Data

    from src.config import PhysicsConfig
    from src.core_physics.physics_kernels import PhysicsKernels
    from src.utils.kinematics_physics_terms import (
        PDE_FLOOR_CONT, PDE_FLOOR_MOM, attach_pde_floors, compute_kinematics_physics_terms)

    # structured grid -> real WLS operators, so the labels carry genuine stencil residual
    gx, gy = torch.meshgrid(torch.linspace(0, 0.01, 14), torch.linspace(-1e-3, 1e-3, 9),
                            indexing="ij")
    pos = torch.stack([gx.flatten(), gy.flatten()], dim=1)
    n = pos.shape[0]
    dist = torch.cdist(pos, pos)
    ei = (dist < 1.2e-3).nonzero(as_tuple=False).t()
    ei = ei[:, ei[0] != ei[1]]

    d = Data(x=torch.zeros(n, 18), edge_index=ei)
    d.num_nodes = n
    d.x[:, 0:2] = pos
    d.mask_wall = (pos[:, 1].abs() > 9.5e-4)
    d.mask_inlet = pos[:, 0] < 1e-4
    d.mask_outlet = pos[:, 0] > 9.9e-3
    d.mask_wound = torch.zeros(n, dtype=torch.bool)
    d.u_ref = torch.tensor([0.1])
    d.d_bar = torch.tensor([1.5e-3])
    d.is_anchor = torch.tensor([True])
    d.u_inlet_bc = torch.full((n, 1), 0.1)
    d.mu_inlet_bc = torch.ones(n, 1)
    d.mu_wall_bc = torch.ones(n, 1)

    row, col = d.edge_index
    dr = pos[col] - pos[row]
    dx, dy = dr[:, 0], dr[:, 1]
    d.V = torch.stack([dx, dy, 0.5 * dx**2, dx * dy, 0.5 * dy**2], dim=1)
    d.W = torch.ones(d.edge_index.size(1))
    d.M_inv = torch.eye(5).unsqueeze(0).repeat(n, 1, 1)

    # a Poiseuille-ish label field, non-trivial in every channel the momentum term reads
    yn = pos[:, 1] / 1e-3
    d.y = torch.zeros(n, 5)
    d.y[:, 0] = 1.0 - yn**2
    d.y[:, 1] = 0.02 * yn * (pos[:, 0] / 1e-2)
    d.y[:, 2] = -pos[:, 0] / 1e-3
    d.y[:, 3] = 1.0
    d.y_valid_mask = torch.ones(n, 5, dtype=torch.bool)

    kern = PhysicsKernels(PhysicsConfig(phase="kinematics"))
    raw = compute_kinematics_physics_terms(d.y.clone(), d, kern, phase="kinematics")
    assert attach_pde_floors(d, kern), "labels are non-trivial, a floor must be attachable"
    assert getattr(d, PDE_FLOOR_CONT).shape == (n,)
    assert getattr(d, PDE_FLOOR_MOM).shape == (n,)

    hinged = compute_kinematics_physics_terms(d.y.clone(), d, kern, phase="kinematics")
    assert float(raw["l_cont"]) > 0.0, "the labels DO carry stencil residual (else no test)"
    assert float(hinged["l_cont"]) == 0.0, "COMSOL's own answer must cost nothing"
    assert float(hinged["l_mom"]) == 0.0, "COMSOL's own answer must cost nothing"

    # ... and the hinge must still bite on a field that is worse than the labels
    g = torch.Generator().manual_seed(0)
    worse = d.y.clone()
    worse[:, 0:2] += 0.1 * torch.randn(n, 2, generator=g)
    t_worse = compute_kinematics_physics_terms(worse, d, kern, phase="kinematics")
    assert float(t_worse["l_cont"]) > 0.0, "a model worse than the labels must pay"
    assert float(t_worse["l_mom"]) > 0.0, "a model worse than the labels must pay"


def test_pde_floor_is_not_attached_to_an_unsolved_vessel():
    """B27: 39/250 packs of the 2026-08-28 cohort have an all-zero `y` -- COMSOL never solved
    them.  Granting those a floor would grant a floor of exactly zero everywhere, which reads as
    "the labels are perfect" when there are no labels.  They must stay un-floored."""
    from torch_geometric.data import Data

    from src.config import PhysicsConfig
    from src.core_physics.physics_kernels import PhysicsKernels
    from src.utils.kinematics_physics_terms import compute_pde_floors

    n = 16
    d = Data(x=torch.zeros(n, 18), edge_index=torch.tensor([[0, 1], [1, 0]]))
    d.num_nodes = n
    d.y = torch.zeros(n, 5)          # the unsolved-vessel placeholder
    d.is_anchor = torch.tensor([False])
    assert compute_pde_floors(d, PhysicsKernels(PhysicsConfig(phase="kinematics"))) is None


# --- B28: the Carreau shear rate is already non-dimensional --------------------------------

def test_carreau_target_uses_the_nd_shear_rate_directly():
    """B28: `_compute_carreau_viscosity` must NOT rescale the WLS shear rate.

    The WLS operators are built on `x_nd = x / d_bar` and the velocities are `u / u_ref`, so
    `d(u_nd)/d(x_nd)` is already non-dimensional -- the contract `graph_gradient_operators`
    documents and the one `BiochemPhysicsKernels._compute_shear_rate` and
    `clot_kinematics_fields` rely on when they multiply by `u_ref / d_bar` to reach SI 1/s.
    This function used to scale by `d_bar / u_ref` instead, ~7.7x too small on the 250-vessel
    cohort, which pushed the Carreau law toward its `mu_0` plateau: the target came out ~2x
    COMSOL's own `mu` on every vessel (`l_rheo` at `pred = y` had median 6.107).
    """
    from torch_geometric.data import Data

    from src.config import PhysicsConfig
    from src.core_physics.physics_kernels import PhysicsKernels
    from src.utils.rheology import carreau_yasuda_viscosity

    cfg = PhysicsConfig(phase="kinematics")
    kern = PhysicsKernels(cfg)

    n = 8
    d = Data(x=torch.zeros(n, 18), edge_index=torch.tensor([[0, 1], [1, 0]]))
    d.num_nodes = n
    d.u_ref = torch.tensor([0.105])
    d.d_bar = torch.tensor([0.0133])

    # a pure shear `du_nd/dy_nd = g`, so the invariant is exactly `g`
    g = torch.linspace(0.5, 40.0, n)
    du_ij = torch.zeros(n, 4)
    du_ij[:, 1] = g

    got = kern._compute_carreau_viscosity(du_ij, d)
    lambda_nd = cfg.lam * (float(d.u_ref) / float(d.d_bar))
    want = carreau_yasuda_viscosity(
        gamma_dot_nd=g, mu_inf_nd=torch.tensor(kern.mu_inf_nd),
        mu_0_nd=torch.tensor(kern.mu_0_nd), lambda_nd=torch.tensor(lambda_nd),
        n=cfg.n, a=cfg.a,
    )
    assert torch.allclose(got, want, rtol=2e-3), (
        "the Carreau argument must be `lam * gamma_SI` = `lambda_nd * gamma_nd`, with the "
        "WLS shear rate used as-is"
    )

    # and the regression itself: the old `d_bar / u_ref` rescale sits far up the mu_0 plateau
    stale = carreau_yasuda_viscosity(
        gamma_dot_nd=g * (float(d.d_bar) / float(d.u_ref)), mu_inf_nd=torch.tensor(kern.mu_inf_nd),
        mu_0_nd=torch.tensor(kern.mu_0_nd), lambda_nd=torch.tensor(lambda_nd),
        n=cfg.n, a=cfg.a,
    )
    assert float(stale.median() / got.median()) > 1.4, (
        "guard is only meaningful if the two conventions actually differ here"
    )


# --- B27: the throat has to be resolved, and the repair path has to be exact ----------------

def _straight_stenosed_walls(n=60, length=0.1, width=0.018, throat_frac=0.20):
    """Two wall polylines with a Gaussian throat, in SI metres."""
    import numpy as np

    x = np.linspace(0.0, length, n)
    bump = 1.0 - (1.0 - throat_frac) * np.exp(-0.5 * ((x - 0.5 * length) / (0.06 * length)) ** 2)
    half = 0.5 * width * bump
    top = np.stack([x, half], axis=1)
    bot = np.stack([x, -half], axis=1)
    return top, bot


def test_mesh_sizing_resolves_the_stenosis_throat(tmp_path):
    """B27: a uniform element size spans a severe throat instead of resolving it.

    The 2026-08-28 cohort meshed at a uniform 1 mm (x0.75), which puts about five elements
    across a 3.7 mm throat.  COMSOL then fails to converge: all 39 unsolved vessels were
    stenosis geometries and the failure rate climbed monotonically with stenosis ratio (2.9%
    below 1.5, 40.6% above 3.0).

    The guard is that the throat gets the requested resolution AND that the open lumen does not
    pay for it -- a global refine would work too but multiplies the node count everywhere.
    """
    import numpy as np
    import pytest

    gmsh = pytest.importorskip("gmsh")

    from src.config import VesselConfig
    from src.data_gen.lib.vessel_generator import _mesh_geometry
    from src.data_gen.lib.vessel_geometry import compute_geometry_from_walls

    cfg = VesselConfig(phase="kinematics")
    cfg_dict = {
        "mesh_lc": cfg.mesh_lc,
        "mesh_size_factor": cfg.mesh_size_factor,
        "mesh_min_elems_across": cfg.mesh_min_elems_across,
        "mesh_lc_min_ratio": cfg.mesh_lc_min_ratio,
        "unit": "m",
        "TAGS": cfg.TAGS,
        "base_length": cfg.base_length,
    }

    def mesh(top, bot, idx):
        geom = compute_geometry_from_walls(top, bot, idx=idx, unit="m")
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", 1)
            gmsh.option.setNumber("Mesh.SaveAll", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFactor", cfg_dict["mesh_size_factor"])
            lc_min = cfg.mesh_lc * cfg.mesh_lc_min_ratio
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cfg.mesh_lc)
            _, ok, err = _mesh_geometry(geom, cfg_dict, str(tmp_path))
            assert ok, err
        finally:
            gmsh.finalize()
        lines = (tmp_path / f"vessel_{idx}.msh").read_text().splitlines()
        i = lines.index("$Nodes")
        n = int(lines[i + 1])
        return np.array([[float(v) for v in lines[i + 2 + k].split()[1:3]] for k in range(n)])

    top_s, bot_s = _straight_stenosed_walls(throat_frac=0.20)
    pts_s = mesh(top_s, bot_s, 9001)
    top_o, bot_o = _straight_stenosed_walls(throat_frac=1.0)   # same vessel, no throat
    pts_o = mesh(top_o, bot_o, 9002)

    from scipy.spatial import cKDTree

    def spacing_near(pts, x0, r):
        near = pts[np.abs(pts[:, 0] - x0) < r]
        assert len(near) > 8
        dd, _ = cKDTree(near).query(near, k=2)
        return float(np.median(dd[:, 1]))

    throat_w = 0.018 * 0.20
    h_throat = spacing_near(pts_s, 0.05, throat_w)
    across = throat_w / h_throat
    assert across >= float(cfg.mesh_min_elems_across), (
        f"throat resolved by only {across:.1f} elements, asked for "
        f"{cfg.mesh_min_elems_across} -- the size callback is not taking effect.  Setting `lc` "
        f"on `addPoint` does NOT work here: the wall stations are B-spline control points, not "
        f"model vertices, so `Mesh.MeshSizeFromPoints` ignores them and the node count comes "
        f"back byte-identical to a uniform mesh."
    )

    # The inlet, far from the throat, must keep the OPEN-LUMEN size: refinement is local.
    # Asserted against the configured size rather than as a ratio to the throat, so it does not
    # drift when `mesh_lc` changes -- lowering `mesh_lc` to match deployment shrinks the open
    # lumen and would silently erode a ratio threshold.
    h_inlet = spacing_near(pts_s, 0.01, 0.008)
    h_open = cfg.mesh_lc * cfg.mesh_size_factor
    assert h_inlet > 0.8 * h_open, (
        f"inlet spacing {h_inlet:.2e} vs the configured open-lumen size {h_open:.2e} -- the "
        f"whole vessel refined, which multiplies node count everywhere instead of where it is "
        f"needed"
    )
    assert h_inlet > 1.5 * h_throat, (
        f"inlet {h_inlet:.2e} vs throat {h_throat:.2e} -- no meaningful local contrast"
    )
    # and the cost stays modest against the same vessel with no throat at all
    assert len(pts_s) < 2.0 * len(pts_o), (
        f"{len(pts_s)} nodes with a throat vs {len(pts_o)} without -- too expensive"
    )


# --- B27 repair: the re-draw ladder has to actually get easier -------------------------------

def test_severity_scale_softens_a_max_stenosis_draw():
    """The repair ladder was inert, and this is the mechanism that fixes it.

    `max_stenosis` pins the sampler at the class maximum, so a "re-draw of the same class" came
    back at the same severity every time -- the 0.70x / 0.50x / 0.35x rungs all measured 5.00.
    Which is why 38 substitutions left 36 vessels unsolved: an equally extreme vessel fails for
    the same reason the original did.

    `severity_scale` must soften BOTH halves: the wall offset AND the max-magnitude shape
    presets.  A max_stenosis draw takes a sharper transition (`std_dev` 0.02-0.05n against
    0.04-0.10n), and sharpness is a large part of what breaks the solve.
    """
    import numpy as np

    from src.config import VesselConfig
    from src.data_gen.lib.vessel_generator import _sample_params, _wall_severity
    from src.data_gen.lib.vessel_geometry import compute_geometry_from_params

    from src.data_gen.lib.vessel_generator import VesselGenerator

    cfg = VesselConfig(phase="kinematics")
    cfg_dict = VesselGenerator(phase="kinematics")._cfg_dict()

    def sev(scale):
        rng = np.random.default_rng(20260830)
        p = _sample_params(7, 0, cfg, rng, pathology_mode="max_stenosis",
                           severity_scale=scale)
        g = compute_geometry_from_params(p, cfg_dict)
        return _wall_severity(g.top_coords, g.bot_coords, "stenosis"), p

    full, p_full = sev(1.0)
    assert full > 3.0, f"a max_stenosis draw should be extreme, got {full:.2f}"

    prev = full
    for scale in (0.7, 0.5, 0.3):
        s_i, _ = sev(scale)
        assert s_i < prev, (
            f"severity_scale={scale} gave {s_i:.2f}, not below {prev:.2f} -- the ladder is "
            f"inert and every rung re-draws an equally unsolvable vessel"
        )
        prev = s_i
    assert prev < 0.6 * full, (
        f"the easiest rung only reached {prev:.2f} against {full:.2f}; a replacement has to be "
        f"meaningfully easier than the thing it replaces"
    )

    # scale 1.0 must reproduce the unmodified draw exactly -- normal generation is untouched
    rng = np.random.default_rng(20260830)
    p_plain = _sample_params(7, 0, cfg, rng, pathology_mode="max_stenosis")
    assert np.allclose(np.asarray(p_plain["offsets"], dtype=float),
                       np.asarray(p_full["offsets"], dtype=float)), \
        "severity_scale=1.0 changed a normal draw"


# --- B32: a repair round must attempt the vessels it repaired --------------------------------

def _fake_cohort(root: Path, n: int, solved: set[int]):
    """`n` CFD-ready vessels; those in `solved` also have a .npz."""
    meshes = root / "meshes"
    out = root / "npz"
    meshes.mkdir(parents=True)
    out.mkdir(parents=True)
    for i in range(n):
        (meshes / f"vessel_{i}.json").write_text("{}")
        (meshes / f"vessel_{i}.nas").write_text("x")
        (meshes / f"vessel_{i}.msh").write_text("x")
        if i in solved:
            (out / f"vessel_{i}.npz").write_bytes(b"x")
    return meshes, out


def test_repair_round_targets_only_the_rebuilt_vessels(tmp_path):
    """B32: with `allow_overwrite=True` the pool includes ALREADY-SOLVED vessels.

    A repair round runs with overwrite on (it inherits it from a `--overwrite` cohort) and a
    small `max_new` -- one per vessel it rebuilt.  Without `only_stems` the batch walks the whole
    cohort in index order, spends its budget re-solving healthy geometries, and stops before
    attempting a single repaired one.  Observed as `target new successes=7, candidate pool=50`
    followed by a screen of `Finished solving study` on vessels that were never broken.
    """
    from src.data_gen.lib.anchor_generator import select_anchor_candidates

    meshes, out = _fake_cohort(tmp_path, n=50, solved=set(range(43)))
    broken = [f"vessel_{i}" for i in range(43, 50)]

    # the bug: overwrite widens the pool to everything, and index order puts the healthy first
    wide, _ = select_anchor_candidates(meshes, out, allow_overwrite=True)
    assert len(wide) == 50
    assert [p.stem for p in wide][:5] != broken[:5], (
        "the first candidates are healthy vessels -- a max_new of 7 is spent before the "
        "repaired ones are reached"
    )

    # the fix
    got, missing = select_anchor_candidates(
        meshes, out, allow_overwrite=True, only_stems=broken)
    assert sorted(p.stem for p in got) == sorted(broken)
    assert missing == []

    # and a stem that is not CFD-ready is reported, not silently dropped
    got2, missing2 = select_anchor_candidates(
        meshes, out, allow_overwrite=True, only_stems=broken + ["vessel_999"])
    assert sorted(p.stem for p in got2) == sorted(broken)
    assert missing2 == ["vessel_999"]

    # without overwrite the pool is already just the unsolved ones
    narrow, _ = select_anchor_candidates(meshes, out, allow_overwrite=False)
    assert sorted(p.stem for p in narrow) == sorted(broken)


# --- mesh resolution is set in the units the model consumes -----------------------------------

def test_mesh_resolution_is_independent_of_vessel_size(tmp_path):
    """Resolution must be a property of the DESIGN, not of how big the vessel happens to be.

    Packs store positions as `x / d_bar`, so `h_nd` is what the WLS stencil and the edge features
    are built from.  A fixed physical `mesh_lc` made it a property of `d_bar` instead: measured
    on the 2026-08-30 cohort, `d_bar` spanned 3.9x and `h_nd` spanned 4.1x at spearman **-0.965**
    against it, leaving only 57% of vessels inside deployment's p10-p90 band while the median sat
    at 0.94x.  `mesh_h_nd_target` sets the size as a fraction of each vessel's own `d_bar`.
    """
    import numpy as np
    import pytest

    gmsh = pytest.importorskip("gmsh")

    from src.config import VesselConfig
    from src.data_gen.lib.vessel_generator import VesselGenerator, _mesh_geometry
    from src.data_gen.lib.vessel_geometry import compute_geometry_from_walls

    cfg = VesselConfig(phase="kinematics")
    cfg_dict = VesselGenerator(phase="kinematics")._cfg_dict()

    def mesh_h_nd(width, idx):
        n, length = 60, 0.1
        x = np.linspace(0.0, length, n)
        half = np.full(n, 0.5 * width)
        top = np.stack([x, half], axis=1)
        bot = np.stack([x, -half], axis=1)
        geom = compute_geometry_from_walls(top, bot, idx=idx, unit="m")
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", 1)
            gmsh.option.setNumber("Mesh.SaveAll", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFactor", cfg_dict["mesh_size_factor"])
            _, ok, err = _mesh_geometry(geom, cfg_dict, str(tmp_path))
            assert ok, err
        finally:
            gmsh.finalize()
        lines = (tmp_path / f"vessel_{idx}.msh").read_text().splitlines()
        i = lines.index("$Nodes")
        nn = int(lines[i + 1])
        pos = {}
        for k in range(nn):
            f = lines[i + 2 + k].split()
            pos[int(f[0])] = (float(f[1]), float(f[2]))
        j = lines.index("$Elements")
        edges = set()
        for k in range(int(lines[j + 1])):
            f = lines[j + 2 + k].split()
            if int(f[1]) != 2:
                continue
            nt = int(f[2])
            a, b, c = (int(v) for v in f[3 + nt:6 + nt])
            for u, v in ((a, b), (b, c), (c, a)):
                edges.add((min(u, v), max(u, v)))
        d_bar = float(geom.d_bar)
        el = [np.hypot(pos[u][0] - pos[v][0], pos[u][1] - pos[v][1]) / d_bar for u, v in edges]
        return float(np.median(el))

    small = mesh_h_nd(0.008, 9101)     # 8 mm
    large = mesh_h_nd(0.024, 9102)     # 24 mm, 3x the size

    ratio = max(small, large) / min(small, large)
    assert ratio < 1.15, (
        f"h_nd {small:.4f} vs {large:.4f} ({ratio:.2f}x) across a 3x change in vessel size -- "
        f"resolution is tracking d_bar instead of the target"
    )
    target = cfg.mesh_h_nd_target
    for got in (small, large):
        assert abs(got - target) / target < 0.15, (
            f"h_nd {got:.4f} against target {target:.4f}"
        )


# --- selection: the metric, the packs, and what promotion ranks on -----------------------------

def test_selection_metric_is_maximised_at_pred_equals_gt():
    """A PERFECT flow field must read gate Jaccard 1.0 of its own ceiling.

    The consumer differentiates a predicted field at ``hops=6`` and GT at ``hops=3``, and the
    shipped ``PRED_DSRX_GAIN = 3.0`` was least-squares fitted against the OLD surrogate -- so it
    carries that surrogate's ~1.35x under-resolution on top of the ~2.2x stencil attenuation.
    Selecting a RETRAIN against it rewards a model for staying under-resolved: measured on the
    deploy packs, feeding COMSOL's own velocity in as the prediction reads a median gate Jaccard
    of **0.835 at gain 3.0 against 0.941 at the stencil-only gain**.

    ``gate_jaccard_frac`` divides by the per-vessel ceiling, so it is 1.0 at the truth by
    construction and a cohort mean of it cannot mix model quality with metric defect.
    """
    import numpy as np
    import pytest
    import torch

    from src.utils.kinematics_select_packs import load_selection_packs
    from src.utils.kinematics_selection import GAIN_STENCIL, wall_shear_selection_metrics

    packs = load_selection_packs(limit=3, prior_source="analytic", verbose=False)
    if not packs:
        pytest.skip("no deploy selection packs on this machine")

    for g in packs:
        y = g.y[0] if g.y.dim() == 3 else g.y
        m = wall_shear_selection_metrics(y[:, :2], g, gain=GAIN_STENCIL)
        assert m, f"{g.graph_stem}: no metrics"
        assert m["gate_jaccard_frac"] == pytest.approx(1.0, abs=1e-9), (
            f"{g.graph_stem}: pred == GT must sit exactly at the ceiling"
        )
        # The ceiling itself is well below 1.0 and that is the metric's own defect, not a bug.
        assert 0.3 <= m["gate_jaccard_ceiling"] <= 1.0
        assert m["dsrx_corr"] > 0.9


def test_selection_packs_exclude_every_sealed_vessel():
    """Choosing a checkpoint on a vessel is tuning on it.

    ``docs/SEALED_SPLIT.md``: FINAL_HALF is reserved for the project's one final read, and
    VIZ_HALF may be shown but never used to select or tune.
    """
    from src.core_physics.wall_cohort_splits import DEV, FIT, SEALED, VIZ_RELEASED
    from src.utils.kinematics_select_packs import selection_pack_stems

    stems = set(selection_pack_stems())
    assert not (stems & set(SEALED)), sorted(stems & set(SEALED))
    assert not (stems & set(VIZ_RELEASED)), sorted(stems & set(VIZ_RELEASED))
    assert stems <= (set(FIT) | set(DEV))
    assert len(stems) >= 15


def test_selection_packs_default_to_deploy_legal_priors(monkeypatch):
    """`resolve_prior_source` defaults to "stored", and on these packs that IS the CFD solution.

    A gate Jaccard read off the s17 Z2 leak measures nothing: the analytic-prior arm posts
    rel-L2 0.02 that way against its true 0.147.
    """
    import pytest

    from src.utils.kinematics_select_packs import load_selection_packs

    monkeypatch.delenv("SPECIES_PRIOR_SOURCE", raising=False)
    packs = load_selection_packs(limit=2, verbose=False)
    if not packs:
        pytest.skip("no deploy selection packs on this machine")
    for g in packs:
        y = g.y[0] if g.y.dim() == 3 else g.y
        rel = float((g.x[:, 11:13] - y[:, :2]).norm() / y[:, :2].norm().clamp(min=1e-30))
        assert rel > 0.05, f"{g.graph_stem}: prior block is the CFD solution (rel-L2 {rel:.4f})"


# --- the corpus's own labels: what P2 elevation costs the gate's dominant branch ---------------

def test_elevation_drops_the_dead_wls_operators_without_changing_the_graph():
    """`V`/`W`/`M_inv` are 47% of an elevated graph and nothing in training reads them.

    `graph_gradient_operators` defaults to MLS and rebuilds from positions + connectivity;
    the stored arrays are read only under `BIOCHEM_GRAD_OPERATOR=legacy`.  Keeping them is also
    the B13 hazard -- an operator that no longer matches the graph stored beside it.
    """
    import torch

    from src.data_gen.lib.p2_elevation import elevate_to_p2

    d = _load_first_kine_pack()
    keep = elevate_to_p2(d, keep_wls=True)
    drop = elevate_to_p2(d, keep_wls=False)

    assert all(hasattr(keep, k) for k in ("V", "W", "M_inv"))
    assert not any(hasattr(drop, k) for k in ("V", "W", "M_inv"))
    # Everything the model consumes is bit-identical, width derivatives included.
    assert torch.equal(keep.x, drop.x)
    assert torch.equal(keep.edge_index, drop.edge_index)
    assert int(keep.num_nodes) == int(drop.num_nodes)


def test_band_shear_terms_are_zero_at_the_labels_in_both_band_modes(monkeypatch):
    """`l_band_sr` / `l_band_dsrx` must vanish when the prediction IS the ground truth.

    That is the well-posedness statement for the corner-view arm: moving where the shear terms
    are evaluated must not move their optimum.
    """
    import torch

    from src.config import PhysicsConfig
    from src.core_physics.physics_kernels import PhysicsKernels
    from src.data_gen.lib.p2_elevation import elevate_to_p2
    from src.utils.anchor_mask import anchor_node_mask
    from src.utils.kinematics_physics_terms import corner_view, wall_band_shear_losses

    monkeypatch.setenv("KINEMATICS_NORMALIZE_SHEAR_GRAD", "1")
    kern = PhysicsKernels(phys_cfg=PhysicsConfig(phase="kinematics"))
    e = elevate_to_p2(_load_first_kine_pack(), keep_wls=False)
    nia = anchor_node_mask(e)
    pred = torch.zeros(int(e.num_nodes), 8)
    pred[:, : e.y.shape[1]] = e.y
    pred.requires_grad_(True)

    assert corner_view(e) is not None, "an elevated graph must expose its P1 corner view"

    for mode in ("0", "1"):
        monkeypatch.setenv("KINEMATICS_BAND_ON_CORNERS", mode)
        l_sr, l_dsrx, l_gate, _l_floor, _l_tail = wall_band_shear_losses(
            pred, e, kern, hops=3, node_is_anchor=nia
        )
        assert float(l_sr) < 1e-8, f"mode={mode} l_band_sr={float(l_sr)}"
        assert float(l_dsrx) < 1e-8, f"mode={mode} l_band_dsrx={float(l_dsrx)}"
        # The soft gate has a finite temperature, so its floor at the truth is small, not zero.
        assert float(l_gate) < 0.25, f"mode={mode} l_band_gate={float(l_gate)}"


def _load_first_kine_pack():
    import pytest
    import torch

    from src.utils.kinematics_paths import kinematics_training_graph_dir

    files = sorted(kinematics_training_graph_dir(rheology="carreau").glob("vessel_*.pt"))
    if not files:
        pytest.skip("no synthetic kinematics packs on this machine")
    return torch.load(files[0], map_location="cpu", weights_only=False)


def test_wall_shear_prior_floor_is_one_sided(monkeypatch):
    """The prior must be a floor in the SHEAR channel, and only a floor.

    Zero when the prediction is the ground truth, zero when it IS the prior (it sits exactly on
    the floor by construction), positive only when the model is worse than the field it was
    handed.  This exists because T6's velocity-only floor is cleared by a surrogate that still
    lands 8 points of gate Jaccard behind that same prior (§16.4).
    """
    import torch

    from src.config import PhysicsConfig
    from src.core_physics.physics_kernels import PhysicsKernels
    from src.data_gen.lib.legal_priors import apply_prior_source
    from src.data_gen.lib.p2_elevation import elevate_to_p2
    from src.utils.anchor_mask import anchor_node_mask
    from src.utils.kinematics_physics_terms import wall_band_shear_losses

    monkeypatch.setenv("KINEMATICS_NORMALIZE_SHEAR_GRAD", "1")
    monkeypatch.setenv("KINEMATICS_BAND_SHEAR_FLOOR", "1")
    monkeypatch.setenv("KINEMATICS_BAND_ON_CORNERS", "0")
    kern = PhysicsKernels(phys_cfg=PhysicsConfig(phase="kinematics"))
    e = apply_prior_source(elevate_to_p2(_load_first_kine_pack(), keep_wls=False), "analytic")
    nia = anchor_node_mask(e)

    def floor_at(uv):
        pred = torch.zeros(int(e.num_nodes), 8)
        pred[:, : e.y.shape[1]] = e.y
        pred[:, :2] = uv
        pred.requires_grad_(True)
        return float(wall_band_shear_losses(pred, e, kern, hops=3, node_is_anchor=nia)[3])

    assert floor_at(e.y[:, :2]) == 0.0, "the truth cannot be worse than the prior"
    assert floor_at(e.x[:, 11:13]) == 0.0, "the prior sits exactly on its own floor"
    assert floor_at(0.5 * e.y[:, :2]) > 0.0, "a shrunk field must be penalised"

    monkeypatch.setenv("KINEMATICS_BAND_SHEAR_FLOOR", "0")
    assert floor_at(0.5 * e.y[:, :2]) == 0.0, "unset must be a no-op"


def test_deploy_training_packs_are_disjoint_from_selection_and_carry_no_chemistry(monkeypatch):
    """Training on the deploy packs is only legitimate if two things hold.

    1. **Disjoint from selection.**  A vessel cannot be both trained on and used to choose the
       checkpoint; that would measure memorisation, not transfer.  Both halves of the old SEALED
       set stay out on top of that.
    2. **No chemistry leaks into a kinematics label.**  These packs carry `biochem_v1_16ch`:
       `u_nd, v_nd, p_nd, mu_eff_nd` and then twelve species.  `PredChannels.WSS` is 4, so an
       untruncated `y` would have supervised the WSS head against `RP_log1p_nd` at weight 5.35.
    """
    import pytest
    import torch

    from src.config import PhysicsConfig
    from src.core_physics.physics_kernels import PhysicsKernels
    from src.core_physics.wall_cohort_splits import SEALED, VIZ_RELEASED
    from src.utils.kinematics_select_packs import (
        load_deploy_training_packs, selection_subset_stems,
    )

    monkeypatch.setenv("KINEMATICS_SELECT_MAX_GRAPHS", "8")
    monkeypatch.setenv("SPECIES_PRIOR_SOURCE", "analytic")
    packs = load_deploy_training_packs(verbose=False)
    if not packs:
        pytest.skip("no deploy packs on this machine")

    stems = {p.graph_stem for p in packs}
    assert not (stems & set(selection_subset_stems())), sorted(stems & set(selection_subset_stems()))
    assert not (stems & set(SEALED)), sorted(stems & set(SEALED))
    assert not (stems & set(VIZ_RELEASED)), sorted(stems & set(VIZ_RELEASED))

    kern = PhysicsKernels(phys_cfg=PhysicsConfig(phase="kinematics"))
    for g in packs:
        assert g.y.shape[1] == 4, f"{g.graph_stem}: y has {g.y.shape[1]} channels, expected 4"
        assert bool(g.is_anchor.all()), f"{g.graph_stem}: COMSOL labels every node here"
        # the prior block must not be the CFD solution
        rel = float((g.x[:, 11:13] - g.y[:, :2]).norm() / g.y[:, :2].norm().clamp(min=1e-30))
        assert rel > 0.05, f"{g.graph_stem}: leaked prior (rel-L2 {rel:.4f})"
        pred = torch.zeros(int(g.num_nodes), 5)
        pred[:, :4] = g.y
        assert float(kern.wall_shear_stress_loss(pred, g)) == 0.0, (
            f"{g.graph_stem}: the WSS term must self-disable on a 4-channel y"
        )


def test_elevation_prefers_true_midside_labels_when_the_pack_carries_them():
    """A mid-side label must be COMSOL's own value where one was solved, the corner mean where
    it was not, and exactly zero on the wall either way.

    The corner mean makes the field piecewise-linear along the half-edge by construction, so a
    quadratic fit through it reads ~zero curvature -- and `dsrx`, the gate branch that decides
    ~91% of firing wall nodes at the FIT median, IS that curvature.
    """
    import torch

    from src.data_gen.lib.p2_elevation import elevate_to_p2, undirected_edges

    d = _load_first_kine_pack()
    # Strip any probe set the pack already carries -- the rebuilt corpus ships one on every
    # vessel -- so the fallback branch is exercised from a known-empty baseline.
    for attr in ("p2_probe_xy_nd", "p2_probe_y"):
        if hasattr(d, attr):
            delattr(d, attr)
    base = elevate_to_p2(d, keep_wls=False)
    assert int(getattr(base, "p2_midside_true", 0)) == 0, "no probe set -> no true mid-side labels"

    # Manufacture a probe set at the true midpoints with a value the corner mean cannot produce.
    n = int(d.num_nodes)
    pairs = undirected_edges(d.edge_index)
    a, b = pairs[:, 0], pairs[:, 1]
    mid_xy = 0.5 * (d.x[a, 0:2] + d.x[b, 0:2])
    half = pairs.shape[0] // 2                      # only half get a probe
    d.p2_probe_xy_nd = mid_xy[:half].clone()
    marker = 7.5
    d.p2_probe_y = torch.full((half, 4), marker, dtype=torch.float32)

    out = elevate_to_p2(d, keep_wls=False)
    assert int(out.p2_midside_true) == half, f"matched {out.p2_midside_true} of {half}"

    mid_y = out.y[n:]
    lin = 0.5 * (d.y[a] + d.y[b])
    wall = d.mask_wall.reshape(-1).bool()
    wall_mid = (wall[a] & wall[b])

    probed = torch.zeros(pairs.shape[0], dtype=torch.bool)
    probed[:half] = True
    interior_probed = probed & ~wall_mid
    if bool(interior_probed.any()):
        assert torch.allclose(mid_y[interior_probed][:, 0],
                              torch.full((int(interior_probed.sum()),), marker)), \
            "a probed interior mid-side must take COMSOL's value, not the corner mean"
    unprobed = ~probed
    if bool(unprobed.any()):
        assert torch.allclose(mid_y[unprobed], lin[unprobed]), \
            "an unprobed mid-side must fall back to the corner mean"
    if bool(wall_mid.any()):
        assert float(mid_y[wall_mid][:, 0:2].abs().max()) == 0.0, \
            "no-slip is a boundary condition, not a sample"
