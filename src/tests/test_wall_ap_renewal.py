"""Tests for the time-varying wall-AP ODE (wall_ap_renewal.py)."""

from __future__ import annotations
from src.utils.paths import anchor_packs_dir

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Repo root by marker, not by depth: this file may move between
# scripts/ and scripts/<subdir>/ without silently resolving one level off.
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig
from src.core_physics.physics_wall_model import T0Fields, integrate_mat_trajectory
from src.core_physics.wall_ap_renewal import WallApRenewal, make_species_from_renewal

# Find a sample pack for realistic geometry (we need edge_index and positions for upwind)
PACK_DIR = anchor_packs_dir()
SAMPLE_PACK_PATH = PACK_DIR / "comsol020.pt"

@pytest.fixture(scope="module")
def sample_pack():
    if not SAMPLE_PACK_PATH.exists():
        pytest.skip(f"Sample pack not found at {SAMPLE_PACK_PATH}")
    return torch.load(SAMPLE_PACK_PATH, map_location="cpu", weights_only=False)

@pytest.fixture(scope="module")
def bio_cfg():
    return BiochemConfig(phase="biochem")


def _make_dummy_fields(data):
    N = int(data.num_nodes)
    return T0Fields(
        sr=np.full(N, 100.0),
        dsrx=np.zeros(N),
        gate_low=np.zeros(N),
        gate_sep=np.zeros(N),
        gate=np.ones(N),
        u=np.full(N, 0.1),
        v=np.zeros(N),
    )


def test_renewal_scale_zero_is_identity(sample_pack, bio_cfg):
    """renewal_scale=0 returns frozen ap0, bit-identical to not passing it."""
    fields = _make_dummy_fields(sample_pack)
    renewal = WallApRenewal(renewal_scale=0.0)
    
    # 1. Base integration with no species/renewal
    traj_base, _ = integrate_mat_trajectory(
        sample_pack, bio_cfg, fields.gate,
    )
    
    # 2. Integration via wall_ap_renewal convenience kwargs
    traj_renew, _ = integrate_mat_trajectory(
        sample_pack, bio_cfg, fields.gate,
        wall_ap_renewal=renewal,
        wall_ap_fields=fields,
    )
    
    # They must be exactly identical
    assert np.array_equal(traj_base, traj_renew)


def test_monotone_depletion(sample_pack, bio_cfg):
    """With non-zero gate and no renewal (or low renewal), AP must decrease."""
    fields = _make_dummy_fields(sample_pack)
    renewal = WallApRenewal(renewal_scale=0.0001)  # tiny renewal so consumption dominates
    
    _, ap_traj = make_species_from_renewal(
        sample_pack, bio_cfg, fields, renewal=renewal
    )
    
    wall_mask = sample_pack.mask_wall.reshape(-1).bool().cpu().numpy()
    
    # ap_traj is [T, N]. Check that for wall nodes, AP at T-1 is less than AP at 0
    ap0 = ap_traj[0, wall_mask]
    ap_final = ap_traj[-1, wall_mask]
    
    assert np.all(ap_final < ap0), "AP did not deplete on wall nodes"
    
    # Check off-wall nodes (they should remain exactly at ap0)
    off_wall_mask = ~wall_mask
    if np.any(off_wall_mask):
        ap0_off = ap_traj[0, off_wall_mask]
        ap_final_off = ap_traj[-1, off_wall_mask]
        assert np.array_equal(ap0_off, ap_final_off), "Off-wall AP changed"


def test_no_gt_leak_when_fields_provided(sample_pack, bio_cfg):
    """If u and v are in fields, the function never touches data.y after t=0.
    
    t=0 is the initial condition (wall_platelet_constants reads ap0/rp0 from it).
    We poison all t > 0 data.y to prove no future GT flow or chemistry is leaked.
    """
    import copy
    data_no_y = copy.copy(sample_pack)
    data_no_y.y = data_no_y.y.clone()
    data_no_y.y[1:] = float("nan")  # Poison future GT

    
    fields = _make_dummy_fields(data_no_y)
    renewal = WallApRenewal(renewal_scale=1.0)
    
    # Should not raise an error
    _, ap_traj = make_species_from_renewal(
        data_no_y, bio_cfg, fields, renewal=renewal
    )
    assert np.all(np.isfinite(ap_traj))


def test_no_wound_pack_safety(sample_pack, bio_cfg):
    """A pack without `mask_wound` still evaluates perfectly."""
    import copy
    data_no_wound = copy.copy(sample_pack)
    if hasattr(data_no_wound, "mask_wound"):
        delattr(data_no_wound, "mask_wound")
        
    fields = _make_dummy_fields(data_no_wound)
    renewal = WallApRenewal(renewal_scale=1.0)
    
    _, ap_traj = make_species_from_renewal(
        data_no_wound, bio_cfg, fields, renewal=renewal
    )
    assert np.all(np.isfinite(ap_traj))
