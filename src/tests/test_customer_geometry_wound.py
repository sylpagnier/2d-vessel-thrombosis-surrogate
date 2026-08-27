from __future__ import annotations

import torch
from torch_geometric.data import Data

from src.data_gen.lib.customer_geometry_import import apply_customer_mirrored_wound


def _straight_channel() -> Data:
    # Top/bottom wall nodes at each of five axial locations, plus inlet/outlet anchors
    # that orient the wound interval from left to right.
    pos = []
    for x in range(5):
        pos.extend(((float(x), -1.0), (float(x), 1.0)))
    pos.extend(((-0.2, 0.0), (4.2, 0.0)))
    n = len(pos)
    wall = torch.zeros(n, dtype=torch.bool)
    wall[:10] = True
    inlet = torch.zeros(n, dtype=torch.bool)
    outlet = torch.zeros(n, dtype=torch.bool)
    inlet[-2] = True
    outlet[-1] = True
    return Data(
        x=torch.tensor(pos, dtype=torch.float32),
        mask_wall=wall,
        mask_inlet=inlet,
        mask_outlet=outlet,
        num_nodes=n,
    )


def test_customer_mirrored_wound_marks_both_wall_sides_and_preserves_solid_boundary():
    data = _straight_channel()
    wounded = apply_customer_mirrored_wound(
        data, enabled=True, position_frac=0.50, width_frac=0.30
    )

    # The central top and bottom points are both selected; wall and wound stay disjoint.
    assert wounded.mask_wound[4]
    assert wounded.mask_wound[5]
    assert not bool((wounded.mask_wall & wounded.mask_wound).any())
    assert torch.equal(wounded.mask_wall | wounded.mask_wound, data.mask_wall)


def test_customer_wound_can_be_removed_without_changing_the_vessel_boundary():
    data = _straight_channel()
    wounded = apply_customer_mirrored_wound(data, enabled=True, position_frac=0.50, width_frac=0.30)
    restored = apply_customer_mirrored_wound(wounded, enabled=False)

    assert not bool(restored.mask_wound.any())
    assert torch.equal(restored.mask_wall, data.mask_wall)
