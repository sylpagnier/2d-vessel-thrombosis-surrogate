"""Customer deploy defaults to local FEM t=0 flow (same as research sweeps)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data_gen.lib.customer_geometry_import import (
    build_parametric_customer_graph,
    default_customer_mesh_cache_dir,
)
from src.inference.customer_pipeline import DEFAULT_CUSTOMER_FLOW


def test_default_customer_flow_is_fem():
    assert DEFAULT_CUSTOMER_FLOW == "fem"


@pytest.mark.slow
def test_parametric_customer_graph_persists_mesh_for_fem():
    data = build_parametric_customer_graph(
        width=0.010,
        angle_span=0.0,
        amplitude=0.0,
        t_final_s=8000.0,
        n_steps=8,
    )
    mesh_path = getattr(data, "mesh_path", None)
    assert mesh_path is not None
    p = Path(mesh_path)
    assert p.is_file()
    assert p.parent == default_customer_mesh_cache_dir().resolve()
    stem = str(getattr(data, "graph_stem", ""))
    assert stem.startswith("customer_")
