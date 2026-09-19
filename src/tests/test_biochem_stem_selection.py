"""Tests for multi-stem COMSOL extract selection parsing."""

from __future__ import annotations

from src.data_gen.lib.biochem_comsol_auto_export import resolve_stem_selection


def _table(n: int) -> list[str]:
    return [f"comsol{i:03d}" for i in range(1, n + 1)]


def test_resolve_indices_and_ranges():
    table = _table(11)
    assert resolve_stem_selection("5", table) == ["comsol005"]
    assert resolve_stem_selection("5,8,9", table) == ["comsol005", "comsol008", "comsol009"]
    assert resolve_stem_selection("8-10", table) == ["comsol008", "comsol009", "comsol010"]
    assert resolve_stem_selection("5, 8-10", table) == [
        "comsol005",
        "comsol008",
        "comsol009",
        "comsol010",
    ]


def test_resolve_comsol_names():
    table = _table(7)
    assert resolve_stem_selection("comsol007", table) == ["comsol007"]
    assert resolve_stem_selection("comsol005,comsol007", table) == ["comsol005", "comsol007"]
