"""Tests for the centralized diagnostics package."""

from __future__ import annotations

import importlib

import pytest

from src.tools.diagnostics.registry import DIAGNOSTICS, resolve_main


@pytest.mark.parametrize("slug", sorted(DIAGNOSTICS))
def test_diagnostic_modules_expose_main(slug: str) -> None:
    main = resolve_main(slug)
    assert callable(main)


def test_registry_matches_package_exports() -> None:
    from src.tools.diagnostics import DIAGNOSTICS as exported

    assert exported == DIAGNOSTICS


def test_diagnostics_cli_list() -> None:
    from src.tools.diagnostics.__main__ import main

    assert main(["list"]) == 0


def test_pi_flux_help() -> None:
    from src.tools.diagnostics.pi_flux_interaction import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
