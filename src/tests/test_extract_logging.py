"""Quiet-by-default COMSOL extract logging."""

from __future__ import annotations

import logging

import pytest

from src.data_gen.lib import extract_logging as el


@pytest.fixture(autouse=True)
def _reset_extract_logging(monkeypatch):
    monkeypatch.delenv("BIOCHEM_EXTRACT_VERBOSE", raising=False)
    el._verbose_forced = None
    yield
    el._verbose_forced = None
    monkeypatch.delenv("BIOCHEM_EXTRACT_VERBOSE", raising=False)


def test_quiet_comsol_extract_logs_hides_mph_info():
    el.quiet_comsol_extract_logs()
    mph = logging.getLogger("mph")
    export = logging.getLogger("src.data_gen.lib.biochem_comsol_mph_export")
    datasets = logging.getLogger("src.data_gen.lib.biochem_comsol_datasets")
    assert mph.level == logging.WARNING
    assert export.level == logging.WARNING
    assert datasets.level == logging.WARNING
    assert not mph.isEnabledFor(logging.INFO)
    assert mph.isEnabledFor(logging.WARNING)


def test_verbose_flag_sticks_for_later_quiet_calls():
    el.quiet_comsol_extract_logs(verbose=True)
    el.quiet_comsol_extract_logs()
    assert el.extract_verbose_requested()
    assert logging.getLogger("mph").level == logging.INFO
    assert logging.getLogger("mph").isEnabledFor(logging.INFO)


def test_env_verbose_without_cli_flag(monkeypatch):
    monkeypatch.setenv("BIOCHEM_EXTRACT_VERBOSE", "1")
    el.quiet_comsol_extract_logs()
    assert logging.getLogger("mph").level == logging.INFO
