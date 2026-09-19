"""Retrain subprocess hygiene: unique anchor names, and no run outliving the UI that started it."""
from __future__ import annotations

import subprocess
import sys
import threading

import pytest

from src.utils.paths import get_project_root

ROOT = get_project_root()


def _eof_within(stream, timeout_s: float) -> bool:
    """True once every writer of ``stream`` has exited (read hits EOF) within the timeout."""
    done = threading.Event()

    def drain() -> None:
        while stream.read(1024):
            pass
        done.set()

    threading.Thread(target=drain, daemon=True).start()
    return done.wait(timeout_s)


# A parent that starts a grandchild sharing its stdout, reports the grandchild's start, then
# sleeps.  The pipe reaches EOF only when both have exited.
_PARENT = r"""
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", sys.argv[1], str(__import__("os").getpid())])
time.sleep(120)
"""


def test_watchdog_child_exits_when_its_parent_is_killed():
    child = (
        "import sys, time; sys.path.insert(0, %r);"
        "from src.utils.parent_watchdog import exit_when_parent_dies;"
        "exit_when_parent_dies(int(sys.argv[1])); print('up', flush=True); time.sleep(120)"
    ) % str(ROOT)
    parent = subprocess.Popen([sys.executable, "-c", _PARENT, child], stdout=subprocess.PIPE)
    try:
        assert parent.stdout.readline().strip() == b"up"
        parent.kill()
        parent.wait(timeout=10)
        assert _eof_within(parent.stdout, 20), "grandchild outlived its killed parent"
    finally:
        parent.kill()


def test_kill_process_tree_reaches_grandchildren():
    from src.inference.customer_retrain_pipeline import kill_process_tree

    child = "import time; print('up', flush=True); time.sleep(120)"
    parent = subprocess.Popen(
        [sys.executable, "-c", _PARENT, child], stdout=subprocess.PIPE,
        start_new_session=sys.platform != "win32",
    )
    try:
        assert parent.stdout.readline().strip() == b"up"
        kill_process_tree(parent)
        assert _eof_within(parent.stdout, 20), "grandchild survived kill_process_tree"
    finally:
        parent.kill()


def test_retrain_anchor_names_never_collide():
    from scripts.customer_retrain_run import _safe_stem

    taken: dict[str, object] = {}
    for name in ("a b", "a_b", "a-b", "customer_a_b"):
        stem = _safe_stem(name, taken)
        assert stem not in taken
        taken[stem] = object()
    assert set(taken) == {"customer_a_b", "customer_a_b_2", "customer_a_b_3", "customer_a_b_4"}


def test_untrusted_loader_refuses_torch_without_the_weights_only_fix(monkeypatch, tmp_path):
    import torch

    from src.utils import safe_load

    monkeypatch.setattr(torch, "__version__", "2.5.1+cu121")
    with pytest.raises(RuntimeError, match="upgrade"):
        safe_load.load_untrusted_graph(tmp_path / "missing.pt")
    monkeypatch.setattr(torch, "__version__", "2.11.0+cu128")
    assert safe_load._torch_version() == (2, 11)
