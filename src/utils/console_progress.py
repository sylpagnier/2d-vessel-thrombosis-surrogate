"""Compact console + progress policy for the data-generation pipelines.

Long cohort runs used to bury their own results.  Three habits caused it:

*   Full-width ``tqdm`` bars.  Anything else touching the console breaks the
    carriage return, so a 275-file loop left 275 wide bars in the scrollback.
*   Per-sample INFO chatter -- mesh import, dataset resolution, study start --
    repeating the same sentence once per vessel.
*   Third-party start-up logs (``mph``/COMSOL) narrating the JVM handshake.

The helpers here keep bars short and throttled, route log records through
``tqdm.write`` so a bar is never torn, and mute the third-party narration.  Set
``PIPELINE_VERBOSE=1`` to get every original line back, ``PIPELINE_PROGRESS=0``
to drop the bars for a decile line instead.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from typing import Iterable, Iterator, Optional

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Bars stay narrow on purpose: a torn redraw then costs one short line, not one
# 158-column line, and the numbers still fit next to the pipeline's own output.
_BAR_FORMAT = "{desc} {percentage:3.0f}%|{bar:18}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"


def verbose() -> bool:
    """True when the caller asked for the full, unfiltered log stream."""
    return os.environ.get("PIPELINE_VERBOSE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _bars_enabled() -> bool:
    raw = os.environ.get("PIPELINE_PROGRESS", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    # Redirected output has no carriage return, so a bar there is pure spam.
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        return False


class _PlainProgress:
    """Bar-free fallback: one line per decile, so a log file stays readable."""

    def __init__(self, iterable: Iterable, *, desc: str, total: Optional[int]):
        self._iterable = iterable
        self._desc = desc
        self._total = total
        self._log = logging.getLogger(__name__)

    def __iter__(self) -> Iterator:
        last_decile = -1
        for i, item in enumerate(self._iterable, start=1):
            yield item
            if not self._total:
                continue
            decile = (10 * i) // self._total
            if decile != last_decile:
                last_decile = decile
                self._log.info("%s %d/%d (%d%%)", self._desc, i, self._total, 10 * decile)

    def set_postfix_str(self, _s: str = "", **_kw) -> None:
        return None


def progress(
    iterable: Iterable,
    *,
    desc: str,
    unit: str = "it",
    total: Optional[int] = None,
):
    """A short, throttled progress bar -- or a decile log line when bars are off.

    The returned object is always safe to iterate and always exposes
    ``set_postfix_str``, so callers need no branching of their own.
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None
    if not _bars_enabled():
        return _PlainProgress(iterable, desc=desc, total=total)
    try:
        sys.stdout.flush()
        return tqdm(
            iterable,
            desc=desc,
            total=total,
            unit=unit,
            bar_format=_BAR_FORMAT,
            ascii=sys.platform == "win32",
            dynamic_ncols=False,
            mininterval=0.5,
            leave=True,
        )
    except OSError:
        # Broken console pipe (Windows/PyCharm): never let cosmetics kill a batch.
        return _PlainProgress(iterable, desc=desc, total=total)


class _PlainCounter:
    """``.update()``-driven fallback for loops that own their own counting."""

    def __init__(self, *, desc: str, total: Optional[int]):
        self._desc = desc
        self._total = total
        self._n = 0
        self._last_decile = -1
        self._log = logging.getLogger(__name__)

    def update(self, k: int = 1) -> None:
        self._n += k
        if not self._total:
            return
        decile = (10 * self._n) // self._total
        if decile != self._last_decile:
            self._last_decile = decile
            self._log.info("%s %d/%d (%d%%)", self._desc, self._n, self._total, 10 * decile)

    def set_postfix_str(self, _s: str = "", **_kw) -> None:
        return None


@contextmanager
def counter(*, desc: str, total: Optional[int], unit: str = "it"):
    """A short bar driven by ``.update()`` -- the manual counterpart of ``progress``."""
    if not _bars_enabled():
        yield _PlainCounter(desc=desc, total=total)
        return
    bar = tqdm(
        total=total,
        desc=desc,
        unit=unit,
        bar_format=_BAR_FORMAT,
        ascii=sys.platform == "win32",
        dynamic_ncols=False,
        mininterval=0.5,
        leave=True,
    )
    try:
        with logging_redirect_tqdm():
            yield bar
    finally:
        bar.close()


@contextmanager
def logs_above_bar():
    """Route log records through ``tqdm.write`` so records never tear the bar."""
    if not _bars_enabled():
        yield
        return
    with logging_redirect_tqdm():
        yield


_QUIET_APPLIED = False


def quiet_pipeline_logs() -> None:
    """Silence the ``mph``/COMSOL start-up narration of the JVM handshake.

    Idempotent.  Under ``PIPELINE_VERBOSE=1`` it does the opposite: the per-sample
    detail these pipelines log at DEBUG is turned back on.
    """
    global _QUIET_APPLIED
    if _QUIET_APPLIED:
        return
    _QUIET_APPLIED = True
    if verbose():
        logging.getLogger().setLevel(logging.DEBUG)
        return
    for name in ("mph", "mph.client", "mph.session", "comsol", "jpype", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)
