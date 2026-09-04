"""One place that knows which environment knobs this project reads.

WHY THIS EXISTS
---------------
Configuration here grew as ``os.environ.get("SOME_KNOB", "default")`` written at
the point of use.  That is cheap to add and expensive to live with: at the time
this module was introduced the tree read **471** distinct variables across ~750
sites, **275** of which were never set anywhere in the repository and never
mentioned in any doc or script.  A reader had to understand each one before they
could conclude it did not matter, and the same knob could be read with different
fallbacks in different modules (``CLOT_V2_NUCLEATION_HOPS`` had four).

The fix is not to rewrite 750 call sites at once -- it is to stop the growth and
give the existing set a single index.  So:

* :data:`KNOWN_ENV` is the frozen inventory of knobs that existed when this
  registry was added.  It is an allowlist, not an endorsement.
* ``src/tests/test_env_registry.py`` fails if a knob appears in the tree that is
  not in the inventory, which forces genuinely new configuration through the
  typed configs (``BiochemRuntimeConfig``, ``PushforwardConfig``,
  ``PhysicsConfig``, ``VesselConfig``) instead of a fresh ad-hoc variable.
* Knobs bound to the typed runtime are listed in :data:`TYPED_ENV`; those are the
  ones with a documented precedence and a dataclass field behind them.

ADDING CONFIGURATION
--------------------
Add a field to the relevant typed config and, if it needs an environment
override, bind it in ``runtime_config.RUNTIME_ENV_TO_FIELD``.  Do not add a bare
``os.environ.get`` -- the test will reject it.

REMOVING CONFIGURATION
----------------------
Deleting a dead knob is always allowed: drop the read, drop it from the
inventory.  The inventory is expected to shrink over time, never grow.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_ALLOWLIST = Path(__file__).with_name("_env_allowlist.json")


@lru_cache(maxsize=1)
def _load() -> dict:
    return json.loads(_ALLOWLIST.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def known_env() -> frozenset[str]:
    """Every environment variable the tree was known to read when frozen."""
    return frozenset(_load()["known"])


@lru_cache(maxsize=1)
def typed_env() -> frozenset[str]:
    """Knobs bound to a typed runtime field, i.e. with a real precedence rule."""
    return frozenset(_load()["typed_runtime_bound"])


def is_known(name: str) -> bool:
    return str(name) in known_env()


def untyped_env() -> frozenset[str]:
    """Knobs still read ad-hoc. This set should only ever shrink."""
    return known_env() - typed_env()
