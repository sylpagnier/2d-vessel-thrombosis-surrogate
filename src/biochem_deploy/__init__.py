"""Canonical import alias for ``src.biochem_gnn``.

The ``biochem_gnn`` model class this alias was originally written for is retired --
see ``docs/BIOCHEM_GNN.md``. What ``src.biochem_gnn`` still exports (checkpoint-path
helpers and env-application config, not a model) is re-exported here unchanged so
existing imports of this alias keep working.
"""

from src.biochem_gnn import *  # noqa: F403
from src.biochem_gnn import __all__  # noqa: F401
