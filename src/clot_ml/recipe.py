"""The one canonical `clot_gnn` training recipe.

WHY THIS EXISTS
---------------
The same hyperparameters were written out three times -- in
``scripts/promote_clot_gnn_v4.py`` (promotion), ``scripts/run_phase9_cv.py``
(strict cross-validation) and ``scripts/customer_retrain_run.py`` (the retrain a
customer runs on their own vessels).  All three agreed on fourteen of fifteen
shared keys, which is exactly the state that rots quietly: a tuning change lands
in one copy, and the model a customer retrains stops matching the model that was
validated, with nothing failing to say so.

The values below are the shipped objective.  Anything a caller varies on purpose
is applied as an explicit override at the call site, so a diff shows the
intent rather than a whole second copy of the recipe.

``empty_gt_loss="none"`` and ``shape_w=0.0`` in particular are the C0 objective:
they must stay in lockstep across promotion and CV or the two stop measuring the
same model.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

#: Architecture and optimiser settings shared by every clot_gnn training entry point.
CLOT_GNN_BASE: Mapping[str, Any] = MappingProxyType(
    {
        # architecture
        "dim": 64,
        "layers": 4,
        "drop": 0.1,
        # optimiser
        "lr": 3e-3,
        "wd": 1e-4,
        # objective
        "pos_weight": 30.0,
        "reg_w": 1.0,
        "metric_w": 2.0,
        "metric_start": 0.3,
        "metric": "legacy",
        "off_mult": 1.0,
        "empty_gt_loss": "none",
        "burden_w": 0.0,
        "shape_w": 0.0,
        "clot_free_w": 1.0,
    }
)

#: Promotion and strict CV train the full schedule.
PROMOTION_EPOCHS = 80
#: The customer retrain is deliberately shorter -- it runs on a laptop, on a
#: handful of the customer's own vessels, and is reviewed before it ships.
CUSTOMER_RETRAIN_EPOCHS = 40


def recipe(**overrides: Any) -> dict[str, Any]:
    """The canonical recipe with explicit, visible deviations applied.

    >>> recipe(epochs=PROMOTION_EPOCHS)["pos_weight"]
    30.0
    """
    return {**CLOT_GNN_BASE, **overrides}
