"""THE source of truth for which trained artifact anything loads.

WHY THIS MODULE EXISTS.  Artifact identity used to live in ~20 places: `DEFAULT_NAME` in
`v0.py`, `_LOCKED_CUSTOMER_CLOT_MODEL` in the customer pipeline, `clot_ml_model` in the
publication config, a `BASE`/`DEFAULT_BASE` constant in three promotion scripts, and an
`argparse` default in a dozen more.  They drifted, because nothing made them agree: on
2026-09-04 the shipped stack was `DeployClot2_0` while `eval_clot_ml_0.py --baseline` still
defaulted to `clot_gnn_v5w`, two generations back, and `run_research_sweep.py` was serving the
legacy `clot_ml_v0` stub to the customer pipeline for an entire sweep campaign
(docs/DEPLOYCLOT.md 21 and 26).

The fix is to stop naming artifacts in code at all.  A caller asks for a ROLE -- "the shipped
unified model", "the wound complement underneath it" -- and this module answers from the
locked pointer.  Repointing then changes every consumer at once, which is what a pointer was
always supposed to mean.

    from src.clot_ml.artifacts import shipped, UNIFIED, WOUND, BASE

    shipped(UNIFIED)     # -> "DeployClot2_0"   whatever is currently promoted
    shipped(WOUND)       # -> "DeployClot2_w"   derived from its manifest chain
    shipped(BASE)        # -> "DeployClot2"

THE CHAIN IS DERIVED, NOT LISTED.  A `unified_v0` manifest names its `base_model`, which is a
`temporal_v4_wound` naming its own base, which is the `temporal_v4` GNN.  Walking that chain
means the three roles cannot disagree with each other, and a new generation needs no edit
here -- only a promotion.

EXPLICIT NAMES ALWAYS WIN.  `resolve("DeployClot_0")` returns exactly that.  Pinned
comparisons against a named past generation are the whole reason the ablation tables are
readable, and this module must never quietly retarget one.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOCKED = REPO / "outputs" / "clot_ml" / "locked"

#: The locked pointer, written by `scripts/promote_clot_ml_0.py --repoint`.
POINTER = REPO / "data/reference/clot_gnn_locked.json"

# --- roles -------------------------------------------------------------------------------
#: The unified stack a caller almost always wants: wall GNN + wound complement + chemistry.
UNIFIED = "unified"
#: The wound-capable GNN underneath it (`temporal_v4_wound`) -- the baseline an ablation
#: scores against, because it is the same weights without the chemistry replacement.
WOUND = "wound"
#: The bare temporal GNN (`temporal_v4`) at the bottom of the chain.
BASE = "base"
ROLES = (UNIFIED, WOUND, BASE)

#: Kinds, as recorded on each manifest.  Used to verify the chain is the shape we think.
KIND_FOR_ROLE = {UNIFIED: "unified_v0", WOUND: "temporal_v4_wound", BASE: "temporal_v4"}

#: The compiled-in fallback, used only when the pointer is missing or unreadable.  It is a
#: NAME, not a directory: no generation has ever been stored under it.
DEFAULT_NAME = "clot_ml_0"
#: Older spellings of "whatever is shipped", kept so pre-2026-09 configs still resolve.
LEGACY_NAMES = frozenset({"clot_ml_v0", "clot_ml_0"})


def _manifest(name: str) -> dict:
    try:
        return json.loads((LOCKED / name / "manifest.json").read_text())
    except (OSError, ValueError):
        return {}


def pointer() -> dict:
    """The locked pointer as a dict, or ``{}`` when absent/unreadable.

    Never raises: an ordinary checkout with no promoted artifact must still import.
    """
    try:
        return json.loads(POINTER.read_text())
    except (OSError, ValueError):
        return {}


def pointer_name() -> str | None:
    """The ``unified_v0`` artifact the pointer names, if it names one that exists."""
    ptr = pointer()
    if ptr.get("kind") != KIND_FOR_ROLE[UNIFIED]:
        return None
    name = str(ptr.get("name") or "").strip()
    return name if name and (LOCKED / name).is_dir() else None


def shipped(role: str = UNIFIED) -> str:
    """Name of the currently promoted artifact for ``role``.

    Falls back to `DEFAULT_NAME` for `UNIFIED` when there is no usable pointer, and raises
    for the other roles -- a chain that cannot be walked is a broken promotion, and silently
    substituting a guess there is how the wrong model gets served for a whole campaign.
    """
    if role not in ROLES:
        raise ValueError(f"unknown artifact role {role!r}; expected one of {ROLES}")
    top = pointer_name()
    if top is None:
        if role == UNIFIED:
            return DEFAULT_NAME
        raise LookupError(
            f"no usable locked pointer at {POINTER}, so the {role!r} artifact cannot be "
            f"derived. Promote with scripts/promote_clot_ml_0.py --repoint, or pass an "
            f"explicit name.")
    if role == UNIFIED:
        return top
    wound = str((_manifest(top).get("v0") or {}).get("base_model")
                or _manifest(top).get("base_model") or "").strip()
    if not wound:
        raise LookupError(f"{top}'s manifest names no base_model; the chain is broken.")
    if role == WOUND:
        return wound
    base = str(_manifest(wound).get("base_model") or "").strip()
    if not base:
        raise LookupError(f"{wound}'s manifest names no base_model; the chain is broken.")
    return base


def resolve(name: str | None = None, role: str = UNIFIED) -> str:
    """Resolve a caller-supplied artifact id.

    ``None`` or a legacy alias means "whatever is shipped for this role"; anything else is
    returned verbatim so a pinned past generation is never retargeted.
    """
    n = (name or "").strip()
    if not n or n in LEGACY_NAMES:
        return shipped(role)
    return n


def root(name: str | None = None, role: str = UNIFIED) -> Path:
    """On-disk directory for an artifact, resolving roles and aliases first."""
    canonical = resolve(name, role)
    p = LOCKED / canonical
    if p.is_dir():
        return p
    # the pre-2026-09 stub, kept so a checkout with no promoted artifact still loads
    legacy = LOCKED / "clot_ml_v0"
    return legacy if canonical == DEFAULT_NAME and legacy.is_dir() else p


def describe() -> dict:
    """Every role and what it currently resolves to -- for logs and `--version` output."""
    out: dict[str, str | None] = {}
    for r in ROLES:
        try:
            out[r] = shipped(r)
        except LookupError:
            out[r] = None
    out["pointer"] = str(POINTER)
    return out
