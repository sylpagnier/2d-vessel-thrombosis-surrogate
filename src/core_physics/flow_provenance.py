"""Which t=0 velocity field was in the packs when an artifact was measured.

`pack.u0_pred` is ONE slot that every reconstructed flow source writes, so its contents are a
MODE the packs are currently in, not a property of the packs.  Two legitimate consumers want
different things in it at the same time:

    a full-pool checkpoint                the same checkpoint (`E5_band_gateup`) for every vessel
    leak-free flow diagnostics            per vessel, the cross-fit fold that never saw it

Both cannot hold, so the slot gets rewritten whenever the other measurement is wanted -- and
nothing downstream recorded which mode produced it.  `outputs/runs/flow_diagnostics.json` was
a bare list of per-vessel rows with no checkpoint field anywhere, so on 2026-09-06 a full-pool
precache silently invalidated it and the file still looked current.  A number whose inputs
cannot be identified is not reproducible, however carefully it was computed.

So: read the stamp `precache_rgp_deq.py` already writes onto each pack, summarise it across a
cohort, and put that summary INTO the artifact.  Consumers that care can then assert it
instead of assuming.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

#: `u0_pred` holds a solved field with no checkpoint behind it.
SOLVED = "fem-solve"
#: Nothing has written the slot.
ABSENT = "absent"

#: Cohort-level modes `classify` can report.
FULL_POOL = "full-pool"
CROSS_FIT = "cross-fit"
MIXED = "mixed"


def pack_stamp(data: Any) -> dict:
    """The flow provenance of one pack, as `{run, checkpoint}`.

    `run` is the directory the weights came from, which is the part that identifies a
    cross-fit fold -- every arm writes the same basename, so the basename alone cannot tell
    `E8_xf5_f3` from the full-pool arm.  That is exactly the confusion this module exists for.
    """
    raw = getattr(data, "u0_pred_provenance", None)
    if not raw:
        return {"run": ABSENT, "checkpoint": ABSENT}
    try:
        path = Path(json.loads(raw)["checkpoint"]["path"])
    except (ValueError, KeyError, TypeError):
        return {"run": "(unreadable)", "checkpoint": "(unreadable)"}
    return {"run": path.parent.name, "checkpoint": path.name}


def is_fold_run(run: str) -> bool:
    """True for a cross-fit fold arm (`E8_xf5_f3`), false for a full-pool one.

    A single vessel cannot be classified by `classify` -- one run is one run -- so anything
    asking "is THIS pack's field a held-out fold's?" needs this instead.
    """
    r = str(run)
    return "_f" in r and r.rsplit("_f", 1)[-1].isdigit()


def classify(runs: Iterable[str]) -> str:
    """Name the cohort-level mode from the set of run directories seen.

    A cross-fit cohort is identifiable by construction: several `*_f<i>` fold arms appear,
    because each vessel is served by the one that held it out.  Anything else with more than
    one non-fold run in it is `mixed`, which is a state no measurement should be quoted from
    without saying so.
    """
    seen = {r for r in runs if r not in (ABSENT,)}
    if not seen:
        return ABSENT
    # A solved field has no arm behind it, so it is its own mode.  Calling a cohort of FEM
    # solves "full-pool" would name a training pool that had no part in producing it.
    if seen == {SOLVED}:
        return SOLVED
    folds = {r for r in seen if is_fold_run(r)}
    if len(folds) > 1:
        return CROSS_FIT
    if len(seen) == 1:
        return FULL_POOL
    return MIXED


def cohort_stamp(stems: Iterable[str], packs_dir: Path) -> dict:
    """Summarise the flow mode across a cohort, cheaply.

    Reads only the provenance attribute, never the tensors: loading 50 packs to answer a
    metadata question would make stamping cost more than the measurement it annotates.
    """
    import torch

    per_vessel: dict[str, dict] = {}
    for stem in sorted(set(stems)):
        p = Path(packs_dir) / f"{stem}.pt"
        if not p.is_file():
            continue
        try:
            d = torch.load(p, map_location="meta", weights_only=False)
        except Exception:                      # noqa: BLE001 - a pack we cannot read is data
            try:
                d = torch.load(p, map_location="cpu", weights_only=False)
            except Exception:                  # noqa: BLE001
                per_vessel[stem] = {"run": "(unreadable)", "checkpoint": "(unreadable)"}
                continue
        per_vessel[stem] = pack_stamp(d)

    runs = [v["run"] for v in per_vessel.values()]
    return {
        "mode": classify(runs),
        "n_vessels": len(per_vessel),
        "runs": dict(sorted(Counter(runs).items())),
        "per_vessel": per_vessel,
        "note": ("`u0_pred` is one slot every reconstructed flow source writes; this records "
                 "which weights were resident when the artifact was measured, so the artifact "
                 "can be told apart from one measured in a different mode."),
    }


def read_diagnostics(path: Path) -> tuple[list[dict], dict]:
    """Load a flow-diagnostics file, old shape or new, as ``(rows, provenance)``.

    Files written before 2026-09-06 are a bare list of per-vessel rows with no provenance at
    all; they load with an empty stamp rather than failing, because the honest answer for
    those really is "unknown", and a reader that crashed on them would just get bypassed.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, {"mode": ABSENT, "note": "file predates flow provenance stamping"}
    return list(payload.get("vessels") or []), dict(payload.get("provenance") or {})


__all__ = ["SOLVED", "ABSENT", "FULL_POOL", "CROSS_FIT", "MIXED",
           "pack_stamp", "is_fold_run", "classify", "cohort_stamp", "read_diagnostics"]
