"""The strict-CV readout families, and the per-vessel cut adaptation.

These are library functions: the locked ``clot_ml_0`` artifact reads its committed
set through them (``src.clot_ml.locked._committed_set_v4``), so a promoted model
cannot be scored without them.  They lived in ``scripts/eval_strict.py``, which
meant the library imported from a script -- the dependency ran backwards, and the
script could not be treated as an entry point.

``scripts/eval_strict.py`` now imports them from here and keeps its tuners, which
are genuinely script-level: they search for the cuts, whereas these apply them.

Distinct from :mod:`src.clot_ml.readouts`, which is the newer registry
(``thresh`` / ``expected`` / ``topk`` / ``blend``) used by the deploy readout
selection.  The two coexist on purpose: promoted artifacts reference families by
name, so renaming or merging them would retarget models already locked.
"""

from __future__ import annotations

import numpy as np

#: Which per-vessel statistic the adaptive cut is a function of.
ADAPT_STAT = "mean"


def readout_plain(S, sc, th):
    """One cut per domain."""
    tw, to = th
    w = S["wall"]
    return (w & (sc >= tw)) | (~w & (sc >= to))


def readout_resid(S, sc, th):
    """Separate cuts for keeping a physics-positive node and adding a physics-negative one.

    Wall error is two opposite failure modes (PHASE7 10.3: weak-separation false positives
    on 018/019/025 against ungated false negatives on 012/028) and one cut cannot serve
    both.  This is the readout ``scripts/train_clot_gnn.py`` already uses.
    """
    kw, aw, ko, ao = th
    w, ph = S["wall"], S["phys_mask"]
    return ((w & ph & (sc >= kw)) | (w & ~ph & (sc >= aw))
            | (~w & ph & (sc >= ko)) | (~w & ~ph & (sc >= ao)))


#: family name -> the function that turns thresholds into a mask.
READOUTS = {"plain": readout_plain, "resid": readout_resid}

#: free scalars per domain, per family -- ``resid`` has twice ``plain``'s.
N_PARAMS = {"plain": 1, "resid": 2}


def vessel_stat(S, sc, dom, name: str = ADAPT_STAT) -> float:
    d = np.asarray(dom, dtype=bool)
    v = np.asarray(sc, dtype=np.float64)[d]
    if v.size == 0:
        return 0.0
    if name == "mean":
        return float(v.mean())
    if name == "q90":
        return float(np.quantile(v, 0.90))
    if name == "physfrac":
        return float((S["phys_mask"] & d).sum() / max(d.sum(), 1))
    raise ValueError(name)


def apply_adapt(S, sc, family, th, dom_of, b, med, lo=None, hi=None):
    """Perturb the cohort cut by the fitted slope on this vessel's own statistic.

    ``lo``/``hi`` bound the statistic to the support `tune_adapt` fitted over.  Inside that
    support clamping is an EXACT no-op -- verified bit-identical on all 19 pool vessels,
    under both the geometry-stratified partition and one that deliberately holds the
    extremes out -- so it cannot flatter anything the cohort measures.  Outside it, the cut
    is held at the most extreme perturbation the fit actually saw rather than continuing a
    line no labelled vessel supports.  Both default to ``None``, which reproduces the
    unbounded behaviour exactly.
    """
    stat = vessel_stat(S, sc, dom_of(S))
    if lo is not None and hi is not None:
        stat = float(min(max(stat, float(lo)), float(hi)))
    off = b * (stat - med)
    return READOUTS[family](S, sc, tuple(np.clip(np.array(th) + off, 0.02, 0.98)))
