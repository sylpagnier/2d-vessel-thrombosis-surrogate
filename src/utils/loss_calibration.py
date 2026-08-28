"""Set Stage-A loss weights by measured gradient share, not by sweeping.

**Why not a sweep.**  There are eight weights and each configuration costs a training run, so a
grid is out of reach and a random search would spend its budget learning what a single forward
pass can measure.  The weights are also doing two unrelated jobs at once -- unit conversion
between terms on wildly different scales, and expression of priority -- and a sweep searches
that tangle rather than untangling it.

**Why not the Kendall weighter.**  `DynamicLossWeighter` is already in the tree and already
disabled: "avoids negative weighted PDE collapse" (`train_kinematics_predictor.py`).
Homoscedastic uncertainty weighting assumes every term is a log-likelihood of the same data; a
PDE residual is not, and the learned precisions run away.

**What this does instead.**  The quantity that matters is not a term's *value* but the size of
the gradient it puts on the parameters.  For each term `L_i` measured separately on a fixed
batch:

    g_i = || dL_i / d(theta) ||        w_i = share_i / g_i

so that after weighting every term contributes its intended fraction of the total gradient
norm.  The weights then stop being unit conversions and the only remaining judgement is
`share` -- a short, readable statement of priority that someone can disagree with on the
merits.

One forward/backward per term per graph; no training runs.  Re-runnable mid-training if the
balance drifts, though the point is to set it once from a sane reference state.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import torch

#: Intended share of the total gradient norm.  These are priorities, not scales -- argue with
#: them directly.  Ordering follows the one measurement that predicts the downstream outcome:
#: gate union Jaccard tracks the clot model's oracle-F1 at +0.918, so the gate term leads
#: (RGP_DEQ_REPAIR_PLAN.md s10.3).
DEFAULT_SHARES: Dict[str, float] = {
    "l_band_gate": 0.30,     # the metric that predicts the clot outcome
    "l_data_kine": 0.22,     # supervised velocity -- the model must still fit the field
    "l_band_sr": 0.08,       # wall-band shear magnitude
    "l_band_dsrx": 0.08,     # wall-band shear gradient (what the gate is built from)
    "l_cont": 0.10,          # continuity: a real constraint, but it is not the objective
    "l_prior_floor": 0.07,   # never be worse than the analytic prior
    "l_wss": 0.05,
    "l_shear_grad": 0.04,
    "l_io": 0.03,
    "l_bc": 0.02,
    "l_mom": 0.01,
}


def measure_gradient_norms(
    model,
    graphs: Iterable,
    kernels,
    device,
    *,
    terms: Iterable[str] | None = None,
    stage: int = 3,
    carreau_n: float = 0.6,
) -> Dict[str, float]:
    """Mean ``|| dL_i / d(theta) ||`` per term over ``graphs``, each measured in isolation."""
    from src.utils.kinematics_physics_terms import compute_kinematics_physics_terms

    names = list(terms) if terms is not None else list(DEFAULT_SHARES)
    acc: Dict[str, list] = {n: [] for n in names}
    params = [p for p in model.parameters() if p.requires_grad]

    for g in graphs:
        gg = g.clone().to(device)
        out = model(gg, solver="anderson")
        pred = out[0] if isinstance(out, tuple) else out
        vals = compute_kinematics_physics_terms(pred, gg, kernels, carreau_n=carreau_n)
        for name in names:
            t = vals.get(name)
            if t is None or not torch.is_tensor(t) or not t.requires_grad:
                continue
            grads = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
            n2 = sum(float((x.detach() ** 2).sum()) for x in grads if x is not None)
            if n2 > 0:
                acc[name].append(n2**0.5)
        del gg, pred, vals

    # MEDIAN, not mean.  Derivative-of-derivative terms are heavy-tailed across graphs -- on one
    # 6-graph run `l_shear_grad` averaged 3.0e+09 against 7.0e+03 on a 4-graph run, because a
    # single vessel dominated. A mean would let that one graph set every weight in the recipe.
    import statistics

    out: Dict[str, float] = {}
    spread: Dict[str, float] = {}
    for n, v in acc.items():
        if not v:
            out[n] = float("nan")
            continue
        out[n] = float(statistics.median(v))
        spread[n] = float(max(v) / max(min(v), 1e-30))
    out["_spread"] = spread
    return out


def weights_from_gradient_norms(
    norms: Mapping[str, float],
    shares: Mapping[str, float] | None = None,
    *,
    reference: str = "l_data_kine",
    inert_rel: float = 1e-4,
) -> Dict[str, float]:
    """Turn measured gradient norms into weights that realise ``shares``.

    Normalised so ``reference`` keeps weight 1.0 -- the absolute scale is absorbed by the
    learning rate, and pinning one term keeps the numbers readable against the old recipe.
    """
    sh = dict(shares or DEFAULT_SHARES)
    # Anchor "inert" to the REFERENCE term, not to the maximum.  Anchoring to the max lets a
    # single heavy-tailed term (`l_shear_grad` again) set the threshold and silently drop every
    # other term -- observed, and it produced a recipe containing exactly one loss.
    ref_norm = norms.get(reference, float("nan"))
    if ref_norm is None or ref_norm != ref_norm or ref_norm <= 0.0:
        finite = [float(v) for k, v in norms.items()
                  if k != "_spread" and v is not None and v == v and v > 0.0]
        if not finite:
            return {}
        ref_norm = sorted(finite)[len(finite) // 2]
    cutoff = float(ref_norm) * float(inert_rel)

    raw: Dict[str, float] = {}
    for name, share in sh.items():
        g = norms.get(name, float("nan"))
        if g is None or g != g or g <= 0.0:
            continue
        if g < cutoff:
            # STRUCTURALLY INERT -- do not amplify it.  `l_bc` measures 3.8e-09 against a
            # cohort max of ~7e+03 because the hard BC `u = uv_prior + sdf * uvp` satisfies the
            # boundary condition by construction; the term has nothing left to say.  Solving
            # `w = share / g` for such a term asks for a weight of ~1e+07, which would turn
            # numerical noise into the dominant gradient.  Dropping it is the correct answer:
            # a term with no gradient cannot be given influence, only amplified noise.
            continue
        raw[name] = float(share) / g
    if not raw:
        return {}
    ref = raw.get(reference) or max(raw.values())
    return {k: v / ref for k, v in raw.items()}


__all__ = ["DEFAULT_SHARES", "measure_gradient_norms", "weights_from_gradient_norms"]
