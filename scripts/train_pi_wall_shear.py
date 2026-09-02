"""STEPS 3-5 -- fit the physics baseline, add the bounded residual, score LOVO.

Reports three arms on the SAME held-out vessels, so the learned term is credited only with
what it adds over the physics:

  null       sr_pred = sr0                    do nothing
  prior      A(dmu) only                      Tier 1, the shipped analytic attenuation
  physics    A(dmu) * (h0/h)^p, p fitted      STEP 3 -- no learning
  full       + bounded eps_theta              STEP 4 -- LOVO, vessel held out entirely

The score is mean |error| in `sr/sr0`, the deposition gate's own quantity, matched to the
Tier 1.5 harness so the numbers are directly comparable with
`outputs/diag_corrector_severe_occlusion.json` (null 0.630, prior 0.327).  Also reported in
log space, where the model is actually fitted, and as wall-shear correlation -- the thing the
existing arms fail at (both were weakly or negatively correlated with FEM).

LOVO is by VESSEL, never by case: cases from one vessel share its geometry, so splitting on
cases would leak and the reported gain would be fictional.

    python scripts/train_pi_wall_shear.py --corpus outputs/pi_corpus
    python scripts/train_pi_wall_shear.py --corpus outputs/pi_corpus --epochs 400
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core_physics.wall_shear_attenuation import DELTA_MU_HALF_SI  # noqa: E402
from src.core_physics.pi_wall_shear import (  # noqa: E402
    FEATURE_NAMES, P_FLUX_INIT, PIWallShear, assemble_features, hydraulic_h,
    physics_log_ratio)

KEYS = ("sr0", "sr_fem", "delta_mu", "h_over_h0", "s_signed",
        "width_nd", "width_d1", "width_d2", "sdf_nd", "in_clot", "case_id")


def load_corpus(corpus_dir: Path, *, min_sr0: float = 1e-2, max_abs_log: float = 4.0,
                scope: str = "clot") -> dict:
    """Load the corpus and restrict it to the rows the gate decision actually turns on.

    TWO FILTERS, both load-bearing, both chosen before any score was read:

    * ``scope="clot"`` keeps only wall nodes UNDER the occlusion.  Most wall nodes in a vessel
      are far from the clot and read `sr/sr0 = 1` exactly -- the corpus median ratio is 1.000 --
      so scoring over the whole wall would let every arm (including "do nothing") look accurate
      by predicting no change almost everywhere.  The Tier 1.5 harness scored on `occ & wall`
      for the same reason, and matching it is what makes these numbers comparable to its
      null 0.630 / prior 0.327.  ``scope="all"`` reports the whole wall instead.
    * A node whose clot-free shear is ~0 carries no ratio information and its denominator would
      dominate a log target -- patient005 produces a ratio of 1053 that way.  ``min_sr0`` and
      ``max_abs_log`` drop those rather than let Huber quietly absorb them.
    """
    if scope not in ("clot", "all"):
        raise ValueError("scope must be 'clot' or 'all'")
    out = {}
    for f in sorted(corpus_dir.glob("*.npz")):
        z = np.load(f)
        d = {k: z[k].astype(np.float64) for k in KEYS if k in z}
        keep = (d["sr0"] > min_sr0) & (d["sr_fem"] > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log(np.where(keep, d["sr_fem"] / np.maximum(d["sr0"], 1e-12), 1.0))
        keep &= np.abs(lr) <= max_abs_log
        if scope == "clot":
            keep &= d["in_clot"] > 0.5
        d = {k: v[keep] for k, v in d.items()}
        if len(d["sr0"]) >= 50:
            out[f.stem] = d
    return out


def fit_p(data: dict, half: float = DELTA_MU_HALF_SI, *, hydraulic: bool = True) -> float:
    """STEP 3: least squares for the flux exponent in LOG space -- closed form, no optimiser.

    log(sr/sr0) + log1p(dmu/half) = -p * log(h_use)  =>  p = -<a,b>/<b,b> with b = log(h_use).

    ``hydraulic`` selects the effective lumen (only the SOLID fraction of the clot blocks flux)
    over the raw geometric one.      Both are fitted and reported; see `python -m src.tools.diagnostics pi-flux-interaction` for why a
    constant exponent on geometric `h` is wrong.
    """
    y, b = [], []
    for d in data.values():
        h = (hydraulic_h(d["delta_mu"], d["h_over_h0"], delta_mu_half=half) if hydraulic
             else np.clip(d["h_over_h0"], 1e-3, 1.0))
        y.append(np.log(d["sr_fem"] / d["sr0"]) + np.log1p(d["delta_mu"] / half))
        b.append(np.log(np.clip(h, 1e-9, 1.0)))
    y, b = np.concatenate(y), np.concatenate(b)
    denom = float(b @ b)
    return float(-(y @ b) / denom) if denom > 1e-12 else P_FLUX_INIT


def arm_scores(d: dict, log_ratio_pred: np.ndarray) -> dict:
    """Three scores, because no single one of them is honest on its own.

    ``mae_log``  -- PER NODE, in the space the model is fitted in.  This is the headline: the
                    ratio has a heavy right tail (nodes at `sr/sr0` ~ 50 exist inside the clot
                    scope), and a mean absolute error in RATIO space is set by that tail rather
                    than by typical behaviour.  Log is scale-symmetric: halving and doubling
                    cost the same, which is what "wrong by 2x" should mean.
    ``mae_ratio_case`` -- PER CASE, on the median ratio in the clot.  This is the ONLY column
                    directly comparable with the Tier 1.5 harness (null 0.630, prior 0.327),
                    because that harness scored per-case medians, not per-node values.  Quoting
                    the per-node number against those is an error -- they are different
                    statistics of different distributions.
    ``corr_log`` -- does the arm rank nodes correctly?  Both existing arms failed here (weak or
                    negative against FEM), so this is where a real improvement must show.
    """
    ratio_true = d["sr_fem"] / d["sr0"]
    ratio_pred = np.exp(log_ratio_pred)
    lt = np.log(ratio_true)
    per_case = []
    cid = d.get("case_id")
    if cid is not None:
        for c in np.unique(cid):
            m = cid == c
            per_case.append(abs(float(np.median(ratio_pred[m])) - float(np.median(ratio_true[m]))))
    return dict(
        mae_log=float(np.mean(np.abs(log_ratio_pred - lt))),
        mae_ratio_case=float(np.mean(per_case)) if per_case else float("nan"),
        mae_ratio_node=float(np.mean(np.abs(ratio_pred - ratio_true))),
        corr_log=float(np.corrcoef(log_ratio_pred, lt)[0, 1]) if lt.std() > 1e-12 else float("nan"),
        corr_sr=float(np.corrcoef(d["sr0"] * ratio_pred, d["sr_fem"])[0, 1]),
        n=int(len(lt)), n_cases=len(per_case),
    )


def train_one(train: dict, *, epochs: int, lr: float, eps_max: float,
              hidden: int, seed: int, hydraulic: bool = True) -> PIWallShear:
    torch.manual_seed(seed)
    X = np.concatenate([assemble_features(d) for d in train.values()])
    dmu = np.concatenate([d["delta_mu"] for d in train.values()])
    h = np.concatenate([d["h_over_h0"] for d in train.values()])
    y = np.concatenate([np.log(d["sr_fem"] / d["sr0"]) for d in train.values()])

    m = PIWallShear(n_features=X.shape[1], hidden=hidden, eps_max=eps_max,
                    hydraulic=hydraulic)
    m.set_normalizer(X)
    Xt = torch.tensor(X, dtype=torch.float32)
    dt = torch.tensor(dmu, dtype=torch.float32)
    ht = torch.tensor(h, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)

    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad()
        # Huber in LOG space: the target spans 0.03-1.6 multiplicatively, and a handful of
        # near-stagnant nodes would otherwise set the whole gradient.
        loss = torch.nn.functional.huber_loss(m(Xt, dt, ht), yt, delta=0.5)
        loss.backward()
        opt.step()
        sched.step()
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="outputs/pi_corpus")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--eps-max", type=float, default=0.4)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scope", default="clot", choices=("clot", "all"),
                    help="'clot' = wall nodes under the occlusion (comparable to Tier 1.5)")
    ap.add_argument("--min-sr0", type=float, default=1e-2)
    ap.add_argument("--out", default="outputs/pi_wall_shear_lovo.json")
    args = ap.parse_args()

    data = load_corpus(REPO / args.corpus, min_sr0=args.min_sr0, scope=args.scope)
    if len(data) < 3:
        print(f"[ERR] need >=3 vessels in {args.corpus}, found {len(data)}")
        return 1
    print(f"[i] {len(data)} vessels, "
          f"{sum(len(d['sr0']) for d in data.values())} wall rows (scope={args.scope}), "
          f"{len(FEATURE_NAMES)} residual features")

    # ---- STEP 3: the physics baseline, fitted globally (no learning) -------------------
    p_geom = fit_p(data, hydraulic=False)
    p_hyd = fit_p(data, hydraulic=True)
    print("")
    print(f"STEP 3  flux exponent fitted on all vessels   (Poiseuille = {P_FLUX_INIT:.1f})")
    print(f"        geometric h : p = {p_geom:.3f}")
    print(f"        hydraulic h : p = {p_hyd:.3f}   <- only the SOLID fraction blocks flux")

    # ---- STEPS 4-5: LOVO ---------------------------------------------------------------
    rows = []
    for held in sorted(data):
        train = {k: v for k, v in data.items() if k != held}
        d = data[held]
        # both exponents fitted WITHOUT the held-out vessel
        p_g = fit_p(train, hydraulic=False)
        p_h = fit_p(train, hydraulic=True)
        zeros = np.zeros_like(d["sr0"])
        ones = np.ones_like(d["h_over_h0"])
        arms = {
            "null": arm_scores(d, zeros),
            "prior": arm_scores(d, physics_log_ratio(d["delta_mu"], ones, hydraulic=False)),
            "phys_geom": arm_scores(d, physics_log_ratio(d["delta_mu"], d["h_over_h0"],
                                                         p=p_g, hydraulic=False)),
            "phys_hyd": arm_scores(d, physics_log_ratio(d["delta_mu"], d["h_over_h0"],
                                                        p=p_h, hydraulic=True)),
        }
        m = train_one(train, epochs=args.epochs, lr=args.lr, eps_max=args.eps_max,
                      hidden=args.hidden, seed=args.seed, hydraulic=True)
        p_tr = p_h
        X = torch.tensor(assemble_features(d), dtype=torch.float32)
        with torch.no_grad():
            lr_full = m(X, torch.tensor(d["delta_mu"], dtype=torch.float32),
                        torch.tensor(d["h_over_h0"], dtype=torch.float32)).numpy()
        arms["full"] = arm_scores(d, lr_full.astype(np.float64))
        rows.append(dict(held=held, p_train=p_tr, p_learned=m.p,
                         p_geom=p_g, half_learned=m.delta_mu_half, arms=arms))
        print(f"  [{held:20s}] p_tr={p_tr:5.2f} p_lrn={m.p:5.2f} "
              f"half={m.delta_mu_half:.4f}  "
              + "  ".join(f"{k} {arms[k]['mae_log']:.3f}" for k in
                          ("null", "prior", "phys_geom", "phys_hyd", "full")), flush=True)

    print("\nLOVO MEAN (held-out vessels only)")
    print(f"{'arm':>9} {'MAE log':>9} {'MAEratio/case':>14} {'corr log':>9} {'corr sr':>9}")
    summary = {}
    for arm in ("null", "prior", "phys_geom", "phys_hyd", "full"):
        s = {k: float(np.nanmean([r["arms"][arm][k] for r in rows]))
             for k in ("mae_log", "mae_ratio_case", "mae_ratio_node", "corr_log", "corr_sr")}
        summary[arm] = s
        print(f"{arm:>9} {s['mae_log']:>9.4f} {s['mae_ratio_case']:>14.4f} "
              f"{s['corr_log']:>9.3f} {s['corr_sr']:>9.3f}")

    win_ph = sum(r["arms"]["phys_hyd"]["mae_log"] < r["arms"]["prior"]["mae_log"] for r in rows)
    win_hg = sum(r["arms"]["phys_hyd"]["mae_log"] < r["arms"]["phys_geom"]["mae_log"] for r in rows)
    win_fu = sum(r["arms"]["full"]["mae_log"] < r["arms"]["phys_hyd"]["mae_log"] for r in rows)
    n = len(rows)
    print(f"\nphysics beats prior on {win_ph}/{n} held-out vessels")
    print(f"full    beats physics on {win_fu}/{n} held-out vessels")
    print(f"learned p across folds: {np.mean([r['p_learned'] for r in rows]):.3f} "
          f"+/- {np.std([r['p_learned'] for r in rows]):.3f}  (init {P_FLUX_INIT})")

    out = dict(p_all=p_hyd, p_all_geom=p_geom, n_vessels=len(data), features=list(FEATURE_NAMES),
               eps_max=args.eps_max, epochs=args.epochs, scope=args.scope,
               summary=summary, folds=rows)
    Path(REPO / args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(REPO / args.out).write_text(json.dumps(out, indent=2))
    print(f"[save] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
