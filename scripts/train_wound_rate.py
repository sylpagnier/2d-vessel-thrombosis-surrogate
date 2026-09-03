"""Fit the wound rate model, leave-one-vessel-out, and save the artifact.

The learned quantity is a **rate coefficient inside COMSOL's own surface ODE**, not a label:
``src/clot_ml/wound.py`` runs the ungated wound law with a two-regime gate and this script
fits ``(G_pre, G_post)`` -- globally, and optionally with a per-node network residual --
against GT ``Mat`` in log space over the whole trajectory.

Three arms, all evaluated leave-one-vessel-out on 3 wound vessels:

    physics    G == 1              COMSOL-faithful, zero parameters
    const      G_pre, G_post       two global scalars
    net        + WoundRateNet      per-node residual on both

At n=3 the honest expectation is that ``const`` is the arm that survives; ``net`` is fitted
and reported so the comparison exists rather than being assumed. Read the numbers, not the
architecture.

Usage:
    python scripts/train_wound_rate.py
    python scripts/train_wound_rate.py --epochs 400 --hidden 32
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

import sys

# Run directly (`python scripts/train_wound_rate.py`) needs the repo root importable.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.wound import (
    G_POST0, G_PRE0, RP_C0, WOUND_FEATURES, WoundRateNet, mat_trajectory_torch,
    onset_from_traj, prepare_vessel,
)
# The per-node wound ODE is a Python loop over ~80 time steps of SMALL tensors, run under
# autograd for every epoch, fold and arm.  Torch's default intra-op thread pool costs more in
# dispatch than those ops cost to execute -- measured 2026-09-02, the six-vessel fit made no
# progress in 75 minutes on 22 threads.  One thread is strictly faster here, and it also stops
# this fit from starving the GPU pipeline it runs beside.
torch.set_num_threads(1)

from src.biochem_gnn.wall_cohort_constants import WOUND_COHORT
from src.config import BiochemConfig

GRAPH_DIR = Path("data/processed/graphs_biochem_anchors")
OUT_DIR = Path("outputs/clot_ml/wound_rate")
#: Every wound simulation on disk.  Was a hardcoded three until 2026-09-02; the list now
#: lives with the rest of the cohort so a new wound run enters every fit that uses it.
WOUND_STEMS = WOUND_COHORT
EPS = 1e-3  # floor in Mat/crit units, so log10 is finite where nothing has deposited


def load_vessel(stem: str, bio, flow: str = "gt") -> dict:
    from src.core_physics.clot_phi_simple import mat_si_for_gelation_from_log1p

    data = torch.load(GRAPH_DIR / f"{stem}.pt", map_location="cpu", weights_only=False)
    if flow == "fem":
        # The deploy arm fits the wound rate against the flow the deploy stack will actually
        # see.  `prepare_vessel` reads `u0_pred`, so the solve has to land in the pack first.
        from src.clot_ml.v0 import solve_fem_into_pack
        if not str(getattr(data, "graph_stem", "") or ""):
            data.graph_stem = stem
        solve_fem_into_pack(data)
    V = prepare_vessel(data, bio, flow=flow)
    V["stem"] = stem
    V["data"] = data
    T = int(data.y.shape[0])
    mi = data.y_channel_names.split(",").index("Mat_log1p_nd")
    mat_gt = mat_si_for_gelation_from_log1p(data.y[:, :, mi], bio).reshape(T, -1)
    w = V["wound"]
    V["idx"] = np.flatnonzero(w)
    V["mat_gt_w"] = mat_gt[:, w].double()
    V["T"] = T
    return V


def subset(V: dict, wound_ap_closure: bool = True) -> dict:
    """The per-node ODE is uncoupled, so training only needs the wound rows.

    ``wound_ap_closure=False`` drops the wall-AP CONSUMPTION closure at the wound.  That
    closure (`src/core_physics/ap_closure.py`) is a Damkohler balance for a GATED wall
    reaction depleting activated platelets faster than shear renews them.  A wound deletes
    the gate and is a net platelet PRODUCER, and COMSOL says so: on `wound_patient003` GT
    `AP` ends at 10.3x its initial value where the closure predicts 0.96, and on
    `wound_patient006` the closure suppresses `AP` 5x (multiplier 0.188) where GT leaves it
    at 0.93 after the first interval.  Applying a depletion model to a source has the wrong
    sign, and it is the single largest error in the wound ODE -- see docs/DEPLOYCLOT.md 5c.
    """
    i = V["idx"]
    C = V["C"] if wound_ap_closure else dataclasses.replace(V["C"], ap_C=0.0)
    return dict(t=V["t"], rp=V["rp"][i], ap=V["ap"][i], sr=V["sr"][i], C=C)


def curve_loss(traj: torch.Tensor, mat_gt: torch.Tensor, crit: float) -> torch.Tensor:
    a = torch.log10(traj / crit + EPS)
    b = torch.log10(mat_gt / crit + EPS)
    return (a - b).abs().mean()


def fold_metrics(traj_np: np.ndarray, V: dict, crit: float) -> dict:
    gt = V["mat_gt_w"].numpy()
    T = V["T"]
    on_p, on_g = onset_from_traj(traj_np, crit), onset_from_traj(gt, crit)
    live = on_g < T
    a = np.log10(traj_np / crit + EPS)
    b = np.log10(gt / crit + EPS)
    return dict(
        curve_l1=float(np.abs(a - b).mean()),
        onset_err_med=float(np.median(on_p[live] - on_g[live])) if live.any() else float("nan"),
        onset_mae=float(np.abs(on_p[live] - on_g[live]).mean()) if live.any() else float("nan"),
        onset_mae_frac=float(np.abs(on_p[live] - on_g[live]).mean() / T) if live.any() else float("nan"),
        recall=float((traj_np[-1] >= crit).mean()),
        final_ratio=float(np.median(traj_np[-1]) / max(np.median(gt[-1]), 1e-30)),
        T=T,
    )


def train_fold(train_V: list, hidden: int, epochs: int, lr: float, mu, sd, crit: float,
               fit_rp_C: bool = False, wound_ap_closure: bool = True):
    net = WoundRateNet(len(WOUND_FEATURES), hidden=hidden, g_pre0=G_PRE0, g_post0=G_POST0,
                       fit_rp_C=fit_rp_C, rp_C0=RP_C0)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    packs = []
    for V in train_V:
        x = torch.tensor((V["feats"][V["idx"]] - mu) / sd, dtype=torch.float32)
        packs.append((x, subset(V, wound_ap_closure), V["mat_gt_w"]))
    for _ in range(epochs):
        opt.zero_grad()
        loss = 0.0
        for x, sub, gt in packs:
            g_pre, g_post = net(x)
            rc = None if net.log_rp_C is None else torch.exp(net.log_rp_C).double()
            traj = mat_trajectory_torch(gate_pre=g_pre.double(), gate_post=g_post.double(),
                                        rp_C=rc, **sub)
            loss = loss + curve_loss(traj, gt, crit)
        loss = loss / len(packs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
    return net, float(loss.detach())


@torch.no_grad()
def apply_net(net, V, mu, sd, wound_ap_closure: bool = True) -> np.ndarray:
    x = torch.tensor((V["feats"][V["idx"]] - mu) / sd, dtype=torch.float32)
    g_pre, g_post = net(x)
    rc = None if net.log_rp_C is None else torch.exp(net.log_rp_C).double()
    return mat_trajectory_torch(gate_pre=g_pre.double(), gate_post=g_post.double(),
                                rp_C=rc, **subset(V, wound_ap_closure)).numpy()


@torch.no_grad()
def flat_gate(V, g: float) -> np.ndarray:
    n = len(V["idx"])
    gg = torch.full((n,), float(g), dtype=torch.float64)
    return mat_trajectory_torch(gate_pre=gg, gate_post=gg, **subset(V)).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--flow", default="gt", choices=["gt", "pred", "fem"],
                    help="t=0 velocity the wound ODE's shear terms read.  `fem` is the "
                         "deploy-legal arm: the local Carreau solve, no COMSOL field.")
    ap.add_argument("--stems", nargs="*", default=None,
                    help="override the wound vessel list (default: WOUND_COHORT)")
    args = ap.parse_args()

    bio = BiochemConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    stems = args.stems or list(WOUND_STEMS)
    print("[i] loading %d wound vessels (flow=%s)" % (len(stems), args.flow))
    Vs = [load_vessel(s, bio, flow=args.flow) for s in stems]
    for V in Vs:
        print(f"    {V['stem']:20s} T={V['T']:4d} wound_nodes={len(V['idx']):4d}")

    # ARMS, in increasing capacity.  `const_rp` adds ONE scalar to `const` -- the
    # resting-platelet renewal coefficient of the same Damkohler balance the wall-AP closure
    # already applies to `ap` -- and recovers `const` continuously as that scalar goes to 0,
    # so this is a nested comparison, not two unrelated models.
    rows: dict[str, dict[str, dict]] = {"physics": {}, "const": {}, "const_rp": {}, "const_noapc": {}, "net": {}}
    folds: dict[str, dict] = {}
    for k, held in enumerate(Vs):
        train_V = [V for V in Vs if V is not held]
        feats = np.concatenate([V["feats"][V["idx"]] for V in train_V], axis=0)
        mu, sd = feats.mean(0), feats.std(0) + 1e-6

        const_net, l_const = train_fold(train_V, 0, args.epochs, args.lr, mu, sd, crit)
        crp_net, l_crp = train_fold(train_V, 0, args.epochs, args.lr, mu, sd, crit,
                                    fit_rp_C=True)
        napc_net, l_napc = train_fold(train_V, 0, args.epochs, args.lr, mu, sd, crit,
                                      wound_ap_closure=False)
        net, l_net = train_fold(train_V, args.hidden, args.epochs, args.lr, mu, sd, crit)

        rows["physics"][held["stem"]] = fold_metrics(flat_gate(held, 1.0), held, crit)
        rows["const"][held["stem"]] = fold_metrics(apply_net(const_net, held, mu, sd), held, crit)
        rows["const_rp"][held["stem"]] = fold_metrics(apply_net(crp_net, held, mu, sd), held, crit)
        rows["const_noapc"][held["stem"]] = fold_metrics(
            apply_net(napc_net, held, mu, sd, wound_ap_closure=False), held, crit)
        rows["net"][held["stem"]] = fold_metrics(apply_net(net, held, mu, sd), held, crit)

        gp = float(torch.exp(const_net.log_g_pre0)); gq = float(torch.exp(const_net.log_g_post0))
        gp2 = float(torch.exp(crp_net.log_g_pre0)); gq2 = float(torch.exp(crp_net.log_g_post0))
        print(f"\n[fold {k}] held out {held['stem']}   train loss "
              f"const={l_const:.4f} const_rp={l_crp:.4f} net={l_net:.4f}")
        print(f"    const    G_pre={gp:.2f} G_post={gq:.2f}")
        print(f"    const_rp G_pre={gp2:.2f} G_post={gq2:.2f} rp_C={crp_net.rp_C:.1f}")
        for arm in ("physics", "const", "const_rp", "const_noapc", "net"):
            m = rows[arm][held["stem"]]
            print(f"    {arm:8s} curveL1={m['curve_l1']:.3f}  onset_MAE={m['onset_mae']:6.1f} "
                  f"({m['onset_mae_frac']*100:5.1f}% of horizon)  recall={m['recall']:.3f} "
                  f"  final Mat ratio={m['final_ratio']:.2f}")
        folds[held["stem"]] = dict(g_pre=gp, g_post=gq,
                                   g_pre_rp=gp2, g_post_rp=gq2, rp_C=crp_net.rp_C,
                                   train_loss_const=l_const, train_loss_const_rp=l_crp,
                                   train_loss_net=l_net)

    print("\n" + "=" * 84)
    print(f"{'LEAVE-ONE-VESSEL-OUT MEAN':34s} {'curveL1':>9s} {'onset MAE':>11s} {'% horizon':>11s} {'recall':>8s}")
    summary = {}
    for arm in ("physics", "const", "const_rp", "const_noapc", "net"):
        cl = float(np.mean([m["curve_l1"] for m in rows[arm].values()]))
        om = float(np.mean([m["onset_mae"] for m in rows[arm].values()]))
        of = float(np.mean([m["onset_mae_frac"] for m in rows[arm].values()]))
        rc = float(np.mean([m["recall"] for m in rows[arm].values()]))
        summary[arm] = dict(curve_l1=cl, onset_mae=om, onset_mae_frac=of, recall=rc)
        print(f"{arm:34s} {cl:9.3f} {om:11.1f} {of*100:10.1f}% {rc:8.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # The deploy artifact refits the constants on ALL THREE vessels; the LOVO table above is
    # what says whether that is worth anything, and is reported next to it.
    feats = np.concatenate([V["feats"][V["idx"]] for V in Vs], axis=0)
    mu, sd = feats.mean(0), feats.std(0) + 1e-6
    final_const, _ = train_fold(Vs, 0, args.epochs, args.lr, mu, sd, crit)
    final_crp, _ = train_fold(Vs, 0, args.epochs, args.lr, mu, sd, crit, fit_rp_C=True)
    final_napc, _ = train_fold(Vs, 0, args.epochs, args.lr, mu, sd, crit,
                               wound_ap_closure=False)
    final_net, _ = train_fold(Vs, args.hidden, args.epochs, args.lr, mu, sd, crit)
    torch.save(dict(const=final_const.state_dict(), const_rp=final_crp.state_dict(),
                    const_noapc=final_napc.state_dict(), net=final_net.state_dict(),
                    mu=mu, sd=sd, hidden=args.hidden, features=list(WOUND_FEATURES)),
               out / "wound_rate.pt")
    (out / "lovo.json").write_text(json.dumps(
        dict(summary=summary, per_vessel=rows, folds=folds,
             fitted_all=dict(g_pre=float(torch.exp(final_const.log_g_pre0)),
                             g_post=float(torch.exp(final_const.log_g_post0))),
             fitted_all_rp=dict(g_pre=float(torch.exp(final_crp.log_g_pre0)),
                                g_post=float(torch.exp(final_crp.log_g_post0)),
                                rp_C=final_crp.rp_C),
             fitted_all_noapc=dict(g_pre=float(torch.exp(final_napc.log_g_pre0)),
                                   g_post=float(torch.exp(final_napc.log_g_post0)),
                                   wound_ap_closure=False),
             epochs=args.epochs, hidden=args.hidden, n_vessels=len(Vs),
             flow=args.flow, stems=list(stems)), indent=2))
    print(f"\n[save] {out/'wound_rate.pt'}")
    print(f"[save] {out/'lovo.json'}")
    print(f"[i] refit on all {len(Vs)}: const   G_pre={float(torch.exp(final_const.log_g_pre0)):.2f} "
          f"G_post={float(torch.exp(final_const.log_g_post0)):.2f}")
    print(f"[i] refit on all {len(Vs)}: const_noapc G_pre={float(torch.exp(final_napc.log_g_pre0)):.2f} "
          f"G_post={float(torch.exp(final_napc.log_g_post0)):.2f} (wall-AP closure OFF at the wound)")
    print(f"[i] refit on all {len(Vs)}: const_rp G_pre={float(torch.exp(final_crp.log_g_pre0)):.2f} "
          f"G_post={float(torch.exp(final_crp.log_g_post0)):.2f} rp_C={final_crp.rp_C:.1f}")


if __name__ == "__main__":
    main()
