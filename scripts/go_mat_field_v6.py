"""v6 -- build, train and evaluate the LEARNED surface ``Mat`` field.

Rationale and the measurements that force this design are in :mod:`src.clot_ml.mat_field`
and docs/WOUND_PROGRESS.md 16/17.  In one line: the off-wall architecture is correct and the
ODE's ``Mat`` is the single broken component, so v6 replaces that component and changes
nothing else.

    python scripts/go_mat_field_v6.py cache                     # ~20 min, once
    python scripts/go_mat_field_v6.py train --epochs 60         # GPU
    python scripts/go_mat_field_v6.py eval                      # wound vessels + cohort

The holdout is enforced here rather than inside the module so it is visible at the call site:
``wound_patient003`` and the four SEALED vessels are never in ``train``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.mat_field import (  # noqa: E402
    EXTRA_CHANNELS, OFF_ATT, MatFieldConfig, build_mat_field_entry, build_static_graph,
    crossing_target, extra_channels, make_model,
)
from src.config import BiochemConfig  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
CACHE = REPO / "outputs/mat_field_cache"
CKPT = REPO / "outputs/clot_ml/mat_field_v6"
TARGET = "wound_patient003"
# SEALED (WALL_COHORT_V2_GENERALIZATION) -- spent once, never tuned against.
SEALED = ("patient007", "patient013", "patient031", "patient043")


def cache_stems() -> list[str]:
    out = []
    for p in sorted(PACKS.glob("*.pt")):
        if p.name.endswith(".prenormalfix"):
            continue
        out.append(p.stem)
    return out


def train_stems(all_stems: list[str], extra_holdout=()) -> list[str]:
    """Everything legal: all packs minus SEALED, the target vessel, and any extra holdout.

    A holdout drops its MIRROR too (``patient005_mirror_y`` is the same vessel reflected), or
    the "held-out" score would be read off a vessel the model trained on.
    """
    drop = {TARGET, *SEALED}
    for s in extra_holdout:
        drop.add(s)
        drop.add(f"{s}_mirror_y")
        if s.endswith("_mirror_y"):
            drop.add(s[: -len("_mirror_y")])
    return [s for s in all_stems if s not in drop]


# ---------------------------------------------------------------------------
def cmd_cache(args) -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    bio = BiochemConfig(phase="biochem")
    from src.clot_ml.locked import load_temporal_v4_wound

    w = load_temporal_v4_wound(name=args.artifact)["wound"]
    wr = (float(w["g_pre"]), float(w["g_post"]))
    stems = args.stems or cache_stems()
    print(f"[i] caching {len(stems)} packs -> {CACHE}  (wound_rate={wr})", flush=True)
    for i, stem in enumerate(stems, 1):
        out = CACHE / f"{stem}.npz"
        if out.exists() and not args.force:
            print(f"  [skip] {stem}", flush=True)
            continue
        t0 = time.time()
        try:
            data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
            e = build_mat_field_entry(data, bio, flow=args.flow,
                                      n_times=args.n_times, wound_rate=wr)
            np.savez_compressed(out, **e)
            print(f"  [{i}/{len(stems)}] {stem}  N={e['X'].shape[0]} "
                  f"K={len(e['t_idx'])}  {time.time() - t0:.0f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {stem}: {type(exc).__name__}: {exc}", flush=True)
    return 0


def load_entries(stems) -> dict:
    E = {}
    for s in stems:
        p = CACHE / f"{s}.npz"
        if not p.exists():
            continue
        with np.load(p, allow_pickle=False) as z:
            E[s] = {k: z[k] for k in z.files}
    return E


# ---------------------------------------------------------------------------
def cmd_train(args) -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        print("[WARN] no CUDA; this will be slow", flush=True)
    cfg = MatFieldConfig(dim=args.dim, layers=args.layers, epochs=args.epochs,
                         lr=args.lr, cls_w=args.cls_w, reg_w=args.reg_w,
                         pos_weight=args.pos_weight, reg_mag_w=args.reg_mag_w,
                         clot_free_w=args.clot_free_w, seed=args.seed)
    stems = train_stems(cache_stems(), args.holdout)
    E = load_entries(stems)
    if not E:
        print("[ERR] cache empty -- run `cache` first", flush=True)
        return 1
    held = sorted({TARGET, *SEALED, *args.holdout})
    print(f"[i] train on {len(E)} vessels, dev={dev}, seed={cfg.seed}\n"
          f"    held out: {', '.join(held)}", flush=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    keys = sorted(E)
    mu = np.concatenate([E[a]["X"] for a in keys]).mean(0)
    sd = np.concatenate([E[a]["X"] for a in keys]).std(0)
    sd[sd < 1e-6] = 1.0

    # Only the STATIC graph is resident on the card.  The per-time tensors are small
    # ([N,4] and three [N] vectors) and 47 vessels x 16 times of them would add ~0.5 GB to a
    # 4 GB device that also has to hold six message-passing layers' activations; building
    # them per step costs a negligible transfer next to the forward pass.
    G = {a: build_static_graph(E[a], mu, sd, dev) for a in keys}
    sel = {a: torch.tensor(np.asarray(E[a]["solid"], bool), device=dev) for a in keys}
    # detected from the labels, not imported from `wall_cohort_splits.CLOT_FREE`, so a cache
    # built from any vessel set behaves the same way
    vessel_w = {a: (cfg.clot_free_w if float(E[a]["gt_t"].max()) <= 0.0 else 1.0)
                for a in keys}
    n_free = sum(1 for a in keys if vessel_w[a] != 1.0)
    print(f"    {n_free} clot-free vessels down-weighted to {cfg.clot_free_w}", flush=True)

    def slice_at(a, k):
        e = E[a]
        return (extra_channels(e, k, dev),
                torch.tensor(np.asarray(e["ode_t"][k], np.float32), device=dev),
                torch.tensor(crossing_target(e, k), device=dev),
                torch.tensor(np.asarray(e["gt_t"][k], np.float32), device=dev))

    model = make_model(G[keys[0]]["x"].shape[1], G[keys[0]]["ea"].shape[1], cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    steps = max(cfg.epochs * sum(len(E[a]["t_idx"]) for a in keys), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps,
                                                pct_start=0.25)
    pw = torch.tensor(cfg.pos_weight, device=dev)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    CKPT.mkdir(parents=True, exist_ok=True)

    pairs = [(a, k) for a in keys for k in range(len(E[a]["t_idx"]))]
    best = float("inf")
    for ep in range(cfg.epochs):
        model.train()
        tot = n = 0.0
        for idx in np.random.permutation(len(pairs)):
            a, k = pairs[idx]
            g = G[a]
            ex_k, base_k, tc_k, tr_k = slice_at(a, k)
            opt.zero_grad(set_to_none=True)
            logit, reg = model(g["x"], g["ei"], g["ea"], g["w_up"], g["w_dn"],
                               base_k, extra=ex_k)
            s = sel[a]
            loss = cfg.cls_w * bce(logit[s], tc_k[s], pos_weight=pw)
            # magnitude-weighted regression: the nodes that decide the off-wall rule are the
            # large ones, and an unweighted mean over a field that is zero at the median
            # simply learns the zero.
            w = 1.0 + cfg.reg_mag_w * tr_k[s]
            l1 = torch.nn.functional.smooth_l1_loss(reg[s], tr_k[s], reduction="none")
            loss = loss + cfg.reg_w * (w * l1).sum() / w.sum().clamp_min(1e-6)
            loss = loss * vessel_w[a]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            n += 1
        msg = f"  ep {ep + 1:3d}/{cfg.epochs}  loss {tot / max(n, 1):.5f}"
        if tot / max(n, 1) < best:
            best = tot / max(n, 1)
            torch.save(dict(state=model.state_dict(), mu=mu, sd=sd,
                            cfg=vars(cfg), in_dim=int(G[keys[0]]["x"].shape[1]),
                            edim=int(G[keys[0]]["ea"].shape[1]),
                            extra=list(EXTRA_CHANNELS), off_att=OFF_ATT,
                            train_stems=keys, holdout=TARGET),
                       CKPT / f"best_seed{cfg.seed}.pth")
            msg += "  [save]"
        print(msg, flush=True)
    print(f"[OK] best loss {best:.5f} -> {CKPT / f'best_seed{cfg.seed}.pth'}", flush=True)
    return 0


# ---------------------------------------------------------------------------
def load_model(seed: int, dev):
    p = CKPT / f"best_seed{seed}.pth"
    if not p.exists():
        raise FileNotFoundError(f"no checkpoint at {p} -- run `train` first")
    ck = torch.load(p, map_location=dev, weights_only=False)
    cfg = MatFieldConfig(**{k: v for k, v in ck["cfg"].items()
                            if k in MatFieldConfig.__dataclass_fields__})
    m = make_model(ck["in_dim"], ck["edim"], cfg).to(dev)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


def _auc(score, y):
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(np.asarray(score, float))) + 1.0
    npos, nneg = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def solid_shells(entry: dict, max_depth: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Corner shells off the WHOLE solid boundary, plus a nearest-solid owner for each.

    Thin wrapper around :func:`src.core_physics.physics_lumen_model.solid_boundary_shells`
    so existing diag scripts keep their import.  See that function for why shells are
    solid-anchored and why shell 1 stays the shipped first corner shell.
    """
    from src.core_physics.physics_lumen_model import solid_boundary_shells

    return solid_boundary_shells(
        np.asarray(entry["pos"], dtype=np.float64),
        np.asarray(entry["solid"], bool),
        np.asarray(entry["edge_index"]),
        shell1=np.asarray(entry["shell"], bool),
        town=np.asarray(entry["town"]),
        max_depth=int(max_depth),
    )


def cmd_eval(args) -> int:
    """Score the v6 field through the OTHERWISE-UNCHANGED off-wall rule.

    Two arms, because the two heads answer different questions.  ``cls`` reads the classifier,
    which was trained on exactly one bar (``Mat >= crit/0.16``) and so speaks only for the
    first shell.  ``reg`` reads the magnitude head, which can be compared against a bar at any
    DEPTH (``att**k``) at the cost of needing its top end to be right -- the extrapolation risk
    ``diag_mat_magnitude_cohort.py`` flagged, since 003's ``Mat`` p90 is the dataset maximum.

    Both are strictly additive on the shipped committed set, so the wall domain is untouched
    by construction and any off-wall move is attributable to the learned field alone.
    """
    from src.clot_ml.evaluate import domain_score
    from src.clot_ml.locked import (
        build_sample, load_temporal_v4_wound, predict_temporal_v4_wound,
    )
    from src.clot_ml.mat_field import predict_entry
    from src.config import PhysicsConfig
    from src.core_physics.t0_mu_physics import gt_clot_phi_at_time

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, mu, sd = [], None, None
    if not args.oracle:
        for s in args.seeds:
            try:
                models.append(load_model(s, dev))
            except FileNotFoundError as exc:
                print(f"  [skip seed {s}] {exc}", flush=True)
        if not models:
            return 1
        mu, sd = models[0][1]["mu"], models[0][1]["sd"]
    else:
        print("[i] ORACLE: substituting GT Mat for both heads -- this is the CEILING the "
              "learned field is chasing, never a deploy path", flush=True)
    phys, bio = PhysicsConfig(phase="biochem"), BiochemConfig(phase="biochem")
    bundle = load_temporal_v4_wound(name=args.artifact)
    thrs = np.asarray(args.thresholds, dtype=float)
    rows, summary = [], {}

    for stem in args.stems:
        p = CACHE / f"{stem}.npz"
        if not p.exists():
            print(f"[skip] {stem}: not cached", flush=True)
            continue
        with np.load(p, allow_pickle=False) as z:
            e = {k: z[k] for k in z.files}
        data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
        ei_t = torch.tensor(np.asarray(e["edge_index"]))
        solid = np.asarray(e["solid"], bool)
        off, shell, town = ~solid, np.asarray(e["shell"], bool), np.asarray(e["town"])
        has = town >= 0
        ti = [int(x) for x in e["t_idx"]]
        S = build_sample(data, bio, flow=args.flow, variant="v4")
        base_series = predict_temporal_v4_wound(bundle, data, ti, flow=args.flow,
                                                sample=S)["series"]
        shells, owner = solid_shells(e, args.depth)

        P, R = [], []
        for k in range(len(ti)):
            if args.oracle:
                g = np.asarray(e["gt_t"][k], dtype=np.float64)
                R.append(g)
                P.append((g >= np.log1p(1.0 / OFF_ATT)).astype(np.float64))
            else:
                out = [predict_entry(m, e, mu, sd, dev, k) for m, _ in models]
                P.append(np.mean([o[0] for o in out], axis=0))
                R.append(np.mean([o[1] for o in out], axis=0))

        base_w, base_o = [], []
        arms: dict[str, list[float]] = {}
        for k, t in enumerate(ti):
            gt = gt_clot_phi_at_time(data, t, phys).numpy() > 0.5
            bp = np.asarray(base_series[t], bool)
            base_w.append(domain_score(bp, gt, ei_t, np.asarray(e["wall"], bool), solid))
            base_o.append(domain_score(bp, gt, ei_t, off, solid))
            # `union` ADDS to the shipped off-wall verdict; `replace` hands the off-wall
            # domain to the rule alone and keeps the base only on solid.  They are not close:
            # the shipped readout carries off-wall FALSE POSITIVES the rule would not make, so
            # unioning a PERFECT field with it scores 0.6558 on 003 where the same field alone
            # scores 0.7897.  The base cannot be assumed harmless off-wall.
            seed = bp if args.combine == "union" else (bp & solid)
            for th in thrs:  # classifier: trained at one bar, so first shell only
                m = (shells[0] & (P[k][owner] >= th)) | seed
                arms.setdefault(f"cls@{th:g}", []).append(
                    domain_score(m, gt, ei_t, off, solid))
            for att in args.atts:  # magnitude: comparable at any depth via att**k
                for d in range(1, args.depth + 1):
                    m = seed.copy()
                    for j in range(d):
                        bar = np.log1p(1.0 / max(float(att) ** (j + 1), 1e-30))
                        m = m | (shells[j] & (R[k][owner] >= bar))
                    arms.setdefault(f"reg@{att:g}d{d}", []).append(
                        domain_score(m, gt, ei_t, off, solid))

        k = len(ti) - 1
        gt = gt_clot_phi_at_time(data, ti[k], phys).numpy() > 0.5
        cand = shells[0] & has
        a_p = _auc(P[k][owner[cand]], gt[cand]) if cand.any() else float("nan")
        a_r = _auc(R[k][owner[cand]], gt[cand]) if cand.any() else float("nan")
        q = np.percentile(np.expm1(R[k][solid]), [50, 90, 99])
        fin = {a: v[-1] for a, v in arms.items()}
        best = max(fin, key=lambda a: fin[a])
        print("=" * 104)
        print(f"{stem}   wall(final) {base_w[-1]:.4f}   off base(final) {base_o[-1]:.4f}")
        print(f"  ownerAUC  cls {a_p:.4f}   reg {a_r:.4f}   "
              f"reg Mat/crit p50/p90/p99 {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}")
        for a in sorted(arms):
            # early frames have no GT clot anywhere, where `domain_score` is NaN by design
            # (`empty_gt="nan"`); a plain mean over them would poison the row.
            print(f"    {a:14s} final {arms[a][-1]:7.4f}   "
                  f"mean-t {np.nanmean(arms[a]):7.4f}")
        print(f"  BEST {best} -> final {fin[best]:.4f}")
        rows.append((stem, base_w[-1], base_o[-1], best, fin[best]))
        summary[stem] = {"wall_final": base_w[-1], "off_base_final": base_o[-1],
                         "owner_auc_cls": a_p, "owner_auc_reg": a_r,
                         "final": fin,
                         "mean_t": {a: float(np.nanmean(v)) for a, v in arms.items()}}

    print("=" * 104)
    print(f"{'stem':22s} {'wall':>8s} {'off base':>9s} {'best arm':>14s} {'off v6':>8s}")
    for r in rows:
        print(f"{r[0]:22s} {r[1]:8.4f} {r[2]:9.4f} {r[3]:>14s} {r[4]:8.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[save] {args.out}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache")
    c.add_argument("--stems", nargs="*")
    c.add_argument("--flow", default="gt")
    c.add_argument("--n-times", type=int, default=16)
    c.add_argument("--artifact", default="clot_gnn_v5w")
    c.add_argument("--force", action="store_true")
    c.set_defaults(fn=cmd_cache)

    t = sub.add_parser("train")
    t.add_argument("--dim", type=int, default=64)
    t.add_argument("--layers", type=int, default=6)
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--cls-w", type=float, default=1.0)
    t.add_argument("--reg-w", type=float, default=0.3)
    t.add_argument("--pos-weight", type=float, default=12.0)
    t.add_argument("--reg-mag-w", type=float, default=1.0)
    t.add_argument("--clot-free-w", type=float, default=0.25)
    t.add_argument("--holdout", nargs="*", default=[],
                   help="extra vessels to hold out, for an honest generalization read")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=cmd_train)

    v = sub.add_parser("eval")
    v.add_argument("--stems", nargs="*",
                   default=["wound_patient003", "wound_patient001", "wound_patient002"])
    v.add_argument("--seeds", nargs="*", type=int, default=[0])
    v.add_argument("--flow", default="gt")
    v.add_argument("--artifact", default="clot_gnn_v5w")
    v.add_argument("--thresholds", nargs="*", type=float,
                   default=[0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9])
    v.add_argument("--atts", nargs="*", type=float, default=[0.16, 0.23, 0.35, 0.5])
    v.add_argument("--depth", type=int, default=3)
    v.add_argument("--combine", choices=("union", "replace"), default="replace")
    v.add_argument("--oracle", action="store_true",
                   help="substitute GT Mat for the learned field: the reachable ceiling")
    v.add_argument("--out", default=None)
    v.set_defaults(fn=cmd_eval)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
