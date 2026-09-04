"""v6 -- build, train and evaluate the LEARNED surface ``Mat`` field for wound off-wall clot.

Rationale and the measurements that force this design are in :mod:`src.clot_ml.mat_field`
and docs/WOUND_PROGRESS.md 16/17.  In one line: the off-wall architecture is correct and the
physics ``Mat`` field is the single broken component, so v6 replaces that component and
changes nothing else -- same shells, same owner map, same ``att``/``depth``/``scope``, same
monotone union, via ``predict_clot_ml_0(..., mat_field=...)``.

    python scripts/go_mat_field_v6.py cache --flow fem
    python scripts/go_mat_field_v6.py lovo  --flow fem --epochs 30
    python scripts/go_mat_field_v6.py eval  --flow fem --lovo

WHAT CHANGED SINCE THE 2026-08 RUN (docs/WOUND_PROGRESS.md 17), and why it is re-run:

* **six wound vessels, not three.**  17.3's verdict was that ``wound_patient003`` is out of
  distribution and the residual collapses onto the physics there.  That was measured with
  ONE stagnation-regime wound in the corpus and it in the test set.  ``wound_patient004/005/
  006`` arrived 2026-09-02 and 006 is a second stagnation wound, so the claim is now testable
  rather than structural.
* **deploy-legal flow.**  v6 was trained and scored on COMSOL's velocity field.  Everything
  here runs on ``flow="fem"`` -- the local Carreau solve -- so a win is a deployable win.
* **the residual base is the SHIPPED field.**  v6 sat on ``ode_trajectory``; the wound
  off-wall readout has shipped ``v0.chemistry_mat_trajectory`` since 2026-08-25.  Sitting on
  the plain ODE measured a residual against a field nothing uses.  ``--mat-base chem`` (the
  default) makes an UNTRAINED v6 bit-identical to what ships, so every move is the residual's.
* **leave-one-vessel-out over the wound cohort**, matching the protocol the wound rate and
  ``replace_scope`` were licensed under, instead of one hardcoded held-out vessel.

SEALED is never cached, never trained on and never scored here.  It was spent on 2026-09-03
and using it to select this arm would be a second read.
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
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.biochem_gnn.wall_cohort_constants import WOUND_COHORT  # noqa: E402
from src.clot_ml.mat_field import (  # noqa: E402
    EXTRA_CHANNELS, OFF_ATT, MatFieldConfig, build_mat_field_entry, build_static_graph,
    crossing_target, extra_channels, make_model,
)
from src.config import BiochemConfig  # noqa: E402
from src.core_physics.wall_cohort_splits import SEALED  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
CACHE_FOR_FLOW = {"gt": REPO / "outputs/mat_field_cache",
                  "fem": REPO / "outputs/mat_field_cache_fem"}
CKPT = REPO / "outputs/clot_ml/mat_field_v6"


def cache_dir(flow: str) -> Path:
    return CACHE_FOR_FLOW.get(flow, REPO / f"outputs/mat_field_cache_{flow}")


def cache_stems() -> list[str]:
    """Every pack that may legally be read.

    SEALED is dropped HERE, not at the call site, so no subcommand can reach it by
    forgetting a flag.
    """
    out = []
    for p in sorted(PACKS.glob("*.pt")):
        if p.name.endswith(".prenormalfix") or p.stem in SEALED:
            continue
        out.append(p.stem)
    return out


def train_stems(all_stems: list[str], holdout=()) -> list[str]:
    """Everything legal minus the holdout.

    A holdout drops its MIRROR too (``patient005_mirror_y`` is the same vessel reflected), or
    the "held-out" score would be read off a vessel the model trained on.
    """
    drop: set[str] = set()
    for s in holdout:
        drop.add(s)
        drop.add(f"{s}_mirror_y")
        if s.endswith("_mirror_y"):
            drop.add(s[: -len("_mirror_y")])
    return [s for s in all_stems if s not in drop]


# ---------------------------------------------------------------------------
def cmd_cache(args) -> int:
    CACHE = cache_dir(args.flow)
    CACHE.mkdir(parents=True, exist_ok=True)
    bio = BiochemConfig(phase="biochem")
    from src.clot_ml.v0 import load_v0_bundle, solve_fem_into_pack

    b = load_v0_bundle(args.artifact)
    w = b["base"].get("wound") or {}
    wr = (float(w["g_pre"]), float(w["g_post"]))
    cfg0 = b["cfg"]
    stems = args.stems or cache_stems()
    print(f"[i] caching {len(stems)} packs -> {CACHE}  flow={args.flow} "
          f"base={args.mat_base} wound_rate={wr}", flush=True)
    for i, stem in enumerate(stems, 1):
        out = CACHE / f"{stem}.npz"
        if out.exists() and not args.force:
            print(f"  [skip] {stem}", flush=True)
            continue
        t0 = time.time()
        try:
            data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
            data.graph_stem = stem
            if args.flow == "fem":
                solve_fem_into_pack(data)
            e = build_mat_field_entry(data, bio, flow=args.flow, n_times=args.n_times,
                                      wound_rate=wr, mat_base=args.mat_base, v0_cfg=cfg0)
            np.savez_compressed(out, **e)
            print(f"  [{i}/{len(stems)}] {stem}  N={e['X'].shape[0]} "
                  f"K={len(e['t_idx'])}  gtmax={float(e['gt_t'].max()):.2f} "
                  f"basemax={float(e['ode_t'].max()):.2f}  {time.time() - t0:.0f}s",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {stem}: {type(exc).__name__}: {exc}", flush=True)
    return 0


def load_entries(stems, flow: str) -> dict:
    CACHE = cache_dir(flow)
    E = {}
    for s in stems:
        p = CACHE / f"{s}.npz"
        if not p.exists():
            continue
        with np.load(p, allow_pickle=False) as z:
            E[s] = {k: z[k] for k in z.files if k != "mat_base"}
    return E


# ---------------------------------------------------------------------------
def _fit(cfg: MatFieldConfig, E: dict, dev, tag: str, quiet: bool = False):
    """Train one field on the entries in ``E``.  Returns ``(state, mu, sd, in_dim, edim)``."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    keys = sorted(E)
    stacked = np.concatenate([E[a]["X"] for a in keys])
    mu, sd = stacked.mean(0), stacked.std(0)
    sd[sd < 1e-6] = 1.0
    del stacked

    # Only the STATIC graph is resident on the card.  The per-time tensors are small
    # ([N,4] and three [N] vectors) and 46 vessels x K times of them would add ~0.5 GB to a
    # 4 GB device that also has to hold six message-passing layers' activations; building
    # them per step costs a negligible transfer next to the forward pass.
    G = {a: build_static_graph(E[a], mu, sd, dev) for a in keys}
    sel = {a: torch.tensor(np.asarray(E[a]["solid"], bool), device=dev) for a in keys}
    # detected from the labels, not imported from `wall_cohort_splits.CLOT_FREE`, so a cache
    # built from any vessel set behaves the same way
    vessel_w = {a: (cfg.clot_free_w if float(E[a]["gt_t"].max()) <= 0.0 else 1.0)
                for a in keys}
    if not quiet:
        n_free = sum(1 for a in keys if vessel_w[a] != 1.0)
        print(f"    {len(keys)} vessels, {n_free} clot-free down-weighted to "
              f"{cfg.clot_free_w}", flush=True)

    def slice_at(a, k):
        e = E[a]
        return (extra_channels(e, k, dev),
                torch.tensor(np.asarray(e["ode_t"][k], np.float32), device=dev),
                torch.tensor(crossing_target(e, k), device=dev),
                torch.tensor(np.asarray(e["gt_t"][k], np.float32), device=dev))

    model = make_model(G[keys[0]]["x"].shape[1], G[keys[0]]["ea"].shape[1], cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    _kpe = int(getattr(cfg, "_k_per_epoch", 0) or 0)
    _b1 = float(np.log1p(1.0 / OFF_ATT))
    steps = max(cfg.epochs * sum(
        min(len(E[a]["t_idx"]),
            (_kpe if float(E[a]["gt_t"].max()) >= _b1 else max(1, _kpe // 3))
            if _kpe else 10 ** 9)
        for a in keys), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps,
                                                pct_start=0.25)
    pw = torch.tensor(cfg.pos_weight, device=dev)
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    pairs = [(a, k) for a in keys for k in range(len(E[a]["t_idx"]))]
    # Each epoch sees a random `k_per_epoch` of every vessel's time samples rather than all
    # of them.  The target is a monotone crossing, so consecutive samples carry nearly the
    # same label and a full pass spends most of its steps re-deriving one frame from its
    # neighbour; sampling turns that redundancy into wall-clock.  `0` keeps every sample.
    kpe = int(getattr(cfg, "_k_per_epoch", 0) or 0)
    # Vessels that never reach the bar the off-wall rule asks about carry no positive at any
    # time, so every one of their samples teaches the same thing.  They stay in -- a false
    # positive on a quiet vessel is a scored failure -- but at a third of the sampling rate,
    # which is where the wall-clock goes: WOUND_PROGRESS 17.3 counted 30 of 47 vessels with
    # wall `Mat` p90 at or below 1.01x crit.
    bar1 = float(np.log1p(1.0 / OFF_ATT))
    kpe_of = {a: (kpe if float(E[a]["gt_t"].max()) >= bar1 else max(1, kpe // 3))
              for a in keys}
    best, best_state = float("inf"), None
    for ep in range(cfg.epochs):
        model.train()
        tot = n = 0.0
        if kpe:
            ep_pairs = []
            for a in keys:
                ks = np.random.permutation(len(E[a]["t_idx"]))[:kpe_of[a]]
                ep_pairs.extend((a, int(k)) for k in ks)
        else:
            ep_pairs = pairs
        for idx in np.random.permutation(len(ep_pairs)):
            a, k = ep_pairs[idx]
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
        avg = tot / max(n, 1)
        star = ""
        if avg < best:
            best = avg
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            star = "  *"
        if not quiet:
            print(f"  [{tag}] ep {ep + 1:3d}/{cfg.epochs}  loss {avg:.5f}{star}", flush=True)
    return best_state, mu, sd, int(G[keys[0]]["x"].shape[1]), int(G[keys[0]]["ea"].shape[1])


def _cfg_from(args) -> MatFieldConfig:
    cfg = MatFieldConfig(dim=args.dim, layers=args.layers, epochs=args.epochs, lr=args.lr,
                         cls_w=args.cls_w, reg_w=args.reg_w, pos_weight=args.pos_weight,
                         reg_mag_w=args.reg_mag_w, clot_free_w=args.clot_free_w,
                         seed=args.seed)
    object.__setattr__(cfg, "_k_per_epoch", int(getattr(args, "k_per_epoch", 0) or 0))
    return cfg


def _save(path: Path, state, mu, sd, in_dim, edim, cfg, keys, holdout, flow, mat_base):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=state, mu=mu, sd=sd, cfg=vars(cfg), in_dim=in_dim, edim=edim,
                    extra=list(EXTRA_CHANNELS), off_att=OFF_ATT, train_stems=list(keys),
                    holdout=list(holdout), flow=flow, mat_base=mat_base), path)


def cmd_lovo(args) -> int:
    """One field per wound vessel, that vessel held out.  This is the honest read."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _cfg_from(args)
    allst = cache_stems()
    targets = list(args.targets or WOUND_COHORT)
    print(f"[i] LOVO over {len(targets)} wound vessels, dev={dev}, flow={args.flow}, "
          f"seed={cfg.seed}, epochs={cfg.epochs}", flush=True)
    for tgt in targets:
        out = CKPT / f"lovo_{args.flow}_{tgt}_seed{cfg.seed}.pth"
        if out.exists() and not args.force:
            print(f"  [skip] {tgt}: {out.name} exists", flush=True)
            continue
        keys = train_stems(allst, [tgt])
        E = load_entries(keys, args.flow)
        if not E:
            print(f"  [ERR] {tgt}: cache empty -- run `cache` first", flush=True)
            return 1
        t0 = time.time()
        state, mu, sd, ind, ed = _fit(cfg, E, dev, tgt, quiet=args.quiet)
        _save(out, state, mu, sd, ind, ed, cfg, sorted(E), [tgt], args.flow, args.mat_base)
        print(f"[OK] {tgt}: {len(E)} train vessels, {time.time() - t0:.0f}s -> {out.name}",
              flush=True)
        del E
    return 0


def cmd_train(args) -> int:
    """One field on every legal vessel -- the DEPLOY field, not a generalization read."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _cfg_from(args)
    keys = train_stems(cache_stems(), args.holdout)
    E = load_entries(keys, args.flow)
    if not E:
        print("[ERR] cache empty -- run `cache` first", flush=True)
        return 1
    print(f"[i] train on {len(E)} vessels, dev={dev}, flow={args.flow}, seed={cfg.seed}\n"
          f"    held out: {', '.join(args.holdout) or '(none beyond SEALED)'}", flush=True)
    state, mu, sd, ind, ed = _fit(cfg, E, dev, "all", quiet=args.quiet)
    out = CKPT / f"deploy_{args.flow}_seed{cfg.seed}.pth"
    _save(out, state, mu, sd, ind, ed, cfg, sorted(E), args.holdout, args.flow, args.mat_base)
    print(f"[OK] -> {out}", flush=True)
    return 0


# ---------------------------------------------------------------------------
def load_ckpt(path: Path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    cfg = MatFieldConfig(**{k: v for k, v in ck["cfg"].items()
                            if k in MatFieldConfig.__dataclass_fields__})
    m = make_model(ck["in_dim"], ck["edim"], cfg).to(dev)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


@torch.no_grad()
def learned_mat_series(model, ck, entry: dict, dev, crit: float) -> np.ndarray:
    """``[K, N]`` SI ``Mat`` from the regression head at every cached time in ``entry``.

    The REGRESSION head is what feeds ``replace_depth_mask``: the classifier speaks for one
    bar only (``crit/off_att``, i.e. shell 1) while the rule compares against ``crit/att**d``
    at every depth.  ``head_reg`` is a zero-init residual on the base field, so this returns
    the shipped field exactly when the residual is zero.
    """
    g = build_static_graph(entry, ck["mu"], ck["sd"], dev)
    out = np.empty((len(entry["t_idx"]), len(entry["solid"])), dtype=np.float64)
    for k in range(len(entry["t_idx"])):
        ex = extra_channels(entry, k, dev)
        base = torch.tensor(np.asarray(entry["ode_t"][k], np.float32), device=dev)
        _, reg = model(g["x"], g["ei"], g["ea"], g["w_up"], g["w_dn"], base, extra=ex)
        out[k] = np.expm1(np.maximum(reg.cpu().numpy().astype(np.float64), 0.0)) * crit
    return out


def cmd_eval(args) -> int:
    """Score the learned field through the OTHERWISE-UNCHANGED shipped readout.

    Both arms call ``predict_clot_ml_0``; the only difference is ``mat_field=``.  So the wall
    domain is untouched by construction and any wound off-wall move is the field's alone.
    """
    import dataclasses

    from eval_wound_complement import gt_series, score_domains

    from src.clot_ml.locked import build_sample
    from src.clot_ml.v0 import (
        REPLACE_SCOPE_ALL_LUMEN, REPLACE_SCOPE_WOUND_REGION, load_v0_bundle,
        predict_clot_ml_0, solve_fem_into_pack,
    )
    from src.clot_ml.wound import solid_mask, wound_region_masks
    from src.config import PhysicsConfig

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    bundle = load_v0_bundle(args.artifact)
    w = bundle["base"].get("wound") or {}
    wr = (float(w["g_pre"]), float(w["g_post"]))
    cfg0 = bundle["cfg"]
    crit = float(bio.viscosity_mat_crit)
    scopes = args.scopes or [REPLACE_SCOPE_WOUND_REGION, REPLACE_SCOPE_ALL_LUMEN]

    rows: dict[str, dict] = {}
    for stem in (args.stems or list(WOUND_COHORT)):
        ck_path = (CKPT / f"lovo_{args.flow}_{stem}_seed{args.seed}.pth" if args.lovo
                   else CKPT / f"deploy_{args.flow}_seed{args.seed}.pth")
        if not ck_path.exists():
            print(f"[skip] {stem}: no checkpoint {ck_path.name}", flush=True)
            continue
        t0 = time.time()
        data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
        data.graph_stem = stem
        if args.flow == "fem":
            solve_fem_into_pack(data)
        T = int(data.y.shape[0])
        times = sorted({*range(0, T, max(args.every, 1)), T - 1})
        # The entry is rebuilt at EVERY frame rather than read from the sparse training
        # cache: the readout is scored at `times`, and a step-held sparse field would make
        # the learned arm and the chemistry arm disagree about which frame they are on.
        entry = build_mat_field_entry(data, bio, flow=args.flow, n_times=T,
                                      wound_rate=wr, mat_base=args.mat_base, v0_cfg=cfg0)
        model, ck = load_ckpt(ck_path, dev)
        learned = learned_mat_series(model, ck, entry, dev, crit)   # [T, N]

        S = build_sample(data, bio, flow=args.flow, variant="v4")
        ei = torch.tensor(np.asarray(S["edge_index"]))
        gts = gt_series(data, phys, times)
        wall = np.asarray(S["wall"], dtype=bool)
        reg_m, lum_m, far_m = wound_region_masks(data)
        domains = dict(wall=wall, wnd=solid_mask(data) & ~wall, w_reg=reg_m, w_lum=lum_m,
                       far=far_m, full=np.ones(len(wall), dtype=bool))
        last = times[-1]

        row: dict[str, dict] = {}
        over = {k: v for k, v in (("replace_att", args.replace_att),
                                  ("replace_depth", args.replace_depth),
                                  ("att_beta", args.att_beta)) if v is not None}
        for scope in scopes:
            cfg = dataclasses.replace(cfg0, replace_scope=scope, **over)
            b = dict(bundle, cfg=cfg)
            for arm, fld in (("chem", None), ("v6", learned)):
                out = predict_clot_ml_0(b, data, times, flow=args.flow, sample=S,
                                        mat_field=fld, preflight="off")
                row[f"{arm}/{scope}"] = score_domains(out["series"][last], gts[last], ei,
                                                      wall, domains)
        # how far the learned field moved the magnitude, which is what 17.3 said was broken
        solid = solid_mask(data)
        row["_mag"] = dict(
            chem_p90=float(np.percentile(entry["ode_t"][last][solid], 90)),
            v6_p90=float(np.percentile(np.log1p(learned[last][solid] / crit), 90)),
            gt_p90=float(np.percentile(entry["gt_t"][last][solid], 90)),
        )
        rows[stem] = row
        print(f"[{stem}] {time.time() - t0:.0f}s  "
              + "  ".join(f"{k} w_lum {v['w_lum']:.4f} far {v['far']:.4f}"
                          for k, v in row.items() if k != "_mag"), flush=True)
        del entry, learned

    if not rows:
        print("[ERR] nothing scored", flush=True)
        return 1
    hdr = [k for k in next(iter(rows.values())) if k != "_mag"]
    print()
    for dom in ("wall", "w_reg", "w_lum", "far"):
        print(f"--- {dom}")
        print(f"{'vessel':20s} " + " ".join(f"{h:>26s}" for h in hdr))
        for stem, r in rows.items():
            print(f"{stem:20s} " + " ".join(f"{r[h][dom]:26.4f}" for h in hdr))
        print(f"{'MEAN':20s} " + " ".join(
            f"{np.nanmean([r[h][dom] for r in rows.values()]):26.4f}" for h in hdr))
    print()
    print(f"{'vessel':20s} {'chem p90':>10s} {'v6 p90':>10s} {'GT p90':>10s}  "
          f"(log1p Mat/crit on solid, final frame)")
    for stem, r in rows.items():
        m = r["_mag"]
        print(f"{stem:20s} {m['chem_p90']:10.3f} {m['v6_p90']:10.3f} {m['gt_p90']:10.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            dict(flow=args.flow, artifact=args.artifact, lovo=bool(args.lovo),
                 seed=args.seed, mat_base=args.mat_base, per_vessel=rows), indent=2),
            encoding="utf-8")
        print(f"[save] {args.out}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--flow", default="fem", choices=["gt", "fem"])
        p.add_argument("--mat-base", default="chem", choices=["chem", "ode"])
        p.add_argument("--artifact", default=None)

    c = sub.add_parser("cache")
    common(c)
    c.add_argument("--stems", nargs="*")
    c.add_argument("--n-times", type=int, default=12)
    c.add_argument("--force", action="store_true")
    c.set_defaults(fn=cmd_cache)

    def trainargs(p):
        common(p)
        p.add_argument("--dim", type=int, default=64)
        p.add_argument("--layers", type=int, default=6)
        p.add_argument("--epochs", type=int, default=30)
        p.add_argument("--lr", type=float, default=2e-3)
        p.add_argument("--cls-w", type=float, default=1.0)
        p.add_argument("--reg-w", type=float, default=0.3)
        p.add_argument("--pos-weight", type=float, default=12.0)
        p.add_argument("--reg-mag-w", type=float, default=1.0)
        p.add_argument("--clot-free-w", type=float, default=0.25)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--quiet", action="store_true")
        p.add_argument("--k-per-epoch", type=int, default=0,
                       help="time samples per vessel per epoch (0 = all)")

    lo = sub.add_parser("lovo")
    trainargs(lo)
    lo.add_argument("--targets", nargs="*")
    lo.add_argument("--force", action="store_true")
    lo.set_defaults(fn=cmd_lovo)

    t = sub.add_parser("train")
    trainargs(t)
    t.add_argument("--holdout", nargs="*", default=[])
    t.set_defaults(fn=cmd_train)

    v = sub.add_parser("eval")
    common(v)
    v.add_argument("--stems", nargs="*")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--lovo", action="store_true",
                   help="score each vessel with the field that held IT out")
    v.add_argument("--every", type=int, default=8)
    v.add_argument("--scopes", nargs="*")
    v.add_argument("--replace-att", type=float, default=None)
    v.add_argument("--replace-depth", type=int, default=None)
    v.add_argument("--att-beta", type=float, default=None)
    v.add_argument("--out", default=None)
    v.set_defaults(fn=cmd_eval)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
