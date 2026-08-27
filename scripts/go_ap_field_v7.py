"""v7 -- build, train and evaluate the ClotGNN residual on the AP field.

CONTEXT (docs/WOUND_PROGRESS.md §18.3). The deploy stack requires a time-varying AP field
to drive the surface ODE. The pure advective-upwind model (wall_ap_renewal) drops
AP_owner AUC to 0.22, so we train a ClotGNN residual on top of it.

    python scripts/go_ap_field_v7.py cache
    python scripts/go_ap_field_v7.py train --epochs 60
    python scripts/go_ap_field_v7.py eval
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.ap_field import (
    ApFieldConfig, EXTRA_CHANNELS, AP_TARGET_SCALE, build_ap_field_entry, build_static_graph,
    extra_channels, make_model,
)
from src.config import BiochemConfig, PhysicsConfig

PACKS = REPO / "data/processed/graphs_biochem_anchors"
CACHE = REPO / "outputs/ap_field_cache"
CKPT = REPO / "outputs/clot_ml/ap_field_v7"
TARGET = "wound_patient003"
SEALED = ("patient007", "patient013", "patient031", "patient043")


def cache_stems() -> list[str]:
    out = []
    for p in sorted(PACKS.glob("*.pt")):
        if p.name.endswith(".prenormalfix"):
            continue
        out.append(p.stem)
    return out


def train_stems(all_stems: list[str], extra_holdout=()) -> list[str]:
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
    stems = args.stems or cache_stems()
    print(f"[i] caching {len(stems)} packs -> {CACHE}", flush=True)
    for i, stem in enumerate(stems, 1):
        out = CACHE / f"{stem}.npz"
        if out.exists() and not args.force:
            print(f"  [skip] {stem}", flush=True)
            continue
        t0 = time.time()
        try:
            data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
            e = build_ap_field_entry(data, bio, flow="gt", n_times=args.n_times)
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
            raise FileNotFoundError(f"missing cache for {s} -- run `cache` first")
        E[s] = dict(np.load(p))
    return E


def cmd_train(args) -> int:
    cfg = ApFieldConfig(epochs=args.epochs, seed=args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] device {dev}  seed {cfg.seed}")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    all_stems = cache_stems()
    keys = train_stems(all_stems, args.holdout)
    print(f"[i] loading {len(keys)} training entries...")
    E = load_entries(keys)

    # 1. Feature normalization based on the training set
    X_all = np.concatenate([E[k]["X"] for k in keys], axis=0)
    mu, sd = X_all.mean(0).astype(np.float32), X_all.std(0).astype(np.float32)
    sd = np.where(sd > 1e-6, sd, 1.0)
    del X_all

    # 2. Build static graph parts
    G = {k: build_static_graph(E[k], mu, sd, dev) for k in keys}

    # 3. Restrict training loss to solid nodes
    sel = {k: torch.tensor(E[k]["solid"], device=dev, dtype=torch.bool) for k in keys}

    def slice_at(a, k):
        """Returns extra, base, and target tensors for vessel `a` at time step index `k`."""
        e = E[a]
        return (extra_channels(e, k, dev),
                torch.tensor(np.asarray(e["ode_t"][k], np.float32), device=dev),
                torch.tensor(np.asarray(e["gt_t"][k], np.float32), device=dev))

    model = make_model(G[keys[0]]["x"].shape[1], G[keys[0]]["ea"].shape[1], cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    steps = max(cfg.epochs * sum(len(E[a]["t_idx"]) for a in keys), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps,
                                                pct_start=0.25)
    
    CKPT.mkdir(parents=True, exist_ok=True)
    pairs = [(a, k) for a in keys for k in range(len(E[a]["t_idx"]))]
    best = float("inf")
    
    for ep in range(cfg.epochs):
        model.train()
        tot = n = 0.0
        for idx in np.random.permutation(len(pairs)):
            a, k = pairs[idx]
            g = G[a]
            ex_k, base_k, tr_k = slice_at(a, k)
            opt.zero_grad(set_to_none=True)
            # ignore cls logit
            _, reg = model(g["x"], g["ei"], g["ea"], g["w_up"], g["w_dn"],
                           base_k, extra=ex_k)
            s = sel[a]
            if not s.any():
                continue
                
            loss = F.smooth_l1_loss(reg[s].squeeze(), tr_k[s], reduction="mean")
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
                            extra=list(EXTRA_CHANNELS),
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
    cfg = ApFieldConfig(**{k: v for k, v in ck["cfg"].items()
                           if k in ApFieldConfig.__dataclass_fields__})
    m = make_model(ck["in_dim"], ck["edim"], cfg).to(dev)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


def cmd_eval(args) -> int:
    """Evaluate the trained field via diag_wall_ap_renewal logic."""
    from scripts.diag_chem_oracle_v6 import ATTS, DEPTH, _score_field
    from scripts.diag_wound_ode_closure_cell import auc
    from scripts.go_mat_field_v6 import solid_shells
    from src.clot_ml.evaluate import domain_score
    from src.clot_ml.locked import load_temporal_v4_wound
    from src.clot_ml.wound import solid_mask, wound_rate_blockage, wound_region_masks
    from src.core_physics.ap_closure import SHIPPED_DA_SCALE
    from src.core_physics.physics_lumen_model import first_corner_shell, topological_owner
    from src.core_physics.physics_wall_model import (
        WASHOUT_LAMBDA, integrate_mat_trajectory, t0_flow_fields, deposition_gate, PER_M3_TO_PER_CM3
    )
    from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
    from src.clot_ml.ap_field import predict_entry

    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(args.seed, dev)
    
    stems = ["wound_patient001", "wound_patient002", "wound_patient003"]
    if not args.wound_only:
        stems += ["patient012", "patient032"] # Add some generic cohort packs if needed

    print(f"[i] evaluating seed {args.seed} on {len(stems)} vessels...", flush=True)
    E = load_entries(stems)
    
    w = load_temporal_v4_wound(name="clot_gnn_v5w")["wound"]
    
    for stem in stems:
        data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
        T = int(data.y.shape[0])
        ei_np = data.edge_index.detach().cpu().numpy()
        ei = torch.tensor(ei_np)
        wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
        solid = solid_mask(data)
        off = ~solid
        pos = data.x[:, :2].detach().cpu().numpy().astype(np.float64)
        gt = gt_clot_phi_at_time(data, T - 1, phys).numpy() > 0.5

        shell1 = first_corner_shell(pos, solid, ei_np)
        town = topological_owner(pos, solid, ei_np)
        shells, owner = solid_shells(
            dict(solid=solid, edge_index=ei_np, shell=shell1, pos=pos, town=town), DEPTH)
        seed = gt & solid                     # keep wall/wound as-is; replace off-wall
        _, _, far = wound_region_masks(data)
        cand = shells[0] & off & far & (town >= 0)
        y_far, o_far = gt[cand], town[cand]
        
        # 1. Base upwind-renewal AP field [T, N] (deploy-legal)
        f = t0_flow_fields(data, bio, hops=3, flow_source="gt")
        gate = deposition_gate(data, f, wall=wall, wound_source=True)
        from src.core_physics.wall_ap_renewal import WallApRenewal, make_species_from_renewal
        renewal = WallApRenewal(renewal_scale=1.0)
        rp0, ap_traj_cgs = make_species_from_renewal(data, bio, f, renewal=renewal)

        scales = bio.get_species_scales(device="cpu")
        ap_scale_cgs = float(scales[1]) * PER_M3_TO_PER_CM3
        ode_nd = np.log1p(np.maximum(ap_traj_cgs, 0.0) / ap_scale_cgs).astype(np.float32)

        # 2. Predict full [T, N] AP_log1p_nd field using GNN
        e = E[stem]
        g = build_static_graph(e, ck["mu"], ck["sd"], dev)
        
        ap_traj_pred_nd = np.zeros((T, int(data.num_nodes)), dtype=np.float32)
        model.eval()
        with torch.no_grad():
            for t_step in range(T):
                tf = float(t_step) / max(T - 1, 1)
                # Scale the base identically to how it was scaled in training
                base = ode_nd[t_step] * AP_TARGET_SCALE
                cols = np.stack([
                    np.full(base.shape, tf, dtype=np.float32),
                    base,
                    np.asarray(e["gate"], dtype=np.float32),
                ], axis=1)
                ex_t = torch.tensor(cols, dtype=torch.float32, device=dev)
                base_t = torch.tensor(base, device=dev)
                
                _, reg = model(g["x"], g["ei"], g["ea"], g["w_up"], g["w_dn"], base_t, extra=ex_t)
                # Unscale the network output back to the true ND space
                ap_traj_pred_nd[t_step] = (reg.cpu().numpy() / AP_TARGET_SCALE).reshape(-1)
            
        # 3. Convert back to CGS for ODE integration
        ap_traj_pred_cgs = (np.expm1(np.clip(ap_traj_pred_nd, -10, 8)) * ap_scale_cgs).astype(np.float64)
        species = (rp0, ap_traj_pred_cgs)

        washout_sr = np.broadcast_to(f.sr, (T, int(data.num_nodes)))
        
        wound_blk = wound_rate_blockage(data, bio, g_pre=float(w["g_pre"]), g_post=float(w["g_post"]))
        
        # 3. Integrate ODE with GNN species
        traj, _ = integrate_mat_trajectory(
            data, bio, gate,
            da_scale=SHIPPED_DA_SCALE,
            da_scale_auto=123.0,
            ap_closure=None,
            species=species,
            blockage=wound_blk,
            washout=WASHOUT_LAMBDA,
            washout_sr=washout_sr,
        )
        fld = traj[-1]
        
        # 4. Score
        p90 = float(np.percentile(fld[solid], 90) / crit)
        a_far = auc(fld[o_far], y_far) if cand.any() else float("nan")
        sc = _score_field(fld, shells, owner, seed, gt, ei, off, solid, crit)
        
        # Print
        print("=" * 116)
        wall_sc = domain_score(seed, gt, ei, wall, solid)
        print(f"{stem}  wall_F1={wall_sc:.4f}  "
              f"off GT+={int((gt & off).sum())}  "
              f"far cand={int(cand.sum())} (GT+ {int(y_far.sum())})")
        
        hdr = "  ".join(f"{a:>10s}" for a in [f"att{x:g}d{d}" for x in ATTS for d in range(1, DEPTH + 1)])
        print(f"  {'arm':38s} {'p90x':>7s} {'farAUC':>7s}  {hdr}")
        
        cols = "  ".join(
            f"{sc.get(k, float('nan')):10.4f}"
            for x in ATTS for d in range(1, DEPTH + 1)
            for k in [f"att{x:g}d{d}"]
        )
        print(f"  {'GNN AP residual da=123':38s} {p90:7.2f} {a_far:7.4f}  {cols}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    cp = sub.add_parser("cache")
    cp.add_argument("--stems", nargs="*")
    cp.add_argument("--force", action="store_true")
    cp.add_argument("--n-times", type=int, default=16)

    tp = sub.add_parser("train")
    tp.add_argument("--epochs", type=int, default=60)
    tp.add_argument("--seed", type=int, default=0)
    tp.add_argument("--holdout", nargs="*", default=[])

    ep = sub.add_parser("eval")
    ep.add_argument("--seed", type=int, default=0)
    ep.add_argument("--wound-only", action="store_true")

    args = ap.parse_args()
    if args.cmd == "cache":
        return cmd_cache(args)
    if args.cmd == "train":
        return cmd_train(args)
    if args.cmd == "eval":
        return cmd_eval(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
