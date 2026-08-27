"""Design EDA: how GT flow opens 003 blinds, and what a reduced-order gate could read.

Not a model.  MLS once per pack, then shear at every stored time.  Questions, in order:

  1. Which branch of `gate = A(sep) + B(lss)` actually opens the blinds?
  2. Does that opening LEAD local occupancy (flow reorganisation) or FOLLOW it (stall)?
  3. Is the drop a coating shelter along the wall, a wall-normal displacement layer,
     or a cross-section Q-drop (Arm 2, wrong sign)?
  4. Do 001/002 and FIT vessels have an analogous population, or is 003 unique?

    python scripts/diag_nonlocal_flow_gate.py
    python scripts/diag_nonlocal_flow_gate.py --no-cohort
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.features import adjacency, hop_distance  # noqa: E402
from src.clot_ml.wound import solid_mask, wound_mask, wound_region_masks  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_phi_simple import mat_si_for_gelation_from_log1p  # noqa: E402
from src.core_physics.mls_gradient import (  # noqa: E402
    build_mls_gradient, node_positions, shear_rate_2d,
)
from src.core_physics.physics_wall_model import M_TO_CM, gate_from_shear  # noqa: E402
from src.core_physics.shear_redistribution import (  # noqa: E402
    build_crosssection_operator, sdf_nd,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.wall_cohort_splits import CLOT_FREE, FIT  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
WOUND_STEMS = ("wound_patient001", "wound_patient002", "wound_patient003")
UNREACHED = 99.0


def _load(stem: str):
    p = PACKS / f"{stem}.pt"
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=False)


def _axial(pos: np.ndarray, wall: np.ndarray) -> np.ndarray:
    X = pos[wall] - pos[wall].mean(0)
    _, evecs = np.linalg.eigh(X.T @ X)
    s = pos @ evecs[:, -1]
    return (s - s.min()) / max(float(s.ptp()), 1e-9)


def _gt_mat(data, bio):
    mi = data.y_channel_names.split(",").index("Mat_log1p_nd")
    return mat_si_for_gelation_from_log1p(data.y[:, :, mi], bio).reshape(
        int(data.y.shape[0]), -1).numpy()


def _blinds(data, bio, phys, sr0, gate0):
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    solid, wnd = solid_mask(data), wound_mask(data)
    T = int(data.y.shape[0])
    gt = gt_clot_phi_at_time(data, T - 1, phys).numpy() > 0.5
    pos = node_positions(data)
    if not solid.any():
        return np.zeros(0, dtype=int), wall, solid, wnd
    _, j = cKDTree(pos[solid]).query(pos)
    owner = np.flatnonzero(solid)[j]
    if wnd.any():
        _, lumen, _ = wound_region_masks(data)
        ow = np.unique(owner[(gt & lumen) & ~wnd[owner]])
    else:
        ow = np.unique(owner[(gt & ~wall)])
        ow = ow[wall[ow]]
    return ow[gate0[ow] <= 0], wall, solid, wnd


def _series(data, bio):
    """MLS once; sr/dsrx/A/B/gate at every stored time.  Returns dict of [T, N] arrays."""
    pos = node_positions(data)
    ei = data.edge_index.detach().cpu().numpy()
    Dx, Dy = build_mls_gradient(pos, ei, hops=3)
    u_ref = float(data.u_ref.reshape(-1)[0])
    d_bar = float(data.d_bar.reshape(-1)[0])
    scale = u_ref / d_bar
    sgt = float(bio.sgt) / M_TO_CM
    coef = float(bio.L_char) * M_TO_CM / float(bio.gamma_m)
    lss = float(bio.lss)
    T = int(data.y.shape[0])
    n = int(data.num_nodes)
    sr = np.empty((T, n), dtype=np.float64)
    dsrx = np.empty((T, n), dtype=np.float64)
    speed = np.empty((T, n), dtype=np.float64)
    y = data.y.detach().cpu().numpy().astype(np.float64)
    for ti in range(T):
        u, v = y[ti, :, 0], y[ti, :, 1]
        speed[ti] = np.hypot(u, v)
        s = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * scale
        sr[ti] = s
        dsrx[ti] = (Dx @ s) / (d_bar * M_TO_CM)
    A = (dsrx < sgt) * coef * np.abs(dsrx)
    B = (sr < lss).astype(np.float64)
    gate = A + B
    return dict(sr=sr, dsrx=dsrx, A=A, B=B, gate=gate, speed=speed,
                pos=pos, ei=ei, lss=lss, sgt=sgt, coef=coef)


def _first_true(mask_tn: np.ndarray) -> np.ndarray:
    """Per-node first time a [T,N] bool is true; T if never."""
    T = mask_tn.shape[0]
    any_t = mask_tn.any(0)
    idx = np.where(any_t, mask_tn.argmax(0), T)
    return idx.astype(int)


def _spearman(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float("nan")
    ra = np.argsort(np.argsort(a[m])).astype(np.float64)
    rb = np.argsort(np.argsort(b[m])).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def _run_wound(stem: str, data, bio, phys) -> None:
    print("=" * 92)
    print(f"{stem}")
    S = _series(data, bio)
    gtm = _gt_mat(data, bio)
    crit = float(bio.viscosity_mat_crit)
    hot = gtm >= crit
    T, n = hot.shape
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    blinds, wall, solid, wnd = _blinds(data, bio, phys, S["sr"][0], S["gate"][0] * wall)
    A = adjacency(data.edge_index.detach().cpu().numpy(), n)
    pos = S["pos"]
    s_ax = _axial(pos, wall)
    sdf = sdf_nd(data)
    hops_wnd = hop_distance(wnd, A, max_h=24) if wnd.any() else np.full(n, UNREACHED)
    hops_solid0 = hop_distance(solid, A, max_h=12)
    Bop = build_crosssection_operator(pos, sdf, wall, radius_mult=1.0)

    print(f"  T={T}  wall={int(wall.sum())}  wound={int(wnd.sum())}  "
          f"blinds={blinds.size}  lss={S['lss']:.1f}")
    if blinds.size == 0:
        print("  [i] no t=0-ungated owners of wound-region lumen clot.  "
              "This vessel does not have a 003-style blind population.")
        _kernel_amp_table(S, wall, hot, solid, A, T)
        _flip_summary(S, wall, hot, solid, wnd)
        return

    hb = hops_wnd[blinds]
    print("  blinds hops-to-wound: " + " ".join(f"{h:.0f}" for h in np.sort(hb)))
    print(f"  blinds t0 sr med {float(np.median(S['sr'][0, blinds])):.1f} /s  "
          f"need amp < {S['lss'] / max(float(np.median(S['sr'][0, blinds])), 1e-9):.3f} for B")

    steps = [0, 2, 5, 10, 20, 40, min(80, T - 1), T - 1]
    steps = sorted(set(min(i, T - 1) for i in steps))
    print(f"  {'t':>4s}  {'n_occ_w':>7s}  {'n_occ_s':>7s}  "
          f"{'amp':>6s}  {'sr':>7s}  {'B%':>5s}  {'A%':>5s}  {'g%':>5s}  "
          f"{'h_w':>4s}  {'h_wl':>4s}  {'phi':>5s}")
    for ti in steps:
        occ_w = hot[ti] & wnd
        occ_s = hot[ti] & solid
        occ_wl = hot[ti] & wall
        hw = hop_distance(occ_w, A, max_h=24) if occ_w.any() else np.full(n, UNREACHED)
        hwl = hop_distance(occ_wl, A, max_h=24) if occ_wl.any() else np.full(n, UNREACHED)
        amp = S["sr"][ti, blinds] / np.maximum(S["sr"][0, blinds], 1e-9)
        phi = np.asarray(Bop @ occ_s.astype(np.float64)).reshape(-1)
        print(f"  {ti:4d}  {int(occ_w.sum()):7d}  {int(occ_s.sum()):7d}  "
              f"{float(np.median(amp)):6.3f}  "
              f"{float(np.median(S['sr'][ti, blinds])):7.1f}  "
              f"{100 * float((S['B'][ti, blinds] > 0).mean()):4.0f}%  "
              f"{100 * float((S['A'][ti, blinds] > 0).mean()):4.0f}%  "
              f"{100 * float((S['gate'][ti, blinds] > 0).mean()):4.0f}%  "
              f"{float(np.median(hw[blinds])):4.0f}  "
              f"{float(np.median(hwl[blinds])):4.0f}  "
              f"{float(np.median(phi[blinds])):5.3f}")

    # How each blind opens
    tA = _first_true(S["A"][:, blinds] > 0)
    tB = _first_true(S["B"][:, blinds] > 0)
    tG = _first_true(S["gate"][:, blinds] > 0)
    t_own = _first_true(hot[:, blinds])
    print("\n  per-blind first-open (T means never):")
    print(f"    {'i':>3s}  {'hopW':>4s}  {'s':>5s}  {'sr0':>6s}  "
          f"{'tA':>4s}  {'tB':>4s}  {'tG':>4s}  {'tMat':>4s}  {'via':>6s}")
    via_b = via_a = via_both = via_never = 0
    for k, nd in enumerate(blinds):
        via = "never"
        if tG[k] < T:
            a_on = tA[k] <= tG[k] and tA[k] < T
            b_on = tB[k] <= tG[k] and tB[k] < T
            if a_on and b_on:
                via = "A+B"
                via_both += 1
            elif b_on:
                via = "B"
                via_b += 1
            elif a_on:
                via = "A"
                via_a += 1
        else:
            via_never += 1
        print(f"    {k:3d}  {hb[k]:4.0f}  {s_ax[nd]:5.3f}  {S['sr'][0, nd]:6.1f}  "
              f"{tA[k]:4d}  {tB[k]:4d}  {tG[k]:4d}  {t_own[k]:4d}  {via:>6s}")
    print(f"  via  B={via_b}  A={via_a}  A+B={via_both}  never={via_never}")

    # Lead-lag at the instant the GATE first opens, vs occupancy
    print("\n  lead-lag at tG (hops from that blind to gelled sets; unreached="
          f"{UNREACHED:.0f}):")
    lead_w = lead_wl = follow_w = 0
    for k, nd in enumerate(blinds):
        tg = int(tG[k])
        if tg >= T:
            continue
        occ_w = hot[tg] & wnd
        occ_wl = hot[tg] & wall
        occ_s = hot[tg] & solid
        hw = hop_distance(occ_w, A, max_h=24)[nd] if occ_w.any() else UNREACHED
        hwl = hop_distance(occ_wl, A, max_h=24)[nd] if occ_wl.any() else UNREACHED
        hs = hop_distance(occ_s, A, max_h=24)[nd] if occ_s.any() else UNREACHED
        # lead = gate opened while no gelled WALL within 8 hops
        if hwl > 8:
            lead_wl += 1
        else:
            follow_w += 1
        if hw > 8:
            lead_w += 1
        print(f"    b{k:02d} tG={tg:3d}  hop_wound_occ={hw:.0f}  hop_wall_occ={hwl:.0f}  "
              f"hop_solid_occ={hs:.0f}  amp={S['sr'][tg, nd] / max(S['sr'][0, nd], 1e-9):.3f}")
    print(f"  opened with gelled WALL >8 hops away: {lead_wl}/{(tG < T).sum()}  "
          f"(wound occ >8 hops: {lead_w})")

    # Amp vs deployable correlators, pooled over blinds x sampled times after t=2
    amps, feat_hw, feat_hwl, feat_phi, feat_depth, feat_ds = [], [], [], [], [], []
    for ti in range(2, T, max(T // 20, 1)):
        occ_w = hot[ti] & wnd
        occ_wl = hot[ti] & wall
        occ_s = hot[ti] & solid
        hw = hop_distance(occ_w, A, max_h=24) if occ_w.any() else np.full(n, UNREACHED)
        hwl = hop_distance(occ_wl, A, max_h=24) if occ_wl.any() else np.full(n, UNREACHED)
        phi = np.asarray(Bop @ occ_s.astype(np.float64)).reshape(-1)
        depth = 0.0
        if (hot[ti] & ~solid & wnd).any() or (hot[ti] & wnd).any():
            gel_lumen = hot[ti] & ~solid
            if gel_lumen.any() and wnd.any():
                # coating depth: max sdf among gelled lumen nodes in the wound hop-4 region
                region, _, _ = wound_region_masks(data)
                sel = gel_lumen & region
                depth = float(sdf[sel].max()) if sel.any() else 0.0
        amp = S["sr"][ti, blinds] / np.maximum(S["sr"][0, blinds], 1e-9)
        amps.append(amp)
        feat_hw.append(hw[blinds])
        feat_hwl.append(hwl[blinds])
        feat_phi.append(phi[blinds])
        feat_depth.append(np.full(blinds.size, depth))
        feat_ds.append(np.abs(s_ax[blinds] - float(np.median(s_ax[wnd]))) if wnd.any()
                       else np.zeros(blinds.size))
    amps = np.concatenate(amps)
    print("\n  spearman(amp, feature) over blinds x time (amp down = collapse):")
    for name, feat in (("hops_to_gelled_wound", np.concatenate(feat_hw)),
                       ("hops_to_gelled_wall", np.concatenate(feat_hwl)),
                       ("slice_phi Arm2", np.concatenate(feat_phi)),
                       ("wound_region coating sdf max", np.concatenate(feat_depth)),
                       ("|s - s_wound|", np.concatenate(feat_ds))):
        print(f"    {name:32s}  {_spearman(amps, feat):+.3f}")

    # Wall-normal speed in the blind axial band vs a far band
    s_b = float(np.median(s_ax[blinds]))
    band = np.abs(s_ax - s_b) < 0.05
    far = np.abs(s_ax - 0.75) < 0.05
    print(f"\n  wall-normal speed ratio (t_final / t0) in axial bands:")
    print(f"    {'band':10s}  {'hop0-1':>8s}  {'hop2-3':>8s}  {'hop4-6':>8s}  "
          f"{'hop>=7':>8s}  {'core sdf>p75':>12s}")
    for name, msk in (("blinds", band), ("s~0.75", far)):
        cells = []
        h = hops_solid0
        for lo, hi in ((0, 1), (2, 3), (4, 6), (7, 12)):
            sel = msk & (h >= lo) & (h <= hi) & ~solid
            if not sel.any():
                cells.append(float("nan"))
                continue
            r = S["speed"][-1, sel] / np.maximum(S["speed"][0, sel], 1e-9)
            cells.append(float(np.median(r)))
        core = msk & (sdf >= np.percentile(sdf[msk], 75)) & ~solid if msk.any() else np.zeros(n, bool)
        cr = (float(np.median(S["speed"][-1, core] / np.maximum(S["speed"][0, core], 1e-9)))
              if core.any() else float("nan"))
        print(f"    {name:10s}  {cells[0]:8.3f}  {cells[1]:8.3f}  {cells[2]:8.3f}  "
              f"{cells[3]:8.3f}  {cr:12.3f}")

    # Opposite wall in the same axial band: wall nodes far from blinds in Euclidean space
    if blinds.size:
        tree_b = cKDTree(pos[blinds])
        dmin = tree_b.query(pos)[0]
        opp = wall & band & (dmin > np.percentile(dmin[wall & band], 75))
        amp_here = S["sr"][-1, blinds] / np.maximum(S["sr"][0, blinds], 1e-9)
        if opp.any():
            amp_opp = S["sr"][-1, opp] / np.maximum(S["sr"][0, opp], 1e-9)
            g0_opp = S["gate"][0, opp] > 0
            gF_opp = S["gate"][-1, opp] > 0
            print(f"  opposite wall in blind band: n={int(opp.sum())}  "
                  f"amp med {float(np.median(amp_opp)):.3f}  "
                  f"gate  {100 * float(g0_opp.mean()):.0f}% -> {100 * float(gF_opp.mean()):.0f}%")
        print(f"  blinds themselves amp med {float(np.median(amp_here)):.3f}")

    # Nearest gelled WALL at tG: t=0-gated (independent station) vs ungated (march)
    g0w = S["gate"][0] * wall
    print("\n  nearest gelled wall at tG (euclidean nearest occupied wall node):")
    n_t0 = n_march = n_self = n_none = 0
    for k, nd in enumerate(blinds):
        tg = int(tG[k])
        if tg >= T:
            continue
        occ_wl = hot[tg] & wall
        if not occ_wl.any():
            n_none += 1
            print(f"    b{k:02d} tG={tg:3d}  wound_only (no gelled wall yet)")
            continue
        dhop = hop_distance(occ_wl, A, max_h=24)[nd]
        ids = np.flatnonzero(occ_wl)
        src_id = ids[int(np.argmin(np.linalg.norm(pos[ids] - pos[nd], axis=1)))]
        if dhop == 0:
            kind = "self"
            n_self += 1
        elif g0w[src_id] > 0:
            kind = "t0_gated"
            n_t0 += 1
        else:
            kind = "ungated_march"
            n_march += 1
        print(f"    b{k:02d} tG={tg:3d}  hop_wall={dhop:.0f}  nearest={kind}")
    print(f"  counts  self={n_self}  t0_gated={n_t0}  ungated_march={n_march}  wound_only={n_none}")

    _kernel_amp_table(S, wall, hot, solid, A, T)
    _flip_summary(S, wall, hot, solid, wnd)


def _kernel_amp_table(S, wall, hot, solid, A, T):
    """amp(h) on t=0-ungated wall that is not yet gelled, vs hop to committed solid."""
    print("\n  kernel amp(h) on ungated wall that is not yet gelled (pooled over time):")
    print(f"    {'h':>4s}  {'n':>7s}  {'amp_med':>8s}  {'amp_p25':>8s}  {'amp_p75':>8s}  {'B%':>5s}")
    buckets = {h: [] for h in range(0, 9)}
    buckets["9+"] = []
    b_open = {h: [0, 0] for h in list(range(0, 9)) + ["9+"]}
    sample_t = list(range(0, T, max(T // 16, 1)))
    if sample_t[-1] != T - 1:
        sample_t.append(T - 1)
    ung = wall & (S["gate"][0] <= 0)
    for ti in sample_t:
        occ_s = hot[ti] & solid
        if not occ_s.any():
            continue
        dist = hop_distance(occ_s, A, max_h=12)
        live = ung & ~hot[ti]
        if not live.any():
            continue
        amp = S["sr"][ti, live] / np.maximum(S["sr"][0, live], 1e-9)
        bb = S["B"][ti, live] > 0
        hh = dist[live]
        for h in range(0, 9):
            m = hh == h
            if m.any():
                buckets[h].append(amp[m])
                b_open[h][0] += int(bb[m].sum())
                b_open[h][1] += int(m.sum())
        m = hh >= 9
        if m.any():
            buckets["9+"].append(amp[m])
            b_open["9+"][0] += int(bb[m].sum())
            b_open["9+"][1] += int(m.sum())
    for h in list(range(0, 9)) + ["9+"]:
        if not buckets[h]:
            continue
        a = np.concatenate(buckets[h])
        n_obs = b_open[h][1]
        print(f"    {str(h):>4s}  {a.size:7d}  {float(np.median(a)):8.3f}  "
              f"{float(np.percentile(a, 25)):8.3f}  {float(np.percentile(a, 75)):8.3f}  "
              f"{100 * b_open[h][0] / max(n_obs, 1):4.0f}%")


def _flip_summary(S, wall, hot, solid, wnd):
    """Vessel-wide: t=0-ungated wall whose GT gate later opens, by branch."""
    g0 = S["gate"][0] * wall
    ung = wall & (g0 <= 0)
    opened = ung & (S["gate"][-1] > 0)
    a_only = opened & (S["A"][-1] > 0) & (S["B"][-1] <= 0)
    b_only = opened & (S["B"][-1] > 0) & (S["A"][-1] <= 0)
    both = opened & (S["A"][-1] > 0) & (S["B"][-1] > 0)
    print(f"  vessel ungated wall {int(ung.sum())}: GT-final extra gates {int(opened.sum())}  "
          f"A-only {int(a_only.sum())}  B-only {int(b_only.sum())}  A+B {int(both.sum())}")
    # extra B among those that never gel
    never = ung & ~hot[-1] & solid
    fp_b = never & (S["B"][-1] > 0)
    print(f"  extra B on ungated wall that never gels: {int(fp_b.sum())}")


def _cohort_snap(bio, phys, stems, label: str) -> None:
    print("=" * 92)
    print(f"{label}: t=0 vs t_final GT gate flips on t=0-ungated wall")
    print(f"  {'stem':22s}  {'ung':>5s}  {'dA':>5s}  {'dB':>5s}  {'dG':>5s}  "
          f"{'GT+':>5s}  {'dG&GT+':>7s}  {'dG&~GT':>7s}  {'amp_med':>7s}")
    tot = dict(ung=0, dA=0, dB=0, dG=0, gt=0, tp=0, fp=0)
    for stem in stems:
        d = _load(stem)
        if d is None:
            continue
        wall = d.mask_wall.reshape(-1).bool().cpu().numpy()
        pos = node_positions(d)
        ei = d.edge_index.detach().cpu().numpy()
        Dx, Dy = build_mls_gradient(pos, ei, hops=3)
        u_ref = float(d.u_ref.reshape(-1)[0])
        d_bar = float(d.d_bar.reshape(-1)[0])
        scale = u_ref / d_bar
        T = int(d.y.shape[0])
        y = d.y.detach().cpu().numpy().astype(np.float64)

        def sr_ds(ti):
            u, v = y[ti, :, 0], y[ti, :, 1]
            sr = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * scale
            dsrx = (Dx @ sr) / (d_bar * M_TO_CM)
            return sr, dsrx

        sr0, ds0 = sr_ds(0)
        srF, dsF = sr_ds(T - 1)
        g0 = gate_from_shear(sr0, ds0, bio, wall=wall)
        gF = gate_from_shear(srF, dsF, bio, wall=wall)
        A0 = (ds0 < float(bio.sgt) / M_TO_CM) & wall
        AF = (dsF < float(bio.sgt) / M_TO_CM) & wall
        B0 = (sr0 < float(bio.lss)) & wall
        BF = (srF < float(bio.lss)) & wall
        ung = wall & (g0 <= 0)
        dA = ung & (~A0) & AF
        dB = ung & (~B0) & BF
        dG = ung & (gF > 0)
        gt = gt_clot_phi_at_time(d, T - 1, phys).numpy() > 0.5
        amp = srF[ung] / np.maximum(sr0[ung], 1e-9) if ung.any() else np.array([1.0])
        print(f"  {stem:22s}  {int(ung.sum()):5d}  {int(dA.sum()):5d}  {int(dB.sum()):5d}  "
              f"{int(dG.sum()):5d}  {int((ung & gt).sum()):5d}  "
              f"{int((dG & gt).sum()):7d}  {int((dG & ~gt).sum()):7d}  "
              f"{float(np.median(amp)):7.3f}")
        tot["ung"] += int(ung.sum())
        tot["dA"] += int(dA.sum())
        tot["dB"] += int(dB.sum())
        tot["dG"] += int(dG.sum())
        tot["gt"] += int((ung & gt).sum())
        tot["tp"] += int((dG & gt).sum())
        tot["fp"] += int((dG & ~gt).sum())
    print(f"  {'SUM':22s}  {tot['ung']:5d}  {tot['dA']:5d}  {tot['dB']:5d}  "
          f"{tot['dG']:5d}  {tot['gt']:5d}  {tot['tp']:7d}  {tot['fp']:7d}")
    print("  dA/dB/dG = t=0-ungated wall that GAIN that branch by t_final.  "
          "dG&GT+ is the analog of 003 blinds (flow-opened and they do clot).")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=list(WOUND_STEMS))
    ap.add_argument("--no-cohort", action="store_true")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    print("[i] non-local flow->gate design EDA  (GT velocity, MLS hops=3)")
    print("    A = sep branch (dsrx < sgt)*coef*|dsrx|   B = lss indicator (sr < lss)\n")
    for stem in args.stems:
        d = _load(stem)
        if d is None:
            print(f"[miss] {stem}")
            continue
        _run_wound(stem, d, bio, phys)
        print()
    if not args.no_cohort:
        fit = [s for s in FIT if (PACKS / f"{s}.pt").exists()][:12]
        free = [s for s in CLOT_FREE if (PACKS / f"{s}.pt").exists()]
        extra = ["patient012"]
        _cohort_snap(bio, phys, fit, "FIT clot-carrying (n<=12)")
        print()
        _cohort_snap(bio, phys, extra, "patient012 (clot-rich no-wound control)")
        print()
        _cohort_snap(bio, phys, free, "clot-free (extra gates must stay ~0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
