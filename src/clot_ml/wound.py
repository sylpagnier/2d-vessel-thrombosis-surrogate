"""The wound complement to ``clot_gnn_v4``: an ungated surface ODE with a learned rate.

WHY A COMPLEMENT AND NOT A RETRAIN.  Three measurements in
[docs/WOUND_PROGRESS.md](../../docs/WOUND_PROGRESS.md) fix the shape of this module:

1. **The wound does not perturb the healthy wall.** Wall-only F1 of the t=0 gate on the
   wound vessels (0.710 / 0.832 / 0.874) sits inside the no-wound cohort's range
   (0.549-0.816). So v4 stays valid everywhere except the injured segment, and the right
   move is to compose rather than retrain -- especially at n=3 vessels.
2. **The set is free.** COMSOL's wound law is the wall law with the gates deleted
   (``G_wound == 1``), and running the shipped surface ODE with that one substitution gives
   wound recall **1.000** at precision 1.000 on all three vessels. Nothing to learn.
3. **The clock is what is wrong, and it is wrong in a structured way.** ``G == 1`` is ~2x
   too slow (onset step 49 against 24). A flat ``G == 2`` fixes onset to +-1 step on 001/002
   but then undershoots the final magnitude 2.4x against 9.0x crit -- one rate scalar cannot
   fit both, because the real trajectory changes slope at gelation: ``mu1`` steps 80x, the
   near-wall flow stalls, ``sr`` collapses 148 -> 18 /s and the ordinary low-shear gate opens
   on top of the wound law (WOUND_PROGRESS 3.3).

Hence the model here: a **two-regime gate** with a per-node learned correction.

    G_i(t) = G_pre_i  +  (G_post_i - G_pre_i) * sigma((Mat_i/crit - 1)/tau)     on the wound
    G_i(t) = gate_i                                                             on healthy wall

    log G_pre_i  = log G_pre0  + r_pre(x_i)
    log G_post_i = log G_post0 + r_post(x_i)

``G_pre0``/``G_post0`` are two global constants; ``r_*`` are the network's per-node
residuals, read off t=0 features only. The ODE is unchanged COMSOL physics and supplies
monotonicity, the gelation threshold and the set; the network only moves the *rate*. That is
the physics-informed part: the learned quantity is a coefficient inside a conservation law,
not the label.

DEPLOY LEGALITY. Everything here is a function of the t=0 flow field (``flow_source`` is
threaded through to ``t0_flow_fields``) plus mesh geometry. No GT velocity after t=0, no GT
species, no GT ``Mat``. The gate feedback in §3.3 is driven by the rollout's **own** ``Mat``,
so the loop closes on the model's state rather than on the answer.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.clot_ml.features import adjacency, hop_distance, khop_stats
from src.data_gen.lib.mesh_wls import solid_boundary_nodes

M_TO_CM = 100.0
PER_M2_TO_PER_CM2 = 1.0e-4

#: Base constants for the two-regime gate. ``G_pre0 = 2`` is not tuned -- it is
#: ``ungated (1) + the low-shear branch (1)``, the value the mechanism predicts, and it lands
#: onset within +1 step of GT on both healthy-flow vessels. ``G_post0`` is fitted.
#: Resting-platelet renewal coefficient, initial value for the fit.  The wall-AP closure
#: ships `C = 62.42` against a consumption of `gate * k_as`; `k_as / k_rs = 12.16` in the
#: config, so the same physical balance on the resting pool sits near `62.42 * 12 = 750`.
#: That is an initialisation, not a claim -- `scripts/train_wound_rate.py` fits it
#: leave-one-vessel-out alongside the two gate constants.
RP_C0 = 750.0

G_PRE0 = 2.0
G_POST0 = 10.0
#: Softness of the gelation switch, in units of ``Mat/crit``. Small enough to behave like the
#: hard indicator, large enough that onset timing has a gradient.
SWITCH_TAU = 0.05
#: Off-wall lag behind the owning wound node, in grid steps (WOUND_PROGRESS 4; the cohort
#: measurement is "off-wall lags its owner by a median +4 of 11 grid steps").
OFFWALL_LAG_FRAC = 0.04

#: How many corner shells the recursive lumen rule may consider.  NOT a tuned depth -- the
#: depth that actually commits is decided per vessel by `off_att**k * Mat_wound >= crit`, and
#: at `off_att = 0.16` the fourth shell needs `Mat_wound >= 1526 * crit`, which no vessel
#: measured here approaches (the largest is 104x).  This is only a loop bound.
RECURSIVE_MAX_DEPTH = 4
#: Mesh-graph hops over which a neighbour's clot can open a wound node's gate. Set by the
#: measurement it has to separate -- 12-14 hops (externally triggered) against 61
#: (self-triggered) -- not by a fit; the answer is flat over 25-40 (WOUND_PROGRESS 11).
TRIGGER_HOPS = 25


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------
def wound_mask(data) -> np.ndarray:
    m = getattr(data, "mask_wound", None)
    if m is None or not torch.is_tensor(m) or m.numel() == 0:
        return np.zeros(int(data.num_nodes), dtype=bool)
    return m.reshape(-1).bool().cpu().numpy()


def solid_mask(data) -> np.ndarray:
    """Healthy wall union wound -- every no-slip boundary node (COMSOL ``uni1``).

    Delegates to the canonical accessor.  Three modules had grown their own copy of this
    union and one of them (``src/clot_ml/features.py``) was silently missing it, which is
    what MODEL_REVIEW_2026-08-22 5b.3 found; there is one implementation now.
    """
    return solid_boundary_nodes(data)


def has_wound(data) -> bool:
    return bool(wound_mask(data).any())


# ---------------------------------------------------------------------------
# features for the rate network
# ---------------------------------------------------------------------------
WOUND_FEATURES = (
    "log_sr", "sr_over_lss", "log_absdsrx", "gate_low", "gate_sep", "log_gate",
    "speed", "sdf_nd", "width_nd",
    "wall_gate_frac_h4", "wall_gate_frac_h16", "wall_gate_frac_vessel",
    "along_wound", "hop_to_wound_edge", "wound_len_frac",
)


def wound_features(data, f0, bio_cfg) -> np.ndarray:
    """Per-node t=0 features, ``[N, len(WOUND_FEATURES)]``.

    ``wall_gate_frac_vessel`` is the one that carries the across-vessel signal: the wound on
    ``wound_comsol003`` gels 5x faster than the other two and its *vessel* is 35% gated at
    t=0 against 16% / 12%. The k-hop versions let the network localise that.
    """
    n = int(data.num_nodes)
    ei = data.edge_index.detach().cpu().numpy()
    A = adjacency(ei, n)
    wnd, wall = wound_mask(data), data.mask_wall.reshape(-1).bool().cpu().numpy()
    pos = data.x[:, :2].detach().cpu().numpy().astype(np.float64)

    lss = float(bio_cfg.lss)
    sgt = abs(float(bio_cfg.sgt) / M_TO_CM)
    gate_bin = (f0.gate > 0).astype(np.float64)
    wall_gate = gate_bin * wall
    h4, _ = khop_stats(A, wall_gate, 4)
    h16, _ = khop_stats(A, wall_gate, 16)
    vessel_frac = float(gate_bin[wall].mean()) if wall.any() else 0.0

    xs = data.x.detach().cpu().numpy()
    ch = {c: i for i, c in enumerate(data.x_channel_names.split(","))}
    sdf = xs[:, ch["sdf_nd"]] if "sdf_nd" in ch else np.zeros(n, np.float32)
    width = xs[:, ch["width_nd"]] if "width_nd" in ch else np.zeros(n, np.float32)

    # The velocity MUST come from the T0Fields the caller resolved, not from `data.y[0]`:
    # under `flow="pred"` the latter is GT and would leak COMSOL's answer into a channel the
    # module's docstring promises is deploy-legal.  No fallback -- a missing field is a
    # caller bug, and silently substituting GT is the failure this check exists to prevent.
    if f0.u is None or f0.v is None:
        raise ValueError("wound_features needs T0Fields carrying u/v; build it with "
                         "t0_flow_fields(..., flow_source=...) rather than by hand")
    u = np.asarray(f0.u, np.float64)
    v = np.asarray(f0.v, np.float64)

    # position along the wound, and how deep inside it a node sits
    along = np.zeros(n)
    hop_edge = np.zeros(n)
    len_frac = 0.0
    if wnd.any():
        xw = pos[wnd, 0]
        lo, hi = float(xw.min()), float(xw.max())
        along = np.clip((pos[:, 0] - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        # hops from the wound's boundary with the healthy wall: interior of the patch is
        # where the ungated law has been running longest without an edge effect.
        edge_seed = wnd & (A @ (~wnd & wall).astype(np.int8) > 0)
        hop_edge = np.minimum(hop_distance(edge_seed, A), 12).astype(np.float64)
        len_frac = (hi - lo) / max(float(np.ptp(pos[:, 0])), 1e-9)

    cols = {
        "log_sr": np.log1p(np.maximum(f0.sr, 0.0)),
        "sr_over_lss": np.clip(f0.sr / lss, 0, 40),
        "log_absdsrx": np.log1p(np.abs(f0.dsrx)),
        "gate_low": f0.gate_low,
        "gate_sep": f0.gate_sep,
        "log_gate": np.log1p(np.maximum(f0.gate, 0.0)),
        "speed": np.hypot(u, v),
        "sdf_nd": sdf.astype(np.float64),
        "width_nd": width.astype(np.float64),
        "wall_gate_frac_h4": h4,
        "wall_gate_frac_h16": h16,
        "wall_gate_frac_vessel": np.full(n, vessel_frac),
        "along_wound": along,
        "hop_to_wound_edge": hop_edge,
        "wound_len_frac": np.full(n, len_frac),
        "_dsrx_scale": np.full(n, sgt),  # not a column; keeps sgt referenced for clarity
    }
    return np.stack([cols[k] for k in WOUND_FEATURES], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# the differentiable two-regime surface ODE
# ---------------------------------------------------------------------------
@dataclass
class OdeConstants:
    """Everything ``integrate_mat_trajectory`` reads out of the config, pulled out once."""

    k_rs: float
    k_as: float
    k_aa: float
    minf: float
    da: float
    crit: float
    gate_s: float
    slope: float
    ap_C: float
    ap_q: float
    #: RESTING-platelet renewal coefficient, the same Damkohler balance the wall-AP closure
    #: applies to `ap` (`src/core_physics/ap_closure.py`) but with `k_rs` as the consumption
    #: constant.  ``0.0`` reproduces the frozen-`rp` model bit-for-bit, which is every
    #: artifact promoted before 2026-09-03.  See `RP_C_DEFAULT` for why it exists.
    rp_C: float = 0.0

    @classmethod
    def from_cfg(cls, bio_cfg, da_scale: float, ap_closure,
                 rp_C: float = 0.0) -> "OdeConstants":
        return cls(
            k_rs=float(bio_cfg.k_rs) * M_TO_CM,
            k_as=float(bio_cfg.k_as) * M_TO_CM,
            k_aa=float(bio_cfg.k_aa) * M_TO_CM,
            minf=float(bio_cfg.Minf) * PER_M2_TO_PER_CM2,
            da=float(bio_cfg.surface_damkohler) * float(da_scale),
            crit=float(bio_cfg.viscosity_mat_crit),
            gate_s=float(bio_cfg.surface_time_gate_s),
            slope=float(bio_cfg.surface_time_gate_slope),
            ap_C=float(ap_closure.C),
            ap_q=float(ap_closure.q),
            rp_C=float(rp_C),
        )


def mat_trajectory_torch(
    *,
    t: torch.Tensor,
    gate_pre: torch.Tensor,
    gate_post: torch.Tensor,
    rp: torch.Tensor,
    ap: torch.Tensor,
    sr: torch.Tensor,
    C: OdeConstants,
    tau: float = SWITCH_TAU,
    ext_weight: torch.Tensor | None = None,
    reach: torch.Tensor | None = None,
    rp_C: "torch.Tensor | float | None" = None,
) -> torch.Tensor:
    """``[T, N]`` ``Mat`` in COMSOL model units, differentiable in the two gate fields.

    An exact torch mirror of :func:`integrate_mat_trajectory` (``static`` ap-closure kernel,
    no washout), with one addition: the gate moves from ``gate_pre`` to ``gate_post`` as the
    node's own ``Mat`` crosses ``crit``. With ``gate_post == gate_pre`` and a hard switch it
    reproduces the numpy path to float tolerance -- pinned by
    ``src/tests/test_wound_complement.py``.

    The switch is soft (sigmoid of width ``tau`` in units of ``Mat/crit``) so the *timing* of
    the transition carries a gradient; ``tau = 0.05`` is 5% of the threshold and behaves like
    the indicator.

    ``ext_weight`` / ``reach`` add the NEIGHBOUR trigger. The self-trigger above assumes a
    node's gate opens because of its *own* clot, which is what happens on a wound in healthy
    flow. It is not what happens when the wound sits near wall that was already going to clot:
    on ``wound_comsol003`` 21 wall nodes gel at step 2, the wound's shear falls 128 -> 84 /s
    and its gate opens to 42% by step 3 -- all before the wound's own gelation at step 5
    (WOUND_PROGRESS 11). ``ext_weight`` is ``[T, M]`` committed weight in [0, 1] on some other
    set of nodes and ``reach`` is ``[N, M]`` boolean neighbourhood; the switch then takes the
    max of the node's own weight and its neighbourhood's. Both ``None`` reproduces the
    self-triggered model exactly.
    """
    # `rp_C` overrides `C.rp_C` so the FIT can pass a live tensor (the coefficient is
    # learned) while DEPLOY passes a plain float on the constants object.  Same value, two
    # lifetimes.
    rc = C.rp_C if rp_C is None else rp_C
    rc_off = isinstance(rc, float) and rc == 0.0
    n = gate_pre.shape[0]
    dev, dt = gate_pre.device, gate_pre.dtype
    mas = torch.zeros(n, device=dev, dtype=dt)
    mat = torch.zeros(n, device=dev, dtype=dt)
    traj = [mat]
    sr_f = torch.clamp(sr, min=1.0e-3)
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        step2t = torch.sigmoid(torch.clamp((t[i] - C.gate_s) * C.slope, -50, 50))
        w = torch.sigmoid((mat / C.crit - 1.0) / tau)
        if ext_weight is not None and reach is not None:
            nb = (reach * ext_weight[i].unsqueeze(0)).max(dim=1).values
            w = torch.maximum(w, nb.to(w.dtype))
        g = gate_pre + (gate_post - gate_pre) * w
        sat = torch.clamp(1.0 - mas / C.minf, 0.0, 1.0)
        # static ap-closure kernel: consumption = g * k_as, x = consumption / sr^q
        ap_i = ap / (1.0 + C.ap_C * (g * C.k_as) / torch.pow(sr_f, C.ap_q))
        # THE SAME BALANCE ON THE RESTING POOL.  `rp` is the FEEDSTOCK -- `ap` is what the
        # reaction produces, which is why `ap` accumulates in a dead zone while `rp` does
        # not.  Measured 2026-09-03 over all six wounds (374 nodes): resting-platelet
        # survival after one 150 s interval is 0.994 wherever wall shear exceeds 20 /s and
        # 0.0000 below 5 /s, and it separates the wound nodes that ever clot from those that
        # never do -- 100.0% of nodes with RP survival >= 0.90 clot against 67.0% below it.
        # `rp_C = 0` reproduces the frozen-`rp` model bit-for-bit.
        rp_i = (rp if rc_off
                else rp / (1.0 + rc * (g * C.k_rs) / torch.pow(sr_f, C.ap_q)))
        dep = sat * (C.k_rs * rp_i + C.k_as * ap_i)
        auto = (mas / C.minf) * C.k_aa * ap_i
        mas = mas + h * C.da * g * dep * step2t
        mat = mat + h * g * C.da * (dep + auto) * step2t
        traj.append(mat)
    return torch.stack(traj, dim=0)


# ---------------------------------------------------------------------------
# the learned rate
# ---------------------------------------------------------------------------
class WoundRateNet(nn.Module):
    """Per-node log-rate residuals on the two-regime gate.

    Deliberately small: 3 vessels and 186 wound nodes do not support anything else. The
    global ``log G_pre0`` / ``log G_post0`` are learnable too, so with ``hidden=0`` this
    degrades exactly to the two-constant physics baseline -- which is the arm it has to beat.
    """

    def __init__(self, in_dim: int, hidden: int = 32, *,
                 g_pre0: float = G_PRE0, g_post0: float = G_POST0,
                 max_resid: float = 1.5, fit_rp_C: bool = False,
                 rp_C0: float = RP_C0):
        super().__init__()
        self.log_g_pre0 = nn.Parameter(torch.tensor(float(np.log(g_pre0))))
        self.log_g_post0 = nn.Parameter(torch.tensor(float(np.log(g_post0))))
        # THE THIRD SCALAR, and it is a physical one: the resting-platelet renewal
        # coefficient of the same Damkohler balance the wall-AP closure already applies to
        # `ap`.  Held OUT of the model unless asked for, so the two-constant arm this has to
        # beat stays exactly what it was.  `rp_C -> 0` recovers that arm continuously, which
        # is what makes the comparison a nested one rather than two unrelated models.
        self.log_rp_C = (nn.Parameter(torch.tensor(float(np.log(rp_C0))))
                         if fit_rp_C else None)
        self.max_resid = float(max_resid)
        if hidden > 0:
            self.body = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, 2),
            )
            # zero-init the head: the net starts exactly at the physics baseline.
            nn.init.zeros_(self.body[-1].weight)
            nn.init.zeros_(self.body[-1].bias)
        else:
            self.body = None

    @property
    def rp_C(self) -> float:
        """Fitted resting-platelet renewal coefficient, or 0.0 when it is not in the model."""
        return 0.0 if self.log_rp_C is None else float(torch.exp(self.log_rp_C.detach()))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` is ``[N, F]`` normalised features -> ``(G_pre, G_post)`` each ``[N]``."""
        base_pre = self.log_g_pre0.expand(x.shape[0])
        base_post = self.log_g_post0.expand(x.shape[0])
        if self.body is None:
            return torch.exp(base_pre), torch.exp(base_post)
        r = torch.tanh(self.body(x)) * self.max_resid
        g_pre = torch.exp(base_pre + r[:, 0])
        g_post = torch.exp(base_post + r[:, 1])
        # the post-gelation rate cannot be slower than the pre-gelation one: the gate only
        # ever gains the low-shear branch when the clot stalls the flow.
        return g_pre, torch.maximum(g_post, g_pre)


# ---------------------------------------------------------------------------
# assembling one vessel
# ---------------------------------------------------------------------------
def prepare_vessel(data, bio_cfg, *, flow: str = "gt", hops: int = 3,
                   da_scale: float | None = None) -> dict:
    """Everything the wound ODE and the rate net need for one pack, computed once."""
    from src.core_physics.ap_closure import SHIPPED, SHIPPED_DA_SCALE
    from src.core_physics.physics_wall_model import t0_flow_fields, wall_platelet_constants

    f0 = t0_flow_fields(data, bio_cfg, hops=hops, flow_source=flow)
    rp, ap = wall_platelet_constants(data, bio_cfg)
    C = OdeConstants.from_cfg(bio_cfg, SHIPPED_DA_SCALE if da_scale is None else da_scale,
                              SHIPPED)
    wnd, wall = wound_mask(data), data.mask_wall.reshape(-1).bool().cpu().numpy()
    return dict(
        f0=f0, C=C, wound=wnd, wall=wall, solid=wall | wnd,
        t=torch.tensor(data.t.reshape(-1).detach().cpu().numpy(), dtype=torch.float64),
        rp=torch.tensor(np.asarray(rp, np.float64)),
        ap=torch.tensor(np.asarray(ap, np.float64)),
        sr=torch.tensor(np.asarray(f0.sr, np.float64)),
        gate=torch.tensor(np.asarray(f0.gate, np.float64)),
        feats=wound_features(data, f0, bio_cfg),
        edge_index=data.edge_index.detach().cpu().numpy(),
        pos=data.x[:, :2].detach().cpu().numpy().astype(np.float64),
        n=int(data.num_nodes),
    )


def gate_fields(V: dict, g_pre: torch.Tensor, g_post: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter the wound's learned rates into full-mesh gate fields.

    Healthy wall keeps the shipped ``t=0`` gate in both regimes -- this module changes
    nothing v4 already models. Off-boundary nodes get zero: ``Mat`` is a wall-sourced field.
    """
    wnd = torch.tensor(V["wound"])
    solid = torch.tensor(V["solid"]).to(V["gate"].dtype)
    base = V["gate"] * solid
    pre = torch.where(wnd, g_pre.to(base.dtype), base)
    post = torch.where(wnd, g_post.to(base.dtype), base)
    return pre * solid, post * solid


def onset_from_traj(traj: np.ndarray, crit: float) -> np.ndarray:
    """First index at or above ``crit``; ``T`` (i.e. never) where it does not cross."""
    hot = traj >= crit
    T = traj.shape[0]
    return np.where(hot.any(axis=0), hot.argmax(axis=0), T).astype(np.float64)


def wound_rate_blockage(data, bio_cfg, *, g_pre: float = G_PRE0, g_post: float = G_POST0,
                        inner=None):
    """``blockage(mat, gate0, step) -> gate`` running the wound at its FITTED rate.

    WHY THIS EXISTS.  WOUND_PROGRESS 14.6 gave the injured patch a ``Mat`` source inside the
    shared ODE, with COMSOL's own ``srf2`` prefactor of a hard 1.  That is the correct
    transcription of the law and it leaves the stack saying two different things about the
    same patch: :func:`predict_wound_series` integrates the fitted two-regime rate
    (``G_pre ~ 2 -> G_post ~ 14``) while :func:`~src.clot_ml.temporal.ode_trajectory`
    integrates 1, so every ``Mat`` consumer that is not the complement itself -- ``mat_phys``,
    ``mat_owner_t``, ``mat_self_t``, the advective source, ``oon`` -- sees a wound an order of
    magnitude too quiet.  Measured at the final time, wound ``Mat``/crit:

        vessel   ODE at prefactor 1   this   GT
        001            1.35          12.72   9.04
        002            1.40          11.49   8.70
        003            2.31          17.19  103.84

    That matters because the off-wall commitment is a MAGNITUDE condition
    (``Mat_owner >= crit / off_att``, 6.25x crit at the shipped attenuation).  At 1.35x crit
    no wound-owned lumen node can ever clear it, which is exactly WOUND_PROGRESS 12.2's
    finding that the zero-parameter physics arm scores 0.0000 on ``w_lum``.

    ``g_pre``/``g_post`` are the complement's own constants, so the two paths cannot diverge.
    The switch is the node's own ``Mat`` crossing ``crit`` -- the same self-trigger
    :func:`mat_trajectory_torch` uses, hard rather than sigmoid because the numpy ODE has no
    gradient to preserve.

    ``inner`` composes with an existing blockage (near-stall, gelation wake): the wound patch
    is rewritten AFTER it, because ``srf2`` is ungated and must not be replaced by a stalled
    ``srf1``.  Returns ``inner`` unchanged on a pack with no wound, so this is a structural
    no-op off a wound pack rather than a numerically-small one.
    """
    wnd = wound_mask(data)
    if not wnd.any():
        return inner
    crit = float(bio_cfg.viscosity_mat_crit)
    pre, post = float(g_pre), float(max(g_post, g_pre))

    def blockage(mat, gate0, step):
        g = gate0 if inner is None else inner(mat, gate0, step)
        g = np.array(g, dtype=np.float64, copy=True)
        g[wnd] = np.where(np.asarray(mat, dtype=np.float64)[wnd] >= crit, post, pre)
        return g

    return blockage


# ---------------------------------------------------------------------------
# ownership + composition with v4
# ---------------------------------------------------------------------------
def wound_owned_masks(data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(wound, owned_off, owner)`` -- the nodes this module takes responsibility for.

    ``owned_off`` is the first corner shell off the **solid** boundary whose nearest solid
    node is a wound node. Topological, not a distance: the packs are quadratic meshes and a
    length-based shell straddles the empty wall-normal mid-side family
    (PHASE7_FINDINGS 8). On 001/002 the GT wound clot is exactly the wound plus this one
    shell; on 003 it runs deeper, which the attenuation rule below handles by magnitude.
    """
    from scipy.spatial import cKDTree

    from src.core_physics.physics_lumen_model import first_corner_shell

    pos = data.x[:, :2].detach().cpu().numpy().astype(np.float64)
    ei = data.edge_index.detach().cpu().numpy()
    wnd, solid = wound_mask(data), solid_mask(data)
    owner = np.zeros(len(solid), dtype=np.int64)
    if solid.any():
        _, j = cKDTree(pos[solid]).query(pos)
        owner = np.flatnonzero(solid)[j]
    if not wnd.any():
        return wnd, np.zeros_like(wnd), owner
    shell = first_corner_shell(pos, solid, ei)
    return wnd, shell & ~solid & wnd[owner], owner


def wound_transport_attenuation(V: dict, wnd: np.ndarray, owner: np.ndarray,
                               src_weight: np.ndarray | None = None) -> np.ndarray:
    """Per-node ``Mat_off / Mat_owner``, computed from COMSOL's own transport operator.

    THIS IS WHAT REPLACES THE 0.16 CONSTANT (roadmap item C1, MODEL_REVIEW 5b.5(2a)).

    `off_att = 0.16` is the cohort median of `Mat_off / Mat_owner` (PHASE7 12.5) and it is a
    *nearest-wall-node* rule: it transports information along the mesh NORMAL, the one
    direction `dMat/dt + u.grad(Mat) = 0` does not transport along.  PHASE7 12.5 also measured
    that the ratio spans 0.12-0.19 **within** a single vessel, and near a threshold that
    spread is the whole off-wall gap.

    So solve the operator instead.  `src/clot_ml/transport.py` already discretises it; seed it
    on the WOUND boundary alone -- the healthy wall's contribution is v4's job and adding it
    here would double-count -- and read the ratio each node reaches relative to its owner.
    Zero parameters: the only constant is `crit`, which is physics.

    The operator is LINEAR and the flow is frozen at t=0, so this ratio is time-independent
    for a fixed source SHAPE and one solve serves every stored time.  `src_weight` sets that
    shape (default: uniform over the wound, i.e. the geometric answer).
    """
    from src.clot_ml.features_v4 import horizon_for
    from src.clot_ml.transport import _node_volume, _solve_upwind, upwind_operator

    pos = np.asarray(V["pos"], np.float64)
    ei = np.asarray(V["edge_index"])
    u = np.asarray(V["f0"].u, np.float64)
    v = np.asarray(V["f0"].v, np.float64)
    solid = np.asarray(V["solid"], bool)

    ws = np.zeros(len(u), np.float64)
    ws[wnd] = 1.0 if src_weight is None else np.asarray(src_weight, np.float64)[wnd]
    if not ws.any():
        return np.zeros(len(u))

    H = horizon_for(pos, u, v, solid)
    F, out = upwind_operator(pos, ei, u, v)
    vol = _node_volume(pos, ei)
    adv = _solve_upwind(F, out, ws * vol, vol, H)
    adv = np.maximum(adv, 0.0)

    # relative to the OWNING wound node's own transported value, so the quantity is the
    # dimensionless attenuation the 0.16 constant stands for, not a raw concentration whose
    # units depend on the horizon.  PHASE10 14.4 killed a threshold on an unnormalised
    # transported field; this keeps the decision in `Mat` units.
    den = np.maximum(adv[owner], 1e-30)
    return np.clip(adv / den, 0.0, 1.0)


def wound_shells(data, max_depth: int = 4) -> tuple[list, np.ndarray]:
    """Successive corner shells off the wound, and every node's owning wound node.

    THE MESH IS QUADRATIC, so the shells sit at EVEN hop distances: the odd rows are P2
    mid-side nodes, which carry structurally zero ``Mat`` (PHASE7 8.5).  Measured on the three
    wound packs, GT ``Mat`` relative to the wound's own, by hop:

        hop      1       2       3       4       5       6
        001   0.0000  0.2028  0.0000  0.0384  0.0000  0.0070
        002   0.0000  0.2007  0.0000  0.0361  0.0000  0.0066
        003   0.0000  0.1306  0.0000  0.0248  0.0000  0.0045

    so shell ``k`` is hop ``2k``, and each shell attenuates the previous by **0.180-0.190** on
    every vessel and at every depth.  That is the same 0.16-class constant the first shell
    uses, applied recursively -- not a new parameter.
    """
    from src.clot_ml.features import adjacency, hop_distance

    wnd = wound_mask(data)
    n = int(data.num_nodes)
    A = adjacency(data.edge_index.detach().cpu().numpy(), n)
    hop = hop_distance(wnd, A)
    solid = solid_mask(data)
    shells = [(hop == 2 * k) & ~solid for k in range(1, int(max_depth) + 1)]

    pos = data.x[:, :2].detach().cpu().numpy().astype(np.float64)
    owner = np.zeros(n, dtype=np.int64)
    wi = np.flatnonzero(wnd)
    if wi.size:
        from scipy.spatial import cKDTree
        owner = wi[cKDTree(pos[wi]).query(pos)[1]]
    return shells, owner


def predict_wound_series(
    data, bio_cfg, times, *, g_pre: float = G_PRE0, g_post: float = G_POST0,
    flow: str = "gt", off_att: float = 0.16, lag_frac: float = OFFWALL_LAG_FRAC,
    prepared: dict | None = None, trigger: str = "self", k_hops: int = TRIGGER_HOPS,
    trigger_gate_scale: float = 1.0, lumen: str = "shell",
    base_onset: np.ndarray | None = None, rp_C: float = 0.0,
    wound_ap_closure: bool = True,
) -> dict:
    """``{mask, onset, mat, owned}`` for the wound and the shell it feeds.

    ``off_att`` is the off-wall attenuation: a shell node commits when its owning wound
    node's ``Mat`` reaches ``crit / off_att``. 0.16 is the cohort constant measured in
    PHASE7 12.5 (median ``Mat_off / Mat_owner``), not a fit on these three vessels.

    ``rp_C`` is the resting-platelet renewal coefficient (:func:`mat_trajectory_torch`).
    ``0.0``, the default, reproduces every artifact promoted before 2026-09-03 bit-for-bit.

    ``wound_ap_closure=False`` drops the wall-AP CONSUMPTION closure at the wound, where it
    has the wrong sign -- the closure models a gated wall depleting `AP`, and an ungated
    wound is a net producer (docs/DEPLOYCLOT.md 5c).  Leave-one-vessel-out over six wounds it
    takes onset MAE 9.2 -> 7.2 steps and recall 0.877 -> 1.000 at an unchanged curve L1, with
    no new parameter.  ``True``, the default, is every artifact before 2026-09-03.

    ``lumen`` selects how the off-boundary nodes are decided:
      ``"shell"``      the shipped rule -- commit when the owner reaches ``crit/off_att``,
                       then lag by ``lag_frac`` of the horizon.  Two fixed constants.
      ``"transport"``  C1: the per-node attenuation comes from COMSOL's own operator
                       (:func:`wound_transport_attenuation`) and the node commits the first
                       time ``att_node * Mat_owner(t) >= crit``.  **Replaces BOTH constants**
                       -- the attenuation becomes per-node and the timing falls out of
                       evaluating the same inequality at each stored time.
      ``"union"``      shell OR transport.  Monotone in the committed set, so it can only add
                       nodes -- the safe first variant MODEL_REVIEW 5b.5 asks for, because
                       001/002 already score `w_lum` 0.97 with a thrombus that genuinely is
                       one shell, and a depth-unlimited field can only add false positives
                       there.
      ``"recursive"``  C2: the SAME attenuation applied shell after shell.  GT ``Mat`` decays
                       geometrically with depth at 0.18-0.19 per shell on all three vessels
                       (:func:`wound_shells`), so shell ``k`` commits when
                       ``off_att**k * Mat_owner(t) >= crit``.  **The depth is not a
                       parameter** -- it falls out of how much ``Mat`` the wound reaches:
                       001/002 reach 9x crit and admit one shell, 003 reaches 104x and admits
                       two, which is exactly the 214 deep GT nodes 5b.4 measured as missing.

    ``trigger`` selects what may open the two-regime gate:
      ``"self"``   the node's own ``Mat`` only -- correct for a wound in healthy flow;
      ``"wall"``   also any healthy-wall node within ``k_hops``, from the shipped wall ODE
                   (deploy-legal, and currently INERT -- see ``wall_trigger_field``);
      ``"oracle"`` the same but from GT wall ``Mat``. A ceiling, never a deploy path.
    """
    V = prepared or prepare_vessel(data, bio_cfg, flow=flow)
    crit = V["C"].crit
    wnd, owned_off, owner = wound_owned_masks(data)
    n, T = V["n"], len(V["t"])
    idx = np.flatnonzero(wnd)

    # C2: OWNERSHIP IS NOT ONE SHELL.  The shipped `owned_off` is the first corner shell, and
    # everything deeper falls to v4 -- which 5b.3 measured as blind there.  `recursive` widens
    # it to every shell the physics admits, and the depth is emergent rather than chosen: at
    # `off_att = 0.16`, shell k needs `Mat_wound >= crit / 0.16**k`, so a wound reaching 9x
    # crit owns one shell and one reaching 104x owns two.  `depth_of` records which shell each
    # owned node is in, so each commits at its own bar.
    depth_of = np.ones(n, dtype=np.int64)
    if lumen == "recursive":
        # STRICTLY ADDITIVE, and that is load-bearing.  The shipped `owned_off` comes from
        # `first_corner_shell`, which navigates the P2 mid-side family; a plain hop-2 ring is
        # a DIFFERENT and smaller set (80 nodes vs 43 on `wound_comsol001`).  Rebuilding
        # shell 1 from hops therefore replaces a good shell with a worse one and costs
        # 001/002 `w_lum` 0.0160 -- measured.  So shell 1 stays exactly as shipped and the
        # recursion only ADDS deeper rings, which keeps the 5b.5 gate satisfiable by
        # construction: on a vessel whose wound reaches 9x crit no deeper shell clears its
        # bar, so nothing is added and the result is bit-identical to `shell`.
        #
        # Additive in the committed SET, not just here: widening ownership is only half of
        # it, because `compose_with_v4` lets an owned node OVERRIDE v4 downward.  The
        # un-committed deep nodes are released again after `commits` is known, below.
        shells, owner_r = wound_shells(data, max_depth=RECURSIVE_MAX_DEPTH)
        seen = wnd | owned_off
        for k, sh in enumerate(shells[1:], start=2):
            add = sh & ~seen
            if not add.any():
                continue
            owned_off = owned_off | add
            depth_of[add] = k
            seen = seen | add
        owner = np.where(owned_off & (depth_of > 1), owner_r, owner)

    mask = np.zeros(n, dtype=bool)
    onset = np.full(n, -1.0)
    mat_full = np.zeros(n)
    if idx.size:
        gp = torch.full((idx.size,), float(g_pre), dtype=torch.float64)
        gq = torch.full((idx.size,), float(g_post), dtype=torch.float64)
        ext = rch = None
        if trigger in ("wall", "oracle", "model"):
            reach_np, wall_idx = neighbour_reach(data, k_hops=k_hops)
            if trigger == "oracle":
                field = gt_trigger_field(data, bio_cfg, V, wall_idx=wall_idx)
            elif trigger == "model":
                if base_onset is None:
                    raise ValueError("base_onset required for model trigger")
                field = np.zeros((T, len(wall_idx)), dtype=np.float64)
                on_wall = base_onset[wall_idx]
                for ti in range(T):
                    field[ti, :] = (on_wall <= ti) & (on_wall >= 0)
            else:
                field = wall_trigger_field(data, bio_cfg, V, wall_idx=wall_idx,
                                         gate_scale=trigger_gate_scale)
            ext = torch.tensor(field, dtype=torch.float64)
            rch = torch.tensor(reach_np.astype(np.float64))
        elif trigger != "self":
            raise ValueError(f"unknown trigger {trigger!r}")
        _C = V["C"] if wound_ap_closure else dataclasses.replace(V["C"], ap_C=0.0)
        traj = mat_trajectory_torch(t=V["t"], gate_pre=gp, gate_post=gq,
                                    rp=V["rp"][idx], ap=V["ap"][idx], sr=V["sr"][idx],
                                    C=_C, ext_weight=ext, reach=rch,
                                    rp_C=float(rp_C)).numpy()
        on_w = onset_from_traj(traj, crit)
        mask[idx] = traj[-1] >= crit
        onset[idx] = np.where(on_w < T, on_w, -1.0)
        mat_full[idx] = traj[-1]

        # off-wall.  `owned_off` is the shell this module takes responsibility for; which
        # of its nodes commit, and when, is what `lumen` selects.
        oi = np.flatnonzero(owned_off)
        if oi.size:
            pos_in_idx = np.searchsorted(idx, owner[oi])
            hit = np.minimum(pos_in_idx, idx.size - 1)
            ok = (pos_in_idx < idx.size) & (idx[hit] == owner[oi])

            def _shell():
                on_off = onset_from_traj(traj, crit / max(off_att, 1e-6))
                lag = max(int(round(lag_frac * T)), 1)
                src = np.where(ok, on_off[hit], T)
                return src < T, np.minimum(src + lag, T - 1)

            def _transport():
                att = wound_transport_attenuation(V, wnd, owner)[oi]
                # `traj[:, hit]` is the owner's Mat(t); the node commits the first time the
                # transported fraction of it reaches crit.  No lag constant -- the delay is
                # however long the owner takes to get there, which is the physics.
                own_t = traj[:, hit] * att[None, :]
                own_t = np.where(ok[None, :], own_t, 0.0)
                hot = own_t >= crit
                first = np.where(hot.any(0), hot.argmax(0), T)
                return first < T, np.minimum(first, T - 1)

            if lumen == "shell":
                commits, when = _shell()
            elif lumen == "recursive":
                # `owned_off` has already been widened to the shells the physics admits (see
                # above); each node commits at its own depth's bar.
                a_k = np.power(float(off_att), depth_of[oi])
                on_k = np.stack([onset_from_traj(traj, crit / max(float(a), 1e-30))
                                 for a in a_k])
                src = np.where(ok, on_k[np.arange(oi.size), hit], T)
                commits = src < T
                when = np.minimum(src, T - 1)
                # OWNERSHIP IS NOT THE SAME THING AS COMMITMENT, and conflating them made
                # this rule SUBTRACTIVE.  `compose_with_v4` overwrites v4's verdict on every
                # owned node, so widening `owned_off` to a deeper ring hands v4's nodes to a
                # module that may decline them: measured on `wound_comsol003`, recursive
                # removed 2 committed nodes and added none.  It happened to remove two false
                # positives, which is luck, not a property.  A deeper ring may therefore only
                # claim the nodes it actually commits -- then "strictly additive" is true of
                # the committed SET and not merely of the ownership map, which is what the
                # 5b.5 gate and `test_dispatcher_recursive_is_inert_on_a_nine_x_wound` mean.
                # Shell 1 is untouched, so `shell` and the depth-1 semantics are bit-identical.
                deep = depth_of[oi] > 1
                drop = deep & ~commits
                if drop.any():
                    owned_off[oi[drop]] = False
            elif lumen == "transport":
                commits, when = _transport()
            elif lumen == "union":
                c1, w1 = _shell()
                c2, w2 = _transport()
                commits = c1 | c2
                when = np.where(c1 & c2, np.minimum(w1, w2), np.where(c1, w1, w2))
            else:
                raise ValueError(f"unknown lumen rule {lumen!r}")
            mask[oi] = commits
            onset[oi] = np.where(commits, when, -1.0)

    owned = wnd | owned_off
    from src.clot_ml.temporal import mask_series
    return dict(mask=mask, onset=onset, mat=mat_full, owned=owned,
                series=mask_series(onset, mask, times))


def compose_with_v4(base: dict, wound_out: dict, times, data=None, bio_cfg=None) -> dict:
    """v4 everywhere it is valid; the wound module on the nodes it owns.

    v4 is left bit-identical off the wound -- the healthy wall is measurably unperturbed by
    the injury (WOUND_PROGRESS 4), so there is nothing to correct there and every reason not
    to disturb a validated artifact.
    """
    from src.clot_ml.temporal import mask_series

    owned = wound_out["owned"]
    mask = base["mask"].copy()
    onset = np.asarray(base["onset"], dtype=np.float64).copy()
    
    if data is not None and bio_cfg is not None:
        from src.clot_ml.temporal import ode_trajectory
        from src.core_physics.physics_wall_model import first_crossing
        traj_stall, _ = ode_trajectory(data, bio_cfg, flow="gt", stall=True, wound_source=True)
        onset_stall = first_crossing(traj_stall, float(bio_cfg.viscosity_mat_crit))
        wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
        stall_ign = (onset_stall >= 0) & wall
        update = stall_ign & ((onset < 0) | (onset_stall < onset))
        onset[update] = onset_stall[update]
        mask[stall_ign] = True

    mask[owned] = wound_out["mask"][owned]
    onset[owned] = wound_out["onset"][owned]
    out = dict(base)
    out["mask"], out["onset"] = mask, onset
    out["series"] = mask_series(onset, mask, times)
    out["owned"] = owned
    return out


# ---------------------------------------------------------------------------
# the neighbour trigger
# ---------------------------------------------------------------------------
def neighbour_reach(data, *, k_hops: int = TRIGGER_HOPS) -> tuple[np.ndarray, np.ndarray]:
    """``(reach [n_wound, n_wall], wall_idx)`` -- wall nodes within ``k_hops`` of each wound node.

    Mesh-graph hops, not a length: the discrimination it has to make is 12-14 hops
    (``wound_comsol003``, externally triggered) against 61 (``001``, self-triggered), so
    nothing here is delicate. Restricted to the healthy wall because the wound's own nodes
    are already covered by the self-trigger.
    """
    n = int(data.num_nodes)
    A = adjacency(data.edge_index.detach().cpu().numpy(), n)
    wnd = wound_mask(data)
    wall_idx = np.flatnonzero(data.mask_wall.reshape(-1).bool().cpu().numpy())
    idx = np.flatnonzero(wnd)
    reach = np.zeros((idx.size, wall_idx.size), dtype=bool)
    for a, j in enumerate(idx):
        seed = np.zeros(n, dtype=bool)
        seed[j] = True
        reach[a] = hop_distance(seed, A, max_h=k_hops + 1)[wall_idx] <= k_hops
    return reach, wall_idx


def committed_weight(mat: np.ndarray, crit: float, tau: float = SWITCH_TAU) -> np.ndarray:
    """``[T, M]`` soft indicator that a node has gelled -- the trigger field's currency."""
    return 1.0 / (1.0 + np.exp(-np.clip((mat / crit - 1.0) / tau, -50.0, 50.0)))


def wall_trigger_field(data, bio_cfg, V: dict, *, wall_idx: np.ndarray,
                       gate_scale: float = 1.0) -> np.ndarray:
    """Deploy-legal trigger: the shipped gated wall ODE's own committed field, ``[T, M]``.

    ``gate_scale`` exists to make one negative result reproducible rather than to be tuned.
    On ``wound_comsol003`` a scale of 20 does fix the wound (onset MAE 18.0 -> 4.7) by making
    the wall gel early enough to trigger it -- and the 12-vessel no-wound cohort says the
    same scale takes wall-onset MAE from **18.1% to 43.7%** of the horizon, with 1 the best
    value on 8 of 12 vessels. It is a ``wound_comsol003``-shaped fudge that would wreck the
    wall model, so the default is 1 and it stays 1.
    """
    from src.core_physics.ap_closure import SHIPPED, SHIPPED_DA_SCALE, make_rollout_hook
    from src.core_physics.physics_wall_model import integrate_mat_trajectory

    hook = make_rollout_hook(SHIPPED, bio_cfg, V["f0"].sr)
    traj, _ = integrate_mat_trajectory(
        data, bio_cfg, V["f0"].gate * V["wall"] * float(gate_scale),
        da_scale=SHIPPED_DA_SCALE, ap_closure=hook)
    return committed_weight(traj[:, wall_idx], V["C"].crit)


def gt_trigger_field(data, bio_cfg, V: dict, *, wall_idx: np.ndarray) -> np.ndarray:
    """ORACLE trigger: GT wall ``Mat``. Illegal to ship -- it is a ceiling, not a model.

    It answers one question: how much of the residual is the *coupling* worth if the wall's
    timing were right? On ``wound_comsol003``, onset MAE 18.0 -> 6.6 at 25 hops.
    """
    from src.core_physics.clot_phi_simple import mat_si_for_gelation_from_log1p

    T = int(data.y.shape[0])
    mi = data.y_channel_names.split(",").index("Mat_log1p_nd")
    mat = mat_si_for_gelation_from_log1p(data.y[:, :, mi], bio_cfg).reshape(T, -1).numpy()
    return committed_weight(mat[:, wall_idx], V["C"].crit)


#: Radius, in mesh-graph hops, of the region a wound's thrombus is scored over. The packs are
#: quadratic, so one corner shell is TWO hops (PHASE7_FINDINGS 8); 8 hops is four corner
#: shells. Chosen so the band contains the whole GT wound thrombus with margin rather than
#: clipping it: on 001/002 the GT clot count saturates by hop 4 (162 nodes) and 8 adds only
#: true negatives; on 003 it reaches 148 of 176. Positive rate 0.19 / 0.19 / 0.33 -- which is
#: the point, because the wound boundary alone is 100% GT clot and cannot be scored.
WOUND_REGION_HOPS = 8


def wound_region_masks(data, *, k_hops: int = WOUND_REGION_HOPS
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(region, lumen, far_lumen)`` -- the domains the wound must actually be scored on.

    WHY THIS EXISTS.  ``mask_wound`` (COMSOL ``sel1``) is **100% GT clot on every vessel**, so
    a deploy score restricted to it is 1.0 for any model that commits the patch, which the
    ungated law does for free.  It measures coverage, not skill.  Meanwhile the thrombus the
    wound actually grows extends *into the lumen* -- 82 of 162 GT clot nodes within 4 hops on
    ``wound_comsol001`` are off the boundary -- and those were being scored in the global
    off-wall domain, pooled with clot from healthy wall elsewhere in the vessel.  Neither
    domain answered "did we get the wound's thrombus".

    - ``region``     every node within ``k_hops`` of the wound: boundary and lumen together.
    - ``lumen``      the off-boundary subset -- the clot the wound pushes into the flow.
    - ``far_lumen``  off-boundary and *beyond* ``k_hops``, i.e. everything the wound did not
                     cause. Reported separately so a wound gain cannot hide healthy-wall
                     off-wall behaviour, or vice versa.
    """
    n = int(data.num_nodes)
    wnd = wound_mask(data)
    solid = solid_mask(data)
    if not wnd.any():
        empty = np.zeros(n, dtype=bool)
        return empty, empty, ~solid
    A = adjacency(data.edge_index.detach().cpu().numpy(), n)
    hops = hop_distance(wnd, A, max_h=int(k_hops) + 1)
    region = hops <= int(k_hops)
    return region, region & ~solid, (~solid) & ~region


# ---------------------------------------------------------------------------
# which flow regime a wound sits in -- deploy-legal, t=0 flow only
# ---------------------------------------------------------------------------
#: Fraction of a wound's own nodes at which the RAW t=0 shear gate must fire before the wound
#: counts as sitting in a STAGNATION zone rather than in flowing blood.  Measured, not tuned:
#: the gate fires on 0.0% of the wound on `wound_comsol001`-`005` and on 77.9% on `006`, so
#: any cut strictly inside (0, 0.78) separates them and there is nothing here to fit.
GATE_ON_STAGNANT = 0.50


def wound_gate_on_fraction(data, bio_cfg, *, flow: str = "gt") -> float:
    """Fraction of this pack's wound nodes where the RAW t=0 shear gate already fires.

    THE PREMISE OF THE WHOLE WOUND BRANCH is an ungated patch inside a gated wall: the
    injured segment deposits because `srf2` deletes the shear gates, and the surrounding
    healthy wall does not because `srf1` keeps them.  This measures whether that premise
    holds on a given vessel, from the t=0 velocity alone -- no labels, so it is legal at
    deploy time as well as at promotion time.

    **Read the raw gate from `t0_flow_fields`, never `deposition_gate(..., wound_source=True)`.**
    That helper forces the gate to 1 on wound nodes *by construction*, because forcing it
    there IS the wound law; reading it reports every wound as 100% gated, makes this test
    vacuous, and silently disables any coverage requirement built on it.  That mistake
    promoted one bad artifact on 2026-09-02 before the gate's own reporting caught it.

    Measured on the six-vessel wound cohort, `flow="fem"`:

        wound_comsol001 / 002 / 003 / 004 / 005     0.0%     flowing
        wound_comsol006                            77.9%     stagnation

    `wound_comsol006` is the only vessel whose wound sits in a dead zone (wall shear p50
    3.5 /s on the nodes that clot, 0.7 /s on the nodes that never do, against 127-146 /s
    elsewhere).  There the two-regime constants under-predict wound `Mat` by 8.4x, because
    they were fitted where species supply is never limiting -- see docs/DEPLOYCLOT.md 5b.
    """
    from src.core_physics.physics_wall_model import t0_flow_fields

    from src.clot_ml.temporal import _flow_hops

    w = wound_mask(data)
    if not w.any():
        return 0.0
    f = t0_flow_fields(data, bio_cfg, hops=_flow_hops(flow), flow_source=flow)
    return float((np.asarray(f.gate)[w] > 0).mean())


def wound_flow_regime(data, bio_cfg, *, flow: str = "gt") -> tuple[str, float]:
    """``("flowing" | "stagnation", gate_on_fraction)`` for this pack's wound."""
    frac = wound_gate_on_fraction(data, bio_cfg, flow=flow)
    return ("stagnation" if frac >= GATE_ON_STAGNANT else "flowing"), frac
