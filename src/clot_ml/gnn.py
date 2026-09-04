"""Flow-aware message passing on the mesh, with the physics backbone as a residual base.

Design follows the measured physics rather than a generic GNN recipe:

  * **Anisotropic messages.** PHASE6_RESULTS 3.4 measured that isotropic mesh smoothing of
    the source makes the fit *worse* -- the non-locality is advective, not diffusive.  So
    every edge carries the projection of the t=0 velocity onto it, and messages are gated
    by upstream/downstream sign.  An isotropic GNN is the wrong prior here.
  * **Physics as a base, not a competitor.** The backbone's own ``log(Mat/crit)`` enters as
    a feature *and* as an additive base for the regression head, so the network learns a
    residual and ``residual = 0`` recovers the physics.
  * **Two heads.** GT clot IS ``{Mat >= crit}``, so regressing ``log1p(Mat/crit)`` is the
    physically-faithful target and the classifier is the readout the score actually uses.
    Training both shares the representation and keeps the regression honest.
  * **Domain embedding.** Wall / first-shell / interior behave differently (a wall node
    accumulates its own flux, a shell node inherits ~0.16x its owner's), so the node type
    is an explicit input rather than something to infer.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#: Speeds below ``FLOW_DIR_DEADBAND * median(|u|)`` carry no flow DIRECTION.
#:
#: The direction cosines below are ``(edge . velocity) / |velocity|``, and the normaliser
#: amplifies whatever is in the numerator without limit as ``|velocity| -> 0``.  COMSOL's
#: no-slip wall velocity is **exactly** 0.0, so under ``flow="gt"`` every edge into a wall
#: node gets ``cos = 0`` and ``w_up = w_dn = 0`` -- the locked ensembles were trained with
#: wall nodes receiving no anisotropic messages at all.  RGP-DEQ cannot reproduce that: its
#: hard BC is ``u = uv_prior + sdf * uvp`` and on these packs ``sdf_nd`` at the wall is
#: clamped to 1e-6 while ``u_prior`` is ~2.9e-5, so its wall speed is ~5e-6 -- physically
#: zero (4e-6 of the lumen median) but a thousand times the old ``+1e-9`` floor, which
#: turned float noise into a UNIT direction vector.  Measured 2026-08-23: mean ``|cos_d|``
#: on wall-destination edges 0.0000 (GT) vs 0.70 (RGP-DEQ), total up/down aggregation mass
#: 0 vs ~1100, and per-node F1 recovers 0.377 -> 0.619 with the deadband in place.
#:
#: A *relative* floor is what makes this safe.  Under GT no node falls in the band at all
#: (the field is exactly 0 at the wall or O(1) in the lumen), so the GT arm is unchanged --
#: pinned by ``test_flow_direction_deadband``.  The value is not tuned: 1e-3 and 1e-2 give
#: identical scores on all 10 measured vessels, which is the signature of a noise floor
#: rather than a threshold.
FLOW_DIR_DEADBAND = 1e-3


def edge_features(pos: np.ndarray, ei: np.ndarray, u: np.ndarray, v: np.ndarray,
                  h_edge: float) -> np.ndarray:
    src, dst = ei[0], ei[1]
    d = (pos[dst] - pos[src]) / max(h_edge, 1e-12)
    ln = np.linalg.norm(d, axis=1, keepdims=True)
    dh = d / (ln + 1e-9)
    fs = np.stack([u[src], v[src]], 1)
    fd = np.stack([u[dst], v[dst]], 1)
    ns = np.linalg.norm(fs, axis=1, keepdims=True)
    nd = np.linalg.norm(fd, axis=1, keepdims=True)
    # Below the deadband a "direction" is numerical noise, not flow -- see FLOW_DIR_DEADBAND.
    floor = FLOW_DIR_DEADBAND * float(np.median(np.hypot(u, v)))
    cos_s = (dh * fs).sum(1, keepdims=True) / (ns + 1e-9) * (ns > floor)
    cos_d = (dh * fd).sum(1, keepdims=True) / (nd + 1e-9) * (nd > floor)
    spd_s = np.log1p(ns + 1e-9)
    return np.concatenate([dh, ln, cos_s, cos_d, cos_s * spd_s, spd_s], axis=1).astype(np.float32)


class MPLayer(nn.Module):
    """One anisotropic message-passing step: separate upstream / downstream aggregation."""

    def __init__(self, dim: int, edim: int, drop: float = 0.1):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(2 * dim + edim, dim), nn.SiLU(),
                                 nn.Linear(dim, dim))
        self.upd = nn.Sequential(nn.Linear(4 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, ei, ea, w_up, w_dn):
        src, dst = ei[0], ei[1]
        m = self.msg(torch.cat([x[src], x[dst], ea], dim=1))
        n = x.shape[0]
        agg_u = torch.zeros_like(x).index_add_(0, dst, m * w_up)
        agg_d = torch.zeros_like(x).index_add_(0, dst, m * w_dn)
        cu = torch.zeros(n, 1, device=x.device).index_add_(0, dst, w_up).clamp_min(1e-6)
        cd = torch.zeros(n, 1, device=x.device).index_add_(0, dst, w_dn).clamp_min(1e-6)
        mx = torch.zeros_like(x).index_reduce_(0, dst, m, "amax", include_self=False)
        h = self.upd(torch.cat([x, agg_u / cu, agg_d / cd, mx], dim=1))
        return self.norm(x + self.drop(h))


class ClotGNN(nn.Module):
    def __init__(self, in_dim: int, edim: int, dim: int = 96, layers: int = 6,
                 drop: float = 0.1, extra_dim: int = 0):
        super().__init__()
        self.extra_dim = int(extra_dim)
        self.enc = nn.Sequential(nn.Linear(in_dim + self.extra_dim, dim), nn.SiLU(),
                                 nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.mp = nn.ModuleList([MPLayer(dim, edim, drop) for _ in range(layers)])
        self.head_cls = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.head_reg = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        # residual base: start at the physics prediction exactly
        nn.init.zeros_(self.head_reg[-1].weight)
        nn.init.zeros_(self.head_reg[-1].bias)

    def forward(self, x, ei, ea, w_up, w_dn, mat_phys, extra=None):
        h = self.enc(x if extra is None else torch.cat([x, extra], dim=1))
        for layer in self.mp:
            h = layer(h, ei, ea, w_up, w_dn)
        logit = self.head_cls(h).reshape(-1)
        reg = mat_phys + self.head_reg(h).reshape(-1)
        return logit, reg


def to_device(S: dict, mu: np.ndarray, sd: np.ndarray, dev: torch.device) -> dict:
    ei = S["edge_index"]
    pos, u, v = S["pos"], S["u"], S["v"]
    h_edge = float(np.median(np.linalg.norm(pos[ei[0]] - pos[ei[1]], axis=1)))
    ea = edge_features(pos, ei, u, v, h_edge)
    cos_s = ea[:, 4:5]
    w_up = np.clip(cos_s, 0.0, None)
    w_dn = np.clip(-cos_s, 0.0, None)
    t = lambda a, d=torch.float32: torch.tensor(np.ascontiguousarray(a), dtype=d, device=dev)
    return dict(
        x=t((S["X"] - mu) / sd), ei=t(ei, torch.long), ea=t(ea),
        w_up=t(w_up), w_dn=t(w_dn),
        mat_phys=t(np.log1p(np.maximum(S["mat_phys"], 0.0) / 2e7)),
        y=t(S["y"]), mat_gt=t(S["mat_gt"]),
        wall=t(S["wall"].astype(np.float32)),
        # `solid` = wall | wound.  Carried so the training domains can match the EVAL
        # domains exactly (`src/clot_ml/data.eval_domains`): off-wall is `~solid`, true
        # lumen, not `~wall`.  Identical on every no-wound pack, and older caches that
        # predate the geometry union fall back to `wall`.
        solid=t(np.asarray(S.get("solid", S["wall"])).astype(np.float32)),
        n=int(len(S["wall"])))


# --- graph assembly and refinement rollout ---------------------------------------
# Moved out of `scripts/train_clot_gnn.py`: `clot_ml.locked` builds and rolls the
# locked ensemble with these, so the library depended on a training script to score
# a promoted model.  Both are unchanged.

import numpy as np  # noqa: E402

from src.clot_ml.recurrent import (  # noqa: E402
    advective_operators, feedback_channels, feedback_channels_advective, neighbour_operator,
)
from src.clot_ml.softmetric import (  # noqa: E402
    dilation_operator, soft_dilate, to_torch_sparse,
)


def to_weighted_sparse(M, dev_t):
    """Like ``to_torch_sparse`` but KEEPS the values -- upwind weights are not indicators."""
    C = M.tocoo()
    idx = torch.tensor(np.stack([C.row, C.col]), dtype=torch.long, device=dev_t)
    val = torch.tensor(C.data, dtype=torch.float32, device=dev_t)
    return torch.sparse_coo_tensor(idx, val, M.shape).coalesce()


def build_graph(S, mu, sd, dev_t, *, need_soft=False, need_fb=False, adv_fb=False):
    g = to_device(S, mu, sd, dev_t)
    g["phys"] = torch.tensor(S["phys_mask"].astype(np.float32), device=dev_t)
    if need_fb:
        g["At"] = to_torch_sparse(neighbour_operator(S["edge_index"], len(S["wall"])), dev_t)
        g["owner"] = torch.tensor(S["owner"].astype(np.int64), device=dev_t)
        if adv_fb:
            Wu, Wd = advective_operators(S["pos"], S["edge_index"], S["u"], S["v"])
            g["Wup"] = to_weighted_sparse(Wu, dev_t)
            g["Wdn"] = to_weighted_sparse(Wd, dev_t)
    if need_soft:
        D = dilation_operator(S["edge_index"], len(S["wall"]), 2)
        g["D"] = to_torch_sparse(D, dev_t)
        g["gt_dil"] = soft_dilate(g["y"], g["D"]).detach()
    # TRUE LUMEN, matching `src/clot_ml/data.off_domain`: `1 - solid`, not `1 - wall`.
    g["off"] = 1.0 - g["solid"]
    return g


def rollout(model, g, rounds, adv_fb=False):
    """K shared-weight refinement rounds; round 0 occlusion is the physics mask."""
    if rounds <= 1:
        return model(g["x"], g["ei"], g["ea"], g["w_up"], g["w_dn"], g["mat_phys"])
    p = g["phys"].clone()
    R = int(rounds)
    for k in range(R):
        extra = (feedback_channels_advective(p, g["At"], g["Wup"], g["Wdn"],
                                            g["owner"])
                 if adv_fb else feedback_channels(p, g["At"], g["owner"]))
        logit, reg = model(g["x"], g["ei"], g["ea"], g["w_up"], g["w_dn"], g["mat_phys"],
                           extra=extra)
        # truncated BPTT: only the last round carries gradient, so peak memory is one
        # round's activations rather than R (a 4 GB card cannot hold R=3 at dim 96).
        if k < R - 1:
            p = torch.sigmoid(logit).detach()
        else:
            p = torch.sigmoid(logit)
    return logit, reg
