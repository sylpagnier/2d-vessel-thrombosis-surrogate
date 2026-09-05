import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, to_dense_batch
from typing import Optional, Tuple, Union
from torch import Tensor

from src.core_physics.anderson import anderson_acceleration
from src.architecture.spectral_linear import SpectralLinear
from src.architecture.siren_decoder import SIRENDecoder
from src.config import NodeFeat, PhysicsConfig, PredChannels, WIDTH_D1_MAX, WIDTH_D2_MAX
from src.utils.batching import get_batch_tensor

# Set by the Stage-A arm scripts (`scripts/stage_a/run_*.sh`), the only setter and one
# outside the tree the knob sweep grepped -- so sweeping these to plain constants made
# every E-series arm a silent no-op for them.  Read from the environment with the swept
# value as the default: unset behaves exactly as the constant did.
KINEMATICS_BC_ENVELOPE = os.environ.get("KINEMATICS_BC_ENVELOPE", "0")
KINEMATICS_BC_ENVELOPE_DECAY = os.environ.get("KINEMATICS_BC_ENVELOPE_DECAY", "0.0")
KINEMATICS_BC_ENVELOPE_FLOOR = os.environ.get("KINEMATICS_BC_ENVELOPE_FLOOR", "0.0")
KINEMATICS_DECODER_SKIP = os.environ.get("KINEMATICS_DECODER_SKIP", "0")
KINEMATICS_FOURIER_LEARNABLE = "0"
KINEMATICS_PHYS_GAT_PRIORS_MULTIPLY_BEFORE_ADDITIVE = "0"
KINEMATICS_RESIDUAL_GAIN = os.environ.get("KINEMATICS_RESIDUAL_GAIN", "0")
KINEMATICS_RESIDUAL_GAIN_CLAMP = "3.0"
KINEMATICS_RESIDUAL_REZERO = os.environ.get("KINEMATICS_RESIDUAL_REZERO", "0")
KINEMATICS_WSS_FUSE = "0"

def _spectral_or_plain_linear(in_features: int, out_features: int, bias: bool, spectral: bool) -> nn.Module:
    if spectral:
        return SpectralLinear(in_features, out_features, bias=bias)
    return nn.Linear(in_features, out_features, bias=bias)


def _make_activation(name: str) -> nn.Module:
    mode = (name or "silu").strip().lower()
    if mode == "silu":
        return nn.SiLU()
    if mode == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation '{name}'. Supported: silu, gelu.")


class AttentionGlobalMixingBlock(nn.Module):
    """
    Perceiver-style bottleneck: global tokens read each graph via cross-attention,
    reason with an MLP, then broadcast back to nodes.

    Uses :func:`torch_geometric.utils.to_dense_batch` so attention is **strictly within**
    each graph — PyG's batched ``x`` is not treated as one long sequence across vessels.
    """

    def __init__(
        self,
        latent_dim: int,
        num_global_tokens: int = 16,
        num_heads: int = 4,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError(f"latent_dim ({latent_dim}) must be divisible by num_heads ({num_heads})")
        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.randn(1, num_global_tokens, latent_dim))
        self.cross_att_read = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=num_heads, batch_first=True
        )
        self.global_mlp = nn.Sequential(
            _spectral_or_plain_linear(latent_dim, latent_dim, True, use_spectral_norm),
            nn.SiLU(),
            _spectral_or_plain_linear(latent_dim, latent_dim, True, use_spectral_norm),
        )
        self.cross_att_broadcast = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=num_heads, batch_first=True
        )
        # Broadcast attention starts ~inactive so local GNN / SIREN can stabilize first.
        with torch.no_grad():
            nn.init.zeros_(self.cross_att_broadcast.out_proj.weight)
            nn.init.zeros_(self.cross_att_broadcast.out_proj.bias)

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        dense_x, mask = to_dense_batch(x, batch)
        batch_size = dense_x.size(0)
        device, dtype = x.device, x.dtype
        global_t = self.global_tokens.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
        # MHA: True in key_padding_mask = positions to ignore (padding).
        # mask is True for real nodes, so invert for padding slots.
        read_tokens, _ = self.cross_att_read(
            query=global_t,
            key=dense_x,
            value=dense_x,
            key_padding_mask=~mask,
        )
        processed_tokens = self.global_mlp(read_tokens)
        broadcast_update, _ = self.cross_att_broadcast(
            query=dense_x,
            key=processed_tokens,
            value=processed_tokens,
        )
        return broadcast_update[mask]


class MultiHeadPhysicsGATConv(MessagePassing):
    """Physics-modulated multi-head GAT (PM-GAT).

    Edge attention logits receive additive (or multiplicative) biases from
    advection, wall-rheology, and curvature priors before softmax.
    Core of RGP-DEQ; see ``docs/MODEL_NOMENCLATURE.md``.
    """

    def __init__(
        self,
        latent_dim: int,
        edge_dim: int = 3,
        temperature: float = 1.5,
        use_spectral_norm: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('node_dim', 0)
        super().__init__(**kwargs)

        self.temperature = temperature
        self.edge_proj = _spectral_or_plain_linear(edge_dim, latent_dim, True, use_spectral_norm)
        # Candidate toggle: multiply edge projection into logits before additive log-modulators.
        # Env: KINEMATICS_PHYS_GAT_PRIORS_MULTIPLY_BEFORE_ADDITIVE=1
        self.priors_multiply_before_add = bool(
            int(KINEMATICS_PHYS_GAT_PRIORS_MULTIPLY_BEFORE_ADDITIVE)
        )

        self.lin_src = _spectral_or_plain_linear(latent_dim, latent_dim, True, use_spectral_norm)
        self.lin_dst = _spectral_or_plain_linear(latent_dim, latent_dim, True, use_spectral_norm)
        self.att = _spectral_or_plain_linear(latent_dim, 1, True, use_spectral_norm)

    def forward(self,
                x: Union[Tensor, Tuple[Tensor, Tensor]],
                edge_index: Tensor,
                edge_attr: Tensor,
                mod_adv: Tensor,
                mod_rheo: Tensor,
                mod_curve: Tensor,
                size: Optional[Tuple[int, int]] = None) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        x_src = self.lin_src(x[0])
        x_dst = self.lin_dst(x[1])

        alpha_src = self.att(x_src)
        alpha_dst = self.att(x_dst)

        out = self.propagate(
            edge_index,
            size=size,
            x=(x_src, x_dst),
            alpha=(alpha_src, alpha_dst),
            edge_attr=edge_attr,
            mod_adv=mod_adv,
            mod_rheo=mod_rheo,
            mod_curve=mod_curve
        )
        return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: Tensor,
                edge_attr: Tensor, mod_adv: Tensor, mod_rheo: Tensor, mod_curve: Tensor,
                index: Tensor, ptr: Optional[Tensor], size_i: Optional[int]) -> Tensor:
        alpha = (alpha_j + alpha_i) / self.temperature
        # Bias pre-softmax logits with flow-wall directional modulators and curvature.
        if self.priors_multiply_before_add:
            alpha = alpha * self.edge_proj(edge_attr)
            alpha = alpha + mod_adv + mod_rheo + mod_curve
        else:
            # Historical order (kept as default): add additive log-modulators, then scale.
            alpha = alpha + mod_adv + mod_rheo + mod_curve
            alpha = alpha * self.edge_proj(edge_attr)
        alpha = softmax(alpha, index, ptr, size_i)
        return x_j * alpha


class RGPBlock(nn.Module):
    """One RGP-DEQ equilibrium step: PM-GAT + Perceiver global mixing + residual.

    Legacy alias: ``GINOBlock`` (not Li et al. GINO).
    """

    def __init__(
        self,
        latent_dim=64,
        edge_dim=3,
        use_spectral_norm: bool = True,
        activation_fn: str = "silu",
        num_global_tokens: int = 16,
    ):
        super().__init__()
        assert latent_dim % 2 == 0, "latent_dim must be divisible by 2 for multi-head split"

        self.conv = MultiHeadPhysicsGATConv(
            latent_dim, edge_dim=edge_dim, use_spectral_norm=use_spectral_norm
        )
        self.global_mixer = AttentionGlobalMixingBlock(
            latent_dim,
            num_global_tokens=num_global_tokens,
            use_spectral_norm=use_spectral_norm,
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.activation = _make_activation(activation_fn)

    def forward(self, z, edge_index, edge_attr, batch, mod_adv, mod_rheo, mod_curve):
        local_out = self.conv(z, edge_index, edge_attr, mod_adv, mod_rheo, mod_curve)
        global_out = self.global_mixer(z, batch)
        return self.norm(self.activation(z + local_out + global_out))


# Backward-compatible alias (not Li et al. GINO)
GINOBlock = RGPBlock



#: How absolute node coordinates enter the encoder.  ``"absolute"`` reproduces every historical
#: run bit-for-bit and is the default; ``"centered"`` subtracts the graph's own centroid.
KINEMATICS_COORD_MODE_ENV = "KINEMATICS_COORD_MODE"


def _canonical_coords(nodes_nd: Tensor) -> Tensor:
    """Optionally remove the absolute frame from the coordinates the Fourier block sees.

    RGP_DEQ_REPAIR_PLAN.md §8 A2.  ``_apply_fourier_encoding`` feeds ABSOLUTE ``x, y`` through
    16 sin/cos frequencies -- the NeRF construction, whose entire purpose is to let a network
    memorise what happens at a location.  Every synthetic training vessel lives in the same box
    (``x`` in [0, 5.54], ``y`` in [-0.63, 0.56], inlet at ``x = 0``), so absolute ``x`` is a
    near-perfect proxy for "fraction of the way along the vessel" and the shortcut is free.

    Measured: translating a vessel -- which changes nothing physical -- moves the prediction by

    ```
    shift of vessel length      1%      10%     100%
    comsol020  rel change    0.059    0.391    0.552
    comsol041  rel change    0.206    0.264    0.286
    ```

    against a model whose total error versus COMSOL is ~0.20.  A 10% translation moves the
    answer by more than the entire error budget, which is spatial memorisation, not physics.

    Centring on the graph's own centroid makes the encoder exactly translation-invariant.  It
    does NOT give rotation invariance (``wall_normal`` is covariant, not invariant) and it does
    not remove streamwise position as a feature -- position along the vessel is genuine physics
    for a developing flow.  What it removes is the shared absolute frame that makes that
    position memorisable across a corpus of similarly-placed vessels.

    Default is ``"absolute"``: this changes what the weights mean, so it is a retrain-time
    choice, not something to flip under a trained checkpoint.
    """
    mode = os.environ.get(KINEMATICS_COORD_MODE_ENV, "absolute").strip().lower()
    if mode in ("", "absolute", "xy"):
        return nodes_nd
    if mode == "centered":
        return nodes_nd - nodes_nd.mean(dim=0, keepdim=True)
    raise ValueError(
        f"{KINEMATICS_COORD_MODE_ENV} must be 'absolute' or 'centered', got {mode!r}"
    )


class RGP_DEQ(nn.Module):
    """Stage-A flow surrogate: RGP-DEQ (mu-coupled PM-GAT-Perceiver DEQ).

    Equilibrium: z* = f(z*, mu(z*)) via Anderson/Picard; each step uses
    ``RGPBlock`` (physics-modulated GAT + Perceiver global tokens).

    Canonical id: ``rgp_deq_kine`` (acronym RGP-DEQ). Legacy class alias: ``GINO_DEQ``.
    See ``docs/MODEL_NOMENCLATURE.md``.
    """

    def __init__(
        self,
        in_channels=11,
        out_channels=5,
        latent_dim=64,
        max_iters=25,
        num_fourier_freqs=8,
        outer_iters=3,
        mu_inf_nd: Optional[float] = None,
        mu_0_nd: Optional[float] = None,
        phys_cfg: Optional[PhysicsConfig] = None,
        activation_fn: str = "silu",
        fourier_base: float = 2.0,
        use_hard_bcs: bool = False,
        num_global_tokens: int = 16,
        use_siren_decoder: bool = False,
        use_width_priors: bool = False,
        wss_fuse: Optional[bool] = None,
        bc_envelope: Optional[bool] = None,
        bc_lambda: Optional[float] = None,
        bc_envelope_decay: Optional[float] = None,
        bc_envelope_floor: Optional[float] = None,
        fourier_learnable: Optional[bool] = None,
        shear_head: bool = True,
        decoder_skip: Optional[bool] = None,
        residual_gain: Optional[bool] = None,
        residual_rezero: Optional[bool] = None,
    ):
        super().__init__()
        self.shear_head = shear_head
        self.max_iters = max_iters
        self.outer_iters = outer_iters
        self.num_fourier_freqs = num_fourier_freqs
        if phys_cfg is not None:
            mu_scale = float(phys_cfg.mu_viscosity_nd_scale)
            default_mu_inf_nd = float(phys_cfg.mu_inf / mu_scale)
            default_mu_0_nd = float(phys_cfg.mu_0 / mu_scale)
            self.edge_decay_k = float(phys_cfg.gino_edge_decay_k)
            self.curve_log_clamp_min = float(phys_cfg.gino_curve_log_clamp_min)
            self.rheo_log_clamp_min = float(phys_cfg.gino_rheo_log_clamp_min)
            self.adv_log_clamp_min = float(phys_cfg.gino_adv_log_clamp_min)
        else:
            default_mu_inf_nd = 0.03
            default_mu_0_nd = 1.0
            self.edge_decay_k = 5.0
            self.curve_log_clamp_min = 1e-4
            self.rheo_log_clamp_min = 1e-3
            self.adv_log_clamp_min = 1e-3
        self.mu_inf_nd = float(default_mu_inf_nd if mu_inf_nd is None else mu_inf_nd)
        self.mu_0_nd = float(default_mu_0_nd if mu_0_nd is None else mu_0_nd)
        self.activation_fn = (activation_fn or "silu").strip().lower()
        self.fourier_base = float(fourier_base)

        self.use_hard_bcs = bool(use_hard_bcs)

        # Toggles: explicit ctor kwargs (checkpoint restore) override env for A/B sweeps.
        self.bc_envelope = (
            bool(bc_envelope)
            if bc_envelope is not None
            else bool(int(KINEMATICS_BC_ENVELOPE))
        )
        # Snapshotted alongside `bc_envelope`, not read from the environment at load time.
        # It sets the SHAPE of the hard BC -- `u = prior + (1 - exp(-bc_lambda*sdf)) * r` -- so
        # a model trained at one value and rebuilt at another is a different function of the
        # same weights, silently.  As an env-only knob it was never written into the checkpoint
        # config, so every reload snapped back to 10.0 however the run was launched.
        self.bc_lambda = float(
            bc_lambda
            if bc_lambda is not None
            else os.environ.get("KINEMATICS_BC_LAMBDA", "10.0")
        )
        # Far-field decay on the hard-BC envelope, so the residual is BAND-LOCALISED:
        #
        #     env(sdf) = (1 - exp(-bc_lambda*sdf)) * exp(-bc_envelope_decay*sdf)
        #
        # zero at the wall (the BC is untouched), peaking in the near-wall band, decaying in the
        # core.  `0.0` is exactly the plain envelope, so this defaults to a no-op.
        #
        # The plain envelope is backwards for a residual on an already-accurate prior.  Measured
        # on the deploy packs against the FEM prior: the outer 40% of the domain by wall distance
        # carries 17% of the prior's squared error and is handed envelope 1.000, while the
        # near-wall decile that the wall-shear metrics read is damped to 0.45-0.59.  The head's
        # output is correlated with the prior's error in the wall band (+0.25) and uncorrelated
        # globally (-0.03), so the plain envelope gives it the most authority exactly where it
        # has the least signal -- which is how a head that improves the gate still loses rel-L2.
        self.bc_envelope_decay = float(
            bc_envelope_decay
            if bc_envelope_decay is not None
            else KINEMATICS_BC_ENVELOPE_DECAY
        )
        # Lower bound on the far-field decay, so BAND-LOCALISED does not mean CONFINED:
        #
        #     env(sdf) = (1 - exp(-bc_lambda*sdf)) * (floor + (1 - floor)*exp(-decay*sdf))
        #
        # `0.0` is exactly the decayed envelope above, so this defaults to a no-op, and the
        # value at the wall is still exactly zero for any floor -- the hard BC is untouched.
        #
        # Why it exists.  At `decay=12` the envelope in the core is `exp(-12*sdf)`, which is
        # 2.5e-3 at mid-lumen: the head has no authority there at all, by construction.  That
        # is right for a prior whose error is a near-wall film, and it is exactly wrong for the
        # two vessels the deploy score is actually lost on -- `comsol045` and `comsol046`, the
        # highest-peak-velocity vessels in the corpus, whose FEM prior misplaces a downstream
        # shear layer and reads rel-L2 0.53 / 0.67 with the error entirely OFF the wall
        # (DEPLOYCLOT.md s1).  Measured: E5's prediction on those two is bit-identical to the
        # prior it was handed (0.5335 / 0.6705, gate Jaccard 0.487 / 0.234 -> 0.487 / 0.234).
        # Those two carry 37% of the whole FEM-vs-GT wall deploy gap, so a head that cannot
        # reach them cannot close it.  A floor buys core authority back at a fixed, auditable
        # fraction of the wall band's, instead of trading the band away.
        self.bc_envelope_floor = float(
            bc_envelope_floor
            if bc_envelope_floor is not None
            else KINEMATICS_BC_ENVELOPE_FLOOR
        )
        if not 0.0 <= self.bc_envelope_floor <= 1.0:
            raise ValueError(
                "bc_envelope_floor must lie in [0, 1] (0 = the decayed envelope, 1 = the plain "
                f"one); got {self.bc_envelope_floor}")
        self.wss_fuse = (
            bool(wss_fuse)
            if wss_fuse is not None
            else bool(int(KINEMATICS_WSS_FUSE))
        )
        self.fourier_learnable = (
            bool(fourier_learnable)
            if fourier_learnable is not None
            else bool(int(KINEMATICS_FOURIER_LEARNABLE))
        )

        # WSS decoder: either z-only (legacy) or fused with (u,v,p) and mu.
        if self.wss_fuse:
            # Input: z + uvp + mu
            self.wss_decoder = nn.Sequential(
                SpectralLinear(latent_dim + 4, latent_dim),
                _make_activation(self.activation_fn),
                nn.Linear(latent_dim, 1),
            )
            if self.shear_head:
                self.shear_decoder = nn.Sequential(
                    SpectralLinear(latent_dim + 4, latent_dim),
                    _make_activation(self.activation_fn),
                    nn.Linear(latent_dim, 1),
                )
        else:
            self.wss_decoder = nn.Sequential(
                SpectralLinear(latent_dim, latent_dim),
                _make_activation(self.activation_fn),
                nn.Linear(latent_dim, 1),  # Non-recurrent output projection
            )
            if self.shear_head:
                self.shear_decoder = nn.Sequential(
                    SpectralLinear(latent_dim, latent_dim),
                    _make_activation(self.activation_fn),
                    nn.Linear(latent_dim, 1),
                )
        self.use_siren_decoder = bool(use_siren_decoder)
        self.use_width_priors = bool(use_width_priors)
        self.decouple_rheology = False

        # --- Dynamic-range repair (RGP_DEQ_REPAIR_PLAN.md s17) -------------------------------
        # `RGPBlock.forward` ends in `nn.LayerNorm`, so the equilibrium `z*` the decoder reads is
        # LITERALLY a LayerNorm output: every node's latent sits on one fixed shell.  Measured on
        # four deploy packs under the promoted checkpoint, the coefficient of variation of
        # ||z_i|| across nodes is 8.8e-04 - 1.1e-03, i.e. constant to a tenth of a percent.
        # Per-node AMPLITUDE is therefore not something the DEQ output can carry; only direction
        # on that shell is.  The consequence is measurable at the hard BC, comparing the residual
        # the model emits, `uvp = (pred - uv_prior)/sdf`, against the one the labels require:
        #
        #   vessel        |uvp| p99/p50 (model)   (labels need)
        #   comsol003            9.07                24.33
        #   comsol008            8.69                20.58
        #   comsol015            9.80                30.35
        #   comsol021            7.02                26.99
        #
        # Medians agree (0.37-0.62 against 0.30-0.65); the model is short by ~3x on the TAIL, and
        # wall `dsrx` spread is a tail statistic.  This is why s16.10's 48x reweighting of
        # `l_band_dsrx` moved `dsrxScale` "not one part in a thousand" while everything else
        # degraded, and why a single-vessel overfit reaches 0.955 (one vessel needs one scale).
        # The objective cannot move what the architecture normalises away.
        #
        # Both repairs below leave the fixed-point map -- and so every convergence property of
        # the Anderson solve -- untouched: they change only what the DECODER is allowed to read.
        # Both default OFF and are no-ops at initialisation, so they are retrain-time choices.
        self.decoder_skip = (
            bool(decoder_skip)
            if decoder_skip is not None
            else bool(int(KINEMATICS_DECODER_SKIP))
        )
        self.residual_gain = (
            bool(residual_gain)
            if residual_gain is not None
            else bool(int(KINEMATICS_RESIDUAL_GAIN))
        )
        # exp(+-3) -> gain in [0.05, 20]: ~6x headroom over the 3x deficit measured above, and
        # bounded, so a diverging gain cannot take the field with it.
        self.residual_gain_clamp = float(KINEMATICS_RESIDUAL_GAIN_CLAMP)

        freqs = (self.fourier_base ** torch.arange(num_fourier_freqs)) * torch.pi
        if self.fourier_learnable:
            self.fourier_freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("fourier_freqs", freqs)

        fourier_channels = 5 * num_fourier_freqs * 2
        width_extra = 3 if self.use_width_priors else 0
        encoded_channels = (in_channels - 5) + 5 + fourier_channels + width_extra

        self.encoder = nn.Sequential(
            nn.Linear(encoded_channels, latent_dim),
            _make_activation(self.activation_fn),
            nn.Linear(latent_dim, latent_dim)
        )

        self.core = RGPBlock(
            latent_dim,
            edge_dim=3,
            activation_fn=self.activation_fn,
            num_global_tokens=num_global_tokens,
        )
        # With the skip on, the decoder reads [z*, x_enc]: the shell PLUS the encoder's own
        # un-normalised per-node features (sdf, wall_normal, the width priors, the velocity
        # prior).  That is the scale-carrying half `z*` cannot supply.
        decoder_in = latent_dim * 2 if self.decoder_skip else latent_dim
        if self.use_siren_decoder:
            self.siren_decoder = SIRENDecoder(decoder_in)
            self.kinematics_decoder = None
        else:
            self.kinematics_decoder = nn.Linear(decoder_in, 3)
            self.siren_decoder = None

        if self.residual_gain:
            # Per-node log-gain on the hard-BC residual.  Reads `x_enc` and NOT `z*`, on purpose:
            # reading the shell would reintroduce the constraint this exists to lift.  The final
            # layer is zero-initialised, so gain == exp(0) == 1 and a fresh model is bit-identical
            # to `residual_gain=False` -- the flag adds capacity without moving the starting point.
            self.residual_gain_head = nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                _make_activation(self.activation_fn),
                nn.Linear(latent_dim // 2, 1),
            )
            nn.init.zeros_(self.residual_gain_head[-1].weight)
            nn.init.zeros_(self.residual_gain_head[-1].bias)
        else:
            self.residual_gain_head = None

        # ReZero on the hard-BC residual: one learnable scalar, initialised to 0, so a fresh
        # model predicts `u = uv_prior` EXACTLY and every departure from the prior has to be
        # earned by the objective.
        #
        # Without it the decoder is a randomly-initialised map on a LayerNorm shell, so it emits
        # an O(1) field regardless of what the prior needs.  That was harmless while the prior
        # was analytic Poiseuille -- its error is O(0.4) and an O(0.2) initial residual is the
        # right order.  Under the FEM prior the target error is O(0.01), and the SAME
        # initialisation overshoots it by ~20x (measured 19-24x on the deploy packs): the run
        # then spends itself suppressing its own initialisation noise instead of learning where
        # the FEM was wrong, gets to 1.4-3.4x, and stops.  What survives is uncorrelated --
        # corr(delta, prior_error) = +0.03, and the best possible rescaling of it removes 0.1%
        # of the prior's error (`src.tools.diagnostics.residual_head_audit`).
        #
        # The same trick is already used twice in this file, for the same reason: the residual
        # gain head and `AttentionGlobalMixingBlock.cross_att_broadcast` are both zero-initialised
        # so they start inactive.  Defaults OFF, so existing checkpoints are untouched.
        self.residual_rezero = (
            bool(residual_rezero)
            if residual_rezero is not None
            else bool(int(KINEMATICS_RESIDUAL_REZERO))
        )
        if self.residual_rezero:
            self.residual_scale = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("residual_scale", None)

        self.mu_decoder = nn.Sequential(
            SpectralLinear(latent_dim, latent_dim),
            _make_activation(self.activation_fn),
            nn.Linear(latent_dim, 1)
        )
        self.mu_encoder = nn.Linear(1, latent_dim)
        # Prior injector: maps [u_prior, v_prior, p_prior, mu_prior] into latent warm start.
        self.z_prior_proj = SpectralLinear(4, latent_dim)

    def _apply_fourier_encoding(self, x, pos_nd=None):
        # Canonical Phase-1 layout is 15 channels; optional width priors append three more (see NodeFeat).
        xb = x[:, :15] if x.size(1) >= 15 else x
        nodes_nd = pos_nd if pos_nd is not None else xb[:, NodeFeat.XY]
        sdf_nd = xb[:, NodeFeat.SDF]
        shear_pot = xb[:, NodeFeat.SHEAR_POT]
        wall_normal = xb[:, NodeFeat.WALL_NORMAL]

        rest = xb[:, NodeFeat.REST]
        uv_prior = xb[:, NodeFeat.UV_PRIOR]
        mu_prior = xb[:, NodeFeat.MU_PRIOR]
        wss_prior = xb[:, NodeFeat.WSS_PRIOR]

        nodes_nd = _canonical_coords(nodes_nd)
        features_to_encode = torch.cat([nodes_nd, sdf_nd, wall_normal], dim=1)
        N, C = features_to_encode.shape

        x_proj = (features_to_encode.unsqueeze(-1) * self.fourier_freqs).contiguous()
        x_proj = x_proj.view(N, -1)
        fourier_feats = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        encoded_x = torch.cat(
            [shear_pot, features_to_encode, fourier_feats, rest, uv_prior, mu_prior, wss_prior], dim=1)
        if getattr(self, "use_width_priors", False):
            if x.size(1) >= NodeFeat.WIDTH_D2.stop:
                width_features = x[:, NodeFeat.WIDTH_ND.start : NodeFeat.WIDTH_D2.stop].clone()
                # Bounds live in `src.config` so the encoder and
                # `kinematics_inference.clamped_width_priors` cannot drift apart.  They were
                # hardcoded in both places against a 40-vessel corpus with no severe
                # stenosis; on the 250-vessel cohort that clamped 44% / 34% of vessels.
                width_features[:, 1] = torch.clamp(width_features[:, 1], -WIDTH_D1_MAX, WIDTH_D1_MAX)
                width_features[:, 2] = torch.clamp(width_features[:, 2], -WIDTH_D2_MAX, WIDTH_D2_MAX)
            else:
                width_features = torch.zeros(x.size(0), 3, device=x.device, dtype=x.dtype)
            encoded_x = torch.cat([encoded_x, width_features], dim=1)
        return encoded_x, uv_prior

    def _solve_equilibrium_z(
        self,
        data,
        *,
        solver: str = "anderson",
        anderson_beta: float = 0.8,
        anderson_warmup_iters: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Anderson/Picard DEQ solve.

        Returns ``(z_eq, jac_loss, x_enc)``.  ``x_enc`` is handed back rather than recomputed
        because the decoder skip and the residual gain both need the encoder's un-normalised
        per-node features, and re-running ``_apply_fourier_encoding`` would double the encode.
        """
        x_encoded, _ = self._apply_fourier_encoding(data.x)
        x_enc = self.encoder(x_encoded)
        z = x_enc.clone()

        row, col = data.edge_index
        edge_attr = data.edge_attr
        edge_vec = edge_attr[:, :2]
        batch_idx = get_batch_tensor(data, data.x.size(0), data.x.device)

        wall_normals = data.x[:, NodeFeat.WALL_NORMAL]
        e_dir = F.normalize(edge_vec, p=2, dim=-1, eps=1e-8)
        n_dir_row = F.normalize(wall_normals[row], p=2, dim=-1, eps=1e-8)
        n_dir_col = F.normalize(wall_normals[col], p=2, dim=-1, eps=1e-8)

        dot_prod = torch.abs((e_dir * n_dir_row).sum(dim=-1, keepdim=True))
        dot_prod = torch.clamp(dot_prod, max=1.0)

        sdf_nd = data.x[:, NodeFeat.SDF]
        sdf_edge = sdf_nd[row]

        decay_factor = torch.exp(-self.edge_decay_k * sdf_edge)
        curve_dot = (n_dir_row * n_dir_col).sum(dim=-1, keepdim=True)
        mod_curve = torch.log(torch.clamp(1.0 - curve_dot, min=self.curve_log_clamp_min, max=1.0)) * decay_factor

        mod_rheo = torch.log(torch.clamp(dot_prod, min=self.rheo_log_clamp_min, max=1.0)) * decay_factor
        mod_adv = torch.log(torch.clamp((1.0 - dot_prod), min=self.adv_log_clamp_min, max=1.0)) * decay_factor

        def decode_mu(latent_state):
            mu_raw_state = self.mu_decoder(latent_state)
            return self.mu_inf_nd + (self.mu_0_nd - self.mu_inf_nd) * torch.sigmoid(mu_raw_state)

        def f_coupled(curr_z):
            curr_z_flat = curr_z.squeeze(0) if curr_z.ndim == 3 else curr_z
            mu = decode_mu(curr_z_flat)
            if getattr(self, "decouple_rheology", False):
                if hasattr(self, "kinematics_mu_decoder"):
                    with torch.no_grad():
                        t1_mu_raw = self.kinematics_mu_decoder(curr_z_flat)
                        mu_feedback = self.mu_inf_nd + (self.mu_0_nd - self.mu_inf_nd) * torch.sigmoid(t1_mu_raw)
                else:
                    mu_feedback = mu
            else:
                mu_feedback = mu
            mu_enc = self.mu_encoder(mu_feedback)
            z_in = curr_z_flat + x_enc + mu_enc
            out = self.core(z_in, data.edge_index, edge_attr, batch_idx, mod_adv, mod_rheo, mod_curve)
            return out.unsqueeze(0) if curr_z.ndim == 3 else out

        uv_prior = data.x[:, NodeFeat.UV_PRIOR]
        p_prior = data.x[:, NodeFeat.SHEAR_POT]
        mu_prior = data.x[:, NodeFeat.MU_PRIOR]
        priors = torch.cat([uv_prior, p_prior, mu_prior], dim=1)
        z_warm_start = z + self.z_prior_proj(priors)
        z_init = z_warm_start.unsqueeze(0) if z_warm_start.ndim == 2 else z_warm_start

        with torch.no_grad():
            if solver == "picard":
                z_star = z_init
                for _ in range(self.max_iters):
                    z_star = f_coupled(z_star)
            else:
                z_star = anderson_acceleration(
                    f_coupled, z_init, batch_idx=batch_idx,
                    max_iter=self.max_iters, beta=anderson_beta, warmup_iters=anderson_warmup_iters
                )

        z_star_req = z_star.detach().requires_grad_(self.training)
        z_out = f_coupled(z_star_req)
        if self.training:
            eps = torch.randn_like(z_out)
            vjp = torch.autograd.grad(z_out, z_star_req, grad_outputs=eps, create_graph=True)[0]
            jac_loss = torch.mean(vjp ** 2)
            z_eq = z_out.squeeze(0) if z_out.ndim == 3 else z_out
        else:
            z_eq = z_out.squeeze(0) if z_out.ndim == 3 else z_out
            jac_loss = torch.tensor(0.0, device=z_eq.device)
        return z_eq, jac_loss, x_enc

    def _decode_pred_from_z(self, data, z: Tensor, x_enc: Optional[Tensor] = None) -> Tensor:
        """Decode kinematics (+ mu, wss) from an equilibrium latent ``z``."""
        if getattr(self, "decoder_skip", False):
            if x_enc is None:
                raise ValueError("decoder_skip is on but x_enc was not supplied")
            z_dec = torch.cat([z, x_enc], dim=1)
        else:
            z_dec = z
        mu_raw_state = self.mu_decoder(z)
        mu = self.mu_inf_nd + (self.mu_0_nd - self.mu_inf_nd) * torch.sigmoid(mu_raw_state)

        if self.siren_decoder is not None:
            pos_nd = getattr(data, "pos_nd", None)
            if pos_nd is None:
                pos_nd = getattr(data, "pos", None)
            if pos_nd is None:
                pos_nd = data.x[:, NodeFeat.XY]
                # Leaf tensor so autograd can differentiate NS / hard-BC terms w.r.t. coordinates.
                pos_nd = pos_nd.clone().requires_grad_(True)
            # The SIREN is a COORDINATE network -- feeding it the absolute frame is the single
            # strongest memorisation path in the model, stronger than the encoder's Fourier
            # block.  Centring the encoder alone left `comsol041` at 0.715 relative change
            # under a full-span translation (0.284 before); both paths have to be canonicalised
            # together or the invariance is not real.  See `_canonical_coords`.
            pos_nd = _canonical_coords(pos_nd)
            uvp, siren_pos = self.siren_decoder(z_dec, pos_nd)
            data.siren_pos = siren_pos
            u_v_p = uvp[:, PredChannels.KINEMATICS]
        else:
            assert self.kinematics_decoder is not None
            kinematics_out = self.kinematics_decoder(z_dec)
            u_v_p = kinematics_out[:, PredChannels.KINEMATICS]

        if getattr(self, "residual_gain_head", None) is not None:
            # Scale the residual BEFORE the hard BC multiplies it by sdf.  `d/dn(sdf * g*uvp)` at
            # the wall is `g*uvp`, so this is a direct, per-node, learnable gain on wall shear --
            # the one quantity `dsrxScale` measures and the LayerNorm shell cannot vary.
            if x_enc is None:
                raise ValueError("residual_gain is on but x_enc was not supplied")
            log_g = self.residual_gain_head(x_enc).clamp(
                -self.residual_gain_clamp, self.residual_gain_clamp
            )
            u_v_p = torch.cat([u_v_p[:, :2] * torch.exp(log_g), u_v_p[:, 2:3]], dim=1)

        if getattr(self, "residual_scale", None) is not None:
            # Applied to the VELOCITY residual only.  Pressure is decoded absolutely, not as a
            # correction to a prior, so gating it to zero would suppress a head rather than
            # start it neutral.
            u_v_p = torch.cat([u_v_p[:, :2] * self.residual_scale, u_v_p[:, 2:3]], dim=1)

        if self.use_hard_bcs:
            # SDF is already [N, 1]; do not add another singleton (would break broadcast with [N, 2]).
            sdf = data.x[:, NodeFeat.SDF]
            uv_prior = data.x[:, NodeFeat.UV_PRIOR]
            if self.bc_envelope:
                # Soft-envelope hard-BC: exact at sdf=0, but keeps derivatives closer to wall.
                envelope = 1.0 - torch.exp(-self.bc_lambda * sdf)
                if self.bc_envelope_decay > 0.0:
                    far = torch.exp(-self.bc_envelope_decay * sdf)
                    floor = self.bc_envelope_floor
                    if floor > 0.0:
                        far = floor + (1.0 - floor) * far
                    envelope = envelope * far
                u_v_constrained = uv_prior + envelope * u_v_p[:, :2]
            else:
                u_v_constrained = uv_prior + sdf * u_v_p[:, :2]
            u_v_p = torch.cat([u_v_constrained, u_v_p[:, 2:3]], dim=1)

        if self.wss_fuse:
            wss_pred = self.wss_decoder(torch.cat([z, u_v_p, mu], dim=1))
            if getattr(self, "shear_head", True):
                shear_pred = self.shear_decoder(torch.cat([z, u_v_p, mu], dim=1))
        else:
            wss_pred = self.wss_decoder(z)
            if getattr(self, "shear_head", True):
                shear_pred = self.shear_decoder(z)
        
        out_list = [u_v_p, mu, wss_pred]
        if getattr(self, "shear_head", True):
            out_list.append(shear_pred)
            
        return torch.cat(out_list, dim=1)

    @torch.no_grad()
    def solve_latent(
        self,
        data,
        solver: str = "anderson",
        anderson_beta: float = 0.8,
        anderson_warmup_iters: int = 5,
    ) -> torch.Tensor:
        """Frozen inference: DEQ equilibrium latent ``z_kin`` per node, shape ``[N, latent_dim]``."""
        was_training = self.training
        self.eval()
        z, _, x_enc = self._solve_equilibrium_z(
            data,
            solver=solver,
            anderson_beta=anderson_beta,
            anderson_warmup_iters=anderson_warmup_iters,
        )
        if was_training:
            self.train()
        return z

    @torch.no_grad()
    def predict_uv_and_latent(
        self,
        data,
        solver: str = "anderson",
        anderson_beta: float = 0.8,
        anderson_warmup_iters: int = 5,
    ) -> tuple[Tensor, Tensor]:
        """One DEQ solve -> ``(pred [N, C], z_kin [N, latent_dim])`` (inference only)."""
        was_training = self.training
        self.eval()
        z, _, x_enc = self._solve_equilibrium_z(
            data,
            solver=solver,
            anderson_beta=anderson_beta,
            anderson_warmup_iters=anderson_warmup_iters,
        )
        pred = self._decode_pred_from_z(data, z, x_enc)
        if was_training:
            self.train()
        return pred, z

    @torch.enable_grad()
    def forward(self, data, solver="anderson", anderson_beta=0.8, anderson_warmup_iters=5, current_n=None):
        z, jac_loss, x_enc = self._solve_equilibrium_z(
            data,
            solver=solver,
            anderson_beta=anderson_beta,
            anderson_warmup_iters=anderson_warmup_iters,
        )
        pred = self._decode_pred_from_z(data, z, x_enc)
        return (pred, jac_loss) if self.training else pred

# Backward-compatible alias (not Li et al. GINO)
GINO_DEQ = RGP_DEQ

