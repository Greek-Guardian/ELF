"""Flow-matching noise / timestep / sampling utilities (PyTorch).

Translated from src/utils/sampling_utils.py (JAX/jax.jit).
All @jax.jit / static_argnums decorators are removed — PyTorch eager
mode is used; torch.compile() can be applied at the call site if needed.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise scheduler
# ---------------------------------------------------------------------------

def add_noise(
    x0: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    config,
    cond_seq_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Flow-matching interpolation z = t*x0 + (1-t)*noise*scale.

    Conditioning tokens are kept clean (cond_seq_mask positions → x0).
    """
    t_exp = t.reshape(-1, 1, 1)
    z = t_exp * x0 + (1.0 - t_exp) * noise * config.denoiser_noise_scale
    if cond_seq_mask is not None:
        z = cond_seq_mask * x0 + (1.0 - cond_seq_mask) * z
    return z


# ---------------------------------------------------------------------------
# Timestep samplers
# ---------------------------------------------------------------------------

def sample_timesteps(
    batch_size: int,
    P_mean: float = -0.8,
    P_std: float = 0.8,
    time_schedule: str = "logit_normal",
    device: torch.device = None,
) -> torch.Tensor:
    """Sample timesteps using various time schedules → (batch_size,) in [0,1]."""
    if time_schedule == "logit_normal":
        z = torch.randn(batch_size, device=device) * P_std + P_mean
        return torch.sigmoid(z)
    if time_schedule == "uniform":
        return torch.rand(batch_size, device=device)
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def get_sampling_steps(
    n_steps: int,
    time_schedule: str = "logit_normal",
    P_mean: float = -0.8,
    P_std: float = 0.8,
    device: torch.device = None,
) -> torch.Tensor:
    """Return a (n_steps+1,) tensor of t values in [0,1] for a sampling run."""
    if time_schedule == "uniform":
        return torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    if time_schedule == "logit_normal":
        steps = sample_timesteps(
            n_steps - 1, P_mean=P_mean, P_std=P_std,
            time_schedule=time_schedule, device=device,
        )
        steps_sorted, _ = torch.sort(steps)
        return torch.cat([
            torch.tensor([0.0], device=device),
            steps_sorted,
            torch.tensor([1.0], device=device),
        ])
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


# ---------------------------------------------------------------------------
# CFG scale sampler
# ---------------------------------------------------------------------------

def sample_cfg_scale(
    batch_size: int,
    cfg_min: float = 0.0,
    cfg_max: float = 3.0,
    device: torch.device = None,
) -> torch.Tensor:
    """Sample CFG scale from log-uniform distribution in [cfg_min, cfg_max]."""
    u = torch.rand(batch_size, device=device)
    a = torch.tensor(1.0 + cfg_min, device=device)
    b = torch.tensor(1.0 + cfg_max, device=device)
    return a * torch.exp(u * torch.log(b / a)) - 1.0


# ---------------------------------------------------------------------------
# Conditioning helpers
# ---------------------------------------------------------------------------

def restore_cond(
    z_updated: torch.Tensor,
    cond_seq: torch.Tensor,
    cond_seq_mask: torch.Tensor,
) -> torch.Tensor:
    """Restore clean conditioning tokens in z after a denoising step."""
    mask = cond_seq_mask
    while mask.ndim < z_updated.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask > 0, cond_seq, z_updated)


def restore_vx(
    v: torch.Tensor,
    x: torch.Tensor,
    cond_seq: Optional[torch.Tensor],
    cond_seq_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Restore cond positions: x → clean cond_seq, v → 0."""
    if cond_seq is not None:
        x = restore_cond(x, cond_seq, cond_seq_mask)
        v = restore_cond(v, torch.zeros_like(cond_seq), cond_seq_mask)
    return v, x


# ---------------------------------------------------------------------------
# Network output conversion
# ---------------------------------------------------------------------------

def net_out_to_v_x(
    net_out,
    z: torch.Tensor,
    t: torch.Tensor,
    t_eps: float = 5e-2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert x_pred network output to (v, x).

    When net_out is a tuple (denoised_output, decoder_logits),
    decoder logits are discarded here.
    """
    if isinstance(net_out, tuple):
        net_out = net_out[0]
    t_reshaped = t.reshape(-1, 1, 1)
    x = net_out
    v = (x - z) / torch.clamp(1.0 - t_reshaped, min=t_eps)
    return v, x


# ---------------------------------------------------------------------------
# Forward passes for sampling (replaces @jax.jit decorated versions)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _forward_sample_self_cond(
    model,
    z: torch.Tensor,
    t_batch: torch.Tensor,
    x_pred_prev: Optional[torch.Tensor],
    config,
    self_cond_cfg_scale: float,
    cond_seq: Optional[torch.Tensor],
    cond_seq_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass with self-conditioning (no_grad, eager mode)."""
    t_eps = config.t_eps

    def _restore(v, x):
        return restore_vx(v, x, cond_seq, cond_seq_mask)

    if config.num_self_cond_cfg_tokens > 0:
        if x_pred_prev is None:
            x_pred_prev = restore_cond(
                torch.zeros_like(z), cond_seq, cond_seq_mask
            ) if cond_seq is not None else torch.zeros_like(z)
        z_input_cond = torch.cat([z, x_pred_prev], dim=-1)
        sc_scale_batch = torch.full((z.shape[0],), self_cond_cfg_scale,
                                    dtype=z.dtype, device=z.device)
        net_out_cond = model(z_input_cond, t_batch,
                             self_cond_cfg_scale=sc_scale_batch)
        v_cond, x_cond = net_out_to_v_x(net_out_cond, z, t_batch, t_eps)
        return _restore(v_cond, x_cond)

    if config.self_cond_prob == 0:
        net_out = model(z, t_batch)
        v, x = net_out_to_v_x(net_out, z, t_batch, t_eps)
        return _restore(v, x)

    # Combined unconditional + conditional
    if self_cond_cfg_scale != 1 or x_pred_prev is None:
        z_uncond = (
            restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
            if cond_seq is not None else torch.zeros_like(z)
        )
        z_input_uncond = torch.cat([z, z_uncond], dim=-1)
        net_out_uncond = model(z_input_uncond, t_batch)
        v_uncond, x_uncond = net_out_to_v_x(net_out_uncond, z, t_batch, t_eps)
        v_uncond, x_uncond = _restore(v_uncond, x_uncond)
        if self_cond_cfg_scale == 0.0 or x_pred_prev is None:
            return v_uncond, x_uncond

    z_input_cond = torch.cat([z, x_pred_prev], dim=-1)
    net_out_cond = model(z_input_cond, t_batch)
    v_cond, x_cond = net_out_to_v_x(net_out_cond, z, t_batch, t_eps)
    v_cond, x_cond = _restore(v_cond, x_cond)
    if self_cond_cfg_scale == 1:
        return v_cond, x_cond

    v_out = v_uncond + self_cond_cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + self_cond_cfg_scale * (x_cond - x_uncond)
    return _restore(v_out, x_out)


@torch.no_grad()
def _forward_sample(
    model,
    z: torch.Tensor,
    t_batch: torch.Tensor,
    x_pred_prev: Optional[torch.Tensor],
    config,
    cfg_scale: float,
    self_cond_cfg_scale: float,
    cond_seq: Optional[torch.Tensor],
    cond_seq_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass with optional self-conditioning and CFG."""
    v_cond, x_cond = _forward_sample_self_cond(
        model, z, t_batch, x_pred_prev, config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    if cfg_scale == 1.0:
        return v_cond, x_cond

    # Unconditional forward
    z_uncond = (
        restore_cond(z, torch.zeros_like(z), cond_seq_mask)
        if cond_seq_mask is not None else z
    )
    x_pred_prev_uncond = (
        None if x_pred_prev is None
        else restore_cond(x_pred_prev, torch.zeros_like(x_pred_prev), cond_seq_mask)
        if cond_seq_mask is not None else x_pred_prev
    )
    v_uncond, x_uncond = _forward_sample_self_cond(
        model, z_uncond, t_batch, x_pred_prev_uncond, config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=torch.zeros_like(cond_seq) if cond_seq is not None else None,
        cond_seq_mask=cond_seq_mask,
    )

    v_out = v_uncond + cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + cfg_scale * (x_cond - x_uncond)
    return restore_vx(v_out, x_out, cond_seq, cond_seq_mask)


@torch.no_grad()
def _ode_step(
    model,
    z: torch.Tensor,
    t: float,
    t_next: float,
    x_pred_prev: Optional[torch.Tensor],
    config,
    cfg_scale: float,
    self_cond_cfg_scale: float,
    cond_seq: Optional[torch.Tensor],
    cond_seq_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single ODE (Euler) step."""
    t_batch = torch.full((z.shape[0],), t, dtype=z.dtype, device=z.device)
    v_pred, x_pred = _forward_sample(
        model, z, t_batch, x_pred_prev, config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    return z + (t_next - t) * v_pred, x_pred


@torch.no_grad()
def _sde_step(
    model,
    z: torch.Tensor,
    t: float,
    t_next: float,
    x_pred_prev: Optional[torch.Tensor],
    config,
    cfg_scale: float,
    self_cond_cfg_scale: float,
    cond_seq: Optional[torch.Tensor],
    cond_seq_mask: Optional[torch.Tensor],
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-step SDE-style sampler (hybrid noise scaling)."""
    h = t_next - t
    alpha = max(0.0, min(1.0, 1.0 - gamma * h))
    t_back = alpha * t
    eps = torch.randn_like(z) * config.denoiser_noise_scale
    z_back = alpha * z + (1.0 - alpha) * eps
    if cond_seq is not None:
        z_back = restore_cond(z_back, cond_seq, cond_seq_mask)
    t_batch = torch.full((z.shape[0],), t_back, dtype=z.dtype, device=z.device)
    v_pred, x_pred = _forward_sample(
        model, z_back, t_batch, x_pred_prev, config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    return z_back + (t_next - t_back) * v_pred, x_pred
