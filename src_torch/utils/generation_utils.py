"""Generation utilities (PyTorch).

Translated from src/utils/generation_utils.py (JAX/pmap/lax.scan).
Key changes:
  - jax.lax.scan → Python for loop
  - jax.pmap → plain function call (caller manages multi-GPU)
  - jnp ops → torch ops
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from configs.config import Config, SamplingConfig
from utils.logging_utils import log_for_0
from utils.sampling_utils import (
    restore_cond,
    _ode_step,
    _sde_step,
    get_sampling_steps,
    net_out_to_v_x,
)
from modules.t5_encoder import get_encoder


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def mask_after_eos(
    predicted_ids: torch.Tensor,
    eos_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Mask everything at/after the first EOS token per sequence."""
    eos_mask = (predicted_ids == eos_token_id)
    keep_mask = torch.cumsum(eos_mask.long(), dim=1) == 0
    return torch.where(keep_mask, predicted_ids, torch.full_like(predicted_ids, pad_token_id))


def shift_left(
    x: torch.Tensor,
    shift_per_sample: torch.Tensor,
    pad_value: int = 0,
    axis: int = 1,
) -> torch.Tensor:
    """Shift each sample left along the sequence axis; pad emptied positions."""
    if x.ndim < 2:
        raise ValueError("x must have at least 2 dimensions")
    shift_per_sample = shift_per_sample.to(dtype=torch.long, device=x.device)
    if axis != 1:
        x = x.transpose(1, axis)
    seq_len = x.shape[1]
    base_idx = torch.arange(seq_len, device=x.device).unsqueeze(0)    # (1, S)
    gather_idx = shift_per_sample.unsqueeze(1) + base_idx              # (B, S)
    valid = gather_idx < seq_len
    gather_idx = gather_idx.clamp(0, seq_len - 1)
    if x.ndim == 2:
        shifted = x.gather(1, gather_idx)
        shifted = torch.where(valid, shifted, torch.full_like(shifted, pad_value))
    else:
        extra = tuple(range(2, x.ndim))
        idx_exp = gather_idx.view(*gather_idx.shape, *([1] * len(extra))).expand_as(x)
        shifted = x.gather(1, idx_exp)
        valid_exp = valid.view(*valid.shape, *([1] * len(extra))).expand_as(shifted)
        shifted = torch.where(valid_exp, shifted, torch.full_like(shifted, pad_value))
    if axis != 1:
        shifted = shifted.transpose(1, axis)
    return shifted


# ---------------------------------------------------------------------------
# Core generation loop (replaces pmap + lax.scan)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(
    model: nn.Module,
    z: torch.Tensor,
    t_steps: torch.Tensor,
    config: Config,
    sampling_config: SamplingConfig,
    cfg_scale: float,
    self_cond_cfg_scale: float,
    cond_seq: Optional[torch.Tensor] = None,
    cond_seq_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate samples using ODE or SDE sampling.

    Args:
        model:              ELF model (eval mode)
        z:                  initial noise (B, S, d_model)
        t_steps:            (n_steps+1,) timestep array from get_sampling_steps
        config, sampling_config, cfg_scale, self_cond_cfg_scale: sampling config
        cond_seq:           optional conditioning latents (B, S, d_model)
        cond_seq_mask:      optional (B, S) mask, 1=cond token

    Returns:
        final latent z of shape (B, S, d_model)
    """
    method = sampling_config.sampling_method
    batch_size, max_length, d_model = z.shape
    device = z.device

    if cond_seq is None:
        cond_seq = torch.zeros(batch_size, max_length, d_model, device=device)
        cond_seq_mask = torch.zeros(batch_size, max_length, device=device)

    z = restore_cond(z, cond_seq, cond_seq_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)

    step_kwargs = dict(
        model=model,
        config=config,
        cfg_scale=cfg_scale,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq,
        cond_seq_mask=cond_seq_mask,
    )

    sde_gamma = getattr(sampling_config, "sde_gamma", 0.0)
    n_pairs = len(t_steps) - 2   # last step done separately (always ODE)

    for i in range(n_pairs):
        t_cur = t_steps[i].item()
        t_nxt = t_steps[i + 1].item()
        if method == "sde":
            z, x_pred = _sde_step(
                z=z, t=t_cur, t_next=t_nxt, x_pred_prev=x_pred,
                gamma=sde_gamma, **step_kwargs,
            )
        else:
            z, x_pred = _ode_step(
                z=z, t=t_cur, t_next=t_nxt, x_pred_prev=x_pred, **step_kwargs,
            )

    # Last step always ODE
    z, _ = _ode_step(
        z=z, t=t_steps[-2].item(), t_next=t_steps[-1].item(),
        x_pred_prev=x_pred, **step_kwargs,
    )
    return z


# ---------------------------------------------------------------------------
# Decoder head pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_latent_to_ids(
    z: torch.Tensor,
    model: nn.Module,
    t_final_val: float,
    config: Config,
    self_cond_cfg_scale: float,
) -> torch.Tensor:
    """Run the DLM decoder head on z → token ids (B, S)."""
    batch_size = z.shape[0]
    device = z.device
    t_final = torch.full((batch_size,), t_final_val, dtype=z.dtype, device=device)
    sc_scale_batch = (
        torch.full((batch_size,), self_cond_cfg_scale, dtype=z.dtype, device=device)
        if config.num_self_cond_cfg_tokens > 0 else None
    )
    z_input = (
        torch.cat([z, torch.zeros_like(z)], dim=-1) if config.self_cond_prob > 0 else z
    )
    _, decoder_logits = model(
        z_input, t_final,
        self_cond_cfg_scale=sc_scale_batch,
        decoder_step_active=True,
    )
    return decoder_logits.argmax(dim=-1)


# ---------------------------------------------------------------------------
# Shared setup helper
# ---------------------------------------------------------------------------

def setup_generation(state_model: nn.Module, config: Config, batch_size: int, header: str):
    """Log header and compute batch sizing info."""
    log_for_0("\n" + "=" * 70)
    log_for_0(header)
    log_for_0("=" * 70)

    encoder_config, _ = get_encoder(config.encoder_model_name)
    d_model = encoder_config.d_model
    log_for_0(f"T5 d_model: {d_model}")
    log_for_0(f"Batch size: {batch_size}")
    return d_model


def _build_run_name(
    sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
    time_schedule, sde_gamma, suffix,
):
    ts_str = f"-ts_{time_schedule}"
    sccfg_str = f"-sccfg{self_cond_cfg_scale}" if self_cond_cfg_scale != 1.0 else ""
    sde_str = f"-gamma{sde_gamma}" if sampling_method == "sde" else ""
    return (
        f"{sampling_method}-steps{num_sampling_steps}-cfg{cfg_scale}"
        f"{sccfg_str}{ts_str}{sde_str}-{suffix}"
    )
