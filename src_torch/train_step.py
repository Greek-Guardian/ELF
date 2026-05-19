"""Per-step training function for ELF (PyTorch).

Translated from src/train_step.py (JAX/pmap/value_and_grad).
Key changes:
  - jax.value_and_grad → standard loss.backward()
  - jax.lax.cond(decoder_step_active, ...) → Python if/else
  - jax.lax.stop_gradient → .detach()
  - jax.random → torch random
  - pmap allreduce → handled automatically by DDP
  - EMA update is called from the training loop (not inside this fn)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sampling_utils import (
    add_noise,
    net_out_to_v_x,
    restore_cond,
    sample_cfg_scale,
    sample_timesteps,
)
from utils.encoder_utils import encode_text


def train_step(
    model: nn.Module,
    encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    config,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_accum_step: int = 0,
    do_optimizer_step: bool = True,
) -> Dict[str, float]:
    """Perform a single (possibly accumulation) training step.

    Args:
        model:             ELF model (possibly DDP-wrapped)
        encoder:           Frozen T5 encoder
        optimizer:         Torch optimizer
        batch:             Dict of tensors on the correct device
        config:            Config object
        scaler:            Optional GradScaler for AMP (bf16)
        grad_accum_step:   Which accumulation micro-step we are on (0-based)
        do_optimizer_step: Whether to call optimizer.step() after backward

    Returns:
        Dict with 'loss', 'l2_loss', 'ce_loss' as Python floats.
        Losses are per-branch (divided by branch sampling probability).
    """
    t_eps = config.t_eps
    self_cond_prob = config.self_cond_prob
    latent_mean = config.latent_mean
    latent_std = config.latent_std
    decoder_prob = config.decoder_prob
    decoder_noise_scale = config.decoder_noise_scale
    device = next(model.parameters()).device

    # ------------------------------------------------------------------
    # 1. Encode input tokens → x0
    # ------------------------------------------------------------------
    encoder_attention_mask = batch["encoder_attention_mask"]
    input_ids = batch["input_ids"]

    # Label drop: mask attention from x-tokens to cond-tokens
    if config.label_drop_prob > 0:
        drop = batch["label_drop_mask"].float().unsqueeze(1).unsqueeze(1)  # (B, 1, 1)
        cond_mask = batch["cond_seq_mask"]   # (B, S)
        # block_mask is 1 only at (non-cond row, cond col)
        block_mask = (1 - cond_mask).unsqueeze(2) * cond_mask.unsqueeze(1)  # (B, S, S)
        encoder_attention_mask = encoder_attention_mask * (1 - drop * block_mask)

    with torch.no_grad():
        x0 = encode_text(
            input_ids=input_ids,
            attention_mask=encoder_attention_mask,
            encoder=encoder,
            latent_mean=latent_mean,
            latent_std=latent_std,
        )

    batch_size, seq_length = x0.shape[0], x0.shape[1]

    # ------------------------------------------------------------------
    # 2. Sample timesteps and noise
    # ------------------------------------------------------------------
    t = sample_timesteps(
        batch_size,
        P_mean=config.denoiser_p_mean,
        P_std=config.denoiser_p_std,
        time_schedule=config.time_schedule,
        device=device,
    )

    noise = torch.randn_like(x0)
    cond_seq_mask = batch["cond_seq_mask"].unsqueeze(-1)  # (B, S, 1)
    attention_mask = batch["attention_mask"]

    if config.pad_token == "pad":
        loss_mask = attention_mask
    else:
        loss_mask = torch.ones_like(attention_mask)
    loss_mask = loss_mask * (1 - batch["cond_seq_mask"])  # ignore cond positions

    denoiser_z = add_noise(x0, noise, t, config, cond_seq_mask=cond_seq_mask)

    # Label drop on latents
    drop_flag = batch["label_drop_mask"].unsqueeze(1)  # (B, 1)
    if config.label_drop_prob > 0:
        mask = drop_flag.unsqueeze(2).bool() & (cond_seq_mask > 0)
        denoiser_z = torch.where(mask, torch.zeros_like(denoiser_z), denoiser_z)
        x0_for_target = torch.where(mask, torch.zeros_like(x0), x0)
    else:
        x0_for_target = x0

    decoder_targets = batch["input_ids"]  # (B, S) token ids

    # ------------------------------------------------------------------
    # 3. Decoder branch input (logit-normal-noised latent at t=1)
    # ------------------------------------------------------------------
    decoder_z_vals = (
        torch.randn(batch_size * seq_length, device=device)
        * config.decoder_p_std + config.decoder_p_mean
    )
    decoder_lambda_t = torch.sigmoid(decoder_z_vals).reshape(batch_size, seq_length, 1)
    decoder_noise = torch.randn_like(x0) * decoder_noise_scale
    decoder_z = decoder_lambda_t * x0_for_target + (1 - decoder_lambda_t) * decoder_noise

    # ------------------------------------------------------------------
    # 4. Velocity target for denoiser branch
    # ------------------------------------------------------------------
    t_expanded = t.reshape(-1, 1, 1)
    v_target = (x0_for_target - denoiser_z) / torch.clamp(1.0 - t_expanded, min=t_eps)

    # ------------------------------------------------------------------
    # 5. Self-conditioning setup
    # ------------------------------------------------------------------
    if self_cond_prob > 0:
        use_self_cond_mask = (
            (torch.rand(batch_size, device=device) < self_cond_prob)
            .float().reshape(-1, 1, 1)
        )
    else:
        use_self_cond_mask = None

    if config.num_self_cond_cfg_tokens > 0:
        self_cond_cfg_scale = sample_cfg_scale(
            batch_size,
            cfg_min=config.self_cond_cfg_min,
            cfg_max=config.self_cond_cfg_max,
            device=device,
        )
    else:
        self_cond_cfg_scale = None

    # ------------------------------------------------------------------
    # 6. Choose branch: decoder (CE) or denoiser (L2)
    # ------------------------------------------------------------------
    decoder_step_active = (torch.rand(1, device=device).item() < decoder_prob)

    # ------------------------------------------------------------------
    # 7. Loss function
    # ------------------------------------------------------------------
    def reduce_token_loss(per_token_loss, mask):
        mask = mask.float()
        safe = torch.where(mask > 0, per_token_loss, torch.zeros_like(per_token_loss))
        return (safe * mask).sum() / mask.sum().clamp(min=1.0)

    def get_z_input(z, t_input, sc_cfg_input):
        """Optionally prepend self-cond estimate to z."""
        if self_cond_prob == 0:
            return z
        z_uncond = restore_cond(torch.zeros_like(z), x0_for_target, cond_seq_mask)
        z_with_zeros = torch.cat([z, z_uncond], dim=-1)
        with torch.no_grad():
            net_out_init = model(
                z_with_zeros, t_input,
                self_cond_cfg_scale=sc_cfg_input,
            )
            _, x_pred_init = net_out_to_v_x(net_out_init, z, t_input, t_eps)
            x_pred_init = restore_cond(x_pred_init, x0_for_target, cond_seq_mask)
        x_pred_cond = x_pred_init * use_self_cond_mask
        x_pred_cond = restore_cond(x_pred_cond, x0_for_target, cond_seq_mask)
        return torch.cat([z, x_pred_cond], dim=-1)

    if decoder_step_active:
        # ---- Decoder (CE) branch ----
        decoder_t = torch.ones(batch_size, device=device)
        decoder_input = (
            torch.cat([decoder_z, torch.zeros_like(decoder_z)], dim=-1)
            if self_cond_prob > 0 else decoder_z
        )
        _, decoder_logits = model(
            decoder_input, decoder_t,
            self_cond_cfg_scale=self_cond_cfg_scale,
            decoder_step_active=True,
        )
        log_probs = F.log_softmax(decoder_logits.float(), dim=-1)
        ce = -log_probs.gather(
            dim=-1, index=decoder_targets.unsqueeze(-1)
        ).squeeze(-1)
        ce_loss = reduce_token_loss(ce, loss_mask)
        loss = ce_loss
        l2_loss = torch.tensor(0.0, device=device)
    else:
        # ---- Denoiser (L2) branch ----
        denoiser_input = get_z_input(denoiser_z, t, self_cond_cfg_scale)
        net_out, _ = model(
            denoiser_input, t,
            self_cond_cfg_scale=self_cond_cfg_scale,
            decoder_step_active=False,
        )
        v_pred, _ = net_out_to_v_x(net_out, denoiser_z, t, t_eps)

        # Self-conditioning CFG guidance on v_target (optional)
        v_final_target = v_target
        if config.num_self_cond_cfg_tokens > 0 and self_cond_prob > 0:
            with torch.no_grad():
                # Conditional pass
                z_uncond_sc = restore_cond(torch.zeros_like(denoiser_z), x0_for_target, cond_seq_mask)
                z_input_uncond_sc = torch.cat([denoiser_z, z_uncond_sc], dim=-1)
                net_uncond = model(z_input_uncond_sc, t,
                                   self_cond_cfg_scale=self_cond_cfg_scale)
                v_uncond_sc, x_uncond_sc = net_out_to_v_x(net_uncond, denoiser_z, t, t_eps)
                x_uncond_sc = restore_cond(x_uncond_sc, x0_for_target, cond_seq_mask)

                z_input_cond_sc = torch.cat([denoiser_z, x_uncond_sc], dim=-1)
                net_cond = model(z_input_cond_sc, t,
                                 self_cond_cfg_scale=self_cond_cfg_scale)
                v_cond_sc, _ = net_out_to_v_x(net_cond, denoiser_z, t, t_eps)

                sc_w = self_cond_cfg_scale.reshape(-1, 1, 1)
                sc_guidance = (1 - 1 / sc_w) * (v_cond_sc - v_uncond_sc)
                if use_self_cond_mask is not None:
                    sc_guidance = sc_guidance * use_self_cond_mask
                v_final_target = (v_target + sc_guidance).detach()

        per_dim_loss = (v_pred - v_final_target) ** 2
        l2_loss = reduce_token_loss(per_dim_loss.mean(dim=-1), loss_mask)
        loss = l2_loss
        ce_loss = torch.tensor(0.0, device=device)

    # ------------------------------------------------------------------
    # 8. Backward + optimizer step
    # ------------------------------------------------------------------
    # Scale loss for gradient accumulation
    scaled_loss = loss / config.grad_accum_steps

    if scaler is not None:
        scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    metrics = {
        "loss": loss.item(),
        "l2_loss": (l2_loss.item() / (1.0 - decoder_prob) if (1.0 - decoder_prob) > 0 else 0.0),
        "ce_loss": (ce_loss.item() / decoder_prob if decoder_prob > 0 else 0.0),
    }

    if do_optimizer_step:
        if scaler is not None:
            scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    return metrics
