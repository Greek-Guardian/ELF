"""Generation / evaluation runner (PyTorch).

Translated from src/generation.py (JAX/pmap).
Multi-GPU: uses the passed-in model directly; caller wraps with DDP.
"""

import itertools
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from configs.config import Config, SamplingConfig
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import upload_output_dir_to_hf
from utils.data_utils import get_dataloader, get_pad_token_id
from utils.encoder_utils import encode_text
from utils.generation_utils import (
    mask_after_eos, shift_left,
    generate_samples, decode_latent_to_ids,
    setup_generation, _build_run_name,
)
from utils.sampling_utils import get_sampling_steps
from utils.metrics_utils import Metrics as PPLMetrics, compute_bleu, compute_rouge


def _is_rank0() -> bool:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True


def run_generation(
    model: nn.Module,
    ema_params: Optional[Dict[str, torch.Tensor]],
    encoder: nn.Module,
    eval_dataset,
    tokenizer,
    config: Config,
    device: torch.device,
    batch_size: int,
) -> None:
    """Run generation for all sampling configs."""
    for sc_idx, sc in enumerate(config.sampling_configs):
        if len(config.sampling_configs) > 1:
            log_for_0(f"\n--- Sampling config {sc_idx + 1}/{len(config.sampling_configs)} ---")
        kwargs = dict(
            model=model,
            ema_params=ema_params,
            tokenizer=tokenizer,
            config=config,
            sampling_config=sc,
            device=device,
            batch_size=batch_size,
            num_samples=config.num_samples,
        )
        if eval_dataset is None:
            test_generation_uncond(**kwargs)
        else:
            test_generation_cond(**kwargs, encoder=encoder, dataset=eval_dataset)


# ---------------------------------------------------------------------------
# Unconditional generation
# ---------------------------------------------------------------------------

def test_generation_uncond(
    model: nn.Module,
    ema_params: Optional[Dict[str, torch.Tensor]],
    tokenizer,
    config: Config,
    sampling_config: SamplingConfig,
    device: torch.device,
    num_samples: int = 64,
    batch_size: int = 64,
):
    """Unconditional text generation."""
    sampling_method = sampling_config.sampling_method
    time_schedule = sampling_config.time_schedule
    log_for_0(f"Config: {sampling_config}")

    # Use EMA weights for inference
    _apply_ema(model, ema_params)
    model.eval()

    encoder_config, _ = _get_encoder_config(config)
    d_model = encoder_config.d_model

    pad_token_id = get_pad_token_id(tokenizer)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    ppl_metrics = None
    if config.online_eval:
        ppl_metrics = PPLMetrics(
            gen_ppl_eval_model_name_or_path=config.eval_ppl_model,
            eval_ppl_batch_size=config.eval_ppl_batch_size,
            eval_context_size=config.eval_ppl_max_length,
        )

    for num_steps, cfg_scale, sc_cfg in itertools.product(
        sampling_config.num_sampling_steps, [1], sampling_config.self_cond_cfg_scales
    ):
        log_for_0(
            f"\n--- Method: {sampling_method}, Steps: {num_steps}, "
            f"CFG: {cfg_scale}, SC-CFG: {sc_cfg} ---"
        )
        all_generated = []
        generation_time = 0.0
        decode_time = 0.0
        samples_processed = 0
        num_batches = (num_samples + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Generating samples", disable=not _is_rank0()):
            if samples_processed >= num_samples:
                break
            current_bs = min(batch_size, num_samples - samples_processed)

            t_steps = get_sampling_steps(
                num_steps,
                time_schedule=time_schedule,
                P_mean=config.denoiser_p_mean,
                P_std=config.denoiser_p_std,
                device=device,
            )
            z = torch.randn(current_bs, config.max_length, d_model, device=device) * config.denoiser_noise_scale

            gen_start = time.time()
            latent = generate_samples(
                model=model, z=z, t_steps=t_steps,
                config=config, sampling_config=sampling_config,
                cfg_scale=cfg_scale, self_cond_cfg_scale=sc_cfg,
            )
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            generation_time += time.time() - gen_start

            dec_start = time.time()
            t_final_val = t_steps[-1].item()
            predicted_ids = decode_latent_to_ids(latent, model, t_final_val, config, sc_cfg)
            predicted_ids = mask_after_eos(predicted_ids, eos_token_id, pad_token_id)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            decode_time += time.time() - dec_start

            for i in range(predicted_ids.shape[0]):
                if samples_processed >= num_samples:
                    break
                text = tokenizer.decode(predicted_ids[i].cpu().tolist(), skip_special_tokens=True)
                all_generated.append((samples_processed, text))
                samples_processed += 1

        log_for_0(
            f"Generation: {generation_time:.2f}s ({num_steps} steps) | Decode: {decode_time:.2f}s"
        )

        name = _build_run_name(
            sampling_method, num_steps, cfg_scale, sc_cfg,
            time_schedule, getattr(sampling_config, "sde_gamma", 0.0), suffix="uncond",
        )

        if _is_rank0():
            out_dir = os.path.join(config.output_dir, name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "all_generated.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for tid, gen in all_generated:
                    f.write(json.dumps({"id": tid, "generated": gen}, ensure_ascii=False) + "\n")
            log_for_0(f"Saved {len(all_generated)} texts to {out_path}")
            upload_output_dir_to_hf(config.output_dir, config.hf_repo_id, reason="generation")

        if ppl_metrics is not None and _is_rank0():
            ppl_metrics.reset()
            nonempty = [g for _, g in all_generated if g.strip()]
            if nonempty:
                ppl_results = ppl_metrics.record_generative_perplexity(
                    nonempty, max_length=config.eval_ppl_max_length
                )
                log_for_0(f"PPL: {ppl_results['ppl']:.4f}  Entropy: {ppl_results['mean_entropy']:.4f}")


# ---------------------------------------------------------------------------
# Conditional generation
# ---------------------------------------------------------------------------

def test_generation_cond(
    model: nn.Module,
    ema_params: Optional[Dict[str, torch.Tensor]],
    encoder: nn.Module,
    tokenizer,
    config: Config,
    sampling_config: SamplingConfig,
    dataset,
    device: torch.device,
    num_samples: int = 64,
    batch_size: int = 64,
):
    """Conditional text generation."""
    sampling_method = sampling_config.sampling_method
    time_schedule = sampling_config.time_schedule

    _apply_ema(model, ema_params)
    model.eval()

    encoder_config, _ = _get_encoder_config(config)
    d_model = encoder_config.d_model

    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    eos_token_id = tokenizer.eos_token_id

    dataloader = get_dataloader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length,
        distributed=False,
    )

    for num_steps, cfg_scale, sc_cfg in itertools.product(
        sampling_config.num_sampling_steps,
        sampling_config.cfgs,
        sampling_config.self_cond_cfg_scales,
    ):
        log_for_0(
            f"\n--- Steps: {num_steps}, CFG: {cfg_scale}, SC-CFG: {sc_cfg} ---"
        )
        all_generated = []
        generation_time = 0.0
        samples_processed = 0

        for batch in dataloader:
            if samples_processed >= num_samples:
                break

            input_ids = torch.tensor(batch["input_ids"], device=device)
            enc_attn_mask = torch.tensor(batch["encoder_attention_mask"], device=device)
            cond_seq_mask_arr = torch.tensor(batch["cond_seq_mask"], device=device)

            with torch.no_grad():
                cond_seq = encode_text(
                    input_ids=input_ids,
                    attention_mask=enc_attn_mask,
                    encoder=encoder,
                    latent_mean=config.latent_mean,
                    latent_std=config.latent_std,
                )

            t_steps = get_sampling_steps(
                num_steps, time_schedule=time_schedule,
                P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                device=device,
            )
            cur_bs = input_ids.shape[0]
            z = torch.randn(cur_bs, config.max_length, d_model, device=device) * config.denoiser_noise_scale

            gen_start = time.time()
            latent = generate_samples(
                model=model, z=z, t_steps=t_steps,
                config=config, sampling_config=sampling_config,
                cfg_scale=cfg_scale, self_cond_cfg_scale=sc_cfg,
                cond_seq=cond_seq, cond_seq_mask=cond_seq_mask_arr,
            )
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            generation_time += time.time() - gen_start

            t_final_val = t_steps[-1].item()
            predicted_ids = decode_latent_to_ids(latent, model, t_final_val, config, sc_cfg)
            gen_length = config.max_length - (config.max_input_length or 0)
            cond_lens = cond_seq_mask_arr.long().sum(dim=1)
            predicted_ids = shift_left(predicted_ids, cond_lens, 0)[:, :gen_length]
            predicted_ids = mask_after_eos(predicted_ids, eos_token_id, pad_token_id)

            original_texts = batch.get("target", [""] * cur_bs)
            context_texts = batch.get("input", [""] * cur_bs)

            for i in range(min(cur_bs, num_samples - samples_processed)):
                text = tokenizer.decode(predicted_ids[i].cpu().tolist(), skip_special_tokens=True)
                orig = original_texts[i] if isinstance(original_texts, list) else ""
                ctx = context_texts[i] if isinstance(context_texts, list) else ""
                all_generated.append((samples_processed, orig, text, ctx))
                samples_processed += 1

        log_for_0(f"Generation: {generation_time:.2f}s ({num_steps} steps)")

        name = _build_run_name(
            sampling_method, num_steps, cfg_scale, sc_cfg,
            time_schedule, getattr(sampling_config, "sde_gamma", 0.0), suffix="cond",
        )

        if _is_rank0():
            out_dir = os.path.join(config.output_dir, name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "all_generated.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for tid, orig, gen, ctx in all_generated:
                    f.write(json.dumps({"id": tid, "generated": gen}, ensure_ascii=False) + "\n")
            log_for_0(f"Saved {len(all_generated)} texts to {out_path}")
            upload_output_dir_to_hf(config.output_dir, config.hf_repo_id, reason="generation")

            if config.online_eval and all_generated:
                hyps = [g for _, _, g, _ in all_generated]
                refs = [r for _, r, _, _ in all_generated]
                bleu = compute_bleu(hyps, refs)
                rouge = compute_rouge(hyps, refs)
                log_for_0(
                    f"BLEU: {bleu:.2f}  ROUGE-1: {rouge['rouge1']:.2f}  "
                    f"ROUGE-2: {rouge['rouge2']:.2f}  ROUGE-L: {rouge['rougeL']:.2f}"
                )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_ema(model: nn.Module, ema_params: Optional[Dict[str, torch.Tensor]]):
    """Load EMA weights into model for inference (if available)."""
    if ema_params is not None:
        # Load EMA weights; non-matching keys are silently ignored
        current = model.state_dict()
        for k, v in ema_params.items():
            if k in current:
                current[k] = v.to(current[k].device)
        model.load_state_dict(current, strict=False)


def _get_encoder_config(config: Config):
    """Return (encoder_config, None) without instantiating model."""
    from modules.t5_encoder import get_encoder, _T5_CONFIGS
    if config.encoder_model_name in _T5_CONFIGS:
        return _T5_CONFIGS[config.encoder_model_name], None
    return _T5_CONFIGS["t5-small"], None
