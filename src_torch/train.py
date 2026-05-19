#!/usr/bin/env python
"""Main training script for ELF (PyTorch / DDP).

Translated from src/train.py (JAX/pmap/jax_utils).

Launch (H800, 8 GPUs):
    torchrun --nproc_per_node=8 train.py --config configs/training_configs/train_owt_ELF-B_h800_torch.yml

Single-device (CPU / debug):
    python train.py --config configs/training_configs/train_owt_ELF-B_h800_torch.yml
"""

import argparse
import copy
import logging
import os
import sys
import time
import yaml

# Ensure src_torch/ is on sys.path when run as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer
import wandb

from modules.model import ELF_models
from modules.t5_encoder import get_encoder
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import save_checkpoint, load_checkpoint, find_latest_checkpoint
from utils.train_utils import TrainState, get_optimizer, create_learning_rate_fn
from utils.data_utils import get_dataloader, prepare_batch, load_dataset, get_pad_token_id
from train_step import train_step
from generation import run_generation
from configs.config import load_config_from_yaml, apply_config_overrides, load_sampling_configs, SamplingConfig


logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO, force=True,
)
logger = logging.getLogger(__name__)
sys.stdout.reconfigure(line_buffering=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ELF Diffusion Model (PyTorch).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--config_override", action="append", default=[],
        help="Override config values (field=value). Repeatable.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def _is_rank0():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def _world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def setup_ddp():
    """Initialise process group if launched with torchrun / mpirun."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return local_rank
    return None


def get_device(local_rank):
    if local_rank is not None:
        return torch.device(f"cuda:{local_rank}")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_training(config):
    local_rank = setup_ddp()
    device = get_device(local_rank)

    log_for_0("=" * 60)
    log_for_0("ELF Diffusion Model Training (PyTorch)")
    log_for_0("=" * 60)
    log_for_0(f"Model: {config.model}")
    log_for_0(f"Encoder: {config.encoder_model_name}")
    log_for_0(f"Data: {config.data_path}")
    log_for_0(f"Max seq length: {config.max_length}")
    log_for_0(f"Output dir: {config.output_dir}")
    log_for_0(f"Device: {device}  World size: {_world_size()}")
    log_for_0("=" * 60)

    # --- wandb ---
    if config.use_wandb and _is_rank0():
        wandb_config = {k: getattr(config, k) for k in dir(config) if not k.startswith("_")}
        wandb_tags = config.wandb_tag.split(",") if config.wandb_tag else None
        wandb.init(
            project=config.wandb_project, entity=config.wandb_entity,
            name=config.wandb_run_name, id=config.wandb_run_name, resume=config.wandb_resume,
            tags=wandb_tags, config=wandb_config, dir="/tmp",
            settings=wandb.Settings(start_method="thread"),
        )
        log_for_0(f"Wandb: {wandb.run.url}")

    # --- Tokenizer ---
    log_for_0("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    log_for_0(f"Pad token id: {pad_token_id}")

    train_dataset, eval_dataset = load_dataset(config)

    # --- Encoder (frozen T5) ---
    log_for_0(f"Loading encoder: {config.encoder_model_name}...")
    encoder_config, encoder = get_encoder(config.encoder_model_name)
    encoder = encoder.to(device)
    encoder.eval()
    log_for_0(f"Encoder d_model: {encoder_config.d_model}")

    # --- ELF model ---
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = tokenizer.vocab_size

    log_for_0(f"Creating {config.model} model (vocab={vocab_size})...")
    model_fn = ELF_models[config.model]
    model = model_fn(
        text_encoder_dim=encoder_config.d_model,
        max_length=config.max_length,
        attn_drop=config.attn_dropout,
        proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log_for_0(f"ELF parameters: {total_params:,}")

    # --- Wrap with DDP ---
    if local_rank is not None:
        model = DDP(model, device_ids=[local_rank])

    # --- Batch size / step count ---
    num_devices = _world_size()
    if config.global_batch_size is not None:
        total_batch_size = config.global_batch_size
        local_batch_size = total_batch_size // num_devices
        config.batch_size = local_batch_size
    elif config.batch_size is not None:
        total_batch_size = config.batch_size * num_devices
        local_batch_size = config.batch_size
        config.global_batch_size = total_batch_size
    else:
        raise ValueError("Either global_batch_size or batch_size must be specified")

    steps_per_epoch = len(train_dataset) // total_batch_size
    num_train_steps = steps_per_epoch * config.epochs

    if config.warmup_steps >= 0:
        num_warmup_steps = config.warmup_steps
    elif config.warmup_epochs is not None:
        num_warmup_steps = int(config.warmup_epochs * steps_per_epoch)
    else:
        num_warmup_steps = 0

    grad_accum_steps = config.grad_accum_steps
    num_optimizer_steps = num_train_steps // grad_accum_steps
    num_warmup_optimizer_steps = num_warmup_steps // grad_accum_steps

    if config.lr is None or config.lr <= 0:
        config.lr = config.blr * (total_batch_size * grad_accum_steps) / 256

    log_for_0(
        f"local_bs={local_batch_size}, total_bs={total_batch_size} | "
        f"steps/epoch={steps_per_epoch}, total={num_train_steps}, "
        f"warmup={num_warmup_steps}, lr={config.lr:.2e}"
    )

    # --- Optimizer + LR schedule ---
    raw_model = model.module if isinstance(model, DDP) else model
    optimizer = get_optimizer(config, raw_model)
    lr_lambda = create_learning_rate_fn(
        num_optimizer_steps, num_warmup_optimizer_steps,
        config.lr, config.lr_schedule, config.min_lr,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- TrainState ---
    state = TrainState(
        model=raw_model,
        optimizer=optimizer,
        ema_params1=copy.deepcopy({k: v.cpu() for k, v in raw_model.state_dict().items()}),
    )
    optimizer.zero_grad()

    # --- Auto-resume ---
    if not config.resume:
        auto_ckpt = find_latest_checkpoint(config.output_dir)
        if auto_ckpt:
            config.resume = config.output_dir
            log_for_0(f"Auto-resuming from {auto_ckpt}")

    start_epoch, resume_step = 0, 0
    if config.resume:
        try:
            state, resume_step = load_checkpoint(config.resume, state, device=device)
            start_epoch = state.epoch
            log_for_0(f"Resumed from step {resume_step} (epoch {start_epoch})")
            # Advance scheduler to match current step
            for _ in range(resume_step // grad_accum_steps):
                scheduler.step()
        except Exception as e:
            log_for_0(f"Error loading checkpoint: {e}. Starting from scratch.")

    # --- Save config ---
    os.makedirs(config.output_dir, exist_ok=True)
    config_dict = {
        k: ([vars(sc) for sc in v] if isinstance(v, list) and v and isinstance(v[0], SamplingConfig) else v)
        for k, v in vars(config).items()
    }
    config_path = os.path.join(config.output_dir, "config.yml")
    if _is_rank0():
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        log_for_0(f"Config saved to {config_path}")

    # --- Sampling configs ---
    if config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)

    # --- DataLoader ---
    train_dataloader = get_dataloader(
        train_dataset, batch_size=local_batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=True,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length,
        distributed=(local_rank is not None),
    )

    # --- AMP scaler (bf16 on CUDA) ---
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=False)   # bf16 doesn't need scaling

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    log_for_0("\n" + "=" * 60)
    log_for_0("Starting Training")
    log_for_0("=" * 60)

    global_step = start_epoch * steps_per_epoch + (resume_step - start_epoch * steps_per_epoch)
    last_log_step = global_step
    last_log_time = time.time()
    last_save_epoch = float(start_epoch)
    train_metrics_accum = []
    accum_step = 0

    for epoch in range(start_epoch, config.epochs):
        log_for_0(f"\nEpoch {epoch + 1}/{config.epochs}")
        model.train()

        if hasattr(train_dataloader, "sampler") and hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        steps_to_skip = (resume_step - start_epoch * steps_per_epoch) if epoch == start_epoch else 0
        epoch_pbar = tqdm(
            total=steps_per_epoch, desc=f"Epoch {epoch + 1}",
            initial=steps_to_skip, disable=not _is_rank0(),
        )

        for step_in_epoch, batch in enumerate(train_dataloader):
            if epoch == start_epoch and step_in_epoch < steps_to_skip:
                continue

            batch = prepare_batch(batch, config, device=device)
            accum_step = (global_step % grad_accum_steps)
            do_opt = (accum_step == grad_accum_steps - 1) or (step_in_epoch == steps_per_epoch - 1)

            # bf16 autocast on CUDA
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp else torch.autocast(device_type="cpu", enabled=False)
            )
            with amp_ctx:
                metrics = train_step(
                    model=model,
                    encoder=encoder,
                    optimizer=optimizer,
                    batch=batch,
                    config=config,
                    scaler=scaler,
                    grad_accum_step=accum_step,
                    do_optimizer_step=do_opt,
                )

            if do_opt:
                scheduler.step()
                # EMA update
                state.update_ema(config.ema_decay1)
                state.step = global_step + 1

            train_metrics_accum.append(metrics)
            global_step += 1
            epoch_pbar.update(1)

            # --- Logging ---
            if global_step % config.log_freq == 0 and _is_rank0():
                avg_loss = sum(m["loss"] for m in train_metrics_accum) / len(train_metrics_accum)
                avg_l2 = sum(m["l2_loss"] for m in train_metrics_accum) / len(train_metrics_accum)
                avg_ce = sum(m["ce_loss"] for m in train_metrics_accum) / len(train_metrics_accum)
                now = time.time()
                sps = (global_step - last_log_step) / max(now - last_log_time, 1e-8)
                current_lr = optimizer.param_groups[0]["lr"]
                postfix = {
                    "step": global_step, "loss": f"{avg_loss:.4f}",
                    "l2": f"{avg_l2:.4f}", "ce": f"{avg_ce:.4f}",
                    "sps": f"{sps:.1f}", "lr": f"{current_lr:.2e}",
                }
                epoch_pbar.set_postfix(**{k: str(v) for k, v in postfix.items()})
                tqdm.write(
                    f"INFO - Step {global_step}: loss={avg_loss:.4f}, "
                    f"l2={avg_l2:.4f}, ce={avg_ce:.4f}, lr={current_lr:.2e}, sps={sps:.2f}"
                )
                if config.use_wandb:
                    wandb.log({
                        "train_loss": avg_loss, "train_l2_loss": avg_l2,
                        "train_ce_loss": avg_ce, "lr": current_lr,
                        "epoch": epoch + (step_in_epoch + 1) / steps_per_epoch,
                        "step": global_step,
                    }, step=global_step)
                train_metrics_accum = []
                last_log_step = global_step
                last_log_time = now

            # --- Intra-epoch checkpoint ---
            if 0 < config.save_freq < 1:
                progress = epoch + (step_in_epoch + 1) / steps_per_epoch
                if progress - last_save_epoch >= config.save_freq:
                    state.epoch = epoch
                    save_checkpoint(state, config.output_dir, global_step, config.hf_repo_id)
                    log_for_0(f"Checkpoint at epoch {progress:.2f} (step {global_step})")
                    last_save_epoch = progress

        epoch_pbar.close()
        current_epoch = epoch + 1
        state.epoch = current_epoch

        # --- Epoch checkpoint ---
        if config.save_freq >= 1 and current_epoch % int(config.save_freq) == 0:
            save_checkpoint(state, config.output_dir, global_step, config.hf_repo_id)
            log_for_0(f"Checkpoint at epoch {current_epoch} (step {global_step})")

        # --- Evaluation / generation ---
        if config.eval_freq >= 1 and current_epoch % config.eval_freq == 0:
            model.eval()
            run_generation(
                model=raw_model,
                ema_params=state.ema_params1,
                encoder=encoder,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                config=config,
                device=device,
                batch_size=local_batch_size,
            )
            model.train()

    # --- Final save ---
    log_for_0("\n" + "=" * 60)
    log_for_0("Final checkpoint")
    log_for_0("=" * 60)
    save_checkpoint(state, config.output_dir, global_step, config.hf_repo_id)
    log_for_0(f"Final checkpoint saved to {config.output_dir}")
    if config.use_wandb and _is_rank0():
        wandb.finish()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main():
    args = parse_args()
    config = load_config_from_yaml(args.config)
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)
        log_for_0(f"Applied {len(args.config_override)} config override(s)")
    run_training(config)


if __name__ == "__main__":
    main()
