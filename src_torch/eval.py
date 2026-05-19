#!/usr/bin/env python
"""Evaluation script for trained ELF models (PyTorch).

Translated from src/eval.py (JAX).

Usage:
    # Multi-GPU
    torchrun --nproc_per_node=8 eval.py \
        --config configs/training_configs/train_owt_ELF-B_h800_torch.yml \
        --checkpoint_path /path/to/output_dir

    # Single GPU / CPU
    python eval.py \
        --config configs/training_configs/train_owt_ELF-B_h800_torch.yml \
        --checkpoint_path /path/to/checkpoint_10000.pt
"""

import argparse
import copy
import logging
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from modules.model import ELF_models
from modules.t5_encoder import get_encoder
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import load_checkpoint
from utils.train_utils import TrainState, get_optimizer
from utils.data_utils import load_jsonl_dataset, load_dataset_split, get_pad_token_id
from generation import test_generation_uncond, test_generation_cond
from configs.config import load_config_from_yaml, apply_config_overrides, load_sampling_configs


logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO, force=True,
)


def _is_rank0():
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained ELF model (PyTorch).")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--config_override", action="append", default=[])
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seeds (overrides --seed).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Optional DDP init
    local_rank = None
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = (
        torch.device(f"cuda:{local_rank}") if local_rank is not None
        else torch.device("cuda:0") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    log_for_0("Loading configuration...")
    config = load_config_from_yaml(args.config)
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)
        log_for_0(f"Applied {len(args.config_override)} override(s)")

    world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1

    if config.global_batch_size is not None:
        local_batch_size = config.global_batch_size // world_size
        config.batch_size = local_batch_size
    elif config.batch_size is not None:
        local_batch_size = config.batch_size
        config.global_batch_size = config.batch_size * world_size
    else:
        raise ValueError("Either global_batch_size or batch_size must be specified")

    log_for_0(f"Model: {config.model}  Encoder: {config.encoder_model_name}")
    log_for_0(f"Max length: {config.max_length}  Num samples: {config.num_samples}")
    log_for_0(f"Checkpoint: {args.checkpoint_path}")
    log_for_0(f"Device: {device}  World size: {world_size}")

    seed_list = (
        [int(s.strip()) for s in args.seeds.split(",")]
        if args.seeds else [args.seed]
    )
    log_for_0(f"Seeds: {seed_list}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)

    # --- Eval dataset ---
    eval_dataset = None
    if config.eval_data_path is not None:
        log_for_0("Loading eval dataset...")
        if config.eval_data_path.endswith(".jsonl"):
            eval_dataset = load_jsonl_dataset(config.eval_data_path, tokenizer)
        else:
            eval_dataset = load_dataset_split(config.eval_data_path)
        log_for_0(f"Eval size: {len(eval_dataset)}")

    # --- Encoder ---
    log_for_0(f"Loading encoder: {config.encoder_model_name}...")
    encoder_config, encoder = get_encoder(config.encoder_model_name)
    encoder = encoder.to(device).eval()
    log_for_0(f"Encoder d_model: {encoder_config.d_model}")

    # --- ELF model ---
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = tokenizer.vocab_size

    log_for_0(f"Creating {config.model} model...")
    model = ELF_models[config.model](
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

    # Dummy optimizer (needed for TrainState.load_state_dict)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    state = TrainState(
        model=model,
        optimizer=optimizer,
        ema_params1=copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}),
    )

    # --- Sampling configs ---
    if config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)

    # --- Load checkpoint ---
    log_for_0(f"Loading checkpoint: {args.checkpoint_path}...")
    state, _ = load_checkpoint(args.checkpoint_path, state, device=device)
    log_for_0("Checkpoint loaded.")

    # --- Run generation for each seed ---
    original_output_dir = config.output_dir
    for seed_idx, seed_val in enumerate(seed_list):
        if len(seed_list) > 1:
            log_for_0(f"\n{'#' * 70}")
            log_for_0(f"Seed {seed_idx + 1}/{len(seed_list)}: {seed_val}")
        torch.manual_seed(seed_val)

        config.output_dir = (
            os.path.join(original_output_dir, f"seed_{seed_val}")
            if len(seed_list) > 1 else original_output_dir
        )

        for sc_idx, sc in enumerate(config.sampling_configs):
            if len(config.sampling_configs) > 1:
                log_for_0(f"\n--- Sampling config {sc_idx + 1}/{len(config.sampling_configs)} ---")
            common = dict(
                model=model,
                ema_params=state.ema_params1,
                tokenizer=tokenizer,
                config=config,
                sampling_config=sc,
                device=device,
                batch_size=local_batch_size,
                num_samples=config.num_samples,
            )
            if eval_dataset is None:
                test_generation_uncond(**common)
            else:
                test_generation_cond(**common, encoder=encoder, dataset=eval_dataset)

    config.output_dir = original_output_dir
    log_for_0("\nEvaluation complete!")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
