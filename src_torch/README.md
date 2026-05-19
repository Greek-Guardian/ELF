# ELF PyTorch Implementation

This directory contains a pure PyTorch re-implementation of ELF (Embedded Language Flows), 
targeting H800 (sm_90, CUDA 12) multi-GPU nodes.

The original JAX/Flax implementation remains untouched in `../src/`.

## Key Differences from JAX Version

| Aspect | JAX (`src/`) | PyTorch (`src_torch/`) |
|---|---|---|
| Framework | JAX + Flax + optax | PyTorch |
| Multi-GPU | `jax.pmap` | `torch.nn.parallel.DistributedDataParallel` |
| T5 Encoder | Hand-written JAX port (`.pkl` weights) | HuggingFace `T5EncoderModel` |
| Checkpoints | Flax `serialization` + orbax | `torch.save` / `torch.load` |
| Attention | Manual SDPA | `F.scaled_dot_product_attention` (FlashAttn-2) |
| RNG | Functional (key split) | Global state (`torch.manual_seed`) |
| Optimizer | optax (Muon / AdamW) | Custom Muon + `torch.optim.AdamW` |

## Quick Start (H800, 8 GPUs)

```bash
cd src_torch
pip install -r requirements.txt

# Single-node 8xH800
torchrun --nproc_per_node=8 train.py \
    --config configs/training_configs/train_owt_ELF-B_h800_torch.yml
```

## Development (macOS, CPU)

```bash
cd src_torch
pip install -r requirements.txt

# Single-device (CPU) smoke test
python train.py --config configs/training_configs/train_owt_ELF-B_h800_torch.yml \
    --config_override global_batch_size=2 --config_override epochs=1
```

## File Structure

```
src_torch/
├── modules/
│   ├── layers.py          # RMSNorm, RoPE, Attention, SwiGLUFFN, FinalLayer, ...
│   ├── model.py           # ELFBlock, ELF transformer, factory functions
│   └── t5_encoder.py      # HuggingFace T5 frozen encoder wrapper
├── utils/
│   ├── logging_utils.py   # rank-0 logging helpers
│   ├── encoder_utils.py   # encode_text + mask building
│   ├── sampling_utils.py  # noise / timestep / flow-matching helpers
│   ├── data_utils.py      # DataLoader, collate, dataset loading
│   ├── train_utils.py     # TrainState, optimizer, LR schedule
│   ├── checkpoint_utils.py# save/load checkpoints (local + HF Hub)
│   ├── generation_utils.py# generation helpers (ODE/SDE steps, decode)
│   └── metrics_utils.py   # PPL, BLEU, ROUGE
├── optimizers/
│   └── muon.py            # PyTorch Muon optimizer
├── configs/
│   ├── config.py          # Config / SamplingConfig dataclasses
│   ├── sampling_configs/  # (copied from src/)
│   └── training_configs/  # (copied from src/, + torch-specific)
├── train_step.py          # single training step function
├── train.py               # main training entry point (DDP)
├── generation.py          # generation / evaluation runner
├── eval.py                # evaluation entry point
└── requirements.txt
```
