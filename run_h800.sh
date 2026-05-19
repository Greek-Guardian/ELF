#!/usr/bin/env bash
# Launch ELF training/eval on a single 8x H800 node.
#
# Single-host JAX uses one Python process that owns all 8 GPUs and pmaps over
# them — there is no need to spawn 8 ranks. `jax.distributed.initialize()` in
# train.py / eval.py raises on single-host and is caught.
#
# Usage:
#   ./run_h800.sh train [extra args ...]
#   ./run_h800.sh eval  [extra args ...]
#
# Examples:
#   ./run_h800.sh train --config configs/training_configs/train_owt_ELF-B_h800.yml
#   ./run_h800.sh eval  --config configs/training_configs/train_owt_ELF-B_h800.yml \
#                       --checkpoint_path ../assets_download/checkpoints/ELF-B-owt

set -euo pipefail

cd "$(dirname "$0")/src"

# --- Single-host: skip JAX cluster auto-detection ----------------------------
# JAX 0.4.38 sees KUBERNETES_SERVICE_HOST in the env and tries to query the
# K8s API to find peer pods for multi-host training. On a single-node K8s
# container behind a corporate proxy this hangs (504) and crashes the script.
# Setting ELF_SINGLE_HOST=1 makes train.py / eval.py skip the call entirely.
export ELF_SINGLE_HOST=1

# --- GPU visibility -----------------------------------------------------------
# Override CUDA_VISIBLE_DEVICES from the calling shell to use a subset.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# --- JAX memory ---------------------------------------------------------------
# JAX preallocates 75% of each GPU by default. On 80GB H800s, 0.92 leaves ~6GB
# headroom for the T5 encoder, dataloader buffers, and PPL eval (gpt2-large).
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92

# --- XLA codegen / scheduler --------------------------------------------------
# Only flags that exist in jaxlib 0.4.38. XLA aborts with "Unknown flag in
# XLA_FLAGS" if you pass anything it doesn't recognize. The latency-hiding
# scheduler is the biggest win on multi-GPU pmap; the rest is best-effort.
# If you upgrade jaxlib, you can re-add:
#   --xla_gpu_enable_async_collectives=true
#   --xla_gpu_enable_highest_priority_async_stream=true
export XLA_FLAGS="\
--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_enable_triton_gemm=true \
--xla_gpu_triton_gemm_any=true"

# --- TF32 / bf16 matmul -------------------------------------------------------
# Matches what JAX uses internally on Hopper; explicit for reproducibility.
export NVIDIA_TF32_OVERRIDE=1

# --- TF / TFDS noise ----------------------------------------------------------
# `tensorflow` is in requirements.txt but only used transitively. Mute it.
export TF_CPP_MIN_LOG_LEVEL=3
export TF_FORCE_GPU_ALLOW_GROWTH=true

# --- Misc ---------------------------------------------------------------------
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cmd="${1:-train}"
shift || true

case "$cmd" in
    train) exec python train.py "$@" ;;
    eval)  exec python eval.py  "$@" ;;
    *)     echo "Usage: $0 {train|eval} [args...]" >&2; exit 2 ;;
esac
