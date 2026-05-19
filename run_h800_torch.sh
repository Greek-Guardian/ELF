#!/usr/bin/env bash
# run_h800_torch.sh — Launch ELF PyTorch training on a single 8×H800 node.
#
# Usage:
#   bash run_h800_torch.sh [--config <yml>] [extra torchrun args]
#
# Environment variables (optional):
#   ELF_CONFIG     path to training YAML (default: src_torch/configs/training_configs/train_owt_ELF-B_h800_torch.yml)
#   ELF_NPROC      number of GPUs (default: 8)
#   ELF_MASTER_PORT NCCL master port (default: 29500)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src_torch"

ELF_CONFIG="${ELF_CONFIG:-${SRC_DIR}/configs/training_configs/train_owt_ELF-B_h800_torch.yml}"
ELF_NPROC="${ELF_NPROC:-8}"
ELF_MASTER_PORT="${ELF_MASTER_PORT:-29500}"

# -----------------------------------------------------------------------
# CUDA / NCCL performance flags (H800 / sm_90)
# -----------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Use TF32 for matrix multiplications (free precision/speed on Hopper)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# NCCL tuning for H800 NVLink
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN

# Enable Flash Attention kernel selection
export TORCH_SDPA_ENABLE_FLASH=1

# -----------------------------------------------------------------------
# Python path
# -----------------------------------------------------------------------
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

echo "======================================================="
echo " ELF PyTorch H800 Training"
echo " Config : ${ELF_CONFIG}"
echo " GPUs   : ${ELF_NPROC}"
echo " SRC    : ${SRC_DIR}"
echo "======================================================="

torchrun \
    --nproc_per_node="${ELF_NPROC}" \
    --master_port="${ELF_MASTER_PORT}" \
    "${SRC_DIR}/train.py" \
    --config "${ELF_CONFIG}" \
    "$@"
