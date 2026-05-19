"""Checkpoint save / load utilities (PyTorch).

Translated from src/utils/checkpoint_utils.py (JAX/flax).
Format: plain torch.save / torch.load dict.
Also supports uploading to Hugging Face Hub (unchanged from JAX version).
"""

import logging
import os
import re
from typing import Any, Optional, Tuple

import torch

from utils.logging_utils import log_for_0


def _local_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


# ---------------------------------------------------------------------------
# HF Hub upload helper
# ---------------------------------------------------------------------------

def _is_rank0() -> bool:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True


def upload_output_dir_to_hf(output_dir: str, hf_repo_id: Optional[str], reason: str = "artifacts"):
    if not hf_repo_id or not _is_rank0():
        return
    folder_path = _local_path(output_dir)
    if not os.path.isdir(folder_path):
        log_for_0(
            f"HF upload skipped; output directory does not exist: {folder_path}",
            level=logging.WARNING,
        )
        return
    try:
        from huggingface_hub import HfApi
        repo_id = hf_repo_id.strip("/")
        api = HfApi()
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        log_for_0(f"Uploading {reason} to HF: {repo_id}")
        api.upload_folder(repo_id=repo_id, folder_path=folder_path, repo_type="model")
        log_for_0(f"Uploaded {reason} to HF: {repo_id}")
    except Exception as e:
        log_for_0(f"Failed to upload {reason} to HF: {e}", level=logging.WARNING)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(
    state: Any,
    output_dir: str,
    step: int,
    hf_repo_id: Optional[str] = None,
):
    """Save model checkpoint to `output_dir/checkpoint_<step>.pt` (rank 0 only)."""
    if not _is_rank0():
        return

    ckpt_dir = _local_path(output_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{step}.pt")

    log_for_0(f"Saving checkpoint to {ckpt_path}")
    torch.save(state.state_dict(), ckpt_path)
    log_for_0(f"Checkpoint written to {ckpt_path}")

    upload_output_dir_to_hf(output_dir, hf_repo_id, reason="checkpoint")


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _checkpoint_step(name: str) -> int:
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else -1


def find_all_checkpoints(ckpt_dir: str, prefix: str = "checkpoint_"):
    ckpt_dir = _local_path(ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        return []
    names = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith(prefix) and f.endswith(".pt")],
        key=_checkpoint_step,
    )
    return [os.path.join(ckpt_dir, n) for n in names]


def find_latest_checkpoint(ckpt_dir: str, prefix: str = "checkpoint_"):
    all_ckpts = find_all_checkpoints(ckpt_dir, prefix)
    return all_ckpts[-1] if all_ckpts else None


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _split_hf_path(path: str, min_parts: int) -> Optional[Tuple[str, str]]:
    if "://" in path:
        return None
    if path.startswith(("/", ".", "~")):
        return None
    if os.path.exists(_local_path(path)):
        return None
    parts = path.split("/")
    if len(parts) < min_parts:
        return None
    repo_id = "/".join(parts[:2])
    sub_path = "/".join(parts[2:])
    return repo_id, sub_path


def _download_hf_checkpoint(checkpoint_path: str) -> Optional[str]:
    hf_path = _split_hf_path(checkpoint_path, min_parts=2)
    if hf_path is None:
        return None
    repo_id, sub_path = hf_path
    from huggingface_hub import snapshot_download
    log_for_0(
        f"Downloading checkpoint from HF: {repo_id}"
        + (f"/{sub_path}" if sub_path else "")
    )
    local_dir = snapshot_download(
        repo_id=repo_id, repo_type="model",
        allow_patterns=[f"{sub_path}/**"] if sub_path else None,
    )
    return os.path.join(local_dir, sub_path) if sub_path else local_dir


def load_checkpoint(
    checkpoint_path: str,
    state: Any,
    device=None,
) -> Tuple[Any, int]:
    """Load a PyTorch ELF checkpoint into `state` (in-place).

    Accepts:
      - local path to a .pt file
      - local directory (uses latest checkpoint inside)
      - HF repo id (e.g. 'embedded-language-flows/ELF-B-owt')
    """
    log_for_0(f"Loading ELF checkpoint from {checkpoint_path}...")
    errors = []

    def _try_local(path):
        local = _local_path(path)
        # Resolve directory → latest file
        if os.path.isdir(local):
            latest = find_latest_checkpoint(local)
            if latest:
                local = latest
        if os.path.isfile(local):
            ckpt = torch.load(local, map_location=device or "cpu")
            state.load_state_dict(ckpt, device=device)
            step = int(ckpt.get("step", 0))
            log_for_0(f"Loaded checkpoint from {local} (step {step})")
            return state, step
        return None

    result = None
    local = _local_path(checkpoint_path)
    if os.path.exists(local):
        try:
            result = _try_local(checkpoint_path)
        except Exception as e:
            errors.append(f"local: {e}")

    if result is None:
        try:
            hf_local = _download_hf_checkpoint(checkpoint_path)
            if hf_local:
                result = _try_local(hf_local)
        except Exception as e:
            errors.append(f"HF: {e}")

    if result is None and not os.path.exists(local):
        try:
            result = _try_local(checkpoint_path)
        except Exception as e:
            errors.append(f"local-fallback: {e}")

    if result is None:
        raise ValueError(
            f"Failed to load checkpoint from {checkpoint_path}. "
            f"Tried: {'; '.join(errors)}"
        )

    return result
