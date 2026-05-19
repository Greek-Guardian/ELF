"""Rank-0 logging helpers (PyTorch DDP compatible)."""

import logging

logger = logging.getLogger(__name__)


def _is_rank0() -> bool:
    """Return True if this is the main process."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True   # single-process: always rank 0


def log_for_0(msg, level=logging.INFO):
    """Log only on rank 0."""
    if _is_rank0():
        logger.log(level, msg)
