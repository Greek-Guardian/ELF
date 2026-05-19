"""Training utilities: TrainState, optimizer, LR schedule (PyTorch).

Translated from src/utils/train_utils.py (JAX/optax/flax.train_state).
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from utils.logging_utils import log_for_0


# ---------------------------------------------------------------------------
# TrainState
# ---------------------------------------------------------------------------

@dataclass
class TrainState:
    """Minimal mutable training state (mirrors JAX TrainState fields)."""
    model: nn.Module
    optimizer: torch.optim.Optimizer
    step: int = 0
    epoch: int = 0
    ema_params1: Optional[Dict[str, torch.Tensor]] = field(default=None, repr=False)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "params": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "ema_params1": (
                {k: v.cpu() for k, v in self.ema_params1.items()}
                if self.ema_params1 is not None else None
            ),
            "opt_state": self.optimizer.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
        }

    def load_state_dict(self, ckpt: Dict[str, Any], device=None):
        """Load from a state dict saved by `state_dict()`."""
        map_location = device if device is not None else "cpu"
        self.model.load_state_dict(
            {k: v.to(map_location) for k, v in ckpt["params"].items()}
        )
        if ckpt.get("ema_params1") is not None:
            self.ema_params1 = {k: v.to(map_location) for k, v in ckpt["ema_params1"].items()}
        self.optimizer.load_state_dict(ckpt["opt_state"])
        self.step = int(ckpt["step"])
        self.epoch = int(ckpt["epoch"])

    @property
    def ema_state_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        """Current EMA params as a state_dict (for inference)."""
        return self.ema_params1

    def update_ema(self, decay: float):
        """Update EMA params in-place: ema = decay * ema + (1-decay) * params."""
        if self.ema_params1 is None:
            self.ema_params1 = copy.deepcopy(
                {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            )
            return
        with torch.no_grad():
            for k, p in self.model.state_dict().items():
                if k in self.ema_params1:
                    self.ema_params1[k].mul_(decay).add_(p.detach().cpu() * (1.0 - decay))


# ---------------------------------------------------------------------------
# Optimizer builder
# ---------------------------------------------------------------------------

def get_optimizer(config, model: nn.Module) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    if config.optimizer == "muon":
        # Import the local Muon implementation
        from optimizers.muon import Muon
        log_for_0("Using Muon optimizer")
        return Muon(model.parameters(), lr=config.lr or 1e-4)
    elif config.optimizer == "adamw":
        log_for_0("Using AdamW optimizer")
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr or 1e-4,
            weight_decay=config.weight_decay,
            betas=(config.adam_b1, config.adam_b2),
        )
    else:
        raise ValueError(
            f"Unknown optimizer: {config.optimizer!r}. Choose 'adamw' or 'muon'."
        )


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def create_learning_rate_fn(
    num_train_steps: int,
    num_warmup_steps: int,
    learning_rate: float,
    schedule: str = "constant",
    min_lr: float = 0.0,
):
    """Return a LambdaLR-compatible lambda (step → lr multiplier).

    Usage:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    """
    def lr_lambda(current_step: int) -> float:
        # Linear warmup
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)

        if schedule == "cosine":
            import math
            progress = float(current_step - num_warmup_steps) / max(
                1, num_train_steps - num_warmup_steps
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Scale so that at progress=1 we reach min_lr / learning_rate
            alpha = min_lr / learning_rate if learning_rate > 0 else 0.0
            return alpha + (1.0 - alpha) * cosine_decay

        # Constant
        return 1.0

    return lr_lambda
