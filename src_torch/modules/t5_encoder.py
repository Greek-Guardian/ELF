"""HuggingFace T5 encoder wrapper for ELF (PyTorch).

Replaces the hand-written JAX T5 encoder in src/modules/t5_encoder.py.
Uses `transformers.T5EncoderModel` with frozen weights.
The interface mirrors the original: `get_encoder(model_name)` returns
(config, model) where config has a `.d_model` attribute.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from utils.logging_utils import log_for_0


# ---------------------------------------------------------------------------
# Minimal config class (compatible with the rest of the codebase)
# ---------------------------------------------------------------------------

@dataclass
class T5EncoderConfig:
    """Minimal config that mirrors the original JAX T5EncoderConfig."""
    vocab_size: int
    d_model: int
    d_kv: int
    d_ff: int
    num_layers: int
    num_heads: int
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    is_gated_act: bool = True
    model_name: str = "t5-small"


# ---------------------------------------------------------------------------
# Encoder wrapper
# ---------------------------------------------------------------------------

class T5Encoder(nn.Module):
    """Frozen T5 encoder used as a text embedder.

    Weights are loaded from HuggingFace Hub (or a local cache) and
    immediately frozen — no gradients flow through this module.
    """

    def __init__(self, model_name: str = "t5-small"):
        super().__init__()
        from transformers import T5EncoderModel  # deferred import

        log_for_0(f"Loading T5EncoderModel from HuggingFace: {model_name}...")
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.encoder.requires_grad_(False)   # frozen
        self.encoder.eval()
        log_for_0(f"T5 encoder loaded and frozen ({model_name})")

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return last hidden state: (B, S, d_model).

        attention_mask:
          - (B, S)     → standard 1D mask (1=valid, 0=pad)
          - (B, S, S)  → 2D self-attention mask (e.g., for conditional masking)
            In 2D case it is converted to a 1D mask (any non-zero row → valid)
            before being passed to the HF model, because HF T5 only accepts 1D masks.
        """
        if attention_mask is not None and attention_mask.ndim == 3:
            # (B, S, S) → (B, S): a token is "present" if it attends to anything
            attention_mask_1d = (attention_mask.sum(dim=-1) > 0).long()
        elif attention_mask is not None:
            attention_mask_1d = attention_mask.long()
        else:
            attention_mask_1d = None

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask_1d,
        )
        return outputs.last_hidden_state


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_T5_CONFIGS = {
    "t5-small": T5EncoderConfig(
        vocab_size=32128, d_model=512, d_kv=64, d_ff=2048,
        num_layers=6, num_heads=8, is_gated_act=False,
        model_name="t5-small",
    ),
    "t5-base": T5EncoderConfig(
        vocab_size=32128, d_model=768, d_kv=64, d_ff=3072,
        num_layers=12, num_heads=12, is_gated_act=False,
        model_name="t5-base",
    ),
    "t5-large": T5EncoderConfig(
        vocab_size=32128, d_model=1024, d_kv=64, d_ff=4096,
        num_layers=24, num_heads=16, is_gated_act=False,
        model_name="t5-large",
    ),
}


def get_encoder(model_name: str, _dtype=None):
    """Return (config, model) for the requested T5 variant.

    The `_dtype` argument is accepted for API compatibility with the JAX
    version but is ignored here (dtype is controlled via autocast / model.half()).
    """
    if model_name not in _T5_CONFIGS:
        log_for_0(
            f"Warning: unknown T5 model '{model_name}', defaulting to t5-small config."
        )
        config = _T5_CONFIGS["t5-small"]
    else:
        config = _T5_CONFIGS[model_name]

    model = T5Encoder(model_name=config.model_name)
    return config, model
