"""Encoder utilities for ELF (PyTorch).

Translated from src/utils/encoder_utils.py (JAX).
"""

import torch


@torch.no_grad()
def encode_text(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    encoder: "torch.nn.Module",
    latent_mean: float,
    latent_std: float,
) -> torch.Tensor:
    """Run the frozen T5 encoder and normalise the output.

    Args:
        input_ids:      (B, S) token ids
        attention_mask: (B, S) or (B, S, S) — passed straight to T5Encoder
        encoder:        frozen T5Encoder module
        latent_mean:    normalisation mean (scalar)
        latent_std:     normalisation std  (scalar)

    Returns:
        (B, S, d_model) normalised latents
    """
    latents = encoder(input_ids=input_ids, attention_mask=attention_mask)
    return (latents - latent_mean) / latent_std


def build_self_attn_cond_masks(
    is_cond: "numpy.ndarray | torch.Tensor",
    is_valid: "numpy.ndarray | torch.Tensor",
    xp=None,
):
    """Build self-attention conditioning masks from cond/valid token flags.

    Identical logic to the JAX version in src/utils/encoder_utils.py.

    Args:
        is_cond:  bool array (B, S) — True where token is conditioning prefix
        is_valid: bool array (B, S) — True where token is a real (non-pad) token
        xp:       array module to use.  Pass `numpy` for CPU pre-processing
                  inside the DataLoader collate_fn; pass `torch` otherwise.
                  Defaults to numpy.

    Returns:
        (encoder_attention_mask, attention_mask, cond_seq_mask)
        All are float32 arrays of shape (B, S, S), (B, S), (B, S) respectively.
    """
    import numpy as np
    if xp is None:
        xp = np

    encoder_attention_mask = (
        (is_cond[:, :, None] & is_cond[:, None, :]) |
        (~is_cond[:, :, None] & is_valid[:, None, :])
    ).astype(xp.float32)
    attention_mask = is_valid.astype(xp.float32)
    cond_seq_mask = is_cond.astype(xp.float32)
    return encoder_attention_mask, attention_mask, cond_seq_mask
