"""PyTorch ELF Transformer model.

Translated from src/modules/model.py (JAX/Flax).
Model structure and parameter layout are kept identical so that
JAX→PyTorch weight conversion is straightforward.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.layers import (
    Attention,
    BottleneckTextProj,
    FinalLayer,
    RMSNorm,
    SwiGLUFFN,
    TextRotaryEmbeddingFast,
    TimestepEmbedder,
)


# ---------------------------------------------------------------------------
# ELFBlock
# ---------------------------------------------------------------------------

class ELFBlock(nn.Module):
    """ELF Transformer block: pre-norm attention + pre-norm SwiGLU FFN."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        rope_fn: Optional[TextRotaryEmbeddingFast] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_fn=rope_fn, attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# ELF  (main transformer)
# ---------------------------------------------------------------------------

class ELF(nn.Module):
    """Text ELF Transformer.

    Params match the JAX version exactly, including:
    - Prefix-token time / self-cond-CFG / model-mode conditioning
    - Bottleneck text projection
    - Optional self-conditioning (concatenated along the feature axis)
    - Factored decoder unembedding (hidden → text_encoder_dim → vocab)
    """

    def __init__(
        self,
        text_encoder_dim: int,
        max_length: int,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bottleneck_dim: int = 128,
        num_time_tokens: int = 4,
        num_self_cond_cfg_tokens: int = 4,
        num_model_mode_tokens: int = 0,
        vocab_size: int = 0,
    ):
        super().__init__()
        self.text_encoder_dim = text_encoder_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_time_tokens = num_time_tokens
        self.num_self_cond_cfg_tokens = num_self_cond_cfg_tokens
        self.num_model_mode_tokens = num_model_mode_tokens
        self.vocab_size = vocab_size

        # --- Self-conditioning projection (only if self_cond_prob > 0) ---
        # Initialised unconditionally; we'll skip it at forward time if the input
        # is single-dim (non-self-cond). The layer is created so that weights exist
        # even when self_cond_prob > 0 is expected at some point.
        self.self_cond_proj = nn.Linear(2 * text_encoder_dim, text_encoder_dim)
        nn.init.xavier_uniform_(self.self_cond_proj.weight)
        nn.init.zeros_(self.self_cond_proj.bias)

        # --- Text projection ---
        self.text_proj = BottleneckTextProj(text_encoder_dim, hidden_size, bottleneck_dim)

        # --- Time conditioning ---
        if num_time_tokens <= 0:
            raise ValueError("num_time_tokens must be positive")
        self.t_embedder = TimestepEmbedder(hidden_size)
        # Learned tokens added on top of the time embedding (normal 0.02)
        self.t_emb_tokens = nn.Parameter(torch.randn(1, num_time_tokens, hidden_size) * 0.02)

        # --- Self-cond CFG conditioning ---
        if num_self_cond_cfg_tokens > 0:
            self.self_cond_cfg_embedder = TimestepEmbedder(hidden_size)
            self.self_cond_cfg_tokens = nn.Parameter(
                torch.randn(1, num_self_cond_cfg_tokens, hidden_size) * 0.02
            )

        # --- Model-mode tokens (optional) ---
        if num_model_mode_tokens > 0:
            self.mode_tokens = nn.Parameter(
                torch.randn(1, num_model_mode_tokens, hidden_size) * 0.02
            )

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            ELFBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4 <= i < depth // 4 * 3) else 0.0,
                proj_drop=proj_drop if (depth // 4 <= i < depth // 4 * 3) else 0.0,
            )
            for i in range(depth)
        ])

        # --- Final layer ---
        self.final_layer = FinalLayer(hidden_size, patch_size=1, out_channels=text_encoder_dim)

        # --- Factored decoder unembedding ---
        if vocab_size > 0:
            # proj_kernel: hidden_size → text_encoder_dim
            self.proj_kernel = nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(hidden_size, text_encoder_dim))
            )
            self.proj_bias = nn.Parameter(torch.zeros(text_encoder_dim))
            # unembed_kernel: text_encoder_dim → vocab_size
            self.unembed_kernel = nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(text_encoder_dim, vocab_size))
            )
            self.unembed_bias = nn.Parameter(torch.zeros(vocab_size))

    # ------------------------------------------------------------------
    # Context (prefix token) builder
    # ------------------------------------------------------------------

    def build_context(
        self,
        t: torch.Tensor,
        self_cond_cfg_scale: Optional[torch.Tensor],
    ) -> list:
        """Return list of prefix-token tensors, each shape (B, n_tokens, hidden_size)."""
        B = t.shape[0]
        prefix_tokens = []

        time_emb = self.t_embedder(t)                                     # (B, H)
        t_prefix = self.t_emb_tokens.expand(B, -1, -1) + time_emb.unsqueeze(1)
        prefix_tokens.append(t_prefix)

        if self_cond_cfg_scale is not None and self.num_self_cond_cfg_tokens > 0:
            sc_emb = self.self_cond_cfg_embedder(self_cond_cfg_scale)     # (B, H)
            sc_prefix = (
                self.self_cond_cfg_tokens.expand(B, -1, -1) + sc_emb.unsqueeze(1)
            )
            prefix_tokens.append(sc_prefix)

        return prefix_tokens

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        self_cond_cfg_scale: Optional[torch.Tensor] = None,
        decoder_step_active: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, S, C) or (B, S, 2C) when self-conditioning is active.
            t: (B,) timestep values in [0, 1].
            attention_mask: (B, S) float mask, 1=valid, 0=padding.
            self_cond_cfg_scale: (B,) optional CFG scale for self-conditioning.
            decoder_step_active: bool — if True run decoder head, if False skip it,
                if None never compute decoder logits.

        Returns:
            (output, decoder_logits)
            output: (B, S, text_encoder_dim)
            decoder_logits: (B, S, vocab_size) or None
        """
        B, S, _ = x.shape
        patch_size = 1
        head_dim = self.hidden_size // self.num_heads

        # --- Self-conditioning merge ---
        if x.shape[-1] == 2 * self.text_encoder_dim:
            x = self.self_cond_proj(x)

        # --- Text projection ---
        x = self.text_proj(x)    # (B, S, hidden_size)

        # --- Model-mode tokens (gated) ---
        model_mode_offset = 0
        if self.num_model_mode_tokens > 0:
            mode_tokens = self.mode_tokens.expand(B, -1, -1).clone()
            if decoder_step_active is None:
                gate = 0.0
            else:
                gate = 1.0 if decoder_step_active else 0.0
            mode_tokens = mode_tokens * gate
            x = torch.cat([mode_tokens, x], dim=1)
            model_mode_offset = self.num_model_mode_tokens
            if attention_mask is not None:
                mode_mask = torch.ones(B, self.num_model_mode_tokens,
                                       dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([mode_mask, attention_mask], dim=1)

        # --- Prefix time/cfg tokens ---
        context_prefix_tokens = self.build_context(t, self_cond_cfg_scale)
        prefix_len = 0
        if context_prefix_tokens:
            prefix = torch.cat(context_prefix_tokens, dim=1)   # (B, P, H)
            prefix_len = prefix.shape[1]
            x = torch.cat([prefix, x], dim=1)
            if attention_mask is not None:
                prefix_mask = torch.ones(B, prefix_len,
                                         dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # --- RoPE ---
        rope_fn = TextRotaryEmbeddingFast(
            dim=head_dim,
            pt_seq_len=self.max_length,
            num_empty_token=prefix_len + model_mode_offset,
        ).to(x.device)

        # --- Transformer blocks ---
        for block in self.blocks:
            x = block(x, rope_fn=rope_fn, attention_mask=attention_mask)

        # Strip prefix and mode tokens
        x = x[:, prefix_len + model_mode_offset:]   # back to (B, S, H)

        # --- Decoder unembedding (factored) ---
        decoder_logits = None
        if decoder_step_active is not None and self.vocab_size > 0:
            if decoder_step_active:
                hidden_proj = F.gelu(x @ self.proj_kernel + self.proj_bias)
                decoder_logits = hidden_proj @ self.unembed_kernel + self.unembed_bias
            else:
                decoder_logits = torch.zeros(B, S, self.vocab_size,
                                             dtype=x.dtype, device=x.device)

        # --- Final projection to text_encoder_dim ---
        output = self.final_layer(x)      # (B, S, text_encoder_dim)
        return output, decoder_logits


# ---------------------------------------------------------------------------
# Factory functions (match JAX ELF_B / ELF_M / ELF_L)
# ---------------------------------------------------------------------------

def ELF_B(**kwargs) -> ELF:
    return ELF(depth=12, hidden_size=768,  num_heads=12, **kwargs)

def ELF_M(**kwargs) -> ELF:
    return ELF(depth=24, hidden_size=1056, num_heads=16, **kwargs)

def ELF_L(**kwargs) -> ELF:
    return ELF(depth=32, hidden_size=1280, num_heads=16, **kwargs)


ELF_models = {
    'ELF-B': ELF_B,
    'ELF-M': ELF_M,
    'ELF-L': ELF_L,
}
