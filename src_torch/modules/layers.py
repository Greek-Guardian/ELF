"""PyTorch model layers for ELF.

Translated from the original JAX/Flax implementation in src/modules/layers.py.
Numerical behaviour is kept identical:
  - Dense kernels: xavier_uniform; biases: 0
  - TimestepEmbedder MLPs and learned tokens: normal(0.02)
  - final_layer.linear: 0 (zero init)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ---------------------------------------------------------------------------
# Rotation helper for RoPE
# ---------------------------------------------------------------------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input (matches JAX version)."""
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)          # each (..., d)
    x = torch.stack((-x2, x1), dim=-1)  # (..., d, 2)
    return rearrange(x, '... d r -> ... (d r)')


# ---------------------------------------------------------------------------
# TextRotaryEmbeddingFast  (1D RoPE, no learnable params)
# ---------------------------------------------------------------------------

class TextRotaryEmbeddingFast(nn.Module):
    """1D Rotary Position Embedding for text sequence models.

    Matches the original JAX TextRotaryEmbeddingFast exactly.
    No learnable parameters — RoPE frequencies are computed on the fly
    and applied to the input tensor directly.
    """

    def __init__(
        self,
        dim: int,
        pt_seq_len: int = 512,
        ft_seq_len: Optional[int] = None,
        theta: float = 10000.0,
        num_empty_token: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.pt_seq_len = pt_seq_len
        self.ft_seq_len = ft_seq_len if ft_seq_len is not None else pt_seq_len
        self.theta = theta
        self.num_empty_token = num_empty_token

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to t.

        t: (..., seq_len, dim)  where the leading dims can be (B, heads, ...).
        Returns same shape as t.
        """
        dim = self.dim
        freqs = 1.0 / (
            self.theta ** (
                torch.arange(0, dim, 2, dtype=torch.float32, device=t.device)[:dim // 2] / dim
            )
        )

        pos = (
            torch.arange(self.ft_seq_len, dtype=torch.float32, device=t.device)
            / self.ft_seq_len * self.pt_seq_len
        )

        # Outer product: (ft_seq_len, dim//2)
        freqs_main = torch.einsum('..., f -> ... f', pos, freqs)
        freqs_main = repeat(freqs_main, '... n -> ... (n r)', r=2)  # (ft_seq_len, dim)

        D = freqs_main.shape[-1]
        cos_parts, sin_parts = [], []

        if self.num_empty_token > 0:
            cos_parts.append(torch.ones(self.num_empty_token, D, dtype=freqs.dtype, device=t.device))
            sin_parts.append(torch.zeros(self.num_empty_token, D, dtype=freqs.dtype, device=t.device))

        cos_parts.append(torch.cos(freqs_main))
        sin_parts.append(torch.sin(freqs_main))

        freqs_cos = torch.cat(cos_parts, dim=0)  # (total_len, dim)
        freqs_sin = torch.cat(sin_parts, dim=0)

        return t * freqs_cos + rotate_half(t) * freqs_sin


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMS Normalization (matches JAX version, preserves input dtype)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)


# ---------------------------------------------------------------------------
# BottleneckTextProj
# ---------------------------------------------------------------------------

class BottleneckTextProj(nn.Module):
    """Text projection with bottleneck: text_encoder_dim -> bottleneck -> hidden_size."""

    def __init__(self, text_encoder_dim: int, hidden_size: int, bottleneck_dim: int):
        super().__init__()
        self.proj1 = nn.Linear(text_encoder_dim, bottleneck_dim, bias=False)
        self.proj2 = nn.Linear(bottleneck_dim, hidden_size, bias=True)
        nn.init.xavier_uniform_(self.proj1.weight)
        nn.init.xavier_uniform_(self.proj2.weight)
        nn.init.zeros_(self.proj2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.proj1(x))


# ---------------------------------------------------------------------------
# TimestepEmbedder
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp_0 = nn.Linear(frequency_embedding_size, hidden_size)
        self.mlp_2 = nn.Linear(hidden_size, hidden_size)
        # normal(0.02) init to match JAX
        for layer in (self.mlp_0, self.mlp_2):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Sinusoidal timestep embeddings: (N,) -> (N, dim)."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp_2(F.silu(self.mlp_0(t_emb)))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head self-attention with optional QK-norm and RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.attn_drop = attn_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_layer = nn.Dropout(proj_drop)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        nn.init.xavier_uniform_(self.qkv.weight)
        if qkv_bias:
            nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        rope_fn: Optional[TextRotaryEmbeddingFast],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, N, C)
        attention_mask: optional float mask (B, N) or (B, N, N); 1=valid, 0=masked
        Returns: (B, N, C)
        """
        B, N, C = x.shape
        # QKV projection → (B, N, 3, num_heads, head_dim) → 3 × (B, heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)              # each (B, heads, N, head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope_fn is not None:
            q = rope_fn(q)
            k = rope_fn(k)

        # Build additive float mask for F.scaled_dot_product_attention
        # Original: 1=valid, 0=masked → convert to 0 / -1e9 additive float mask
        sdpa_mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:       # (B, N) → (B, 1, 1, N)
                sdpa_mask = attention_mask[:, None, None, :].float()
            elif attention_mask.ndim == 3:     # (B, N, N) → (B, 1, N, N)
                sdpa_mask = attention_mask[:, None, :, :].float()
            else:
                sdpa_mask = attention_mask.float()
            # 1→0, 0→-1e9
            sdpa_mask = (1.0 - sdpa_mask) * -1e9

        dropout_p = self.attn_drop if self.training else 0.0
        # F.scaled_dot_product_attention handles FlashAttention-2 automatically on sm_90
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask, dropout_p=dropout_p)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop_layer(x)


# ---------------------------------------------------------------------------
# SwiGLUFFN
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True):
        super().__init__()
        # Mirror JAX: hidden_dim passed in is mlp_ratio * dim; then 2/3 of that
        actual_hidden = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * actual_hidden, bias=bias)
        self.w3 = nn.Linear(actual_hidden, dim, bias=bias)
        self.drop_layer = nn.Dropout(drop)

        nn.init.xavier_uniform_(self.w12.weight)
        nn.init.xavier_uniform_(self.w3.weight)
        if bias:
            nn.init.zeros_(self.w12.bias)
            nn.init.zeros_(self.w3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = self.drop_layer(F.silu(x1) * x2)
        return self.w3(hidden)


# ---------------------------------------------------------------------------
# FinalLayer
# ---------------------------------------------------------------------------

class FinalLayer(nn.Module):
    """The final layer of ELF: RMSNorm → Linear (zero-initialized)."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        # Zero init (matches JAX ZERO_INIT)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm_final(x))
