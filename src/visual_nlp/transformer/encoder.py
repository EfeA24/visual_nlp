"""Pre-LN Transformer encoder. Operates on prepared embeddings only."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from visual_nlp.transformer.attention import MultiHeadAttention
from visual_nlp.transformer.feed_forward import FeedForward
from visual_nlp.transformer.masks import expand_mask
from visual_nlp.transformer.normalization import LayerNorm


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout, activation)
        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_mask = None
        if padding_mask is not None:
            attn_mask = expand_mask(padding_mask, self.n_heads, query_length=x.size(1))
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, mask=attn_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout, activation) for _ in range(n_layers)]
        )
        self.norm = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        return self.norm(x)
