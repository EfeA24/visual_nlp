"""Pre-LN Transformer decoder. Does not project to a vocabulary."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from visual_nlp.transformer.attention import MultiHeadAttention
from visual_nlp.transformer.feed_forward import FeedForward
from visual_nlp.transformer.masks import decoder_self_mask, encoder_memory_mask
from visual_nlp.transformer.normalization import LayerNorm


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm3 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout, activation)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        cache: tuple[tuple[torch.Tensor, torch.Tensor] | None, tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple]:
        past_self = cache[0] if cache else None
        past_mem = cache[1] if cache else None
        h = self.norm1(x)
        self_out = self.self_attn(h, h, h, mask=tgt_mask, past_kv=past_self, use_cache=use_cache)
        if use_cache:
            self_out, self_kv = self_out
        else:
            self_kv = None
        x = x + self.dropout(self_out)
        h = self.norm2(x)
        cross_out = self.cross_attn(
            h,
            memory,
            memory,
            mask=memory_mask,
            past_kv=past_mem,
            use_cache=use_cache,
            reuse_past_kv=past_mem is not None,
        )
        if use_cache:
            cross_out, mem_kv = cross_out
        else:
            mem_kv = None
        x = x + self.dropout(cross_out)
        x = x + self.dropout(self.ff(self.norm3(x)))
        if use_cache:
            return x, (self_kv, mem_kv)
        return x


class TransformerDecoder(nn.Module):
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
        self.n_heads = n_heads
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff, dropout, activation) for _ in range(n_layers)]
        )
        self.norm = LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        cache: list | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        if tgt_mask is None and tgt_padding_mask is not None:
            tgt_mask = decoder_self_mask(tgt_padding_mask, self.n_heads)
        elif tgt_mask is None:
            pad = torch.zeros(tgt.size(0), tgt.size(1), dtype=torch.bool, device=tgt.device)
            tgt_mask = decoder_self_mask(pad, self.n_heads)
        if memory_mask is None and memory_padding_mask is not None:
            memory_mask = encoder_memory_mask(memory_padding_mask, tgt.size(1), self.n_heads)
        x = tgt
        new_cache = []
        for idx, layer in enumerate(self.layers):
            layer_cache = cache[idx] if cache is not None else None
            if use_cache:
                x, layer_state = layer(
                    x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, cache=layer_cache, use_cache=True
                )
                new_cache.append(layer_state)
            else:
                x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        x = self.norm(x)
        if use_cache:
            return x, new_cache
        return x
