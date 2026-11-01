"""Autoregressive character decoder. Training emits logits, not softmax."""

from __future__ import annotations

import torch
import torch.nn as nn

from visual_nlp.transformer.decoder import TransformerDecoder
from visual_nlp.transformer.positional import SinusoidalPositionalEncoding


class CharacterDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_len: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.decoder = TransformerDecoder(d_model, n_heads, n_layers, d_ff, dropout, activation)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_padding_mask: torch.Tensor | None = None,
        memory_padding_mask: torch.Tensor | None = None,
        start_pos: int = 0,
        cache: list | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        if tgt_padding_mask is None:
            tgt_padding_mask = token_ids.eq(self.pad_id)
        embeddings = self.pos(self.token_embed(token_ids), start_pos=start_pos)
        hidden = self.decoder(
            embeddings,
            memory,
            tgt_padding_mask=tgt_padding_mask,
            memory_padding_mask=memory_padding_mask,
            cache=cache,
            use_cache=use_cache,
        )
        if use_cache:
            hidden, new_cache = hidden
            return self.proj(hidden), new_cache
        return self.proj(hidden)
