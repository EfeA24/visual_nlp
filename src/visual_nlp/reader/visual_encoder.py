"""Image to contextual patch representations. Does not emit characters."""

from __future__ import annotations

import torch
import torch.nn as nn

from visual_nlp.reader.patch_embedding import PatchEmbedding, PatchGrid
from visual_nlp.transformer.encoder import TransformerEncoder
from visual_nlp.transformer.positional import Sinusoidal2DPositionalEncoding


class VisualEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_h: int,
        patch_w: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_h, patch_w, d_model)
        self.pos = Sinusoidal2DPositionalEncoding(d_model, dropout=dropout)
        self.encoder = TransformerEncoder(d_model, n_heads, n_layers, d_ff, dropout, activation)

    def forward(
        self,
        images: torch.Tensor,
        valid_hw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, PatchGrid]:
        patches, grid = self.patch_embed(images, valid_hw=valid_hw)
        encoded = self.encoder(self.pos(patches, grid.grid_h, grid.grid_w), padding_mask=grid.padding_mask)
        return encoded, grid.padding_mask, grid
