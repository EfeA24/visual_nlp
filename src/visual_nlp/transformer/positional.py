"""1D and 2D positional encodings. Layout/box embeddings live in nlp/embeddings.py."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_table(max_len: int, d_model: int, device: torch.device | None = None) -> torch.Tensor:
    if d_model % 2 != 0:
        raise ValueError("Sinusoidal encodings require an even d_model.")
    position = torch.arange(max_len, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    table = torch.zeros(max_len, d_model, device=device)
    table[:, 0::2] = torch.sin(position * div)
    table[:, 1::2] = torch.cos(position * div)
    return table


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("table", sinusoidal_table(max_len, d_model), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        length = x.size(1)
        end = start_pos + length
        if end > self.max_len:
            raise ValueError(f"Sequence end {end} exceeds max_len {self.max_len}.")
        return self.dropout(x + self.table[start_pos:end].unsqueeze(0))


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.max_len = max_len
        self.embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        if length > self.max_len:
            raise ValueError(f"Sequence length {length} exceeds max_len {self.max_len}.")
        positions = torch.arange(length, device=x.device)
        return self.dropout(x + self.embed(positions).unsqueeze(0))


class Sinusoidal2DPositionalEncoding(nn.Module):
    """Encode a patch grid by row and column. Half of d_model is used per axis."""

    def __init__(self, d_model: int, max_h: int = 128, max_w: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("2D sinusoidal encodings require an even d_model.")
        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w
        self.dropout = nn.Dropout(dropout)
        axis_dim = d_model // 2
        self.register_buffer("row_table", sinusoidal_table(max_h, axis_dim), persistent=False)
        self.register_buffer("col_table", sinusoidal_table(max_w, axis_dim), persistent=False)

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        if grid_h > self.max_h or grid_w > self.max_w:
            raise ValueError(f"Grid {grid_h}x{grid_w} exceeds max {self.max_h}x{self.max_w}.")
        rows = self.row_table[:grid_h].unsqueeze(1).expand(grid_h, grid_w, -1)
        cols = self.col_table[:grid_w].unsqueeze(0).expand(grid_h, grid_w, -1)
        pos = torch.cat([rows, cols], dim=-1).reshape(1, grid_h * grid_w, self.d_model)
        return self.dropout(x + pos.to(dtype=x.dtype))
