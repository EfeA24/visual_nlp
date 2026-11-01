"""Non-overlapping patch embedding without a CNN."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PatchGrid:
    grid_h: int
    grid_w: int
    pad_h: int
    pad_w: int
    padding_mask: torch.Tensor


def pad_to_patch(
    images: torch.Tensor,
    patch_h: int,
    patch_w: int,
    value: float = 0.0,
) -> tuple[torch.Tensor, int, int]:
    _, _, height, width = images.shape
    pad_h = (patch_h - height % patch_h) % patch_h
    pad_w = (patch_w - width % patch_w) % patch_w
    if pad_h or pad_w:
        images = F.pad(images, (0, pad_w, 0, pad_h), value=value)
    return images, pad_h, pad_w


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_h: int, patch_w: int, d_model: int) -> None:
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.in_channels = in_channels
        self.proj = nn.Linear(in_channels * patch_h * patch_w, d_model)

    def _patch_padding_mask(
        self,
        batch: int,
        grid_h: int,
        grid_w: int,
        valid_hw: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        if valid_hw is None:
            return torch.zeros(batch, grid_h * grid_w, dtype=torch.bool, device=device)
        rows = torch.arange(grid_h, device=device).view(1, grid_h, 1)
        cols = torch.arange(grid_w, device=device).view(1, 1, grid_w)
        top = rows * self.patch_h
        left = cols * self.patch_w
        valid_h = valid_hw[:, 0].view(batch, 1, 1)
        valid_w = valid_hw[:, 1].view(batch, 1, 1)
        overlaps = (top < valid_h) & (left < valid_w)
        return (~overlaps).reshape(batch, grid_h * grid_w)

    def forward(
        self,
        images: torch.Tensor,
        valid_hw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PatchGrid]:
        images, pad_h, pad_w = pad_to_patch(images, self.patch_h, self.patch_w)
        batch, channels, height, width = images.shape
        grid_h = height // self.patch_h
        grid_w = width // self.patch_w
        patches = images.unfold(2, self.patch_h, self.patch_h).unfold(3, self.patch_w, self.patch_w)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(batch, grid_h * grid_w, channels * self.patch_h * self.patch_w)
        embeddings = self.proj(patches)
        padding_mask = self._patch_padding_mask(batch, grid_h, grid_w, valid_hw, images.device)
        grid = PatchGrid(grid_h=grid_h, grid_w=grid_w, pad_h=pad_h, pad_w=pad_w, padding_mask=padding_mask)
        return embeddings, grid
