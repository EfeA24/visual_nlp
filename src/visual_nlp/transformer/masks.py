"""Attention masks for padding and causal decoding.

Attention mask convention
-------------------------
``True`` means the location is **blocked** (the model must not attend).
``False`` means the location is visible.

Loss mask convention (see ``training.losses``)
----------------------------------------------
``True`` means the location **contributes** to the loss.
These two conventions are intentionally different because they serve
different purposes; do not reuse one as the other.
"""

from __future__ import annotations

from typing import Optional

import torch


def causal_mask(length: int, device: torch.device | None = None) -> torch.Tensor:
    """[T, T] upper-triangular mask. True = future token is blocked."""
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


def padding_mask(valid_lengths: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    """[B, T] mask from lengths. True = pad / blocked."""
    max_length = int(max_length or valid_lengths.max().item())
    positions = torch.arange(max_length, device=valid_lengths.device)
    return positions.unsqueeze(0) >= valid_lengths.unsqueeze(1)


def boolean_padding_mask(valid: torch.Tensor) -> torch.Tensor:
    """Convert a validity map (True = keep) into an attention mask (True = block)."""
    return ~valid.bool()


def expand_mask(
    mask: torch.Tensor,
    num_heads: int,
    query_length: int | None = None,
) -> torch.Tensor:
    """Broadcast a mask to [B, H, T_q, T_k]."""
    if mask.dim() == 2:
        # [T, T] causal or [B, T_k] padding
        if mask.size(0) == mask.size(1) and query_length is None:
            mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            mask = mask.unsqueeze(1).unsqueeze(1)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() != 4:
        raise ValueError(f"Unsupported mask rank: {mask.dim()}")
    if query_length is not None and mask.size(-2) == 1:
        mask = mask.expand(-1, -1, query_length, -1)
    if mask.size(1) == 1:
        mask = mask.expand(-1, num_heads, -1, -1)
    return mask


def combine_masks(*masks: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Logical OR of blocked positions. None arguments are ignored."""
    active = [mask.bool() for mask in masks if mask is not None]
    if not active:
        return None
    combined = active[0]
    for mask in active[1:]:
        combined = combined | mask
    return combined


def decoder_self_mask(
    tgt_padding: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Causal + target padding mask shaped [B, H, T, T]."""
    batch, length = tgt_padding.shape
    causal = expand_mask(causal_mask(length, device=tgt_padding.device), num_heads)
    pad = expand_mask(tgt_padding, num_heads, query_length=length)
    return combine_masks(causal.expand(batch, -1, -1, -1), pad)


def encoder_memory_mask(
    memory_padding: torch.Tensor,
    query_length: int,
    num_heads: int,
) -> torch.Tensor:
    """Encoder padding seen by decoder queries, shape [B, H, T_q, T_k]."""
    return expand_mask(memory_padding, num_heads, query_length=query_length)
