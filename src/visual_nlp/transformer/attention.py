"""Scaled dot-product and multi-head attention.

Attention mask convention: True means the location is blocked.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + M) V.

    Args:
        query: [B, H, T_q, d_k]
        key: [B, H, T_k, d_k]
        value: [B, H, T_k, d_v]
        mask: broadcastable to [B, H, T_q, T_k]; True = blocked.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask.bool(), torch.finfo(scores.dtype).min)
    weights = F.softmax(scores, dim=-1)
    if mask is not None:
        all_blocked = mask.bool().all(dim=-1, keepdim=True)
        weights = weights.masked_fill(all_blocked, 0.0)
    if dropout is not None:
        weights = dropout(weights)
    output = torch.matmul(weights, value)
    return output, weights


class MultiHeadAttention(nn.Module):
    """Self-attention or cross-attention depending on Q/K/V sources."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        tensor = tensor.view(batch, length, self.n_heads, self.d_head)
        return tensor.transpose(1, 2).contiguous()

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, _, length, _ = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch, length, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        reuse_past_kv: bool = False,
    ) -> torch.Tensor | tuple:
        """Args are [B, T, d_model]. Mask True = blocked."""
        q = self._split_heads(self.q_proj(query))
        if reuse_past_kv and past_kv is not None:
            k, v = past_kv
        else:
            k = self._split_heads(self.k_proj(key))
            v = self._split_heads(self.v_proj(value))
            if past_kv is not None:
                k = torch.cat([past_kv[0], k], dim=2)
                v = torch.cat([past_kv[1], v], dim=2)
        context, weights = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.attn_dropout)
        output = self.resid_dropout(self.out_proj(self._merge_heads(context)))
        if use_cache and return_weights:
            return output, weights, (k, v)
        if use_cache:
            return output, (k, v)
        if return_weights:
            return output, weights
        return output
