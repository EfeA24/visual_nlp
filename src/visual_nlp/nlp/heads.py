"""Task heads on top of the shared NLP encoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)


class FieldHead(nn.Module):
    """BIO tags: 0=O, then B/I for each field type."""

    def __init__(self, d_model: int, num_fields: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, 1 + 2 * num_fields)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)


class RelationHead(nn.Module):
    def __init__(self, d_model: int, num_fields: int) -> None:
        super().__init__()
        self.field_queries = nn.Parameter(torch.randn(num_fields, d_model) * 0.02)
        self.left = nn.Linear(d_model, d_model)
        self.right = nn.Linear(d_model, d_model)
        self.score = nn.Linear(d_model, 1)

    def forward(self, hidden: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid = (~padding_mask).unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        states = pooled.unsqueeze(1) + self.field_queries.unsqueeze(0)
        left = self.left(states).unsqueeze(2)
        right = self.right(states).unsqueeze(1)
        return self.score(torch.tanh(left + right)).squeeze(-1)


class DocumentHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, num_classes)

    def forward(self, hidden: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid = (~padding_mask).unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.proj(pooled)
