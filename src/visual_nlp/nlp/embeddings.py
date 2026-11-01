"""Combine token, order, box, and optional page embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn

from visual_nlp.transformer.positional import SinusoidalPositionalEncoding


class DocumentEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_length: int,
        max_pages: int = 32,
        dropout: float = 0.1,
        use_layout: bool = True,
        use_page: bool = True,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.use_layout = use_layout
        self.use_page = use_page
        self.token = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_length, dropout=0.0)
        self.box_proj = nn.Linear(4, d_model)
        self.page = nn.Embedding(max_pages, d_model)
        self.layout_scale = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(dropout)
        self.default_box = torch.zeros(4)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_boxes: torch.Tensor | None = None,
        page_ids: torch.Tensor | None = None,
        use_layout: bool | None = None,
    ) -> torch.Tensor:
        hidden = self.pos(self.token(input_ids))
        layout_on = self.use_layout if use_layout is None else use_layout
        if layout_on:
            if token_boxes is None:
                boxes = hidden.new_zeros(*hidden.shape[:2], 4)
            else:
                boxes = token_boxes.to(hidden.dtype)
            hidden = hidden + self.layout_scale * self.box_proj(boxes)
            if self.use_page:
                if page_ids is None:
                    pages = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
                else:
                    pages = page_ids.clamp(min=0, max=self.page.num_embeddings - 1)
                hidden = hidden + self.page(pages)
        return self.dropout(hidden)
