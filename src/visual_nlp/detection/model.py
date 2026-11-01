"""Query-based line detector. Queries answer where text is, not what it says."""

from __future__ import annotations

import torch
import torch.nn as nn

from visual_nlp.reader.visual_encoder import VisualEncoder
from visual_nlp.transformer.decoder import TransformerDecoder
from visual_nlp.utils.config import Config


class DetectorModel(nn.Module):
    def __init__(
        self,
        visual_encoder: VisualEncoder,
        decoder: TransformerDecoder,
        num_queries: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.visual_encoder = visual_encoder
        self.decoder = decoder
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.class_head = nn.Linear(d_model, 2)
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
            nn.Sigmoid(),
        )
        self.num_queries = num_queries

    @classmethod
    def from_config(cls, config: Config) -> "DetectorModel":
        in_channels = 3 if bool(config.input.get("color", True)) else 1
        encoder = VisualEncoder(
            in_channels=in_channels,
            patch_h=int(config.patch.height),
            patch_w=int(config.patch.width),
            d_model=int(config.encoder.d_model),
            n_heads=int(config.encoder.n_heads),
            n_layers=int(config.encoder.n_layers),
            d_ff=int(config.encoder.d_ff),
            dropout=float(config.encoder.get("dropout", 0.1)),
            activation=str(config.encoder.get("activation", "gelu")),
        )
        decoder = TransformerDecoder(
            d_model=int(config.decoder.d_model),
            n_heads=int(config.decoder.n_heads),
            n_layers=int(config.decoder.n_layers),
            d_ff=int(config.decoder.d_ff),
            dropout=float(config.decoder.get("dropout", 0.1)),
            activation=str(config.decoder.get("activation", "gelu")),
        )
        return cls(encoder, decoder, int(config.queries.num_queries), int(config.encoder.d_model))

    def forward(self, images: torch.Tensor, valid_hw: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        memory, memory_mask, _ = self.visual_encoder(images, valid_hw=valid_hw)
        batch = images.size(0)
        queries = self.query_embed.weight.unsqueeze(0).expand(batch, -1, -1)
        # Queries are unordered detections, not a causal token sequence.
        tgt_mask = torch.zeros(
            batch,
            self.decoder.n_heads,
            self.num_queries,
            self.num_queries,
            dtype=torch.bool,
            device=images.device,
        )
        hidden = self.decoder(
            queries,
            memory,
            memory_padding_mask=memory_mask,
            tgt_mask=tgt_mask,
        )
        return {"logits": self.class_head(hidden), "boxes": self.bbox_head(hidden), "hidden": hidden}
