"""Visual reader: encode a cropped line and decode characters."""

from __future__ import annotations

import torch
import torch.nn as nn

from visual_nlp.reader.character_decoder import CharacterDecoder
from visual_nlp.reader.visual_encoder import VisualEncoder
from visual_nlp.utils.config import Config


class ReaderModel(nn.Module):
    def __init__(
        self,
        visual_encoder: VisualEncoder,
        character_decoder: CharacterDecoder,
    ) -> None:
        super().__init__()
        self.visual_encoder = visual_encoder
        self.character_decoder = character_decoder

    @classmethod
    def from_config(cls, config: Config, vocab_size: int, pad_id: int = 0) -> "ReaderModel":
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
        decoder = CharacterDecoder(
            vocab_size=vocab_size,
            d_model=int(config.decoder.d_model),
            n_heads=int(config.decoder.n_heads),
            n_layers=int(config.decoder.n_layers),
            d_ff=int(config.decoder.d_ff),
            max_len=int(config.decoder.get("max_output_length", 128)) + 1,
            dropout=float(config.decoder.get("dropout", 0.1)),
            activation=str(config.decoder.get("activation", "gelu")),
            pad_id=pad_id,
        )
        return cls(encoder, decoder)

    def encode(
        self,
        images: torch.Tensor,
        valid_hw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory, memory_mask, _ = self.visual_encoder(images, valid_hw=valid_hw)
        return memory, memory_mask

    def forward(
        self,
        images: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        valid_hw: torch.Tensor | None = None,
        tgt_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory, memory_mask = self.encode(images, valid_hw=valid_hw)
        return self.character_decoder(
            decoder_input_ids,
            memory,
            tgt_padding_mask=tgt_padding_mask,
            memory_padding_mask=memory_mask,
        )
