from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_nlp.transformer.decoder import TransformerDecoder
from visual_nlp.transformer.encoder import TransformerEncoder
from visual_nlp.transformer.masks import (
    combine_masks,
    decoder_self_mask,
    expand_mask,
    padding_mask,
)


def test_padding_mask_blocks_pad_positions() -> None:
    mask = padding_mask(torch.tensor([2, 4]), max_length=4)
    assert mask.tolist() == [[False, False, True, True], [False, False, False, False]]


def test_future_tokens_are_blocked() -> None:
    tgt_pad = torch.zeros(1, 4, dtype=torch.bool)
    mask = decoder_self_mask(tgt_pad, num_heads=2)
    assert mask[0, 0, 1, 2]
    assert not mask[0, 0, 2, 1]


def test_combined_mask_or() -> None:
    a = torch.tensor([[True, False]])
    b = torch.tensor([[False, True]])
    assert combine_masks(a, b).tolist() == [[True, True]]


def test_expand_matches_batch_and_heads() -> None:
    mask = expand_mask(torch.tensor([[False, True]]), num_heads=3, query_length=2)
    assert mask.shape == (1, 3, 2, 2)
    assert mask[0, 2, 1, 1]


def test_future_target_change_does_not_affect_past_decoder_outputs() -> None:
    torch.manual_seed(0)
    decoder = TransformerDecoder(d_model=16, n_heads=4, n_layers=2, d_ff=32)
    encoder = TransformerEncoder(d_model=16, n_heads=4, n_layers=1, d_ff=32)
    memory = encoder(torch.randn(1, 5, 16))
    embed = nn.Embedding(12, 16)
    tokens_a = torch.tensor([[1, 2, 3, 4]])
    tokens_b = tokens_a.clone()
    tokens_b[0, -1] = 9
    out_a = decoder(embed(tokens_a), memory)
    out_b = decoder(embed(tokens_b), memory)
    assert torch.allclose(out_a[:, :-1], out_b[:, :-1], atol=1e-6)
    assert not torch.allclose(out_a[:, -1], out_b[:, -1], atol=1e-6)
