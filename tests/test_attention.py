from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.numpy.attention_numpy import scaled_dot_product_attention as numpy_attention
from visual_nlp.transformer.attention import MultiHeadAttention, scaled_dot_product_attention
from visual_nlp.transformer.masks import causal_mask, expand_mask


def test_attention_matches_hand_example() -> None:
    query = torch.tensor([[[[1.0, 0.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    value = torch.tensor([[[[2.0, 0.0], [0.0, 4.0]]]])
    output, weights = scaled_dot_product_attention(query, key, value)
    np_out, np_w = numpy_attention(query.numpy(), key.numpy(), value.numpy())
    assert torch.allclose(output, torch.tensor(np_out, dtype=output.dtype), atol=1e-6)
    assert torch.allclose(weights, torch.tensor(np_w, dtype=weights.dtype), atol=1e-6)
    assert weights[0, 0, 0, 0] > weights[0, 0, 0, 1]


def test_multihead_shapes_and_cross_attention() -> None:
    attn = MultiHeadAttention(d_model=16, n_heads=4)
    query = torch.randn(2, 3, 16)
    key = torch.randn(2, 5, 16)
    value = torch.randn(2, 5, 16)
    out, weights = attn(query, key, value, return_weights=True)
    assert out.shape == (2, 3, 16)
    assert weights.shape == (2, 4, 3, 5)


def test_gradients_are_finite() -> None:
    attn = MultiHeadAttention(d_model=8, n_heads=2)
    x = torch.randn(1, 4, 8, requires_grad=True)
    out = attn(x, x, x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_masked_positions_do_not_contribute() -> None:
    query = torch.randn(1, 1, 2, 4)
    key = torch.randn(1, 1, 2, 4)
    value = torch.tensor([[[[1.0, 0, 0, 0], [100.0, 0, 0, 0]]]])
    mask = torch.tensor([[[[False, True]]]])
    output, weights = scaled_dot_product_attention(query, key, value, mask=mask)
    assert torch.allclose(weights[..., 1], torch.zeros_like(weights[..., 1]))
    assert torch.allclose(output, value[:, :, :1, :])


def test_causal_mask_expansion() -> None:
    mask = expand_mask(causal_mask(3), num_heads=2)
    assert mask.shape == (1, 2, 3, 3)
    assert mask[0, 0, 0, 1]
    assert not mask[0, 0, 1, 0]
