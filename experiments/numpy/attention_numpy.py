"""Step-by-step NumPy attention used only for math checks."""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    d_k = query.shape[-1]
    scores = query @ np.swapaxes(key, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    weights = softmax(scores, axis=-1)
    if mask is not None:
        all_blocked = np.all(mask, axis=-1, keepdims=True)
        weights = np.where(all_blocked, 0.0, weights)
    return weights @ value, weights


def split_heads(x: np.ndarray, n_heads: int) -> np.ndarray:
    batch, length, dim = x.shape
    return x.reshape(batch, length, n_heads, dim // n_heads).transpose(0, 2, 1, 3)


def merge_heads(x: np.ndarray) -> np.ndarray:
    batch, n_heads, length, d_head = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, length, n_heads * d_head)
