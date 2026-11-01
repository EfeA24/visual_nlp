"""NumPy positional encoding tables for inspection."""

from __future__ import annotations

import math

import numpy as np


def sinusoidal_table(max_len: int, d_model: int) -> np.ndarray:
    position = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    table = np.zeros((max_len, d_model), dtype=np.float64)
    table[:, 0::2] = np.sin(position * div)
    table[:, 1::2] = np.cos(position * div)
    return table


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)
