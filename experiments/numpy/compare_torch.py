"""Compare a NumPy attention step with the PyTorch module. Not used in training."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.numpy.attention_numpy import scaled_dot_product_attention as numpy_attention
from experiments.numpy.positional_numpy import layer_norm, sinusoidal_table
from visual_nlp.transformer.attention import scaled_dot_product_attention
from visual_nlp.transformer.normalization import LayerNorm
from visual_nlp.transformer.positional import sinusoidal_table as torch_table


def main() -> None:
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 5, 4)
    value = torch.randn(1, 2, 5, 4)
    out_t, w_t = scaled_dot_product_attention(query, key, value)
    out_n, w_n = numpy_attention(query.numpy(), key.numpy(), value.numpy())
    print("attn_out", float(np.max(np.abs(out_t.detach().numpy() - out_n))))
    print("attn_w", float(np.max(np.abs(w_t.detach().numpy() - w_n))))
    table_t = torch_table(16, 8).numpy()
    table_n = sinusoidal_table(16, 8)
    print("pos", float(np.max(np.abs(table_t - table_n))))
    x = torch.randn(2, 3, 8)
    ln = LayerNorm(8)
    with torch.no_grad():
        ln.weight.fill_(1.0)
        ln.bias.zero_()
    print("ln", float(np.max(np.abs(ln(x).detach().numpy() - layer_norm(x.numpy())))))


if __name__ == "__main__":
    main()
