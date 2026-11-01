from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_nlp.reader.model import ReaderModel
from visual_nlp.reader.tokenizer import CharacterTokenizer, default_charset
from visual_nlp.reader.visual_encoder import VisualEncoder
from visual_nlp.training.losses import sequence_cross_entropy
from visual_nlp.utils.config import load_config


def _tiny_model(vocab: int) -> ReaderModel:
    encoder = VisualEncoder(3, 16, 8, 32, 4, 1, 64, dropout=0.0)
    from visual_nlp.reader.character_decoder import CharacterDecoder

    decoder = CharacterDecoder(vocab, 32, 4, 1, 64, max_len=64, dropout=0.0)
    return ReaderModel(encoder, decoder)


def test_patch_count_and_logits() -> None:
    tokenizer = CharacterTokenizer(default_charset())
    model = _tiny_model(tokenizer.vocab_size)
    images = torch.randn(2, 3, 32, 40)
    valid_hw = torch.tensor([[32, 20], [32, 40]])
    tokens = torch.tensor([tokenizer.decoder_inputs("A1"), tokenizer.decoder_inputs("BB")])
    tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=tokenizer.pad_id)
    logits = model(images, tokens, valid_hw=valid_hw, tgt_padding_mask=tokens.eq(tokenizer.pad_id))
    memory, mask, grid = model.visual_encoder(images, valid_hw=valid_hw)
    assert grid.grid_h == 2
    assert grid.grid_w == 5
    assert memory.shape == (2, 10, 32)
    assert mask[0].sum() >= 1
    assert logits.shape == (2, tokens.size(1), tokenizer.vocab_size)


def test_padding_batch_backward() -> None:
    tokenizer = CharacterTokenizer(default_charset())
    model = _tiny_model(tokenizer.vocab_size)
    images = torch.randn(2, 3, 32, 48, requires_grad=False)
    tokens = torch.tensor([[tokenizer.bos_id, 10, tokenizer.pad_id], [tokenizer.bos_id, 11, 12]])
    targets = torch.tensor([[10, tokenizer.eos_id, tokenizer.pad_id], [11, 12, tokenizer.eos_id]])
    loss_mask = targets.ne(tokenizer.pad_id)
    logits = model(images, tokens, tgt_padding_mask=tokens.eq(tokenizer.pad_id))
    loss = sequence_cross_entropy(logits, targets, loss_mask)
    loss.backward()
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_reader_config_loads() -> None:
    config = load_config(ROOT / "configs/reader.yaml", model_name="reader")
    assert int(config.encoder.d_model) % int(config.encoder.n_heads) == 0
