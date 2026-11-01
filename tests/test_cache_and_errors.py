from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_nlp.reader.character_decoder import CharacterDecoder
from visual_nlp.reader.decoding import greedy_decode
from visual_nlp.reader.model import ReaderModel
from visual_nlp.reader.tokenizer import CharacterTokenizer, default_charset
from visual_nlp.reader.visual_encoder import VisualEncoder
from visual_nlp.training.metrics import reader_error_summary


def test_reader_error_types() -> None:
    summary = reader_error_summary(["İstanbul 12."], ["Istanbul 13,"])
    assert summary["turkish_case_errors"] >= 1
    assert summary["digit_errors"] >= 1
    assert summary["punctuation_errors"] >= 1


def test_cached_decode_matches_full_decode() -> None:
    torch.manual_seed(0)
    tokenizer = CharacterTokenizer(default_charset())
    model = ReaderModel(
        VisualEncoder(3, 16, 8, 32, 4, 1, 64, dropout=0.0),
        CharacterDecoder(tokenizer.vocab_size, 32, 4, 1, 64, max_len=64, dropout=0.0),
    )
    model.eval()
    images = torch.randn(2, 3, 32, 48)
    cached = greedy_decode(model, images, tokenizer, max_length=8, use_cache=True)
    full = greedy_decode(model, images, tokenizer, max_length=8, use_cache=False)
    assert cached.token_ids == full.token_ids
    assert cached.texts == full.texts
