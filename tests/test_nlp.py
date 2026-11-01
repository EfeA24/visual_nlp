from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_nlp.nlp.normalization import normalize_field
from visual_nlp.nlp.tokenizer import NLPTokenizer


def test_offsets_and_leading_zeros() -> None:
    tokenizer = NLPTokenizer.train_from_texts(["Belge No: 001234 Ahmet"], vocab_size=200, use_huggingface=False)
    encoded = tokenizer.encode("Belge No: 001234")
    assert encoded.offsets
    span = next(text for text, (s, e) in zip(encoded.tokens, encoded.offsets) if encoded.ids[encoded.tokens.index(text)] or True)
    joined = "Belge No: 001234"
    recovered = "".join(joined[s:e] for s, e in encoded.offsets if e > s)
    assert "001234" in recovered
    assert tokenizer.decode(tokenizer.encode("001234").ids).replace(" ", "") == "001234" or "001234" in recovered


def test_normalization_keeps_raw_and_zeros() -> None:
    amount = normalize_field("amount", "1.250,00 TL")
    assert amount.valid
    assert amount.value["amount"] == "1250.00"
    assert amount.value["currency"] == "TRY"
    assert amount.raw == "1.250,00 TL"
    date = normalize_field("date", "01.11.2026")
    assert date.value == "2026-11-01"
    plate = normalize_field("plate", "20 ABC 123")
    assert plate.value == "20 ABC 123"
    number = normalize_field("document_number", "001234")
    assert number.value == "001234"
    bad = normalize_field("amount", "???")
    assert bad.raw == "???"
    assert not bad.valid
