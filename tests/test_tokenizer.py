from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_nlp.reader.tokenizer import CharacterTokenizer, default_charset


def test_encode_decode_roundtrip(tmp_path: Path) -> None:
    tokenizer = CharacterTokenizer(default_charset())
    text = "Ahmet Yılmaz 001234 İı"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    path = tmp_path / "tok.json"
    tokenizer.save(path)
    loaded = CharacterTokenizer.load(path)
    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.tokens == tokenizer.tokens


def test_turkish_case_and_leading_zeros() -> None:
    tokenizer = CharacterTokenizer(default_charset())
    for char in "IİıiĞğŞşÇçÖöÜü":
        assert tokenizer.decode(tokenizer.encode(char)) == char
    assert tokenizer.decode(tokenizer.encode("001234")) == "001234"


def test_special_tokens_and_unknown_policy() -> None:
    tokenizer = CharacterTokenizer(default_charset(), unk_policy="unk")
    ids = tokenizer.decoder_inputs("A")
    assert ids[0] == tokenizer.bos_id
    assert tokenizer.targets("A")[-1] == tokenizer.eos_id
    assert tokenizer.encode("§") == [tokenizer.unk_id]
    reject = CharacterTokenizer(default_charset(), unk_policy="reject")
    try:
        reject.encode("§")
        raise AssertionError("expected reject")
    except ValueError:
        pass
