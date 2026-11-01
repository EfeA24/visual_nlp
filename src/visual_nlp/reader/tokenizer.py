"""Character tokenizer. Case is preserved; I/İ/ı/i stay distinct."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"
SPECIAL_TOKENS = (PAD, BOS, EOS, UNK)

LOWER = "abcçdefgğhıijklmnoöpqrsştuüvwxyz"
UPPER = "ABCÇDEFGĞHIİJKLMNOÖPQRSŞTUÜVWXYZ"
DIGITS = "0123456789"
PUNCTUATION = " .,;:!?-_/()[]{}'\"%&@#+*=<>€$£₺°"


def default_charset() -> list[str]:
    seen: list[str] = []
    for token in (*SPECIAL_TOKENS, *LOWER, *UPPER, *DIGITS, *PUNCTUATION):
        if token not in seen:
            seen.append(token)
    return seen


class CharacterTokenizer:
    def __init__(self, tokens: Sequence[str], unk_policy: str = "unk") -> None:
        if list(tokens[:4]) != list(SPECIAL_TOKENS):
            raise ValueError("First four tokens must be <PAD> <BOS> <EOS> <UNK>.")
        self.tokens = list(tokens)
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.unk_policy = unk_policy
        self.pad_id = self.token_to_id[PAD]
        self.bos_id = self.token_to_id[BOS]
        self.eos_id = self.token_to_id[EOS]
        self.unk_id = self.token_to_id[UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def vocab_id(self) -> str:
        return "|".join(self.tokens)

    def encode_char(self, char: str) -> int:
        if char in self.token_to_id:
            return self.token_to_id[char]
        if self.unk_policy == "reject":
            raise ValueError(f"Unsupported character: {char!r}")
        return self.unk_id

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids = [self.encode_char(char) for char in text]
        if add_special:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        chars: list[str] = []
        for idx in ids:
            if idx < 0 or idx >= len(self.tokens):
                continue
            token = self.tokens[idx]
            if token == EOS:
                break
            if skip_special and token in SPECIAL_TOKENS:
                continue
            chars.append(token)
        return "".join(chars)

    def unsupported_characters(self, text: str) -> list[str]:
        return [char for char in text if char not in self.token_to_id]

    def decoder_inputs(self, text: str) -> list[int]:
        return [self.bos_id, *self.encode(text)]

    def targets(self, text: str) -> list[int]:
        return [*self.encode(text), self.eos_id]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tokens": self.tokens,
            "unk_policy": self.unk_policy,
            "special_tokens": list(SPECIAL_TOKENS),
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["tokens"], unk_policy=payload.get("unk_policy", "unk"))

    @classmethod
    def from_config(cls, path: str | Path, unk_policy: str = "unk") -> "CharacterTokenizer":
        target = Path(path)
        if target.exists():
            tokenizer = cls.load(target)
            tokenizer.unk_policy = unk_policy
            return tokenizer
        tokenizer = cls(default_charset(), unk_policy=unk_policy)
        tokenizer.save(target)
        return tokenizer
