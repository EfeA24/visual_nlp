"""Subword tokenizer with character offsets. Distinct from the reader charset."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PAD, UNK, MASK, BOS, EOS = "<PAD>", "<UNK>", "<MASK>", "<BOS>", "<EOS>"
SPECIAL_TOKENS = [PAD, UNK, MASK, BOS, EOS]
_TOKEN_RE = re.compile(r"\s+|[A-Za-zÇĞİÖŞÜçğıöşü]+|\d+|[^\sA-Za-zÇĞİÖŞÜçğıöşü\d]+", re.UNICODE)


@dataclass
class EncodedText:
    ids: list[int]
    tokens: list[str]
    offsets: list[tuple[int, int]]


class NLPTokenizer:
    def __init__(self, token_to_id: dict[str, int], id_to_token: list[str] | None = None) -> None:
        self.token_to_id = dict(token_to_id)
        self.id_to_token = id_to_token or [""] * (max(token_to_id.values()) + 1)
        if id_to_token is None:
            for token, idx in token_to_id.items():
                if idx >= len(self.id_to_token):
                    self.id_to_token.extend([""] * (idx + 1 - len(self.id_to_token)))
                self.id_to_token[idx] = token
        self.pad_id = self.token_to_id[PAD]
        self.unk_id = self.token_to_id[UNK]
        self.mask_id = self.token_to_id[MASK]
        self.bos_id = self.token_to_id[BOS]
        self.eos_id = self.token_to_id[EOS]
        self._hf = None

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def _split(self, text: str) -> list[tuple[str, int, int]]:
        pieces: list[tuple[str, int, int]] = []
        for match in _TOKEN_RE.finditer(text):
            token = match.group(0)
            if token.isspace():
                continue
            pieces.append((token, match.start(), match.end()))
        return pieces or [(text, 0, len(text))] if text else []

    def encode(self, text: str, max_length: int | None = None) -> EncodedText:
        if self._hf is not None:
            encoded = self._hf.encode(text)
            ids = list(encoded.ids)
            tokens = list(encoded.tokens)
            offsets = [(int(a), int(b)) for a, b in encoded.offsets]
        else:
            pieces = self._split(text)
            tokens = [piece[0] for piece in pieces]
            offsets = [(piece[1], piece[2]) for piece in pieces]
            ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        ids = [self.bos_id, *ids, self.eos_id]
        tokens = [BOS, *tokens, EOS]
        offsets = [(0, 0), *offsets, (len(text), len(text))]
        if max_length is not None and len(ids) > max_length:
            ids = ids[: max_length - 1] + [self.eos_id]
            tokens = tokens[: max_length - 1] + [EOS]
            offsets = offsets[: max_length - 1] + [(len(text), len(text))]
        return EncodedText(ids=ids, tokens=tokens, offsets=offsets)

    def decode(self, ids: Iterable[int]) -> str:
        pieces = []
        for idx in ids:
            if idx in {self.pad_id, self.bos_id}:
                continue
            if idx == self.eos_id:
                break
            pieces.append(self.id_to_token[idx] if 0 <= idx < len(self.id_to_token) else UNK)
        return " ".join(pieces)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"token_to_id": self.token_to_id, "id_to_token": self.id_to_token}
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if self._hf is not None:
            self._hf.save(str(target.with_suffix(".hf.json")))

    @classmethod
    def load(cls, path: str | Path) -> "NLPTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(payload["token_to_id"], payload.get("id_to_token"))
        hf_path = Path(path).with_suffix(".hf.json")
        if hf_path.exists():
            from tokenizers import Tokenizer

            tokenizer._hf = Tokenizer.from_file(str(hf_path))
        return tokenizer

    @classmethod
    def train_from_texts(
        cls,
        texts: Iterable[str],
        vocab_size: int = 16000,
        min_frequency: int = 2,
        use_huggingface: bool = True,
    ) -> "NLPTokenizer":
        corpus = [text for text in texts if text]
        if use_huggingface:
            try:
                return cls._train_hf(corpus, vocab_size, min_frequency)
            except Exception:
                pass
        counts: dict[str, int] = {}
        for text in corpus:
            for token, _s, _e in cls({PAD: 0, UNK: 1, MASK: 2, BOS: 3, EOS: 4})._split(text):
                counts[token] = counts.get(token, 0) + 1
        ranked = [token for token, freq in sorted(counts.items(), key=lambda item: (-item[1], item[0])) if freq >= min_frequency]
        tokens = SPECIAL_TOKENS + ranked
        tokens = tokens[: max(vocab_size, len(SPECIAL_TOKENS))]
        token_to_id = {token: idx for idx, token in enumerate(tokens)}
        return cls(token_to_id, tokens)

    @classmethod
    def _train_hf(cls, texts: list[str], vocab_size: int, min_frequency: int) -> "NLPTokenizer":
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer

        hf = Tokenizer(BPE(unk_token=UNK))
        hf.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=SPECIAL_TOKENS)
        hf.train_from_iterator(texts or ["Adı Soyadı 001234"], trainer)
        token_to_id = hf.get_vocab()
        id_to_token = [""] * (max(token_to_id.values()) + 1)
        for token, idx in token_to_id.items():
            id_to_token[idx] = token
        tokenizer = cls(token_to_id, id_to_token)
        tokenizer._hf = hf
        return tokenizer

    @classmethod
    def from_config(cls, path: str | Path, texts: Iterable[str] | None = None, vocab_size: int = 16000) -> "NLPTokenizer":
        target = Path(path)
        if target.exists():
            return cls.load(target)
        tokenizer = cls.train_from_texts(texts or ["Adı: Ahmet Yılmaz 001234"], vocab_size=vocab_size)
        tokenizer.save(target)
        return tokenizer
