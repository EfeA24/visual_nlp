from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_nlp.datasets.nlp_dataset import chunk_document
from visual_nlp.nlp.tokenizer import NLPTokenizer


def test_chunking_does_not_cut_fields() -> None:
    words = [f"kelime{i}" for i in range(20)]
    tokenizer = NLPTokenizer.train_from_texts(words + ["Ahmet", "Yilmaz", "Adi"], vocab_size=200, use_huggingface=False)
    name = "Ahmet Yilmaz"
    text = " ".join(words) + f" Adi: {name} " + " ".join(f"son{i}" for i in range(20))
    start = text.index(name)
    end = start + len(name)
    chunks = chunk_document(
        text,
        tokenizer,
        max_length=16,
        fields=[{"start": start, "end": end, "type": "person_name"}],
        source_id="doc1",
    )
    assert len(chunks) >= 2
    assert all(chunk["source_id"] == "doc1" for chunk in chunks)
    covering = [chunk for chunk in chunks if chunk["fields"]]
    assert covering
    for chunk in covering:
        field = chunk["fields"][0]
        assert chunk["text"][field["start"] : field["end"]] == name


def test_short_document_stays_one_chunk() -> None:
    tokenizer = NLPTokenizer.train_from_texts(["kisa metin"], vocab_size=40, use_huggingface=False)
    chunks = chunk_document("kisa metin", tokenizer, max_length=64, source_id="s")
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == 0
