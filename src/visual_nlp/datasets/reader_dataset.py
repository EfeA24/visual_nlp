"""Cropped line image + transcription -> reader training example."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from visual_nlp.datasets.transforms import ReaderTransform
from visual_nlp.reader.tokenizer import CharacterTokenizer
from visual_nlp.utils.config import find_project_root


def load_jsonl(path: str | Path) -> list[dict]:
    target = Path(path)
    if not target.exists():
        return []
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]


class ReaderDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: CharacterTokenizer,
        transform: ReaderTransform,
        root: str | Path | None = None,
    ) -> None:
        self.rows = load_jsonl(manifest_path)
        self.tokenizer = tokenizer
        self.transform = transform
        manifest = Path(manifest_path)
        self.root = Path(root) if root else (manifest.parent.parent / "synthetic")
        if not self.root.exists():
            self.root = find_project_root() / "data" / "synthetic"

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = self.root / image_path
        image = Image.open(image_path)
        prepared = self.transform(image)
        text = row["text"]
        return {
            "image": prepared.image,
            "valid_hw": torch.tensor(prepared.valid_hw, dtype=torch.long),
            "decoder_input_ids": torch.tensor(self.tokenizer.decoder_inputs(text), dtype=torch.long),
            "target_ids": torch.tensor(self.tokenizer.targets(text), dtype=torch.long),
            "text": text,
            "sample_id": row.get("id", str(index)),
            "source_id": row.get("source_id", row.get("id", str(index))),
            "image_path": str(image_path),
        }
