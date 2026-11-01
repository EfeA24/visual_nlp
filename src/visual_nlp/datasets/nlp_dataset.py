"""NLP pretraining chunks and fine-tuning field/relation examples."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from visual_nlp.datasets.reader_dataset import load_jsonl


def _inside_field(pos: int, fields: list[dict]) -> bool:
    return any(int(field["start"]) < pos < int(field["end"]) for field in fields)


def _shift_fields(fields: list[dict], start: int, end: int) -> list[dict]:
    shifted = []
    for field in fields:
        fs, fe = int(field["start"]), int(field["end"])
        if fe <= start or fs >= end:
            continue
        if fs < start or fe > end:
            continue
        item = dict(field)
        item["start"] = fs - start
        item["end"] = fe - start
        shifted.append(item)
    return shifted


def chunk_document(
    text: str,
    tokenizer,
    max_length: int,
    fields: list[dict] | None = None,
    source_id: str = "",
    extra: dict | None = None,
) -> list[dict]:
    """Split long text without cutting annotated fields in half."""
    fields = fields or []
    encoded = tokenizer.encode(text)
    if len(encoded.ids) <= max_length:
        record = {"text": text, "fields": fields, "chunk_id": 0, "source_id": source_id, **(extra or {})}
        return [record]
    # offsets include BOS/EOS dummy spans; use real token ends as candidate cuts
    cuts = [0]
    last_safe = 0
    token_budget = 2  # BOS + EOS
    for token, (start, end) in zip(encoded.tokens[1:-1], encoded.offsets[1:-1]):
        token_budget += 1
        if end > last_safe and not _inside_field(end, fields):
            last_safe = end
        if token_budget >= max_length:
            cut = last_safe if last_safe > cuts[-1] else end
            if _inside_field(cut, fields):
                covering = next(field for field in fields if int(field["start"]) < cut < int(field["end"]))
                cut = int(covering["end"])
            if cut <= cuts[-1]:
                cut = min(len(text), cuts[-1] + 1)
            cuts.append(cut)
            token_budget = 2
            last_safe = cut
    if cuts[-1] < len(text):
        cuts.append(len(text))
    chunks = []
    for idx, (start, end) in enumerate(zip(cuts, cuts[1:])):
        chunk_fields = _shift_fields(fields, start, end)
        chunk_extra = dict(extra or {})
        relations = []
        present = {field["type"] for field in chunk_fields}
        for rel in chunk_extra.get("relations") or []:
            if rel.get("head") in present and rel.get("tail") in present:
                relations.append(rel)
        chunk_extra["relations"] = relations
        chunks.append(
            {
                "text": text[start:end],
                "fields": chunk_fields,
                "chunk_id": idx,
                "source_id": source_id,
                "char_start": start,
                **chunk_extra,
            }
        )
    return chunks


class NLPPretrainDataset(Dataset):
    def __init__(self, manifest_path: str | Path, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        raw = load_jsonl(manifest_path)
        self.rows: list[dict] = []
        for row in raw:
            self.rows.extend(
                chunk_document(
                    row["text"],
                    tokenizer,
                    max_length,
                    source_id=str(row.get("source_id", row.get("id", ""))),
                    extra={"id": row.get("id")},
                )
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        text = row["text"]
        encoded = self.tokenizer.encode(text, max_length=self.max_length)
        return {
            "input_ids": torch.tensor(encoded.ids, dtype=torch.long),
            "offsets": encoded.offsets,
            "text": text,
            "sample_id": f"{row.get('id', index)}_{row.get('chunk_id', 0)}",
            "source_id": row.get("source_id", row.get("id", str(index))),
            "chunk_id": int(row.get("chunk_id", 0)),
            "has_field": False,
            "has_relation": False,
            "has_document": False,
        }


class NLPFinetuneDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer,
        max_length: int,
        field_labels: list[str],
        document_classes: list[str],
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.field_labels = field_labels
        self.document_classes = document_classes
        self.label_to_id = {label: idx for idx, label in enumerate(field_labels)}
        self.doc_to_id = {label: idx for idx, label in enumerate(document_classes)}
        raw = load_jsonl(manifest_path)
        self.rows: list[dict] = []
        for row in raw:
            extra = {
                "id": row.get("id"),
                "relations": row.get("relations") or [],
                "document_class": row.get("document_class"),
                "page_id": row.get("page_id", 0),
                "token_boxes": row.get("token_boxes"),
                "boxes": row.get("boxes"),
                "box_cxcywh": row.get("box_cxcywh"),
            }
            self.rows.extend(
                chunk_document(
                    row["text"],
                    tokenizer,
                    max_length,
                    fields=row.get("fields") or [],
                    source_id=str(row.get("source_id", row.get("id", ""))),
                    extra=extra,
                )
            )

    def __len__(self) -> int:
        return len(self.rows)

    def _align_bio(self, offsets: list[tuple[int, int]], fields: list[dict], seq_len: int) -> list[int]:
        tags = [0] * seq_len
        for field in fields:
            start, end = int(field["start"]), int(field["end"])
            label = field["type"]
            if label not in self.label_to_id:
                continue
            base = 1 + 2 * self.label_to_id[label]
            first = True
            for idx, (tok_s, tok_e) in enumerate(offsets):
                if tok_e <= start or tok_s >= end or tok_s == tok_e:
                    continue
                tags[idx] = base if first else base + 1
                first = False
        return tags

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        text = row["text"]
        encoded = self.tokenizer.encode(text, max_length=self.max_length)
        fields = row.get("fields") or []
        bio = self._align_bio(encoded.offsets, fields, len(encoded.ids))
        boxes = row.get("token_boxes") or row.get("boxes") or []
        if boxes and len(boxes) == len(encoded.ids):
            token_boxes = torch.tensor(boxes, dtype=torch.float32)
        else:
            page_box = row.get("box_cxcywh") or [0.0, 0.0, 0.0, 0.0]
            token_boxes = torch.tensor([page_box] * len(encoded.ids), dtype=torch.float32)
        page_ids = torch.full((len(encoded.ids),), int(row.get("page_id", 0)), dtype=torch.long)
        relations = row.get("relations") or []
        relation_tensor = torch.zeros(len(self.field_labels), len(self.field_labels), dtype=torch.float32)
        for rel in relations:
            src, dst = rel.get("head"), rel.get("tail")
            if src in self.label_to_id and dst in self.label_to_id:
                relation_tensor[self.label_to_id[src], self.label_to_id[dst]] = 1.0
        doc_label = row.get("document_class")
        return {
            "input_ids": torch.tensor(encoded.ids, dtype=torch.long),
            "offsets": encoded.offsets,
            "text": text,
            "field_labels": torch.tensor(bio, dtype=torch.long),
            "token_boxes": token_boxes,
            "page_ids": page_ids,
            "relations": relation_tensor,
            "document_label": torch.tensor(
                self.doc_to_id.get(doc_label, 0) if doc_label is not None else -100,
                dtype=torch.long,
            ),
            "has_field": bool(fields) or "fields" in row,
            "has_relation": "relations" in row,
            "has_document": doc_label is not None,
            "sample_id": f"{row.get('id', index)}_{row.get('chunk_id', 0)}",
            "source_id": row.get("source_id", row.get("id", str(index))),
            "chunk_id": int(row.get("chunk_id", 0)),
        }
