"""Pad variable-size examples. Does not run models."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def _pad_images(images: Sequence[torch.Tensor]) -> torch.Tensor:
    max_h = max(image.size(-2) for image in images)
    max_w = max(image.size(-1) for image in images)
    batch = []
    for image in images:
        pad_h = max_h - image.size(-2)
        pad_w = max_w - image.size(-1)
        batch.append(F.pad(image, (0, pad_w, 0, pad_h)))
    return torch.stack(batch, dim=0)


def _pad_1d(sequences: Sequence[torch.Tensor], pad_value: int | float) -> torch.Tensor:
    max_len = max(seq.size(0) for seq in sequences)
    out = []
    for seq in sequences:
        pad = max_len - seq.size(0)
        if pad:
            extra = torch.full((pad, *seq.shape[1:]), pad_value, dtype=seq.dtype)
            seq = torch.cat([seq, extra], dim=0)
        out.append(seq)
    return torch.stack(out, dim=0)


def reader_collate(batch: list[dict], pad_id: int = 0) -> dict:
    images = _pad_images([item["image"] for item in batch])
    decoder_inputs = _pad_1d([item["decoder_input_ids"] for item in batch], pad_id)
    targets = _pad_1d([item["target_ids"] for item in batch], pad_id)
    loss_mask = targets.ne(pad_id)
    tgt_padding = decoder_inputs.eq(pad_id)
    return {
        "images": images,
        "valid_hw": torch.stack([item["valid_hw"] for item in batch]),
        "decoder_input_ids": decoder_inputs,
        "target_ids": targets,
        "tgt_padding_mask": tgt_padding,
        "loss_mask": loss_mask,
        "texts": [item["text"] for item in batch],
        "sample_ids": [item["sample_id"] for item in batch],
    }


def detection_collate(batch: list[dict]) -> dict:
    return {
        "images": _pad_images([item["image"] for item in batch]),
        "valid_hw": torch.stack([item["valid_hw"] for item in batch]),
        "boxes_cxcywh": [item["boxes_cxcywh"] for item in batch],
        "boxes_xyxy": [item["boxes_xyxy"] for item in batch],
        "boxes_xyxy_original": [item.get("boxes_xyxy_original", item["boxes_xyxy"]) for item in batch],
        "original_size": torch.stack([item["original_size"] for item in batch]),
        "pads": torch.stack([item["pad"] for item in batch]),
        "scales": torch.tensor([item["scale"] for item in batch], dtype=torch.float32),
        "sample_ids": [item["sample_id"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
    }


def nlp_collate(batch: list[dict], pad_id: int = 0) -> dict:
    input_ids = _pad_1d([item["input_ids"] for item in batch], pad_id)
    attention_blocked = input_ids.eq(pad_id)
    out = {
        "input_ids": input_ids,
        "padding_mask": attention_blocked,
        "loss_mask": input_ids.ne(pad_id),
        "texts": [item["text"] for item in batch],
        "sample_ids": [item["sample_id"] for item in batch],
        "has_field": torch.tensor([item.get("has_field", False) for item in batch]),
        "has_relation": torch.tensor([item.get("has_relation", False) for item in batch]),
        "has_document": torch.tensor([item.get("has_document", False) for item in batch]),
    }
    if "token_boxes" in batch[0]:
        out["token_boxes"] = _pad_1d([item["token_boxes"] for item in batch], 0.0)
        out["page_ids"] = _pad_1d([item["page_ids"] for item in batch], 0)
        out["field_labels"] = _pad_1d([item["field_labels"] for item in batch], -100)
        out["relations"] = torch.stack([item["relations"] for item in batch])
        out["document_label"] = torch.stack([item["document_label"] for item in batch])
        out["offsets"] = [item["offsets"] for item in batch]
    return out
