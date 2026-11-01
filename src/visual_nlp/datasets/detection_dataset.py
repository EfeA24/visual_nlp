"""Full-page image + line boxes. Transforms move boxes with the image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from visual_nlp.datasets.reader_dataset import load_jsonl
from visual_nlp.datasets.transforms import DetectorTransform
from visual_nlp.utils.config import find_project_root


def xyxy_to_cxcywh_normalized(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    x1, y1, x2, y2 = boxes.T
    cx = ((x1 + x2) / 2.0) / max(width, 1)
    cy = ((y1 + y2) / 2.0) / max(height, 1)
    w = (x2 - x1) / max(width, 1)
    h = (y2 - y1) / max(height, 1)
    return np.stack([cx, cy, w, h], axis=1).astype(np.float32)


class DetectionDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        transform: DetectorTransform,
        root: str | Path | None = None,
    ) -> None:
        self.rows = load_jsonl(manifest_path)
        self.transform = transform
        manifest = Path(manifest_path)
        self.root = Path(root) if root else manifest.parent.parent / "synthetic"
        if not self.root.exists():
            self.root = find_project_root() / "data" / "synthetic"

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = self.root / image_path
        image = Image.open(image_path).convert("RGB")
        boxes = np.asarray(row.get("boxes_xyxy") or ([row["box_xyxy"]] if "box_xyxy" in row else []), dtype=np.float32)
        prepared, shifted = self.transform(image, boxes)
        page_h, page_w = prepared.image.shape[-2:]
        return {
            "image": prepared.image,
            "valid_hw": torch.tensor(prepared.valid_hw, dtype=torch.long),
            "boxes_cxcywh": torch.tensor(xyxy_to_cxcywh_normalized(shifted, page_h, page_w), dtype=torch.float32),
            "boxes_xyxy": torch.tensor(shifted, dtype=torch.float32),
            "boxes_xyxy_original": torch.tensor(boxes, dtype=torch.float32),
            "original_size": torch.tensor(prepared.original_size, dtype=torch.long),
            "scale": prepared.scale,
            "pad": torch.tensor(prepared.pad, dtype=torch.long),
            "sample_id": row.get("id", str(index)),
            "source_id": row.get("source_id", row.get("id", str(index))),
            "image_path": str(image_path),
        }
