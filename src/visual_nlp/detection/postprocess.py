"""Turn raw detector outputs into usable line boxes. Does not read text."""

from __future__ import annotations

from typing import Sequence

import torch

from visual_nlp.training.losses import box_iou, cxcywh_to_xyxy


def xyxy_to_cxcywh(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = ((x1 + x2) / 2.0) / max(width, 1)
    cy = ((y1 + y2) / 2.0) / max(height, 1)
    w = (x2 - x1) / max(width, 1)
    h = (y2 - y1) / max(height, 1)
    return torch.stack([cx, cy, w, h], dim=-1)


def clip_xyxy(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    boxed = boxes.clone()
    boxed[:, 0].clamp_(0, width)
    boxed[:, 2].clamp_(0, width)
    boxed[:, 1].clamp_(0, height)
    boxed[:, 3].clamp_(0, height)
    return boxed


def valid_box_mask(
    boxes_cxcywh: torch.Tensor,
    min_width: float = 0.01,
    min_height: float = 0.005,
) -> torch.Tensor:
    if boxes_cxcywh.numel() == 0:
        return boxes_cxcywh.new_zeros((0,), dtype=torch.bool)
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    return (w >= min_width) & (h >= min_height) & (cx >= 0) & (cx <= 1) & (cy >= 0) & (cy <= 1)


def merge_by_iou(boxes_xyxy: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    if iou_threshold <= 0 or boxes_xyxy.numel() == 0:
        return boxes_xyxy, scores
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel():
        current = int(order[0].item())
        keep.append(current)
        if order.numel() == 1:
            break
        ious, _ = box_iou(boxes_xyxy[current : current + 1], boxes_xyxy[order[1:]])
        order = order[1:][ious[0] <= iou_threshold]
    idx = torch.tensor(keep, device=boxes_xyxy.device)
    return boxes_xyxy[idx], scores[idx]


def invert_boxes_to_original(
    boxes_xyxy: torch.Tensor,
    pad: torch.Tensor | Sequence[int],
    original_hw: torch.Tensor | Sequence[int],
    scale: float,
) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy
    left, top, *_ = [int(v) for v in pad]
    orig_h, orig_w = [int(v) for v in original_hw]
    inverted = boxes_xyxy.clone()
    inverted[:, [0, 2]] = (inverted[:, [0, 2]] - left) / max(scale, 1e-6)
    inverted[:, [1, 3]] = (inverted[:, [1, 3]] - top) / max(scale, 1e-6)
    return clip_xyxy(inverted, orig_h, orig_w)


def predictions_to_boxes(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    threshold: float,
    min_width: float = 0.01,
    min_height: float = 0.005,
    merge_iou: float = 0.0,
    image_hw: tuple[int, int] | None = None,
) -> list[dict[str, torch.Tensor]]:
    scores = logits.softmax(-1)[..., 1]
    results = []
    height, width = image_hw or (1, 1)
    for score, box in zip(scores, boxes):
        keep = (score >= threshold) & valid_box_mask(box, min_width, min_height)
        kept_boxes = box[keep]
        kept_scores = score[keep]
        xyxy = cxcywh_to_xyxy(kept_boxes)
        if image_hw is not None:
            xyxy = xyxy * torch.tensor([width, height, width, height], device=xyxy.device, dtype=xyxy.dtype)
            xyxy = clip_xyxy(xyxy, height, width)
        xyxy, kept_scores = merge_by_iou(xyxy, kept_scores, merge_iou)
        results.append(
            {
                "scores": kept_scores,
                "boxes_cxcywh": kept_boxes if image_hw is None else xyxy_to_cxcywh(xyxy, height, width),
                "boxes_xyxy": xyxy,
            }
        )
    return results
