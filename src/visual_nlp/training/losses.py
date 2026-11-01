"""Task losses. Attention masks and loss masks are different contracts.

Loss mask convention: True means the position contributes to the loss.
Padding must not change the loss scale; we average over valid items only.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def sequence_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    vocab = logits.size(-1)
    token_loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), reduction="none")
    token_loss = token_loss.view_as(targets)
    valid = loss_mask.bool()
    if valid.sum() == 0:
        return logits.new_zeros(())
    return token_loss[valid].mean()


def masked_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    return sequence_cross_entropy(logits, targets, loss_mask)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6), union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area.clamp(min=1e-6)


def match_boxes(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    class_cost: float,
    bbox_cost: float,
    giou_cost: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target_boxes.numel() == 0:
        return pred_logits.new_empty((0,), dtype=torch.long), pred_logits.new_empty((0,), dtype=torch.long)
    prob = pred_logits.softmax(-1)[:, 1]
    cost_class = -prob[:, None]
    cost_bbox = torch.cdist(pred_boxes, target_boxes, p=1)
    cost_giou = -generalized_box_iou(cxcywh_to_xyxy(pred_boxes), cxcywh_to_xyxy(target_boxes))
    cost = class_cost * cost_class + bbox_cost * cost_bbox + giou_cost * cost_giou
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    return pred_logits.new_tensor(rows, dtype=torch.long), pred_logits.new_tensor(cols, dtype=torch.long)


def detector_loss(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    targets: Sequence[torch.Tensor],
    presence_weight: float = 1.0,
    bbox_l1_weight: float = 5.0,
    giou_weight: float = 2.0,
    no_object_weight: float = 0.1,
    class_cost: float = 1.0,
    bbox_cost: float = 5.0,
    giou_cost: float = 2.0,
) -> dict[str, torch.Tensor]:
    batch, num_queries, _ = pred_logits.shape
    target_classes = torch.zeros(batch, num_queries, dtype=torch.long, device=pred_logits.device)
    box_losses = []
    giou_losses = []
    for idx in range(batch):
        gt = targets[idx]
        src_idx, tgt_idx = match_boxes(
            pred_logits[idx], pred_boxes[idx], gt, class_cost, bbox_cost, giou_cost
        )
        if src_idx.numel():
            target_classes[idx, src_idx] = 1
            matched_pred = pred_boxes[idx, src_idx]
            matched_gt = gt[tgt_idx]
            box_losses.append((matched_pred - matched_gt).abs().mean())
            giou = torch.diag(generalized_box_iou(cxcywh_to_xyxy(matched_pred), cxcywh_to_xyxy(matched_gt)))
            giou_losses.append((1.0 - giou).mean())
    weight = torch.ones(2, device=pred_logits.device)
    weight[0] = no_object_weight
    cls_loss = F.cross_entropy(
        pred_logits.reshape(-1, 2),
        target_classes.reshape(-1),
        weight=weight,
    )
    zero = pred_logits.new_zeros(())
    l1 = torch.stack(box_losses).mean() if box_losses else zero
    giou = torch.stack(giou_losses).mean() if giou_losses else zero
    total = presence_weight * cls_loss + bbox_l1_weight * l1 + giou_weight * giou
    return {"loss": total, "loss_cls": cls_loss, "loss_l1": l1, "loss_giou": giou}


def nlp_multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    total = None
    if "mlm_logits" in outputs:
        mlm = sequence_cross_entropy(outputs["mlm_logits"], batch["mlm_labels"], batch["mlm_mask"])
        losses["loss_mlm"] = mlm
        total = mlm if total is None else total + mlm
    if "field_logits" in outputs and batch["has_field"].any():
        mask = batch["has_field"].unsqueeze(1) & batch["field_labels"].ne(-100)
        field = sequence_cross_entropy(outputs["field_logits"], batch["field_labels"].clamp(min=0), mask)
        losses["loss_field"] = field
        total = field * weights.get("field", 1.0) if total is None else total + field * weights.get("field", 1.0)
    if "relation_logits" in outputs and batch["has_relation"].any():
        rel = F.binary_cross_entropy_with_logits(
            outputs["relation_logits"][batch["has_relation"]],
            batch["relations"][batch["has_relation"]],
        )
        losses["loss_relation"] = rel
        total = rel * weights.get("relation", 1.0) if total is None else total + rel * weights.get("relation", 1.0)
    if "document_logits" in outputs and batch["has_document"].any():
        labels = batch["document_label"]
        valid = batch["has_document"] & labels.ne(-100)
        if valid.any():
            doc = F.cross_entropy(outputs["document_logits"][valid], labels[valid])
            losses["loss_document"] = doc
            total = doc * weights.get("document", 1.0) if total is None else total + doc * weights.get("document", 1.0)
    if total is None:
        total = next(iter(outputs.values())).new_zeros(())
    losses["loss"] = total
    return losses
