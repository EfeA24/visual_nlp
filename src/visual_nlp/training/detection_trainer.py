"""Detector training with Hungarian matching and original-coordinate metrics."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from visual_nlp.detection.postprocess import invert_boxes_to_original, predictions_to_boxes
from visual_nlp.training.checkpoint import save_checkpoint
from visual_nlp.training.losses import detector_loss
from visual_nlp.training.metrics import detection_prf
from visual_nlp.utils.config import Config
from visual_nlp.utils.device import mixed_precision_enabled, move_to_device
from visual_nlp.utils.logging import RunLogger


class DetectionTrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        config: Config,
        device: torch.device,
        logger: RunLogger,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.logger = logger
        self.use_amp = mixed_precision_enabled(config) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.best_metric = -float("inf")
        self.step = 0

    def _loss(self, batch: dict) -> dict[str, torch.Tensor]:
        images = batch["images"].to(self.device)
        valid_hw = batch["valid_hw"].to(self.device)
        outputs = self.model(images, valid_hw=valid_hw)
        targets = [boxes.to(self.device) for boxes in batch["boxes_cxcywh"]]
        return detector_loss(
            outputs["logits"],
            outputs["boxes"],
            targets,
            presence_weight=float(self.config.loss.presence_weight),
            bbox_l1_weight=float(self.config.loss.bbox_l1_weight),
            giou_weight=float(self.config.loss.giou_weight),
            no_object_weight=float(self.config.loss.no_object_weight),
            class_cost=float(self.config.loss.class_cost),
            bbox_cost=float(self.config.loss.bbox_cost),
            giou_cost=float(self.config.loss.giou_cost),
        )

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total = 0.0
        count = 0
        accum = int(self.config.train.get("grad_accum", 1))
        self.optimizer.zero_grad(set_to_none=True)
        for idx, batch in enumerate(loader, start=1):
            losses = self._loss(batch)
            loss = losses["loss"] / accum
            self.scaler.scale(loss).backward()
            if idx % accum == 0:
                if float(self.config.train.get("grad_clip", 0)) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.train.grad_clip))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.step += 1
            total += float(losses["loss"].item())
            count += 1
        mean = total / max(count, 1)
        self.logger.log_metrics("train", self.step, {"loss": mean, "epoch": epoch})
        return mean

    @torch.no_grad()
    def validate(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.eval()
        losses = []
        pred_boxes, pred_scores, tgt_boxes = [], [], []
        for batch in loader:
            batch_dev = {
                "images": batch["images"].to(self.device),
                "valid_hw": batch["valid_hw"].to(self.device),
                "boxes_cxcywh": batch["boxes_cxcywh"],
            }
            outputs = self.model(batch_dev["images"], valid_hw=batch_dev["valid_hw"])
            loss = detector_loss(outputs["logits"], outputs["boxes"], [b.to(self.device) for b in batch["boxes_cxcywh"]])
            losses.append(float(loss["loss"].item()))
            image_h, image_w = int(batch["images"].size(-2)), int(batch["images"].size(-1))
            decoded = predictions_to_boxes(
                outputs["logits"],
                outputs["boxes"],
                threshold=0.0,
                image_hw=(image_h, image_w),
            )
            originals = batch.get("boxes_xyxy_original") or batch["boxes_xyxy"]
            for item, pad, orig, scale, target in zip(
                decoded, batch["pads"], batch["original_size"], batch["scales"], originals
            ):
                pred_orig = invert_boxes_to_original(item["boxes_xyxy"].cpu(), pad, orig, float(scale))
                pred_boxes.append(pred_orig)
                pred_scores.append(item["scores"].cpu())
                tgt_boxes.append(target.cpu() if torch.is_tensor(target) else target)
        prf = detection_prf(
            pred_boxes,
            pred_scores,
            tgt_boxes,
            score_threshold=float(self.config.predict.score_threshold),
            box_format="xyxy",
        )
        metrics = {"val_loss": sum(losses) / max(len(losses), 1), **prf}
        self.logger.log_metrics("val", self.step, metrics, epoch=epoch)
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None) -> None:
        ckpt_dir = Path(self.config.checkpoint.dir)
        for epoch in range(1, int(self.config.train.max_epochs) + 1):
            self.train_epoch(train_loader, epoch)
            metrics = self.validate(val_loader, epoch) if val_loader is not None else {"f1": 0.0}
            save_checkpoint(
                ckpt_dir / "last.pt",
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                epoch=epoch,
                step=self.step,
                best_metric=self.best_metric,
                config=self.config,
            )
            value = metrics.get(str(self.config.checkpoint.get("best_metric", "f1")), 0.0)
            if value > self.best_metric:
                self.best_metric = value
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    epoch=epoch,
                    step=self.step,
                    best_metric=self.best_metric,
                    config=self.config,
                )
