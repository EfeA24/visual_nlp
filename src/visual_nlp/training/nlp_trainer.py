"""NLP pretraining (MLM) and fine-tuning with optional task masks."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from visual_nlp.training.checkpoint import save_checkpoint
from visual_nlp.training.losses import nlp_multitask_loss
from visual_nlp.training.metrics import span_f1
from visual_nlp.utils.config import Config
from visual_nlp.utils.device import mixed_precision_enabled, move_to_device
from visual_nlp.utils.logging import RunLogger


def apply_mlm(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    vocab_size: int,
    mask_id: int,
    mask_prob: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = input_ids.clone()
    probability = torch.full(input_ids.shape, mask_prob, device=input_ids.device)
    masked = torch.bernoulli(probability).bool() & loss_mask
    labels = labels.masked_fill(~masked, -100)
    random_ids = torch.randint(0, vocab_size, input_ids.shape, device=input_ids.device)
    replace = input_ids.clone()
    replace[masked] = mask_id
    bernoulli = torch.rand(input_ids.shape, device=input_ids.device)
    random_mask = masked & (bernoulli >= 0.8) & (bernoulli < 0.9)
    replace[random_mask] = random_ids[random_mask]
    keep = masked & (bernoulli >= 0.9)
    replace[keep] = input_ids[keep]
    return replace, labels, masked


class NLPTrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        config: Config,
        device: torch.device,
        logger: RunLogger,
        mode: str = "pretrain",
        field_labels: list[str] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.logger = logger
        self.mode = mode
        self.field_labels = field_labels or []
        train_cfg = config.pretrain.train if mode == "pretrain" else config.finetune.train
        self.train_cfg = train_cfg
        self.use_amp = mixed_precision_enabled(Config({"train": train_cfg})) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.best_metric = float("inf") if mode == "pretrain" else -float("inf")
        self.step = 0

    def _step_losses(self, batch: dict) -> dict[str, torch.Tensor]:
        batch = move_to_device(batch, self.device)
        if self.mode == "pretrain":
            masked, labels, mask = apply_mlm(
                batch["input_ids"],
                batch["loss_mask"],
                vocab_size=self.model.vocab_size,
                mask_id=self.model.mask_id,
                mask_prob=float(self.config.pretrain.mask_prob),
            )
            batch["input_ids"] = masked
            batch["mlm_labels"] = labels
            batch["mlm_mask"] = mask
            outputs = self.model(batch, mode="pretrain")
            return nlp_multitask_loss(outputs, batch, {})
        outputs = self.model(batch, mode="finetune")
        weights = dict(self.config.finetune.get("loss_weights", {}))
        return nlp_multitask_loss(outputs, batch, weights)

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total = 0.0
        count = 0
        accum = int(self.train_cfg.get("grad_accum", 1))
        self.optimizer.zero_grad(set_to_none=True)
        for idx, batch in enumerate(loader, start=1):
            losses = self._step_losses(batch)
            loss = losses["loss"] / accum
            self.scaler.scale(loss).backward()
            if idx % accum == 0:
                if float(self.train_cfg.get("grad_clip", 0)) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.train_cfg.grad_clip))
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
        gold_spans, pred_spans = [], []
        for batch in loader:
            losses.append(float(self._step_losses(batch)["loss"].item()))
            if self.mode == "finetune":
                moved = move_to_device(batch, self.device)
                outputs = self.model(moved, mode="finetune")
                pred = outputs["field_logits"].argmax(-1)
                for labels, preds, mask, text, offsets in zip(
                    moved["field_labels"], pred, moved["loss_mask"], batch["texts"], batch["offsets"]
                ):
                    gold_spans.extend(_bio_to_spans(labels.tolist(), offsets, text, self.field_labels))
                    pred_spans.extend(_bio_to_spans(preds.tolist(), offsets, text, self.field_labels, valid=mask.tolist()))
        metrics = {"val_loss": sum(losses) / max(len(losses), 1)}
        if self.mode == "finetune":
            metrics.update(span_f1(gold_spans, pred_spans))
        self.logger.log_metrics("val", self.step, metrics, epoch=epoch)
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None) -> None:
        ckpt_dir = Path(self.config.pretrain.checkpoint if self.mode == "pretrain" else self.config.checkpoint.dir)
        for epoch in range(1, int(self.train_cfg.max_epochs) + 1):
            self.train_epoch(train_loader, epoch)
            metrics = self.validate(val_loader, epoch) if val_loader is not None else {}
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
            if self.mode == "pretrain":
                value = metrics.get("val_loss", float("inf"))
                better = value < self.best_metric
            else:
                value = metrics.get("field_f1", 0.0)
                better = value > self.best_metric
            if better:
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


def _bio_to_spans(
    tags: list[int],
    offsets: list[tuple[int, int]],
    text: str,
    field_labels: list[str],
    valid: list[bool] | None = None,
) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    current = None
    for idx, tag in enumerate(tags):
        if valid is not None and (idx >= len(valid) or not valid[idx]):
            continue
        if tag <= 0 or idx >= len(offsets):
            if current:
                spans.append(current)
                current = None
            continue
        label_idx = (tag - 1) // 2
        is_begin = (tag - 1) % 2 == 0
        if label_idx >= len(field_labels):
            continue
        label = field_labels[label_idx]
        start, end = offsets[idx]
        if current is None or is_begin or current[2] != label:
            if current:
                spans.append(current)
            current = (start, end, label)
        else:
            current = (current[0], end, label)
    if current:
        spans.append(current)
    return spans
