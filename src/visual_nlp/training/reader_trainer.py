"""Reader training loop. Teacher-forcing loss is not free-run accuracy."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from visual_nlp.reader.decoding import decode_images
from visual_nlp.reader.model import ReaderModel
from visual_nlp.reader.tokenizer import CharacterTokenizer
from visual_nlp.training.checkpoint import save_checkpoint
from visual_nlp.training.losses import sequence_cross_entropy
from visual_nlp.training.metrics import reader_metrics
from visual_nlp.utils.config import Config
from visual_nlp.utils.device import mixed_precision_enabled, move_to_device
from visual_nlp.utils.logging import RunLogger


class ReaderTrainer:
    def __init__(
        self,
        model: ReaderModel,
        optimizer: torch.optim.Optimizer,
        scheduler,
        tokenizer: CharacterTokenizer,
        config: Config,
        device: torch.device,
        logger: RunLogger,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.logger = logger
        self.use_amp = mixed_precision_enabled(config) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.best_metric = float("inf")
        self.step = 0

    def _forward_loss(self, batch: dict) -> torch.Tensor:
        batch = move_to_device(batch, self.device)
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            logits = self.model(
                batch["images"],
                batch["decoder_input_ids"],
                valid_hw=batch["valid_hw"],
                tgt_padding_mask=batch["tgt_padding_mask"],
            )
            return sequence_cross_entropy(logits, batch["target_ids"], batch["loss_mask"])

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total = 0.0
        count = 0
        accum = int(self.config.train.get("grad_accum", 1))
        self.optimizer.zero_grad(set_to_none=True)
        for idx, batch in enumerate(loader, start=1):
            loss = self._forward_loss(batch) / accum
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
            total += float(loss.item()) * accum
            count += 1
        mean = total / max(count, 1)
        self.logger.log_metrics("train", self.step, {"loss": mean, "epoch": epoch, "lr": self.scheduler.get_last_lr()[0]})
        return mean

    @torch.no_grad()
    def validate(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.eval()
        losses = []
        refs: list[str] = []
        hyps: list[str] = []
        max_decode = int(self.config.train.get("max_decode_samples", 64))
        for batch in loader:
            batch = move_to_device(batch, self.device)
            loss = self._forward_loss(batch)
            losses.append(float(loss.item()))
            if len(refs) < max_decode:
                decoded = decode_images(
                    self.model,
                    batch["images"],
                    self.tokenizer,
                    strategy=str(self.config.decode.get("strategy", "greedy")),
                    max_length=int(self.config.decode.get("max_length", 128)),
                    beam_size=int(self.config.decode.get("beam_size", 5)),
                    valid_hw=batch["valid_hw"],
                )
                refs.extend(batch["texts"])
                hyps.extend(decoded.texts)
        metrics = {"val_loss": sum(losses) / max(len(losses), 1), **reader_metrics(refs, hyps)}
        self.logger.log_metrics("val", self.step, metrics, epoch=epoch)
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None) -> None:
        ckpt_dir = Path(self.config.checkpoint.dir)
        for epoch in range(1, int(self.config.train.max_epochs) + 1):
            self.train_epoch(train_loader, epoch)
            metrics = self.validate(val_loader, epoch) if val_loader is not None else {"val_loss": float("inf")}
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
                tokenizer_id=self.tokenizer.vocab_id,
            )
            metric_name = str(self.config.checkpoint.get("best_metric", "cer"))
            value = metrics.get(metric_name, metrics.get("val_loss", float("inf")))
            better = value < self.best_metric if self.config.checkpoint.get("best_mode", "min") == "min" else value > self.best_metric
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
                    tokenizer_id=self.tokenizer.vocab_id,
                )
                save_checkpoint(
                    ckpt_dir / "best_infer.pt",
                    self.model,
                    config=self.config,
                    tokenizer_id=self.tokenizer.vocab_id,
                    mode="infer",
                )
