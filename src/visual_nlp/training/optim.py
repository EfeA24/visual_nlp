"""Optimizer and scheduler factories used by training scripts."""

from __future__ import annotations

import math
from typing import Any

import torch

from visual_nlp.utils.config import Config, ConfigError


def build_optimizer(model: torch.nn.Module, train: Config | dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(train["learning_rate"])
    weight_decay = float(train.get("weight_decay", 0.0))
    name = str(train.get("optimizer", "adamw")).lower()
    params = [param for param in model.parameters() if param.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ConfigError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, train: Config | dict[str, Any], steps_per_epoch: int):
    name = str(train.get("scheduler", "none")).lower()
    epochs = int(train.get("max_epochs", 1))
    warmup = int(train.get("warmup_steps", 0))
    total = max(epochs * max(steps_per_epoch, 1), 1)
    if name in {"none", "constant"}:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if name == "cosine":

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step + 1) / max(warmup, 1)
            progress = (step - warmup) / max(total - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ConfigError(f"Unsupported scheduler: {name}")
