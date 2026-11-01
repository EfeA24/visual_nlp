"""Save and restore training or inference packages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch

from visual_nlp.utils.config import Config, save_config
from visual_nlp.utils.seed import load_rng_state, rng_state


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int = 0,
    step: int = 0,
    best_metric: float | None = None,
    config: Config | dict | None = None,
    tokenizer_id: str | None = None,
    extra: dict[str, Any] | None = None,
    mode: Literal["train", "infer"] = "train",
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
        "tokenizer_id": tokenizer_id,
        "mode": mode,
    }
    if config is not None:
        payload["config"] = config.to_dict() if isinstance(config, Config) else dict(config)
        save_config(payload["config"], target.with_suffix(".yaml"))
    if mode == "train":
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        if scaler is not None:
            payload["scaler"] = scaler.state_dict()
        payload["rng"] = rng_state()
    if extra:
        payload["extra"] = extra
    torch.save(payload, target)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    mode: Literal["resume", "finetune", "infer"] = "infer",
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if mode == "resume":
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and "scheduler" in payload:
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and "scaler" in payload:
            scaler.load_state_dict(payload["scaler"])
        if "rng" in payload:
            load_rng_state(payload["rng"])
    return payload
