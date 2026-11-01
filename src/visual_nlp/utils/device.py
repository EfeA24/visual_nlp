"""Device selection. This module does not install or change CUDA."""

from __future__ import annotations

from typing import Any

import torch

from visual_nlp.utils.config import Config, ConfigError


def resolve_device(
    requested: str = "cuda",
    allow_cpu_fallback: bool = True,
    gpu_index: int | None = None,
) -> torch.device:
    requested = requested.lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested not in {"cuda", "gpu"}:
        raise ConfigError(f"Unknown device request: {requested}")
    if not torch.cuda.is_available():
        if allow_cpu_fallback:
            return torch.device("cpu")
        raise ConfigError("CUDA was requested but is not available.")
    index = 0 if gpu_index is None else gpu_index
    if index >= torch.cuda.device_count():
        raise ConfigError(f"GPU index {index} is out of range ({torch.cuda.device_count()} devices).")
    return torch.device(f"cuda:{index}")


def device_from_config(config: Config) -> torch.device:
    train = config.get("train") or config.get_path("pretrain.train") or config.get_path("finetune.train") or {}
    requested = str(train.get("device", config.get("device", "cuda")))
    allow = bool(train.get("allow_cpu_fallback", True))
    gpu_index = train.get("gpu_index")
    return resolve_device(requested, allow_cpu_fallback=allow, gpu_index=gpu_index)


def move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(value, device) for value in obj)
    return obj


def mixed_precision_enabled(config: Config) -> bool:
    train = config.get("train") or {}
    value = train.get("mixed_precision", False)
    if value in {False, None, "none", "fp32"}:
        return False
    if value in {True, "amp", "fp16"}:
        if not torch.cuda.is_available():
            return False
        return True
    raise ConfigError(f"Unsupported mixed_precision setting: {value}")
