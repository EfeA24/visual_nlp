"""Structured training and inference logs. Prefer sample IDs over raw PII."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Mapping


def setup_logger(name: str, log_dir: str | Path | None = None, run_id: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_dir is not None:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        filename = f"{run_id or name}.log"
        file_handler = logging.FileHandler(path / filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


class RunLogger:
    def __init__(self, log_dir: str | Path, run_id: str, experiment: str) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.experiment = experiment
        self.logger = setup_logger(f"visual_nlp.{experiment}", self.log_dir, run_id)
        self.metrics_path = self.log_dir / f"{run_id}.jsonl"
        self._start = time.perf_counter()

    def log_metrics(self, split: str, step: int, metrics: Mapping[str, Any], **extra: Any) -> None:
        payload = {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "split": split,
            "step": step,
            "elapsed_sec": round(time.perf_counter() - self._start, 3),
            **{key: _jsonable(value) for key, value in metrics.items()},
            **{key: _jsonable(value) for key, value in extra.items()},
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        formatted = " ".join(f"{key}={payload[key]}" for key in metrics)
        self.logger.info("%s step=%s %s", split, step, formatted)

    def info(self, message: str, *args: Any) -> None:
        self.logger.info(message, *args)

    def warning(self, message: str, *args: Any) -> None:
        self.logger.warning(message, *args)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, float):
        return round(value, 6)
    return value


def new_run_id(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
