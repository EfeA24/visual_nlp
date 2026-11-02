"""Resolve API config and checkpoint paths from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from visual_nlp.utils.config import find_project_root


@dataclass(frozen=True)
class APISettings:
    reader_config: str | None
    reader_checkpoint: str | None
    detector_config: str | None
    detector_checkpoint: str | None
    nlp_config: str | None
    nlp_checkpoint: str | None


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    if value is None or value.strip() == "":
        return None
    return value


def _resolve(value: str | None, root: Path) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return str(path)


def _checkpoint(env_name: str, default_relative: str, root: Path) -> str | None:
    explicit = _env(env_name)
    if explicit is not None:
        return _resolve(explicit, root)
    candidate = root / default_relative
    return str(candidate) if candidate.is_file() else None


def load_settings(root: Path | None = None) -> APISettings:
    root = root or find_project_root()
    nlp_config = _env("VISUAL_NLP_NLP_CONFIG")
    nlp_checkpoint = _checkpoint("VISUAL_NLP_NLP_CHECKPOINT", "outputs/checkpoints/nlp/best.pt", root)
    if nlp_config is None and nlp_checkpoint is not None:
        nlp_config = "configs/nlp.yaml"
    return APISettings(
        reader_config=_resolve(_env("VISUAL_NLP_READER_CONFIG", "configs/reader.yaml"), root),
        reader_checkpoint=_checkpoint("VISUAL_NLP_READER_CHECKPOINT", "outputs/checkpoints/reader/best.pt", root),
        detector_config=_resolve(_env("VISUAL_NLP_DETECTOR_CONFIG", "configs/detector.yaml"), root),
        detector_checkpoint=_checkpoint("VISUAL_NLP_DETECTOR_CHECKPOINT", "outputs/checkpoints/detector/best.pt", root),
        nlp_config=_resolve(nlp_config, root),
        nlp_checkpoint=nlp_checkpoint
    )
