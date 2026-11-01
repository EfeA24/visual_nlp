"""Load, resolve, and validate YAML experiment configs.

Paths are resolved against the project root (directory that contains
``pyproject.toml``), not the process working directory.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class ConfigError(ValueError):
    """Invalid or incomplete configuration."""


class Config(dict):
    """Dict that also supports attribute access for nested mappings."""

    def __init__(self, data: Mapping[str, Any] | None = None, source_path: Path | None = None) -> None:
        super().__init__()
        self._source_path = source_path
        for key, value in (data or {}).items():
            super().__setitem__(key, _wrap(value))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self[name] = _wrap(value)

    def to_dict(self) -> dict[str, Any]:
        return {key: _unwrap(value) for key, value in self.items()}

    def get_path(self, dotted: str, default: Any = None) -> Any:
        current: Any = self
        for part in dotted.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current[part]
        return current


def _wrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value
    if isinstance(value, Mapping):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value.to_dict()
    if isinstance(value, list):
        return [_unwrap(item) for item in value]
    return value


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise ConfigError("Could not locate project root (missing pyproject.toml).")


def load_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path)
    if not yaml_path.is_absolute():
        yaml_path = find_project_root() / yaml_path
    if not yaml_path.exists():
        raise ConfigError(f"Config file not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"Config root must be a mapping: {yaml_path}")
    return dict(data)


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


_PATH_KEYS = {
    "tokenizer.path",
    "data.train_manifest",
    "data.val_manifest",
    "checkpoint.dir",
    "pretrain.corpus_manifest",
    "pretrain.checkpoint",
    "finetune.pretrain_checkpoint",
    "finetune.train_manifest",
    "finetune.val_manifest",
}


def _set_dotted(config: Config, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    current: Any = config
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], Config):
            current[part] = Config()
        current = current[part]
    current[parts[-1]] = value


def resolve_paths(config: Config, root: Path | None = None) -> Config:
    root = root or find_project_root()
    resolved = Config(config.to_dict(), source_path=config._source_path)
    for key in _PATH_KEYS:
        value = resolved.get_path(key)
        if isinstance(value, str):
            _set_dotted(resolved, key, str(resolve_path(value, root)))
    return resolved


def _require(config: Config, keys: Iterable[str]) -> None:
    missing = [key for key in keys if config.get_path(key) is None]
    if missing:
        raise ConfigError(f"Missing required config keys: {', '.join(missing)}")


def _validate_transformer_block(block: Mapping[str, Any], name: str) -> None:
    d_model = int(block["d_model"])
    n_heads = int(block["n_heads"])
    n_layers = int(block["n_layers"])
    d_ff = int(block["d_ff"])
    if d_model <= 0 or n_heads <= 0 or n_layers <= 0 or d_ff <= 0:
        raise ConfigError(f"{name} dimensions must be positive.")
    if d_model % n_heads != 0:
        raise ConfigError(f"{name}.d_model ({d_model}) must be divisible by n_heads ({n_heads}).")


def validate_reader_config(config: Config) -> None:
    _require(
        config,
        [
            "input.target_height",
            "input.max_width",
            "patch.height",
            "patch.width",
            "encoder.d_model",
            "encoder.n_heads",
            "decoder.d_model",
            "decoder.n_heads",
            "tokenizer.path",
            "train.batch_size",
            "train.learning_rate",
        ],
    )
    _validate_transformer_block(config.encoder, "encoder")
    _validate_transformer_block(config.decoder, "decoder")
    if int(config.encoder.d_model) != int(config.decoder.d_model):
        raise ConfigError("encoder.d_model and decoder.d_model must match.")
    if int(config.patch.height) <= 0 or int(config.patch.width) <= 0:
        raise ConfigError("patch sizes must be positive.")
    if int(config.input.target_height) % int(config.patch.height) != 0:
        raise ConfigError("input.target_height must be divisible by patch.height.")
    _validate_train_block(config.train)


def validate_detector_config(config: Config) -> None:
    _require(
        config,
        [
            "input.page_size",
            "patch.height",
            "patch.width",
            "encoder.d_model",
            "decoder.d_model",
            "queries.num_queries",
            "data.box_level",
            "train.batch_size",
        ],
    )
    _validate_transformer_block(config.encoder, "encoder")
    _validate_transformer_block(config.decoder, "decoder")
    if int(config.encoder.d_model) != int(config.decoder.d_model):
        raise ConfigError("encoder.d_model and decoder.d_model must match.")
    if config.data.box_level != "line":
        raise ConfigError("First detector target must use data.box_level=line.")
    if int(config.queries.num_queries) <= 0:
        raise ConfigError("queries.num_queries must be positive.")
    _validate_train_block(config.train)


def validate_nlp_config(config: Config) -> None:
    _require(config, ["tokenizer.path", "model.d_model", "model.n_heads", "model.n_layers", "model.d_ff"])
    _validate_transformer_block(config.model, "model")
    if config.get_path("pretrain.train") is not None:
        _validate_train_block(config.pretrain.train)
    if config.get_path("finetune.train") is not None:
        _validate_train_block(config.finetune.train)


def _validate_train_block(train: Mapping[str, Any]) -> None:
    if int(train["batch_size"]) <= 0:
        raise ConfigError("train.batch_size must be positive.")
    if float(train["learning_rate"]) <= 0:
        raise ConfigError("train.learning_rate must be positive.")
    optimizer = str(train.get("optimizer", "adamw")).lower()
    if optimizer not in {"adamw", "adam", "sgd"}:
        raise ConfigError(f"Unsupported optimizer: {optimizer}")
    if int(train.get("grad_accum", 1)) <= 0:
        raise ConfigError("train.grad_accum must be positive.")


_VALIDATORS = {
    "reader": validate_reader_config,
    "detector": validate_detector_config,
    "nlp": validate_nlp_config,
}


def load_config(path: str | Path, model_name: str | None = None) -> Config:
    yaml_path = Path(path)
    if not yaml_path.is_absolute():
        yaml_path = find_project_root() / yaml_path
    raw = load_yaml(yaml_path)
    config = Config(raw, source_path=yaml_path)
    name = model_name or config.get("model_name")
    if name in _VALIDATORS:
        _VALIDATORS[name](config)
    return resolve_paths(config)


def save_config(config: Config | Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict() if isinstance(config, Config) else deepcopy(dict(config))
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
