#!/usr/bin/env python3
"""Train the visual reader. Model layers are not defined here."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_nlp.datasets.collate import reader_collate
from visual_nlp.datasets.reader_dataset import ReaderDataset
from visual_nlp.datasets.transforms import ReaderTransform
from visual_nlp.reader.model import ReaderModel
from visual_nlp.reader.tokenizer import CharacterTokenizer
from visual_nlp.training.checkpoint import load_checkpoint
from visual_nlp.training.optim import build_optimizer, build_scheduler
from visual_nlp.training.reader_trainer import ReaderTrainer
from visual_nlp.utils.config import load_config, save_config
from visual_nlp.utils.device import device_from_config
from visual_nlp.utils.logging import RunLogger, new_run_id
from visual_nlp.utils.seed import seed_everything, seed_worker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reader.yaml")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, model_name="reader")
    seed_everything(int(config.train.seed))
    device = device_from_config(config)
    tokenizer = CharacterTokenizer.from_config(config.tokenizer.path, unk_policy=str(config.tokenizer.get("unk_policy", "unk")))
    transform_train = ReaderTransform(
        target_height=int(config.input.target_height),
        max_width=int(config.input.max_width),
        color=bool(config.input.color),
        pad_value=int(config.input.get("pad_value", 0)),
        pad_side=str(config.input.get("pad_side", "right")),
        augment=bool(config.data.get("augment", False)),
    )
    transform_val = ReaderTransform(
        target_height=int(config.input.target_height),
        max_width=int(config.input.max_width),
        color=bool(config.input.color),
        pad_value=int(config.input.get("pad_value", 0)),
        pad_side=str(config.input.get("pad_side", "right")),
        augment=False,
    )
    train_ds = ReaderDataset(config.data.train_manifest, tokenizer, transform_train)
    val_ds = ReaderDataset(config.data.val_manifest, tokenizer, transform_val)
    collate = partial(reader_collate, pad_id=tokenizer.pad_id)
    generator = torch.Generator()
    generator.manual_seed(int(config.train.seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config.train.batch_size),
        shuffle=True,
        num_workers=int(config.train.get("num_workers", 0)),
        collate_fn=collate,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config.train.batch_size),
        shuffle=False,
        num_workers=int(config.train.get("num_workers", 0)),
        collate_fn=collate,
    )
    model = ReaderModel.from_config(config, tokenizer.vocab_size, pad_id=tokenizer.pad_id)
    optimizer = build_optimizer(model, config.train)
    scheduler = build_scheduler(optimizer, config.train, max(len(train_loader), 1))
    if args.resume:
        load_checkpoint(args.resume, model, optimizer, scheduler, mode="resume", map_location=device)
    run_id = new_run_id("reader")
    logger = RunLogger("outputs/logs", run_id, "reader")
    save_config(config, Path(config.checkpoint.dir) / f"{run_id}.yaml")
    trainer = ReaderTrainer(model, optimizer, scheduler, tokenizer, config, device, logger)
    trainer.fit(train_loader, val_loader if len(val_ds) else None)


if __name__ == "__main__":
    main()
