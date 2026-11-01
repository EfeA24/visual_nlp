#!/usr/bin/env python3
"""Fine-tune a pretrained NLP checkpoint on labeled document tasks."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_nlp.datasets.collate import nlp_collate
from visual_nlp.datasets.nlp_dataset import NLPFinetuneDataset
from visual_nlp.nlp.model import NLPModel
from visual_nlp.nlp.tokenizer import NLPTokenizer
from visual_nlp.training.checkpoint import load_checkpoint
from visual_nlp.training.nlp_trainer import NLPTrainer
from visual_nlp.training.optim import build_optimizer, build_scheduler
from visual_nlp.utils.config import Config, load_config, save_config
from visual_nlp.utils.device import device_from_config
from visual_nlp.utils.logging import RunLogger, new_run_id
from visual_nlp.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nlp.yaml")
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, model_name="nlp")
    seed_everything(int(config.finetune.train.seed))
    tokenizer = NLPTokenizer.from_config(config.tokenizer.path, vocab_size=int(config.tokenizer.vocab_size))
    field_labels = list(config.finetune.field_labels)
    doc_classes = list(config.finetune.document_classes)
    train_ds = NLPFinetuneDataset(config.finetune.train_manifest, tokenizer, int(config.model.max_length), field_labels, doc_classes)
    val_ds = NLPFinetuneDataset(config.finetune.val_manifest, tokenizer, int(config.model.max_length), field_labels, doc_classes)
    collate = partial(nlp_collate, pad_id=tokenizer.pad_id)
    train_loader = DataLoader(train_ds, batch_size=int(config.finetune.train.batch_size), shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=int(config.finetune.train.batch_size), shuffle=False, collate_fn=collate)
    model = NLPModel.from_config(config, tokenizer.vocab_size, tokenizer.mask_id, tokenizer.pad_id)
    ckpt = args.checkpoint or config.finetune.get("pretrain_checkpoint")
    if ckpt and Path(ckpt).exists():
        load_checkpoint(ckpt, model, mode="finetune")
    optimizer = build_optimizer(model, config.finetune.train)
    scheduler = build_scheduler(optimizer, config.finetune.train, max(len(train_loader), 1))
    device = device_from_config(Config({"train": config.finetune.train.to_dict()}))
    run_id = new_run_id("nlp_finetune")
    logger = RunLogger("outputs/logs", run_id, "nlp_finetune")
    save_config(config, Path(config.checkpoint.dir) / f"{run_id}.yaml")
    NLPTrainer(
        model,
        optimizer,
        scheduler,
        config,
        device,
        logger,
        mode="finetune",
        field_labels=field_labels,
    ).fit(train_loader, val_loader if len(val_ds) else None)


if __name__ == "__main__":
    main()
