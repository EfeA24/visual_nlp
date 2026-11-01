#!/usr/bin/env python3
"""Pretrain the NLP encoder with MLM. This does not teach document fields."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_nlp.datasets.collate import nlp_collate
from visual_nlp.datasets.nlp_dataset import NLPPretrainDataset
from visual_nlp.datasets.reader_dataset import load_jsonl
from visual_nlp.nlp.model import NLPModel
from visual_nlp.nlp.tokenizer import NLPTokenizer
from visual_nlp.training.nlp_trainer import NLPTrainer
from visual_nlp.training.optim import build_optimizer, build_scheduler
from visual_nlp.utils.config import Config, load_config, save_config
from visual_nlp.utils.device import device_from_config
from visual_nlp.utils.logging import RunLogger, new_run_id
from visual_nlp.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nlp.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, model_name="nlp")
    seed_everything(int(config.pretrain.train.seed))
    texts = [row["text"] for row in load_jsonl(config.pretrain.corpus_manifest)]
    tokenizer = NLPTokenizer.from_config(config.tokenizer.path, texts=texts, vocab_size=int(config.tokenizer.vocab_size))
    dataset = NLPPretrainDataset(config.pretrain.corpus_manifest, tokenizer, int(config.model.max_length))
    loader = DataLoader(
        dataset,
        batch_size=int(config.pretrain.train.batch_size),
        shuffle=True,
        collate_fn=partial(nlp_collate, pad_id=tokenizer.pad_id),
    )
    model = NLPModel.from_config(config, tokenizer.vocab_size, tokenizer.mask_id, tokenizer.pad_id)
    optimizer = build_optimizer(model, config.pretrain.train)
    scheduler = build_scheduler(optimizer, config.pretrain.train, max(len(loader), 1))
    device = device_from_config(Config({"train": config.pretrain.train.to_dict()}))
    run_id = new_run_id("nlp_pretrain")
    logger = RunLogger("outputs/logs", run_id, "nlp_pretrain")
    save_config(config, Path(config.pretrain.checkpoint) / f"{run_id}.yaml")
    NLPTrainer(model, optimizer, scheduler, config, device, logger, mode="pretrain").fit(loader, loader if len(dataset) else None)


if __name__ == "__main__":
    main()
