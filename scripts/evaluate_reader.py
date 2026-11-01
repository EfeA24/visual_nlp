#!/usr/bin/env python3
"""Evaluate a saved reader on a split. Do not tune on the test set."""

from __future__ import annotations

import argparse
import json
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
from visual_nlp.reader.decoding import decode_images
from visual_nlp.reader.model import ReaderModel
from visual_nlp.reader.tokenizer import CharacterTokenizer
from visual_nlp.training.checkpoint import load_checkpoint
from visual_nlp.training.metrics import reader_error_summary, reader_metrics
from visual_nlp.utils.config import load_config
from visual_nlp.utils.device import device_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reader.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=None, help="Defaults to validation manifest.")
    parser.add_argument("--output", default="outputs/predictions/reader_eval.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, model_name="reader")
    device = device_from_config(config)
    tokenizer = CharacterTokenizer.from_config(config.tokenizer.path)
    transform = ReaderTransform(
        target_height=int(config.input.target_height),
        max_width=int(config.input.max_width),
        color=bool(config.input.color),
        augment=False,
    )
    dataset = ReaderDataset(args.manifest or config.data.val_manifest, tokenizer, transform)
    loader = DataLoader(
        dataset,
        batch_size=int(config.train.batch_size),
        shuffle=False,
        collate_fn=partial(reader_collate, pad_id=tokenizer.pad_id),
    )
    model = ReaderModel.from_config(config, tokenizer.vocab_size, pad_id=tokenizer.pad_id).to(device)
    load_checkpoint(args.checkpoint, model, mode="infer", map_location=device)
    model.eval()
    refs, hyps, rows = [], [], []
    for batch in loader:
        images = batch["images"].to(device)
        valid_hw = batch["valid_hw"].to(device)
        decoded = decode_images(
            model,
            images,
            tokenizer,
            strategy=str(config.decode.get("strategy", "greedy")),
            max_length=int(config.decode.get("max_length", 128)),
            valid_hw=valid_hw,
        )
        for sample_id, ref, hyp, completed in zip(batch["sample_ids"], batch["texts"], decoded.texts, decoded.completed):
            refs.append(ref)
            hyps.append(hyp)
            rows.append({"id": sample_id, "reference": ref, "prediction": hyp, "completed": completed})
    metrics = {**reader_metrics(refs, hyps), **reader_error_summary(refs, hyps)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"metrics": metrics, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
