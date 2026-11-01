#!/usr/bin/env python3
"""Evaluate NLP field spans. Token accuracy is not the reported metric."""

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

from visual_nlp.datasets.collate import nlp_collate
from visual_nlp.datasets.nlp_dataset import NLPFinetuneDataset
from visual_nlp.nlp.model import NLPModel
from visual_nlp.nlp.normalization import normalize_field
from visual_nlp.nlp.tokenizer import NLPTokenizer
from visual_nlp.training.checkpoint import load_checkpoint
from visual_nlp.training.metrics import span_f1, value_exact
from visual_nlp.training.nlp_trainer import _bio_to_spans
from visual_nlp.utils.config import Config, load_config
from visual_nlp.utils.device import device_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nlp.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default="outputs/predictions/nlp_eval.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, model_name="nlp")
    device = device_from_config(Config({"train": config.finetune.train.to_dict()}))
    tokenizer = NLPTokenizer.from_config(config.tokenizer.path, vocab_size=int(config.tokenizer.vocab_size))
    labels = list(config.finetune.field_labels)
    dataset = NLPFinetuneDataset(
        args.manifest or config.finetune.val_manifest,
        tokenizer,
        int(config.model.max_length),
        labels,
        list(config.finetune.document_classes),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.finetune.train.batch_size),
        shuffle=False,
        collate_fn=partial(nlp_collate, pad_id=tokenizer.pad_id),
    )
    model = NLPModel.from_config(config, tokenizer.vocab_size, tokenizer.mask_id, tokenizer.pad_id).to(device)
    load_checkpoint(args.checkpoint, model, mode="infer", map_location=device)
    model.eval()
    gold, pred, gold_values, pred_values, rows = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                {
                    "input_ids": batch["input_ids"].to(device),
                    "padding_mask": batch["padding_mask"].to(device),
                    "token_boxes": batch.get("token_boxes", torch.zeros(1)).to(device) if "token_boxes" in batch else None,
                    "page_ids": batch["page_ids"].to(device) if "page_ids" in batch else None,
                },
                mode="finetune",
            )
            tags = outputs["field_logits"].argmax(-1).cpu()
            for labels_row, preds_row, mask, text, offsets, sample_id in zip(
                batch["field_labels"], tags, batch["loss_mask"], batch["texts"], batch["offsets"], batch["sample_ids"]
            ):
                gold_spans = _bio_to_spans(labels_row.tolist(), offsets, text, labels)
                pred_spans = _bio_to_spans(preds_row.tolist(), offsets, text, labels, valid=mask.tolist())
                gold.extend(gold_spans)
                pred.extend(pred_spans)
                for start, end, label in gold_spans:
                    gold_values.append(text[start:end])
                    match = next((item for item in pred_spans if item == (start, end, label)), None)
                    pred_values.append(text[match[0] : match[1]] if match else "")
                rows.append(
                    {
                        "id": sample_id,
                        "text": text,
                        "gold": [
                            {"span": [s, e], "type": lab, "raw": text[s:e], "normalized": normalize_field(lab, text[s:e]).value}
                            for s, e, lab in gold_spans
                        ],
                        "pred": [{"span": [s, e], "type": lab, "raw": text[s:e]} for s, e, lab in pred_spans],
                    }
                )
    metrics = {**span_f1(gold, pred), "value_exact": value_exact(gold_values, pred_values)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"metrics": metrics, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
