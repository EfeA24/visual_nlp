#!/usr/bin/env python3
"""Evaluate a saved detector in original page-pixel coordinates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_nlp.datasets.collate import detection_collate
from visual_nlp.datasets.detection_dataset import DetectionDataset
from visual_nlp.datasets.transforms import DetectorTransform
from visual_nlp.detection.model import DetectorModel
from visual_nlp.detection.postprocess import invert_boxes_to_original, predictions_to_boxes
from visual_nlp.training.checkpoint import load_checkpoint
from visual_nlp.training.metrics import detection_prf
from visual_nlp.utils.config import load_config
from visual_nlp.utils.device import device_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--output", default="outputs/predictions/detector_eval.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, model_name="detector")
    device = device_from_config(config)
    page_h, page_w = [int(v) for v in config.input.page_size]
    transform = DetectorTransform((page_h, page_w), color=bool(config.input.color), augment=False)
    dataset = DetectionDataset(args.manifest or config.data.val_manifest, transform, root=args.root)
    loader = DataLoader(dataset, batch_size=int(config.train.batch_size), shuffle=False, collate_fn=detection_collate)
    model = DetectorModel.from_config(config).to(device)
    load_checkpoint(args.checkpoint, model, mode="infer", map_location=device)
    model.eval()
    pred_boxes, pred_scores, tgt_boxes, rows = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            valid_hw = batch["valid_hw"].to(device)
            outputs = model(images, valid_hw=valid_hw)
            decoded = predictions_to_boxes(
                outputs["logits"],
                outputs["boxes"],
                threshold=0.0,
                image_hw=(int(images.size(-2)), int(images.size(-1))),
            )
            for item, pad, orig, scale, target, sample_id in zip(
                decoded,
                batch["pads"],
                batch["original_size"],
                batch["scales"],
                batch["boxes_xyxy_original"],
                batch["sample_ids"],
            ):
                pred_orig = invert_boxes_to_original(item["boxes_xyxy"].cpu(), pad, orig, float(scale))
                pred_boxes.append(pred_orig)
                pred_scores.append(item["scores"].cpu())
                tgt_boxes.append(target)
                rows.append(
                    {
                        "id": sample_id,
                        "pred_xyxy": pred_orig.tolist(),
                        "target_xyxy": target.tolist(),
                        "scores": item["scores"].cpu().tolist(),
                    }
                )
    metrics = detection_prf(
        pred_boxes,
        pred_scores,
        tgt_boxes,
        score_threshold=float(config.predict.score_threshold),
        box_format="xyxy",
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"metrics": metrics, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
