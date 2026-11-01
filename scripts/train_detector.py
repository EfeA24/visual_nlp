#!/usr/bin/env python3
"""Train the line detector. Reader-only manifests without boxes are rejected."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_nlp.datasets.collate import detection_collate
from visual_nlp.datasets.detection_dataset import DetectionDataset
from visual_nlp.datasets.reader_dataset import load_jsonl
from visual_nlp.datasets.transforms import DetectorTransform
from visual_nlp.detection.model import DetectorModel
from visual_nlp.training.checkpoint import load_checkpoint
from visual_nlp.training.detection_trainer import DetectionTrainer
from visual_nlp.training.optim import build_optimizer, build_scheduler
from visual_nlp.utils.config import ConfigError, load_config, save_config
from visual_nlp.utils.device import device_from_config
from visual_nlp.utils.logging import RunLogger, new_run_id
from visual_nlp.utils.seed import seed_everything, seed_worker


def assert_detector_manifest(path: str | Path) -> None:
    rows = load_jsonl(path)
    if not rows:
        raise ConfigError(f"Detector manifest is empty: {path}")
    sample = rows[0]
    if "boxes_xyxy" not in sample and "box_xyxy" not in sample:
        raise ConfigError(
            f"Manifest {path} has no bounding boxes. Reader line crops cannot be used as detector data."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, model_name="detector")
    assert_detector_manifest(config.data.train_manifest)
    seed_everything(int(config.train.seed))
    device = device_from_config(config)
    page_h, page_w = [int(v) for v in config.input.page_size]
    transform = DetectorTransform(
        page_size=(page_h, page_w),
        color=bool(config.input.color),
        pad_value=int(config.input.get("pad_value", 0)),
        keep_aspect=bool(config.input.get("keep_aspect", True)),
        augment=bool(config.data.get("augment", False)),
    )
    val_transform = DetectorTransform((page_h, page_w), color=bool(config.input.color), augment=False)
    train_ds = DetectionDataset(config.data.train_manifest, transform)
    val_ds = DetectionDataset(config.data.val_manifest, val_transform)
    generator = torch.Generator()
    generator.manual_seed(int(config.train.seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config.train.batch_size),
        shuffle=True,
        num_workers=int(config.train.get("num_workers", 0)),
        collate_fn=detection_collate,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config.train.batch_size),
        shuffle=False,
        collate_fn=detection_collate,
    )
    model = DetectorModel.from_config(config)
    optimizer = build_optimizer(model, config.train)
    scheduler = build_scheduler(optimizer, config.train, max(len(train_loader), 1))
    if args.resume:
        load_checkpoint(args.resume, model, optimizer, scheduler, mode="resume", map_location=device)
    run_id = new_run_id("detector")
    logger = RunLogger("outputs/logs", run_id, "detector")
    save_config(config, Path(config.checkpoint.dir) / f"{run_id}.yaml")
    DetectionTrainer(model, optimizer, scheduler, config, device, logger).fit(
        train_loader, val_loader if len(val_ds) else None
    )


if __name__ == "__main__":
    main()
