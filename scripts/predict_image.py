#!/usr/bin/env python3
"""Predict a line image or a full page. Models are loaded once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_nlp.pipeline.image import build_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image file or directory.")
    parser.add_argument("--mode", choices=["line", "full"], default="line")
    parser.add_argument("--reader-config", default="configs/reader.yaml")
    parser.add_argument("--reader-checkpoint", default=None)
    parser.add_argument("--detector-config", default="configs/detector.yaml")
    parser.add_argument("--detector-checkpoint", default=None)
    parser.add_argument("--nlp-config", default=None)
    parser.add_argument("--nlp-checkpoint", default=None)
    parser.add_argument("--use-nlp", action="store_true")
    parser.add_argument("--output", default="outputs/predictions")
    parser.add_argument("--draw-boxes", action="store_true")
    return parser.parse_args()


def iter_images(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})
    return [path]


def main() -> None:
    args = parse_args()
    pipeline = build_pipeline(
        reader_config=args.reader_config,
        reader_checkpoint=args.reader_checkpoint,
        detector_config=args.detector_config if args.mode == "full" else None,
        detector_checkpoint=args.detector_checkpoint,
        nlp_config=args.nlp_config if args.use_nlp else None,
        nlp_checkpoint=args.nlp_checkpoint,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for image_path in iter_images(Path(args.input)):
        result = pipeline.run(image_path, mode=args.mode, use_nlp=args.use_nlp, document_id=image_path.stem)
        target = output / f"{image_path.stem}.json"
        target.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        raw_text = "\n".join(reading.text for page in result.pages for reading in page.readings)
        (output / f"{image_path.stem}.txt").write_text(raw_text, encoding="utf-8")
        if args.draw_boxes:
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            for page in result.pages:
                for region in page.regions:
                    draw.rectangle(region.box.as_xyxy(), outline=(220, 40, 40), width=2)
            image.save(output / f"{image_path.stem}_boxes.png")
        print(f"{image_path.name}: {raw_text!r}")


if __name__ == "__main__":
    main()
