#!/usr/bin/env python3
"""Run the document pipeline on a PDF or image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_nlp.pipeline.document import DocumentPipeline
from visual_nlp.pipeline.image import build_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--mode", choices=["line", "full"], default="full")
    parser.add_argument("--reader-config", default="configs/reader.yaml")
    parser.add_argument("--reader-checkpoint", default=None)
    parser.add_argument("--detector-config", default="configs/detector.yaml")
    parser.add_argument("--detector-checkpoint", default=None)
    parser.add_argument("--nlp-config", default="configs/nlp.yaml")
    parser.add_argument("--nlp-checkpoint", default=None)
    parser.add_argument("--use-nlp", action="store_true")
    parser.add_argument("--output", default="outputs/predictions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_pipeline = build_pipeline(
        reader_config=args.reader_config,
        reader_checkpoint=args.reader_checkpoint,
        detector_config=args.detector_config if args.mode == "full" else None,
        detector_checkpoint=args.detector_checkpoint,
        nlp_config=args.nlp_config if args.use_nlp else None,
        nlp_checkpoint=args.nlp_checkpoint,
    )
    result = DocumentPipeline(image_pipeline).run(args.input, mode=args.mode, use_nlp=args.use_nlp)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{result.document_id}.json"
    target.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"document_id": result.document_id, "pages": len(result.pages), "warnings": result.warnings}, ensure_ascii=False))


if __name__ == "__main__":
    main()
