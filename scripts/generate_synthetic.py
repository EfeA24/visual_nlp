#!/usr/bin/env python3
"""Generate synthetic line images and split manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_nlp.datasets.synthetic import generate_synthetic_lines, split_manifests, write_detector_manifests
from visual_nlp.reader.tokenizer import CharacterTokenizer
from visual_nlp.utils.config import find_project_root
from visual_nlp.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic OCR lines.")
    parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", default="configs/vocabs/reader_charset.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = find_project_root()
    seed_everything(args.seed)
    tokenizer = CharacterTokenizer.from_config(root / args.tokenizer)
    output = root / args.output
    samples = generate_synthetic_lines(output, count=args.count, seed=args.seed, tokenizer=tokenizer)
    splits = root / "data" / "splits"
    split_manifests(output / "manifest.jsonl", splits, prefix="reader", seed=args.seed)
    write_detector_manifests("reader", splits)
    print(f"generated={len(samples)} output={output}")


if __name__ == "__main__":
    main()
