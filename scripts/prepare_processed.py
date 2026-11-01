#!/usr/bin/env python3
"""Turn immutable raw sources into processed pages, line crops, and manifests.

This script does not train models. Crops keep a pointer back to the source page
and original-pixel boxes. The ASIDE PDF is English research text, so it is used
as a PDF/layout source, not as Turkish field-label training data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_nlp.pipeline.document import open_pages
from visual_nlp.pipeline.reading_order import order_regions
from visual_nlp.utils.config import find_project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw", help="File or directory of images/PDFs.")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--scale", type=float, default=2.0)
    return parser.parse_args()


def iter_sources(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files: list[Path] = []
    for suffix in (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        files.extend(sorted(path.rglob(f"*{suffix}")))
    return files


def crop_region(image, box) -> tuple[object, list[int]]:
    x1, y1, x2, y2 = [int(round(v)) for v in box.as_xyxy()]
    x1 = max(0, min(image.size[0] - 1, x1))
    y1 = max(0, min(image.size[1] - 1, y1))
    x2 = max(x1 + 1, min(image.size[0], x2))
    y2 = max(y1 + 1, min(image.size[1], y2))
    return image.crop((x1, y1, x2, y2)), [x1, y1, x2, y2]


def main() -> None:
    args = parse_args()
    root = find_project_root()
    source = Path(args.input)
    if not source.is_absolute():
        source = root / source
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    pages_dir = output / "pages"
    lines_dir = output / "lines"
    pages_dir.mkdir(parents=True, exist_ok=True)
    lines_dir.mkdir(parents=True, exist_ok=True)
    page_rows = []
    line_rows = []
    for path in iter_sources(source):
        source_id = path.stem
        opened = open_pages(path, scale=args.scale)
        for item in opened:
            page_name = f"{source_id}_p{item['page']:03d}.png"
            image_path = pages_dir / page_name
            item["image"].save(image_path)
            regions, warnings = order_regions(list(item.get("regions") or []))
            page_rows.append(
                {
                    "id": f"{source_id}_{item['page']}",
                    "source_id": source_id,
                    "source_path": str(path),
                    "page": item["page"],
                    "image_path": str(image_path.relative_to(root)).replace("\\", "/"),
                    "width": item["image"].size[0],
                    "height": item["image"].size[1],
                    "boxes_xyxy": [region.box.as_xyxy() for region in regions],
                    "region_texts": [region.text for region in regions],
                    "used_text_layer": bool(regions),
                    "warnings": warnings,
                }
            )
            for line_idx, region in enumerate(regions):
                crop, box = crop_region(item["image"], region.box)
                line_name = f"{source_id}_p{item['page']:03d}_l{line_idx:03d}.png"
                line_path = lines_dir / line_name
                crop.save(line_path)
                line_rows.append(
                    {
                        "id": f"{source_id}_{item['page']}_{line_idx}",
                        "source_id": source_id,
                        "source_page_id": f"{source_id}_{item['page']}",
                        "page": item["page"],
                        "image_path": str(line_path.relative_to(root)).replace("\\", "/"),
                        "page_image_path": str(image_path.relative_to(root)).replace("\\", "/"),
                        "box_xyxy": box,
                        "text": region.text,
                        "kind": "pdf_line" if region.source == "pdf_text_layer" else "crop",
                    }
                )
    (output / "pages.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in page_rows) + ("\n" if page_rows else ""),
        encoding="utf-8",
    )
    (output / "lines.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in line_rows) + ("\n" if line_rows else ""),
        encoding="utf-8",
    )
    print(f"pages={len(page_rows)} lines={len(line_rows)} output={output}")


if __name__ == "__main__":
    main()
