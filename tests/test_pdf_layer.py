from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_nlp.pipeline.document import open_pages, pdf_text_regions


def _aside_pdf() -> Path | None:
    candidates = [
        ROOT / "data" / "raw" / "documents" / "aside_v0.1.pdf",
        Path.home() / ".cursor" / "projects" / "c-Users-DELL-Desktop-visual-nlp" / "attachments" / "01b46d25-0afa-4aff-9bf1-945d6566f7de" / "ASIDE_v0.1_arXiv.pdf",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def test_pdf_text_layer_has_boxes() -> None:
    pdf = _aside_pdf()
    if pdf is None:
        return
    pages = open_pages(pdf, scale=2.0)
    assert pages
    first = pages[0]
    assert first["image"].size[0] > 0
    assert first["regions"]
    region = first["regions"][0]
    assert region.box.x2 > region.box.x1
    assert region.box.y2 > region.box.y1
    assert region.text
    assert region.source == "pdf_text_layer"
    width, height = first["image"].size
    assert 0 <= region.box.x1 <= width
    assert 0 <= region.box.y1 <= height
