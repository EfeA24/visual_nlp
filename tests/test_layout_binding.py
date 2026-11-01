from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_nlp.pipeline.reading_order import assemble_reading_stream, link_label_values, regions_for_span
from visual_nlp.pipeline.schemas import ExtractedField, PixelBox, ReadingResult, TextRegion


def test_field_binds_only_own_region() -> None:
    regions = [
        TextRegion("r0", 0, PixelBox(0, 0, 40, 10), 1.0, text="Adı:"),
        TextRegion("r1", 0, PixelBox(50, 0, 140, 10), 1.0, text="Ahmet"),
    ]
    readings = [
        ReadingResult("Adı:", [], [], True, False, "r0", 0),
        ReadingResult("Ahmet", [], [], True, False, "r1", 0),
    ]
    text, owners, _ = assemble_reading_stream(readings, regions)
    assert text == "Adı:\nAhmet"
    start = text.index("Ahmet")
    assert regions_for_span(owners, start, start + 5) == ["r1"]
    assert regions_for_span(owners, 0, 4) == ["r0"]


def test_label_value_link_keeps_raw() -> None:
    fields = [
        ExtractedField("label", "Adı", "Adı", True, False, (0, 3), ["r0"], 0),
        ExtractedField("person_name", "Ahmet", "Ahmet", True, False, (5, 10), ["r1"], 0),
    ]
    links = link_label_values(fields)
    assert len(links) == 1
    assert links[0].tail_type == "person_name"
    assert fields[1].raw == "Ahmet"
