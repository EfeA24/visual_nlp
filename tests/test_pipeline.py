from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_nlp.pipeline.reading_order import order_regions
from visual_nlp.pipeline.schemas import PixelBox, TextRegion


def test_single_column_order() -> None:
    regions = [
        TextRegion("b", 0, PixelBox(10, 40, 80, 55), 0.9),
        TextRegion("a", 0, PixelBox(10, 10, 80, 25), 0.9),
        TextRegion("c", 0, PixelBox(90, 10, 140, 25), 0.8),
    ]
    ordered, warnings = order_regions(regions)
    assert [region.region_id for region in ordered] == ["a", "c", "b"]
    assert not any("Overlapping" in item for item in warnings)
