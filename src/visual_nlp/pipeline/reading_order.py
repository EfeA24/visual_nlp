"""Order text regions. Y-then-X is only a single-column heuristic."""

from __future__ import annotations

from visual_nlp.pipeline.schemas import ExtractedField, FieldLink, ReadingResult, TextRegion


def _vertical_overlap(a: TextRegion, b: TextRegion) -> float:
    top = max(a.box.y1, b.box.y1)
    bottom = min(a.box.y2, b.box.y2)
    height = min(a.box.y2 - a.box.y1, b.box.y2 - b.box.y1)
    if height <= 0:
        return 0.0
    return max(0.0, bottom - top) / height


def group_lines(regions: list[TextRegion], overlap: float = 0.5) -> list[list[TextRegion]]:
    remaining = sorted(regions, key=lambda region: (region.box.y1, region.box.x1))
    lines: list[list[TextRegion]] = []
    while remaining:
        anchor = remaining.pop(0)
        line = [anchor]
        kept = []
        for region in remaining:
            if _vertical_overlap(anchor, region) >= overlap:
                line.append(region)
            else:
                kept.append(region)
        remaining = kept
        line.sort(key=lambda region: region.box.x1)
        lines.append(line)
    lines.sort(key=lambda line: min(region.box.y1 for region in line))
    return lines


def looks_multicolumn(regions: list[TextRegion]) -> bool:
    if len(regions) < 4:
        return False
    xs = sorted((region.box.x1 + region.box.x2) / 2.0 for region in regions)
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    if not gaps:
        return False
    return max(gaps) > 2.5 * (sum(gaps) / len(gaps))


def overlapping_pairs(regions: list[TextRegion], iou_min: float = 0.3) -> list[tuple[str, str]]:
    flagged = []
    for i, left in enumerate(regions):
        for right in regions[i + 1 :]:
            ix1 = max(left.box.x1, right.box.x1)
            iy1 = max(left.box.y1, right.box.y1)
            ix2 = min(left.box.x2, right.box.x2)
            iy2 = min(left.box.y2, right.box.y2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area_l = max(0.0, left.box.x2 - left.box.x1) * max(0.0, left.box.y2 - left.box.y1)
            area_r = max(0.0, right.box.x2 - right.box.x1) * max(0.0, right.box.y2 - right.box.y1)
            union = area_l + area_r - inter
            if union > 0 and inter / union >= iou_min:
                flagged.append((left.region_id, right.region_id))
    return flagged


def order_regions(regions: list[TextRegion]) -> tuple[list[TextRegion], list[str]]:
    warnings: list[str] = []
    if looks_multicolumn(regions):
        warnings.append("Possible multi-column layout; y-then-x order may be wrong.")
    for left, right in overlapping_pairs(regions):
        warnings.append(f"Overlapping regions {left} and {right}.")
    ordered = [region for line in group_lines(regions) for region in line]
    return ordered, warnings


def ordered_text(readings: list[ReadingResult], regions: list[TextRegion]) -> str:
    return assemble_reading_stream(readings, regions)[0]


def assemble_reading_stream(
    readings: list[ReadingResult],
    regions: list[TextRegion],
) -> tuple[str, list[str | None], list[tuple[int, int, str]]]:
    """Join readings and remember which region owns each character."""
    order = {region.region_id: idx for idx, region in enumerate(regions)}
    sorted_reads = sorted(readings, key=lambda item: order.get(item.region_id, 10**9))
    parts: list[str] = []
    owners: list[str | None] = []
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for idx, item in enumerate(sorted_reads):
        if idx:
            parts.append("\n")
            owners.append(None)
            cursor += 1
        start = cursor
        parts.append(item.text)
        owners.extend([item.region_id] * len(item.text))
        cursor += len(item.text)
        spans.append((start, cursor, item.region_id))
    return "".join(parts), owners, spans


def regions_for_span(owners: list[str | None], start: int, end: int) -> list[str]:
    found: list[str] = []
    for idx in range(max(0, start), min(end, len(owners))):
        owner = owners[idx]
        if owner and owner not in found:
            found.append(owner)
    return found


_VALUE_TYPES = {"person_name", "date", "plate", "amount", "document_number"}


def link_label_values(fields: list[ExtractedField]) -> list[FieldLink]:
    """Pair a label with the nearest later value. Does not rewrite text."""
    labels = [field for field in fields if field.field_type == "label"]
    values = [field for field in fields if field.field_type in _VALUE_TYPES]
    used: set[tuple[int, int, str]] = set()
    links: list[FieldLink] = []
    for label in labels:
        candidates = [
            value
            for value in values
            if value.page == label.page
            and value.text_span[0] >= label.text_span[1]
            and (value.text_span[0], value.text_span[1], value.field_type) not in used
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda value: value.text_span[0] - label.text_span[1])
        used.add((best.text_span[0], best.text_span[1], best.field_type))
        links.append(
            FieldLink(
                head_type=label.field_type,
                tail_type=best.field_type,
                head_span=label.text_span,
                tail_span=best.text_span,
                method="reading_order",
            )
        )
    return links
