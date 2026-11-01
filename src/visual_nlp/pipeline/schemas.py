"""Shared data records. External coordinates are original page pixels."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PixelBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def as_xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def as_cxcywh_normalized(self, height: int, width: int) -> list[float]:
        return [
            ((self.x1 + self.x2) / 2.0) / max(width, 1),
            ((self.y1 + self.y2) / 2.0) / max(height, 1),
            (self.x2 - self.x1) / max(width, 1),
            (self.y2 - self.y1) / max(height, 1),
        ]


@dataclass
class TextRegion:
    region_id: str
    page: int
    box: PixelBox
    score: float
    source: str = "detector"
    text: str = ""


@dataclass
class ReadingResult:
    text: str
    token_ids: list[int]
    scores: list[float]
    completed: bool
    truncated: bool
    region_id: str
    page: int


@dataclass
class ExtractedField:
    field_type: str
    raw: str
    normalized: Any
    valid: bool
    uncertain: bool
    text_span: tuple[int, int]
    region_ids: list[str]
    page: int


@dataclass
class PageResult:
    page: int
    width: int
    height: int
    regions: list[TextRegion] = field(default_factory=list)
    readings: list[ReadingResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    used_text_layer: bool = False


@dataclass
class FieldLink:
    head_type: str
    tail_type: str
    head_span: tuple[int, int]
    tail_span: tuple[int, int]
    method: str


@dataclass
class DocumentResult:
    document_id: str
    pages: list[PageResult]
    fields: list[ExtractedField] = field(default_factory=list)
    links: list[FieldLink] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
