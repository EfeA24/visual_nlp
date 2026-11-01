"""Multi-page documents. PDF parsing is not a Transformer responsibility."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from visual_nlp.pipeline.image import ImagePipeline
from visual_nlp.pipeline.reading_order import assemble_reading_stream, link_label_values, order_regions
from visual_nlp.pipeline.schemas import DocumentResult, PageResult, PixelBox, ReadingResult, TextRegion


def pdf_text_regions(page, scale: float, page_index: int) -> list[TextRegion]:
    """Extract text rectangles and convert PDF coords to rendered image pixels."""
    try:
        textpage = page.get_textpage()
    except Exception:
        return []
    page_height = float(page.get_height())
    regions: list[TextRegion] = []
    try:
        count = textpage.count_rects()
    except Exception:
        return []
    for idx in range(count):
        left, bottom, right, top = textpage.get_rect(idx)
        piece = textpage.get_text_bounded(left=left, bottom=bottom, right=right, top=top).replace("\x00", "").strip()
        if not piece:
            continue
        regions.append(
            TextRegion(
                region_id=f"pdf-{page_index}-{idx}",
                page=page_index,
                box=PixelBox(left * scale, (page_height - top) * scale, right * scale, (page_height - bottom) * scale),
                score=1.0,
                source="pdf_text_layer",
                text=piece,
            )
        )
    return regions


def open_pages(path: str | Path, scale: float = 2.0) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(path))
        pages = []
        for index, page in enumerate(document):
            regions = pdf_text_regions(page, scale=scale, page_index=index)
            bitmap = page.render(scale=scale)
            pages.append(
                {
                    "page": index,
                    "image": bitmap.to_pil().convert("RGB"),
                    "regions": regions,
                    "source_path": str(path),
                }
            )
        return pages
    image = Image.open(path).convert("RGB")
    return [{"page": 0, "image": image, "regions": [], "source_path": str(path)}]


class DocumentPipeline:
    def __init__(self, image_pipeline: ImagePipeline, prefer_text_layer: bool = True) -> None:
        self.image_pipeline = image_pipeline
        self.prefer_text_layer = prefer_text_layer

    def run(
        self,
        path: str | Path,
        mode: str = "full",
        use_nlp: bool = True,
        document_id: str | None = None,
    ) -> DocumentResult:
        path = Path(path)
        pages = open_pages(path)
        collected: list[DocumentResult] = []
        warnings: list[str] = []
        for item in pages:
            regions = list(item.get("regions") or [])
            if self.prefer_text_layer and regions:
                collected.append(self._from_text_layer(item, regions, use_nlp, document_id or path.stem))
            else:
                try:
                    collected.append(
                        self.image_pipeline.run(
                            item["image"],
                            mode=mode,
                            use_nlp=use_nlp,
                            document_id=document_id or path.stem,
                            page=item["page"],
                        )
                    )
                except Exception as exc:
                    warnings.append(f"page {item['page']} failed: {exc}")
                    collected.append(
                        DocumentResult(
                            document_id=document_id or path.stem,
                            pages=[
                                PageResult(
                                    page=item["page"],
                                    width=item["image"].size[0],
                                    height=item["image"].size[1],
                                    warnings=[str(exc)],
                                )
                            ],
                            warnings=[str(exc)],
                        )
                    )
        return DocumentResult(
            document_id=document_id or path.stem,
            pages=[page for result in collected for page in result.pages],
            fields=[field for result in collected for field in result.fields],
            links=[link for result in collected for link in result.links],
            warnings=warnings + [warning for result in collected for warning in result.warnings],
            model_versions=collected[0].model_versions if collected else {},
        )

    def _from_text_layer(self, item: dict, regions: list[TextRegion], use_nlp: bool, document_id: str) -> DocumentResult:
        regions, order_warnings = order_regions(regions)
        readings = [
            ReadingResult(region.text, [], [], True, False, region.region_id, item["page"]) for region in regions
        ]
        page_result = PageResult(
            page=item["page"],
            width=item["image"].size[0],
            height=item["image"].size[1],
            regions=regions,
            readings=readings,
            warnings=order_warnings,
            used_text_layer=True,
        )
        fields, links = [], []
        if use_nlp:
            stream, owners, _ = assemble_reading_stream(readings, regions)
            fields = self.image_pipeline._run_nlp(
                stream,
                regions,
                item["page"],
                owners=owners,
                page_hw=(item["image"].size[1], item["image"].size[0]),
            )
            links = link_label_values(fields)
        return DocumentResult(document_id=document_id, pages=[page_result], fields=fields, links=links)
