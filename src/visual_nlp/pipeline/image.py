"""Run loaded models on one image. Models are constructed once by the caller."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from visual_nlp.datasets.transforms import DetectorTransform, ReaderTransform
from visual_nlp.detection.model import DetectorModel
from visual_nlp.detection.postprocess import invert_boxes_to_original, predictions_to_boxes
from visual_nlp.nlp.model import NLPModel
from visual_nlp.nlp.normalization import normalize_field
from visual_nlp.nlp.tokenizer import NLPTokenizer
from visual_nlp.pipeline.reading_order import (
    assemble_reading_stream,
    link_label_values,
    order_regions,
    regions_for_span,
)
from visual_nlp.pipeline.schemas import DocumentResult, ExtractedField, PageResult, PixelBox, ReadingResult, TextRegion
from visual_nlp.reader.decoding import decode_images
from visual_nlp.reader.model import ReaderModel
from visual_nlp.reader.tokenizer import CharacterTokenizer
from visual_nlp.training.checkpoint import load_checkpoint
from visual_nlp.training.nlp_trainer import _bio_to_spans
from visual_nlp.utils.config import load_config
from visual_nlp.utils.device import device_from_config


class ImagePipeline:
    def __init__(
        self,
        reader=None,
        reader_tokenizer=None,
        reader_config=None,
        detector=None,
        detector_config=None,
        nlp=None,
        nlp_tokenizer=None,
        nlp_config=None,
        device: torch.device | None = None,
    ) -> None:
        self.reader = reader
        self.reader_tokenizer = reader_tokenizer
        self.reader_config = reader_config
        self.detector = detector
        self.detector_config = detector_config
        self.nlp = nlp
        self.nlp_tokenizer = nlp_tokenizer
        self.nlp_config = nlp_config
        self.device = device or torch.device("cpu")

    def _read_crops(self, crops: list[Image.Image]) -> list[dict]:
        if not crops:
            return []
        transform = ReaderTransform(
            target_height=int(self.reader_config.input.target_height),
            max_width=int(self.reader_config.input.max_width),
            color=bool(self.reader_config.input.color),
            augment=False,
        )
        tensors = []
        valid = []
        for crop in crops:
            prepared = transform(crop)
            tensors.append(prepared.image)
            valid.append(torch.tensor(prepared.valid_hw))
        max_w = max(item.size(-1) for item in tensors)
        images = torch.stack(
            [torch.nn.functional.pad(item, (0, max_w - item.size(-1))) for item in tensors]
        ).to(self.device)
        valid_hw = torch.stack(valid).to(self.device)
        decoded = decode_images(
            self.reader,
            images,
            self.reader_tokenizer,
            strategy=str(self.reader_config.decode.get("strategy", "greedy")),
            max_length=int(self.reader_config.decode.get("max_length", 128)),
            valid_hw=valid_hw,
        )
        return [
            {
                "text": text,
                "token_ids": ids,
                "scores": scores,
                "completed": completed,
                "truncated": truncated,
            }
            for text, ids, scores, completed, truncated in zip(
                decoded.texts, decoded.token_ids, decoded.scores, decoded.completed, decoded.truncated
            )
        ]

    def _detect(self, image: Image.Image) -> tuple[list[TextRegion], list[Image.Image], list[str]]:
        page_h, page_w = [int(v) for v in self.detector_config.input.page_size]
        transform = DetectorTransform((page_h, page_w), color=bool(self.detector_config.input.color), augment=False)
        prepared, _ = transform(image, None)
        tensor = prepared.image.unsqueeze(0).to(self.device)
        valid_hw = torch.tensor([prepared.valid_hw], device=self.device)
        outputs = self.detector(tensor, valid_hw=valid_hw)
        decoded = predictions_to_boxes(
            outputs["logits"],
            outputs["boxes"],
            threshold=float(self.detector_config.predict.score_threshold),
            min_width=float(self.detector_config.box.min_width),
            min_height=float(self.detector_config.box.min_height),
            merge_iou=float(self.detector_config.predict.get("merge_iou", 0.0)),
            image_hw=(prepared.image.shape[-2], prepared.image.shape[-1]),
        )[0]
        original = invert_boxes_to_original(
            decoded["boxes_xyxy"].cpu(),
            prepared.pad,
            prepared.original_size,
            prepared.scale,
        )
        regions = []
        crops = []
        warnings = []
        width, height = image.size
        for idx, (box, score) in enumerate(zip(original.tolist(), decoded["scores"].tolist())):
            x1, y1, x2, y2 = [int(max(0, v)) for v in box]
            x2 = min(width, max(x1 + 1, x2))
            y2 = min(height, max(y1 + 1, y2))
            region = TextRegion(f"r{idx}", 0, PixelBox(x1, y1, x2, y2), float(score))
            regions.append(region)
            crops.append(image.crop((x1, y1, x2, y2)))
        if not regions:
            warnings.append("Detector returned no regions.")
        return regions, crops, warnings

    def _run_nlp(
        self,
        text: str,
        regions: list[TextRegion],
        page: int,
        owners: list[str | None] | None = None,
        page_hw: tuple[int, int] | None = None,
    ) -> list[ExtractedField]:
        if self.nlp is None or self.nlp_tokenizer is None:
            return []
        encoded = self.nlp_tokenizer.encode(text, max_length=int(self.nlp_config.model.max_length))
        lookup = {region.region_id: region for region in regions}
        height, width = page_hw or (1, 1)
        token_boxes = []
        for start, end in encoded.offsets:
            region_ids = regions_for_span(owners or [], start, end) if owners else []
            box_region = lookup.get(region_ids[0]) if region_ids else (regions[0] if regions else None)
            token_boxes.append(box_region.box.as_cxcywh_normalized(height, width) if box_region else [0.0, 0.0, 0.0, 0.0])
        batch = {
            "input_ids": torch.tensor([encoded.ids], device=self.device),
            "padding_mask": torch.zeros(1, len(encoded.ids), dtype=torch.bool, device=self.device),
            "token_boxes": torch.tensor([token_boxes], device=self.device, dtype=torch.float32),
            "page_ids": torch.full((1, len(encoded.ids)), page, dtype=torch.long, device=self.device),
        }
        outputs = self.nlp(batch, mode="finetune")
        if "field_logits" not in outputs:
            return []
        tags = outputs["field_logits"][0].argmax(-1).tolist()
        labels = list(self.nlp_config.finetune.field_labels)
        fields = []
        for start, end, label in _bio_to_spans(tags, encoded.offsets, text, labels):
            raw = text[start:end]
            normalized = normalize_field(label, raw)
            fields.append(
                ExtractedField(
                    field_type=label,
                    raw=raw,
                    normalized=normalized.value,
                    valid=normalized.valid,
                    uncertain=normalized.uncertain,
                    text_span=(start, end),
                    region_ids=regions_for_span(owners or [], start, end) if owners else [region.region_id for region in regions],
                    page=page,
                )
            )
        return fields

    def run(
        self,
        image: Image.Image | str | Path,
        mode: str = "line",
        use_nlp: bool = False,
        document_id: str = "image",
        page: int = 0,
    ) -> DocumentResult:
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        warnings: list[str] = []
        if mode == "line":
            if self.reader is None:
                raise RuntimeError("Reader is not loaded.")
            decoded = self._read_crops([image])[0]
            region = TextRegion("r0", page, PixelBox(0, 0, image.size[0], image.size[1]), 1.0, source="given")
            reading = ReadingResult(
                decoded["text"],
                decoded["token_ids"],
                decoded["scores"],
                decoded["completed"],
                decoded["truncated"],
                "r0",
                page,
            )
            regions, readings = [region], [reading]
            if not decoded["completed"]:
                warnings.append("Reading stopped at max length and is not marked complete.")
        elif mode == "full":
            if self.detector is None or self.reader is None:
                raise RuntimeError("Full-image mode requires detector and reader.")
            regions, crops, detect_warnings = self._detect(image)
            warnings.extend(detect_warnings)
            regions, order_warnings = order_regions(regions)
            warnings.extend(order_warnings)
            decoded = self._read_crops(crops)
            readings = [
                ReadingResult(
                    item["text"],
                    item["token_ids"],
                    item["scores"],
                    item["completed"],
                    item["truncated"],
                    region.region_id,
                    page,
                )
                for item, region in zip(decoded, regions)
            ]
        else:
            raise ValueError(f"Unknown image mode: {mode}")
        stream, owners, _ = assemble_reading_stream(readings, regions)
        fields = (
            self._run_nlp(stream, regions, page, owners=owners, page_hw=(image.size[1], image.size[0]))
            if use_nlp
            else []
        )
        page_result = PageResult(
            page=page,
            width=image.size[0],
            height=image.size[1],
            regions=regions,
            readings=readings,
            warnings=warnings,
        )
        return DocumentResult(
            document_id=document_id,
            pages=[page_result],
            fields=fields,
            links=link_label_values(fields),
            warnings=warnings,
            model_versions={
                "reader": "loaded" if self.reader is not None else "absent",
                "detector": "loaded" if self.detector is not None else "absent",
                "nlp": "loaded" if self.nlp is not None else "absent",
            },
        )


def build_pipeline(
    reader_config: str | None = None,
    reader_checkpoint: str | None = None,
    detector_config: str | None = None,
    detector_checkpoint: str | None = None,
    nlp_config: str | None = None,
    nlp_checkpoint: str | None = None,
    device: torch.device | None = None,
) -> ImagePipeline:
    reader = reader_tokenizer = reader_cfg = None
    detector = detector_cfg = None
    nlp = nlp_tokenizer = nlp_cfg = None
    if reader_config:
        reader_cfg = load_config(reader_config, model_name="reader")
        device = device or device_from_config(reader_cfg)
        reader_tokenizer = CharacterTokenizer.from_config(reader_cfg.tokenizer.path)
        reader = ReaderModel.from_config(reader_cfg, reader_tokenizer.vocab_size, reader_tokenizer.pad_id).to(device)
        if reader_checkpoint:
            load_checkpoint(reader_checkpoint, reader, mode="infer", map_location=device)
        reader.eval()
    if detector_config:
        detector_cfg = load_config(detector_config, model_name="detector")
        device = device or device_from_config(detector_cfg)
        detector = DetectorModel.from_config(detector_cfg).to(device)
        if detector_checkpoint:
            load_checkpoint(detector_checkpoint, detector, mode="infer", map_location=device)
        detector.eval()
    if nlp_config:
        nlp_cfg = load_config(nlp_config, model_name="nlp")
        device = device or device_from_config(nlp_cfg)
        nlp_tokenizer = NLPTokenizer.from_config(nlp_cfg.tokenizer.path, vocab_size=int(nlp_cfg.tokenizer.vocab_size))
        nlp = NLPModel.from_config(nlp_cfg, nlp_tokenizer.vocab_size, nlp_tokenizer.mask_id, nlp_tokenizer.pad_id).to(device)
        if nlp_checkpoint:
            load_checkpoint(nlp_checkpoint, nlp, mode="infer", map_location=device)
        nlp.eval()
    return ImagePipeline(
        reader=reader,
        reader_tokenizer=reader_tokenizer,
        reader_config=reader_cfg,
        detector=detector,
        detector_config=detector_cfg,
        nlp=nlp,
        nlp_tokenizer=nlp_tokenizer,
        nlp_config=nlp_cfg,
        device=device or torch.device("cpu"),
    )
