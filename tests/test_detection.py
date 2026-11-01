from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_nlp.detection.model import DetectorModel
from visual_nlp.detection.postprocess import invert_boxes_to_original, predictions_to_boxes
from visual_nlp.reader.visual_encoder import VisualEncoder
from visual_nlp.training.losses import detector_loss, match_boxes
from visual_nlp.training.metrics import detection_prf
from visual_nlp.transformer.decoder import TransformerDecoder


def test_hungarian_matches_closest_box() -> None:
    logits = torch.tensor([[0.0, 4.0], [4.0, 0.0], [0.0, 3.0]])
    pred = torch.tensor([[0.2, 0.2, 0.2, 0.1], [0.8, 0.8, 0.1, 0.1], [0.21, 0.22, 0.2, 0.1]])
    target = torch.tensor([[0.2, 0.2, 0.2, 0.1]])
    rows, cols = match_boxes(logits, pred, target, 1.0, 5.0, 2.0)
    assert rows.tolist() == [0]
    assert cols.tolist() == [0]


def test_detector_shapes_and_empty_page() -> None:
    encoder = VisualEncoder(3, 16, 16, 32, 4, 1, 64, dropout=0.0)
    decoder = TransformerDecoder(32, 4, 1, 64, dropout=0.0)
    model = DetectorModel(encoder, decoder, num_queries=8, d_model=32)
    images = torch.randn(2, 3, 64, 64)
    outputs = model(images)
    assert outputs["logits"].shape == (2, 8, 2)
    assert outputs["boxes"].shape == (2, 8, 4)
    empty = [torch.zeros(0, 4), torch.tensor([[0.5, 0.5, 0.2, 0.1]])]
    losses = detector_loss(outputs["logits"], outputs["boxes"], empty)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])


def test_postprocess_inverts_pad_and_scale() -> None:
    boxes = torch.tensor([[10.0, 10.0, 30.0, 20.0]])
    inverted = invert_boxes_to_original(boxes, pad=(4, 2, 0, 0), original_hw=(100, 200), scale=0.5)
    assert torch.allclose(inverted, torch.tensor([[12.0, 16.0, 52.0, 36.0]]))
    logits = torch.tensor([[[0.0, 5.0], [5.0, 0.0]]])
    pred_boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.1], [0.1, 0.1, 0.001, 0.001]]])
    result = predictions_to_boxes(logits, pred_boxes, threshold=0.5, min_width=0.01, min_height=0.005)[0]
    assert result["boxes_cxcywh"].shape[0] == 1


def test_detection_prf_uses_original_xyxy() -> None:
    pred = [torch.tensor([[10.0, 10.0, 30.0, 20.0]], dtype=torch.float32)]
    scores = [torch.tensor([0.9])]
    target = [torch.tensor([[12.0, 11.0, 29.0, 21.0]], dtype=torch.float32)]
    metrics = detection_prf(pred, scores, target, score_threshold=0.5, box_format="xyxy")
    assert metrics["tp"] == 1
    assert metrics["f1"] == 1.0
