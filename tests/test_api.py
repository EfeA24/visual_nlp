from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw, ImageFont

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_nlp.api.app import create_app
from visual_nlp.pipeline.image import ImagePipeline
from visual_nlp.reader.character_decoder import CharacterDecoder
from visual_nlp.reader.model import ReaderModel
from visual_nlp.reader.tokenizer import CharacterTokenizer, default_charset
from visual_nlp.reader.visual_encoder import VisualEncoder
from visual_nlp.utils.config import load_config


def _tiny_reader(vocab: int) -> ReaderModel:
    return ReaderModel(
        VisualEncoder(3, 16, 8, 32, 4, 1, 64, dropout=0.0),
        CharacterDecoder(vocab, 32, 4, 1, 64, max_len=129, dropout=0.0),
    )


def _reader_pipeline() -> ImagePipeline:
    tokenizer = CharacterTokenizer(default_charset())
    config = load_config(ROOT / "configs/reader.yaml")
    return ImagePipeline(
        reader=_tiny_reader(tokenizer.vocab_size),
        reader_tokenizer=tokenizer,
        reader_config=config,
        device=torch.device("cpu"),
    )


def _png_bytes() -> bytes:
    image = Image.new("RGB", (160, 32), "white")
    ImageDraw.Draw(image).text((4, 4), "A1", fill="black", font=ImageFont.load_default())
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_reports_loaded_reader() -> None:
    with TestClient(create_app(pipeline=_reader_pipeline())) as client:
        response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["models"]["reader"] == "loaded"
    assert payload["models"]["detector"] == "absent"


def test_predict_line_returns_reading() -> None:
    with TestClient(create_app(pipeline=_reader_pipeline())) as client:
        response = client.post(
            "/predict",
            files={"file": ("line.png", _png_bytes(), "image/png")},
            data={"mode": "line"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["pages"][0]["readings"][0]["text"] is not None
    assert payload["model_versions"]["reader"] == "loaded"


def test_predict_full_without_detector_is_unavailable() -> None:
    with TestClient(create_app(pipeline=_reader_pipeline())) as client:
        response = client.post(
            "/predict",
            files={"file": ("page.png", _png_bytes(), "image/png")},
            data={"mode": "full"},
        )
    assert response.status_code == 503
    assert "detector" in response.json()["detail"].lower()
