"""Architecture smoke test: models connect and a synthetic line can be forwarded."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_nlp.datasets.transforms import ReaderTransform
from visual_nlp.detection.model import DetectorModel
from visual_nlp.nlp.model import NLPModel
from visual_nlp.nlp.tokenizer import NLPTokenizer
from visual_nlp.pipeline.image import ImagePipeline
from visual_nlp.reader.model import ReaderModel
from visual_nlp.reader.tokenizer import CharacterTokenizer, default_charset
from visual_nlp.reader.visual_encoder import VisualEncoder
from visual_nlp.reader.character_decoder import CharacterDecoder
from visual_nlp.transformer.decoder import TransformerDecoder
from visual_nlp.utils.config import load_config


def _tiny_reader(vocab: int) -> ReaderModel:
    return ReaderModel(
        VisualEncoder(3, 16, 8, 32, 4, 1, 64, dropout=0.0),
        CharacterDecoder(vocab, 32, 4, 1, 64, max_len=129, dropout=0.0),
    )


def test_end_to_end_line_forward(tmp_path: Path) -> None:
    tokenizer = CharacterTokenizer(default_charset())
    image = Image.new("RGB", (160, 32), "white")
    ImageDraw.Draw(image).text((4, 4), "A1", fill="black", font=ImageFont.load_default())
    transform = ReaderTransform(32, 256, augment=False)
    prepared = transform(image)
    model = _tiny_reader(tokenizer.vocab_size)
    tokens = torch.tensor([tokenizer.decoder_inputs("A1")])
    logits = model(prepared.image.unsqueeze(0), tokens)
    assert logits.shape[-1] == tokenizer.vocab_size
    config = load_config(ROOT / "configs/reader.yaml")
    pipeline = ImagePipeline(
        reader=model,
        reader_tokenizer=tokenizer,
        reader_config=config,
        device=torch.device("cpu"),
    )
    result = pipeline.run(image, mode="line", document_id="smoke")
    assert result.pages[0].readings[0].text is not None
    assert result.model_versions["reader"] == "loaded"


def test_detector_and_nlp_forward() -> None:
    detector = DetectorModel(
        VisualEncoder(3, 16, 16, 32, 4, 1, 64, dropout=0.0),
        TransformerDecoder(32, 4, 1, 64, dropout=0.0),
        num_queries=4,
        d_model=32,
    )
    outputs = detector(torch.randn(1, 3, 64, 64))
    assert outputs["boxes"].shape[-1] == 4
    nlp_tok = NLPTokenizer.train_from_texts(["Adı Ahmet"], vocab_size=64, use_huggingface=False)
    nlp_cfg = load_config(ROOT / "configs/nlp.yaml", model_name="nlp")
    nlp_cfg.model.d_model = 32
    nlp_cfg.model.n_heads = 4
    nlp_cfg.model.n_layers = 1
    nlp_cfg.model.d_ff = 64
    nlp = NLPModel.from_config(nlp_cfg, nlp_tok.vocab_size, nlp_tok.mask_id, nlp_tok.pad_id)
    encoded = nlp_tok.encode("Adı Ahmet")
    batch = {
        "input_ids": torch.tensor([encoded.ids]),
        "padding_mask": torch.zeros(1, len(encoded.ids), dtype=torch.bool),
    }
    pre = nlp(batch, mode="pretrain")
    fine = nlp(batch, mode="finetune")
    assert pre["mlm_logits"].shape[-1] == nlp_tok.vocab_size
    assert "field_logits" in fine
