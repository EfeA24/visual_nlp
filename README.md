# Visual NLP

Three separately trained models share Transformer *code*, not weights:

| Component | Question | Output |
|-----------|----------|--------|
| Detector | Where is the text? | Line boxes |
| Reader | What characters are there? | A string |
| NLP | What do those strings mean? | Fields and relations |

The first working path is **cropped text line → character sequence**. A trained line reader is not a finished document-understanding system.

## Supported and not yet supported

Supported now:

- Cropped, mostly horizontal text lines
- Synthetic line generation and reader training
- Query-based line detection on full pages
- Turkish NLP pretraining (MLM) and field fine-tuning
- Image and PDF pipelines with optional text-layer extraction

Not a claim of this repository:

- Robust reading of every document
- Silent correction of OCR errors by the NLP model
- Reliable multi-column reading order
- Character-level boxes from cross-attention maps
- Rotated or strongly perspective text (planned later)

## Architecture

```
image/PDF → detector → crop lines → reader → reading order → NLP → normalization
                 ↘ cropped line can also skip detector
```

Common math lives in `src/visual_nlp/transformer/`. Reader, detector, and NLP each instantiate those layers with their own parameters.

Attention mask: `True` = blocked. Loss mask: `True` = this position contributes to the loss.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -e ".[dev]"
```

Install a CUDA build of PyTorch from [pytorch.org](https://pytorch.org) when a GPU is available. This project does not pin or rewrite the CUDA toolkit; it uses whatever PyTorch already sees.

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
pytest
```

## Data layout

- `data/raw/` — immutable sources (images, PDFs). Do not resize in place.
- `data/synthetic/` — generated images + `manifest.jsonl`
- `data/processed/` — derived page images, line crops, and manifests that point back to the source page and original-pixel boxes
- `data/splits/` — train/val/test manifests, split by source document
- `data/fonts/` — preferred fonts; system fonts are a fallback

Prepare processed artifacts from raw files:

```bash
python scripts/prepare_processed.py --input data/raw --output data/processed
```

The bundled ASIDE PDF is English research text. It is a PDF/layout source, not Turkish field-label training data. `prepare_processed.py` keeps the PDF text-layer rectangle granularity; those boxes are not mixed with handwritten line labels.

Filenames are not labels. Manifests bind images, text, boxes, fonts, and seeds.

Reader manifest fields: `id`, `source_id`, `image_path`, `text`.

Detector manifest fields: `id`, `source_id`, `image_path`, `boxes_xyxy` (line boxes only). A reader crop list without boxes is rejected.

## Training order

1. Generate synthetic lines:

```bash
python scripts/generate_synthetic.py --count 200 --seed 42
```

2. Train the reader (teacher-forcing loss and free-run CER are reported separately):

```bash
python scripts/train_reader.py --config configs/reader.yaml
python scripts/evaluate_reader.py --config configs/reader.yaml --checkpoint outputs/checkpoints/reader/best.pt
python scripts/evaluate_detector.py --config configs/detector.yaml --checkpoint outputs/checkpoints/detector/best.pt
python scripts/evaluate_nlp.py --config configs/nlp.yaml --checkpoint outputs/checkpoints/nlp/best.pt
```

3. Train the detector on line-box pages (not on reader crops):

```bash
python scripts/train_detector.py --config configs/detector.yaml
```

4. Pretrain then fine-tune NLP. Pretraining only learns masked-token representations.

```bash
python scripts/pretrain_nlp.py --config configs/nlp.yaml
python scripts/finetune_nlp.py --config configs/nlp.yaml
```

Checkpoints store weights, optimizer, scheduler, step, config, tokenizer id, and RNG state. Use `resume` to continue the same run and `finetune` to load weights under new settings. Tokenizer IDs must match the checkpoint; the reader never lowercases `I/İ/ı/i`.

## Prediction

```bash
python scripts/predict_image.py --input path/to/line.png --mode line --reader-checkpoint outputs/checkpoints/reader/best.pt
python scripts/predict_image.py --input path/to/page.png --mode full --reader-checkpoint ... --detector-checkpoint ... --draw-boxes
python scripts/predict_document.py --input path/to/doc.pdf --use-nlp --nlp-checkpoint outputs/checkpoints/nlp/best.pt
```

The same JSON is also available over HTTP:

```bash
pip install -e ".[api]"
python scripts/serve.py
```

`GET /health` reports which models are loaded. `POST /predict` accepts an image or PDF (`mode=line|full`, `use_nlp=true|false`) and returns the same `DocumentResult` payload as the CLI. Interactive docs: `http://127.0.0.1:8000/docs`.

Checkpoint paths can be set with `VISUAL_NLP_READER_CHECKPOINT`, `VISUAL_NLP_DETECTOR_CHECKPOINT`, and `VISUAL_NLP_NLP_CHECKPOINT` (and matching `*_CONFIG` variables). If a checkpoint env var is omitted, `outputs/checkpoints/<model>/best.pt` is used when that file exists.

Outputs keep raw text, completion flags, original-pixel boxes, normalized field values, and model versions. Truncated decodes are not marked complete. Confidence scores are not accuracy percentages.

## Known limits

- NLP must not silently rewrite `20 A8C 123` into `20 ABC 123`.
- Leading zeros stay strings (`001234` is not the integer 1234).
- Empty or failed pages are listed in `warnings`.
- Multi-column documents need a better reading-order model than y-then-x.
