"""Generate synthetic line images with manifests. Filenames are not labels."""

from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from visual_nlp.reader.tokenizer import CharacterTokenizer, default_charset
from visual_nlp.utils.config import find_project_root

TURKISH_WORDS = [
    "Adı", "Soyadı", "Tarih", "Tutar", "Plaka", "Belge", "Numara", "Toplam",
    "Ahmet", "Yılmaz", "İstanbul", "Ankara", "şirket", "fatura", "öğrenci",
    "kimlik", "doğum", "düzenleme", "ödeme", "lira", "gün", "ay", "yıl",
    "Türkiye", "görev", "başvuru", "onay", "imza", "kaşe", "vergi",
]
SENTENCES = [
    "Adı: Ahmet Yılmaz",
    "Plaka: 20 ABC 123",
    "Toplam: 1.250,00 TL",
    "Düzenleme tarihi 01.11.2026",
    "Belge No: 001234",
    "İstanbul Üniversitesi",
    "Ödeme tutarı 250,50 ₺",
]


@dataclass
class SyntheticSample:
    image_path: str
    text: str
    font: str
    font_size: int
    seed: int
    kind: str
    distortions: list[str]
    box_xyxy: list[int]


def iter_font_paths(extra_dir: Path | None = None) -> list[Path]:
    root = find_project_root()
    candidates: list[Path] = []
    search = [root / "data" / "fonts"]
    if extra_dir is not None:
        search.insert(0, Path(extra_dir))
    search.extend(
        [
            Path("C:/Windows/Fonts"),
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
            Path.home() / ".fonts",
            Path.home() / "Library" / "Fonts",
        ]
    )
    seen: set[str] = set()
    for folder in search:
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.suffix.lower() not in {".ttf", ".otf", ".ttc"}:
                continue
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(path)
    return candidates


def font_supports_text(font_path: Path, text: str) -> bool:
    try:
        from fontTools.ttLib import TTFont
    except Exception:
        return True
    try:
        font = TTFont(str(font_path), fontNumber=0)
        cmap = font.getBestCmap() or {}
        font.close()
    except Exception:
        return False
    needed = {ord(char) for char in text if not char.isspace()}
    return all(code in cmap for code in needed)


def load_pillow_font(font_path: Path, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(font_path), size=size)
    except OSError:
        return ImageFont.load_default()


def _random_date(rng: random.Random) -> str:
    return f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{rng.randint(1990, 2026)}"


def _random_amount(rng: random.Random) -> str:
    value = rng.randint(1, 999999)
    whole = f"{value:,}".replace(",", ".")
    return f"{whole},{rng.randint(0, 99):02d} TL"


def _random_plate(rng: random.Random) -> str:
    letters = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVYZ") for _ in range(rng.choice([2, 3])))
    return f"{rng.randint(1, 81):02d} {letters} {rng.randint(1, 999):03d}"


def _random_docno(rng: random.Random) -> str:
    return f"{rng.randint(0, 999999):06d}"


def sample_text(kind: str, rng: random.Random, tokenizer: CharacterTokenizer) -> str:
    if kind == "char":
        pool = [tok for tok in tokenizer.tokens[4:] if tok.strip()]
        return rng.choice(pool)
    if kind == "word":
        return rng.choice(TURKISH_WORDS)
    if kind == "sentence":
        return rng.choice(SENTENCES)
    if kind == "date":
        return _random_date(rng)
    if kind == "amount":
        return _random_amount(rng)
    if kind == "plate":
        return _random_plate(rng)
    if kind == "document_number":
        return _random_docno(rng)
    extra = "".join(rng.choice(string.ascii_letters + "0123456789") for _ in range(rng.randint(3, 8)))
    return f"{rng.choice(TURKISH_WORDS)} {extra}"


def render_text_image(text: str, font: ImageFont.ImageFont, rng: random.Random) -> tuple[Image.Image, list[int], list[str]]:
    dummy = Image.new("RGB", (8, 8), "white")
    draw = ImageDraw.Draw(dummy)
    box = draw.textbbox((0, 0), text, font=font)
    text_w = max(1, box[2] - box[0])
    text_h = max(1, box[3] - box[1])
    pad_x, pad_y = 8, 6
    image = Image.new("RGB", (text_w + 2 * pad_x, text_h + 2 * pad_y), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    origin = (pad_x - box[0], pad_y - box[1])
    draw.text(origin, text, fill=(20, 20, 20), font=font)
    text_box = [origin[0], origin[1], origin[0] + text_w, origin[1] + text_h]
    distortions: list[str] = []
    if rng.random() < 0.4:
        image = image.point(lambda p: max(0, min(255, int(p * rng.uniform(0.85, 1.1)))))
        distortions.append("brightness")
    return image, text_box, distortions


def choose_font(fonts: Iterable[Path], text: str, rng: random.Random) -> Path | None:
    compatible = [path for path in fonts if font_supports_text(path, text)]
    if not compatible:
        return None
    return rng.choice(compatible)


def generate_synthetic_lines(
    output_dir: str | Path,
    count: int,
    seed: int = 42,
    kinds: list[str] | None = None,
    tokenizer: CharacterTokenizer | None = None,
) -> list[SyntheticSample]:
    output = Path(output_dir)
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = tokenizer or CharacterTokenizer(default_charset())
    kinds = kinds or ["char", "word", "sentence", "date", "amount", "plate", "document_number"]
    fonts = iter_font_paths()
    rng = random.Random(seed)
    samples: list[SyntheticSample] = []
    attempts = 0
    while len(samples) < count and attempts < count * 20:
        attempts += 1
        kind = kinds[len(samples) % len(kinds)]
        text = sample_text(kind, rng, tokenizer)
        if tokenizer.unsupported_characters(text):
            continue
        font_path = choose_font(fonts, text, rng)
        if font_path is None:
            continue
        size = rng.randint(18, 36)
        font = load_pillow_font(font_path, size)
        sample_seed = seed + len(samples)
        image, box, distortions = render_text_image(text, font, random.Random(sample_seed))
        rel = f"images/{len(samples):06d}.png"
        image.save(images_dir / f"{len(samples):06d}.png")
        samples.append(
            SyntheticSample(
                image_path=rel,
                text=text,
                font=str(font_path),
                font_size=size,
                seed=sample_seed,
                kind=kind,
                distortions=distortions,
                box_xyxy=box,
            )
        )
    manifest_path = output / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for idx, sample in enumerate(samples):
            record = {
                "id": f"syn_{idx:06d}",
                "source_id": f"family_{sample.seed}",
                "image_path": sample.image_path,
                "text": sample.text,
                "font": sample.font,
                "font_size": sample.font_size,
                "seed": sample.seed,
                "kind": sample.kind,
                "distortions": sample.distortions,
                "box_xyxy": sample.box_xyxy,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return samples


def split_manifests(
    manifest_path: Path,
    output_dir: Path,
    prefix: str,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> None:
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    families: dict[str, list[dict]] = {}
    for row in rows:
        families.setdefault(str(row.get("source_id", row["id"])), []).append(row)
    keys = sorted(families)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_train = int(len(keys) * ratios[0])
    n_val = int(len(keys) * ratios[1])
    groups = {
        "train": keys[:n_train],
        "val": keys[n_train : n_train + n_val],
        "test": keys[n_train + n_val :],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, family_keys in groups.items():
        path = output_dir / f"{prefix}_{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for key in family_keys:
                for row in families[key]:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_detector_manifests(reader_prefix: str, output_dir: Path) -> None:
    """Copy line samples that already have boxes; do not invent detector labels."""
    output_dir = Path(output_dir)
    for split in ("train", "val", "test"):
        source = output_dir / f"{reader_prefix}_{split}.jsonl"
        if not source.exists():
            continue
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
        target = output_dir / f"detector_{split}.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                if "box_xyxy" not in row and "boxes_xyxy" not in row:
                    continue
                payload = dict(row)
                if "boxes_xyxy" not in payload:
                    payload["boxes_xyxy"] = [payload["box_xyxy"]]
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
