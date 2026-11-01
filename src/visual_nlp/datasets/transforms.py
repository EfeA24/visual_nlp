"""Image transforms. Horizontal flips are off by default because they invalidate labels."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass
class TransformOutput:
    image: torch.Tensor
    valid_hw: tuple[int, int]
    scale: float
    pad: tuple[int, int, int, int]
    original_size: tuple[int, int]


def _to_mode(image: Image.Image, color: bool) -> Image.Image:
    return image.convert("RGB" if color else "L")


def resize_keep_aspect(
    image: Image.Image,
    target_height: int,
    max_width: int,
) -> tuple[Image.Image, float]:
    width, height = image.size
    scale = target_height / max(height, 1)
    new_w = max(1, min(max_width, int(round(width * scale))))
    new_h = max(1, int(round(height * scale)))
    if new_h != target_height:
        new_h = target_height
    return image.resize((new_w, new_h), Image.Resampling.BILINEAR), scale


def pad_image(
    image: Image.Image,
    target_hw: tuple[int, int],
    value: int = 0,
    pad_side: str = "right",
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    target_h, target_w = target_hw
    width, height = image.size
    pad_h = max(0, target_h - height)
    pad_w = max(0, target_w - width)
    if pad_side == "both":
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
    else:
        left, top = 0, 0
        right, bottom = pad_w, pad_h
    padded = ImageOps.expand(image, border=(left, top, right, bottom), fill=value)
    return padded, (left, top, right, bottom)


def to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[None, ...]
    else:
        array = array.transpose(2, 0, 1)
    return torch.from_numpy(array)


def _augment(image: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.5:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.3:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.2)))
    if random.random() < 0.3:
        array = np.asarray(image).astype(np.int16)
        noise = np.random.randint(-18, 19, size=array.shape, dtype=np.int16)
        image = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))
    if random.random() < 0.2:
        image = image.rotate(random.uniform(-3, 3), resample=Image.Resampling.BILINEAR, fillcolor=255)
    return image


class ReaderTransform:
    def __init__(
        self,
        target_height: int,
        max_width: int,
        color: bool = True,
        pad_value: int = 0,
        pad_side: str = "right",
        augment: bool = False,
    ) -> None:
        self.target_height = target_height
        self.max_width = max_width
        self.color = color
        self.pad_value = pad_value
        self.pad_side = pad_side
        self.augment = augment

    def __call__(self, image: Image.Image) -> TransformOutput:
        original = image.size[::-1]
        image = _to_mode(image, self.color)
        if self.augment:
            image = _augment(image)
        image, scale = resize_keep_aspect(image, self.target_height, self.max_width)
        valid = (image.size[1], image.size[0])
        image, pad = pad_image(image, (self.target_height, image.size[0]), self.pad_value, self.pad_side)
        return TransformOutput(to_tensor(image), valid, scale, pad, original)


class DetectorTransform:
    def __init__(
        self,
        page_size: tuple[int, int],
        color: bool = True,
        pad_value: int = 0,
        keep_aspect: bool = True,
        augment: bool = False,
    ) -> None:
        self.page_h, self.page_w = page_size
        self.color = color
        self.pad_value = pad_value
        self.keep_aspect = keep_aspect
        self.augment = augment

    def __call__(self, image: Image.Image, boxes_xyxy: np.ndarray | None = None) -> tuple[TransformOutput, np.ndarray]:
        original_w, original_h = image.size
        image = _to_mode(image, self.color)
        if self.augment:
            image = _augment(image)
        if self.keep_aspect:
            scale = min(self.page_w / max(original_w, 1), self.page_h / max(original_h, 1))
            new_w = max(1, int(round(original_w * scale)))
            new_h = max(1, int(round(original_h * scale)))
            image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        else:
            scale = 1.0
            image = image.resize((self.page_w, self.page_h), Image.Resampling.BILINEAR)
        valid = (image.size[1], image.size[0])
        image, pad = pad_image(image, (self.page_h, self.page_w), self.pad_value, "right")
        boxes = np.zeros((0, 4), dtype=np.float32) if boxes_xyxy is None else boxes_xyxy.astype(np.float32).copy()
        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * (valid[1] / original_w) + pad[0]
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * (valid[0] / original_h) + pad[1]
        output = TransformOutput(to_tensor(image), valid, scale, pad, (original_h, original_w))
        return output, boxes
