from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def _crop_center(image: Image.Image, keep_ratio: float) -> Image.Image:
    width, height = image.size
    new_w = max(1, int(width * keep_ratio))
    new_h = max(1, int(height * keep_ratio))
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    return image.crop((left, top, left + new_w, top + new_h)).resize((width, height), Image.BILINEAR)


def _resize_compress(image: Image.Image, scale: float = 0.55) -> Image.Image:
    width, height = image.size
    small = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.BILINEAR)
    return small.resize((width, height), Image.BILINEAR)


def _watermark(image: Image.Image, text: str = "COPY") -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    width, height = out.size
    box = (width // 10, height // 2 - 16, width * 9 // 10, height // 2 + 20)
    draw.rectangle(box, fill=(255, 255, 255, 80))
    draw.text((box[0] + 8, box[1] + 6), text, fill=(0, 0, 0, 140))
    return out


def _bg_color_change(image: Image.Image) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    tint = np.array([12.0, -8.0, 18.0], dtype=np.float32)
    arr = np.clip(arr + tint, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def apply_attack(image: Image.Image, attack: str) -> Image.Image:
    """Apply one named query/training attack.

    Names use Python-safe identifiers. CLI also accepts legacy hyphen names by
    replacing ``-`` with ``_``.
    """

    attack = attack.replace("-", "_")
    if attack == "original":
        return image.copy()
    if attack == "rotate_m5":
        return image.rotate(-5, resample=Image.BILINEAR, expand=False)
    if attack == "rotate_m10":
        return image.rotate(-10, resample=Image.BILINEAR, expand=False)
    if attack == "rotate5":
        return image.rotate(5, resample=Image.BILINEAR, expand=False)
    if attack == "rotate15":
        return image.rotate(15, resample=Image.BILINEAR, expand=False)
    if attack == "rotate30":
        return image.rotate(30, resample=Image.BILINEAR, expand=False)
    if attack == "rotate40":
        return image.rotate(40, resample=Image.BILINEAR, expand=False)
    if attack == "crop5":
        return _crop_center(image, 0.95)
    if attack == "crop10":
        return _crop_center(image, 0.90)
    if attack == "crop20":
        return _crop_center(image, 0.80)
    if attack == "crop40":
        return _crop_center(image, 0.60)
    if attack == "bright_light":
        return ImageEnhance.Brightness(image).enhance(1.15)
    if attack == "bright":
        return ImageEnhance.Brightness(image).enhance(1.35)
    if attack == "heavy_bright":
        return ImageEnhance.Brightness(image).enhance(1.75)
    if attack == "flip_h":
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    if attack == "flip_v":
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    if attack == "flip":
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    if attack == "resize_compress":
        return _resize_compress(image, 0.50)
    if attack == "resize":
        return _resize_compress(image, 0.75)
    if attack == "watermark_text":
        return _watermark(image, "TEXT")
    if attack == "watermark_asset":
        return _watermark(image, "ASSET")
    if attack == "mix_rotate10_bright":
        return ImageEnhance.Brightness(apply_attack(image, "rotate_m10")).enhance(1.35)
    if attack == "mix_crop20_watermark":
        return _watermark(apply_attack(image, "crop20"), "ASSET")
    if attack == "bg_color_change":
        return _bg_color_change(image)
    if attack == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=1.5))
    if attack == "contrast":
        return ImageEnhance.Contrast(image).enhance(1.55)
    raise ValueError(f"Unknown attack: {attack}")


@dataclass
class AttackChooser:
    names: tuple[str, ...]
    seed: int = 20260519

    def choose(self, key: str, epoch: int, slot: int = 0) -> str:
        rng = random.Random(f"{self.seed}:{epoch}:{slot}:{key}")
        return rng.choice(tuple(self.names))
