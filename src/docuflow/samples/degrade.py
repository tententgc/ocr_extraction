"""Simulate a photocopied / phone-photographed scan.

Clean digital PDFs make OCR look better than it is. Running part of the corpus
through this filter keeps the demo honest.
"""

from __future__ import annotations

import random

from PIL import Image, ImageEnhance, ImageFilter


def degrade_image(image: Image.Image, intensity: float, seed: int = 0) -> Image.Image:
    """Return a scanned-looking copy of `image`.

    `intensity` runs 0 (untouched) to 1 (badly photocopied).
    """
    intensity = max(0.0, min(1.0, intensity))
    if intensity == 0.0:
        return image

    rng = random.Random(seed)
    result = image.convert("RGB")

    # Slight skew, as if the page were not square on the glass.
    angle = rng.uniform(-1.0, 1.0) * intensity
    result = result.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(248, 247, 244))

    # Paper tint and uneven exposure.
    result = ImageEnhance.Color(result).enhance(1.0 - 0.55 * intensity)
    result = ImageEnhance.Contrast(result).enhance(1.0 - 0.22 * intensity)
    result = ImageEnhance.Brightness(result).enhance(1.0 - 0.06 * intensity)

    # Optical softness from the scanner lens.
    result = result.filter(ImageFilter.GaussianBlur(radius=0.4 + 0.8 * intensity))

    # Sensor noise and toner speckle.
    noise_amount = int(60 * intensity)
    if noise_amount:
        pixels = result.load()
        width, height = result.size
        for _ in range(noise_amount * width * height // 900):
            x, y = rng.randrange(width), rng.randrange(height)
            shade = rng.randint(120, 205)
            pixels[x, y] = (shade, shade, shade)

    return result


def save_as_scan(image: Image.Image, path, intensity: float, seed: int = 0) -> None:
    """Degrade and write as a JPEG, including compression artefacts."""
    degraded = degrade_image(image, intensity, seed=seed)
    quality = int(92 - 42 * max(0.0, min(1.0, intensity)))
    degraded.save(path, format="JPEG", quality=max(35, quality))
