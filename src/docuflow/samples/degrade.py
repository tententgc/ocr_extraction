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


# --------------------------------------------------------------------------- #
# How documents actually arrive                                                 #
# --------------------------------------------------------------------------- #
# `degrade_image` above simulates a mildly tired photocopier, which is enough to
# stop a corpus looking artificially clean. These two go further, because the
# preprocessing stage exists for damage this side of a scanner and a corpus that
# never contains any cannot show whether it works.

def photograph(image: Image.Image, seed: int = 0) -> Image.Image:
    """A phone picture of a document lying on a desk.

    Skewed because nobody squares a page up before photographing it, lit from
    one side because there is one lamp, and slightly soft because a hand moves.

    Deliberately no perspective distortion. A page photographed from directly
    above has none, and correcting keystoning needs to find the page's corners -
    a different and much larger piece of work than straightening a scan. Putting
    it in the corpus would only prove the preprocessing does not do something it
    does not claim to do.
    """
    import numpy as np

    rng = random.Random(seed)
    result = image.convert("RGB")

    pixels = np.asarray(result, dtype=np.float32)
    height, width = pixels.shape[:2]
    rows, columns = np.mgrid[0:height, 0:width]
    # A lamp somewhere off one corner, falling away across the sheet.
    lamp_x, lamp_y = rng.uniform(0.0, 0.35), rng.uniform(0.0, 0.3)
    falloff = 0.45 + 0.55 * np.exp(
        -(((columns / width - lamp_x) ** 2 + (rows / height - lamp_y) ** 2) / 0.4)
    )
    result = Image.fromarray(np.clip(pixels * falloff[..., None], 0, 255).astype(np.uint8))

    result = result.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.8, 1.4)))
    # Held at an angle, and never a small enough one to ignore.
    angle = rng.choice([-1, 1]) * rng.uniform(2.2, 6.0)
    result = result.rotate(
        angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(64, 62, 60)
    )
    return _speckle(result, rng, amount=30)


def photocopy(image: Image.Image, seed: int = 0) -> Image.Image:
    """A third-generation copy: grey, flat and slightly crooked."""
    rng = random.Random(seed)
    result = image.convert("RGB")
    # The ink loses its blacks and the paper loses its whites, which is what
    # collapses the ink-to-paper separation the contrast metric measures.
    result = ImageEnhance.Contrast(result).enhance(0.34)
    result = ImageEnhance.Brightness(result).enhance(1.12)
    result = result.filter(ImageFilter.GaussianBlur(radius=0.9))
    result = result.rotate(
        rng.choice([-1, 1]) * rng.uniform(0.6, 2.0),
        resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(236, 234, 230),
    )
    return _speckle(result, rng, amount=45)


def _speckle(image: Image.Image, rng: random.Random, amount: int) -> Image.Image:
    """Toner dust and sensor noise."""
    pixels = image.load()
    width, height = image.size
    for _ in range(amount * width * height // 900):
        x, y = rng.randrange(width), rng.randrange(height)
        shade = rng.randint(110, 210)
        pixels[x, y] = (shade, shade, shade)
    return image
