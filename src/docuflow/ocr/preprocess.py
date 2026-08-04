"""Clean a page image before the OCR model reads it.

Documents do not arrive as clean digital PDFs. They arrive as a photo taken at a
desk under one lamp, a fax of a fax, a scan fed in crooked. Typhoon OCR copes
with a lot of that, but every degree of skew and every stop of lost contrast is
accuracy spent on decoding instead of reading, and the loss lands on exactly the
characters that matter - the digits in an amount.

Three rules shape this module, and they are the whole design.

**Measure first, correct second.** Every correction is gated on a measurement
that says the page needs it. Sharpening an already-crisp digital PDF adds ringing
around glyph edges and makes OCR *worse*; stretching the contrast of a page that
already spans black to white just clips ink. A clean page must come out of here
byte-identical to how it went in, and for most of the corpus it does.

**Never make it worse.** Each correction is applied, then re-measured. If the
metric it was supposed to improve did not improve, the correction is thrown away
and the previous image is kept. This costs one extra measurement per correction
and removes the entire class of failure where a heuristic misfires on an unusual
page and quietly degrades it.

**The human still sees the original.** The prepared image is written alongside
the rendered page, not over it. OCR reads the prepared one; the reviewer looking
at the document sees what actually arrived. A pipeline that silently shows people
a retouched version of the evidence is not auditable.

Deliberately not attempted: 180-degree rotation. Telling upside-down text from
right-way-up needs to know which way the glyphs face, which needs recognition -
by the time you can answer it you have already run the OCR. Guessing gets it
wrong half the time on the pages where it matters, so this reports the page as it
found it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from docuflow.ocr.base import PageImage

logger = logging.getLogger(__name__)

# --- what counts as "needs fixing" ----------------------------------------- #
# Calibrated against the sample corpus: clean renders sit far on the good side of
# every one of these, photographed and faxed pages far on the bad side.

# Skew beyond this many degrees is worth correcting. Below it the rotation costs
# more in resampling blur than it wins back in line alignment.
SKEW_WORTH_FIXING = 0.4
# Past this the page is not a crooked scan, it is a photograph of a document
# lying at an angle - or the estimate is wrong. Either way, leave it alone and
# say so rather than rotating a page into nonsense.
MAX_SKEW_DEGREES = 12.0

# Variance of the Laplacian, on luma scaled to 0..1. A crisply rendered page of
# text lands around 2e-3 and up; a soft phone photo an order of magnitude below.
SHARPNESS_FLOOR = 1.2e-3
# Sharpening a page already at or above the floor makes it worse, so the only
# question is how much a blurred one gets.
UNSHARP_RADIUS = 1.6
UNSHARP_PERCENT = 130
UNSHARP_THRESHOLD = 3

# How far the ink has to sit from the paper. Measured across the corpus: a
# rendered page lands near 0.32, a blurred one still manages 0.24 because blur
# moves edges rather than levels, and a faded photocopy drops to 0.07. The floor
# goes between the last two so that fading is caught and blur is left to the
# metric that actually describes it.
CONTRAST_FLOOR = 0.18
# Clipped at these percentiles when stretching, so a few black speckles or a
# blown-out highlight cannot define the range for the whole page.
STRETCH_LOW_PERCENTILE = 2.0
STRETCH_HIGH_PERCENTILE = 98.0

# How much the page background may drift across the sheet before it is treated
# as uneven lighting rather than as content. Measured on a 16x16 reduction.
ILLUMINATION_FLOOR = 0.18
# Radius of the blur that estimates the background. Must be far larger than a
# glyph and far smaller than the page, or it either erases the text or tracks
# nothing.
BACKGROUND_BLUR_FRACTION = 0.04

# Measurements run on a reduction, not the full page: skew and contrast are
# properties of the layout, and a 900px-tall copy carries them just as well for a
# fraction of the arithmetic.
MEASURE_DIM = 900
# Coarse sweep, then a fine one around the winner. Two passes cost ~40 rotations
# of a small image instead of the ~240 a single 0.1-degree sweep would need.
SKEW_COARSE_STEP = 1.0
SKEW_FINE_STEP = 0.1
# A 90-degree turn has to beat the page as it stands by this much before it is
# believed. Landscape tables score well both ways round, and turning one of those
# is a much worse outcome than leaving a sideways page for the OCR to handle.
ROTATION_MARGIN = 1.6

PREPARED_SUFFIX = "-prepared"


@dataclass(frozen=True)
class PageQuality:
    """What a page looks like, in numbers a threshold can be set against."""

    skew_degrees: float = 0.0
    sharpness: float = 0.0
    contrast: float = 0.0
    illumination_spread: float = 0.0

    @property
    def is_skewed(self) -> bool:
        return SKEW_WORTH_FIXING <= abs(self.skew_degrees) <= MAX_SKEW_DEGREES

    @property
    def is_blurred(self) -> bool:
        return self.sharpness < SHARPNESS_FLOOR

    @property
    def is_flat(self) -> bool:
        return self.contrast < CONTRAST_FLOOR

    @property
    def is_unevenly_lit(self) -> bool:
        return self.illumination_spread > ILLUMINATION_FLOOR

    @property
    def is_clean(self) -> bool:
        """Nothing here is worth touching."""
        return not (self.is_skewed or self.is_blurred or self.is_flat or self.is_unevenly_lit)

    def to_json(self) -> dict[str, Any]:
        return {
            "skew_degrees": round(self.skew_degrees, 2),
            "sharpness": round(self.sharpness, 5),
            "contrast": round(self.contrast, 3),
            "illumination_spread": round(self.illumination_spread, 3),
        }


@dataclass(frozen=True)
class PagePrep:
    """What was done to one page, and what it did to the numbers."""

    page_number: int
    before: PageQuality
    after: PageQuality
    actions: tuple[str, ...] = ()
    reverted: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    path: Path | None = None

    @property
    def changed(self) -> bool:
        return bool(self.actions)

    def to_json(self) -> dict[str, Any]:
        return {
            "page": self.page_number,
            "actions": list(self.actions),
            "reverted": list(self.reverted),
            "notes": list(self.notes),
            "before": self.before.to_json(),
            "after": self.after.to_json(),
        }


@dataclass(frozen=True)
class PrepReport:
    """The preprocessing stage's account of itself."""

    pages: tuple[PagePrep, ...] = ()
    duration_ms: int = 0
    enabled: bool = True

    @property
    def changed_pages(self) -> tuple[PagePrep, ...]:
        return tuple(page for page in self.pages if page.changed)

    @property
    def actions(self) -> tuple[str, ...]:
        """Every distinct correction applied, in a stable order."""
        seen: dict[str, None] = {}
        for page in self.pages:
            for action in page.actions:
                seen.setdefault(action, None)
        return tuple(seen)

    @property
    def summary(self) -> str:
        if not self.enabled:
            return "preprocessing off"
        if not self.pages:
            return "no pages to prepare"
        changed = self.changed_pages
        if not changed:
            return f"{len(self.pages)} page(s) already clean"
        return (
            f"{len(changed)}/{len(self.pages)} page(s) corrected: "
            + ", ".join(self.actions)
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "pages": [page.to_json() for page in self.pages],
            "corrected": len(self.changed_pages),
            "actions": list(self.actions),
            "summary": self.summary,
        }


# --------------------------------------------------------------------------- #
# Measurement                                                                   #
# --------------------------------------------------------------------------- #

def _luma(image: Image.Image, max_dim: int = MEASURE_DIM) -> np.ndarray:
    """Greyscale pixels in 0..1, reduced to something quick to work on."""
    grey = image.convert("L")
    longest = max(grey.size)
    if longest > max_dim:
        scale = max_dim / float(longest)
        grey = grey.resize(
            (max(1, int(grey.width * scale)), max(1, int(grey.height * scale))),
            Image.Resampling.BILINEAR,
        )
    return np.asarray(grey, dtype=np.float32) / 255.0


def sharpness(luma: np.ndarray) -> float:
    """Variance of the Laplacian: how abruptly the image changes at edges.

    Text is nothing but edges, so a page of it scores high when the glyphs are
    crisp and collapses towards zero as they smear together.
    """
    if luma.shape[0] < 3 or luma.shape[1] < 3:
        return 0.0
    middle = luma[1:-1, 1:-1]
    laplacian = (
        4.0 * middle
        - luma[:-2, 1:-1]
        - luma[2:, 1:-1]
        - luma[1:-1, :-2]
        - luma[1:-1, 2:]
    )
    return float(laplacian.var())


def contrast(luma: np.ndarray) -> float:
    """How far the ink sits from the paper it is printed on.

    Measured between the two populations rather than as a percentile spread over
    every pixel, because a page is mostly paper: on a clean invoice the 5th and
    95th percentiles are both white, and a percentile spread reports perfect
    contrast as none at all. Measured that way first, and every clean page in the
    corpus came back as needing rescue.

    A page with no ink returns 1.0 - there is nothing to read, which is a problem
    for OCR but not one more contrast would solve.
    """
    if luma.size == 0:
        return 1.0
    ink = _ink_mask(luma).astype(bool)
    paper = ~ink
    if ink.sum() < 50 or paper.sum() < 50:
        return 1.0
    return float(luma[paper].mean() - luma[ink].mean())


GRID = 16
# Within one cell, the brightness of paper rather than of the darkest thing in
# it. High enough to sit above the text, below the maximum so a single blown
# pixel cannot speak for the cell.
PAPER_PERCENTILE = 90.0


def illumination_spread(luma: np.ndarray) -> float:
    """How much the *paper's* brightness drifts from one corner to another.

    The page is cut into a grid and each cell reports how bright its paper is -
    not how bright the cell is on average. That distinction is the whole metric.
    Averaging each cell instead measures ink density: a cell of dense table rows
    averages far darker than an empty margin, and a perfectly lit invoice reports
    as though a lamp were shining on one half of it. Written that way first, and
    it condemned a clean synthetic page as unevenly lit while passing a real
    photograph, because real pages happen to have flatter text density.

    Reading the paper level per cell removes content from the measurement
    entirely: there is white space between glyphs everywhere on a page, so every
    cell can say what its background looks like, and only the lighting moves it.
    """
    if luma.size == 0:
        return 0.0
    height, width = luma.shape
    if height < GRID or width < GRID:
        return 0.0

    # Trim to a whole number of cells so the grid can be taken by reshaping,
    # which is far quicker than slicing 256 windows out one at a time.
    cell_h, cell_w = height // GRID, width // GRID
    trimmed = luma[: cell_h * GRID, : cell_w * GRID]
    cells = trimmed.reshape(GRID, cell_h, GRID, cell_w).transpose(0, 2, 1, 3)
    paper = np.percentile(cells.reshape(GRID * GRID, -1), PAPER_PERCENTILE, axis=1)

    low, high = np.percentile(paper, [10.0, 90.0])
    return float(high - low)


def _line_score(binary: np.ndarray) -> float:
    """How sharply the ink separates into horizontal lines.

    Sum the ink in each row, then measure how violently that profile jumps from
    row to row. When the text lines are level, every line is a plateau and every
    gap a trough, and the jumps between them are large. Tilt the page and the
    lines smear across each other until the profile is nearly flat.

    Squared differences rather than the variance of the profile: variance also
    rewards a page that is simply darker at the top, which is a lighting
    artefact rather than an alignment one.
    """
    profile = binary.sum(axis=1)
    if profile.size < 2:
        return 0.0
    return float(np.square(np.diff(profile)).sum())


def _ink_mask(luma: np.ndarray) -> np.ndarray:
    """Ink as 1.0, paper as 0.0.

    Thresholded relative to the page's own statistics rather than at a fixed
    level, because a fax and a laser print disagree completely about what "dark"
    means and both have to work.
    """
    threshold = luma.mean() - 0.5 * luma.std()
    return (luma < threshold).astype(np.float32)


def estimate_skew(image: Image.Image) -> float:
    """Degrees the page is rotated by, positive meaning anticlockwise.

    Coarse sweep for the neighbourhood, fine sweep for the value. Returns 0.0
    when nothing scores meaningfully better than leaving it alone, which is the
    honest answer for a page with too little text to tell.
    """
    luma = _luma(image, MEASURE_DIM)
    ink = _ink_mask(luma)
    if ink.sum() < 50:            # a blank or near-blank page has no lines to level
        return 0.0

    source = Image.fromarray((ink * 255).astype(np.uint8))

    def score(angle: float) -> float:
        if angle == 0.0:
            return _line_score(ink)
        rotated = source.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
        return _line_score(np.asarray(rotated, dtype=np.float32) / 255.0)

    coarse = np.arange(-MAX_SKEW_DEGREES, MAX_SKEW_DEGREES + SKEW_COARSE_STEP, SKEW_COARSE_STEP)
    best_angle = max(coarse, key=score)

    fine = np.arange(
        best_angle - SKEW_COARSE_STEP, best_angle + SKEW_COARSE_STEP + SKEW_FINE_STEP, SKEW_FINE_STEP
    )
    best_angle = max(fine, key=score)

    # The rotation that levels the lines is the negative of the page's skew.
    return float(round(-best_angle, 2))


def measure(image: Image.Image) -> PageQuality:
    """Everything the corrections are gated on, for one page."""
    luma = _luma(image)
    return PageQuality(
        skew_degrees=estimate_skew(image),
        sharpness=sharpness(luma),
        contrast=contrast(luma),
        illumination_spread=illumination_spread(luma),
    )


def _remeasure(image: Image.Image, previous: PageQuality, *, skew: bool = False) -> PageQuality:
    """Re-read the cheap metrics after a correction.

    Skew estimation is the expensive one and only the rotation changes it, so
    everything else carries the angle forward rather than paying for it again.
    """
    luma = _luma(image)
    return PageQuality(
        skew_degrees=estimate_skew(image) if skew else previous.skew_degrees,
        sharpness=sharpness(luma),
        contrast=contrast(luma),
        illumination_spread=illumination_spread(luma),
    )


# --------------------------------------------------------------------------- #
# Corrections                                                                   #
# --------------------------------------------------------------------------- #

def _paper_colour(image: Image.Image) -> int:
    """The page's background level, used to fill in what rotation exposes.

    Black corners on a rotated scan are not neutral: they change the page's
    measured contrast and give the OCR model a hard edge to find text along.
    """
    luma = _luma(image, 200)
    return int(round(float(np.percentile(luma, 90)) * 255))


def deskew(image: Image.Image, degrees: float) -> Image.Image:
    """Rotate the page level, growing the canvas so no text is cut off."""
    fill = _paper_colour(image)
    return image.rotate(
        -degrees,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(fill, fill, fill) if image.mode == "RGB" else fill,
    )


def flatten_illumination(image: Image.Image) -> Image.Image:
    """Divide the page by its own background, removing shadow and lamp fall-off.

    The background is what is left after a blur wide enough to erase every glyph
    but narrow enough to follow the lighting. Dividing by it turns a page that is
    bright on one side and grey on the other into one even sheet, which is the
    single largest win available on a photographed document.
    """
    grey = image.convert("L")
    radius = max(8.0, min(grey.size) * BACKGROUND_BLUR_FRACTION)
    background = np.asarray(grey.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    foreground = np.asarray(grey, dtype=np.float32)

    # Guard the divisor: a genuinely black region would otherwise divide by ~0
    # and explode into noise.
    ratio = foreground / np.maximum(background, 1.0)
    flattened = np.clip(ratio * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(flattened).convert(image.mode)


def stretch_contrast(image: Image.Image) -> Image.Image:
    """Map the page's actual ink-to-paper range onto the full scale."""
    grey = np.asarray(image.convert("L"), dtype=np.float32)
    low, high = np.percentile(grey, [STRETCH_LOW_PERCENTILE, STRETCH_HIGH_PERCENTILE])
    if high - low < 1.0:          # a uniform page has no range to stretch
        return image
    stretched = np.clip((grey - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    return Image.fromarray(stretched).convert(image.mode)


def sharpen(image: Image.Image) -> Image.Image:
    """Unsharp mask, sized for text rather than for photographs."""
    return image.filter(
        ImageFilter.UnsharpMask(
            radius=UNSHARP_RADIUS, percent=UNSHARP_PERCENT, threshold=UNSHARP_THRESHOLD
        )
    )


def _upright_score(image: Image.Image) -> float:
    """How much this page looks like level lines of text."""
    return _line_score(_ink_mask(_luma(image)))


def detect_quarter_turn(image: Image.Image) -> int:
    """0 or 90: whether the page reads better turned a quarter to the right.

    Lines of text make a strongly periodic horizontal profile and almost none at
    all when the page is on its side, so the two orientations are easy to tell
    apart - for a page of prose. A landscape table scores respectably both ways,
    which is why the turn has to win by a clear margin before it is taken.

    Only a quarter turn, and only one direction: distinguishing 90 from 270 needs
    to know which way the glyphs face, and that is a question for the OCR model,
    not for a projection profile.
    """
    upright = _upright_score(image)
    turned = _upright_score(image.rotate(90, expand=True))
    if upright <= 0:
        return 0
    return 90 if turned > upright * ROTATION_MARGIN else 0


# --------------------------------------------------------------------------- #
# The stage                                                                     #
# --------------------------------------------------------------------------- #

def prepare_image(image: Image.Image, quality: PageQuality | None = None) -> tuple[Image.Image, PagePrep]:
    """Apply every correction the page's own measurements ask for.

    Returns the prepared image and an account of what happened, including the
    corrections that were tried and thrown away for not helping.
    """
    before = quality if quality is not None else measure(image)
    current = image
    state = before
    actions: list[str] = []
    reverted: list[str] = []
    notes: list[str] = []

    if detect_quarter_turn(image):
        # Reported, not corrected. Which quarter turn puts it right needs to know
        # which way the glyphs face, and this stage cannot see that - turning it
        # the wrong way leaves the OCR model reading upside down, which is worse
        # than leaving it on its side. So the page goes through as it is and the
        # trace says why, which is what a reviewer needs in order to rescan it.
        notes.append("page appears to be sideways; it was not turned, because which way is up "
                     "cannot be told from the layout alone")

    if abs(before.skew_degrees) > MAX_SKEW_DEGREES:
        notes.append(
            f"page appears rotated by {before.skew_degrees:.1f}°, which is too far to be a "
            f"crooked scan; left as it is"
        )
    elif before.is_skewed:
        candidate = deskew(current, before.skew_degrees)
        measured = _remeasure(candidate, state, skew=True)
        # The test is whether the lines actually ended up straighter, not whether
        # the estimate said they would.
        if abs(measured.skew_degrees) < abs(state.skew_degrees):
            current, state = candidate, measured
            actions.append(f"deskewed {before.skew_degrees:+.1f}°")
        else:
            reverted.append("deskew")

    if state.is_unevenly_lit:
        candidate = flatten_illumination(current)
        measured = _remeasure(candidate, state)
        if measured.illumination_spread < state.illumination_spread:
            current, state = candidate, measured
            actions.append("evened out the lighting")
        else:
            reverted.append("flatten")

    if state.is_flat:
        candidate = stretch_contrast(current)
        measured = _remeasure(candidate, state)
        if measured.contrast > state.contrast:
            current, state = candidate, measured
            actions.append("raised the contrast")
        else:
            reverted.append("contrast")

    # Sharpening last: deskewing and stretching both change how sharp the page
    # measures, so asking before them would answer the wrong question.
    if state.is_blurred:
        candidate = sharpen(current)
        measured = _remeasure(candidate, state)
        if measured.sharpness > state.sharpness:
            current, state = candidate, measured
            actions.append("sharpened")
        else:
            reverted.append("sharpen")

    return current, PagePrep(
        page_number=0,
        before=before,
        after=state,
        actions=tuple(actions),
        reverted=tuple(reverted),
        notes=tuple(notes),
    )


def _prepared_path(page: PageImage) -> Path:
    return page.path.with_name(f"{page.path.stem}{PREPARED_SUFFIX}{page.path.suffix}")


def prepare_pages(pages: list[PageImage], enabled: bool = True) -> tuple[list[PageImage], PrepReport]:
    """Prepare every rendered page for OCR.

    Pages that need nothing are returned untouched and no second file is
    written - the overwhelmingly common case for a digital PDF, and the reason
    this stage costs almost nothing on a clean corpus.

    A page that cannot be prepared is returned as it was. Preprocessing is an
    optimisation; failing it must never fail the document.
    """
    if not enabled or not pages:
        return pages, PrepReport(enabled=enabled)

    started = time.perf_counter()
    out: list[PageImage] = []
    preps: list[PagePrep] = []

    for page in pages:
        try:
            with Image.open(page.path) as opened:
                opened.load()
                prepared, prep = prepare_image(opened.convert("RGB"))
        except Exception as exc:  # noqa: BLE001 - a page we cannot clean is still readable
            logger.warning("could not prepare page %s: %s", page.page_number, exc)
            out.append(page)
            preps.append(
                PagePrep(
                    page_number=page.page_number,
                    before=PageQuality(),
                    after=PageQuality(),
                    notes=(f"could not be prepared: {exc}",),
                )
            )
            continue

        prep = replace(prep, page_number=page.page_number)
        if not prep.changed:
            out.append(page)
            preps.append(replace(prep, path=page.path))
            continue

        target = _prepared_path(page)
        prepared.save(target, format="JPEG", quality=95)
        out.append(
            PageImage(
                page_number=page.page_number,
                path=target,
                width=prepared.width,
                height=prepared.height,
                embedded_text=page.embedded_text,
            )
        )
        preps.append(replace(prep, path=target))

    return out, PrepReport(
        pages=tuple(preps),
        duration_ms=int((time.perf_counter() - started) * 1000),
        enabled=True,
    )


def describe_angle(degrees: float) -> str:
    """A skew angle as a person would say it."""
    if abs(degrees) < SKEW_WORTH_FIXING:
        return "straight"
    direction = "clockwise" if degrees < 0 else "anticlockwise"
    return f"{abs(degrees):.1f}° {direction}"
