"""Cleaning the page before the model reads it.

The risk in this stage is not that it fails to help - it is that it helps
uninvited. A sharpening filter run over a crisp digital PDF rings the glyph
edges; a contrast stretch over a page that already spans black to white clips
ink off the thin strokes. Both make OCR worse while reporting that something was
improved, and neither shows up anywhere except in a wrong digit weeks later.

So most of what is tested here is restraint: a clean page must come out
untouched, and a correction that does not measurably improve the thing it exists
to improve must be thrown away.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from docuflow.ocr.base import PageImage
from docuflow.ocr.preprocess import (
    CONTRAST_FLOOR,
    MAX_SKEW_DEGREES,
    SHARPNESS_FLOOR,
    PageQuality,
    contrast,
    estimate_skew,
    illumination_spread,
    measure,
    prepare_image,
    prepare_pages,
    sharpness,
    _luma,
)

WIDTH, HEIGHT = 900, 1200
# Drawn at this multiple and scaled back down, so the glyph edges are
# antialiased the way a real render's are. Hard-edged marks measure several times
# sharper than any real page and no amount of blur brings them under the floor,
# which makes every threshold in this file untestable.
SUPERSAMPLE = 3


def page() -> Image.Image:
    """A sheet of text-like marks: ragged lines, word gaps, even margins."""
    image = Image.new("L", (WIDTH * SUPERSAMPLE, HEIGHT * SUPERSAMPLE), 255)
    draw = ImageDraw.Draw(image)
    random = np.random.default_rng(7)

    y = 70 * SUPERSAMPLE
    while y < (HEIGHT - 70) * SUPERSAMPLE:
        x = 70 * SUPERSAMPLE
        right = (WIDTH - 70) * SUPERSAMPLE - int(random.integers(0, 90)) * SUPERSAMPLE
        while x < right:
            glyph = int(random.integers(5, 11)) * SUPERSAMPLE
            draw.rectangle([x, y, x + glyph, y + 11 * SUPERSAMPLE], fill=30)
            x += glyph + 4 * SUPERSAMPLE
            if random.random() < 0.18:
                x += 6 * SUPERSAMPLE          # a word gap
        y += 26 * SUPERSAMPLE

    return image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).convert("RGB")


def blurred(image: Image.Image, radius: float = 3.2) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius))


def page_dir(root: Path) -> Path:
    """A directory holding nothing but the page, so "what was written" is answerable.

    Named so that it cannot collide with the data directories the settings
    fixture already lays down inside `tmp_path` - `pages` among them.
    """
    directory = root / "render-out"
    directory.mkdir(exist_ok=True)
    return directory


def faded(image: Image.Image, amount: float = 0.22) -> Image.Image:
    return ImageEnhance.Contrast(image).enhance(amount)


def tilted(image: Image.Image, degrees: float) -> Image.Image:
    """Rotate the page anticlockwise, which is the direction skew is reported in."""
    return image.rotate(
        degrees, expand=True, fillcolor=(255, 255, 255), resample=Image.Resampling.BICUBIC
    )


def lit_from_one_side(image: Image.Image) -> Image.Image:
    """A photograph taken with a lamp at the upper left."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = pixels.shape[:2]
    rows, columns = np.mgrid[0:height, 0:width]
    falloff = 0.42 + 0.58 * np.exp(
        -(((columns / width - 0.15) ** 2 + (rows / height - 0.1) ** 2) / 0.35)
    )
    shaded = np.clip(pixels * falloff[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(shaded)


class TestMeasuring:
    def test_a_straight_page_measures_straight(self):
        assert abs(estimate_skew(page())) < 0.5

    @pytest.mark.parametrize("degrees", [-6.0, -2.5, 3.0, 7.0])
    def test_the_angle_it_finds_is_the_angle_that_was_applied(self, degrees):
        found = estimate_skew(tilted(page(), degrees))
        assert found == pytest.approx(degrees, abs=0.6)

    def test_a_blank_page_is_not_reported_as_crooked(self):
        """Nothing to level, so the honest answer is zero rather than noise."""
        assert estimate_skew(Image.new("RGB", (WIDTH, HEIGHT), "white")) == 0.0

    def test_blur_shows_up_as_lost_sharpness(self):
        clean = sharpness(_luma(page()))
        soft = sharpness(_luma(blurred(page())))
        assert soft < SHARPNESS_FLOOR < clean

    def test_contrast_is_measured_between_ink_and_paper(self):
        """Not as a spread over every pixel.

        A page is mostly paper: the 5th and 95th percentiles of a clean invoice
        are both white, and measuring it that way reports perfect contrast as
        none at all. That was the first implementation, and it condemned every
        clean page in the corpus as needing rescue.
        """
        assert contrast(_luma(page())) > CONTRAST_FLOOR
        assert contrast(_luma(faded(page()))) < CONTRAST_FLOOR

    def test_a_page_with_no_ink_is_not_a_contrast_problem(self):
        assert contrast(_luma(Image.new("RGB", (400, 400), "white"))) == 1.0

    def test_uneven_lighting_is_visible_and_even_lighting_is_not(self):
        assert illumination_spread(_luma(page())) < 0.18
        assert illumination_spread(_luma(lit_from_one_side(page()))) > 0.18

    def test_lighting_is_measured_on_the_paper_not_on_the_ink(self):
        """A page with dense text and wide margins is not unevenly lit.

        Averaging each cell of the page instead of reading its paper level makes
        ink density look like a shadow, and every densely typeset invoice comes
        out flagged.
        """
        dense = Image.new("RGB", (WIDTH, HEIGHT), "white")
        draw = ImageDraw.Draw(dense)
        # Solid ink over the top half would be a fair reading of "dark", since a
        # cell with no paper in it genuinely cannot report a paper level. Text is
        # not solid: there is white between every line of it, everywhere.
        for y in range(150, HEIGHT // 2, 6):
            draw.rectangle([150, y, WIDTH - 150, y + 3], fill=(40, 40, 40))

        assert illumination_spread(_luma(dense)) < 0.18


class TestRestraint:
    """What it does when nothing is wrong."""

    def test_a_clean_page_is_left_completely_alone(self):
        _, prep = prepare_image(page())
        assert prep.actions == ()
        assert prep.changed is False

    def test_a_clean_page_is_not_even_rewritten(self, tmp_path):
        """No second file, so a clean corpus costs nothing but the measurement."""
        directory = page_dir(tmp_path)
        source = directory / "page-001.jpg"
        page().save(source)
        pages = [PageImage(page_number=1, path=source, width=WIDTH, height=HEIGHT)]

        prepared, report = prepare_pages(pages)
        assert prepared[0].path == source
        assert list(directory.iterdir()) == [source]
        assert report.changed_pages == ()

    def test_a_page_tilted_further_than_a_scanner_could_be_is_left_as_it_is(self):
        """At 30 degrees this is a photograph of a document on a desk, or the
        estimate is wrong. Rotating on either reading makes it worse."""
        _, prep = prepare_image(page(), PageQuality(skew_degrees=MAX_SKEW_DEGREES + 18.0))
        assert "deskew" not in " ".join(prep.actions)
        assert prep.notes and "too far" in prep.notes[0]

    def test_a_sideways_page_is_reported_rather_than_turned(self):
        """Which quarter turn puts it right needs to know which way the glyphs
        face. Turning it the wrong way leaves the model reading upside down,
        which is worse than leaving it on its side."""
        sideways = page().rotate(90, expand=True)
        _, prep = prepare_image(sideways)

        assert any("sideways" in note for note in prep.notes)
        assert not any("deskewed" in action for action in prep.actions)

    def test_an_upright_page_is_not_called_sideways(self):
        from docuflow.ocr.preprocess import detect_quarter_turn

        assert detect_quarter_turn(page()) == 0

    def test_preprocessing_can_be_switched_off_entirely(self, tmp_path):
        source = page_dir(tmp_path) / "page-001.jpg"
        tilted(page(), 7.0).save(source)
        pages = [PageImage(page_number=1, path=source, width=WIDTH, height=HEIGHT)]

        prepared, report = prepare_pages(pages, enabled=False)
        assert prepared == pages
        assert report.enabled is False
        assert report.summary == "preprocessing off"


class TestCorrecting:
    def test_a_crooked_page_comes_out_straight(self):
        _, prep = prepare_image(tilted(page(), 7.0))
        assert any("deskewed" in action for action in prep.actions)
        assert abs(prep.after.skew_degrees) < abs(prep.before.skew_degrees)
        assert abs(prep.after.skew_degrees) < 1.0

    def test_a_soft_page_comes_out_sharper(self):
        _, prep = prepare_image(blurred(page()))
        assert "sharpened" in prep.actions
        assert prep.after.sharpness > prep.before.sharpness

    def test_a_faded_page_comes_out_readable(self):
        _, prep = prepare_image(faded(page()))
        assert "raised the contrast" in prep.actions
        assert prep.after.contrast > CONTRAST_FLOOR

    def test_a_photograph_is_relit_and_straightened(self):
        """The whole point of the stage: a phone photo of an invoice."""
        photo = tilted(blurred(lit_from_one_side(page()), 1.0), 2.5)
        _, prep = prepare_image(photo)

        assert any("deskewed" in action for action in prep.actions)
        assert "evened out the lighting" in prep.actions
        assert prep.after.illumination_spread < prep.before.illumination_spread
        assert abs(prep.after.skew_degrees) < 1.0

    def test_the_lighting_is_fixed_before_the_angle_is_trusted(self):
        """Found by running the corpus, and the reason the order is what it is.

        Skew is estimated from a mask of where the ink is, thresholded against
        the page's own mean brightness. On a page lit from one side that mask
        marks the *shadow* rather than the text, so the angle describes the
        gradient. Two corpus documents estimated -2.8 degrees, rotated, and
        measured +2.8 - the same tilt mirrored, which is what rotating a
        gradient does. The safety net reverted both and they stayed crooked.
        """
        photo = tilted(lit_from_one_side(page()), 3.0)
        _, prep = prepare_image(photo)

        assert prep.actions.index("evened out the lighting") < next(
            index for index, action in enumerate(prep.actions) if "deskewed" in action
        ), "the lighting has to be corrected before the angle is measured"
        assert "deskew" not in prep.reverted
        assert abs(prep.after.skew_degrees) < 1.0

    def test_the_prepared_page_is_written_beside_the_original(self, tmp_path):
        """OCR reads the clean copy; the reviewer still has the real one."""
        source = page_dir(tmp_path) / "page-001.jpg"
        tilted(page(), 6.0).save(source)
        pages = [PageImage(page_number=1, path=source, width=WIDTH, height=HEIGHT)]

        prepared, report = prepare_pages(pages)
        assert prepared[0].path != source
        assert prepared[0].path.name == "page-001-prepared.jpg"
        assert source.exists(), "the rendered page is never overwritten"
        assert report.changed_pages and report.changed_pages[0].page_number == 1

    def test_the_recorded_size_matches_the_file_that_was_written(self, tmp_path):
        """Rotation grows the canvas, and a stale size would misplace every
        bounding box drawn against it."""
        source = page_dir(tmp_path) / "page-001.jpg"
        tilted(page(), 6.0).save(source)
        prepared, _ = prepare_pages(
            [PageImage(page_number=1, path=source, width=WIDTH, height=HEIGHT)]
        )
        with Image.open(prepared[0].path) as written:
            assert (prepared[0].width, prepared[0].height) == written.size


class TestNeverMakingItWorse:
    def test_a_correction_that_does_not_help_is_thrown_away(self, monkeypatch):
        """The safety net under every heuristic here.

        A sharpen that does not sharpen is a filter that only added artefacts,
        so the previous image is kept and the attempt is reported as reverted.
        """
        from docuflow.ocr import preprocess

        monkeypatch.setattr(preprocess, "sharpen", lambda image: blurred(image, 4.0))
        _, prep = preprocess.prepare_image(blurred(page()))

        assert "sharpened" not in prep.actions
        assert "sharpen" in prep.reverted

    def test_a_page_that_cannot_be_opened_is_passed_through(self, tmp_path):
        """Preprocessing is an optimisation. Failing it must not fail the run."""
        broken = page_dir(tmp_path) / "page-001.jpg"
        broken.write_bytes(b"this is not an image")
        pages = [PageImage(page_number=1, path=broken, width=10, height=10)]

        prepared, report = prepare_pages(pages)
        assert prepared == pages
        assert report.pages[0].notes and "could not be prepared" in report.pages[0].notes[0]


class TestReportingHowLegibleThePageWas:
    """Separate from what was corrected, and the difference carries weight.

    A page straightened by a degree and a half was perfectly readable to begin
    with. A page whose ink had to be rescued from the paper was not, and the OCR
    engine's own quality score cannot tell you that - it scores the text that
    came back, and text that came back confidently wrong scores as well as text
    that came back right.
    """

    def test_a_faded_page_is_reported_as_barely_legible(self):
        _, prep = prepare_image(faded(page()))
        from docuflow.ocr.preprocess import PrepReport

        assert PrepReport(pages=(prep,)).rescued_pages

    def test_a_soft_page_is_too(self):
        _, prep = prepare_image(blurred(page()))
        from docuflow.ocr.preprocess import PrepReport

        assert PrepReport(pages=(prep,)).rescued_pages

    def test_a_merely_crooked_page_is_not(self):
        """Straightening says nothing about whether the characters were clear."""
        _, prep = prepare_image(tilted(page(), 6.0))
        from docuflow.ocr.preprocess import PrepReport

        assert prep.actions, "it was corrected"
        assert PrepReport(pages=(prep,)).rescued_pages == (), "but it was always legible"

    def test_a_clean_page_is_not_either(self):
        _, prep = prepare_image(page())
        from docuflow.ocr.preprocess import PrepReport

        assert PrepReport(pages=(prep,)).rescued_pages == ()


class TestTheReport:
    def test_it_says_what_it_did_to_how_many_pages(self):
        clean = PageQuality(skew_degrees=0.0, sharpness=1.0, contrast=1.0)
        from docuflow.ocr.preprocess import PagePrep, PrepReport

        report = PrepReport(
            pages=(
                PagePrep(page_number=1, before=clean, after=clean, actions=("deskewed +3.0°",)),
                PagePrep(page_number=2, before=clean, after=clean),
            )
        )
        assert "1/2 page(s) corrected" in report.summary
        assert report.actions == ("deskewed +3.0°",)

    def test_an_untouched_document_says_so_plainly(self):
        clean = PageQuality(sharpness=1.0, contrast=1.0)
        from docuflow.ocr.preprocess import PagePrep, PrepReport

        report = PrepReport(pages=(PagePrep(page_number=1, before=clean, after=clean),))
        assert report.summary == "1 page(s) already clean"
