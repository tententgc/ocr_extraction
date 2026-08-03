"""Fallback recognisers used when no Typhoon model is reachable.

They keep the demo running end-to-end (and CI fast) but are markedly weaker on
layout, so the pipeline marks any run that used them as degraded and lowers the
confidence of every field read from that text.
"""

from __future__ import annotations

import logging
import shutil
import subprocess

from docuflow.ocr.base import OcrProvider, OcrUnavailableError, PageImage

logger = logging.getLogger(__name__)

TESSERACT_TIMEOUT_SECONDS = 120
# Thai + English; falls back to English alone if the Thai data pack is missing.
TESSERACT_LANGS = "tha+eng"


class PdfTextLayerOcr(OcrProvider):
    """Zero-cost reader for PDFs that already contain selectable text."""

    name = "pdf-text-layer"
    # Not recognition at all - the characters the PDF itself stores.
    purpose_built = True

    def available(self) -> tuple[bool, str]:
        return True, ""

    def recognize(self, page: PageImage) -> str:
        if not page.embedded_text:
            raise OcrUnavailableError("page has no embedded text layer")
        return page.embedded_text.strip()


class TesseractOcr(OcrProvider):
    """Offline fallback using the Tesseract CLI."""

    name = "tesseract"

    def __init__(self, languages: str = TESSERACT_LANGS) -> None:
        self.languages = languages

    def available(self) -> tuple[bool, str]:
        if shutil.which("tesseract") is None:
            return False, "tesseract is not installed (brew install tesseract tesseract-lang)"
        return True, ""

    def _installed_languages(self) -> set[str]:
        try:
            result = subprocess.run(
                ["tesseract", "--list-langs"], capture_output=True, text=True, timeout=20
            )
            return {line.strip() for line in result.stdout.splitlines()[1:] if line.strip()}
        except (subprocess.SubprocessError, OSError):
            return set()

    def recognize(self, page: PageImage) -> str:
        ok, reason = self.available()
        if not ok:
            raise OcrUnavailableError(reason)

        installed = self._installed_languages()
        wanted = [lang for lang in self.languages.split("+") if not installed or lang in installed]
        languages = "+".join(wanted) or "eng"

        try:
            result = subprocess.run(
                ["tesseract", str(page.path), "stdout", "-l", languages, "--psm", "4"],
                capture_output=True,
                text=True,
                timeout=TESSERACT_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise OcrUnavailableError("tesseract timed out") from exc

        if result.returncode != 0:
            raise OcrUnavailableError(f"tesseract failed: {result.stderr.strip()[:200]}")
        return result.stdout.strip()
