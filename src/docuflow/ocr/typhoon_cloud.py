"""Typhoon Cloud OCR via the official `typhoon-ocr` package (task type v1.5)."""

from __future__ import annotations

import logging

from docuflow.ocr.base import OcrProvider, OcrUnavailableError, PageImage

logger = logging.getLogger(__name__)


class TyphoonCloudOcr(OcrProvider):
    """Hosted Typhoon OCR 1.5 - fastest path, requires an API key."""

    name = "typhoon-ocr1.5-cloud"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.opentyphoon.ai/v1",
        model: str = "typhoon-ocr",
        target_image_dim: int = 1800,
        figure_language: str = "English",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.target_image_dim = target_image_dim
        self.figure_language = figure_language

    def available(self) -> tuple[bool, str]:
        if not self.api_key.strip():
            return False, "TYPHOON_API_KEY is not set"
        try:
            import typhoon_ocr  # noqa: F401
        except ImportError:
            return False, "typhoon-ocr is not installed"
        return True, ""

    def recognize(self, page: PageImage) -> str:
        ok, reason = self.available()
        if not ok:
            raise OcrUnavailableError(reason)

        from typhoon_ocr import ocr_document

        # Pages are already rendered to single images, so page_num is always 1.
        text = ocr_document(
            pdf_or_image_path=str(page.path),
            task_type="v1.5",
            target_image_dim=self.target_image_dim,
            page_num=1,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            figure_language=self.figure_language,
        )
        return (text or "").strip()
