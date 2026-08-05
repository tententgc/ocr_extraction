"""Typhoon Cloud OCR via the official `typhoon-ocr` package (task type v1.5)."""

from __future__ import annotations

import logging

from docuflow.ocr.base import OcrProvider, OcrTransientError, OcrUnavailableError, PageImage

logger = logging.getLogger(__name__)

# Statuses that describe the call rather than the page. 408 is the one that
# matters here: Typhoon returns it when its own model run overruns, which some
# pages provoke far more often than others - Thai numerals, in this corpus.
TRANSIENT_STATUSES = (408, 409, 425, 429, 500, 502, 503, 504)


def _is_transient(error: Exception) -> bool:
    """Whether asking this engine again is worth doing.

    The `typhoon-ocr` package surfaces the OpenAI client's errors, which carry a
    `status_code`. When they do not, the code is still in the message, and
    reading it there is worth the ugliness: the alternative is treating a
    two-second timeout as a reason to spend ten minutes on a local model.
    """
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status in TRANSIENT_STATUSES
    message = str(error)
    return any(f"Error code: {code}" in message for code in TRANSIENT_STATUSES)


class TyphoonCloudOcr(OcrProvider):
    """Hosted Typhoon OCR 1.5 - fastest path, requires an API key."""

    name = "typhoon-ocr1.5-cloud"
    # Typhoon OCR is a document-transcription model.
    purpose_built = True

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

        try:
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
        except Exception as exc:  # noqa: BLE001 - re-raised either way, only the kind changes
            if _is_transient(exc):
                raise OcrTransientError(str(exc)) from exc
            raise
        return (text or "").strip()
