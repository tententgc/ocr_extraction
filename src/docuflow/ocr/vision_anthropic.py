"""OCR using Claude's vision capability."""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from docuflow.ocr.base import OcrProvider, OcrUnavailableError, PageImage
from docuflow.ocr.prompts import v15_prompt

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 600.0
DEFAULT_MODEL = "claude-sonnet-5"


class AnthropicVisionOcr(OcrProvider):
    """Read a page image with Claude."""

    name = "claude-vision"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        max_tokens: int = 8192,
        figure_language: str = "English",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or None
        self.max_tokens = max_tokens
        self.figure_language = figure_language
        self._client = None
        self._send_temperature = True

    def available(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "ANTHROPIC_API_KEY is not set"
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "the anthropic package is not installed (uv pip install anthropic)"
        return True, ""

    def _get_client(self):
        if self._client is None:
            import anthropic

            kwargs = {"api_key": self.api_key, "timeout": REQUEST_TIMEOUT_SECONDS}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def recognize(self, page: PageImage) -> str:
        ok, reason = self.available()
        if not ok:
            raise OcrUnavailableError(reason)

        import anthropic

        encoded = base64.b64encode(Path(page.path).read_bytes()).decode("ascii")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded},
                    },
                    {"type": "text", "text": v15_prompt(self.figure_language)},
                ],
            }
        ]

        # The newest Claude models reject `temperature`; drop it on complaint
        # rather than keeping a list of model names that goes stale.
        for _ in range(2):
            kwargs = {"model": self.model, "max_tokens": self.max_tokens, "messages": messages}
            if self._send_temperature:
                kwargs["temperature"] = 0.0
            try:
                response = self._get_client().messages.create(**kwargs)
            except anthropic.BadRequestError as exc:
                if "temperature" in str(exc).lower() and self._send_temperature:
                    logger.info("claude-vision: dropping temperature, which %s rejects", self.model)
                    self._send_temperature = False
                    continue
                raise
            return "".join(block.text for block in response.content if block.type == "text").strip()

        raise OcrUnavailableError("claude-vision: could not agree request parameters")
