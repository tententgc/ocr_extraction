"""OCR through any OpenAI-compatible vision endpoint (Ollama, OpenAI, vLLM)."""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from openai import BadRequestError

from docuflow.ocr.base import OcrProvider, OcrUnavailableError, PageImage
from docuflow.ocr.prompts import v15_prompt

logger = logging.getLogger(__name__)

# Published Typhoon OCR 1.5 decoding parameters; harmless for other models.
TEMPERATURE = 0.1
TOP_P = 0.6
REQUEST_TIMEOUT_SECONDS = 600.0
MAX_PARAM_NEGOTIATIONS = 4


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


class OpenAiVisionOcr(OcrProvider):
    """Send a page image plus the OCR prompt to a vision chat model."""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        base_url: str | None,
        api_key: str,
        max_tokens: int = 8192,
        figure_language: str = "English",
        missing_key_reason: str = "API key is not set",
        extra_body: dict | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.figure_language = figure_language
        self._missing_key_reason = missing_key_reason
        self._extra_body = extra_body or {}
        self._client = None
        # Negotiated on first use; see `_adapt`.
        self._token_param = "max_tokens"
        self._send_sampling = True

    def available(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, self._missing_key_reason
        try:
            import openai  # noqa: F401
        except ImportError:
            return False, "the openai package is not installed"
        return True, ""

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=REQUEST_TIMEOUT_SECONDS
            )
        return self._client

    def _adapt(self, message: str) -> bool:
        """Drop or rename a parameter the endpoint rejected. See the chat adapter."""
        lowered = message.lower()
        if "max_completion_tokens" in lowered and self._token_param == "max_tokens":
            self._token_param = "max_completion_tokens"
            return True
        if "temperature" in lowered and self._send_sampling:
            logger.info("%s: dropping sampling parameters this model rejects", self.name)
            self._send_sampling = False
            return True
        if "top_p" in lowered and self._send_sampling:
            self._send_sampling = False
            return True
        return False

    def recognize(self, page: PageImage) -> str:
        ok, reason = self.available()
        if not ok:
            raise OcrUnavailableError(reason)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": v15_prompt(self.figure_language)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(page.path)}"},
                    },
                ],
            }
        ]

        for _ in range(MAX_PARAM_NEGOTIATIONS):
            kwargs = {"model": self.model, "messages": messages, self._token_param: self.max_tokens}
            if self._send_sampling:
                kwargs["temperature"] = TEMPERATURE
                kwargs["top_p"] = TOP_P
            if self._extra_body:
                kwargs["extra_body"] = self._extra_body
            try:
                response = self._get_client().chat.completions.create(**kwargs)
            except BadRequestError as exc:
                if self._adapt(str(exc)):
                    continue
                raise
            return (response.choices[0].message.content or "").strip()

        raise OcrUnavailableError(f"{self.name}: could not agree request parameters with the endpoint")
