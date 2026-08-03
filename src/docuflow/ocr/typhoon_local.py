"""Typhoon OCR 1.5 running on-device via MLX (Apple Silicon).

This is the default provider: fully open source, no API key, no data leaving the
machine. The model is loaded once per process and reused across documents.
"""

from __future__ import annotations

import logging
import platform
import threading

from docuflow.ocr.base import OcrProvider, OcrUnavailableError, PageImage
from docuflow.ocr.prompts import v15_prompt

logger = logging.getLogger(__name__)

# Decoding parameters published by the Typhoon team for OCR 1.5.
TEMPERATURE = 0.1
TOP_P = 0.6
REPETITION_PENALTY = 1.1
DEFAULT_MAX_TOKENS = 6144

_load_lock = threading.Lock()
_cache: dict[str, tuple] = {}


def _load_model(model_id: str):
    """Load (and memoise) the MLX model, processor and config."""
    if model_id in _cache:
        return _cache[model_id]

    with _load_lock:
        if model_id in _cache:
            return _cache[model_id]
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        logger.info("loading local OCR model %s (first run downloads weights)", model_id)
        model, processor = load(model_id)
        config = load_config(model_id)
        _cache[model_id] = (model, processor, config)
        return _cache[model_id]


class TyphoonLocalOcr(OcrProvider):
    """On-device Typhoon OCR 1.5."""

    name = "typhoon-ocr1.5-local"

    def __init__(
        self,
        model_id: str = "mlx-community/typhoon-ocr1.5-2b-8bit",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        figure_language: str = "English",
    ) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.figure_language = figure_language

    def available(self) -> tuple[bool, str]:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return False, "local Typhoon OCR needs Apple Silicon (MLX); use OCR_PROVIDER=typhoon_api"
        try:
            import mlx_vlm  # noqa: F401
        except ImportError:
            return False, "mlx-vlm is not installed; run: uv pip install -e '.[local]'"
        return True, ""

    def warm_up(self) -> None:
        ok, reason = self.available()
        if not ok:
            raise OcrUnavailableError(reason)
        _load_model(self.model_id)

    def recognize(self, page: PageImage) -> str:
        ok, reason = self.available()
        if not ok:
            raise OcrUnavailableError(reason)

        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        model, processor, config = _load_model(self.model_id)
        prompt = apply_chat_template(
            processor, config, v15_prompt(self.figure_language), num_images=1
        )
        result = generate(
            model,
            processor,
            prompt,
            image=[str(page.path)],
            max_tokens=self.max_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            verbose=False,
        )
        text = getattr(result, "text", result)
        return str(text).strip()
