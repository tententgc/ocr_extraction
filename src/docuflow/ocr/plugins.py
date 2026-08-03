"""OCR-engine plugin registry.

Mirrors `llm/plugins.py`: one `register_ocr(...)` call adds an engine to the
factory, the model picker and the benchmark command at once.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from typing import Callable

from docuflow.config import Settings
from docuflow.ocr.base import OcrProvider
from docuflow.ocr.fallbacks import PdfTextLayerOcr, TesseractOcr
from docuflow.ocr.ollama_ocr import OllamaVisionOcr
from docuflow.ocr.typhoon_cloud import TyphoonCloudOcr
from docuflow.ocr.typhoon_local import TyphoonLocalOcr
from docuflow.ocr.vision_anthropic import AnthropicVisionOcr
from docuflow.ocr.vision_openai import OpenAiVisionOcr

logger = logging.getLogger(__name__)

Builder = Callable[[Settings, str | None], OcrProvider]


@dataclass(frozen=True)
class OcrPlugin:
    """One selectable OCR engine."""

    key: str
    label: str
    build: Builder
    default_model: Callable[[Settings], str]
    list_models: Callable[[Settings], list[str]] = lambda _settings: []
    tags: tuple[str, ...] = field(default_factory=tuple)
    # Order tried when the configured engine cannot run; lower runs first.
    fallback_priority: int = 100
    # The Settings field holding this engine's model name, so a UI selection can
    # be written back without a lookup table living somewhere else.
    model_field: str | None = None


_REGISTRY: dict[str, OcrPlugin] = {}


def register_ocr(plugin: OcrPlugin) -> None:
    _REGISTRY[plugin.key] = plugin


def ocr_plugins() -> dict[str, OcrPlugin]:
    return dict(_REGISTRY)


def get_ocr_plugin(key: str) -> OcrPlugin | None:
    return _REGISTRY.get(key)


def fallback_order() -> list[str]:
    return [plugin.key for plugin in sorted(_REGISTRY.values(), key=lambda item: item.fallback_priority)]


def _ollama_vision_models(settings: Settings) -> list[str]:
    from docuflow.models_registry import _is_vision
    from docuflow.ocr.ollama_ocr import server_models

    return [name for name in (server_models(settings.ollama_base_url) or []) if _is_vision(name)]


def _typhoon_ocr_models(settings: Settings) -> list[str]:
    from docuflow.llm.plugins import _openai_style_models

    return [
        name
        for name in _openai_style_models(settings.typhoon_base_url, settings.typhoon_api_key)
        if "ocr" in name
    ]


# ---------------------------------------------------------------------------
# Built-in engines
# ---------------------------------------------------------------------------

register_ocr(
    OcrPlugin(
        key="typhoon_api",
        model_field="ocr_cloud_model",
        label="Typhoon Cloud OCR",
        build=lambda settings, model: TyphoonCloudOcr(
            api_key=settings.typhoon_api_key,
            base_url=settings.typhoon_base_url,
            model=model or settings.ocr_cloud_model,
            target_image_dim=settings.ocr_image_dim,
        ),
        default_model=lambda settings: settings.ocr_cloud_model,
        list_models=_typhoon_ocr_models,
        tags=("cloud", "fastest", "document OCR"),
        fallback_priority=10,
    )
)

register_ocr(
    OcrPlugin(
        key="ollama",
        model_field="ocr_ollama_model",
        label="Ollama (local)",
        build=lambda settings, model: OllamaVisionOcr(
            model=model or settings.ocr_ollama_model,
            base_url=settings.ollama_base_url,
            num_ctx=settings.ollama_num_ctx,
            num_predict=settings.ocr_max_tokens,
            keep_alive=settings.ollama_keep_alive,
        ),
        default_model=lambda settings: settings.ocr_ollama_model,
        list_models=_ollama_vision_models,
        tags=("local", "no key"),
        fallback_priority=20,
    )
)

register_ocr(
    OcrPlugin(
        key="local",
        model_field="ocr_local_model",
        label="MLX (Apple Silicon)",
        build=lambda settings, model: TyphoonLocalOcr(model_id=model or settings.ocr_local_model),
        default_model=lambda settings: settings.ocr_local_model,
        list_models=lambda settings: [settings.ocr_local_model],
        tags=("local", "no key", "document OCR"),
        fallback_priority=30,
    )
)

register_ocr(
    OcrPlugin(
        key="openai",
        model_field="ocr_openai_model",
        label="OpenAI vision",
        build=lambda settings, model: OpenAiVisionOcr(
            name="openai-vision",
            model=model or settings.ocr_openai_model,
            base_url=settings.openai_base_url or None,
            api_key=settings.openai_api_key,
            max_tokens=settings.ocr_max_tokens,
            missing_key_reason="OPENAI_API_KEY is not set",
        ),
        default_model=lambda settings: settings.ocr_openai_model,
        list_models=lambda settings: [settings.ocr_openai_model] if settings.openai_api_key else [],
        tags=("cloud", "general vision"),
        fallback_priority=40,
    )
)

register_ocr(
    OcrPlugin(
        key="claude",
        model_field="ocr_anthropic_model",
        label="Claude vision",
        build=lambda settings, model: AnthropicVisionOcr(
            api_key=settings.anthropic_api_key,
            model=model or settings.ocr_anthropic_model,
            base_url=settings.anthropic_base_url or None,
            max_tokens=settings.ocr_max_tokens,
        ),
        default_model=lambda settings: settings.ocr_anthropic_model,
        list_models=lambda settings: [settings.ocr_anthropic_model] if settings.anthropic_api_key else [],
        tags=("cloud", "general vision"),
        fallback_priority=50,
    )
)

register_ocr(
    OcrPlugin(
        key="pdftext",
        label="PDF text layer (no model)",
        build=lambda _settings, _model: PdfTextLayerOcr(),
        default_model=lambda _settings: "-",
        tags=("instant", "digital PDFs only"),
        fallback_priority=60,
    )
)

register_ocr(
    OcrPlugin(
        key="tesseract",
        label="Tesseract (no model)",
        build=lambda _settings, _model: TesseractOcr(),
        default_model=lambda _settings: "-",
        list_models=lambda _settings: ["-"] if shutil.which("tesseract") else [],
        tags=("offline", "weak on layout"),
        fallback_priority=70,
    )
)
