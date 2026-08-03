"""Agent-model plugin registry.

Adding a backend is one `register_llm(...)` call: give it a builder, a default
model and a way to discover model names. Nothing else in the codebase changes -
the factory, the UI picker and the CLI all read this registry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from docuflow.config import Settings
from docuflow.llm.anthropic_chat import AnthropicChat
from docuflow.llm.base import LlmProvider
from docuflow.llm.local_mlx import LocalMlxChat
from docuflow.llm.openai_compatible import OpenAICompatibleChat

logger = logging.getLogger(__name__)

ModelBuilder = Callable[[Settings, str | None], LlmProvider | None]
ModelLister = Callable[[Settings], list[str]]


@dataclass(frozen=True)
class LlmPlugin:
    """One selectable agent backend."""

    key: str
    label: str
    build: ModelBuilder
    default_model: Callable[[Settings], str]
    list_models: ModelLister = lambda _settings: []
    tags: tuple[str, ...] = field(default_factory=tuple)
    supports_tools: bool = True
    # Order tried when LLM_PROVIDER=auto; lower runs first.
    auto_priority: int = 100
    # The Settings field holding this backend's model name, so a UI selection can
    # be written back without a lookup table living somewhere else.
    model_field: str | None = None


_REGISTRY: dict[str, LlmPlugin] = {}


def register_llm(plugin: LlmPlugin) -> None:
    _REGISTRY[plugin.key] = plugin


def llm_plugins() -> dict[str, LlmPlugin]:
    return dict(_REGISTRY)


def get_llm_plugin(key: str) -> LlmPlugin | None:
    return _REGISTRY.get(key)


def auto_order() -> list[str]:
    """Plugin keys in the order `auto` should try them."""
    return [plugin.key for plugin in sorted(_REGISTRY.values(), key=lambda item: item.auto_priority)]


# ---------------------------------------------------------------------------
# Model discovery helpers
# ---------------------------------------------------------------------------


def _openai_style_models(base_url: str, api_key: str, exclude: tuple[str, ...] = ()) -> list[str]:
    """List models from an OpenAI-compatible `/models` endpoint."""
    if not api_key:
        return []
    import httpx

    try:
        response = httpx.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.debug("model listing failed for %s: %s", base_url, exc)
        return []
    payload = response.json()
    items = payload if isinstance(payload, list) else payload.get("data", [])
    names = [item.get("id", "") for item in items if isinstance(item, dict)]
    return [name for name in names if name and not any(word in name for word in exclude)]


def _anthropic_models(settings: Settings) -> list[str]:
    if not settings.anthropic_api_key:
        return []
    import httpx

    try:
        response = httpx.get(
            f"{(settings.anthropic_base_url or 'https://api.anthropic.com').rstrip('/')}/v1/models",
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
            },
            timeout=5.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.debug("Anthropic model listing failed: %s", exc)
        return []
    return [item.get("id", "") for item in response.json().get("data", []) if item.get("id")]


def _ollama_chat_models(settings: Settings) -> list[str]:
    from docuflow.models_registry import _is_vision
    from docuflow.ocr.ollama_ocr import server_models

    names = server_models(settings.ollama_base_url) or []
    return [
        name
        for name in names
        if not _is_vision(name) and "embed" not in name.lower() and "bge" not in name.lower()
    ]


# ---------------------------------------------------------------------------
# Built-in plugins
# ---------------------------------------------------------------------------

register_llm(
    LlmPlugin(
        key="typhoon",
        model_field="llm_cloud_model",
        label="Typhoon Cloud",
        build=lambda settings, model: OpenAICompatibleChat(
            name="typhoon-cloud",
            model=model or settings.llm_cloud_model,
            base_url=settings.typhoon_base_url,
            api_key=settings.typhoon_api_key,
            missing_key_reason="TYPHOON_API_KEY is not set",
        ),
        default_model=lambda settings: settings.llm_cloud_model,
        list_models=lambda settings: [
            name
            for name in _openai_style_models(
                settings.typhoon_base_url, settings.typhoon_api_key, exclude=("ocr", "asr")
            )
        ],
        tags=("cloud", "Thai-native"),
        auto_priority=10,
    )
)

register_llm(
    LlmPlugin(
        key="ollama",
        model_field="llm_ollama_model",
        label="Ollama (local)",
        build=lambda settings, model: OpenAICompatibleChat(
            name="ollama",
            model=model or settings.llm_ollama_model,
            base_url=settings.ollama_base_url,
            api_key="ollama",  # Ollama ignores it; the SDK requires a value.
            missing_key_reason="",
        ),
        default_model=lambda settings: settings.llm_ollama_model,
        list_models=_ollama_chat_models,
        tags=("local", "no key"),
        auto_priority=20,
    )
)

register_llm(
    LlmPlugin(
        key="openai",
        model_field="llm_openai_model",
        label="OpenAI",
        build=lambda settings, model: OpenAICompatibleChat(
            name="openai",
            model=model or settings.llm_openai_model,
            base_url=settings.openai_base_url or None,
            api_key=settings.openai_api_key,
            missing_key_reason="OPENAI_API_KEY is not set",
        ),
        default_model=lambda settings: settings.llm_openai_model,
        list_models=lambda settings: _openai_style_models(
            settings.openai_base_url or "https://api.openai.com/v1",
            settings.openai_api_key,
            exclude=("embedding", "whisper", "tts", "dall-e", "moderation"),
        ),
        tags=("cloud",),
        auto_priority=30,
    )
)

register_llm(
    LlmPlugin(
        key="claude",
        model_field="llm_anthropic_model",
        label="Claude (Anthropic)",
        build=lambda settings, model: AnthropicChat(
            api_key=settings.anthropic_api_key,
            model=model or settings.llm_anthropic_model,
            base_url=settings.anthropic_base_url or None,
        ),
        default_model=lambda settings: settings.llm_anthropic_model,
        list_models=_anthropic_models,
        tags=("cloud", "strong reasoning"),
        auto_priority=40,
    )
)

register_llm(
    LlmPlugin(
        key="local",
        model_field="llm_local_model",
        label="MLX (Apple Silicon)",
        build=lambda settings, model: LocalMlxChat(model_id=model or settings.llm_local_model),
        default_model=lambda settings: settings.llm_local_model,
        list_models=lambda settings: [settings.llm_local_model],
        tags=("local", "no key"),
        supports_tools=False,
        auto_priority=50,
    )
)

register_llm(
    LlmPlugin(
        key="rules",
        label="No agent - deterministic rules only",
        build=lambda _settings, _model: None,
        default_model=lambda _settings: "-",
        tags=("no model", "always available"),
        supports_tools=False,
        auto_priority=999,
    )
)
