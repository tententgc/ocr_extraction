"""Discover which OCR and agent models are selectable right now.

Everything is derived from the plugin registries, so a newly registered plugin
appears in the UI picker without any change here.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Literal

from docuflow.config import Settings

logger = logging.getLogger(__name__)

Kind = Literal["ocr", "llm"]

# Name fragments that mark an Ollama model as able to read images.
VISION_MARKERS = ("vl", "vision", "-ocr", "ocr1", "llava", "minicpm", "moondream", "pixtral")
# Models purpose-built for document OCR, promoted in the picker.
DOCUMENT_OCR_MARKERS = ("typhoon-ocr", "ocr1.5", "dots.ocr", "olmocr", "got-ocr", "nanonets-ocr")


def _is_vision(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in VISION_MARKERS)


def _is_document_ocr(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in DOCUMENT_OCR_MARKERS)


@dataclass(frozen=True)
class ModelOption:
    """One selectable model, as shown in the picker."""

    key: str  # "<provider>:<model>" - the value the UI sends back
    provider: str
    model: str
    label: str
    kind: Kind
    available: bool
    reason: str = ""
    recommended: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)


def _options_for(kind: Kind, settings: Settings) -> list[ModelOption]:
    from docuflow.llm.plugins import llm_plugins
    from docuflow.ocr.plugins import ocr_plugins

    plugins = ocr_plugins() if kind == "ocr" else llm_plugins()
    options: list[ModelOption] = []

    for key, plugin in plugins.items():
        # Probe availability once per plugin using its default model.
        try:
            probe = plugin.build(settings, None)
        except Exception as exc:  # noqa: BLE001
            logger.debug("plugin %s could not be built: %s", key, exc)
            continue

        if probe is None:  # the rules-only agent
            options.append(
                ModelOption(
                    key=f"{key}:-",
                    provider=key,
                    model="-",
                    label=plugin.label,
                    kind=kind,
                    available=True,
                    tags=tuple(plugin.tags),
                )
            )
            continue

        available, reason = probe.available()

        try:
            discovered = plugin.list_models(settings) or []
        except Exception as exc:  # noqa: BLE001 - discovery is best-effort
            logger.debug("model discovery failed for %s: %s", key, exc)
            discovered = []

        models = discovered or [plugin.default_model(settings)]
        for model in models:
            if not model:
                continue
            short = model.removesuffix(":latest").split("/")[-1]
            tags = list(plugin.tags)
            if kind == "ocr" and _is_document_ocr(model):
                tags.append("document OCR")
            options.append(
                ModelOption(
                    key=f"{key}:{model}",
                    provider=key,
                    model=model,
                    label=f"{short} · {plugin.label}" if model != "-" else plugin.label,
                    kind=kind,
                    available=available,
                    reason=reason,
                    recommended=(
                        _is_document_ocr(model) if kind == "ocr" else "typhoon" in model.lower()
                    ),
                    tags=tuple(dict.fromkeys(tags)),
                )
            )

    def sort_key(option: ModelOption) -> tuple:
        return (not option.available, not option.recommended, option.label.lower())

    return sorted(options, key=sort_key)


def list_models(settings: Settings) -> dict[str, list[dict[str, object]]]:
    """Every selectable OCR and agent model, with availability."""
    return {
        "ocr": [asdict(option) | {"tags": list(option.tags)} for option in _options_for("ocr", settings)],
        "llm": [asdict(option) | {"tags": list(option.tags)} for option in _options_for("llm", settings)],
    }
