"""Provider selection and fallback ordering, driven by the plugin registry."""

from __future__ import annotations

import logging

from docuflow.config import Settings
from docuflow.ocr.base import OcrProvider
from docuflow.ocr.plugins import fallback_order, get_ocr_plugin, ocr_plugins

logger = logging.getLogger(__name__)


def build_provider(name: str, settings: Settings, model: str | None = None) -> OcrProvider:
    """Instantiate one engine by plugin key."""
    plugin = get_ocr_plugin(name)
    if plugin is None:
        raise ValueError(f"Unknown OCR provider: {name}")
    return plugin.build(settings, model)


# Not an engine so much as a reader: the characters are already in the file.
TEXT_LAYER = "pdftext"


def provider_chain(settings: Settings) -> list[OcrProvider]:
    """Ordered engines to try for each page.

    The configured engine goes first; the rest are fallbacks so a missing model
    download or an expired key degrades the demo instead of breaking it.

    Ahead of all of them, on a PDF that carries its own text, sits the text
    layer. This is the cheapest correct answer available: the characters were
    written into the file by whatever produced it, so recognition has nothing
    to add and every model call spent on such a page buys nothing.

    It is safe to put first because it removes itself. `PdfTextLayerOcr` raises
    `OcrUnavailableError` on a page with no embedded text, and the chain skips
    a provider that says it cannot help - so a photograph, a JPG, or a scan
    reaches exactly the engine it would have reached before. Text that is
    present but poor meets the same `score_page_text` gate as any other engine's
    output and escalates when it fails.

    `ocr_strict` still wins: a benchmark that names an engine must measure that
    engine, and reading the file instead would score the PDF writer's work.
    """
    primary = settings.resolved_ocr_provider()
    if settings.ocr_strict:
        # Measure exactly the engine that was asked for.
        order = [primary]
    else:
        order = [primary] + [key for key in fallback_order() if key != primary]
        if settings.ocr_prefer_text_layer and primary != TEXT_LAYER:
            order = [TEXT_LAYER] + [key for key in order if key != TEXT_LAYER]

    chain: list[OcrProvider] = []
    for key in order:
        try:
            provider = build_provider(key, settings)
        except ValueError:
            continue
        available, reason = provider.available()
        if available:
            chain.append(provider)
        elif key == primary:
            logger.warning("configured OCR engine '%s' is unavailable: %s", key, reason)
    return chain


def describe_providers(settings: Settings) -> list[dict[str, object]]:
    """Engine availability, surfaced in the UI status panel."""
    active = settings.resolved_ocr_provider()
    rows: list[dict[str, object]] = []
    for key, plugin in sorted(ocr_plugins().items(), key=lambda item: item[1].fallback_priority):
        try:
            provider = plugin.build(settings, None)
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not hide the rest
            rows.append({"key": key, "name": plugin.label, "available": False, "reason": str(exc), "active": False})
            continue
        available, reason = provider.available()
        rows.append(
            {
                "key": key,
                "name": provider.name,
                "label": plugin.label,
                "model": plugin.default_model(settings),
                "available": available,
                "reason": reason,
                "active": key == active,
                "tags": list(plugin.tags),
            }
        )
    return rows
