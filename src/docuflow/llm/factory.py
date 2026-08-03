"""Agent-model selection with graceful degradation, driven by the registry.

`auto` walks the registry in priority order and takes the first backend that can
actually run, ending at `rules` - which returns no model at all and leaves the
pipeline on its deterministic extractor. The demo therefore never hard-fails on
a missing key or an un-pulled model.
"""

from __future__ import annotations

import logging

from docuflow.config import Settings
from docuflow.llm.base import LlmProvider
from docuflow.llm.plugins import auto_order, get_llm_plugin, llm_plugins

logger = logging.getLogger(__name__)


def build_llm(name: str, settings: Settings, model: str | None = None) -> LlmProvider | None:
    """Instantiate one backend by plugin key, or None for rules-only mode."""
    plugin = get_llm_plugin(name)
    if plugin is None:
        raise ValueError(f"Unknown LLM provider: {name}")
    return plugin.build(settings, model)


def resolve_llm(settings: Settings) -> tuple[LlmProvider | None, str]:
    """Return `(provider, note)`; `note` explains any fallback."""
    configured = settings.llm_provider
    order = [configured] if configured != "auto" else auto_order()

    notes: list[str] = []
    for key in order:
        try:
            provider = build_llm(key, settings)
        except ValueError:
            notes.append(f"unknown provider '{key}'")
            continue
        if provider is None:
            return None, "; ".join(notes)
        available, reason = provider.available()
        if available:
            return provider, "; ".join(notes)
        notes.append(f"{key} unavailable ({reason})")
        logger.warning("agent backend '%s' unavailable: %s", key, reason)

    notes.append("falling back to rule-based extraction only")
    return None, "; ".join(notes)


def describe_llms(settings: Settings) -> list[dict[str, object]]:
    """Availability of each agent backend, for the UI status panel."""
    active, _ = resolve_llm(settings)
    active_name = active.name if active else None

    rows: list[dict[str, object]] = []
    for key, plugin in sorted(llm_plugins().items(), key=lambda item: item[1].auto_priority):
        try:
            provider = plugin.build(settings, None)
        except Exception as exc:  # noqa: BLE001
            rows.append({"key": key, "name": plugin.label, "available": False, "reason": str(exc), "active": False})
            continue
        if provider is None:
            rows.append(
                {
                    "key": key,
                    "name": "rules-only",
                    "label": plugin.label,
                    "model": "-",
                    "available": True,
                    "reason": "deterministic extraction, always available",
                    "active": active is None,
                    "tags": list(plugin.tags),
                    "supports_tools": False,
                }
            )
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
                "active": available and provider.name == active_name,
                "tags": list(plugin.tags),
                "supports_tools": plugin.supports_tools and provider.supports_tools,
            }
        )
    return rows
