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


def resolve_verify_llm(
    settings: Settings, primary: LlmProvider | None
) -> tuple[LlmProvider | None, str]:
    """Pick the model for the second extraction pass.

    Prefers a backend *different* from the primary: two passes of the same model
    over the same text tend to repeat the same error, so agreement between them
    would be a much weaker signal than it looks. Falls back to the primary model
    with a different prompt when nothing else is reachable, and says so.
    """
    configured = settings.llm_verify_provider
    if configured == "off" or primary is None:
        return None, "verification disabled" if configured == "off" else "no primary model"

    if configured != "auto":
        provider = build_llm(configured, settings, settings.llm_verify_model or None)
        if provider is None:
            return None, "verification provider is 'rules'"
        available, reason = provider.available()
        return (provider, "") if available else (None, f"{configured} unavailable ({reason})")

    # Auto: first available backend whose model differs from the primary's.
    primary_model = getattr(primary, "model", None) or getattr(primary, "model_id", None)
    for key in auto_order():
        try:
            provider = build_llm(key, settings)
        except ValueError:
            continue
        if provider is None:
            break  # reached the rules-only backend
        available, _ = provider.available()
        if not available:
            continue
        candidate_model = getattr(provider, "model", None) or getattr(provider, "model_id", None)
        if candidate_model != primary_model:
            return provider, ""

    # Only one model is reachable. Still useful - the verification prompt reads
    # the page label-first rather than field-first - but weaker, so it is named.
    return primary, "same model as the primary pass; independence is limited"


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
