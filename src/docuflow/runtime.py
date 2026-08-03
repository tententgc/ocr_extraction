"""Runtime model selection.

Lets the UI switch OCR and agent models without editing `.env` or restarting.
The selection is an overlay on the configured `Settings`; the base configuration
is never mutated, so a restart always returns to the declared defaults.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from docuflow.config import Settings


def _ocr_field(provider: str) -> str | None:
    from docuflow.ocr.plugins import get_ocr_plugin

    plugin = get_ocr_plugin(provider)
    return plugin.model_field if plugin else None


def _llm_field(provider: str) -> str | None:
    from docuflow.llm.plugins import get_llm_plugin

    plugin = get_llm_plugin(provider)
    return plugin.model_field if plugin else None


def _known_ocr(provider: str) -> bool:
    from docuflow.ocr.plugins import ocr_plugins

    return provider == "auto" or provider in ocr_plugins()


def _known_llm(provider: str) -> bool:
    from docuflow.llm.plugins import llm_plugins

    return provider == "auto" or provider in llm_plugins()


class SelectionError(ValueError):
    """The requested model selection is not valid."""


def parse_key(key: str) -> tuple[str, str]:
    """Split a `"<provider>:<model>"` key. Model names may contain colons."""
    if ":" not in key:
        return key, "-"
    provider, model = key.split(":", 1)
    return provider, model or "-"


@dataclass(frozen=True)
class Selection:
    ocr_key: str | None = None
    llm_key: str | None = None


class SelectionStore:
    """Process-wide current selection, safe to read from worker threads."""

    def __init__(self, base: Settings) -> None:
        self._base = base
        self._selection = Selection()
        self._lock = threading.Lock()

    @property
    def base(self) -> Settings:
        return self._base

    def current(self) -> Selection:
        with self._lock:
            return self._selection

    def select(self, ocr_key: str | None = None, llm_key: str | None = None) -> Selection:
        """Validate and store a new selection."""
        if ocr_key:
            provider, _ = parse_key(ocr_key)
            if not _known_ocr(provider):
                raise SelectionError(f"'{provider}' is not a registered OCR engine.")
        if llm_key:
            provider, _ = parse_key(llm_key)
            if not _known_llm(provider):
                raise SelectionError(f"'{provider}' is not a registered agent backend.")

        with self._lock:
            self._selection = Selection(
                ocr_key=ocr_key or self._selection.ocr_key,
                llm_key=llm_key or self._selection.llm_key,
            )
            return self._selection

    def reset(self) -> Selection:
        with self._lock:
            self._selection = Selection()
            return self._selection

    @staticmethod
    def _first_available_llm(settings: Settings) -> str:
        """Which backend `auto` will actually pick."""
        from docuflow.llm.factory import build_llm
        from docuflow.llm.plugins import auto_order

        for key in auto_order():
            try:
                provider = build_llm(key, settings)
            except ValueError:
                continue
            if provider is None:  # the rules-only backend
                return key
            available, _ = provider.available()
            if available:
                return key
        return "rules"

    def settings(self) -> Settings:
        """Base settings with the current selection applied."""
        selection = self.current()
        updates: dict[str, object] = {}

        if selection.ocr_key:
            provider, model = parse_key(selection.ocr_key)
            updates["ocr_provider"] = provider
            field = _ocr_field(provider)
            if field and model != "-":
                updates[field] = model

        if selection.llm_key:
            provider, model = parse_key(selection.llm_key)
            updates["llm_provider"] = provider
            field = _llm_field(provider)
            if field and model != "-":
                updates[field] = model

        return self._base.model_copy(update=updates) if updates else self._base

    def describe(self) -> dict[str, str]:
        """The effective selection, for display.

        `auto` is resolved to the backend that will actually run, so the UI
        picker can highlight a real entry rather than the word "auto".
        """
        settings = self.settings()

        ocr_provider = settings.resolved_ocr_provider()
        ocr_field = _ocr_field(ocr_provider)
        ocr_model = getattr(settings, ocr_field) if ocr_field else "-"

        llm_provider = settings.llm_provider
        if llm_provider == "auto":
            llm_provider = self._first_available_llm(settings)
        llm_field = _llm_field(llm_provider)
        llm_model = getattr(settings, llm_field) if llm_field else "-"
        return {
            "ocr_key": f"{ocr_provider}:{ocr_model}",
            "ocr_provider": ocr_provider,
            "ocr_model": ocr_model,
            "llm_key": f"{llm_provider}:{llm_model}",
            "llm_provider": llm_provider,
            "llm_model": llm_model,
        }
