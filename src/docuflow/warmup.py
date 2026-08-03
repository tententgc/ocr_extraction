"""Load the local models before the first document arrives.

Ollama evicts an idle model, and reloading a vision model costs far more than
recognising a page with it (168s against 15s on an M4). Nothing here changes a
result - it only decides whether the wait lands on the first document or on
server start, where nobody is watching.
"""

from __future__ import annotations

import logging
import threading

from docuflow.config import Settings
from docuflow.ocr.ollama_ocr import preload

logger = logging.getLogger(__name__)


def _agent_runs_on_ollama(settings: Settings) -> bool:
    """Whether the agent backend actually resolves to Ollama.

    `auto` may land anywhere, so ask the factory rather than guess - warming a
    2.5GB model the run will never call is worse than not warming at all.
    """
    if settings.llm_provider == "ollama":
        return True
    if settings.llm_provider != "auto":
        return False
    from docuflow.llm.factory import resolve_llm

    provider, _ = resolve_llm(settings)
    base_url = str(getattr(provider, "base_url", "") or "")
    return base_url.rstrip("/") == settings.ollama_base_url.rstrip("/")


def ollama_models_in_use(settings: Settings) -> list[tuple[str, int | None]]:
    """Ollama models this configuration will call, with the context each needs.

    The context matters: warming a model at one window and then calling it at
    another makes Ollama reload it, which is the cost the warm-up exists to
    avoid. Only OCR pins a window; the agent runs at the model's default.
    """
    models: list[tuple[str, int | None]] = []
    if settings.resolved_ocr_provider() == "ollama":
        models.append((settings.ocr_ollama_model, settings.ollama_num_ctx))
    if _agent_runs_on_ollama(settings):
        if settings.llm_ollama_model not in [name for name, _ in models]:
            models.append((settings.llm_ollama_model, None))
    return models


def warm_ollama(settings: Settings) -> list[str]:
    """Preload every Ollama model in use. Returns the ones that loaded."""
    if not settings.ollama_preload:
        return []
    return [
        model
        for model, num_ctx in ollama_models_in_use(settings)
        if preload(settings.ollama_base_url, model, settings.ollama_keep_alive, num_ctx)
    ]


def warm_ollama_in_background(settings: Settings) -> threading.Thread | None:
    """Warm the models without holding up startup.

    A failure here is not an error: Ollama may simply not be running, in which
    case the configured engine is unavailable and the chain falls back as usual.
    """
    if not settings.ollama_preload or not ollama_models_in_use(settings):
        return None

    def _run() -> None:
        try:
            warm_ollama(settings)
        except Exception as exc:  # noqa: BLE001 - warming must never break startup
            logger.info("model warm-up skipped: %s", exc)

    thread = threading.Thread(target=_run, name="ollama-warmup", daemon=True)
    thread.start()
    return thread
