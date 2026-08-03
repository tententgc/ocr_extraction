"""Ollama server discovery.

Recognition itself goes through `vision_openai.OpenAiVisionOcr`, since Ollama
speaks the OpenAI protocol. What is Ollama-specific is finding out which models
are pulled and whether the server is up at all.

    ollama pull scb10x/typhoon-ocr1.5-3b
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

PING_TIMEOUT_SECONDS = 2.0
SHOW_TIMEOUT_SECONDS = 5.0


def _api_root(base_url: str) -> str:
    """Ollama's native API sits beside the `/v1` OpenAI-compatible endpoint."""
    return base_url.rstrip("/").removesuffix("/v1")


def server_models(base_url: str) -> list[str] | None:
    """Model names the local server has, or None when it is not running."""
    try:
        response = httpx.get(f"{_api_root(base_url)}/api/tags", timeout=PING_TIMEOUT_SECONDS)
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    return [model.get("name", "") for model in response.json().get("models", [])]


def model_present(base_url: str, model: str) -> tuple[bool, str]:
    """Whether `model` is pulled, tolerating a missing `:latest` suffix."""
    models = server_models(base_url)
    if models is None:
        return False, f"Ollama is not reachable at {base_url} (start it with: ollama serve)"
    wanted = model if ":" in model else f"{model}:latest"
    if wanted in models or model in models:
        return True, ""
    return False, f"model '{model}' is not pulled (run: ollama pull {model})"


def model_capabilities(base_url: str, model: str) -> list[str]:
    """Capabilities Ollama reports for a model, e.g. `["vision", "tools"]`.

    Treat this as a hint only: a model can advertise `tools` and still fail at
    generation time if its chat template cannot render tool calls.
    """
    try:
        response = httpx.post(
            f"{_api_root(base_url)}/api/show", json={"model": model}, timeout=SHOW_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.debug("could not read capabilities for %s: %s", model, exc)
        return []
    return list(response.json().get("capabilities") or [])
