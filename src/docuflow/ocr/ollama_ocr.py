"""Ollama: server discovery and recognition over the native API.

Ollama speaks the OpenAI protocol, but its compatibility layer cannot express
the two settings that decide whether local OCR is usable:

  * `keep_alive` is accepted and silently ignored, so the model is evicted five
    minutes after the last page. The next document then pays a cold load -
    measured at 168s against 15s warm on an M4, an 11x penalty on almost every
    document in a demo, where documents arrive minutes apart.
  * `num_ctx` cannot be set at all and defaults to 4096. A page image alone
    costs ~3,100 tokens, leaving under 1,000 for the transcription; a dense
    page overruns that and Ollama shifts the context out from under the model,
    which starts transcribing text it can no longer see.

Both are ordinary fields on the native endpoint, so recognition goes there.

    ollama pull scb10x/typhoon-ocr1.5-3b
"""

from __future__ import annotations

import base64
import logging

import httpx

from docuflow.ocr.base import OcrProvider, OcrUnavailableError, PageImage
from docuflow.ocr.prompts import v15_prompt

logger = logging.getLogger(__name__)

PING_TIMEOUT_SECONDS = 2.0
SHOW_TIMEOUT_SECONDS = 5.0
GENERATE_TIMEOUT_SECONDS = 600.0
PRELOAD_TIMEOUT_SECONDS = 300.0
# Published Typhoon OCR 1.5 decoding parameters.
TEMPERATURE = 0.1
TOP_P = 0.6


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


def preload(base_url: str, model: str, keep_alive: str, num_ctx: int | None = None) -> bool:
    """Load `model` into memory now and hold it there.

    A request with no prompt is Ollama's documented way to warm a model. Called
    at server start so the first document does not pay the cold load in front of
    whoever is watching the demo.

    `num_ctx` has to match what recognition will ask for. Ollama reloads a model
    when the requested context changes, so warming at the default window and
    then recognising at a larger one pays the load twice - measured as 30.8s on
    the first page instead of 8.6s.
    """
    payload: dict[str, object] = {"model": model, "keep_alive": keep_alive}
    if num_ctx:
        payload["options"] = {"num_ctx": num_ctx}
    try:
        response = httpx.post(
            f"{_api_root(base_url)}/api/generate",
            json=payload,
            timeout=PRELOAD_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.info("could not preload %s: %s", model, exc)
        return False
    logger.info("preloaded %s (keep_alive=%s)", model, keep_alive)
    return True


class OllamaVisionOcr(OcrProvider):
    """Recognise a page with a local vision model over Ollama's native API."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        num_ctx: int,
        num_predict: int,
        keep_alive: str,
        figure_language: str = "English",
    ) -> None:
        self.name = f"ollama:{model}"
        self.model = model
        self.base_url = base_url
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.keep_alive = keep_alive
        self.figure_language = figure_language

    @property
    def purpose_built(self) -> bool:
        """Ollama hosts anything, so the model decides, not the host.

        `scb10x/typhoon-ocr1.5-3b` is the same document-OCR model the cloud
        engine runs and deserves the same trust; `granite3.2-vision` pointed at
        an invoice is a general vision model guessing. Naming a purpose-built
        OCR model after the task it performs is near-universal, so the name is
        the signal available without a hand-maintained list of every model.
        """
        return "ocr" in self.model.lower()

    def available(self) -> tuple[bool, str]:
        return model_present(self.base_url, self.model)

    def recognize(self, page: PageImage) -> str:
        ok, reason = self.available()
        if not ok:
            raise OcrUnavailableError(reason)

        payload = {
            "model": self.model,
            "prompt": v15_prompt(self.figure_language),
            "images": [base64.b64encode(page.path.read_bytes()).decode("ascii")],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        try:
            response = httpx.post(
                f"{_api_root(self.base_url)}/api/generate",
                json=payload,
                timeout=GENERATE_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OcrUnavailableError(f"Ollama request failed: {exc}") from exc

        body = response.json()
        # The context is what the model could actually see. Overrunning it means
        # Ollama dropped part of the page, so say so rather than returning text
        # that looks complete.
        used = int(body.get("prompt_eval_count") or 0) + int(body.get("eval_count") or 0)
        if used >= self.num_ctx:
            logger.warning(
                "%s: page %s filled the %s-token context; raise OLLAMA_NUM_CTX",
                self.name,
                page.page_number,
                self.num_ctx,
            )
        return (body.get("response") or "").strip()
