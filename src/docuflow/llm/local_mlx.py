"""Local instruct model via MLX, for a fully offline pipeline."""

from __future__ import annotations

import logging
import platform
import threading

from docuflow.llm.base import ChatMessage, LlmProvider, LlmUnavailableError

logger = logging.getLogger(__name__)

_load_lock = threading.Lock()
_cache: dict[str, tuple] = {}


def _load(model_id: str):
    if model_id in _cache:
        return _cache[model_id]
    with _load_lock:
        if model_id in _cache:
            return _cache[model_id]
        from mlx_lm import load

        logger.info("loading local LLM %s", model_id)
        _cache[model_id] = load(model_id)
        return _cache[model_id]


class LocalMlxChat(LlmProvider):
    """On-device Typhoon 2.5 (or any MLX-compatible instruct model)."""

    name = "typhoon-local"
    supports_json = False

    def __init__(self, model_id: str = "tawankri/typhoon2.5-qwen3-4b-mlx-4Bit") -> None:
        self.model_id = model_id

    def available(self) -> tuple[bool, str]:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return False, "local LLM needs Apple Silicon (MLX)"
        try:
            import mlx_lm  # noqa: F401
        except ImportError:
            return False, "mlx-lm is not installed; run: uv pip install -e '.[local]'"
        return True, ""

    def complete(self, messages: list[ChatMessage], max_tokens: int = 2048, temperature: float = 0.0) -> str:
        ok, reason = self.available()
        if not ok:
            raise LlmUnavailableError(reason)

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = _load(self.model_id)
        payload = [{"role": message.role, "content": message.content} for message in messages]
        prompt = tokenizer.apply_chat_template(payload, add_generation_prompt=True, tokenize=False)
        text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
            verbose=False,
        )
        return str(text).strip()
