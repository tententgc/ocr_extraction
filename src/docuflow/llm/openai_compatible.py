"""Chat provider for any OpenAI-compatible endpoint.

Typhoon Cloud, Ollama and OpenAI itself all speak this protocol, so one adapter
serves all three - they differ only in base URL, credential and model name.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import BadRequestError

from docuflow.llm.base import (
    AssistantTurn,
    ChatMessage,
    LlmProvider,
    LlmUnavailableError,
    ToolCall,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 300.0
# Reasoning models can consume the whole budget before answering.
RETRY_TOKEN_MULTIPLIER = 4
MAX_RETRY_TOKENS = 16_000
# How many parameter-shape corrections to attempt before giving up.
MAX_PARAM_NEGOTIATIONS = 4


class OpenAICompatibleChat(LlmProvider):
    """Configurable client for an OpenAI-style `/chat/completions` endpoint."""

    supports_json = True
    supports_tools = True

    def __init__(
        self,
        *,
        name: str,
        model: str,
        base_url: str,
        api_key: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        supports_tools: bool = True,
        missing_key_reason: str = "API key is not set",
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.supports_tools = supports_tools
        self._missing_key_reason = missing_key_reason
        self._extra_body = extra_body or {}
        self._client = None
        # Negotiated on first use; see `_adapt`.
        self._token_param = "max_tokens"
        self._send_temperature = True

    # --- availability ------------------------------------------------------
    def available(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, self._missing_key_reason
        try:
            import openai  # noqa: F401
        except ImportError:
            return False, "the openai package is not installed"
        return True, ""

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        return self._client

    def _ensure_ready(self) -> None:
        ok, reason = self.available()
        if not ok:
            raise LlmUnavailableError(reason)

    # --- request construction ---------------------------------------------
    def _create(self, messages: list[dict[str, Any]], max_tokens: int, temperature: float, **extra: Any):
        """Call the endpoint, adapting to per-model parameter quirks.

        Newer OpenAI models renamed `max_tokens` to `max_completion_tokens` and
        reject a non-default `temperature`. Rather than keep a list of model
        names that goes stale, read the rejection and retry - then remember the
        working shape for the rest of the process.
        """
        for _ in range(MAX_PARAM_NEGOTIATIONS):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                self._token_param: max_tokens,
                **extra,
            }
            if self._send_temperature:
                kwargs["temperature"] = temperature
            if self._extra_body:
                kwargs["extra_body"] = self._extra_body

            try:
                return self._get_client().chat.completions.create(**kwargs)
            except BadRequestError as exc:
                if not self._adapt(str(exc)):
                    raise
        raise RuntimeError(f"{self.name}: could not agree request parameters with the endpoint")

    def _adapt(self, message: str) -> bool:
        """Adjust one request parameter based on the endpoint's complaint."""
        lowered = message.lower()
        if "max_completion_tokens" in lowered and self._token_param == "max_tokens":
            logger.info("%s: switching to max_completion_tokens", self.name)
            self._token_param = "max_completion_tokens"
            return True
        if "temperature" in lowered and self._send_temperature:
            logger.info("%s: dropping temperature, which this model rejects", self.name)
            self._send_temperature = False
            return True
        return False

    def complete(self, messages: list[ChatMessage], max_tokens: int = 2048, temperature: float = 0.0) -> str:
        self._ensure_ready()
        payload = [{"role": message.role, "content": message.content} for message in messages]
        response = self._create(payload, max_tokens, temperature)
        choice = response.choices[0]
        content = (choice.message.content or "").strip()

        # Reasoning models spend their budget thinking and can return an empty
        # answer. The thinking is billed but not visible, so retry once with
        # more room rather than reporting a failure.
        if not content and choice.finish_reason == "length":
            retry_budget = min(max_tokens * RETRY_TOKEN_MULTIPLIER, MAX_RETRY_TOKENS)
            logger.info("%s returned no content within %s tokens; retrying with %s",
                        self.name, max_tokens, retry_budget)
            response = self._create(payload, retry_budget, temperature)
            content = (response.choices[0].message.content or "").strip()
        return content

    # --- tool calling ------------------------------------------------------
    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> AssistantTurn:
        self._ensure_ready()
        extra: dict[str, Any] = {"tools": tools, "tool_choice": "auto"} if tools else {}
        response = self._create(messages, max_tokens, temperature, **extra)
        message = response.choices[0].message
        calls: list[ToolCall] = []
        for call in getattr(message, "tool_calls", None) or []:
            raw_arguments = getattr(call.function, "arguments", "") or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                logger.debug("tool arguments were not valid JSON: %s", raw_arguments[:200])
                arguments = {}
            calls.append(
                ToolCall(
                    id=call.id or f"call_{len(calls)}",
                    name=call.function.name,
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )
        return AssistantTurn(content=(message.content or "").strip(), tool_calls=tuple(calls))
