"""Claude adapter.

Anthropic's Messages API differs from the OpenAI shape in three ways this
adapter absorbs: the system prompt is a top-level argument, tools use a flat
`input_schema`, and results come back as typed content blocks.
"""

from __future__ import annotations

import logging
from typing import Any

from docuflow.llm.base import (
    AssistantTurn,
    ChatMessage,
    LlmProvider,
    LlmUnavailableError,
    ToolCall,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_MODEL = "claude-opus-5"


class AnthropicChat(LlmProvider):
    """Claude via the Anthropic Messages API."""

    supports_json = True
    supports_tools = True

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.name = "claude"
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or None
        self.timeout = timeout
        self._client = None
        # Newer Claude models reject `temperature`; negotiated on first use.
        self._send_temperature = True

    def available(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "ANTHROPIC_API_KEY is not set"
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "the anthropic package is not installed (uv pip install anthropic)"
        return True, ""

    def _get_client(self):
        if self._client is None:
            import anthropic

            kwargs: dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def _ensure_ready(self) -> None:
        ok, reason = self.available()
        if not ok:
            raise LlmUnavailableError(reason)

    def _create(self, **kwargs: Any):
        """Call Messages, dropping parameters the model rejects.

        `temperature` is deprecated on the newest Claude models. Reading the
        rejection beats maintaining a list of model names.
        """
        import anthropic

        for _ in range(2):
            payload = dict(kwargs)
            if not self._send_temperature:
                payload.pop("temperature", None)
            try:
                return self._get_client().messages.create(**payload)
            except anthropic.BadRequestError as exc:
                if "temperature" in str(exc).lower() and self._send_temperature:
                    logger.info("claude: dropping temperature, which %s rejects", self.model)
                    self._send_temperature = False
                    continue
                raise
        raise RuntimeError("claude: could not agree request parameters with the endpoint")

    @staticmethod
    def _split_system(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """Claude takes the system prompt separately from the turn list."""
        system_parts = [
            message["content"] for message in messages if message.get("role") == "system"
        ]
        rest = [message for message in messages if message.get("role") != "system"]
        return "\n\n".join(part for part in system_parts if part), rest

    @staticmethod
    def _to_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translate OpenAI-shaped turns, including tool calls and results."""
        converted: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")

            if role == "tool":
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.get("tool_call_id", ""),
                                "content": str(message.get("content", "")),
                            }
                        ],
                    }
                )
                continue

            if role == "assistant" and message.get("tool_calls"):
                blocks: list[dict[str, Any]] = []
                if message.get("content"):
                    blocks.append({"type": "text", "text": message["content"]})
                for call in message["tool_calls"]:
                    function = call.get("function", {})
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        import json

                        try:
                            arguments = json.loads(arguments or "{}")
                        except json.JSONDecodeError:
                            arguments = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call.get("id", ""),
                            "name": function.get("name", ""),
                            "input": arguments if isinstance(arguments, dict) else {},
                        }
                    )
                converted.append({"role": "assistant", "content": blocks})
                continue

            converted.append({"role": role or "user", "content": message.get("content", "")})
        return converted

    def complete(self, messages: list[ChatMessage], max_tokens: int = 2048, temperature: float = 0.0) -> str:
        self._ensure_ready()
        raw = [{"role": message.role, "content": message.content} for message in messages]
        system, turns = self._split_system(raw)
        response = self._create(
            model=self.model,
            system=system or None,
            messages=self._to_anthropic_messages(turns) or [{"role": "user", "content": "."}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> AssistantTurn:
        self._ensure_ready()
        system, turns = self._split_system(messages)

        # OpenAI wraps the schema in `function`; Claude expects it flat.
        anthropic_tools = [
            {
                "name": tool["function"]["name"],
                "description": tool["function"].get("description", ""),
                "input_schema": tool["function"].get("parameters", {"type": "object", "properties": {}}),
            }
            for tool in tools
        ]

        response = self._create(
            model=self.model,
            system=system or None,
            messages=self._to_anthropic_messages(turns),
            tools=anthropic_tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input) if isinstance(block.input, dict) else {},
                    )
                )
        return AssistantTurn(content="".join(text_parts).strip(), tool_calls=tuple(calls))
