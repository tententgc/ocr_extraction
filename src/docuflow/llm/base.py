"""LLM provider interface and tolerant JSON parsing.

Models return JSON wrapped in prose or fences more often than not, so the
parsing here is deliberately forgiving - but it never invents structure.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

MAX_TOOL_ARG_CHARS = 8_000

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
# Some models emit a reasoning block before the answer.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ToolCall:
    """One tool the model asked to run."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AssistantTurn:
    """A single assistant reply, which may request tool calls."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ToolsUnsupportedError(RuntimeError):
    """This provider or model cannot do tool calling."""


class LlmProvider(ABC):
    """Minimal chat interface: messages in, text out."""

    name: str = "base"
    supports_json: bool = False
    supports_tools: bool = False

    @abstractmethod
    def complete(self, messages: list[ChatMessage], max_tokens: int = 2048, temperature: float = 0.0) -> str:
        """Return the assistant's reply as raw text."""

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> AssistantTurn:
        """One turn of a tool-calling conversation.

        `messages` are raw OpenAI-format dicts because the loop has to append
        assistant tool-call messages and tool results between turns.
        """
        raise ToolsUnsupportedError(f"{self.name} does not support tool calling")

    def available(self) -> tuple[bool, str]:
        return True, ""


class LlmUnavailableError(RuntimeError):
    """The configured LLM cannot run."""


def _balanced_slice(text: str, open_char: str, close_char: str) -> str | None:
    """Extract the first balanced {...} or [...] block, ignoring braces in strings."""
    start = text.find(open_char)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json(text: str) -> Any | None:
    """Pull a JSON value out of a model reply, or return None.

    Returning None (rather than raising or guessing) matters: the caller treats
    an unparseable reply as "no extraction", which routes the document to a human
    instead of posting made-up values.
    """
    if not text:
        return None

    cleaned = _THINK_RE.sub("", text).strip()

    fenced = _FENCE_RE.search(cleaned)
    candidates = [fenced.group(1).strip()] if fenced else []
    candidates.append(cleaned)
    for opener, closer in (("{", "}"), ("[", "]")):
        block = _balanced_slice(cleaned, opener, closer)
        if block:
            candidates.append(block)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Trailing commas are the single most common defect.
            repaired = re.sub(r",(\s*[}\]])", r"\1", candidate)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue
    return None
