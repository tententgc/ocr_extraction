"""The tool-calling loop.

Runs a bounded conversation in which the model may call tools before answering.
Falls back to a single prompted turn when the backend cannot do tool calling, so
the same agent code works against Ollama, Typhoon, OpenAI, Claude or nothing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from docuflow.agents.tools import Tool, run_tool, to_schemas
from docuflow.llm.base import (
    ChatMessage,
    LlmProvider,
    LlmUnavailableError,
    ToolsUnsupportedError,
    extract_json,
)

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 6
MAX_TOOL_RESULT_CHARS = 2_500
# Providers cap the message list (Typhoon rejects more than 50). A loop that
# makes several tool calls per turn reaches that quickly, so the transcript is
# trimmed rather than allowed to grow until the request is refused.
MAX_MESSAGES = 40


@dataclass(frozen=True)
class ToolInvocation:
    """One tool call and its result, kept for the audit trail."""

    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass(frozen=True)
class AgentResult:
    """Outcome of an agent run."""

    text: str = ""
    payload: dict[str, Any] | None = None
    calls: tuple[ToolInvocation, ...] = ()
    iterations: int = 0
    used_tools: bool = False
    ok: bool = False
    note: str = ""

    @property
    def tool_names(self) -> list[str]:
        return [call.name for call in self.calls]


def _truncate(value: str) -> str:
    return value if len(value) <= MAX_TOOL_RESULT_CHARS else value[:MAX_TOOL_RESULT_CHARS] + " …[truncated]"


def _trim_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the transcript under the provider's message limit.

    The system prompt and the original request are always kept. Older turns are
    dropped from the front of the middle section - but only at a turn boundary,
    because an assistant message carrying `tool_calls` must keep the matching
    `tool` replies next to it or the request is rejected as malformed.
    """
    if len(messages) <= MAX_MESSAGES:
        return messages

    head, tail = messages[:2], messages[2:]
    budget = MAX_MESSAGES - len(head)

    # Walk back from the end and cut at the first assistant turn that fits.
    start = len(tail) - budget
    while start < len(tail) and tail[start].get("role") != "assistant":
        start += 1
    if start >= len(tail):  # no clean boundary; keep the most recent turn only
        start = max(0, len(tail) - 2)

    logger.debug("trimmed agent transcript from %s to %s messages", len(messages), len(head) + len(tail) - start)
    return head + tail[start:]


def _run_without_tools(
    llm: LlmProvider, system: str, user: str, max_tokens: int
) -> AgentResult:
    """Single-turn fallback for backends without tool calling."""
    try:
        text = llm.complete(
            [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)],
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except (LlmUnavailableError, Exception) as exc:  # noqa: BLE001
        logger.warning("agent call failed: %s", exc)
        return AgentResult(ok=False, note=f"agent call failed: {exc}")

    payload = extract_json(text)
    return AgentResult(
        text=text,
        payload=payload if isinstance(payload, dict) else None,
        ok=payload is not None,
        iterations=1,
        note="" if payload is not None else "reply was not valid JSON",
    )


def run_agent(
    llm: LlmProvider | None,
    system: str,
    user: str,
    tools: list[Tool],
    max_iterations: int = MAX_ITERATIONS,
    max_tokens: int = 1500,
) -> AgentResult:
    """Let the model use `tools`, then return its final JSON answer."""
    if llm is None:
        return AgentResult(ok=False, note="no agent model configured")

    if not tools or not llm.supports_tools:
        return _run_without_tools(llm, system, user, max_tokens)

    schemas = to_schemas(tools)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    calls: list[ToolInvocation] = []

    for iteration in range(1, max_iterations + 1):
        messages = _trim_history(messages)
        try:
            turn = llm.complete_with_tools(messages, schemas, max_tokens=max_tokens, temperature=0.0)
        except ToolsUnsupportedError:
            logger.info("%s cannot use tools; falling back to a single turn", llm.name)
            return _run_without_tools(llm, system, user, max_tokens)
        except Exception as exc:  # noqa: BLE001 - the agent is advisory
            logger.warning("agent turn %s failed: %s", iteration, exc)
            if calls:
                break
            # Nothing was accomplished yet. A model whose chat template cannot
            # render tool calls fails exactly here, and Ollama's advertised
            # capabilities do not reliably predict it - so try the plain path
            # once before giving up.
            logger.info("retrying %s without tools after a first-turn failure", llm.name)
            result = _run_without_tools(llm, system, user, max_tokens)
            return AgentResult(
                text=result.text,
                payload=result.payload,
                iterations=iteration,
                ok=result.ok,
                note=result.note or f"tool calling unavailable ({str(exc)[:120]}); answered without tools",
            )

        if not turn.wants_tools:
            payload = extract_json(turn.content)
            return AgentResult(
                text=turn.content,
                payload=payload if isinstance(payload, dict) else None,
                calls=tuple(calls),
                iterations=iteration,
                used_tools=bool(calls),
                ok=payload is not None,
                note="" if payload is not None else "final reply was not valid JSON",
            )

        # Record the assistant's tool request, then answer each call.
        messages.append(
            {
                "role": "assistant",
                "content": turn.content or None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    }
                    for call in turn.tool_calls
                ],
            }
        )
        for call in turn.tool_calls:
            result = run_tool(tools, call.name, call.arguments)
            calls.append(ToolInvocation(name=call.name, arguments=call.arguments, result=result))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": _truncate(json.dumps(result, ensure_ascii=False, default=str)),
                }
            )

    # Out of iterations: ask once more for the answer, with tools switched off.
    messages.append(
        {
            "role": "user",
            "content": "Stop using tools now and give your final JSON answer.",
        }
    )
    try:
        final = llm.complete_with_tools(_trim_history(messages), [], max_tokens=max_tokens, temperature=0.0)
        payload = extract_json(final.content)
    except Exception as exc:  # noqa: BLE001
        return AgentResult(
            calls=tuple(calls),
            iterations=max_iterations,
            used_tools=bool(calls),
            ok=False,
            note=f"agent did not finish: {exc}",
        )

    return AgentResult(
        text=final.content,
        payload=payload if isinstance(payload, dict) else None,
        calls=tuple(calls),
        iterations=max_iterations,
        used_tools=bool(calls),
        ok=payload is not None,
        note="" if payload is not None else "agent ran out of turns before answering",
    )
