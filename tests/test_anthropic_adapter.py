"""What the Claude adapter puts on the wire.

Anthropic's Messages API differs from the OpenAI shape in ways this adapter
absorbs, and the ones worth pinning are the ones whose failure mode is a 400
that reads like something else entirely.

The system prompt is the example. It is optional, but sending an empty one is
not: `system=None` comes back as *"system: Input should be a valid array"*,
which looks like a credentials or billing problem and was diagnosed as one for
most of an evening. Every prompt in the pipeline carries a system message, so
only the paths that build their own turn list ever hit it - which is exactly the
kind of gap a test is for.

The client is stubbed rather than called. What is being checked is the payload,
and a network round trip would test Anthropic's servers instead.
"""

from __future__ import annotations

import pytest

from docuflow.llm.anthropic_chat import AnthropicChat
from docuflow.llm.base import ChatMessage


class Block:
    def __init__(self, text: str) -> None:
        self.type, self.text = "text", text


class Reply:
    def __init__(self, text: str = "ok") -> None:
        self.content = [Block(text)]
        self.stop_reason = "end_turn"


class RecordingClient:
    """Captures the keyword arguments the adapter would send."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return Reply()


@pytest.fixture
def claude() -> tuple[AnthropicChat, RecordingClient]:
    pytest.importorskip("anthropic")
    provider = AnthropicChat(api_key="test-key", model="claude-opus-5")
    client = RecordingClient()
    provider._client = client
    return provider, client


class TestTheSystemPrompt:
    def test_absent_means_the_key_is_not_sent_at_all(self, claude):
        """Not sent as null. Messages rejects null with a message about arrays.

        Found when a probe with no system prompt failed against a working key,
        and the 400 was read as the account being out of credit.
        """
        provider, client = claude
        provider.complete([ChatMessage(role="user", content="Say ok")])

        assert "system" not in client.calls[0]

    def test_present_means_it_is_sent_as_a_top_level_argument(self, claude):
        """Not as a message. Claude takes it out of the turn list entirely."""
        provider, client = claude
        provider.complete([
            ChatMessage(role="system", content="Be terse."),
            ChatMessage(role="user", content="Say ok"),
        ])

        call = client.calls[0]
        assert call["system"] == "Be terse."
        assert [message["role"] for message in call["messages"]] == ["user"]

    def test_several_are_joined_rather_than_dropped(self, claude):
        provider, client = claude
        provider.complete([
            ChatMessage(role="system", content="Be terse."),
            ChatMessage(role="system", content="Answer in English."),
            ChatMessage(role="user", content="Say ok"),
        ])

        assert client.calls[0]["system"] == "Be terse.\n\nAnswer in English."

    def test_the_tool_calling_path_is_no_different(self, claude):
        """The same omission, on the path that actually runs the agent rules."""
        provider, client = claude
        provider.complete_with_tools(
            [{"role": "user", "content": "Look it up"}],
            [{"function": {"name": "lookup", "description": "", "parameters": {}}}],
        )

        assert "system" not in client.calls[0]


class TestTheShapeOfTheRequest:
    def test_tools_are_flattened_out_of_the_openai_wrapper(self, claude):
        provider, client = claude
        provider.complete_with_tools(
            [{"role": "user", "content": "x"}],
            [{"function": {
                "name": "lookup_vendor",
                "description": "find a vendor",
                "parameters": {"type": "object", "properties": {"name": {"type": "string"}}},
            }}],
        )

        tool = client.calls[0]["tools"][0]
        assert tool["name"] == "lookup_vendor"
        assert tool["description"] == "find a vendor"
        assert tool["input_schema"]["properties"] == {"name": {"type": "string"}}
        assert "function" not in tool

    def test_a_turn_list_that_would_be_empty_still_carries_one_message(self, claude):
        """Messages refuses an empty list, and a system-only prompt is a real
        thing to send."""
        provider, client = claude
        provider.complete([ChatMessage(role="system", content="Be terse.")])

        assert client.calls[0]["messages"], "never sends an empty turn list"


class TestTemperature:
    def test_it_is_sent_until_the_model_says_otherwise(self, claude):
        provider, client = claude
        provider.complete([ChatMessage(role="user", content="x")], temperature=0.0)

        assert "temperature" in client.calls[0]

    def test_a_model_that_rejects_it_is_retried_without_it(self, claude):
        """Reading the rejection beats maintaining a list of model names."""
        import anthropic

        provider, client = claude
        refusals = {"count": 0}
        real_create = client.create

        import httpx

        def create(**kwargs):
            if "temperature" in kwargs and refusals["count"] == 0:
                refusals["count"] += 1
                raise anthropic.BadRequestError(
                    message="temperature: unsupported",
                    response=httpx.Response(
                        400, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
                    ),
                    body=None,
                )
            return real_create(**kwargs)

        client.create = create
        provider.complete([ChatMessage(role="user", content="x")], temperature=0.0)

        assert refusals["count"] == 1
        assert "temperature" not in client.calls[-1]
        assert provider._send_temperature is False, "and it is remembered"
