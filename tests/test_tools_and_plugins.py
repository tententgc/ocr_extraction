"""Agent tools, the tool-calling loop, plugin registries and model selection."""

from __future__ import annotations

import json

import pytest

from docuflow.agents.agent_loop import run_agent
from docuflow.agents.tools import build_toolbox, run_tool, to_schemas
from docuflow.llm.base import AssistantTurn, ChatMessage, LlmProvider, ToolCall
from docuflow.llm.plugins import auto_order, llm_plugins
from docuflow.ocr.plugins import fallback_order, ocr_plugins
from docuflow.runtime import SelectionError, SelectionStore, parse_key
from docuflow.store.records import RecordStore

PAGE_TEXT = (
    "Northwind Logistics Ltd.\n"
    "Tax ID: 0105561071237\n"
    "Invoice No.: NW-88214\n"
    "PO Reference: PO-2026-0455\n"
    "Subtotal 64,000.00 THB\n"
    "VAT 7% 4,480.00 THB\n"
    "Amount Due 86,480.00 THB\n"
)


@pytest.fixture
def toolbox(settings, master):
    return build_toolbox(
        ocr_text=PAGE_TEXT,
        master=master,
        records=RecordStore(settings.db_path),
        doc_type="invoice",
        document_id="DOC-TEST",
    )


class TestToolSchemas:
    def test_every_tool_exposes_a_valid_function_schema(self, toolbox):
        for schema in to_schemas(toolbox):
            assert schema["type"] == "function"
            function = schema["function"]
            assert function["name"] and function["description"]
            assert function["parameters"]["type"] == "object"
            # Schemas must be JSON-serialisable to cross the wire.
            json.dumps(schema)

    def test_tool_names_are_unique(self, toolbox):
        names = [tool.name for tool in toolbox]
        assert len(names) == len(set(names))

    def test_unknown_tool_returns_an_error_not_an_exception(self, toolbox):
        assert "error" in run_tool(toolbox, "no_such_tool", {})


class TestReadDocument:
    def test_finds_text_that_is_on_the_page(self, toolbox):
        result = run_tool(toolbox, "read_document", {"query": "Amount Due"})
        assert result["found"] is True
        assert "86,480.00" in result["matches"][0]["text"]

    def test_reports_text_that_is_absent(self, toolbox):
        result = run_tool(toolbox, "read_document", {"query": "68,480.00"})
        assert result["found"] is False

    def test_ignores_punctuation_and_case(self, toolbox):
        assert run_tool(toolbox, "read_document", {"query": "invoice no nw 88214"})["found"] is True

    def test_missing_query_is_an_error(self, toolbox):
        assert "error" in run_tool(toolbox, "read_document", {})


class TestMasterDataTools:
    def test_lookup_vendor_by_tax_id(self, toolbox):
        result = run_tool(toolbox, "lookup_vendor", {"tax_id": "0105561071237"})
        assert result["found"] is True
        assert result["vendor_id"] == "V-1002"
        assert result["bank_account"] == "9876543210"

    def test_lookup_vendor_reports_close_matches_when_absent(self, toolbox):
        result = run_tool(toolbox, "lookup_vendor", {"name": "Completely Unknown Trading"})
        assert result["found"] is False
        # Suggestions must be labelled as name-similarity only: given a bare
        # list, the agent reported the nearest name as "the actual vendor".
        assert "similar_names_only" in result
        assert "NOT" in result["warning"]

    def test_lookup_purchase_order(self, toolbox):
        result = run_tool(toolbox, "lookup_purchase_order", {"po_number": "PO-2026-0455"})
        assert result["found"] is True
        assert result["amount"] == 64_000.0
        assert result["vendor_name"] == "Northwind Logistics Ltd."

    def test_missing_purchase_order(self, toolbox):
        assert run_tool(toolbox, "lookup_purchase_order", {"po_number": "PO-9999"})["found"] is False

    def test_lookup_contract(self, toolbox):
        assert run_tool(toolbox, "lookup_contract", {"contract_number": "CT-2025-018"})["found"] is True

    def test_find_posted_records_when_nothing_posted(self, toolbox):
        assert run_tool(toolbox, "find_posted_records", {"reference": "NW-88214"})["found"] is False


class TestDeterministicTools:
    def test_arithmetic_detects_the_mismatch(self, toolbox):
        result = run_tool(
            toolbox,
            "check_arithmetic",
            {"subtotal": 64000, "tax_rate": 7, "tax_amount": 4480, "total": 86480},
        )
        assert result["consistent"] is False
        assert result["computed_total"] == 68_480.0
        assert result["difference"] == pytest.approx(-18_000.0)

    def test_arithmetic_accepts_a_correct_document(self, toolbox):
        result = run_tool(
            toolbox,
            "check_arithmetic",
            {"subtotal": 64000, "tax_rate": 7, "tax_amount": 4480, "total": 68480},
        )
        assert result["consistent"] is True

    def test_arithmetic_tolerates_missing_inputs(self, toolbox):
        assert run_tool(toolbox, "check_arithmetic", {})["consistent"] is True

    def test_tax_id_checksum(self, toolbox):
        assert run_tool(toolbox, "validate_tax_id", {"tax_id": "0105543000129"})["valid"] is True
        assert run_tool(toolbox, "validate_tax_id", {"tax_id": "0105543000128"})["valid"] is False

    def test_non_thai_tax_id_is_not_judged(self, toolbox):
        assert run_tool(toolbox, "validate_tax_id", {"tax_id": "GB123456789"})["valid"] is None


class FakeLlm(LlmProvider):
    """Scripted provider: replays a list of turns, recording what it was sent."""

    name = "fake"
    supports_tools = True

    def __init__(self, turns: list[AssistantTurn], fail_first: bool = False) -> None:
        self.turns = turns
        self.fail_first = fail_first
        self.calls = 0
        self.plain_calls = 0

    def complete(self, messages, max_tokens=2048, temperature=0.0) -> str:
        self.plain_calls += 1
        return '{"assessment": "answered without tools", "recommended_action": "review"}'

    def complete_with_tools(self, messages, tools, max_tokens=2048, temperature=0.0) -> AssistantTurn:
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("template cannot render tool calls")
        return self.turns[min(self.calls - 1, len(self.turns) - 1)]


class TestAgentLoop:
    def test_agent_calls_a_tool_then_answers(self, toolbox):
        turns = [
            AssistantTurn(tool_calls=(ToolCall(id="1", name="check_arithmetic",
                                               arguments={"subtotal": 64000, "tax_amount": 4480, "total": 86480}),)),
            AssistantTurn(content='{"assessment": "totals wrong", "recommended_action": "review"}'),
        ]
        llm = FakeLlm(turns)
        result = run_agent(llm, "sys", "user", toolbox)

        assert result.ok is True
        assert result.used_tools is True
        assert result.tool_names == ["check_arithmetic"]
        assert result.calls[0].result["consistent"] is False
        assert result.payload["recommended_action"] == "review"

    def test_agent_answering_immediately_is_fine(self, toolbox):
        llm = FakeLlm([AssistantTurn(content='{"assessment": "ok", "recommended_action": "post"}')])
        result = run_agent(llm, "sys", "user", toolbox)
        assert result.ok is True
        assert result.used_tools is False

    def test_first_turn_failure_degrades_to_a_plain_prompt(self, toolbox):
        # Reproduces the real Ollama case: the model advertises tool support but
        # its chat template cannot render tool calls.
        llm = FakeLlm([AssistantTurn(content="{}")], fail_first=True)
        result = run_agent(llm, "sys", "user", toolbox)
        assert llm.plain_calls == 1
        assert result.ok is True
        assert "without tools" in result.payload["assessment"]

    def test_loop_stops_at_the_iteration_limit(self, toolbox):
        forever = AssistantTurn(tool_calls=(ToolCall(id="1", name="read_document", arguments={"query": "Tax"}),))
        llm = FakeLlm([forever])
        result = run_agent(llm, "sys", "user", toolbox, max_iterations=3)
        assert result.iterations == 3
        assert len(result.calls) == 3

    def test_non_json_reply_is_reported_not_invented(self, toolbox):
        llm = FakeLlm([AssistantTurn(content="I think the invoice looks fine, honestly.")])
        result = run_agent(llm, "sys", "user", toolbox)
        assert result.ok is False
        assert result.payload is None

    def test_no_model_returns_a_clear_note(self, toolbox):
        result = run_agent(None, "sys", "user", toolbox)
        assert result.ok is False
        assert "no agent model" in result.note

    def test_provider_without_tool_support_uses_the_plain_path(self, toolbox):
        llm = FakeLlm([])
        llm.supports_tools = False
        result = run_agent(llm, "sys", "user", toolbox)
        assert llm.plain_calls == 1
        assert result.ok is True


class TestPluginRegistries:
    def test_core_ocr_engines_are_registered(self):
        keys = set(ocr_plugins())
        assert {"typhoon_api", "ollama", "local", "openai", "claude", "pdftext", "tesseract"} <= keys

    def test_core_agent_backends_are_registered(self):
        keys = set(llm_plugins())
        assert {"typhoon", "ollama", "openai", "claude", "local", "rules"} <= keys

    def test_rules_is_the_last_agent_fallback(self):
        assert auto_order()[-1] == "rules"

    def test_ocr_fallback_order_is_deterministic(self):
        assert fallback_order() == sorted(
            ocr_plugins(), key=lambda key: ocr_plugins()[key].fallback_priority
        )

    def test_every_plugin_builds_without_credentials(self, settings):
        for key, plugin in ocr_plugins().items():
            assert plugin.build(settings, None) is not None, key
        for key, plugin in llm_plugins().items():
            plugin.build(settings, None)  # `rules` legitimately returns None

    def test_unconfigured_cloud_engines_report_why(self, settings):
        provider = ocr_plugins()["openai"].build(settings, None)
        available, reason = provider.available()
        assert available is False
        assert "OPENAI_API_KEY" in reason


class TestModelSelection:
    def test_parse_key_keeps_colons_inside_model_names(self):
        assert parse_key("ollama:scb10x/typhoon-ocr1.5-3b:latest") == (
            "ollama",
            "scb10x/typhoon-ocr1.5-3b:latest",
        )

    def test_selecting_an_engine_changes_the_effective_settings(self, settings):
        store = SelectionStore(settings)
        store.select(ocr_key="ollama:my-ocr-model")
        effective = store.settings()
        assert effective.ocr_provider == "ollama"
        assert effective.ocr_ollama_model == "my-ocr-model"
        # The base configuration is never mutated.
        assert settings.ocr_provider == "pdftext"

    def test_selecting_an_agent_model(self, settings):
        store = SelectionStore(settings)
        store.select(llm_key="typhoon:some-model")
        effective = store.settings()
        assert effective.llm_provider == "typhoon"
        assert effective.llm_cloud_model == "some-model"

    def test_unknown_provider_is_refused(self, settings):
        store = SelectionStore(settings)
        with pytest.raises(SelectionError):
            store.select(ocr_key="not_a_provider:x")

    def test_reset_returns_to_configuration(self, settings):
        store = SelectionStore(settings)
        store.select(ocr_key="ollama:x", llm_key="typhoon:y")
        store.reset()
        assert store.settings().ocr_provider == settings.ocr_provider

    def test_describe_reports_the_effective_pair(self, settings):
        store = SelectionStore(settings)
        store.select(ocr_key="ollama:vision-model", llm_key="rules:-")
        described = store.describe()
        assert described["ocr_provider"] == "ollama"
        assert described["ocr_model"] == "vision-model"
        assert described["llm_provider"] == "rules"


class TestTranscriptTrimming:
    """Providers cap the message list; Typhoon rejects more than 50 items."""

    def _transcript(self, turns: int) -> list[dict]:
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        for index in range(turns):
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": f"c{index}", "type": "function",
                         "function": {"name": "read_document", "arguments": "{}"}}
                    ],
                }
            )
            messages.append({"role": "tool", "tool_call_id": f"c{index}", "name": "read_document", "content": "{}"})
        return messages

    def test_short_transcripts_are_untouched(self):
        from docuflow.agents.agent_loop import _trim_history

        messages = self._transcript(3)
        assert _trim_history(messages) is messages

    def test_long_transcripts_are_capped(self):
        from docuflow.agents.agent_loop import MAX_MESSAGES, _trim_history

        trimmed = _trim_history(self._transcript(30))
        assert len(trimmed) <= MAX_MESSAGES

    def test_system_prompt_and_original_request_survive(self):
        from docuflow.agents.agent_loop import _trim_history

        trimmed = _trim_history(self._transcript(30))
        assert trimmed[0]["role"] == "system"
        assert trimmed[1]["role"] == "user"

    def test_no_tool_result_is_orphaned(self):
        """A `tool` message without its `tool_calls` parent is a malformed request."""
        from docuflow.agents.agent_loop import _trim_history

        trimmed = _trim_history(self._transcript(30))
        for index, message in enumerate(trimmed):
            if message["role"] == "tool":
                assert index > 0
                assert "tool_calls" in trimmed[index - 1], "tool result lost its tool_call"


class TestStrictOcrMode:
    def test_fallback_chain_is_used_by_default(self, settings):
        from docuflow.ocr.factory import provider_chain

        assert len(provider_chain(settings)) > 1

    def test_strict_mode_measures_only_the_named_engine(self, settings):
        from docuflow.ocr.factory import provider_chain

        chain = provider_chain(settings.model_copy(update={"ocr_strict": True}))
        assert len(chain) == 1
        assert chain[0].name == "pdf-text-layer"


class TestEngineTrust:
    """Whether OCR output is trusted is a property of the engine, not its name."""

    def test_typhoon_and_the_pdf_text_layer_are_purpose_built(self, settings):
        from docuflow.ocr.factory import build_provider

        for key in ("typhoon_api", "pdftext"):
            assert build_provider(key, settings).purpose_built, key

    def test_a_general_vision_model_is_not(self, settings):
        from docuflow.ocr.factory import build_provider

        assert build_provider("claude", settings).purpose_built is False
        assert build_provider("tesseract", settings).purpose_built is False

    def test_ollama_is_judged_by_its_model_not_its_host(self, settings):
        """Typhoon OCR on Ollama is the same model as Typhoon OCR in the cloud.

        Treating it as a fallback engine degraded every locally-recognised run,
        which cost 0.1 confidence per field and blocked straight-through posting
        on documents that were read perfectly.
        """
        from docuflow.ocr.factory import build_provider

        typhoon = build_provider("ollama", settings.model_copy(
            update={"ocr_ollama_model": "scb10x/typhoon-ocr1.5-3b"}))
        general = build_provider("ollama", settings.model_copy(
            update={"ocr_ollama_model": "granite3.2-vision:2b"}))
        assert typhoon.purpose_built is True
        assert general.purpose_built is False


class TestVerificationBudget:
    """The cross-check must not be able to hold up the document."""

    def test_the_second_pass_gets_the_configured_timeout(self, settings):
        from docuflow.llm.factory import build_llm, resolve_verify_llm

        config = settings.model_copy(update={
            "llm_verify_provider": "ollama", "llm_verify_timeout_seconds": 12.0,
        })
        verify, _ = resolve_verify_llm(config, build_llm("openai", config))
        assert verify is not None and verify.timeout == 12.0

    def test_the_ocr_host_is_not_reused_for_verification(self, settings):
        """A second local model evicts the local OCR model instead of running
        beside it, so every page then pays a cold load."""
        from docuflow.llm.factory import resolve_verify_llm

        config = settings.model_copy(update={"ocr_provider": "ollama"})
        verify, _ = resolve_verify_llm(config, _FakeCloudLlm())
        assert getattr(verify, "base_url", "") != config.ollama_base_url


class _FakeCloudLlm:
    """Stands in for a reachable cloud primary without needing a key."""

    name = "fake-cloud"
    model = "fake-cloud-model"
    base_url = "https://example.invalid/v1"

    def available(self) -> tuple[bool, str]:
        return True, ""


class TestVerificationBackendPreference:
    """Which model gets the second reading, and what happens when it is down."""

    def _config(self, settings, **keys):
        return settings.model_copy(update={"ocr_provider": "pdftext", **keys})

    def test_claude_leads_when_it_is_reachable(self, settings):
        from docuflow.llm.factory import resolve_verify_llm

        config = self._config(settings, anthropic_api_key="x", openai_api_key="y")
        verify, _ = resolve_verify_llm(config, _FakeCloudLlm())
        assert "claude" in str(getattr(verify, "model", "")).lower()

    def test_it_falls_through_when_claude_is_unreachable(self, settings):
        """An outage or a missing key degrades the cross-check, never removes it."""
        from docuflow.llm.factory import resolve_verify_llm

        config = self._config(settings, anthropic_api_key="", openai_api_key="y")
        verify, _ = resolve_verify_llm(config, _FakeCloudLlm())
        assert verify is not None
        assert "claude" not in str(getattr(verify, "model", "")).lower()

    def test_no_cloud_key_falls_back_to_the_local_model(self, settings):
        """Only when it is not the model running OCR - see the contention test."""
        from docuflow.llm.factory import resolve_verify_llm

        config = self._config(settings, anthropic_api_key="", openai_api_key="", typhoon_api_key="")
        verify, _ = resolve_verify_llm(config, _FakeCloudLlm())
        assert verify is not None

    def test_falling_back_to_the_primary_says_independence_is_limited(self, settings):
        """The pass still helps - it reads label-first - but agreement between a
        model and itself is a much weaker signal, so it is never sold as more."""
        from docuflow.llm.factory import resolve_verify_llm
        from docuflow.llm.plugins import get_llm_plugin

        config = self._config(settings, anthropic_api_key="", openai_api_key="",
                              typhoon_api_key="", ocr_provider="ollama")
        # The only backend left is the local one, and it is already the primary.
        primary = get_llm_plugin("local").build(config, None)
        verify, note = resolve_verify_llm(config, primary)
        assert verify is primary
        assert "independence is limited" in note
