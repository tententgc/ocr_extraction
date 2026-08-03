"""Evaluation comparison, reporting, and provider parameter negotiation."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from docuflow.evaluation.benchmark import parse_combination
from docuflow.evaluation.compare import compare_document, compare_value
from docuflow.evaluation.harness import _summarize
from docuflow.evaluation.report import print_report
from docuflow.schemas.registry import FieldSpec, get_schema

MONEY = FieldSpec(name="total_amount", label="Total", type="money", critical=True)
DATE = FieldSpec(name="invoice_date", label="Date", type="date")
TEXT = FieldSpec(name="vendor_name", label="Vendor", critical=True)


class TestValueComparison:
    def test_money_compares_numerically(self):
        assert compare_value(200090.00, 200090.0, MONEY).correct is True

    def test_money_tolerates_float_noise(self):
        assert compare_value(200090.00, 200090.001, MONEY).correct is True

    def test_money_rejects_a_real_difference(self):
        result = compare_value(200090.00, 86480.00, MONEY)
        assert result.correct is False
        assert result.status == "wrong"

    def test_dates_compare_as_iso(self):
        assert compare_value("2026-07-08", "2026-07-08", DATE).correct is True
        assert compare_value("2026-07-08", "2026-07-14", DATE).correct is False

    def test_text_ignores_case_and_punctuation(self):
        assert compare_value("Siam Tech Supply Co., Ltd.", "siam tech supply co ltd", TEXT).correct is True

    def test_a_single_character_ocr_slip_is_near_not_correct(self):
        # "Mirtphol" vs "Mirphol" - wrong, but recognisably the same value.
        result = compare_value("Mirtphol Industries PCL", "Mirphol Industries PCL", TEXT)
        assert result.correct is False
        assert result.near is True
        assert result.status == "near"

    def test_a_different_company_is_simply_wrong(self):
        result = compare_value("Chao Phraya Facility", "Summit Solutions Trading", TEXT)
        assert result.status == "wrong"
        assert result.near is False

    def test_missing_value_is_reported_as_missing(self):
        assert compare_value("INV-1", None, TEXT).status == "missing"

    def test_value_reported_where_none_exists_is_spurious(self):
        assert compare_value(None, "INV-1", TEXT).status == "spurious"

    def test_both_absent_is_correct(self):
        assert compare_value(None, None, TEXT).correct is True

    def test_compare_document_covers_every_expected_field(self):
        specs = {spec.name: spec for spec in get_schema("invoice").fields}
        results = compare_document(
            {"invoice_number": "INV-1", "total_amount": 100.0},
            {"invoice_number": "INV-1", "total_amount": 999.0},
            specs,
        )
        assert len(results) == 2
        assert {r.field: r.correct for r in results} == {"invoice_number": True, "total_amount": False}


def _document(**overrides):
    base = {
        "doc_id": "d1",
        "doc_type": "invoice",
        "classified_as": "invoice",
        "type_correct": True,
        "decision": "needs_review",
        "confidence": 0.5,
        "ocr_provider": "test",
        "ocr_ms": 10,
        "fields_total": 4,
        "fields_correct": 4,
        "fields_near": 0,
        "fields_wrong": 0,
        "fields_missing": 0,
        "wrong_critical": [],
        "silent_error": False,
        "expected_findings": [],
        "findings_fired": [],
        "expected_findings_caught": [],
        "expected_findings_missed": [],
        "comparisons": [
            {"field": "total_amount", "expected": 1, "actual": 1, "status": "correct", "similarity": 1.0}
        ],
    }
    return {**base, **overrides}


class TestSummary:
    def test_clean_run_reports_no_silent_errors(self):
        report = _summarize([_document(), _document(doc_id="d2")], elapsed=1.0)
        assert report["summary"]["silent_error_rate"] == 0.0
        assert report["summary"]["field_accuracy"] == 1.0
        assert report["summary"]["auto_post_precision"] == 1.0

    def test_a_wrong_value_that_auto_posted_is_a_silent_error(self):
        bad = _document(
            doc_id="bad",
            decision="auto_post",
            fields_correct=3,
            fields_wrong=1,
            wrong_critical=["total_amount"],
            silent_error=True,
        )
        report = _summarize([bad, _document(doc_id="ok")], elapsed=1.0)
        assert report["summary"]["silent_errors"] == 1
        assert report["summary"]["silent_error_rate"] == 0.5
        assert report["summary"]["silent_error_documents"] == ["bad"]
        # One of one auto-posted documents was wrong.
        assert report["summary"]["auto_post_precision"] == 0.0

    def test_a_wrong_value_caught_for_review_is_not_a_silent_error(self):
        caught = _document(
            doc_id="caught", decision="needs_review", fields_correct=3,
            fields_wrong=1, wrong_critical=["total_amount"], silent_error=False,
        )
        report = _summarize([caught], elapsed=1.0)
        assert report["summary"]["silent_error_rate"] == 0.0
        assert report["summary"]["field_accuracy"] == 0.75

    def test_defect_recall_counts_planted_defects(self):
        caught = _document(expected_findings=["a", "b"], expected_findings_caught=["a", "b"])
        missed = _document(doc_id="d2", expected_findings=["c"], expected_findings_missed=["c"])
        report = _summarize([caught, missed], elapsed=1.0)
        assert report["summary"]["planted_defects"] == 3
        assert report["summary"]["defect_recall"] == pytest.approx(2 / 3, abs=0.01)

    def test_field_breakdown_is_sorted_worst_first(self):
        docs = [
            _document(comparisons=[
                {"field": "good", "expected": 1, "actual": 1, "status": "correct", "similarity": 1.0},
                {"field": "bad", "expected": 1, "actual": 2, "status": "wrong", "similarity": 0.0},
            ])
        ]
        breakdown = _summarize(docs, elapsed=1.0)["field_breakdown"]
        assert list(breakdown)[0] == "bad"
        assert breakdown["bad"]["accuracy"] == 0.0

    def test_empty_run_does_not_crash(self):
        assert _summarize([], elapsed=0.0)["summary"]["documents"] == 0


class TestReportRendering:
    def test_report_prints_the_headline_metrics(self):
        report = _summarize([_document()], elapsed=1.0)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_report(report)
        output = buffer.getvalue()
        assert "ACCURACY REPORT" in output
        assert "Silent error rate" in output
        assert "PASS" in output

    def test_report_flags_a_failing_run(self):
        bad = _document(decision="auto_post", wrong_critical=["total_amount"], silent_error=True)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_report(_summarize([bad], elapsed=1.0))
        assert "FAIL" in buffer.getvalue()

    def test_empty_report_is_handled(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_report({"summary": {"documents": 0}})
        assert "No documents" in buffer.getvalue()


class TestBenchmarkSpec:
    def test_parses_an_ocr_and_llm_pair(self):
        combo = parse_combination("ocr=ollama:scb10x/typhoon-ocr1.5-3b,llm=typhoon:big-model")
        assert combo.ocr_provider == "ollama"
        assert combo.ocr_model == "scb10x/typhoon-ocr1.5-3b"
        assert combo.llm_provider == "typhoon"
        assert combo.llm_model == "big-model"

    def test_parses_an_engine_with_no_model(self):
        combo = parse_combination("ocr=pdftext")
        assert combo.ocr_provider == "pdftext"
        assert combo.ocr_model is None

    def test_applying_a_combination_overrides_only_that_engine(self, settings):
        combo = parse_combination("ocr=ollama:my-model,llm=claude:my-claude")
        applied = combo.apply(settings)
        assert applied.ocr_provider == "ollama"
        assert applied.ocr_ollama_model == "my-model"
        assert applied.llm_provider == "claude"
        assert applied.llm_anthropic_model == "my-claude"
        # The original is untouched.
        assert settings.ocr_provider == "pdftext"


class FakeBadRequest(Exception):
    """Stands in for openai.BadRequestError."""


class TestParameterNegotiation:
    """Endpoints reject different parameters; the adapter must learn, not guess."""

    def _chat(self, monkeypatch, failures: list[str]):
        from docuflow.llm import openai_compatible as module

        monkeypatch.setattr(module, "BadRequestError", FakeBadRequest, raising=False)

        class FakeCompletions:
            def __init__(self):
                self.seen: list[dict] = []

            def create(self, **kwargs):
                self.seen.append(kwargs)
                if failures:
                    raise FakeBadRequest(failures.pop(0))

                class Message:
                    content = '{"ok": true}'
                    tool_calls = None

                class Choice:
                    message = Message()
                    finish_reason = "stop"

                return type("Response", (), {"choices": [Choice()]})()

        completions = FakeCompletions()
        client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()

        chat = module.OpenAICompatibleChat(
            name="fake", model="m", base_url="http://x", api_key="k"
        )
        chat._client = client
        monkeypatch.setattr(chat, "available", lambda: (True, ""))
        return chat, completions

    def test_switches_to_max_completion_tokens_when_told(self, monkeypatch):
        # Reproduces gpt-5.1: "'max_tokens' is not supported ... use 'max_completion_tokens'".
        import docuflow.llm.openai_compatible as module

        monkeypatch.setattr(module, "BadRequestError", FakeBadRequest, raising=False)
        chat, completions = self._chat(
            monkeypatch, ["Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead."]
        )
        result = chat.complete([module.ChatMessage(role="user", content="hi")], max_tokens=100)
        assert result == '{"ok": true}'
        assert "max_tokens" in completions.seen[0]
        assert "max_completion_tokens" in completions.seen[1]
        assert chat._token_param == "max_completion_tokens"

    def test_drops_temperature_when_the_model_rejects_it(self, monkeypatch):
        import docuflow.llm.openai_compatible as module

        chat, completions = self._chat(monkeypatch, ["`temperature` is deprecated for this model."])
        chat.complete([module.ChatMessage(role="user", content="hi")], max_tokens=100)
        assert "temperature" in completions.seen[0]
        assert "temperature" not in completions.seen[1]

    def test_an_unrelated_error_is_raised_not_swallowed(self, monkeypatch):
        import docuflow.llm.openai_compatible as module

        chat, _ = self._chat(monkeypatch, ["insufficient_quota: you're out of credit"])
        with pytest.raises(FakeBadRequest):
            chat.complete([module.ChatMessage(role="user", content="hi")], max_tokens=100)
