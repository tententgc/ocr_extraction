"""The discrepancy analyst, and what stops it being believed.

It runs only after a check has already failed, it may look anything up, and it
writes a paragraph a reviewer reads before they read anything else. That last
part is the risk: a value it invents is caught by grounding, a sentence it
invents is not.

One real run, on a contract whose renewal clause was extracted and quoted,
announced that the document "contains no mention of renewal", that both parties
were unregistered vendors when no vendor check had failed, and that the system
had wrongly flagged the contract as expiring soon when the rule had said it
expired four days ago. Three sentences, three inventions, printed above the
fields that disproved them.
"""

from __future__ import annotations

import json

import pytest

from docuflow.agents.reconciler import Reconciliation, _fields_summary, reconcile
from docuflow.llm.base import AssistantTurn, LlmProvider
from docuflow.models import RuleFinding


class Replying(LlmProvider):
    """Answers with one payload, and remembers what it was asked."""

    name = "replying"
    supports_tools = False

    def __init__(self, payload: dict) -> None:
        self.payload = json.dumps(payload)
        self.prompts: list[str] = []

    def complete(self, messages, max_tokens=2048, temperature=0.0) -> str:
        self.prompts.append("\n".join(message.content for message in messages))
        return self.payload

    def complete_with_tools(self, messages, tools, max_tokens=2048, temperature=0.0) -> AssistantTurn:
        return AssistantTurn(content=self.payload)


CONTRACT = {
    "contract_number": "CT-2025-018",
    "party_a": "Mirtphol Industries PCL",
    "party_b": "Chao Phraya Facility Services Co., Ltd.",
    "effective_date": "2025-08-01",
    "expiry_date": "2026-07-31",
    "auto_renew": "yes",
    "contract_value": 368_000.00,
    "currency": "THB",
}


PAGE = (
    "FACILITY SERVICES AGREEMENT\nContract No. CT-2025-018\n"
    "This Agreement is made as of 1 August 2025 between Mirtphol Industries PCL "
    "(\"Party A\") and Chao Phraya Facility Services Co., Ltd. (\"Party B\").\n"
    "3. TERM. The initial term ends on 31 July 2026 and shall renew automatically "
    "for successive periods of twelve (12) months unless either party gives written "
    "notice of non-renewal not less than sixty (60) days before expiry.\n"
    "4. FEES. The annual fee is THB 368,000."
)


@pytest.fixture
def expired(make_run):
    """A contract that has ended, with its renewal clause quoted from the page."""
    run = make_run(CONTRACT, doc_type="contract", text=PAGE)
    return run.evolve(
        findings=(
            RuleFinding(
                rule_id="contract_expired",
                passed=False,
                severity="warning",
                message="This agreement expired 4 days ago (2026-07-31).",
                fields=("expiry_date",),
            ),
        )
    )


class TestWhenItIsWorthCalling:
    """It writes a note for a reviewer. No reviewer, no note."""

    def test_a_held_document_with_a_failure_is_investigated(self, expired):
        from docuflow.agents.reconciler import should_run

        assert should_run(expired.evolve(decision="needs_review")) is True

    def test_a_document_that_posted_is_not(self, expired):
        """The warning is still recorded as a finding; the paragraph is not written."""
        from docuflow.agents.reconciler import should_run

        posted = expired.evolve(decision="auto_post")
        assert posted.failed_findings, "the finding still exists"
        assert should_run(posted) is False

    def test_a_clean_document_is_not_either(self, make_run):
        from docuflow.agents.reconciler import should_run

        assert should_run(make_run(CONTRACT, doc_type="contract")) is False

    def test_skipping_it_says_why(self, expired, master, settings):
        from docuflow.store.records import RecordStore

        result = reconcile(expired.evolve(decision="auto_post"),
                           Replying({"assessment": "should never be asked"}),
                           master=master, records=RecordStore(settings.db_path))
        assert result.ran is False
        assert "no reviewer" in result.note
        assert result.assessment == ""


class TestWhatTheAgentIsTold:
    def test_it_is_shown_the_page_text_behind_a_grounded_value(self, expired):
        """It cannot report a value as unsupported while looking at the support."""
        summary = _fields_summary(expired)
        assert "auto_renew = 'yes'" in summary
        assert "page says:" in summary

    def test_an_unverified_value_is_marked_as_one(self, make_run):
        run = make_run(CONTRACT, doc_type="contract")
        ungrounded = run.evolve(
            fields=tuple(
                item.model_copy(update={"grounded": False, "evidence": ""})
                if item.name == "auto_renew" else item
                for item in run.fields
            )
        )
        summary = _fields_summary(ungrounded)
        assert "[UNVERIFIED]" in summary


class TestCorrectionsAreCheckedAgainstThePage:
    def _reconcile(self, run, payload, master, records):
        return reconcile(run, Replying(payload), master=master, records=records)

    def test_a_correction_arguing_with_a_grounded_value_is_refused(
        self, expired, master, settings
    ):
        """"auto_renew is no, there is no renewal clause" - about a quoted clause."""
        from docuflow.store.records import RecordStore

        result = self._reconcile(expired, {
            "assessment": "The document contains no mention of renewal.",
            "recommended_action": "review",
            "corrections": [{
                "field": "auto_renew", "suggested_value": "no",
                "evidence": "no renewal term is mentioned",
                "reason": "No renewal term is mentioned in the contract text.",
            }],
        }, master, RecordStore(settings.db_path))

        correction = result.corrections[0]
        assert correction.grounded is False, "a proven value is not overturned by a guess"
        assert correction.contradicts_evidence is True
        assert result.unreliable is True

    def test_an_assessment_is_left_alone_when_a_correction_holds(
        self, expired, master, settings
    ):
        from docuflow.store.records import RecordStore

        result = self._reconcile(expired, {
            "assessment": "The fee matches the term clause.",
            "recommended_action": "review",
            "corrections": [{
                "field": "contract_value", "suggested_value": "368000.0",
                "evidence": "The annual fee is THB 368,000.",
                "reason": "Confirmed against the fees clause.",
            }],
        }, master, RecordStore(settings.db_path))

        # Same value as the one on the field, quoted from a line that is on the
        # page: nothing is being contradicted and nothing is being invented.
        assert result.corrections[0].contradicts_evidence is False
        assert result.corrections[0].grounded is True
        assert result.unreliable is False

    def test_an_agent_that_proposed_nothing_is_not_called_unreliable(self):
        """Silence is not a failed claim."""
        assert Reconciliation(assessment="Looks consistent.", ran=True).unreliable is False

    def test_a_correction_to_a_field_that_does_not_exist_is_dropped(
        self, expired, master, settings
    ):
        from docuflow.store.records import RecordStore

        result = self._reconcile(expired, {
            "assessment": "x",
            "corrections": [{"field": "profit_margin", "suggested_value": "12",
                             "evidence": "margin 12%", "reason": "y"}],
        }, master, RecordStore(settings.db_path))
        assert result.corrections == ()
