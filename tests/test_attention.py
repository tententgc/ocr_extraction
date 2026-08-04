"""What needs attention, and the briefing written over it.

The agenda is computed; the model only ranks it. So the tests fall in two parts:
the queries have to find the right things and date them correctly, and the
briefing has to be unable to introduce anything the agenda does not contain.
"""

from __future__ import annotations

from datetime import date

import pytest

from docuflow.agents.briefing import summarise, write_briefing
from docuflow.models import RuleFinding
from docuflow.store.attention import Agenda, AttentionStore
from docuflow.store.documents import DocumentStore
from docuflow.store.records import RecordStore

TODAY = date(2026, 8, 4)

INVOICE = {
    "invoice_number": "INV-2026-0731",
    "vendor_name": "Siam Tech Supply Co., Ltd.",
    "vendor_tax_id": "0105543000129",
    "invoice_date": "2026-07-31",
    "currency": "THB",
    "total_amount": 200090.0,
    "purchase_order_ref": "PO-2026-0412",
}


@pytest.fixture
def documents(settings) -> DocumentStore:
    return DocumentStore(settings.db_path)


@pytest.fixture
def store(settings, master) -> AttentionStore:
    return AttentionStore(settings.db_path, today=TODAY)


def post(settings, documents, make_run, fields, vendor="V-1001", document_id="DOC-1"):
    run = make_run(fields).evolve(document_id=document_id, decision="auto_post")
    documents.save(run)
    posted = RecordStore(settings.db_path).post_record(run, vendor)
    documents.save(run.evolve(record_id=posted["record_id"]))
    return posted


class TestPaymentsDue:
    def test_an_invoice_past_its_due_date_is_overdue(self, settings, store, documents, make_run):
        post(settings, documents, make_run, {**INVOICE, "due_date": "2026-08-01"})
        item = next(i for i in store.agenda().items if i.kind == "payment_due")
        assert item.urgency == "overdue"
        assert item.days_left == -3
        assert item.detail == "Payment was due on 2026-08-01 — 3 day(s) ago."

    def test_one_due_next_week_is_due_soon(self, settings, store, documents, make_run):
        post(settings, documents, make_run, {**INVOICE, "due_date": "2026-08-09"})
        assert next(i for i in store.agenda().items if i.kind == "payment_due").urgency == "soon"

    def test_one_due_next_quarter_is_not_listed_yet(self, settings, store, documents, make_run):
        post(settings, documents, make_run, {**INVOICE, "due_date": "2026-12-01"})
        assert not [i for i in store.agenda().items if i.kind == "payment_due"]

    def test_the_item_names_the_document_it_came_from(self, settings, store, documents, make_run):
        """The whole point: the number on the dashboard opens the page behind it."""
        post(settings, documents, make_run, {**INVOICE, "due_date": "2026-08-05"}, document_id="DOC-SRC")
        item = next(i for i in store.agenda().items if i.kind == "payment_due")
        assert item.document_id == "DOC-SRC"
        assert "DOC-SRC" in item.source and "invoice" in item.source


class TestContractsExpiring:
    def test_a_contract_expiring_inside_the_horizon_is_listed(self, store):
        """CT-2025-018 in the demo register runs to 2026-07-31."""
        items = [i for i in store.agenda().items if i.kind == "contract_expiring"]
        assert any(item.item_id == "CTR-CT-2025-018" for item in items)

    def test_an_expired_contract_reads_as_overdue(self, store):
        item = next(i for i in store.agenda().items if i.item_id == "CTR-CT-2025-018")
        assert item.urgency == "overdue"
        assert item.days_left == -4
        assert item.detail == "Contract expired on 2026-07-31 — 4 day(s) ago."

    def test_a_register_entry_with_no_agreement_on_file_says_so(self, store):
        item = next(i for i in store.agenda().items if i.item_id == "CTR-CT-2025-018")
        assert item.document_id is None
        assert "no agreement on file" in item.source


class TestUnbilledCommitments:
    def test_an_open_po_with_no_invoice_against_it_is_flagged(self, store):
        items = [i for i in store.agenda().items if i.kind == "commitment_unbilled"]
        assert {"PO-PO-2026-0412", "PO-PO-2026-0455"} <= {item.item_id for item in items}

    def test_a_po_that_has_been_billed_drops_off(self, settings, store, documents, make_run):
        post(settings, documents, make_run, {**INVOICE, "due_date": "2026-09-30"})
        items = {i.item_id for i in store.agenda().items if i.kind == "commitment_unbilled"}
        assert "PO-PO-2026-0412" not in items, "an invoice bills against this PO"

    def test_a_commitment_carries_no_deadline(self, store):
        """Money already committed does not become overdue on its own."""
        item = next(i for i in store.agenda().items if i.kind == "commitment_unbilled")
        assert item.days_left is None
        assert item.urgency == "watch"


class TestStuckParties:
    def test_a_counterparty_is_grouped_with_the_rules_holding_it(self, settings, store, documents, make_run):
        for index in range(2):
            documents.save(
                make_run(INVOICE).evolve(
                    document_id=f"DOC-H{index}",
                    decision="needs_review",
                    findings=(
                        RuleFinding(rule_id="po_exists", passed=False, severity="blocking", message="no PO"),
                    ),
                )
            )
        party = store.agenda().parties[0]
        assert party.counterparty == "Siam Tech Supply Co., Ltd."
        assert party.documents == 2
        assert party.rules == ("po_exists",)
        assert set(party.document_ids) == {"DOC-H0", "DOC-H1"}

    def test_a_document_with_no_counterparty_is_still_reported(self, store, documents, make_run):
        documents.save(make_run({"total_amount": 10.0}).evolve(document_id="DOC-X", decision="needs_review"))
        assert store.agenda().parties[0].counterparty == "Unidentified counterparty"

    def test_a_posted_document_is_not_stuck(self, settings, store, documents, make_run):
        post(settings, documents, make_run, {**INVOICE, "due_date": "2026-09-30"})
        assert store.agenda().parties == ()


class TestBriefingGrounding:
    """The model ranks the agenda. It cannot add to it."""

    @pytest.fixture
    def agenda(self, settings, store, documents, make_run):
        post(settings, documents, make_run, {**INVOICE, "due_date": "2026-08-01"})
        return store.agenda()

    def test_without_a_model_the_page_still_says_something_true(self, agenda):
        briefing = write_briefing(agenda, llm=None)
        assert briefing.written_by == "rules"
        assert briefing.summary
        assert all(agenda.item(h.item_id) for h in briefing.highlights)

    def test_the_brackets_it_was_shown_are_not_held_against_it(self, agenda, monkeypatch):
        """The prompt lists ids as [PAY-...]; citing them back that way is correct."""
        real = agenda.items[0].item_id
        monkeypatch.setattr(
            "docuflow.agents.briefing.run_agent",
            lambda *a, **k: _result({
                "summary": f"Pay [{real}] first.",
                "highlights": [{"item_id": f"[{real}]", "note": "overdue"}],
            }),
        )
        briefing = write_briefing(agenda, llm=_FakeLlm())
        assert [h.item_id for h in briefing.highlights] == [real]
        assert briefing.dropped == ()

    def test_an_invented_item_id_is_dropped_not_shown(self, agenda, monkeypatch):
        """A claim the reader cannot open is not a claim."""
        real = agenda.items[0].item_id
        monkeypatch.setattr(
            "docuflow.agents.briefing.run_agent",
            lambda *a, **k: _result({
                "summary": "Two things need doing.",
                "highlights": [
                    {"item_id": real, "note": "pay this"},
                    {"item_id": "PAY-REC-DOES-NOT-EXIST", "note": "and this"},
                ],
            }),
        )
        briefing = write_briefing(agenda, llm=_FakeLlm())
        assert [h.item_id for h in briefing.highlights] == [real]
        assert briefing.dropped == ("PAY-REC-DOES-NOT-EXIST",)

    def test_an_empty_summary_falls_back_rather_than_showing_nothing(self, agenda, monkeypatch):
        monkeypatch.setattr(
            "docuflow.agents.briefing.run_agent", lambda *a, **k: _result({"summary": "   "})
        )
        briefing = write_briefing(agenda, llm=_FakeLlm())
        assert briefing.summary == summarise(agenda)
        assert briefing.written_by == "rules"

    def test_a_model_that_returns_nothing_usable_does_not_break_the_page(self, agenda, monkeypatch):
        monkeypatch.setattr("docuflow.agents.briefing.run_agent", lambda *a, **k: _result(None, "timeout"))
        briefing = write_briefing(agenda, llm=_FakeLlm())
        assert briefing.written_by == "rules"
        assert briefing.note == "timeout"

    def test_an_empty_agenda_is_said_plainly(self):
        """And without spending a model call to say it."""
        briefing = write_briefing(Agenda(as_of=TODAY, items=(), parties=()), llm=_FakeLlm())
        assert "no action outstanding" in briefing.summary.lower()
        assert briefing.written_by == "rules"


class _FakeLlm:
    name = "fake"
    model = "fake-model"
    supports_tools = False


def _result(payload, note=""):
    from docuflow.agents.agent_loop import AgentResult

    return AgentResult(payload=payload, ok=payload is not None, note=note)


class TestAttentionEndpoints:
    def test_the_agenda_is_served_without_a_model(self, client, settings, master, documents, make_run):
        post(settings, documents, make_run, {**INVOICE, "due_date": "2026-08-01"})
        payload = client.get("/api/attention").json()
        assert payload["items"]
        assert {"as_of", "items", "parties", "counts"} <= payload.keys()

    def test_every_item_carries_its_source(self, client, settings, master, documents, make_run):
        post(settings, documents, make_run, {**INVOICE, "due_date": "2026-08-01"})
        for item in client.get("/api/attention").json()["items"]:
            assert item["source"], f"{item['item_id']} does not say where it came from"

    def test_the_briefing_endpoint_returns_the_agenda_with_it(self, client, master):
        payload = client.post("/api/attention/briefing").json()
        assert "briefing" in payload and "items" in payload
        assert payload["briefing"]["summary"]
