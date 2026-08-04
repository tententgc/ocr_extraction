"""Per-document-type insights.

The numbers on this page decide where someone spends their next day of work, so
the ones that could quietly mislead - a document counted twice because it broke
two rules, a field ranked by how often it is blank rather than how much it
matters - are what these tests pin down.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from docuflow.models import RuleFinding
from docuflow.store.documents import DocumentStore
from docuflow.store.insights import InsightStore

GOOD_INVOICE = {
    "invoice_number": "INV-2026-0731",
    "vendor_name": "Siam Tech Supply Co., Ltd.",
    "invoice_date": "2026-07-31",
    "total_amount": 200090.0,
    "currency": "THB",
}


def failing(rule_id: str, severity: str = "warning") -> RuleFinding:
    return RuleFinding(rule_id=rule_id, passed=False, severity=severity, message=rule_id)


@pytest.fixture
def store(settings) -> InsightStore:
    return InsightStore(settings.db_path)


@pytest.fixture
def documents(settings) -> DocumentStore:
    return DocumentStore(settings.db_path)


class TestSummaryPerType:
    def test_each_type_is_counted_on_its_own(self, store, documents, make_run):
        documents.save(make_run(GOOD_INVOICE).evolve(document_id="DOC-1", decision="auto_post"))
        documents.save(
            make_run({"contract_title": "Facility"}, doc_type="contract")
            .evolve(document_id="DOC-2", decision="needs_review")
        )

        by_type = {entry["doc_type"]: entry for entry in store.describe_types()}
        assert by_type["invoice"]["total"] == 1
        assert by_type["invoice"]["auto_posted"] == 1
        assert by_type["invoice"]["auto_post_rate"] == 1.0
        assert by_type["contract"]["held"] == 1
        assert by_type["contract"]["auto_post_rate"] == 0.0

    def test_a_type_names_itself_from_its_schema(self, store, documents, make_run):
        """The page should say "Invoice / Tax Invoice", not the database word."""
        documents.save(make_run(GOOD_INVOICE).evolve(document_id="DOC-1"))
        entry = store.describe_types()[0]
        assert entry["title"] and entry["blurb"]
        assert entry["record_kind"] == "payable"

    def test_an_empty_database_reports_nothing_rather_than_failing(self, store):
        assert store.describe_types() == []


class TestFailingRules:
    def test_rules_are_counted_in_documents(self, store, documents, make_run):
        """Two documents failing one rule is 2; one document failing it twice is 1.

        Counting findings instead would report more held documents than exist,
        which is the number a person uses to decide what to fix.
        """
        documents.save(
            make_run(GOOD_INVOICE).evolve(
                document_id="DOC-1", findings=(failing("vendor_known"), failing("vendor_known"))
            )
        )
        documents.save(
            make_run(GOOD_INVOICE).evolve(document_id="DOC-2", findings=(failing("vendor_known"),))
        )
        rules = store.failing_rules_by_type()["invoice"]
        assert [rule["documents"] for rule in rules if rule["rule_id"] == "vendor_known"] == [2]

    def test_a_passing_rule_is_not_listed(self, store, documents, make_run):
        documents.save(
            make_run(GOOD_INVOICE).evolve(
                document_id="DOC-1",
                findings=(RuleFinding(rule_id="totals_add_up", passed=True, severity="blocking", message="ok"),),
            )
        )
        assert store.failing_rules_by_type() == {}

    def test_two_blocking_failures_still_block_one_document(self, store, documents, make_run):
        documents.save(
            make_run(GOOD_INVOICE).evolve(
                document_id="DOC-1",
                findings=(failing("po_exists", "blocking"), failing("duplicate_document", "blocking")),
            )
        )
        assert store.blocked_documents_by_type() == {"invoice": 1}

    def test_a_warning_does_not_count_as_blocked(self, store, documents, make_run):
        documents.save(
            make_run(GOOD_INVOICE).evolve(document_id="DOC-1", findings=(failing("vendor_known"),))
        )
        assert store.blocked_documents_by_type() == {}
        assert store.describe_types()[0]["blocked"] == 0


class TestFieldGaps:
    def test_empty_and_unverified_are_different_problems(self, store, documents, make_run):
        """Nothing read is an extraction gap; read but unsupported is a trust problem."""
        documents.save(
            make_run({**GOOD_INVOICE, "purchase_order_ref": None}).evolve(document_id="DOC-1")
        )
        documents.save(
            make_run({**GOOD_INVOICE, "purchase_order_ref": "PO-9"}, grounded=False)
            .evolve(document_id="DOC-2")
        )
        gaps = {gap["field_name"]: gap for gap in store.field_gaps_by_type()["invoice"]}
        assert gaps["purchase_order_ref"]["missing"] == 1
        assert gaps["purchase_order_ref"]["unverified"] == 1

    def test_a_field_that_is_fine_is_not_a_gap(self, store, documents, make_run):
        documents.save(make_run(GOOD_INVOICE).evolve(document_id="DOC-1"))
        assert store.field_gaps_by_type() == {}

    def test_critical_fields_rank_above_incidental_ones(self, store, documents, make_run):
        """A missing total is not the same size of problem as missing payment terms."""
        documents.save(
            make_run({"payment_terms": None, "total_amount": None}).evolve(document_id="DOC-1")
        )
        ranked = [gap["field_name"] for gap in store.field_gaps_by_type()["invoice"]]
        assert ranked == ["total_amount", "payment_terms"]

    def test_a_field_outside_the_schema_is_still_reported(self, store, documents, make_run):
        """A custom or renamed field has no spec; it is listed, just not ranked up."""
        documents.save(make_run({"mystery_field": None}).evolve(document_id="DOC-1"))
        gap = store.field_gaps_by_type()["invoice"][0]
        assert gap["field_name"] == gap["label"] == "mystery_field"
        assert gap["critical"] is False

    def test_a_gap_carries_the_field_label(self, store, documents, make_run):
        documents.save(make_run({"total_amount": None}).evolve(document_id="DOC-1"))
        gap = store.field_gaps_by_type()["invoice"][0]
        assert gap["label"] == "Total"
        assert gap["critical"] is True


class TestInsightsEndpoint:
    @pytest.fixture
    def client(self, settings, monkeypatch):
        import docuflow.api as api_module

        monkeypatch.setattr(api_module, "settings", settings)
        with TestClient(api_module.app) as test_client:
            yield test_client

    def test_types_come_back_with_their_breakdowns(self, client, documents, make_run):
        documents.save(
            make_run({**GOOD_INVOICE, "purchase_order_ref": None}).evolve(
                document_id="DOC-1", decision="needs_review", findings=(failing("po_exists", "blocking"),)
            )
        )
        payload = client.get("/api/insights").json()
        invoice = next(entry for entry in payload["types"] if entry["doc_type"] == "invoice")
        assert invoice["blocked"] == 1
        assert invoice["failing_rules"][0]["rule_id"] == "po_exists"
        assert any(gap["field_name"] == "purchase_order_ref" for gap in invoice["field_gaps"])
        assert payload["total_documents"] == 1

    def test_every_rule_id_can_be_explained(self, client):
        """An unlabelled rule id on this page is a dead end for the reader."""
        rules = client.get("/api/insights").json()["rules"]
        assert rules
        assert all(summary for summary in rules.values())

    def test_an_empty_database_returns_an_empty_list(self, client):
        payload = client.get("/api/insights").json()
        assert payload["types"] == []
        assert payload["total_documents"] == 0
