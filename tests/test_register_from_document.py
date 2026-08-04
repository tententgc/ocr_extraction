"""Turning a document into the master record it is asking for.

The dangerous version of this feature registers suppliers by itself: an invoice
from a company that does not exist creates the company, and then passes the
check written to catch it. So the tests that matter are the ones about who is
allowed to do it, what is refused, and what the reviewer is warned about before
a bank account reaches the master file.
"""

from __future__ import annotations

import pytest

from docuflow.masterdata.from_document import (
    RegistrationError,
    proposals,
    register,
)
from docuflow.masterdata.repository import MasterDataRepository
from docuflow.store.documents import DocumentStore

UNKNOWN_INVOICE = {
    "invoice_number": "INV-2026-0731",
    "vendor_name": "Andaman Marine Supply Co., Ltd.",
    "vendor_tax_id": "0107544000108",
    "bank_account": "3216549870",
    "currency": "THB",
    "payment_terms": "Net 30",
    "total_amount": 12_000.00,
}


@pytest.fixture
def master(settings) -> MasterDataRepository:
    return MasterDataRepository(settings.db_path)


@pytest.fixture
def documents(settings) -> DocumentStore:
    return DocumentStore(settings.db_path)


@pytest.fixture
def unknown(documents, make_run):
    run = make_run(UNKNOWN_INVOICE).evolve(document_id="DOC-1", decision="needs_review")
    documents.save(run)
    return run


class TestWhatIsOffered:
    def test_an_invoice_offers_its_counterparty(self, master, unknown):
        offer = proposals(unknown, master)[0]
        assert offer.kind == "vendors"
        assert offer.available
        assert offer.values["legal_name"] == "Andaman Marine Supply Co., Ltd."
        assert offer.values["tax_id"] == "0107544000108"

    def test_each_value_carries_the_quote_it_came_from(self, master, unknown):
        """The reviewer is confirming what the page says, not accepting a guess."""
        offer = proposals(unknown, master)[0]
        assert offer.evidence["legal_name"]

    def test_a_counterparty_already_registered_is_not_offered_again(self, master, make_run, documents):
        master.upsert_vendor({"legal_name": "Andaman Marine Supply Co., Ltd.", "tax_id": "0107544000108"})
        run = make_run(UNKNOWN_INVOICE).evolve(document_id="DOC-2")
        documents.save(run)

        offer = proposals(run, master)[0]
        assert not offer.available
        assert "Already registered" in offer.blocked

    def test_a_document_with_no_counterparty_name_offers_nothing(self, master, make_run, documents):
        run = make_run({"invoice_number": "INV-1"}).evolve(document_id="DOC-3")
        documents.save(run)
        assert not proposals(run, master)[0].available

    def test_a_contract_offers_the_contract_register(self, master, make_run, documents):
        run = make_run(
            {"contract_number": "CT-2026-009", "party_a": "Us PCL", "party_b": "Them Ltd",
             "effective_date": "2026-01-01", "expiry_date": "2026-12-31", "contract_value": 90_000.0},
            doc_type="contract",
        ).evolve(document_id="DOC-4")
        documents.save(run)

        offer = proposals(run, master)[0]
        assert offer.kind == "contracts" and offer.available
        assert offer.values["contract_number"] == "CT-2026-009"


class TestWarnings:
    def test_a_bank_account_read_off_an_invoice_is_warned_about(self, master, unknown):
        """This is the value invoice fraud exists to change."""
        offer = proposals(unknown, master)[0]
        assert any("fraud" in warning for warning in offer.warnings)

    def test_an_unverifiable_bank_account_is_warned_about_harder(self, master, make_run, documents):
        run = make_run(UNKNOWN_INVOICE, grounded=False).evolve(document_id="DOC-5")
        documents.save(run)
        offer = proposals(run, master)[0]
        assert "bank_account" in offer.ungrounded
        assert any("could not be verified" in warning for warning in offer.warnings)

    def test_a_tax_id_that_fails_its_check_digit_is_flagged(self, master, make_run, documents):
        run = make_run({**UNKNOWN_INVOICE, "vendor_tax_id": "0105545000124"}).evolve(document_id="DOC-6")
        documents.save(run)
        offer = proposals(run, master)[0]
        assert any("check digit" in warning for warning in offer.warnings)


class TestRegistering:
    def test_the_vendor_lands_in_the_master_file(self, settings, master, unknown):
        result = register(settings, "DOC-1", "vendors")

        assert result["outcome"] == "created"
        assert master.find_vendor("Andaman Marine Supply Co., Ltd.").matched

    def test_the_record_remembers_which_document_made_it(self, settings, master, unknown):
        """A supplier registered off its own invoice is a weaker record than one
        synced from the ERP, and the file should be able to say which is which."""
        register(settings, "DOC-1", "vendors")
        assert master.list_vendors()[0]["source_document_id"] == "DOC-1"

    def test_a_reviewer_edit_wins_over_what_was_read(self, settings, master, unknown):
        register(settings, "DOC-1", "vendors", {"bank_account": "9999999999"})
        assert master.list_vendors()[0]["bank_account"] == "9999999999"

    def test_an_edit_is_validated_like_any_other_input(self, settings, unknown):
        """The same validators the CSV importer uses — one standard, not two."""
        with pytest.raises(RegistrationError, match="must be one of"):
            register(settings, "DOC-1", "vendors", {"status": "probationary"})

    def test_registering_is_written_to_the_audit_trail(self, settings, unknown):
        from docuflow.store.records import RecordStore

        register(settings, "DOC-1", "vendors")
        actions = [entry["action"] for entry in RecordStore(settings.db_path).audit_for("DOC-1")]
        assert "master_record_created" in actions

    def test_a_document_type_that_cannot_become_this_record_is_refused(self, settings, unknown):
        with pytest.raises(RegistrationError, match="cannot become"):
            register(settings, "DOC-1", "contracts")

    def test_a_blocked_proposal_cannot_be_forced_through_the_api(self, settings, master, unknown):
        register(settings, "DOC-1", "vendors")
        with pytest.raises(RegistrationError, match="Already registered"):
            register(settings, "DOC-1", "vendors")

    def test_a_missing_document_is_refused(self, settings):
        with pytest.raises(RegistrationError, match="not found"):
            register(settings, "DOC-NOPE", "vendors")


class TestPurchaseOrders:
    def test_a_po_document_needs_its_vendor_registered_first(self, master, make_run, documents):
        run = make_run(
            {"po_number": "PO-2026-0501", "vendor_name": "Nobody Ltd", "total_amount": 48_150.0,
             "currency": "THB", "po_date": "2026-07-22"},
            doc_type="purchase_order",
        ).evolve(document_id="DOC-7")
        documents.save(run)

        offer = next(o for o in proposals(run, master) if o.kind == "purchase_orders")
        assert not offer.available
        assert "Register the counterparty first" in offer.blocked

    def test_once_the_vendor_exists_the_po_can_be_added(self, settings, master, make_run, documents):
        run = make_run(
            {"po_number": "PO-2026-0501", "vendor_name": "Nobody Ltd", "total_amount": 48_150.0,
             "currency": "THB", "po_date": "2026-07-22"},
            doc_type="purchase_order",
        ).evolve(document_id="DOC-8")
        documents.save(run)
        register(settings, "DOC-8", "vendors")

        result = register(settings, "DOC-8", "purchase_orders")

        assert result["outcome"] == "created"
        assert master.get_purchase_order("PO-2026-0501")["amount"] == 48_150.0


class TestEndpoint:
    def test_the_offer_is_served_with_the_document(self, client, unknown):
        payload = client.get("/api/documents/DOC-1").json()
        assert payload["register"][0]["kind"] == "vendors"
        assert payload["register"][0]["available"] is True

    def test_registering_through_the_api(self, client, master, unknown):
        response = client.post("/api/documents/DOC-1/register", json={"kind": "vendors"})
        assert response.status_code == 200
        assert master.find_vendor("Andaman Marine Supply Co., Ltd.").matched

    def test_a_refusal_is_a_conflict_not_a_crash(self, client, unknown):
        client.post("/api/documents/DOC-1/register", json={"kind": "vendors"})
        again = client.post("/api/documents/DOC-1/register", json={"kind": "vendors"})
        assert again.status_code == 409

    def test_an_unknown_kind_is_rejected_by_the_schema(self, client, unknown):
        response = client.post("/api/documents/DOC-1/register", json={"kind": "widgets"})
        assert response.status_code == 422
