"""Paged lists, and deleting a posted record.

The failure mode of not paging is silent — the table renders, it is simply
missing the rows nobody scrolled to — so the counts are what these tests pin
down: a page that reports the total honestly, and a page number past the end
that returns nothing rather than wrapping round.
"""

from __future__ import annotations

import pytest

from docuflow.store.documents import DocumentStore
from docuflow.store.paging import clamp_page, slice_all
from docuflow.store.records import RecordPostingError, RecordStore

INVOICE = {
    "invoice_number": "INV-2026-0731",
    "vendor_name": "Siam Tech Supply Co., Ltd.",
    "vendor_tax_id": "0105543000129",
    "invoice_date": "2026-07-31",
    "currency": "THB",
    "total_amount": 1_000.00,
}


@pytest.fixture
def documents(settings) -> DocumentStore:
    return DocumentStore(settings.db_path)


@pytest.fixture
def many(documents, make_run):
    """Twenty-five documents: two full pages and a short third."""
    for index in range(25):
        documents.save(
            make_run({**INVOICE, "invoice_number": f"INV-{index:03d}"}).evolve(
                document_id=f"DOC-{index:03d}",
                decision="needs_review" if index % 2 else "auto_post",
            )
        )
    return documents


class TestClamping:
    def test_a_page_below_one_is_the_first_page(self):
        assert clamp_page(0, 10) == (1, 10, 0)
        assert clamp_page(-5, 10) == (1, 10, 0)

    def test_an_absurd_page_size_is_capped(self):
        _, per_page, _ = clamp_page(1, 100_000)
        assert per_page == 200

    def test_the_offset_follows_the_page(self):
        assert clamp_page(4, 10) == (4, 10, 30)

    def test_an_empty_list_is_page_one_of_one(self):
        """"Page 1 of 0" is a number nobody can act on."""
        assert slice_all([], page=1, per_page=10)["pages"] == 1


class TestDocumentPaging:
    def test_a_page_holds_exactly_the_page_size(self, client, many):
        payload = client.get("/api/documents?page=1&per_page=10").json()
        assert len(payload["rows"]) == 10
        assert payload["total"] == 25
        assert payload["pages"] == 3

    def test_the_last_page_holds_the_remainder(self, client, many):
        payload = client.get("/api/documents?page=3&per_page=10").json()
        assert len(payload["rows"]) == 5

    def test_a_page_past_the_end_is_empty_rather_than_wrapping(self, client, many):
        payload = client.get("/api/documents?page=9&per_page=10").json()
        assert payload["rows"] == []
        assert payload["total"] == 25

    def test_pages_do_not_overlap(self, client, many):
        first = {row["document_id"] for row in client.get("/api/documents?page=1&per_page=10").json()["rows"]}
        second = {row["document_id"] for row in client.get("/api/documents?page=2&per_page=10").json()["rows"]}
        assert first.isdisjoint(second)

    def test_the_total_counts_the_filter_not_the_page(self, client, many):
        """Showing "10 of 10" when the filter matches 12 is worse than no number."""
        payload = client.get("/api/documents?decision=needs_review&per_page=10").json()
        assert payload["total"] == 12
        assert len(payload["rows"]) == 10


class TestRecordPaging:
    @pytest.fixture
    def posted(self, settings, master, documents, make_run):
        store = RecordStore(settings.db_path)
        for index in range(12):
            run = make_run({**INVOICE, "invoice_number": f"INV-{index:03d}"}).evolve(
                document_id=f"REC-DOC-{index:03d}", decision="auto_post"
            )
            documents.save(run)
            posted = store.post_record(run, "V-1001")
            documents.save(run.evolve(record_id=posted["record_id"]))
        return store

    def test_records_page(self, client, posted):
        payload = client.get("/api/records?per_page=10").json()
        assert len(payload["rows"]) == 10
        assert payload["total"] == 12

    def test_master_data_pages(self, client, master):
        payload = client.get("/api/master/vendors?page=1&per_page=2").json()
        assert len(payload["rows"]) == 2
        assert payload["total"] == 5


class TestDeletingARecord:
    @pytest.fixture
    def posted(self, settings, master, documents, make_run):
        run = make_run(INVOICE).evolve(document_id="DOC-P", decision="auto_post")
        documents.save(run)
        record = RecordStore(settings.db_path).post_record(run, "V-1001")
        documents.save(run.evolve(record_id=record["record_id"]))
        return record["record_id"]

    def test_the_entry_goes_and_the_document_comes_back(self, settings, documents, posted):
        """The page is still the evidence for the entry that used to exist."""
        RecordStore(settings.db_path).delete(posted)

        assert RecordStore(settings.db_path).get(posted) is None
        assert documents.get("DOC-P") is not None
        assert documents.get("DOC-P").record_id is None

    def test_the_document_is_actionable_again(self, settings, documents, posted):
        """Clearing only the column would leave the stored run saying "posted",
        and everything that refuses to touch a posted document reads that one."""
        from docuflow.deletion import delete_document

        RecordStore(settings.db_path).delete(posted)
        delete_document(settings, "DOC-P")      # would raise if it still looked posted
        assert documents.get("DOC-P") is None

    def test_it_is_written_to_the_audit_trail(self, settings, posted):
        RecordStore(settings.db_path).delete(posted, actor="tester")
        actions = [e["action"] for e in RecordStore(settings.db_path).audit_for("DOC-P")]
        assert "record_deleted" in actions

    def test_deleting_something_that_is_not_there(self, settings):
        with pytest.raises(RecordPostingError, match="not found"):
            RecordStore(settings.db_path).delete("REC-NOPE")

    def test_through_the_api(self, client, settings, posted):
        assert client.delete(f"/api/records/{posted}").status_code == 200
        assert client.get("/api/records").json()["rows"] == []

    def test_a_missing_record_is_a_404(self, client):
        assert client.delete("/api/records/REC-NOPE").status_code == 404


class TestSampleFormats:
    def test_both_renderings_are_offered(self, client, settings):
        """A PDF can carry a text layer; an image cannot."""
        from docuflow.samples.generator import generate_corpus

        generate_corpus(settings.sample_dir)
        sample = next(s for s in client.get("/api/samples").json() if s["available"])
        assert set(sample["formats"]) == {"pdf", "image"}

    def test_an_unknown_format_is_rejected_by_the_route(self, client):
        response = client.post("/api/documents/sample/inv-clean-classic?format=fax")
        assert response.status_code == 422
