"""API surface and the review workflow, exercised without any model calls."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from docuflow.ingest.fetch import FetchError, fetch_url
from docuflow.ingest.store import sniff_suffix, store_bytes
from docuflow.review import ReviewError, apply_review
from docuflow.store.documents import DocumentStore
from docuflow.store.records import RecordStore

GOOD_INVOICE = {
    "invoice_number": "INV-2026-0731",
    "invoice_date": "2026-07-08",
    "due_date": "2026-08-07",
    "vendor_name": "Siam Tech Supply Co., Ltd.",
    "vendor_tax_id": "0105543000129",
    "purchase_order_ref": "PO-2026-0412",
    "currency": "THB",
    "subtotal": 187_000.00,
    "tax_rate": 7.0,
    "tax_amount": 13_090.00,
    "total_amount": 200_090.00,
    "bank_account": "1234567890",
}


class TestUrlFetchSafety:
    """The platform accepts arbitrary URLs, so SSRF defence is load-bearing."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost/secret.pdf",
            "http://127.0.0.1:8000/admin",
            "http://169.254.169.254/latest/meta-data/",  # cloud metadata
            "http://10.0.0.5/internal.pdf",
            "http://192.168.1.1/config",
            "http://[::1]/local",
        ],
    )
    def test_private_and_loopback_addresses_are_refused(self, url):
        with pytest.raises(FetchError) as error:
            fetch_url(url, max_bytes=1024)
        assert "non-public" in str(error.value) or "resolve" in str(error.value)

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x.pdf", "gopher://x"])
    def test_non_http_schemes_are_refused(self, url):
        with pytest.raises(FetchError) as error:
            fetch_url(url, max_bytes=1024)
        assert "http" in str(error.value).lower()

    def test_empty_url_is_refused(self):
        with pytest.raises(FetchError):
            fetch_url("   ", max_bytes=1024)


class TestBlobStore:
    def test_detects_pdf_from_magic_bytes(self, settings):
        blob = store_bytes(b"%PDF-1.7\nbody", "invoice.txt", settings.blob_dir)
        # A PDF uploaded with the wrong extension still renders correctly.
        assert blob.path.suffix == ".pdf"
        assert blob.filename.endswith(".pdf")

    def test_same_content_stores_once(self, settings):
        first = store_bytes(b"%PDF-1.7\nsame", "a.pdf", settings.blob_dir)
        second = store_bytes(b"%PDF-1.7\nsame", "b.pdf", settings.blob_dir)
        assert first.sha256 == second.sha256
        assert first.path == second.path

    def test_path_traversal_in_a_filename_is_neutralised(self, settings):
        blob = store_bytes(b"%PDF-1.7\nx", "../../etc/passwd", settings.blob_dir)
        assert ".." not in str(blob.path)
        assert blob.path.parent == settings.blob_dir

    def test_empty_content_is_refused(self, settings):
        with pytest.raises(ValueError):
            store_bytes(b"", "empty.pdf", settings.blob_dir)

    @pytest.mark.parametrize(
        ("content", "suffix"),
        [(b"%PDF-1.4", ".pdf"), (b"\xff\xd8\xff\xe0", ".jpg"), (b"\x89PNG\r\n\x1a\n", ".png")],
    )
    def test_signature_sniffing(self, content, suffix):
        assert sniff_suffix(content) == suffix


class TestReviewWorkflow:
    def _store(self, settings, master, make_run, fields, findings=(), decision="needs_review"):
        run = make_run(fields).evolve(decision=decision, findings=tuple(findings), confidence=0.6)
        DocumentStore(settings.db_path).save(run)
        return run

    def test_correcting_a_value_then_approving_posts(self, settings, master, make_run):
        from docuflow.rules import RuleContext, run_rules
        from docuflow.schemas.registry import get_schema

        bad = {**GOOD_INVOICE, "total_amount": 86_480.00}
        run = self._store(settings, master, make_run, bad)

        # Approving with the bad total must be refused.
        with pytest.raises(ReviewError) as error:
            apply_review(settings, run.document_id, action="approve")
        assert "Cannot post" in str(error.value)

        outcome = apply_review(
            settings, run.document_id, edits={"total_amount": "200090.00"}, action="approve", reviewer="somchai"
        )
        assert outcome.record_id is not None
        assert outcome.applied == ("total_amount",)

        record = RecordStore(settings.db_path).get(outcome.record_id)
        assert record["amount"] == 200_090.00
        assert record["posted_by"] == "somchai"

    def test_reviewer_edit_is_recorded_for_learning(self, settings, master, make_run):
        from docuflow.store.corrections import CorrectionStore

        run = self._store(settings, master, make_run, {**GOOD_INVOICE, "total_amount": 86_480.00})
        apply_review(settings, run.document_id, edits={"total_amount": "200090.00"}, action="approve")
        hotspots = CorrectionStore(settings.db_path).error_hotspots()
        assert hotspots[0]["field_name"] == "total_amount"

    def test_uninterpretable_edit_is_reported_not_applied(self, settings, master, make_run):
        run = self._store(settings, master, make_run, GOOD_INVOICE)
        outcome = apply_review(settings, run.document_id, edits={"total_amount": "about ten thousand"}, action="save")
        assert outcome.applied == ()
        assert any("total_amount" in message for message in outcome.rejected_edits)

    def test_rejecting_marks_the_document_and_posts_nothing(self, settings, master, make_run):
        run = self._store(settings, master, make_run, GOOD_INVOICE)
        outcome = apply_review(settings, run.document_id, action="reject", note="duplicate of INV-1")
        assert outcome.run.decision == "rejected"
        assert outcome.record_id is None
        assert RecordStore(settings.db_path).list() == []

    def test_already_posted_document_cannot_be_posted_twice(self, settings, master, make_run):
        run = self._store(settings, master, make_run, GOOD_INVOICE)
        first = apply_review(settings, run.document_id, action="approve")
        assert first.record_id
        with pytest.raises(ReviewError) as error:
            apply_review(settings, run.document_id, action="approve")
        assert "already been posted" in str(error.value)

    def test_unknown_document_is_reported(self, settings, master):
        with pytest.raises(ReviewError):
            apply_review(settings, "DOC-NOPE", action="approve")


class TestRecordStore:
    def test_posting_requires_a_cleared_decision(self, settings, master, make_run):
        from docuflow.store.records import RecordPostingError

        run = make_run(GOOD_INVOICE).evolve(decision="needs_review")
        with pytest.raises(RecordPostingError):
            RecordStore(settings.db_path).post_record(run, "V-1001")

    def test_duplicate_detection_ignores_other_vendors(self, settings, master, make_run):
        store = RecordStore(settings.db_path)
        run = make_run(GOOD_INVOICE).evolve(decision="auto_post")
        DocumentStore(settings.db_path).save(run)
        store.post_record(run, "V-1001")

        # Same invoice number, different vendor - suppliers number independently.
        assert store.find_duplicate("invoice", "INV-2026-0731", "V-1002") is None
        assert store.find_duplicate("invoice", "INV-2026-0731", "V-1001") is not None

    def test_reference_key_ignores_formatting(self, settings, master, make_run):
        store = RecordStore(settings.db_path)
        run = make_run(GOOD_INVOICE).evolve(decision="auto_post")
        DocumentStore(settings.db_path).save(run)
        store.post_record(run, "V-1001")
        assert store.find_duplicate("invoice", "inv 2026 0731", "V-1001") is not None


class TestApiEndpoints:
    @pytest.fixture
    def client(self, settings, monkeypatch):
        import docuflow.api as api_module

        monkeypatch.setattr(api_module, "settings", settings)
        with TestClient(api_module.app) as test_client:
            yield test_client

    def test_status_reports_providers_and_thresholds(self, client):
        payload = client.get("/api/status").json()
        assert {"ocr_providers", "llm_providers", "thresholds", "stats"} <= payload.keys()
        assert payload["thresholds"]["auto_post_min_confidence"] == pytest.approx(0.90)

    def test_schemas_expose_field_definitions(self, client):
        payload = client.get("/api/schemas").json()
        names = [field["name"] for field in payload["invoice"]["fields"]]
        assert "total_amount" in names
        assert any(field["critical"] for field in payload["invoice"]["fields"])

    def test_index_page_renders(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "DocuFlow" in response.text

    def test_unknown_document_returns_404(self, client):
        assert client.get("/api/documents/DOC-NOPE").status_code == 404

    def test_empty_upload_is_rejected(self, client):
        response = client.post("/api/documents/upload", files={"file": ("x.pdf", b"", "application/pdf")})
        assert response.status_code == 400

    def test_url_ingest_rejects_private_addresses(self, client):
        response = client.post("/api/documents/url", json={"url": "http://127.0.0.1/x.pdf"})
        assert response.status_code == 400

    def test_records_and_metrics_start_empty(self, client):
        assert client.get("/api/records").json() == []
        assert client.get("/api/metrics").json()["total_documents"] == 0

    def test_master_data_is_seeded(self, client):
        vendors = client.get("/api/master/vendors").json()
        assert any(vendor["vendor_id"] == "V-1001" for vendor in vendors)
