"""Re-evaluating and deleting documents already in the system.

Both operations touch things that are expensive to get wrong: one can post to
the ledger, the other removes files. The tests that matter here are the refusals
- a posted document must survive both, and a policy change must not be able to
overturn a person's rejection by itself.
"""

from __future__ import annotations

import pytest

from docuflow.deletion import DeletionError, delete_document
from docuflow.models import RuleFinding, TraceStep
from docuflow.policy import PolicyStore
from docuflow.revalidate import RevalidationError, reevaluate, reevaluate_many
from docuflow.store.corrections import CorrectionStore
from docuflow.store.documents import DocumentStore
from docuflow.store.records import RecordStore

GOOD_INVOICE = {
    "invoice_number": "INV-2026-0731",
    "vendor_name": "Siam Tech Supply Co., Ltd.",
    "vendor_tax_id": "0105543000129",
    "invoice_date": "2026-07-31",
    "due_date": "2026-08-30",
    "currency": "THB",
    "subtotal": 187000.0,
    "tax_rate": 7.0,
    "tax_amount": 13090.0,
    "total_amount": 200090.0,
    "purchase_order_ref": "PO-2026-0412",
}
# The same invoice with nothing to match against a purchase order.
NO_PO_INVOICE = {key: value for key, value in GOOD_INVOICE.items() if key != "purchase_order_ref"}


@pytest.fixture
def documents(settings) -> DocumentStore:
    return DocumentStore(settings.db_path)


@pytest.fixture
def held(settings, master, documents, make_run):
    """One invoice genuinely stuck in the queue.

    It really does fail `purchase_order_required` - 200,090 with no PO reference,
    above the 50,000 threshold - rather than carrying a stale finding, so that
    re-running the rules reproduces the block instead of clearing it by accident.
    """
    run = make_run(NO_PO_INVOICE).evolve(
        document_id="DOC-HELD",
        decision="needs_review",
        confidence=0.45,
        findings=(
            RuleFinding(
                rule_id="purchase_order_required",
                passed=False,
                severity="blocking",
                message="A purchase order is required above 50,000.",
            ),
        ),
    )
    documents.save(run)
    return run


class TestReevaluate:
    def test_disabling_the_failing_rule_releases_the_document(self, settings, held, documents):
        """The whole point: the queue was filled by a policy that changed."""
        PolicyStore(settings).update(disabled_rules=["purchase_order_required"])

        result = reevaluate(PolicyStore(settings).settings(), "DOC-HELD")

        assert result.before["blocking"] == ["purchase_order_required"]
        assert result.after["blocking"] == []
        assert result.changed
        assert documents.get("DOC-HELD").decision == "auto_post"

    def test_a_released_document_posts_to_the_ledger(self, settings, held):
        """Re-evaluation ends where processing would have ended, not one step short."""
        PolicyStore(settings).update(disabled_rules=["purchase_order_required"])

        result = reevaluate(PolicyStore(settings).settings(), "DOC-HELD")

        assert result.record_id
        assert RecordStore(settings.db_path).get(result.record_id)

    def test_nothing_changes_when_the_policy_did_not(self, settings, held, documents):
        result = reevaluate(settings, "DOC-HELD")
        assert not result.changed
        assert result.record_id is None
        assert documents.get("DOC-HELD").decision == "needs_review"

    def test_the_run_records_what_it_was_judged_against(self, settings, held, documents):
        """A surprising outcome has to be traceable to the policy that produced it."""
        reevaluate(settings, "DOC-HELD")
        step = next(s for s in documents.get("DOC-HELD").trace if s.step == "reevaluate")
        assert step.detail["policy"]["auto_post_min_confidence"] == settings.auto_post_min_confidence
        assert "before" in step.detail and "after" in step.detail

    def test_a_posted_document_is_refused(self, settings, master, documents, make_run):
        run = make_run(GOOD_INVOICE).evolve(document_id="DOC-POSTED", decision="auto_post")
        documents.save(run)
        posted = RecordStore(settings.db_path).post_record(run, "V-1001")
        documents.save(run.evolve(record_id=posted["record_id"]))

        with pytest.raises(RevalidationError, match="already posted"):
            reevaluate(settings, "DOC-POSTED")

    def test_a_missing_document_is_refused(self, settings):
        with pytest.raises(RevalidationError, match="not found"):
            reevaluate(settings, "DOC-NOPE")

    def test_a_document_a_person_rejected_never_posts_itself(self, settings, master, documents, make_run):
        """A policy change may reopen a rejection. It may not overturn one."""
        rejected = make_run(GOOD_INVOICE).evolve(
            document_id="DOC-REJECTED", decision="rejected", confidence=0.95
        ).with_step(TraceStep(step="review", status="error", summary="Rejected by reviewer."))
        documents.save(rejected)

        result = reevaluate(settings, "DOC-REJECTED")

        assert result.after["decision"] == "needs_review"
        assert result.record_id is None
        assert documents.get("DOC-REJECTED").record_id is None

    def test_a_document_the_pipeline_refused_is_not_treated_as_a_human_decision(
        self, settings, master, documents, make_run
    ):
        """An unreadable file was rejected by routing, so re-judging it is free."""
        broken = make_run(GOOD_INVOICE).evolve(document_id="DOC-BROKEN", decision="rejected")
        documents.save(broken)
        assert reevaluate(settings, "DOC-BROKEN").after["decision"] == "auto_post"


class TestReevaluateMany:
    def test_the_whole_queue_is_re_judged_at_once(self, settings, master, documents, make_run):
        for index in range(3):
            documents.save(
                make_run({**NO_PO_INVOICE, "invoice_number": f"INV-{index}"}).evolve(
                    document_id=f"DOC-{index}",
                    decision="needs_review",
                    findings=(
                        RuleFinding(
                            rule_id="purchase_order_required", passed=False,
                            severity="blocking", message="PO required.",
                        ),
                    ),
                )
            )
        PolicyStore(settings).update(disabled_rules=["purchase_order_required"])

        bulk = reevaluate_many(PolicyStore(settings).settings()).to_json()

        assert bulk["checked"] == 3
        assert bulk["changed"] == 3
        assert len(bulk["posted"]) == 3

    def test_posted_documents_are_left_alone(self, settings, master, documents, make_run):
        """The scope is the queue; the ledger is not re-judged."""
        run = make_run(GOOD_INVOICE).evolve(document_id="DOC-POSTED", decision="auto_post")
        documents.save(run)
        posted = RecordStore(settings.db_path).post_record(run, "V-1001")
        documents.save(run.evolve(record_id=posted["record_id"]))

        bulk = reevaluate_many(settings).to_json()
        assert bulk["checked"] == 0

    def test_an_empty_queue_is_not_an_error(self, settings):
        bulk = reevaluate_many(settings).to_json()
        assert bulk["checked"] == 0
        assert "nothing changed" in bulk["summary"]


class TestDelete:
    def test_the_document_and_its_corrections_go_together(self, settings, held, documents):
        CorrectionStore(settings.db_path).record(
            "DOC-HELD", "invoice", "total_amount", 1.0, 2.0
        )

        result = delete_document(settings, "DOC-HELD")

        assert result.corrections == 1
        assert documents.get("DOC-HELD") is None
        assert CorrectionStore(settings.db_path).for_document("DOC-HELD") == []

    def test_the_audit_trail_outlives_the_document(self, settings, held):
        """That the document existed and was deleted is exactly what an audit asks."""
        delete_document(settings, "DOC-HELD", actor="tester")
        entries = RecordStore(settings.db_path).audit_for("DOC-HELD")
        assert any(entry["action"] == "document_deleted" for entry in entries)

    def test_a_posted_document_cannot_be_deleted(self, settings, master, documents, make_run):
        run = make_run(GOOD_INVOICE).evolve(document_id="DOC-POSTED", decision="auto_post")
        documents.save(run)
        posted = RecordStore(settings.db_path).post_record(run, "V-1001")
        documents.save(run.evolve(record_id=posted["record_id"]))

        with pytest.raises(DeletionError, match="cannot be deleted"):
            delete_document(settings, "DOC-POSTED")
        assert documents.get("DOC-POSTED") is not None

    def test_a_missing_document_is_refused(self, settings):
        with pytest.raises(DeletionError, match="not found"):
            delete_document(settings, "DOC-NOPE")

    def test_files_are_removed_with_the_row(self, settings, documents, make_run, tmp_path):
        blob = settings.blob_dir / "abc.pdf"
        blob.write_bytes(b"%PDF-1.4 test")
        pages = settings.page_dir / "DOC-FILES"
        pages.mkdir(parents=True, exist_ok=True)
        (pages / "page-1.jpg").write_bytes(b"jpeg")
        documents.save(make_run(GOOD_INVOICE).evolve(document_id="DOC-FILES", blob_path=str(blob)))

        result = delete_document(settings, "DOC-FILES")

        assert not blob.exists()
        assert not pages.exists()
        assert len(result.files) == 2

    def test_a_blob_shared_with_another_document_is_kept(self, settings, documents, make_run):
        """Ingest storage is content-addressed: two uploads of one file share a blob."""
        blob = settings.blob_dir / "shared.pdf"
        blob.write_bytes(b"%PDF-1.4 shared")
        for document_id in ("DOC-A", "DOC-B"):
            documents.save(
                make_run(GOOD_INVOICE).evolve(document_id=document_id, blob_path=str(blob))
            )

        result = delete_document(settings, "DOC-A")

        assert blob.exists(), "deleting one copy must not take the page from the other"
        assert result.kept and "same content" in result.kept[0]


class TestStartingOver:
    """Clearing a test run.

    Deleting a ledger entry leaves its document behind on purpose - the page is
    the evidence for the entry that used to exist. The consequence is that after
    deleting every record you still have every document, with nothing pointing at
    them, and a counter that reads the routing decision rather than the ledger
    will happily report fifteen records over an empty page.
    """

    def _posted(self, settings, documents, make_run, document_id: str) -> None:
        run = make_run(GOOD_INVOICE).evolve(document_id=document_id, decision="auto_post")
        documents.save(run)
        posted = RecordStore(settings.db_path).post_record(run, "V-1001")
        documents.save(run.evolve(record_id=posted["record_id"]))

    def test_it_removes_documents_and_ledger_entries_together(
        self, settings, master, documents, make_run, held
    ):
        from docuflow.deletion import wipe_documents

        self._posted(settings, documents, make_run, "DOC-POSTED")

        result = wipe_documents(settings)

        assert result.records == 1
        assert result.documents == 2, "the held one and the posted one"
        assert documents.list(limit=50) == []
        assert RecordStore(settings.db_path).count() == 0

    def test_master_data_survives_it(self, settings, master, documents, make_run, held):
        """Vendors are a copy of another system's file, not a product of this run."""
        from docuflow.deletion import wipe_documents

        before = len(master.list_vendors())
        wipe_documents(settings)
        assert len(master.list_vendors()) == before > 0

    def test_the_audit_log_survives_it(self, settings, held):
        from docuflow.deletion import wipe_documents

        wipe_documents(settings, actor="tester")
        entries = RecordStore(settings.db_path).audit_for("DOC-HELD")
        assert any(entry["action"] == "document_deleted" for entry in entries)

    def test_it_sweeps_page_images_no_document_refers_to(self, settings, held):
        """An interrupted run can leave pages behind. Start over means start over."""
        from docuflow.deletion import wipe_documents

        orphan = settings.page_dir / "DOC-GONE"
        orphan.mkdir(parents=True, exist_ok=True)
        (orphan / "page-1.jpg").write_bytes(b"jpeg")

        wipe_documents(settings)

        assert not orphan.exists()

    def test_it_leaves_pages_belonging_to_a_live_document_alone(
        self, settings, documents, make_run
    ):
        """The sweep is scoped by the database, not by "everything in the folder"."""
        from docuflow.deletion import _sweep_orphans

        documents.save(make_run(GOOD_INVOICE).evolve(document_id="DOC-LIVE"))
        pages = settings.page_dir / "DOC-LIVE"
        pages.mkdir(parents=True, exist_ok=True)
        (pages / "page-1.jpg").write_bytes(b"jpeg")

        assert _sweep_orphans(settings) == 0
        assert pages.exists()

    def test_it_clears_staged_uploads_and_orphaned_blobs(self, settings, held):
        """Nothing ever emptied the inbox, so every file ever uploaded was still there."""
        from docuflow.deletion import wipe_documents

        inbox = settings.data_dir / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        staged = inbox / "a1b2c3d4-invoice.pdf"
        staged.write_bytes(b"%PDF-1.4 staged")
        blob = settings.blob_dir / "deadbeef.pdf"
        blob.write_bytes(b"%PDF-1.4 blob")

        wipe_documents(settings)

        assert not staged.exists()
        assert not blob.exists()

    def test_a_file_a_live_document_came_from_is_left_alone(
        self, settings, documents, make_run
    ):
        from docuflow.deletion import _sweep_orphans

        inbox = settings.data_dir / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        staged = inbox / "keep-me.pdf"
        staged.write_bytes(b"%PDF-1.4 keep")
        run = make_run(GOOD_INVOICE).evolve(document_id="DOC-LIVE", blob_path=str(staged))
        documents.save(run.evolve(source=run.source.model_copy(update={"origin": str(staged)})))

        assert _sweep_orphans(settings) == 0
        assert staged.exists()

    def test_an_empty_system_is_not_an_error(self, settings):
        from docuflow.deletion import wipe_documents

        result = wipe_documents(settings)
        assert result.documents == 0
        assert "nothing to remove" in result.message

    def test_the_endpoint_refuses_without_confirmation(self, client, held):
        assert client.post("/api/documents/wipe").status_code == 400
        assert client.post("/api/documents/wipe?confirm=true").json()["documents"] == 1

    def test_the_records_badge_counts_the_ledger_not_the_documents(
        self, client, settings, master, documents, make_run
    ):
        """The bug this pair of tests exists for: 15 records over an empty page."""
        self._posted(settings, documents, make_run, "DOC-POSTED")
        assert client.get("/api/status").json()["stats"]["posted_records"] == 1

        record_id = RecordStore(settings.db_path).list(limit=1)[0]["record_id"]
        client.delete(f"/api/records/{record_id}")

        stats = client.get("/api/status").json()["stats"]
        assert stats["posted_records"] == 0
        assert stats["by_decision"]["auto_post"] == 1, "the document stays, as evidence"


class TestQueueActionEndpoints:
    def test_reevaluating_one_document(self, client, held):
        payload = client.post("/api/documents/DOC-HELD/reevaluate").json()
        assert payload["document_id"] == "DOC-HELD"
        assert payload["changed"] is False

    def test_reevaluating_the_queue(self, client, held):
        payload = client.post("/api/documents/reevaluate", json={}).json()
        assert payload["checked"] == 1

    def test_deleting_through_the_api(self, client, held):
        assert client.delete("/api/documents/DOC-HELD").status_code == 200
        assert client.get("/api/documents/DOC-HELD").status_code == 404

    def test_deleting_a_posted_document_is_a_conflict(self, client, settings, master, documents, make_run):
        run = make_run(GOOD_INVOICE).evolve(document_id="DOC-POSTED", decision="auto_post")
        documents.save(run)
        posted = RecordStore(settings.db_path).post_record(run, "V-1001")
        documents.save(run.evolve(record_id=posted["record_id"]))

        response = client.delete("/api/documents/DOC-POSTED")
        assert response.status_code == 409
        assert "cannot be deleted" in response.json()["detail"]

    def test_deleting_something_that_is_not_there(self, client):
        assert client.delete("/api/documents/DOC-NOPE").status_code == 409
