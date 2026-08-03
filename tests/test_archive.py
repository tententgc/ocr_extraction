"""Filing processed documents under their type."""

from __future__ import annotations

from pathlib import Path

from docuflow.archive import archive_dir, describe_archive, file_document, rebuild, shelf_for
from docuflow.pipeline import Pipeline


class TestShelving:
    def test_known_types_get_their_own_shelf(self):
        for doc_type in ("invoice", "receipt", "purchase_order", "contract", "form"):
            assert shelf_for(doc_type) == doc_type

    def test_anything_else_files_under_other(self):
        """Otherwise a new schema or a typo creates a directory nobody expects."""
        assert shelf_for("unknown") == "other"
        assert shelf_for("bill_of_lading") == "other"


class TestFiling:
    def test_a_document_is_filed_by_type_and_month(self, settings, tmp_path):
        source = tmp_path / "acme.pdf"
        source.write_bytes(b"%PDF-1.7 content")
        filed = file_document(settings, source, "DOC-ABC123", "invoice",
                              "acme-inv-88214.pdf", "2026-08-04T10:00:00Z")
        assert filed is not None
        assert filed.path.parent == archive_dir(settings) / "invoice" / "2026-08"
        assert "DOC-ABC123" in filed.path.name
        assert "acme-inv-88214" in filed.path.name
        assert filed.path.read_bytes() == b"%PDF-1.7 content"

    def test_filing_costs_no_extra_bytes(self, settings, tmp_path):
        """Hard-linked, so ingest storage and the archive share the data."""
        source = tmp_path / "a.pdf"
        source.write_bytes(b"%PDF-1.7 x")
        filed = file_document(settings, source, "DOC-1", "invoice", "a.pdf", "2026-08-04T00:00:00Z")
        assert filed.path.stat().st_ino == source.stat().st_ino

    def test_filing_the_same_document_twice_is_harmless(self, settings, tmp_path):
        source = tmp_path / "a.pdf"
        source.write_bytes(b"%PDF-1.7 x")
        first = file_document(settings, source, "DOC-1", "invoice", "a.pdf", "2026-08-04T00:00:00Z")
        second = file_document(settings, source, "DOC-1", "invoice", "a.pdf", "2026-08-04T00:00:00Z")
        assert first.path == second.path

    def test_a_missing_blob_does_not_raise(self, settings, tmp_path):
        """Filing is a convenience; it must never be able to fail a document."""
        assert file_document(settings, tmp_path / "gone.pdf", "DOC-1", "invoice",
                             "gone.pdf", "2026-08-04T00:00:00Z") is None

    def test_a_hostile_filename_cannot_escape_the_shelf(self, settings, tmp_path):
        source = tmp_path / "a.pdf"
        source.write_bytes(b"%PDF-1.7 x")
        filed = file_document(settings, source, "DOC-1", "invoice",
                              "../../../../etc/passwd", "2026-08-04T00:00:00Z")
        assert filed is not None
        assert archive_dir(settings) in filed.path.parents


class TestArchiveView:
    def test_an_empty_archive_reports_nothing(self, settings):
        report = describe_archive(settings)
        assert report["total_documents"] == 0
        assert report["shelves"] == []

    def test_shelves_are_counted_by_type(self, settings, tmp_path):
        for index, doc_type in enumerate(("invoice", "invoice", "contract")):
            source = tmp_path / f"{index}.pdf"
            source.write_bytes(b"%PDF-1.7 " + str(index).encode())
            file_document(settings, source, f"DOC-{index}", doc_type,
                          f"{index}.pdf", "2026-08-04T00:00:00Z")

        report = describe_archive(settings)
        by_type = {shelf["doc_type"]: shelf["documents"] for shelf in report["shelves"]}
        assert by_type == {"invoice": 2, "contract": 1}
        assert report["total_documents"] == 3


class TestPipelineIntegration:
    def test_processing_files_the_document(self, settings, master, sample_pdf):
        run = Pipeline(settings).process_file(sample_pdf("inv-clean-classic")).run
        assert run.doc_type == "invoice"

        shelf = archive_dir(settings) / "invoice"
        filed = list(shelf.rglob("*.pdf")) + list(shelf.rglob("*.jpg"))
        assert any(run.document_id in item.name for item in filed)
        assert "archive" in {step.step for step in run.trace}

    def test_rebuild_files_documents_processed_earlier(self, settings, master, sample_pdf):
        Pipeline(settings).process_file(sample_pdf("inv-clean-classic"))
        # Wipe the shelves as though filing had never run.
        import shutil
        shutil.rmtree(archive_dir(settings), ignore_errors=True)
        assert describe_archive(settings)["total_documents"] == 0

        assert rebuild(settings)["filed"] == 1
        assert describe_archive(settings)["total_documents"] == 1
