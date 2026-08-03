"""End-to-end pipeline test with no model calls.

Uses the PDF text layer as the OCR provider and the deterministic rule extractor
as the only extraction channel, so the whole orchestrator runs in CI in under a
second and without an API key.
"""

from __future__ import annotations

import pytest

from docuflow.ocr.quality import score_document_text, score_page_text
from docuflow.ocr.render import UnsupportedDocumentError, is_supported, render_document
from docuflow.pipeline import Pipeline
from docuflow.samples.catalog import sample_by_id
from docuflow.samples.generator import render_pdf
from docuflow.store.records import RecordStore


@pytest.fixture
def sample_pdf(settings, tmp_path):
    """Render one clean invoice from the demo corpus."""
    def _render(doc_id: str):
        doc = sample_by_id(doc_id)
        assert doc is not None, f"unknown sample {doc_id}"
        return render_pdf(doc, tmp_path / f"{doc_id}.pdf")
    return _render


@pytest.fixture
def pipeline(settings, master):
    """Pipeline at production thresholds (conservative)."""
    return Pipeline(settings)


@pytest.fixture
def permissive_pipeline(settings, master):
    """Pipeline with a lowered auto-post bar.

    Without a language model only the rule channel runs, so unlabelled fields
    such as the vendor letterhead score too low to clear the 90% production
    threshold. Lowering it here exercises the straight-through posting path
    itself, which is the behaviour under test.
    """
    return Pipeline(settings.model_copy(update={"auto_post_min_confidence": 0.55}))


class TestRendering:
    def test_pdf_renders_to_page_images_with_text(self, sample_pdf, settings, tmp_path):
        pages = render_document(sample_pdf("inv-clean-classic"), tmp_path / "pages")
        assert len(pages) == 1
        assert pages[0].path.exists()
        # A digital PDF carries a text layer, which the pipeline uses for free.
        assert "INV-2026-0731" in pages[0].embedded_text

    def test_multi_page_contract_renders_every_page(self, sample_pdf, tmp_path):
        pages = render_document(sample_pdf("contract-facility"), tmp_path / "pages")
        assert len(pages) >= 1
        assert all(page.path.exists() for page in pages)

    def test_page_limit_is_honoured(self, sample_pdf, tmp_path):
        pages = render_document(sample_pdf("contract-facility"), tmp_path / "pages", max_pages=1)
        assert len(pages) == 1

    def test_unsupported_file_type_is_reported_clearly(self, tmp_path):
        path = tmp_path / "notes.docx"
        path.write_bytes(b"not a document")
        with pytest.raises(UnsupportedDocumentError) as error:
            render_document(path, tmp_path / "pages")
        assert "docx" in str(error.value)

    @pytest.mark.parametrize("name", ["a.pdf", "b.JPG", "c.png", "d.tiff"])
    def test_supported_suffixes(self, tmp_path, name):
        assert is_supported(tmp_path / name) is True

    def test_unsupported_suffix(self, tmp_path):
        assert is_supported(tmp_path / "a.docx") is False


class TestQualityScoring:
    def test_empty_text_is_unacceptable(self):
        report = score_page_text("")
        assert report.acceptable is False
        assert report.score < 0.1

    def test_short_text_is_unacceptable(self):
        assert score_page_text("Invoice").acceptable is False

    def test_full_page_of_text_is_acceptable(self):
        text = (
            "Siam Tech Supply Co., Ltd., 199/22 Ratchadaphisek Road, Din Daeng, Bangkok 10400. "
            "Tax invoice INV-2026-0731 issued 8 July 2026, payment due 7 August 2026 against "
            "purchase order PO-2026-0412. Five Dell Latitude 5450 laptops at 32,000.00 each "
            "and five USB-C docking stations at 5,400.00 each. Subtotal 187,000.00 baht, "
            "value added tax at seven percent 13,090.00, amount payable 200,090.00 baht. "
            "Remit to account 1234567890 under terms Net 30."
        )
        report = score_page_text(text)
        assert report.acceptable is True
        assert report.score > 0.6

    def test_repetition_loop_is_detected(self):
        # The classic vision-model failure: the same phrase forever.
        report = score_page_text("the total amount due is " * 60)
        assert report.acceptable is False
        assert any("repetitive" in reason for reason in report.reasons)

    def test_symbol_soup_is_detected(self):
        report = score_page_text("|#|@#|$%^&*()_+|#|@#|$%^&*()_+" * 20)
        assert report.acceptable is False

    def test_document_score_is_dragged_down_by_the_worst_page(self):
        good = "Invoice number INV-2026-0731 dated 8 July 2026 for services rendered. " * 12
        report = score_document_text([good, ""])
        assert report.acceptable is False


class TestPipelineEndToEnd:
    def test_clean_invoice_is_extracted_and_posted(self, permissive_pipeline, sample_pdf, settings):
        result = permissive_pipeline.process_file(sample_pdf("inv-clean-classic"))
        run = result.run

        assert run.doc_type == "invoice"
        assert run.value("invoice_number") == "INV-2026-0731"
        assert run.value("total_amount") == 200_090.00
        assert run.value("subtotal") == 187_000.00
        assert run.value("invoice_date") == "2026-07-08"
        assert result.vendor_match.matched is True
        assert result.vendor_match.vendor["vendor_id"] == "V-1001"

        assert run.decision == "auto_post"
        assert run.record_id is not None
        record = RecordStore(settings.db_path).get(run.record_id)
        assert record["amount"] == 200_090.00
        assert record["reference"] == "INV-2026-0731"
        assert record["posted_by"] == "system"

    def test_every_stage_appears_in_the_trace(self, pipeline, sample_pdf):
        run = pipeline.process_file(sample_pdf("inv-clean-classic")).run
        steps = [step.step for step in run.trace]
        for expected in ("render", "ocr", "classify", "extract", "validate", "route"):
            assert expected in steps

    def test_arithmetic_error_is_blocked_from_posting(self, pipeline, sample_pdf):
        run = pipeline.process_file(sample_pdf("inv-bad-math")).run
        assert run.decision == "needs_review"
        assert run.record_id is None
        assert "invoice_totals_consistent" in {f.rule_id for f in run.blocking_findings}

    def test_unknown_vendor_is_blocked(self, pipeline, sample_pdf):
        run = pipeline.process_file(sample_pdf("inv-unknown-vendor")).run
        assert run.decision == "needs_review"
        assert "vendor_known" in {f.rule_id for f in run.blocking_findings}

    def test_second_copy_of_an_invoice_is_caught_as_a_duplicate(self, permissive_pipeline, sample_pdf):
        first = permissive_pipeline.process_file(sample_pdf("inv-clean-classic")).run
        assert first.decision == "auto_post"

        second = permissive_pipeline.process_file(sample_pdf("inv-duplicate")).run
        assert second.decision == "needs_review"
        assert "duplicate_document" in {f.rule_id for f in second.blocking_findings}

    def test_purchase_order_without_a_labelled_vendor_is_held_back(self, pipeline, sample_pdf):
        # A purchase order's letterhead is conventionally the buyer, so the
        # letterhead heuristic deliberately does not apply here. Without a
        # language model the vendor is therefore unresolved - and the document
        # is held rather than posted against an unknown party.
        run = pipeline.process_file(sample_pdf("po-orient-paper")).run
        assert run.doc_type == "purchase_order"
        assert run.decision == "needs_review"
        assert "required_fields_present" in {f.rule_id for f in run.blocking_findings}
        assert run.record_id is None

    def test_contract_fields_are_read_from_prose(self, pipeline, sample_pdf):
        run = pipeline.process_file(sample_pdf("contract-facility")).run
        assert run.doc_type == "contract"
        assert run.value("contract_number") == "CT-2025-018"

    def test_form_with_a_bad_tax_id_is_blocked(self, pipeline, sample_pdf):
        run = pipeline.process_file(sample_pdf("form-vendor-onboarding")).run
        assert run.doc_type == "form"
        assert "tax_id_checksum" in {f.rule_id for f in run.blocking_findings}

    def test_corrupt_file_degrades_instead_of_raising(self, pipeline, tmp_path):
        path = tmp_path / "broken.pdf"
        path.write_bytes(b"%PDF-1.7\nthis is not really a pdf")
        run = pipeline.process_file(path).run
        assert run.decision == "rejected"
        assert run.error is not None
        assert run.record_id is None

    def test_line_items_are_read_from_the_invoice_table(self, pipeline, sample_pdf):
        run = pipeline.process_file(sample_pdf("inv-clean-classic")).run
        assert len(run.line_items) == 2
        assert run.line_items[0].amount == 160_000.00

    def test_every_extracted_field_carries_evidence(self, pipeline, sample_pdf):
        run = pipeline.process_file(sample_pdf("inv-clean-classic")).run
        populated = [field for field in run.fields if not field.is_empty]
        assert populated
        for field in populated:
            if field.source in {"llm", "rule"}:
                assert field.evidence, f"{field.name} was extracted without evidence"


class TestFreeModeEndToEnd:
    """RULES_FREE_MODE=true: extract and post, check nothing."""

    @pytest.fixture
    def free_pipeline(self, settings, master):
        return Pipeline(
            settings.model_copy(
                update={"rules_free_mode": True, "auto_post_min_confidence": 0.55}
            )
        )

    def test_a_document_that_normally_blocks_now_posts(self, free_pipeline, sample_pdf):
        """The duplicate is the clean case: it reads perfectly, so the rule was
        the only thing standing between it and the ledger. Compare with
        `test_second_copy_of_an_invoice_is_caught_as_a_duplicate` above."""
        assert free_pipeline.process_file(sample_pdf("inv-clean-classic")).run.decision == "auto_post"

        second = free_pipeline.process_file(sample_pdf("inv-duplicate")).run
        assert second.blocking_findings == ()
        assert second.decision == "auto_post"
        assert second.record_id is not None

    def test_the_record_says_it_was_not_validated(self, free_pipeline, sample_pdf):
        """Silence here would be the dangerous outcome, not the blocked document."""
        run = free_pipeline.process_file(sample_pdf("inv-clean-classic")).run
        assert [finding.rule_id for finding in run.findings] == ["business_rules_disabled"]
        assert "disabled" in run.decision_reasons[0]

    def test_a_badly_read_document_is_still_held_back(self, free_pipeline, sample_pdf):
        """Free mode drops the business checks, not the reading checks. Routing
        still requires the required fields to be present and grounded, so an
        unreadable document does not post just because nothing was validated."""
        run = free_pipeline.process_file(sample_pdf("inv-unknown-vendor")).run
        assert run.decision == "needs_review"
        assert run.record_id is None

    def test_arithmetic_errors_ride_through(self, free_pipeline, sample_pdf):
        """The cost of the mode, stated as a test so nobody discovers it live."""
        run = free_pipeline.process_file(sample_pdf("inv-bad-math")).run
        assert run.blocking_findings == ()

    def test_disabling_one_rule_leaves_the_others_working(self, settings, master, sample_pdf):
        selective = Pipeline(
            settings.model_copy(
                update={"disabled_rules": "vendor_known", "auto_post_min_confidence": 0.55}
            )
        )
        run = selective.process_file(sample_pdf("inv-bad-math")).run
        assert "vendor_known" not in {finding.rule_id for finding in run.findings}
        assert "invoice_totals_consistent" in {f.rule_id for f in run.blocking_findings}
