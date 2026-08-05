"""Reading the answer sheet before grading against it.

On this project the corpus has been wrong more often than the pipeline: a VAT
that was correct when it was meant to be wrong, two contracts with no body text,
four documents claiming fields their layout never drew, two real defects
labelled clean. Each looked like a system fault first, and the expensive part
was the time spent tuning a stage that was fine.

The check that catches most of it is cheap and needs no model: **does the page
actually print this value?** `reference_text` is the PDF's own text layer,
captured before the page was photographed, so it is not an approximation of what
was drawn - it is the string the renderer was handed.

Nothing is silently corrected. A harness that quietly drops the ground truth it
cannot verify reports a better number every time the corpus gets worse.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docuflow.evaluation import integrity
from docuflow.evaluation.corpus import CorpusDocument, load_manifest

CORPUS = Path("real_samples")

PAGE = """
Rungruang Steel Works Co., Ltd.
Bangkok, Thailand
Tax ID: 0105552000110
TAX INVOICE
Invoice No.: RSW-2026-1188
Invoice Date: 14 July 2026
Bill To: Mirtphol Industries PCL
Purchase Order: PO-2026-0501
Hot-rolled steel plate 12mm 20 18,500.00 370,000.00
Subtotal 370,000.00 THB
VAT 7% 25,900.00 THB
Amount Due 395,900.00 THB
Remit to Account No: 1234567890
"""


def _document(ground_truth: dict, reference: str = PAGE) -> CorpusDocument:
    return CorpusDocument(
        order=1,
        file="1_sample_True.pdf",
        doc_type="invoice",
        posts=True,
        ground_truth=ground_truth,
        expected_findings=(),
        reference_text=reference,
        arrival="pdf",
        language="en",
    )


class TestAValueThePageNeverPrints:
    """The bug class: ground truth claiming a field its layout never draws.

    `draw_minimal_invoice` deliberately prints no tax ID and no payment terms -
    "almost nothing is labelled" is the whole point of it. Ground truth naming
    those fields makes them report as missing on every engine anyone tries, and
    the previous time this happened it produced a 50% false-alarm rate that was
    not real.
    """

    def test_a_value_that_is_printed_is_not_flagged(self) -> None:
        assert integrity.undrawn_values(_document({"invoice_number": "RSW-2026-1188"})) == []

    def test_a_value_that_is_not_printed_is_flagged(self) -> None:
        undrawn = integrity.undrawn_values(_document({"payment_terms": "Net 60"}))
        assert undrawn == [("payment_terms", "Net 60")]

    def test_a_date_counts_as_printed_in_any_of_its_written_forms(self) -> None:
        """Ground truth stores `2026-07-14`; no invoice prints that. Searching
        for the stored form alone reported every date on every page as absent."""
        assert integrity.undrawn_values(_document({"invoice_date": "2026-07-14"})) == []

    def test_an_amount_counts_as_printed_with_its_separators(self) -> None:
        assert integrity.undrawn_values(_document({"total_amount": 395900.0})) == []

    def test_a_short_value_is_not_judged(self) -> None:
        """`THB` matches by accident on almost any page, so a currency code
        proves nothing either way and is left out rather than counted as found."""
        assert integrity.undrawn_values(_document({"currency": "THB"})) == []

    def test_an_empty_ground_truth_value_is_not_flagged(self) -> None:
        assert integrity.undrawn_values(_document({"purchase_order_ref": None})) == []


class TestWhenNothingCanBeProved:
    def test_a_page_with_almost_no_text_layer_is_not_judged(self) -> None:
        """A thin reference would report every value as undrawn, which measures
        the capture rather than the corpus."""
        thin = _document({"payment_terms": "Net 60"}, reference="INVOICE 1188")
        assert integrity.undrawn_values(thin) == []

    def test_but_it_is_reported_as_thin(self) -> None:
        report = integrity.check([_document({"total_amount": 395900.0}, reference="INVOICE")])
        assert report["thin_reference"] == ["1_sample_True.pdf"]
        assert report["clean"] is False


class TestTheCorpusLevelReport:
    def test_a_clean_corpus_says_so(self) -> None:
        report = integrity.check([_document({"invoice_number": "RSW-2026-1188"})])
        assert report["clean"] is True
        assert report["undrawn_values"] == 0

    def test_undrawn_values_are_counted_by_field(self) -> None:
        documents = [
            _document({"payment_terms": "Net 60", "vendor_tax_id": "9999999999999"}),
            _document({"payment_terms": "Net 30"}),
        ]
        report = integrity.check(documents)
        assert report["undrawn_by_field"] == {"payment_terms": 2, "vendor_tax_id": 1}
        assert report["undrawn_share"] > 0

    def test_a_document_with_no_ground_truth_cannot_be_graded(self) -> None:
        report = integrity.check([_document({})])
        assert report["documents_without_ground_truth"] == ["1_sample_True.pdf"]

    def test_nothing_is_silently_dropped(self) -> None:
        """The listing is the whole contract: the harness reports, a person
        decides whether the ground truth is wrong or the layout is."""
        report = integrity.check([_document({"payment_terms": "Net 60"})])
        assert report["undrawn"] == [
            {"document": "1_sample_True.pdf", "field": "payment_terms", "value": "Net 60"}
        ]

    def test_an_empty_corpus(self) -> None:
        assert integrity.check([]) == {"documents": 0}


@pytest.mark.skipif(not CORPUS.exists(), reason="the corpus is not generated here")
class TestThisCorpus:
    """What the shipped corpus is actually like. These assert the *shape* of the
    finding rather than an exact count, so regenerating the corpus with fewer
    problems does not fail the suite - only a corpus that got worse does."""

    def test_the_check_runs_over_the_real_manifest(self) -> None:
        report = integrity.check(load_manifest(CORPUS))
        assert report["documents"] == 30
        assert report["gradeable_values"] > 200

    def test_no_document_has_lost_its_ground_truth(self) -> None:
        report = integrity.check(load_manifest(CORPUS))
        assert report["documents_without_ground_truth"] == []

    def test_the_undrawn_values_do_not_grow(self) -> None:
        """Eleven values are claimed by ground truth and not printed on their
        page - three `purchase_order_ref` on layouts with no purchase-order
        line, two `vendor_tax_id` and two `payment_terms` on the `minimal`
        layout that labels nothing, two `payment_method` on receipts that do not
        print one, and two `due_date` that are derivable from the payment terms
        but never written down.

        The bound is what stops it silently getting worse.
        """
        report = integrity.check(load_manifest(CORPUS))
        assert report["undrawn_values"] <= 11, (
            "ground truth now claims more values than the pages print: "
            f"{report['undrawn_by_field']}"
        )
