"""Finding where one document ends and the next begins.

The two mistakes are not equally expensive, and the tests are weighted the same
way the module is.

**A missed boundary is silent.** Two invoices read as one produce a single
invoice number and a single total; the second invoice is never recorded, and
nothing downstream can tell - the run looks like an ordinary document that
posted cleanly.

**A wrong split is loud.** Page two of an invoice becomes a fragment with no
header and no total, fails half the rules, and is in the review queue where
somebody sees it immediately.

So the tests that matter most are the ones proving it does not merge two
documents, and after those, the ones proving it leaves a genuine continuation
page alone.
"""

from __future__ import annotations

import pytest

from docuflow.ingest.splitting import Segment, describe, detect

INVOICE_A = """
Siam Tech Supply Co., Ltd.
TAX INVOICE
Invoice No.: STS-2026-0318
Invoice Date: 24 July 2026
Bill To: Mirtphol Industries PCL
Description Qty Unit Price Amount
UPS 3kVA rack mount 12 9,500.00 114,000.00
Subtotal 135,000.00 THB
Amount Due 144,450.00 THB
"""

INVOICE_B = """
Northwind Logistics Ltd.
TAX INVOICE
Invoice No.: NWL-2026-0518
Invoice Date: 26 July 2026
Bill To: Mirtphol Industries PCL
Description Qty Unit Price Amount
Inbound freight 7 3,500.00 24,500.00
Subtotal 35,000.00 THB
Amount Due 37,450.00 THB
"""

# The second page of INVOICE_A: no heading, no reference of its own.
CONTINUATION = """
Description Qty Unit Price Amount
Installation and commissioning 1 21,000.00 21,000.00
Cable management 4 1,250.00 5,000.00
Page 2 of 2
Payment Terms: Net 30
Remit to Account No: 1234567890
"""

THAI_RECEIPT = """
บริษัท ปตท. น้ำมันและการค้าปลีก จำกัด
ใบเสร็จรับเงิน
เลขที่ใบเสร็จ : PTT-2026-8891
น้ำมันดีเซล B7
รวมทั้งสิ้น 2,140.00 บาท
"""


def _pages(*texts: str) -> list[tuple[int, str]]:
    return [(index + 1, text) for index, text in enumerate(texts)]


class TestTheExpensiveMistake:
    """Two documents must never come back as one."""

    def test_two_invoices_are_two_documents(self) -> None:
        report = detect(_pages(INVOICE_A, INVOICE_B))
        assert report.is_multi
        assert len(report.segments) == 2
        assert report.segments[0] == Segment(1, 1, report.segments[0].reason, "invoice")
        assert report.segments[1].start_page == 2

    def test_the_boundary_names_the_reference_that_changed(self) -> None:
        """So a reviewer can see why, not just that."""
        report = detect(_pages(INVOICE_A, INVOICE_B))
        assert "NWL-2026-0518" in report.segments[1].reason

    def test_a_two_page_invoice_followed_by_another_invoice(self) -> None:
        report = detect(_pages(INVOICE_A, CONTINUATION, INVOICE_B))
        assert [(s.start_page, s.end_page) for s in report.segments] == [(1, 2), (3, 3)]

    def test_documents_of_different_types_in_one_file(self) -> None:
        report = detect(_pages(INVOICE_A, THAI_RECEIPT))
        assert len(report.segments) == 2
        assert report.segments[1].doc_type == "receipt"

    def test_three_in_a_row(self) -> None:
        report = detect(_pages(INVOICE_A, INVOICE_B, THAI_RECEIPT))
        assert [s.start_page for s in report.segments] == [1, 2, 3]


class TestTheCheapMistakeIsStillAMistake:
    """A continuation page must not become its own document."""

    def test_a_single_page_document_is_not_split(self) -> None:
        report = detect(_pages(INVOICE_A))
        assert not report.is_multi
        assert report.segments == (Segment(1, 1, "single page", "invoice"),)

    def test_a_continuation_page_stays_with_its_document(self) -> None:
        report = detect(_pages(INVOICE_A, CONTINUATION))
        assert not report.is_multi, (
            "page 2 of 2 became its own document; it has no header and no total "
            "and would arrive in the queue as an unexplained fragment"
        )
        assert report.segments[0].pages == 2

    def test_page_n_of_m_outranks_a_repeated_heading(self) -> None:
        """Letterheads repeat. An explicit page number does not lie."""
        letterhead_repeat = "Siam Tech Supply Co., Ltd.\nTAX INVOICE\nPage 2 of 2\ncontinued\n"
        report = detect(_pages(INVOICE_A, letterhead_repeat))
        assert not report.is_multi

    def test_the_same_reference_on_both_pages_is_one_document(self) -> None:
        same = "Invoice No.: STS-2026-0318\ncontinued from the previous page\nNet 30\n"
        report = detect(_pages(INVOICE_A, same))
        assert not report.is_multi


class TestWhenItCannotTell:
    """Saying so beats picking an answer."""

    def test_blank_pages_stay_with_what_came_before(self) -> None:
        report = detect(_pages(INVOICE_A, "   \n \n", INVOICE_B))
        assert [(s.start_page, s.end_page) for s in report.segments] == [(1, 2), (3, 3)]
        assert any("too little text" in note for note in report.notes)

    def test_a_file_of_blank_pages_is_reported_uncertain(self) -> None:
        report = detect(_pages(INVOICE_A, "", "  ", "\n"))
        assert report.uncertain, "an unreadable file must not come back as a confident answer"
        assert any("decide where documents begin" in note for note in report.notes)

    def test_a_continuation_quoting_a_different_reference_is_flagged(self) -> None:
        """Contradictory signals: reported rather than silently resolved."""
        odd = "Page 2 of 3\nInvoice No.: NWL-2026-0518\nsomething is wrong here\n"
        report = detect(_pages(INVOICE_A, odd))
        assert any("but quotes" in note for note in report.notes), report.notes

    def test_no_pages_at_all(self) -> None:
        report = detect([])
        assert report.segments == () and not report.is_multi


class TestReferenceReading:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Invoice No.: STS-2026-0318", True),
            ("Tax Invoice No: ABC-991", True),
            ("เลขที่ใบเสร็จ : PTT-2026-8891", True),
            ("Receipt No. R-4471", True),
            ("Page 2 of 4", False),
            ("Invoice Date: 24 July 2026", False),
            ("Total 144,450.00", False),
        ],
    )
    def test_what_counts_as_a_document_reference(self, text: str, expected: bool) -> None:
        from docuflow.ingest.splitting import _first_reference

        assert (_first_reference(text) is not None) is expected, text

    def test_a_bare_number_is_not_a_reference(self) -> None:
        """Otherwise a page number or a year splits the document."""
        from docuflow.ingest.splitting import _first_reference

        assert _first_reference("Invoice No: 2026") is None


class TestHeadingDetection:
    def test_only_the_most_specific_phrase_counts(self) -> None:
        """A Thai till prints an *abbreviated* tax invoice, which is a receipt.

        `ใบกำกับภาษีอย่างย่อ` contains the invoice keyword whole, and matching
        substrings turned every Thai receipt into an invoice once already.
        """
        from docuflow.ingest.splitting import _heading_type

        assert _heading_type("ใบกำกับภาษีอย่างย่อ\nเลขที่ 001\n") == "receipt"

    def test_a_heading_in_the_footer_is_not_a_heading(self) -> None:
        from docuflow.ingest.splitting import _heading_type

        buried = ("terms and conditions apply " * 40) + "\nTAX INVOICE"
        assert _heading_type(buried) != "invoice"


class TestOnARealStackOfPaper:
    """Synthetic strings prove the logic; a real PDF proves the plumbing.

    Three demo invoices concatenated into one file, read through the ordinary
    render and OCR path rather than hand-written text.
    """

    @pytest.fixture(scope="class")
    def stacked(self, tmp_path_factory):
        pdfium = pytest.importorskip("pypdfium2")
        root = tmp_path_factory.mktemp("stack")
        sources = [
            "Demo_Sample/01_PASS_invoice_clean.pdf",
            "Demo_Sample/09_FAIL_invoice_over_purchase_order.pdf",
            "Demo_Sample/12_FAIL_invoice_purchase_order_closed.pdf",
        ]
        from pathlib import Path

        if not all(Path(name).exists() for name in sources):
            pytest.skip("the demo corpus is not generated in this checkout")

        merged = pdfium.PdfDocument.new()
        for name in sources:
            piece = pdfium.PdfDocument(Path(name))
            merged.import_pages(piece, list(range(len(piece))))
        target = root / "stack.pdf"
        merged.save(str(target))
        return target

    def test_three_invoices_in_one_file_come_back_as_three(self, stacked, tmp_path) -> None:
        from docuflow.config import Settings
        from docuflow.ocr.engine import run_ocr
        from docuflow.ocr.render import render_document

        images = render_document(stacked, tmp_path / "pages")
        result = run_ocr(images, Settings())
        report = detect([(page.page_number, page.text) for page in result.pages])

        assert report.is_multi and not report.uncertain
        assert [(s.start_page, s.end_page) for s in report.segments] == [(1, 1), (2, 2), (3, 3)]
        assert all(s.doc_type == "invoice" for s in report.segments)

    def test_each_boundary_names_the_reference_it_saw(self, stacked, tmp_path) -> None:
        """A split a reviewer cannot check is a split they have to trust."""
        from docuflow.config import Settings
        from docuflow.ocr.engine import run_ocr
        from docuflow.ocr.render import render_document

        images = render_document(stacked, tmp_path / "pages")
        result = run_ocr(images, Settings())
        report = detect([(page.page_number, page.text) for page in result.pages])

        for segment in report.segments[1:]:
            assert "reference changed to" in segment.reason


def test_describe_is_readable() -> None:
    single = describe(detect(_pages(INVOICE_A)))
    assert single.startswith("one document")

    many = describe(detect(_pages(INVOICE_A, INVOICE_B)))
    assert many.startswith("2 documents")
    assert "pages 1-1" in many and "pages 2-2" in many
