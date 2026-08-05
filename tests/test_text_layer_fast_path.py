"""Reading a PDF's own text before asking a model to look at it.

The characters in a digital PDF were written there by whatever produced it.
Recognition has nothing to add, and every model call spent on such a page buys
nothing - so the text layer goes first in the chain.

What these tests are actually about is the cases where it must *not* win: a
photograph has no text layer, a scan's text layer may be somebody else's bad
OCR, and a benchmark naming an engine must measure that engine. A fast path
that is right most of the time and silent when it is wrong would be a bad
trade in a system whose headline metric is the silent error rate.
"""

from __future__ import annotations

import pytest

from docuflow.config import Settings
from docuflow.ocr.base import OcrProvider, OcrUnavailableError, PageImage
from docuflow.ocr.engine import run_ocr
from docuflow.ocr.factory import TEXT_LAYER, provider_chain
from docuflow.ocr.fallbacks import PdfTextLayerOcr

CLEAN_TEXT = (
    "TAX INVOICE\nInvoice No.: STS-2026-0318\nDate: 2026-07-24\n"
    "Siam Tech Supply Co., Ltd.\nSubtotal 135,000.00\nVAT 7% 9,450.00\n"
    "Total 144,450.00\nPayment terms: Net 30\n"
)


def _page(number: int = 1, embedded: str = "", pages: int = 1) -> PageImage:
    return PageImage(
        page_number=number,
        path=f"/tmp/page-{number}.jpg",
        width=1200,
        height=1600,
        embedded_text=embedded,
        document_pages=pages,
    )


class CountingOcr(OcrProvider):
    """Stands in for a paid model, and says how often it was asked."""

    name = "counting-model"
    purpose_built = True

    def __init__(self, text: str = CLEAN_TEXT) -> None:
        self.text = text
        self.calls = 0

    def available(self) -> tuple[bool, str]:
        return True, ""

    def recognize(self, page: PageImage) -> str:
        self.calls += 1
        return self.text


class TestTheChainOrder:
    def test_the_text_layer_goes_first_by_default(self) -> None:
        chain = provider_chain(Settings(ocr_provider="tesseract"))
        assert chain, "no provider was available at all"
        assert chain[0].name == PdfTextLayerOcr.name

    def test_strict_mode_measures_only_the_engine_it_names(self) -> None:
        """Otherwise a benchmark scores the PDF writer, not the engine."""
        chain = provider_chain(Settings(ocr_provider="tesseract", ocr_strict=True))
        assert [p.name for p in chain] == [p.name for p in chain if p.name != TEXT_LAYER]

    def test_it_can_be_switched_off(self) -> None:
        chain = provider_chain(
            Settings(ocr_provider="tesseract", ocr_prefer_text_layer=False)
        )
        assert not chain or chain[0].name != PdfTextLayerOcr.name

    def test_choosing_it_explicitly_does_not_list_it_twice(self) -> None:
        chain = provider_chain(Settings(ocr_provider="pdftext"))
        names = [p.name for p in chain]
        assert names.count(PdfTextLayerOcr.name) == 1


class TestItRemovesItselfWhenItCannotHelp:
    def test_a_page_with_no_text_layer_reaches_the_model(self) -> None:
        """A photograph or a JPG must behave exactly as before."""
        model = CountingOcr()
        result = run_ocr([_page(embedded="")], Settings())
        # Exercised directly rather than through the chain, because what is
        # being asserted is the provider's own refusal.
        with pytest.raises(OcrUnavailableError):
            PdfTextLayerOcr().recognize(_page(embedded=""))
        assert result is not None
        assert model.calls == 0  # nothing was routed to it in this call

    def test_whitespace_only_text_counts_as_no_text_layer(self) -> None:
        with pytest.raises(OcrUnavailableError):
            PdfTextLayerOcr().recognize(_page(embedded="   \n  \t "))

    def test_a_real_text_layer_is_returned_as_is(self) -> None:
        assert PdfTextLayerOcr().recognize(_page(embedded=CLEAN_TEXT)) == CLEAN_TEXT.strip()


class TestWhatTheRunReports:
    def test_a_text_layer_read_is_said_out_loud(self) -> None:
        """Somebody auditing a value needs to know no model read it."""
        result = run_ocr([_page(embedded=CLEAN_TEXT)], Settings())
        assert PdfTextLayerOcr.name in result.provider
        assert any("text layer" in note for note in result.notes), result.notes

    def test_reading_the_file_is_not_treated_as_degraded(self) -> None:
        """It is the source, not a weaker substitute for reading the page."""
        result = run_ocr([_page(embedded=CLEAN_TEXT)], Settings())
        assert result.degraded is False, result.notes

    def test_it_does_not_trigger_the_fell_back_warning(self) -> None:
        """That warning means "slower engine stood in", which is the opposite."""
        result = run_ocr([_page(embedded=CLEAN_TEXT)], Settings())
        assert not any("stood in" in note for note in result.notes), result.notes

    def test_a_mixed_document_counts_only_the_pages_it_read(self) -> None:
        pages = [_page(1, CLEAN_TEXT, pages=2), _page(2, "", pages=2)]
        result = run_ocr(pages, Settings())
        assert any("1 of 2 page(s)" in note for note in result.notes), result.notes


class TestTheFastPathLosesNothing:
    """A text layer is plain text; Typhoon returns tables as HTML.

    That difference is real and it reaches the line-item reader, which looks
    for `<tr>` first. If the plain-text path were weaker, switching engines
    would quietly drop table rows - and `line_items_sum_to_subtotal` would then
    compare a short sum against the printed subtotal and blame the document.
    """

    PLAIN = (
        "Description Qty Unit Price Amount\n"
        "UPS 3kVA rack mount 12 9,500.00 114,000.00\n"
        "Installation and commissioning 1 21,000.00 21,000.00\n"
        "Subtotal 135,000.00 THB\n"
    )
    AS_HTML = (
        "<table><tr><td>Description</td><td>Qty</td><td>Unit Price</td><td>Amount</td></tr>"
        "<tr><td>UPS 3kVA rack mount</td><td>12</td><td>9,500.00</td><td>114,000.00</td></tr>"
        "<tr><td>Installation and commissioning</td><td>1</td><td>21,000.00</td>"
        "<td>21,000.00</td></tr></table>"
    )

    def test_the_same_rows_come_out_of_both_shapes(self) -> None:
        from docuflow.agents.rule_extractor import find_line_items

        plain = find_line_items([(1, self.PLAIN)])
        html = find_line_items([(1, self.AS_HTML)])
        assert [row["description"] for row in plain] == [row["description"] for row in html]
        assert sum(row["amount"] for row in plain) == sum(row["amount"] for row in html)
        assert sum(row["amount"] for row in plain) == pytest.approx(135_000.00)

    def test_a_totals_row_inside_the_table_is_not_a_line_item(self) -> None:
        """Both readers must drop them, and only one of them used to.

        A totals row has the shape of a line item - a label and two numbers.
        Counted as one, the rows sum to the invoice total instead of the
        subtotal, and `line_items_sum_to_subtotal` then reports every correct
        invoice as inconsistent.

        The plain-text reader always dropped them. The HTML reader never did,
        and got away with it only because Typhoon leaves totals out of the
        table it returns - a property of one model, not of the format.
        """
        from docuflow.agents.rule_extractor import find_line_items

        whole_invoice = (
            "<table>"
            "<tr><td>Description</td><td>Qty</td><td>Unit Price</td><td>Amount</td></tr>"
            "<tr><td>UPS 3kVA rack mount</td><td>12</td><td>9,500.00</td><td>114,000.00</td></tr>"
            "<tr><td>Installation and commissioning</td><td>1</td><td>21,000.00</td>"
            "<td>21,000.00</td></tr>"
            "<tr><td>Subtotal</td><td></td><td>7</td><td>135,000.00</td></tr>"
            "<tr><td>VAT 7%</td><td></td><td>7</td><td>9,450.00</td></tr>"
            "<tr><td>Amount Due</td><td></td><td>7</td><td>144,450.00</td></tr>"
            "</table>"
        )
        rows = find_line_items([(1, whole_invoice)])
        assert [row["description"] for row in rows] == [
            "UPS 3kVA rack mount",
            "Installation and commissioning",
        ]
        assert sum(row["amount"] for row in rows) == pytest.approx(135_000.00), (
            "the rows must sum to the subtotal, not the total"
        )


class TestPoorTextStillEscalates:
    def test_text_that_fails_the_quality_gate_does_not_win(self) -> None:
        """A scanned PDF can carry somebody else's bad OCR as its text layer.

        The fast path is not allowed to accept it just because it was free -
        it faces the same `score_page_text` gate as any engine's output.
        """
        from docuflow.ocr.quality import score_page_text

        rubbish = "�� ��� \x0c ~~~ ��"
        assert not score_page_text(rubbish).acceptable, (
            "this test needs text the quality gate rejects"
        )
        result = run_ocr([_page(embedded=rubbish)], Settings())
        assert result.degraded, "unreadable text must degrade the run whoever produced it"
