"""Scoring the transcript against what was printed.

Two measurements answering two questions, and the gap between them is the
reason both exist: a page can score a fine character error rate while mangling
exactly the digits that become a payment, because those digits are a vanishing
fraction of the characters on the page.
"""

from __future__ import annotations

import pytest

from docuflow.evaluation.ocr_accuracy import OcrScore, score_document, summarize

PAGE = (
    "TAX INVOICE\nRungruang Steel Works Co., Ltd.\nTax ID: 0105552000110\n"
    "Invoice No.: RSW-2026-1188\nInvoice Date: 14 July 2026\n"
    "Subtotal 415,000.00 THB\nVAT 7% 29,050.00 THB\nAmount Due 444,050.00 THB\n"
    "Remit to Account No: 3120045678"
)

VALUES = {
    "invoice_number": "RSW-2026-1188",
    "vendor_tax_id": "0105552000110",
    "subtotal": 415_000.00,
    "tax_amount": 29_050.00,
    "total_amount": 444_050.00,
    "bank_account": "3120045678",
}


class TestCharacterAccuracy:
    def test_a_perfect_transcript_scores_perfectly(self):
        score = score_document(PAGE, PAGE, VALUES)
        assert score.character_accuracy == 1.0
        assert score.edit_distance == 0

    def test_layout_differences_are_not_errors(self):
        """OCR breaks lines where it saw columns; the text layer where the
        renderer did. Scoring that measures a disagreement about layout."""
        reflowed = PAGE.replace("\n", "   ").replace("  ", " ")
        assert score_document(PAGE, reflowed, VALUES).character_accuracy == 1.0

    def test_table_markup_is_not_an_error_either(self):
        """Typhoon OCR returns HTML tables because a total that stays in its
        cell is a total that can still be found. Scored against a plain text
        layer, every `<td>` would be six characters of "error" for output doing
        exactly its job - and a table-heavy invoice would rank below a page of
        prose the model read badly."""
        reference = "Description Qty Amount\nSteel plate 20 370,000.00"
        marked_up = (
            "<!-- page 1 -->\n<table><tr><td>Description</td><td>Qty</td>"
            "<td>Amount</td></tr><tr><td>Steel plate</td><td>20</td>"
            "<td>370,000.00</td></tr></table>"
        )
        assert score_document(reference, marked_up, {}).character_accuracy == 1.0

    def test_stripping_a_tag_does_not_glue_two_cells_together(self):
        """`<td>A</td><td>B</td>` must not become `AB`, which is a word in
        neither document."""
        from docuflow.evaluation.ocr_accuracy import _collapse

        assert _collapse("<td>Steel</td><td>plate</td>") == "Steel plate"

    def test_a_misread_character_costs_one_edit(self):
        damaged = PAGE.replace("444,050.00", "444,O50.00")
        score = score_document(PAGE, damaged, VALUES)
        assert score.edit_distance == 1
        assert score.character_accuracy > 0.99

    def test_an_empty_transcript_scores_nothing(self):
        assert score_document(PAGE, "", VALUES).character_accuracy == 0.0

    def test_a_transcript_longer_than_the_page_cannot_score_below_zero(self):
        """A model that loops and repeats itself produces more characters than
        the page holds, and an unclamped rate goes negative."""
        score = score_document(PAGE, PAGE + PAGE * 8, VALUES)
        assert score.character_accuracy == 0.0
        assert score.character_error_rate == 1.0


class TestValueRecall:
    def test_every_value_present_is_full_recall(self):
        score = score_document(PAGE, PAGE, VALUES)
        assert score.value_recall == 1.0
        assert score.values_missing == ()

    def test_a_value_the_transcript_lost_is_named(self):
        without_total = PAGE.replace("444,050.00", "")
        score = score_document(PAGE, without_total, VALUES)

        assert "total_amount" in score.values_missing
        assert score.value_recall < 1.0

    def test_an_amount_is_found_however_it_is_punctuated(self):
        """1500.00, 1,500.00 and 1500 are the same number on a page."""
        plain = PAGE.replace("415,000.00", "415000.00").replace("444,050.00", "444050")
        assert score_document(PAGE, plain, VALUES).value_recall == 1.0

    def test_a_date_is_found_however_the_page_writes_it(self):
        """Ground truth stores 2026-07-14; no invoice in the corpus prints that.

        Searching for the stored form alone reported every date on every
        document as lost, taking value recall down by a sixth and blaming the
        OCR engine for the harness.
        """
        dates = {"invoice_date": "2026-07-14"}
        for printed in ("14 July 2026", "14/07/2026", "2026-07-14", "14 Jul 2026"):
            page = f"Invoice Date: {printed}"
            assert score_document(page, page, dates).value_recall == 1.0, printed

    def test_a_buddhist_era_year_is_the_same_date(self):
        """Thai tax invoices print 2569 where the record says 2026."""
        page = "วันที่ 25 July 2569"
        assert score_document(page, page, {"invoice_date": "2026-07-25"}).value_recall == 1.0

    def test_a_date_the_transcript_really_lost_is_still_reported(self):
        """The forgiving matching must not forgive an actual loss."""
        score = score_document("Invoice Date: 14 July 2026", "Invoice Date:",
                               {"invoice_date": "2026-07-14"})
        assert score.values_missing == ("invoice_date",)

    def test_a_short_value_is_not_scored(self):
        """A currency code or a single digit matches by accident too often to
        mean anything."""
        score = score_document(PAGE, PAGE, {"currency": "THB", "tax_rate": 7.0})
        assert score.values_expected == 0
        assert score.value_recall == 1.0, "nothing to find is not a failure"

    def test_the_two_measurements_disagree_and_that_is_the_point(self):
        """A page that reads beautifully and loses the total.

        Character accuracy barely moves - the total is ten characters out of
        two hundred - while value recall drops by a sixth. Reporting only the
        first would call this a good read.
        """
        damaged = PAGE.replace("444,050.00", "144,O5O.OO")
        score = score_document(PAGE, damaged, VALUES)

        assert score.character_accuracy > 0.95
        assert score.value_recall < 0.9
        assert "total_amount" in score.values_missing


class TestAcrossACorpus:
    def _score(self, document: str, chars: int, edits: int, expected: int, found: int) -> OcrScore:
        return OcrScore(document=document, reference_chars=chars, edit_distance=edits,
                        values_expected=expected, values_found=found)

    def test_characters_are_pooled_rather_than_averaged(self):
        """A one-line receipt and a full contract must not count equally
        towards a figure about characters."""
        report = summarize([
            self._score("receipt", chars=100, edits=50, expected=2, found=2),
            self._score("contract", chars=9_900, edits=0, expected=2, found=2),
        ])
        # Averaging the two rates gives 0.75; pooling gives 0.995.
        assert report["character_accuracy"] == pytest.approx(0.995, abs=0.001)

    def test_the_worst_documents_are_named(self):
        report = summarize([
            self._score("clean", chars=1000, edits=0, expected=1, found=1),
            self._score("awful", chars=1000, edits=400, expected=1, found=0),
        ])
        assert report["worst_documents"][0]["document"] == "awful"

    def test_an_empty_corpus_is_not_a_division_by_zero(self):
        assert summarize([]) == {"documents": 0}
