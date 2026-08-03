"""Rule extraction, grounding and channel arbitration."""

from __future__ import annotations

import pytest

from docuflow.agents.grounding import find_evidence, value_supported_by_evidence
from docuflow.agents.llm_extractor import LlmCandidate
from docuflow.agents.merge import merge_field
from docuflow.agents.rule_extractor import extract_with_rules, find_line_items
from docuflow.schemas.registry import get_schema

INVOICE = get_schema("invoice")


def spec(name: str):
    return INVOICE.field(name)


class TestRuleExtractor:
    def test_extracts_labelled_fields(self, ocr_pages):
        found = extract_with_rules(ocr_pages, INVOICE)
        assert found["invoice_number"].raw_value == "INV-2026-0731"
        assert found["purchase_order_ref"].raw_value == "PO-2026-0412"
        assert found["total_amount"].raw_value == "200,090.00"

    def test_subtotal_and_total_are_not_confused(self, ocr_pages):
        # "Subtotal" contains "Total"; the word-boundary check must prevent the
        # subtotal line from being read as the grand total.
        found = extract_with_rules(ocr_pages, INVOICE)
        assert found["subtotal"].raw_value == "187,000.00"
        assert found["total_amount"].raw_value == "200,090.00"

    def test_tax_id_line_is_not_read_as_a_money_amount(self, ocr_pages):
        # "Tax ID: 0105543000129" must not become tax_amount.
        found = extract_with_rules(ocr_pages, INVOICE)
        assert found["tax_amount"].raw_value == "13,090.00"
        assert found["vendor_tax_id"].raw_value == "0105543000129"

    def test_value_on_the_next_line(self, ocr_pages):
        # "Bill To" sits on its own line above the buyer name.
        found = extract_with_rules(ocr_pages, INVOICE)
        assert "Mirtphol" in found["buyer_name"].raw_value

    def test_short_alias_inside_an_identifier_is_ignored(self):
        # The alias "PO" must not eat into the identifier "PO-2026-0412".
        pages = [(1, "<table><tr><td>PO</td><td>PO-2026-0412</td></tr></table>")]
        found = extract_with_rules(pages, INVOICE)
        assert found["purchase_order_ref"].raw_value == "PO-2026-0412"

    def test_html_entities_are_decoded(self):
        pages = [(1, "Vendor: Orient Paper &amp; Print Co., Ltd.")]
        found = extract_with_rules(pages, INVOICE)
        assert found["vendor_name"].raw_value == "Orient Paper & Print Co., Ltd."

    def test_thai_labels_and_numerals(self):
        pages = [(1, "เลขที่: CP-2569-0142\nจำนวนเงินรวมทั้งสิ้น ๙๘,๔๔๐.๐๐\nรวมเป็นเงิน ๙๒,๐๐๐.๐๐")]
        found = extract_with_rules(pages, INVOICE)
        assert found["invoice_number"].raw_value == "CP-2569-0142"
        assert found["total_amount"].raw_value.replace(" ", "") == "๙๘,๔๔๐.๐๐"

    def test_line_items_from_html_table(self):
        pages = [(
            1,
            "<table><tr><td>Description</td><td>Qty</td><td>Unit Price</td><td>Amount</td></tr>"
            "<tr><td>Dell laptop</td><td>5</td><td>32,000.00</td><td>160,000.00</td></tr></table>",
        )]
        items = find_line_items(pages)
        assert len(items) == 1
        assert items[0]["description"] == "Dell laptop"
        assert items[0]["amount"] == 160000.0


class TestGrounding:
    def test_quote_present_on_page(self, ocr_pages):
        assert find_evidence("Invoice No.: INV-2026-0731", ocr_pages).grounded is True

    def test_quote_absent_from_page(self, ocr_pages):
        assert find_evidence("Invoice No.: INV-9999-0000", ocr_pages).grounded is False

    def test_tolerates_spacing_and_punctuation(self, ocr_pages):
        assert find_evidence("invoice no INV 2026 0731", ocr_pages).grounded is True

    def test_empty_evidence_is_not_grounded(self, ocr_pages):
        assert find_evidence("", ocr_pages).grounded is False

    def test_value_must_appear_inside_its_own_quote(self):
        assert value_supported_by_evidence("200090.00", "Amount Due 200,090.00 THB") is True
        # The dangerous case: a real quote paired with a value that is not in it.
        assert value_supported_by_evidence("999999.00", "Amount Due 200,090.00 THB") is False


class TestMerge:
    """Both channels are normally language-model passes; the label matcher only
    fills the second slot when no second model is reachable."""

    def _merge(self, name, primary_value, secondary_value, evidence, ocr_pages, label_channel=False):
        from docuflow.agents.candidates import Candidate

        primary = (
            Candidate(field=name, raw_value=primary_value, evidence=evidence, channel="model")
            if primary_value else None
        )
        secondary = (
            Candidate(
                field=name,
                raw_value=secondary_value,
                evidence=evidence or "",
                page=1,
                channel="label" if label_channel else "verify",
            )
            if secondary_value else None
        )
        return merge_field(spec(name), primary, secondary, ocr_pages, ocr_quality=0.95, ocr_degraded=False)

    def test_agreement_scores_highest(self, ocr_pages):
        field = self._merge("total_amount", "200,090.00", "200,090.00", "Amount Due 200,090.00 THB", ocr_pages)
        assert field.value == 200090.0
        assert field.confidence >= 0.9
        assert field.grounded is True

    def test_disagreement_drops_confidence_and_is_explained(self, ocr_pages):
        field = self._merge("total_amount", "200,090.00", "187,000.00", "Amount Due 200,090.00 THB", ocr_pages)
        assert field.confidence < 0.6
        assert any("disagree" in note for note in field.notes)

    def test_single_ungrounded_reading_is_penalised(self, ocr_pages):
        field = self._merge("total_amount", "999,999.00", None, "Amount Due 999,999.00 THB", ocr_pages)
        assert field.grounded is False
        assert field.confidence < 0.4

    def test_label_channel_value_is_grounded_by_construction(self, ocr_pages):
        field = self._merge("payment_terms", None, "Net 30", "Payment Terms: Net 30", ocr_pages, label_channel=True)
        assert field.value == "Net 30"
        assert field.grounded is True

    def test_quote_that_does_not_contain_the_value_is_rejected(self, ocr_pages):
        field = self._merge("total_amount", "555,555.00", None, "Amount Due 200,090.00 THB", ocr_pages)
        assert field.grounded is False
        assert any("quoted text does not contain" in note for note in field.notes)

    def test_missing_from_both_channels_is_empty(self, ocr_pages):
        field = self._merge("bank_account", None, None, None, ocr_pages)
        assert field.is_empty
        assert field.confidence == 0.0

    def test_second_reading_can_supply_the_evidence(self, ocr_pages):
        """Agreeing values where only the second pass could quote the page."""
        from docuflow.agents.candidates import Candidate

        primary = Candidate(field="total_amount", raw_value="200,090.00",
                            evidence="a line that is not on the page", channel="model")
        secondary = Candidate(field="total_amount", raw_value="200,090.00",
                              evidence="Amount Due 200,090.00 THB", channel="verify")
        field = merge_field(spec("total_amount"), primary, secondary, ocr_pages,
                            ocr_quality=0.95, ocr_degraded=False)
        assert field.value == 200090.0
        assert field.grounded is True
        assert field.confidence >= 0.9
