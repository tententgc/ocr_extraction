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
        assert any("quoted text does not state" in note for note in field.notes)

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


class TestCurrencyFromTheSymbol:
    """A page printing "$154.06" states its currency as plainly as "USD".

    It used to be reported as an unverified guess at 69% - correct, and flagged
    to a human anyway. The symbol is real page text, so it is evidence.
    """

    def _derive(self, settings, pages):
        from docuflow.agents.merge import merge_all
        from docuflow.pipeline import Pipeline
        from docuflow.schemas.registry import get_schema

        schema = get_schema("invoice")
        merged = merge_all(schema, {}, {}, pages, 0.95, False)
        fields = Pipeline(settings)._derive_currency(
            merged, schema, pages, ocr_quality=0.95, ocr_degraded=False
        )
        return next(item for item in fields if item.name == "currency")

    def test_a_dollar_sign_grounds_usd(self, settings, master):
        currency = self._derive(settings, [(1, "INVOICE\nTOTAL $154.06\nThank you\n")])
        assert currency.value == "USD"
        assert currency.grounded is True
        assert "$154.06" in currency.evidence
        assert currency.confidence > 0.75

    def test_the_thai_word_grounds_thb(self, settings, master):
        currency = self._derive(settings, [(1, "Invoice No.: INV-1\nรวมทั้งสิ้น 1,070.00 บาท\n")])
        assert currency.value == "THB"
        assert currency.grounded is True
        assert "บาท" in currency.evidence

    def test_the_evidence_names_the_line_it_came_from(self, settings, master):
        """A reviewer has to be able to see why, not just what."""
        currency = self._derive(settings, [(1, "Header\nAmount Due €900.00\nFooter\n")])
        assert currency.evidence == "Amount Due €900.00"
        assert any("€900.00" in note for note in currency.notes)

    def test_a_page_naming_no_currency_leaves_the_field_alone(self, settings, master):
        from docuflow.agents.merge import merge_all
        from docuflow.pipeline import Pipeline
        from docuflow.schemas.registry import get_schema

        schema = get_schema("invoice")
        pages = [(1, "Invoice No.: INV-1\nTotal 1070.00\n")]
        merged = merge_all(schema, {}, {}, pages, 0.95, False)
        fields = Pipeline(settings)._derive_currency(
            merged, schema, pages, ocr_quality=0.95, ocr_degraded=False
        )
        assert next(f for f in fields if f.name == "currency").is_empty


class TestTypeAwareGrounding:
    """A date written one way still states the date read another way.

    The literal substring test used to call "INVOICE DATE 11/02/2019" evidence
    that does not contain 2019-02-11 - marking a correct reading unverified and
    dropping it to 40% confidence. That was the single most common thing the
    check reported, and it was wrong every time.
    """

    @pytest.mark.parametrize("value,evidence", [
        ("2019-02-11", "INVOICE DATE 11/02/2019"),
        ("2019-02-26", "DUE DATE 26/02/2019"),
        ("2026-07-08", "Invoice Date: 8 July 2026"),
        ("2026-07-08", "Invoice Date: 2026-07-08"),
    ])
    def test_a_date_is_supported_however_it_is_written(self, value, evidence):
        assert value_supported_by_evidence(value, evidence, "date", value) is True

    def test_a_date_the_page_does_not_state_is_still_rejected(self):
        assert value_supported_by_evidence(
            "2019-02-11", "INVOICE DATE 03/09/2021", "date", "2019-02-11") is False

    @pytest.mark.parametrize("value,evidence", [
        ("154.06", "TOTAL $154.06"),
        ("145", "Subtotal 145.00"),
        ("200090.00", "Amount Due 200,090.00 THB"),
    ])
    def test_an_amount_is_supported_however_it_is_formatted(self, value, evidence):
        assert value_supported_by_evidence(value, evidence, "money", value) is True

    def test_an_amount_the_page_does_not_state_is_still_rejected(self):
        assert value_supported_by_evidence(
            "154.06", "TOTAL $999.00", "money", "154.06") is False

    def test_the_screenshot_case_end_to_end(self, ocr_pages):
        """One channel, ISO value, day-first source: verified, not flagged."""
        from docuflow.agents.candidates import Candidate

        pages = [(1, "INVOICE # US-001\nINVOICE DATE 11/02/2019\nDUE DATE 26/02/2019\n")]
        field = merge_field(
            spec("invoice_date"),
            Candidate("invoice_date", "2019-02-11", "INVOICE DATE 11/02/2019", channel="model"),
            None, pages, ocr_quality=0.95, ocr_degraded=False,
        )
        assert field.grounded is True
        assert field.confidence > 0.75


class TestCorroboration:
    """Two independent readings agreeing is why two channels exist."""

    PAGES = [(1, "Bill To\nMirtphol Industries PCL\n88/9 Rama IV Road\nBangkok 10110\n")]

    def _candidate(self, channel, value="Mirtphol Industries PCL", evidence="Bill To"):
        from docuflow.agents.candidates import Candidate

        return Candidate("buyer_name", value, evidence, channel=channel)

    def test_agreement_grounds_a_value_no_single_quote_states(self):
        """The value is split across lines, so neither quote contains it - but
        both channels read the page and reached the same answer."""
        field = merge_field(
            spec("buyer_name"),
            self._candidate("model"),
            self._candidate("verify"),
            self.PAGES, ocr_quality=0.95, ocr_degraded=False,
        )
        assert field.grounded is True
        assert any("both readings" in note for note in field.notes)

    def test_a_quote_that_is_not_on_the_page_is_never_corroborated(self):
        """Agreement relaxes how a value ties to its quote, never whether the
        page contains the quote at all."""
        field = merge_field(
            spec("buyer_name"),
            self._candidate("model", evidence="Bill To: Stark Industries Ltd of Malibu"),
            self._candidate("verify", evidence="Bill To: Stark Industries Ltd of Malibu"),
            self.PAGES, ocr_quality=0.95, ocr_degraded=False,
        )
        assert field.grounded is False

    def test_disagreement_is_untouched_by_corroboration(self):
        field = merge_field(
            spec("buyer_name"),
            self._candidate("model", value="Mirtphol Industries PCL"),
            self._candidate("verify", value="Stark Industries Ltd"),
            self.PAGES, ocr_quality=0.95, ocr_degraded=False,
        )
        assert field.confidence < 0.6
        assert any("disagree" in note for note in field.notes)


class TestAmbiguousDates:
    """`11/02/2019` is two different days, and two models will pick differently.

    On one real page the first pass returned 2019-11-02 and the second
    2019-02-11 — reported as a disagreement at 44% confidence. Neither is a
    reading of the document; both are guesses at a convention the page states.
    """

    PAGE = [(1, "INVOICE # US-001\nINVOICE DATE 11/02/2019\nDUE DATE 26/02/2019\n")]

    def _candidate(self, value, channel):
        from docuflow.agents.candidates import Candidate

        return Candidate("invoice_date", value, "INVOICE DATE 11/02/2019", channel=channel)

    def test_the_page_settles_which_way_round_dates_are_written(self):
        from docuflow.normalize.dates import detect_day_first

        # 26/02 can only be day-first, and it settles 11/02 on the same page.
        assert detect_day_first(self.PAGE[0][1]) is True
        assert detect_day_first("INVOICE DATE 02/11/2019\nDUE DATE 02/26/2019") is False
        assert detect_day_first("DATE 01/02/2019") is None

    def test_two_models_disagreeing_land_on_the_same_day(self):
        field = merge_field(
            spec("invoice_date"),
            self._candidate("2019-11-02", "model"),     # read it month-first
            self._candidate("2019-02-11", "verify"),    # read it day-first
            self.PAGE, ocr_quality=0.95, ocr_degraded=False, day_first=True,
        )
        assert field.value == "2019-02-11"
        assert field.confidence >= 0.9
        assert not any("disagree" in note for note in field.notes)

    def test_the_convention_decides_which_day_it_is(self):
        """The same quote, read under each convention, is two different days."""
        from docuflow.agents.candidates import Candidate

        def read(day_first):
            pages = [(1, "INVOICE DATE 02/11/2019\n")]
            return merge_field(
                spec("invoice_date"),
                Candidate("invoice_date", "2019-01-01", "INVOICE DATE 02/11/2019", channel="model"),
                None, pages, ocr_quality=0.95, ocr_degraded=False, day_first=day_first,
            ).value

        assert read(day_first=False) == "2019-02-11"   # 2 November -> no: February 11
        assert read(day_first=True) == "2019-11-02"    # 2nd of November

    def test_the_convention_is_read_once_for_the_whole_document(self):
        from docuflow.agents.candidates import Candidate
        from docuflow.agents.merge import merge_all

        fields = merge_all(
            INVOICE,
            {"invoice_date": Candidate("invoice_date", "2019-11-02", "INVOICE DATE 11/02/2019", channel="model")},
            {"invoice_date": Candidate("invoice_date", "2019-02-11", "INVOICE DATE 11/02/2019", channel="verify")},
            self.PAGE, 0.95, False,
        )
        assert next(f for f in fields if f.name == "invoice_date").value == "2019-02-11"


class TestEnumGrounding:
    """An enum value is a classification, not a transcription."""

    def test_a_contract_that_automatically_renews_supports_yes(self):
        """The page will never contain the word "yes"; that is not evidence of
        a hallucination, it is what choosing from a list looks like."""
        assert value_supported_by_evidence(
            "yes", "This Agreement shall automatically renew", "enum", "yes") is True

    def test_the_quote_must_still_be_on_the_page(self):
        """Relaxing the value test never relaxes the evidence test."""
        from docuflow.agents.candidates import Candidate
        from docuflow.schemas.registry import get_schema

        pages = [(1, "This Agreement shall automatically renew for successive terms.\n")]
        contract = get_schema("contract")
        real = merge_field(
            contract.field("auto_renew"),
            Candidate("auto_renew", "yes", "shall automatically renew", channel="model"),
            None, pages, ocr_quality=0.95, ocr_degraded=False,
        )
        invented = merge_field(
            contract.field("auto_renew"),
            Candidate("auto_renew", "yes", "the parties agree to perpetual renewal", channel="model"),
            None, pages, ocr_quality=0.95, ocr_degraded=False,
        )
        assert real.grounded is True and real.confidence > 0.75
        assert invented.grounded is False


class TestPassageAgreement:
    """Two readings of a passage differ by where they stopped."""

    LONG = "the laws of the State of California, without regard to its conflict of law principles."
    SHORT = "the laws of the State of California"

    def test_a_shorter_reading_of_the_same_passage_is_not_a_disagreement(self):
        from docuflow.agents.candidates import Candidate
        from docuflow.schemas.registry import get_schema

        page = [(1, "This Agreement shall be governed by and construed in accordance with "
                    f"{self.LONG}\n")]
        field = merge_field(
            get_schema("contract").field("governing_law"),
            Candidate("governing_law", self.LONG, self.LONG, channel="model"),
            Candidate("governing_law", self.SHORT, self.SHORT, channel="verify"),
            page, ocr_quality=0.95, ocr_degraded=False,
        )
        assert field.confidence >= 0.9
        assert not any("disagree" in note for note in field.notes)

    def test_a_genuinely_different_passage_still_disagrees(self):
        from docuflow.agents.candidates import Candidate
        from docuflow.schemas.registry import get_schema

        page = [(1, f"governed by {self.LONG}\n")]
        field = merge_field(
            get_schema("contract").field("governing_law"),
            Candidate("governing_law", self.LONG, self.LONG, channel="model"),
            Candidate("governing_law", "the laws of the State of New York", self.LONG, channel="verify"),
            page, ocr_quality=0.95, ocr_degraded=False,
        )
        assert any("disagree" in note for note in field.notes)

    def test_one_extra_word_is_not_a_disagreement(self):
        """The real case: "laws of Thailand" against "the laws of Thailand"."""
        from docuflow.agents.merge import _values_agree

        assert _values_agree("laws of Thailand", "the laws of Thailand") is True
        assert _values_agree("Net 30", "Net 30 days") is True

    def test_a_different_answer_still_disagrees(self):
        from docuflow.agents.merge import _values_agree

        assert _values_agree("laws of Thailand", "laws of Cambodia") is False
        assert _values_agree(self.LONG, "the laws of the State of New York") is False

    def test_single_tokens_are_never_matched_this_way(self):
        """"INV-001" sits inside "INV-0012"; one token is not a phrase."""
        from docuflow.agents.merge import _values_agree

        assert _values_agree("no", "notice given") is False
        assert _values_agree("INV-001", "INV-0012") is False
        assert _values_agree("Bangkok", "Bangkok Airways") is False
