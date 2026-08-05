"""Every stage that touches text, asked the Thai question directly.

The system is sold as reading Thai and English documents, and the Thai half has
a failure mode the English half does not: **Python answers questions about Thai
script the unhelpful way, and answers them confidently.**

    "๑๒๓๔๕๖๗๘๙๐".isdigit()      -> True
    "เลขที่บัญชี"[:2].isalpha()  -> True
    int("๑๒๓๔๕๖๗๘๙๐")           -> 1234567890

Nothing raises. A branch written for Arabic digits and Latin letters simply
takes the wrong turn, and the value that comes out the far end looks like a
value. That is why these are tests rather than type hints: every one of them
passes through code that never errors.

No model, no OCR, no API key - every assertion here is about a pure function.
"""

from __future__ import annotations

import pytest

from docuflow.agents.classifier import classify_by_keywords
from docuflow.normalize.amounts import detect_currency, parse_money, parse_percent
from docuflow.normalize.apply import normalize_field_value
from docuflow.normalize.dates import parse_date
from docuflow.normalize.identifiers import (
    normalize_account_number,
    parse_tax_id,
    reference_key,
    thai_tax_id_checksum_ok,
)
from docuflow.normalize.text import normalize_for_match, thai_digits_to_arabic
from docuflow.schemas.registry import get_schema

THAI_DIGITS = "๑๒๓๔๕๖๗๘๙๐"
ARABIC_DIGITS = "1234567890"


class TestBankAccountsInThai:
    """The bug this file was written for, and it reached the ledger.

    `"๑๒๓๔๕๖๗๘๙๐".isdigit()` is True, so `normalize_account_number` returned
    early with the Thai numerals untouched. The rule that checks the account
    compares through `reference_key`, which *does* convert - so the document
    matched its vendor, passed every check, auto-posted, and no person ever saw
    it. The export then handed the next system an account number it cannot pay.

    A wrong value that blocks is a bad afternoon. A wrong value that posts is
    the failure this whole system is built to measure.
    """

    def test_thai_numerals_become_arabic(self) -> None:
        assert normalize_account_number(THAI_DIGITS) == ARABIC_DIGITS

    def test_a_thai_label_is_not_part_of_the_account(self) -> None:
        """`"เลขที่บัญชี"[:2].isalpha()` is True, so a labelled Thai account
        took the IBAN branch and the label was stored as part of the number."""
        assert normalize_account_number("เลขที่บัญชี ๘๘๙๐๐๑๔๔๕๕") == "8890014455"
        assert normalize_account_number("เลขที่บัญชี 8890014455") == "8890014455"

    @pytest.mark.parametrize(
        "iban",
        ["GB29NWBK60161331926819", "DE89370400440532013000", "TH12 3456 7890 1234"],
    )
    def test_ibans_still_keep_their_country_code(self, iban: str) -> None:
        """The fix narrows the IBAN branch to ASCII. It must not close it."""
        result = normalize_account_number(iban)
        assert result is not None and result[:2].isalpha()

    def test_the_stored_value_is_the_converted_one(self) -> None:
        """Through the normaliser the pipeline actually calls, not the helper."""
        spec = next(f for f in get_schema("invoice").fields if f.name == "bank_account")
        value, error = normalize_field_value(THAI_DIGITS, spec)
        assert error is None
        assert value == ARABIC_DIGITS

    def test_a_thai_account_still_matches_its_register_entry(self) -> None:
        assert reference_key(THAI_DIGITS) == reference_key(ARABIC_DIGITS)


class TestThaiTaxIds:
    def test_thai_numerals_are_converted(self) -> None:
        assert parse_tax_id("๐๑๐๕๕๕๒๐๐๐๑๑๐") == "0105552000110"

    def test_separators_are_stripped(self) -> None:
        assert parse_tax_id("0-1055-52000-11-0") == "0105552000110"
        assert parse_tax_id("0 1055 52000 11 0") == "0105552000110"

    def test_the_checksum_works_on_thai_numerals(self) -> None:
        """It happens to, because `int()` reads them - but it is load-bearing
        and nothing else states it."""
        assert thai_tax_id_checksum_ok("๐๑๐๕๕๕๒๐๐๐๑๑๐") is True

    def test_a_wrong_check_digit_is_still_caught(self) -> None:
        assert thai_tax_id_checksum_ok("0105552000111") is False


class TestThaiDates:
    """Buddhist era, Thai month names, Thai numerals - and every combination
    of the three, because a real page mixes them."""

    @pytest.mark.parametrize(
        "printed,iso",
        [
            ("14 กรกฎาคม 2569", "2026-07-14"),
            ("๑๔ กรกฎาคม ๒๕๖๙", "2026-07-14"),
            ("14 ก.ค. 2569", "2026-07-14"),
            ("14/07/2569", "2026-07-14"),
            ("31 ธันวาคม 2568", "2025-12-31"),
            ("1 มกราคม 2570", "2027-01-01"),
            # The same page can print an English date beside a Thai one.
            ("14 July 2026", "2026-07-14"),
            ("14/07/2026", "2026-07-14"),
        ],
    )
    def test_it_lands_on_the_right_gregorian_day(self, printed: str, iso: str) -> None:
        assert parse_date(printed) == iso

    def test_a_buddhist_year_is_not_read_as_a_gregorian_one(self) -> None:
        """543 years is not a rounding error - it is a contract that expired
        five centuries from now, and `dates_not_in_future` blocking on it."""
        assert parse_date("14 กรกฎาคม 2569") != "2569-07-14"


class TestThaiAmounts:
    @pytest.mark.parametrize(
        "printed,value",
        [
            ("135,000.00", 135000.0),
            ("135,000.00 บาท", 135000.0),
            ("๑๓๕,๐๐๐.๐๐", 135000.0),
            ("฿135,000.00", 135000.0),
        ],
    )
    def test_money_reads_through_thai(self, printed: str, value: float) -> None:
        assert parse_money(printed) == value

    def test_a_thai_percent_is_a_number(self) -> None:
        """Stored as `"7%"` it reaches `tax_rate_consistent`, which skips
        rather than fails - a check that stopped existing without a word."""
        assert parse_percent("๗%") == 7.0

    @pytest.mark.parametrize("printed", ["รวมทั้งสิ้น 2,140.00 บาท", "฿2,140.00"])
    def test_baht_is_recognised_as_thb(self, printed: str) -> None:
        assert detect_currency(printed) == "THB"


class TestThaiClassification:
    """The nesting that turned every Thai receipt into an invoice.

    A Thai till prints `ใบกำกับภาษีอย่างย่อ` - an *abbreviated* tax invoice,
    which is a receipt - and that string contains the invoice keyword whole.
    Only the most specific matched phrase may vote.
    """

    @pytest.mark.parametrize(
        "text,doc_type",
        [
            ("ใบกำกับภาษีอย่างย่อ\nเลขที่ 001\nรวม 500", "receipt"),
            ("ใบกำกับภาษี\nเลขที่ INV-1\nรวม 500", "invoice"),
            ("ใบเสร็จรับเงิน\nเลขที่ใบเสร็จ R-1", "receipt"),
            ("ใบแจ้งหนี้\nเลขที่ INV-2", "invoice"),
            ("สัญญาจ้างบริการ\nระหว่าง ก และ ข", "contract"),
        ],
    )
    def test_the_most_specific_phrase_decides(self, text: str, doc_type: str) -> None:
        assert classify_by_keywords(text)[0] == doc_type


class TestThaiTextMatching:
    def test_thai_numerals_convert(self) -> None:
        assert thai_digits_to_arabic("๒๕๖๙") == "2569"

    def test_thai_spacing_is_collapsed(self) -> None:
        assert normalize_for_match("สยาม  เทค") == normalize_for_match("สยาม เทค")

    def test_two_different_thai_names_do_not_collapse_together(self) -> None:
        """Matching that is too eager pays the wrong supplier."""
        assert normalize_for_match("บริษัท สยามเทค จำกัด") != normalize_for_match(
            "บริษัท สยามเทรด จำกัด"
        )


class TestThaiPagesReachTheReader:
    """A page the quality gate rejects never gets read at all."""

    THAI_PAGE = (
        "บริษัท รุ่งเรือง สตีล เวิร์คส์ จำกัด\n"
        "ใบกำกับภาษี\n"
        "เลขที่ RSW-2026-1188\n"
        "วันที่ 14 กรกฎาคม 2569\n"
        "เหล็กแผ่นรีดร้อน 12 มม. 20 18,500.00 370,000.00\n"
        "รวมเป็นเงิน 370,000.00 บาท\n"
        "ภาษีมูลค่าเพิ่ม 7% 25,900.00 บาท\n"
        "จำนวนเงินทั้งสิ้น 395,900.00 บาท"
    )

    def test_a_clean_thai_page_passes_the_quality_gate(self) -> None:
        from docuflow.ocr.quality import score_page_text

        assert score_page_text(self.THAI_PAGE).acceptable is True

    def test_thai_numerals_do_not_make_a_page_look_unreadable(self) -> None:
        """They are a reading problem, not a legibility one. Failing the gate
        here would escalate down the provider chain for no reason."""
        from docuflow.ocr.quality import score_page_text

        page = self.THAI_PAGE.replace("370,000.00", "๓๗๐,๐๐๐.๐๐")
        assert score_page_text(page).acceptable is True


class TestThaiDocumentsInAStack:
    """Splitting has to read Thai references and Thai continuation markers."""

    INVOICE_A = "ใบกำกับภาษี\nเลขที่ TH-001\nรวม 1,000.00 บาท\nบริษัท ก จำกัด\n"
    INVOICE_B = "ใบกำกับภาษี\nเลขที่ TH-002\nรวม 2,000.00 บาท\nบริษัท ข จำกัด\n"

    def test_two_thai_invoices_are_two_documents(self) -> None:
        from docuflow.ingest.splitting import detect

        report = detect([(1, self.INVOICE_A), (2, self.INVOICE_B)])
        assert len(report.segments) == 2

    def test_a_thai_continuation_page_stays_with_its_document(self) -> None:
        from docuflow.ingest.splitting import detect

        report = detect([(1, self.INVOICE_A), (2, "หน้า 2 จาก 2\nรายการต่อ\nรวม 500.00 บาท\n")])
        assert report.is_multi is False

    def test_a_thai_receipt_reference_is_a_reference(self) -> None:
        from docuflow.ingest.splitting import _first_reference

        assert _first_reference("เลขที่ใบเสร็จ : PTT-2026-8891") == "PTT-2026-8891"
