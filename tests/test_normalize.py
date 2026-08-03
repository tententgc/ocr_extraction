"""Normalisation is the accuracy foundation - it gets the tightest tests."""

from __future__ import annotations

import pytest

from docuflow.normalize import (
    normalize_field_value,
    parse_date,
    parse_money,
    parse_percent,
    parse_tax_id,
)
from docuflow.normalize.identifiers import reference_key, thai_tax_id_checksum_ok
from docuflow.normalize.text import contains_normalized, normalize_for_match
from docuflow.schemas.registry import FieldSpec


class TestParseMoney:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1,234.56", 1234.56),
            ("฿ 1,234.56", 1234.56),
            ("THB 37,450.00", 37450.00),
            ("1.234,56", 1234.56),  # European convention
            ("1 234,56", 1234.56),
            ("(500.00)", -500.00),  # accounting negative
            ("๑,๒๓๔.๕๖", 1234.56),  # Thai numerals
            ("Total: 99", 99.0),
            ("12,345", 12345.0),
            ("1.234.567", 1234567.0),
            ("USD 1,500.00 per unit", 1500.00),
        ],
    )
    def test_parses_real_world_formats(self, raw, expected):
        assert parse_money(raw) == expected

    @pytest.mark.parametrize("raw", ["", "N/A", "abc", None])
    def test_rejects_non_numeric(self, raw):
        assert parse_money(raw) is None


class TestParseDate:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("2024-03-12", "2024-03-12"),
            ("12/03/2024", "2024-03-12"),  # day-first default
            ("12 March 2024", "2024-03-12"),
            ("March 12, 2024", "2024-03-12"),
            ("31/12/2024", "2024-12-31"),
            ("2567-03-12", "2024-03-12"),  # Buddhist era ISO
            ("12/03/2567", "2024-03-12"),  # Buddhist era numeric
            ("12 มีนาคม 2567", "2024-03-12"),  # Thai month, BE year
            ("๑๒ มีนาคม ๒๕๖๗", "2024-03-12"),  # Thai numerals
            ("1 ม.ค. 2568", "2025-01-01"),  # abbreviated Thai month
        ],
    )
    def test_parses_thai_and_western_dates(self, raw, expected):
        assert parse_date(raw) == expected

    def test_unambiguous_day_overrides_convention(self):
        # 25 cannot be a month, so this is 25 December regardless of day_first.
        assert parse_date("25/12/2024", day_first=False) == "2024-12-25"

    @pytest.mark.parametrize("raw", ["", "sometime soon", "N/A", None])
    def test_returns_none_rather_than_guessing(self, raw):
        assert parse_date(raw) is None

    @pytest.mark.parametrize(
        "raw",
        [
            "Net 45",                        # became 2045 with today's day/month
            "Net 30",
            "Terms Net 45",
            "Quarterly in arrears, Net 15",
            "Due on receipt",
            "30 days",
        ],
    )
    def test_payment_terms_are_never_read_as_dates(self, raw):
        """dateutil fills in whatever the text omits; a partial parse is a guess.

        `Net 45` previously produced 2045-08-03 - the year invented from "45",
        the day and month silently taken from today.
        """
        assert parse_date(raw) is None

    def test_a_bare_year_is_not_a_date(self):
        assert parse_date("2026") is None


class TestTaxId:
    def test_strips_formatting(self):
        assert parse_tax_id("0-1055-43000-11-2") == "0105543000112"

    def test_valid_thai_checksum(self):
        assert thai_tax_id_checksum_ok("0105543000129") is True

    def test_invalid_thai_checksum(self):
        # Same ID with the check digit off by one - the exact class of OCR error
        # that must not reach the downstream system.
        assert thai_tax_id_checksum_ok("0105543000128") is False

    def test_keeps_foreign_vat_numbers(self):
        assert parse_tax_id("GB123456789") == "GB123456789"


class TestPercent:
    @pytest.mark.parametrize(("raw", "expected"), [("7%", 7.0), ("7", 7.0), ("0.07", 7.0), ("VAT 7 %", 7.0)])
    def test_percent_forms(self, raw, expected):
        assert parse_percent(raw) == expected


class TestEvidenceMatching:
    def test_ignores_punctuation_and_case(self):
        assert normalize_for_match("INV-2024/0142") == normalize_for_match("inv 2024 0142")

    def test_finds_quote_in_noisy_text(self):
        page = "Tax Invoice\nInvoice No.:  INV-2024/0142\nTotal  37,450.00"
        assert contains_normalized(page, "INV-2024/0142") is True
        assert contains_normalized(page, "INV-2024/9999") is False

    def test_reference_key_collapses_separators(self):
        assert reference_key("INV-2024/0142") == reference_key("inv 2024 0142")


class TestNormalizeFieldValue:
    def test_money_field(self):
        spec = FieldSpec(name="total_amount", label="Total", type="money")
        assert normalize_field_value("฿1,234.50", spec) == (1234.50, None)

    def test_unparseable_money_explains_itself(self):
        spec = FieldSpec(name="total_amount", label="Total", type="money")
        value, note = normalize_field_value("see attached", spec)
        assert value is None
        assert "could not parse" in note

    def test_currency_enum_accepts_symbol(self):
        spec = FieldSpec(name="currency", label="Currency", type="enum", choices=("THB", "USD"))
        assert normalize_field_value("฿", spec) == ("THB", None)

    def test_enum_rejects_unknown_choice(self):
        spec = FieldSpec(name="currency", label="Currency", type="enum", choices=("THB", "USD"))
        value, note = normalize_field_value("Bitcoin", spec)
        assert value is None
        assert "not one of" in note

    @pytest.mark.parametrize("placeholder", ["N/A", "none", "-", "ไม่ระบุ"])
    def test_placeholder_values_become_none(self, placeholder):
        spec = FieldSpec(name="vendor_name", label="Vendor")
        assert normalize_field_value(placeholder, spec) == (None, None)

    def test_reference_is_uppercased_and_trimmed(self):
        spec = FieldSpec(name="invoice_number", label="Invoice No.")
        assert normalize_field_value("  inv-2024/0142  ", spec) == ("INV-2024/0142", None)
