"""Currency for a document that never printed one."""

from __future__ import annotations

import pytest

from docuflow.confidence import FROM_INFERENCE, FROM_RECORD
from docuflow.normalize.locale import currency_from_country, currency_from_vendor, infer_currency


class TestCountrySignals:
    @pytest.mark.parametrize("text,expected", [
        ("ใบกำกับภาษี\nบริษัท สยาม เทค จำกัด\nรวมทั้งสิ้น 1,070.00", "THB"),
        ("INVOICE\nAcme Corp, Springfield, United States\nTOTAL 154.06", "USD"),
        ("INVOICE\nHanseatic GmbH, Hamburg, Germany\nGesamt 900,00", "EUR"),
        ("INVOICE\nOrchard Pte Ltd, Singapore\nTotal 400.00", "SGD"),
    ])
    def test_the_country_decides(self, text, expected):
        guess = currency_from_country(text)
        assert guess is not None and guess.currency == expected

    def test_a_thai_tax_id_identifies_the_jurisdiction(self):
        """13 digits is a Thai taxpayer number; no other country uses that."""
        guess = currency_from_country("INVOICE\nTax ID: 0105543000129\nTotal 1070.00")
        assert guess is not None and guess.currency == "THB"

    def test_an_incidental_thai_address_does_not_decide(self):
        """A Thai delivery address on an otherwise English invoice from abroad."""
        text = ("INVOICE\nAcme Corp, Springfield, United States\n"
                "Ship to: 88/9 Rama IV Road, กรุงเทพ\nTOTAL 154.06\n") * 3
        guess = currency_from_country(text)
        assert guess is not None and guess.currency == "USD"

    def test_a_document_naming_nowhere_returns_nothing(self):
        """Better an empty field, which stops, than a currency nobody can explain."""
        assert currency_from_country("INVOICE\nTotal 500.00") is None


class TestVendorRecord:
    def test_the_register_beats_the_page(self):
        """A registered billing currency is a fact, not an inference."""
        vendor = {"legal_name": "Bluepeak Software GmbH", "currency": "EUR"}
        guess = infer_currency("INVOICE\nกรุงเทพ\nTotal 900.00", vendor)
        assert guess.currency == "EUR"
        assert guess.from_record is True

    def test_a_vendor_with_no_currency_on_file_falls_through(self):
        guess = infer_currency("ใบกำกับภาษี รวมทั้งสิ้น 1,070.00", {"legal_name": "X"})
        assert guess is not None and guess.currency == "THB"
        assert guess.from_record is False

    def test_an_unmatched_vendor_is_harmless(self):
        assert currency_from_vendor(None) is None


class TestScoring:
    """An inferred currency must never post a document on its own."""

    def test_both_strengths_sit_below_the_auto_post_bar(self, settings):
        assert FROM_RECORD < settings.auto_post_min_confidence
        assert FROM_INFERENCE < FROM_RECORD

    def test_the_schema_constrains_what_can_be_inferred(self):
        """A currency the document type does not accept is not filled in."""
        assert infer_currency("Springfield, United States", None, allowed=("THB", "EUR")) is None
