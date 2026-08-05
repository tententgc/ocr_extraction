"""Is the extracted record fit to hand to another system.

The section that keeps working where accuracy stops: production has no answer
sheet, so completeness, validity and self-consistency are the only quality
signals left on Monday's post.

Two of these encode invariants the project has already been bitten by.

**An absent field is not automatically a defect.** `critical` means *if this
value is here and it is wrong, somebody is paid the wrong amount*. It does not
demand the field exist - an invoice with no purchase-order reference posts, and
whether one is *required* is business policy, configurable, and decided
elsewhere. So this module counts and reports; it does not judge.

**A value in the wrong shape is a check that silently stopped existing.** A tax
rate stored as `"7%"` instead of `7.0` reaches `tax_rate_consistent`, which
skips rather than fails - the quietest possible way for a rule to stop running.
"""

from __future__ import annotations

import pytest

from docuflow.evaluation.data_quality import score_document, summarize

FULL_INVOICE = {
    "invoice_number": "RSW-2026-1188",
    "invoice_date": "2026-07-14",
    "due_date": "2026-09-12",
    "vendor_name": "Rungruang Steel Works Co., Ltd.",
    "vendor_tax_id": "0105552000110",
    "buyer_name": "Mirtphol Industries PCL",
    "purchase_order_ref": "PO-2026-0501",
    "currency": "THB",
    "subtotal": 370000.0,
    "tax_rate": 7.0,
    "tax_amount": 25900.0,
    "total_amount": 395900.0,
    "payment_terms": "Net 60",
    "bank_account": "1234567890",
}


def _score(fields: dict, **kwargs) -> dict:
    grounded = kwargs.pop("grounded", None)
    if grounded is None:
        grounded = {name: True for name in fields}
    return score_document("doc.pdf", "invoice", fields, grounded=grounded, **kwargs)


class TestCompleteness:
    def test_a_full_record_is_complete(self) -> None:
        result = _score(FULL_INVOICE)
        assert result["completeness"] == 1.0
        assert result["critical_completeness"] == 1.0

    def test_an_empty_field_does_not_count_as_filled(self) -> None:
        result = _score({**FULL_INVOICE, "purchase_order_ref": None})
        assert result["fields_filled"] == len(FULL_INVOICE) - 1
        assert result["completeness"] < 1.0

    def test_whitespace_is_not_a_value(self) -> None:
        assert _score({**FULL_INVOICE, "bank_account": "   "})["completeness"] < 1.0

    def test_an_absent_optional_field_is_reported_not_condemned(self) -> None:
        """An invoice with no purchase-order reference is a legitimate invoice.
        Whether one is *required* is business policy, decided elsewhere."""
        result = _score({**FULL_INVOICE, "purchase_order_ref": None})
        assert "purchase_order_ref" not in result["required_missing"]
        assert result["clean"] is True

    def test_critical_completeness_is_reported_apart(self) -> None:
        """A blank total is a different problem from a blank description."""
        result = _score({**FULL_INVOICE, "total_amount": None})
        assert result["critical_completeness"] < 1.0
        assert result["critical_filled"] < result["critical_defined"]


class TestValidity:
    """A value in the wrong shape is one another system has to guess about."""

    def test_a_date_that_is_not_iso_is_invalid(self) -> None:
        result = _score({**FULL_INVOICE, "invoice_date": "14/07/2026"})
        assert [issue["kind"] for issue in result["invalid"]] == ["date_not_iso"]
        assert result["clean"] is False

    def test_an_amount_that_is_a_string_is_invalid(self) -> None:
        result = _score({**FULL_INVOICE, "total_amount": "395,900.00"})
        assert result["invalid"][0]["kind"] == "not_a_number"

    def test_a_tax_rate_carrying_its_percent_sign_is_invalid(self) -> None:
        """The specific silence: `"7%"` reaches `tax_rate_consistent`, which
        skips instead of failing, and the check stops existing without a word."""
        result = _score({**FULL_INVOICE, "tax_rate": "7%"})
        assert result["invalid"][0]["kind"] == "not_a_number"

    def test_a_negative_amount_is_invalid(self) -> None:
        result = _score({**FULL_INVOICE, "total_amount": -100.0})
        assert result["invalid"][0]["kind"] == "negative_amount"

    def test_a_percentage_outside_its_range_is_invalid(self) -> None:
        assert _score({**FULL_INVOICE, "tax_rate": 700.0})["invalid"][0][
            "kind"
        ] == "percent_out_of_range"

    def test_a_currency_that_is_not_a_code_is_invalid(self) -> None:
        result = _score({**FULL_INVOICE, "currency": "Thai Baht"})
        assert result["invalid"][0]["kind"] == "currency_not_a_code"

    def test_a_correct_record_has_nothing_invalid(self) -> None:
        assert _score(FULL_INVOICE)["invalid"] == []

    def test_zero_is_a_valid_amount(self) -> None:
        """Not everything falsy is missing. A zero-rated line is a real value."""
        assert _score({**FULL_INVOICE, "tax_amount": 0.0})["invalid"] == []


class TestConsistency:
    def test_an_ungrounded_value_is_counted(self) -> None:
        """Grounding does not prove a value is right - an OCR misread is
        grounded in the text it misread - but it separates 'read off the page'
        from 'produced from nowhere'."""
        grounded = {name: True for name in FULL_INVOICE}
        grounded["bank_account"] = False
        result = _score(FULL_INVOICE, grounded=grounded)
        assert result["ungrounded"] == ["bank_account"]
        assert result["grounded_share"] < 1.0

    def test_broken_arithmetic_makes_the_record_unusable(self) -> None:
        result = _score(FULL_INVOICE, findings=["invoice_totals_consistent"])
        assert result["broken_arithmetic"] == ["invoice_totals_consistent"]
        assert result["clean"] is False

    def test_an_unrelated_finding_is_not_an_arithmetic_problem(self) -> None:
        """A blocked vendor is a business problem; the numbers still add up."""
        result = _score(FULL_INVOICE, findings=["vendor_active"])
        assert result["broken_arithmetic"] == []
        assert result["clean"] is True


class TestTheCorpusView:
    def test_completeness_is_pooled_over_fields_not_averaged_over_documents(self) -> None:
        """A four-field receipt and a fourteen-field contract must not count
        equally towards a statement about fields."""
        big = {"document": "a", "doc_type": "invoice", "fields_defined": 14,
               "fields_filled": 14, "completeness": 1.0, "critical_defined": 7,
               "critical_filled": 7, "critical_completeness": 1.0,
               "required_missing": [], "invalid": [], "ungrounded": [],
               "grounded_share": 1.0, "broken_arithmetic": [], "clean": True}
        small = {**big, "document": "b", "fields_defined": 2, "fields_filled": 0,
                 "completeness": 0.0, "clean": False}

        result = summarize([big, small])
        # Pooled: 14/16, not the mean of 100% and 0%.
        assert result["completeness"] == round(14 / 16, 3)
        assert result["clean_share"] == 0.5

    def test_nothing_evaluated(self) -> None:
        assert summarize([]) == {"documents": 0}

    def test_invalid_values_are_grouped_by_kind(self) -> None:
        scores = [
            _score({**FULL_INVOICE, "invoice_date": "14/07/2026"}),
            _score({**FULL_INVOICE, "due_date": "12/09/2026"}),
        ]
        result = summarize(scores)
        assert result["invalid_by_kind"] == {"date_not_iso": 2}
        assert result["validity"] < 1.0
