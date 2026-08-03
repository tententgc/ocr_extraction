"""The repair agent: recover what was missed, invent nothing.

The whole value of this stage rests on one property - it can find a value that
was on the page all along, and it cannot produce one that was not. These tests
are mostly about the second half.
"""

from __future__ import annotations

import pytest

from docuflow.agents.repair import _protected, _vet
from docuflow.agents.repair_tools import ProposalBox, build_repair_tools
from docuflow.agents.tools import run_tool
from docuflow.models import ExtractedField
from docuflow.schemas.registry import get_schema

INVOICE = get_schema("invoice")
PAGE = (
    "Siam Tech Supply Co., Ltd.\n"
    "TAX INVOICE\n"
    "Invoice No.: INV-2026-0731\n"
    "Invoice Date: 8 July 2026\n"
    "Due Date: 7 August 2026\n"
    "Subtotal 187,000.00 THB\n"
    "Amount Due 200,090.00 THB\n"
)
PAGES = [(1, PAGE)]


def vet(field: str, value: str, evidence: str, fields=()):
    return _vet(
        {"field": field, "value": value, "evidence": evidence, "reason": "found it"},
        INVOICE, tuple(fields), PAGES, ocr_quality=0.95, ocr_degraded=False,
    )


class TestItCannotInvent:
    """If it is not on the document, it is not on the document."""

    def test_a_quote_that_is_not_on_the_page_is_discarded(self):
        recovered, reason = vet("due_date", "2027-01-01", "Due Date: 1 January 2027")
        assert recovered is None
        assert "not on the page" in reason

    def test_a_real_quote_that_does_not_state_the_value_is_discarded(self):
        """The dangerous shape: a genuine line paired with an invented number."""
        recovered, reason = vet("total_amount", "999,999.00", "Amount Due 200,090.00 THB")
        assert recovered is None
        assert "does not state" in reason

    def test_a_field_the_schema_does_not_have_is_discarded(self):
        recovered, reason = vet("profit_margin", "12", "Amount Due 200,090.00 THB")
        assert recovered is None
        assert "not a field" in reason

    def test_a_value_that_cannot_be_interpreted_is_discarded(self):
        recovered, reason = vet("invoice_date", "sometime in July", "Invoice Date: 8 July 2026")
        assert recovered is None

    def test_a_proposal_with_no_evidence_is_discarded(self):
        recovered, reason = vet("due_date", "2026-08-07", "")
        assert recovered is None
        assert "quote" in reason

    def test_it_cannot_overwrite_a_settled_value(self):
        """Two channels that agreed outrank one agent looking again."""
        settled = ExtractedField(
            name="total_amount", label="Total", value=200_090.00, confidence=0.95,
            grounded=True, source="llm",
        )
        recovered, reason = vet("total_amount", "187,000.00", "Subtotal 187,000.00 THB", [settled])
        assert recovered is None
        assert "not overwritten" in reason

    def test_it_cannot_overwrite_a_human(self):
        corrected = ExtractedField(
            name="due_date", label="Due Date", value="2026-09-01", confidence=1.0,
            grounded=True, source="human",
        )
        assert _protected(corrected) is True


class TestItRecoversWhatIsThere:
    def test_a_value_on_the_page_is_accepted_and_grounded(self):
        recovered, reason = vet("due_date", "7 August 2026", "Due Date: 7 August 2026")
        assert recovered is not None, reason
        assert recovered.value == "2026-08-07"
        assert recovered.grounded is True
        assert recovered.source == "agent"

    def test_a_recovered_value_is_scored_as_one_reading_not_two(self):
        """It is one look at the page, however careful - never agreement."""
        from docuflow.confidence import AGREEMENT_BOTH_CHANNELS

        recovered, _ = vet("due_date", "7 August 2026", "Due Date: 7 August 2026")
        assert recovered.confidence < AGREEMENT_BOTH_CHANNELS

    def test_an_empty_low_confidence_field_can_be_filled(self):
        weak = ExtractedField(
            name="due_date", label="Due Date", value=None, confidence=0.0,
            grounded=False, source="llm",
        )
        recovered, reason = vet("due_date", "7 August 2026", "Due Date: 7 August 2026", [weak])
        assert recovered is not None, reason

    def test_the_note_says_where_the_value_came_from(self):
        recovered, _ = vet("due_date", "7 August 2026", "Due Date: 7 August 2026")
        assert any("repair agent" in note for note in recovered.notes)


class TestTools:
    """Read-only, and honest when a thing is absent."""

    def _tools(self, box=None):
        return build_repair_tools(PAGE, INVOICE, (), (), box or ProposalBox(proposals=[]))

    def test_finding_something_that_is_there(self):
        out = run_tool(self._tools(), "find_on_page", {"query": "Due Date"})
        assert out["found"] >= 1

    def test_finding_something_that_is_not_there_says_so(self):
        out = run_tool(self._tools(), "find_on_page", {"query": "Late Payment Penalty"})
        assert out["found"] == 0
        assert "not report a value the page does not show" in out["note"]

    def test_reading_the_line_after_a_label(self):
        out = run_tool(self._tools(), "read_near_label", {"label": "Due Date"})
        assert "7 August 2026" in out["lines"][0]["line"]

    def test_a_label_the_page_lacks_says_so(self):
        out = run_tool(self._tools(), "read_near_label", {"label": "Discount"})
        assert out["found"] == 0

    def test_missing_fields_are_listed_with_their_labels(self):
        out = run_tool(self._tools(), "show_missing_fields", {})
        names = {row["field"] for row in out["missing"]}
        assert "invoice_number" in names
        assert any(row["labelled_like"] for row in out["missing"])

    def test_a_proposal_is_recorded_not_applied(self):
        """The tool collects; only `_vet` decides."""
        box = ProposalBox(proposals=[])
        out = run_tool(self._tools(box), "propose_value",
                       {"field": "due_date", "value": "x", "evidence": "y"})
        assert out["recorded"] is True
        assert len(box.proposals) == 1

    def test_an_incomplete_proposal_is_refused(self):
        box = ProposalBox(proposals=[])
        out = run_tool(self._tools(box), "propose_value", {"field": "due_date"})
        assert "error" in out
        assert box.proposals == []
