"""What a tool hands back has to be self-explanatory.

A tool is read by a model, and increasingly by a model following a rule that
another model wrote. Neither has seen the schema, the ERP that exported it, or
the conversation where somebody said "of course the order is net of tax". If a
number's meaning is not in the answer, it is not available to the caller at all
- and the caller will assume something reasonable and be wrong.

This file is small on purpose. It does not test what the model concludes; it
tests that the facts it would need were in front of it.
"""

from __future__ import annotations

import pytest

from docuflow.agents.tools import build_toolbox
from docuflow.store.records import RecordStore

PO = "PO-2026-0412"


@pytest.fixture
def toolbox(settings, master):
    return build_toolbox(
        ocr_text=f"Invoice against {PO}",
        master=master,
        records=RecordStore(settings.db_path),
        doc_type="invoice",
        document_id="DOC-TOOLS",
    )


def _tool(toolbox, name):
    return next(t for t in toolbox if t.name == name)


class TestPurchaseOrderLookup:
    """The bug this exists for.

    A purchase order records what was agreed, before tax. An invoice total
    includes it. An agent handed a bare `amount` compared the invoice *total*
    against it and failed a perfectly clean 135,000 order for a 144,450 total -
    every correct invoice looks roughly the VAT rate over.

    The fix is in the tool rather than in each rule's wording, because the tool
    owns the number and a rule can be written by anyone, including a model that
    was never told.
    """

    def test_the_answer_says_the_amount_excludes_tax(self, toolbox) -> None:
        answer = _tool(toolbox, "lookup_purchase_order").run({"po_number": PO})
        assert answer["found"] is True
        assert answer["amount_basis"] == "net of tax"

    def test_the_answer_says_which_invoice_figure_to_compare(self, toolbox) -> None:
        answer = _tool(toolbox, "lookup_purchase_order").run({"po_number": PO})
        note = answer["comparison_note"].lower()
        assert "subtotal" in note, "the note must name the figure to use"
        assert "total" in note, "and the one not to"
        assert "vat" in note or "tax" in note

    def test_a_missing_order_still_answers_without_claiming_a_basis(self, toolbox) -> None:
        """No amount was returned, so there is nothing to explain."""
        answer = _tool(toolbox, "lookup_purchase_order").run({"po_number": "PO-NOPE"})
        assert answer["found"] is False
        assert "amount_basis" not in answer

    def test_the_amount_itself_is_unchanged(self, settings, master, toolbox) -> None:
        """The note explains the number; it must not quietly alter it."""
        answer = _tool(toolbox, "lookup_purchase_order").run({"po_number": PO})
        assert answer["amount"] == pytest.approx(float(master.get_purchase_order(PO)["amount"]))


class TestThePurchaseOrderComparisonIsComputed:
    """The tool owns the number, because the agent could not be trusted with it.

    Asked to add the invoice subtotal to what was already posted and compare
    the result with the order value, the model did the arithmetic in prose -
    and on an invoice for *exactly* the order value, the most ordinary invoice
    there is, the same input returned `pass` and `fail` on consecutive runs.
    Rewording the rule did not fix it. A verdict that changes between identical
    runs cannot be reasoned about.

    Eight of the fifty-one corpus documents bill their order exactly in full
    and three of them must post, so this is not an edge case - it is the
    common case sitting on the boundary.
    """

    @staticmethod
    def _lookup(master, **args):
        from docuflow.agents.tools import build_toolbox

        tools = {t.name: t for t in build_toolbox("", master, None, "invoice", "DOC")}
        return tools["lookup_purchase_order"].run(args)

    def test_billing_an_order_exactly_in_full_is_within_it(self, master) -> None:
        """An order used up is not an order overspent."""
        answer = self._lookup(master, po_number="PO-2026-0412", invoice_subtotal=187_000.00)
        assert answer["within_order_value"] is True
        assert answer["over_by"] == 0.0

    def test_a_single_unit_over_is_over(self, master) -> None:
        answer = self._lookup(master, po_number="PO-2026-0412", invoice_subtotal=187_001.00)
        assert answer["within_order_value"] is False
        assert answer["over_by"] == 1.0

    def test_comfortably_under_is_within(self, master) -> None:
        answer = self._lookup(master, po_number="PO-2026-0412", invoice_subtotal=100_000.00)
        assert answer["within_order_value"] is True

    def test_the_comparison_is_stated_in_words_too(self, master) -> None:
        """The agent reads prose. Giving it only a boolean invites it to
        re-derive the number it was handed."""
        answer = self._lookup(master, po_number="PO-2026-0412", invoice_subtotal=187_000.00)
        assert "within the order" in answer["comparison"]
        over = self._lookup(master, po_number="PO-2026-0412", invoice_subtotal=200_000.00)
        assert "over by" in over["comparison"]

    def test_omitting_the_subtotal_still_returns_the_order(self, master) -> None:
        """The argument is optional: other rules look an order up without
        wanting a comparison, and they must not start erroring."""
        answer = self._lookup(master, po_number="PO-2026-0412")
        assert answer["found"] is True
        assert "within_order_value" not in answer

    def test_a_subtotal_that_is_not_a_number_says_so(self, master) -> None:
        answer = self._lookup(master, po_number="PO-2026-0412", invoice_subtotal="lots")
        assert "comparison_error" in answer
        assert "within_order_value" not in answer

    def test_rounding_noise_does_not_tip_it_over(self, master) -> None:
        """A subtotal reconstructed from line items lands a fraction of a
        satang out, and that must not read as overspending."""
        answer = self._lookup(master, po_number="PO-2026-0412", invoice_subtotal=187_000.004)
        assert answer["within_order_value"] is True
