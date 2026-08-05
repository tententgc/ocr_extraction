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
