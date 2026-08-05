"""The twelve demo documents must mean the same thing in both places.

`Demo_Sample/` is run two ways: through `docuflow` against SQLite and the real
rule engine, and through `docuflow_live.ipynb` in Colab against a dict and a
hand-rolled copy of the same checks. A demonstration where those two disagree
is worse than one that only exists in one place, because the disagreement shows
up in front of an audience rather than here.

The register is compared exactly - it is mirrored into the notebook by hand and
hand-mirrored data drifts. The verdicts are then recomputed for all twelve
documents under the notebook's own thresholds, which are read out of the built
notebook rather than repeated here.

No model is called. What a model reads off a page is a different question, asked
by the corpus; this file asks whether the two setups agree about what the
answers mean once read.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from docuflow.samples.demo_set import DEMO_DOCUMENTS, PURCHASE_ORDERS, VENDORS

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "docuflow_live.ipynb"


def _notebook_source() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"]) for cell in payload["cells"] if cell["cell_type"] == "code"
    )


def _literal(name: str, source: str) -> dict:
    """Pull one top-level dict literal out of the notebook and evaluate it.

    `ast.literal_eval` rather than `exec`: these are data, and evaluating them
    as data means a stray call appearing in the notebook fails here instead of
    running during a test.
    """
    match = re.search(rf"^{name} = (\{{.*?^\}})", source, re.S | re.M)
    assert match, f"{name} is not a top-level dict literal in the built notebook"
    return ast.literal_eval(match.group(1))


@pytest.fixture(scope="module")
def source() -> str:
    assert NOTEBOOK.exists(), "build the notebook first: python notebooks/build_live.py"
    return _notebook_source()


@pytest.fixture(scope="module")
def nb_orders(source: str) -> dict:
    return _literal("PURCHASE_ORDERS", source)


@pytest.fixture(scope="module")
def nb_vendors(source: str) -> dict:
    return _literal("VENDORS", source)


class TestTheRegisterIsMirroredExactly:
    def test_every_purchase_order_is_present_with_the_same_amount(self, nb_orders) -> None:
        system = {o["po_number"]: o for o in PURCHASE_ORDERS}
        assert set(nb_orders) == set(system), (
            "the notebook and Demo_Sample/master disagree about which orders exist"
        )
        for number, order in system.items():
            assert nb_orders[number]["amount"] == pytest.approx(float(order["amount"])), number
            assert nb_orders[number]["vendor_id"] == order["vendor_id"], number

    def test_purchase_order_status_is_mirrored(self, nb_orders) -> None:
        """Documents 06 and 12 are decided by this column and nothing else."""
        for order in PURCHASE_ORDERS:
            assert nb_orders[order["po_number"]]["status"] == order["status"], (
                f"{order['po_number']} is '{order['status']}' in the register and "
                f"'{nb_orders[order['po_number']]['status']}' in the notebook - the one "
                "column the agent-rule pair turns on"
            )

    def test_exactly_one_order_is_closed(self, nb_orders) -> None:
        closed = [n for n, o in nb_orders.items() if o["status"] != "open"]
        assert closed == ["PO-2026-9108"], (
            "a second non-open order would give some other document a defect it was "
            "not designed to have"
        )

    def test_every_vendor_bank_account_is_mirrored(self, nb_vendors) -> None:
        system = {v["vendor_id"]: v for v in VENDORS}
        assert set(nb_vendors) == set(system)
        for vendor_id, vendor in system.items():
            assert nb_vendors[vendor_id]["bank_account"] == vendor["bank_account"], vendor_id
            assert nb_vendors[vendor_id]["tax_id"] == vendor["tax_id"], vendor_id


class TestTheNotebookReachesTheSameVerdicts:
    """Recompute all twelve under the notebook's rules and thresholds."""

    @staticmethod
    def _tolerance(source: str) -> float:
        # Declared alongside AMOUNT_TOLERANCE on one line, and the pairing is
        # deliberate - the notebook states both together precisely so nobody
        # reaches for the wrong one.
        match = re.search(
            r"^AMOUNT_TOLERANCE, ARITHMETIC_TOLERANCE = ([0-9.e-]+), ([0-9.e-]+)", source, re.M
        )
        assert match, "the two tolerances are not stated together in the notebook"
        amount, arithmetic = float(match.group(1)), float(match.group(2))
        assert arithmetic < amount, (
            "the arithmetic tolerance must stay far tighter than the commercial one; "
            "sharing a number let an 810-baht misread post as 'totals add up'"
        )
        return arithmetic

    @staticmethod
    def _verdict(doc, orders, vendors, tolerance) -> list[str]:
        """The notebook's four checks plus its agent rule, as reasons to hold."""
        gt = doc.ground_truth
        held: list[str] = []
        slack = lambda base: max(0.01, abs(base) * tolerance)

        sub, tax = gt.get("subtotal"), gt.get("tax_amount")
        total, rate = gt.get("total_amount"), gt.get("tax_rate")

        if None not in (sub, tax, total) and abs(sub + tax - total) > slack(total):
            held.append("totals_consistent")

        if None not in (sub, tax, rate):
            expected = round(sub * rate / 100, 2)
            if abs(expected - tax) > slack(expected):
                held.append("tax_rate_consistent")

        if doc.line_items and sub is not None:
            rows = round(sum(item["amount"] for item in doc.line_items), 2)
            if abs(rows - sub) > slack(sub):
                held.append("line_items_sum_to_subtotal")

        tax_id = gt.get("vendor_tax_id") or gt.get("merchant_tax_id")
        vendor = next((v for v in vendors.values() if v["tax_id"] == tax_id), None)
        if vendor is None:
            held.append("vendor_not_known")
        elif gt.get("bank_account") and gt["bank_account"] != vendor["bank_account"]:
            held.append("bank_account_matches")

        # The rule a person wrote as a sentence: the order must still be open,
        # and what is billed against it must fit inside it.
        reference = gt.get("purchase_order_ref")
        if reference:
            order = orders.get(reference)
            if order is None:
                held.append("purchase_order_still_has_room")
            else:
                if order["status"] != "open":
                    held.append("purchase_order_still_has_room")
                elif sub is not None and sub > float(order["amount"]) + 0.01:
                    held.append("purchase_order_still_has_room")
        return held

    def test_six_post_and_six_are_held(self, source, nb_orders, nb_vendors) -> None:
        tolerance = self._tolerance(source)
        wrong = []
        for slug, _, expected_post, doc in DEMO_DOCUMENTS:
            held = self._verdict(doc, nb_orders, nb_vendors, tolerance)
            if bool(held) == expected_post:          # held and expected to post, or vice versa
                wrong.append(f"{slug}: expected post={expected_post}, held for {held or 'nothing'}")
        assert not wrong, "the notebook's rules disagree with the filenames:\n" + "\n".join(wrong)

    def test_the_split_is_six_six(self, source, nb_orders, nb_vendors) -> None:
        tolerance = self._tolerance(source)
        posting = [
            not self._verdict(doc, nb_orders, nb_vendors, tolerance)
            for _, _, _, doc in DEMO_DOCUMENTS
        ]
        assert (sum(posting), len(posting) - sum(posting)) == (6, 6)

    @pytest.mark.parametrize("slug,expected", [
        ("invoice_purchase_order_open", []),
        ("invoice_purchase_order_closed", ["purchase_order_still_has_room"]),
    ])
    def test_the_agent_rule_pair_turns_only_on_the_order_status(
        self, source, nb_orders, nb_vendors, slug, expected
    ) -> None:
        """Neither may be held for anything else, or the pair proves nothing."""
        doc = next(d for s, _, _, d in DEMO_DOCUMENTS if s == slug)
        assert self._verdict(doc, nb_orders, nb_vendors, self._tolerance(source)) == expected

    def test_each_held_document_is_held_for_exactly_one_reason(
        self, source, nb_orders, nb_vendors
    ) -> None:
        """Two defects on one page make it ambiguous which one the demo showed."""
        tolerance = self._tolerance(source)
        for slug, _, expected_post, doc in DEMO_DOCUMENTS:
            if expected_post:
                continue
            held = self._verdict(doc, nb_orders, nb_vendors, tolerance)
            assert len(held) == 1, f"{slug} is held for {held}"
