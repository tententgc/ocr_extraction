"""The demonstration set must be demonstrable.

Every file in `Demo_Sample/` states its expected outcome in its own name, and
that claim is only worth anything if the document actually carries the defect -
or the absence of one - that the name promises. These tests check the claim
against the document's own ground truth and against the register it is meant to
reconcile with, so a set that has quietly drifted fails here rather than in
front of an audience.

They deliberately do not call a model. What a model reads off the page is what
`tests/test_evaluation.py` and the corpus are for; what this file asserts is
that the *setup* is coherent: the numbers on the page, the rows in the register,
and the word in the filename all agree about what should happen.
"""

from __future__ import annotations

import pytest

from docuflow.samples.demo_set import (
    DEMO_DOCUMENTS,
    PURCHASE_ORDERS,
    VENDORS,
    master_rows,
)

ARITHMETIC_SLACK = 0.01
# Both formats have to appear: a PDF carries a text layer and a PNG does not,
# so a set of PDFs only demonstrates the half where OCR is barely tested.
REQUIRED_ARRIVALS = {"pdf", "png"}


def _vendor_by_tax_id(tax_id: str) -> dict | None:
    return next((v for v in VENDORS if v["tax_id"] == tax_id), None)


def _order(number: str) -> dict | None:
    return next((p for p in PURCHASE_ORDERS if p["po_number"] == number), None)


class TestTheSetIsBalanced:
    def test_five_documents(self) -> None:
        assert len(DEMO_DOCUMENTS) == 5

    def test_both_arrival_formats_are_present(self) -> None:
        arrivals = {arrival for _, arrival, _, _ in DEMO_DOCUMENTS}
        assert REQUIRED_ARRIVALS <= arrivals, (
            f"the set arrives only as {arrivals}; a PNG is the only version that "
            "puts the OCR model under real load"
        )

    def test_it_shows_both_outcomes(self) -> None:
        posting = [posts for _, _, posts, _ in DEMO_DOCUMENTS]
        assert any(posting) and not all(posting), (
            "a demonstration where everything passes shows nothing being caught, "
            "and one where nothing passes shows a system that never posts"
        )

    def test_every_slug_is_unique(self) -> None:
        slugs = [slug for slug, _, _, _ in DEMO_DOCUMENTS]
        assert len(slugs) == len(set(slugs))


class TestEachDocumentCarriesWhatItsNameClaims:
    """A `_POST` document must be clean; a `_HOLD` must have exactly one defect."""

    @pytest.mark.parametrize("slug,posts,doc", [
        (slug, posts, doc) for slug, _, posts, doc in DEMO_DOCUMENTS
    ])
    def test_the_page_adds_up_unless_that_is_the_defect(
        self, slug: str, posts: bool, doc
    ) -> None:
        gt = doc.ground_truth
        subtotal, tax, total = gt["subtotal"], gt["tax_amount"], gt["total_amount"]
        assert abs(subtotal + tax - total) <= ARITHMETIC_SLACK, (
            f"{slug}: the stated total is not the stated subtotal plus tax, which "
            "is a second defect on a document meant to have at most one"
        )

    @pytest.mark.parametrize("slug,posts,doc", [
        (slug, posts, doc) for slug, _, posts, doc in DEMO_DOCUMENTS
    ])
    def test_the_tax_matches_the_stated_rate_unless_that_is_the_defect(
        self, slug: str, posts: bool, doc
    ) -> None:
        gt = doc.ground_truth
        expected = round(gt["subtotal"] * gt["tax_rate"] / 100.0, 2)
        consistent = abs(expected - gt["tax_amount"]) <= ARITHMETIC_SLACK
        if "vat_wrong" in slug:
            assert not consistent, f"{slug} is named for a VAT defect it does not have"
        else:
            assert consistent, (
                f"{slug}: {gt['tax_rate']:g}% of {gt['subtotal']:,.2f} is "
                f"{expected:,.2f}, not {gt['tax_amount']:,.2f}"
            )

    @pytest.mark.parametrize("slug,posts,doc", [
        (slug, posts, doc) for slug, _, posts, doc in DEMO_DOCUMENTS
    ])
    def test_the_bank_account_matches_the_register_unless_that_is_the_defect(
        self, slug: str, posts: bool, doc
    ) -> None:
        gt = doc.ground_truth
        vendor = _vendor_by_tax_id(gt["vendor_tax_id"])
        assert vendor is not None, f"{slug} quotes a tax id nobody in the register has"
        matches = gt["bank_account"] == vendor["bank_account"]
        if "bank_account_changed" in slug:
            assert not matches, f"{slug} is named for a bank-account defect it does not have"
        else:
            assert matches, (
                f"{slug}: the page pays {gt['bank_account']}, the register holds "
                f"{vendor['bank_account']} for {vendor['legal_name']}"
            )

    @pytest.mark.parametrize("slug,posts,doc", [
        (slug, posts, doc) for slug, _, posts, doc in DEMO_DOCUMENTS
    ])
    def test_the_amount_fits_the_purchase_order_unless_that_is_the_defect(
        self, slug: str, posts: bool, doc
    ) -> None:
        """Compared against the subtotal: orders are recorded net of tax."""
        reference = doc.ground_truth.get("purchase_order_ref")
        if not reference:
            return
        order = _order(reference)
        assert order is not None, f"{slug} quotes {reference}, which is not in the register"
        within = doc.ground_truth["subtotal"] <= float(order["amount"]) + ARITHMETIC_SLACK
        if "over_purchase_order" in slug:
            assert not within, f"{slug} is named for an overrun it does not have"
        else:
            assert within, (
                f"{slug}: bills {doc.ground_truth['subtotal']:,.2f} against "
                f"{reference}, which is for {float(order['amount']):,.2f}"
            )

    @pytest.mark.parametrize("slug,posts,doc", [
        (slug, posts, doc) for slug, _, posts, doc in DEMO_DOCUMENTS
    ])
    def test_line_items_sum_to_the_subtotal(self, slug: str, posts: bool, doc) -> None:
        if not doc.line_items:
            return
        rows = round(sum(item["amount"] for item in doc.line_items), 2)
        assert abs(rows - doc.ground_truth["subtotal"]) <= ARITHMETIC_SLACK, (
            f"{slug}: the rows come to {rows:,.2f} and the subtotal says "
            f"{doc.ground_truth['subtotal']:,.2f}"
        )


class TestTheRegisterSupportsTheSet:
    def test_every_vendor_on_a_page_is_registered(self) -> None:
        for slug, _, _, doc in DEMO_DOCUMENTS:
            assert _vendor_by_tax_id(doc.ground_truth["vendor_tax_id"]), (
                f"{slug} names a vendor the register does not have, so every rule "
                "that needs a lookup will abstain rather than decide"
            )

    def test_every_purchase_order_quoted_exists(self) -> None:
        for slug, _, _, doc in DEMO_DOCUMENTS:
            reference = doc.ground_truth.get("purchase_order_ref")
            if reference:
                assert _order(reference), f"{slug} quotes a purchase order nobody issued"

    def test_each_order_belongs_to_the_vendor_billing_against_it(self) -> None:
        for slug, _, _, doc in DEMO_DOCUMENTS:
            reference = doc.ground_truth.get("purchase_order_ref")
            if not reference:
                continue
            order = _order(reference)
            vendor = _vendor_by_tax_id(doc.ground_truth["vendor_tax_id"])
            assert order["vendor_id"] == vendor["vendor_id"], (
                f"{slug}: {reference} was raised for {order['vendor_id']} and the "
                f"invoice comes from {vendor['vendor_id']} - an unintended second defect"
            )

    def test_no_two_vendors_share_a_tax_id(self) -> None:
        """A tax id is the key a vendor is found by; two rows sharing one is a coin toss."""
        ids = [v["tax_id"] for v in VENDORS]
        assert len(ids) == len(set(ids))

    def test_contracts_are_absent_rather_than_empty(self) -> None:
        """The loader refuses a header with no rows, mid-demonstration."""
        assert master_rows()["contracts"] == ()
