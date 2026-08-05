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
    def test_ten_documents_split_evenly(self) -> None:
        posting = [posts for _, _, posts, _ in DEMO_DOCUMENTS]
        assert len(DEMO_DOCUMENTS) == 10
        assert sum(posting) == 5, "five must pass and five must fail"

    def test_both_arrival_formats_are_present(self) -> None:
        arrivals = {arrival for _, arrival, _, _ in DEMO_DOCUMENTS}
        assert REQUIRED_ARRIVALS <= arrivals, (
            f"the set arrives only as {arrivals}; a PNG is the only version that "
            "puts the OCR model under real load"
        )

    def test_the_failures_fail_for_different_reasons(self) -> None:
        """Five documents failing the same check demonstrate one check."""
        reasons = [doc.expected_findings for _, _, posts, doc in DEMO_DOCUMENTS if not posts]
        assert all(reasons), "every failing document must name what it trips"
        assert len(set(reasons)) == len(reasons), (
            "two failing documents trip the same rule; a demo where every failure "
            "is caught the same way teaches that the system has one mechanism"
        )

    def test_more_than_one_document_type(self) -> None:
        """A classifier that only ever sees invoices demonstrates no classifier."""
        assert len({doc.doc_type for _, _, _, doc in DEMO_DOCUMENTS}) > 1

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
        adds_up = abs(gt["subtotal"] + gt["tax_amount"] - gt["total_amount"]) <= ARITHMETIC_SLACK
        if "total_transposed" in slug:
            assert not adds_up, f"{slug} is named for a total that does not add up"
        else:
            assert adds_up, (
                f"{slug}: the stated total is not the stated subtotal plus tax, "
                "which is a second defect on a document meant to have at most one"
            )

    @pytest.mark.parametrize("slug,posts,doc", [
        (slug, posts, doc) for slug, _, posts, doc in DEMO_DOCUMENTS
    ])
    def test_the_tax_matches_the_stated_rate_unless_that_is_the_defect(
        self, slug: str, posts: bool, doc
    ) -> None:
        gt = doc.ground_truth
        if "tax_rate" not in gt:
            return                      # a till receipt prints tax, never a rate
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
        if "unknown_vendor" in slug:
            assert _vendor_by_tax_id(gt["vendor_tax_id"]) is None, (
                f"{slug} is named for an unregistered vendor and the register has it"
            )
            return
        if "bank_account" not in gt:
            return                      # a receipt names no account to pay into
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
        """Except the one whose whole point is that it is not."""
        for slug, _, _, doc in DEMO_DOCUMENTS:
            tax_id = doc.ground_truth.get("vendor_tax_id") or doc.ground_truth.get("merchant_tax_id")
            assert tax_id, f"{slug} names no counterparty at all"
            registered = _vendor_by_tax_id(tax_id) is not None
            if "unknown_vendor" in slug:
                assert not registered, f"{slug} is meant to be unregistered"
            else:
                assert registered, (
                    f"{slug} names a vendor the register does not have, so every "
                    "rule that needs a lookup will abstain rather than decide"
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
            vendor = _vendor_by_tax_id(doc.ground_truth.get("vendor_tax_id", ""))
            if vendor is None:
                continue                # the unregistered vendor quotes no order
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
