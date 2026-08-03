"""Rules for money-bearing documents: invoices, receipts and purchase orders.

These are the rules that stop a wrong number reaching the ledger, so most of them
are `blocking` rather than advisory.
"""

from __future__ import annotations

from datetime import date

from docuflow.models import RuleFinding
from docuflow.normalize.identifiers import reference_key
from docuflow.rules.context import RuleContext
from docuflow.rules.engine import failed, passed, rule

MONEY_DOCS = ("invoice", "receipt", "purchase_order")
# VAT rates and the purchase-order threshold are finance policy and come from
# configuration via RuleContext, not from constants here.
# Absolute currency units of slack when comparing computed and printed totals.
ARITHMETIC_ABSOLUTE_SLACK = 0.02


def _tolerance(context: RuleContext, base: float) -> float:
    """Relative tolerance with a small absolute floor for rounding."""
    return max(abs(base) * context.amount_tolerance, ARITHMETIC_ABSOLUTE_SLACK)


@rule(*MONEY_DOCS)
def totals_consistent(context: RuleContext) -> RuleFinding | None:
    """Subtotal + tax must equal the stated total."""
    subtotal = context.value("subtotal")
    tax = context.value("tax_amount")
    total = context.value("total_amount")
    if subtotal is None or total is None:
        return None

    computed = float(subtotal) + float(tax or 0.0)
    difference = abs(computed - float(total))
    if difference <= _tolerance(context, total):
        return passed("invoice_totals_consistent", f"Totals add up: {subtotal:,.2f} + {tax or 0:,.2f} = {total:,.2f}.")

    return failed(
        "invoice_totals_consistent",
        f"Totals do not add up: {subtotal:,.2f} + {tax or 0:,.2f} = {computed:,.2f}, "
        f"but the document states {total:,.2f} (difference {difference:,.2f}).",
        severity="blocking",
        fields=["subtotal", "tax_amount", "total_amount"],
        computed_total=round(computed, 2),
        stated_total=round(float(total), 2),
        difference=round(difference, 2),
    )


@rule(*MONEY_DOCS)
def tax_rate_consistent(context: RuleContext) -> RuleFinding | None:
    """The tax amount must match the stated rate applied to the subtotal."""
    subtotal = context.value("subtotal")
    tax = context.value("tax_amount")
    rate = context.value("tax_rate")
    if subtotal is None or tax is None or rate is None or float(subtotal) == 0:
        return None

    expected = float(subtotal) * float(rate) / 100.0
    difference = abs(expected - float(tax))
    if difference <= _tolerance(context, expected or 1.0):
        return passed("tax_rate_consistent", f"Tax of {tax:,.2f} matches {rate:g}% of {subtotal:,.2f}.")
    return failed(
        "tax_rate_consistent",
        f"Tax of {tax:,.2f} does not match {rate:g}% of {subtotal:,.2f} (expected {expected:,.2f}).",
        severity="blocking",
        fields=["subtotal", "tax_rate", "tax_amount"],
        expected_tax=round(expected, 2),
    )


@rule("invoice", "receipt")
def vat_rate_expected(context: RuleContext) -> RuleFinding | None:
    """Flag unusual VAT rates on Thai-currency documents."""
    rate = context.value("tax_rate")
    currency = context.value("currency")
    if rate is None or currency != "THB":
        return None
    if float(rate) in context.expected_vat_rates:
        return passed("vat_rate_expected", f"VAT rate of {rate:g}% is standard for Thailand.")
    return failed(
        "vat_rate_expected",
        f"VAT rate of {rate:g}% is not one of the expected rates "
        f"({', '.join(f'{r:g}%' for r in context.expected_vat_rates)}).",
        severity="warning",
        fields=["tax_rate"],
    )


@rule(*MONEY_DOCS)
def line_items_sum_to_subtotal(context: RuleContext) -> RuleFinding | None:
    """Line item amounts should reconcile to the subtotal."""
    if not context.line_items:
        return None
    amounts = [item.amount for item in context.line_items if item.amount is not None]
    if not amounts:
        return None

    base = context.value("subtotal")
    if base is None:
        base = context.value("total_amount")
        if base is None:
            return None

    line_total = sum(amounts)
    difference = abs(line_total - float(base))
    if difference <= _tolerance(context, base):
        return passed("line_items_sum_to_subtotal", f"{len(amounts)} line items sum to {line_total:,.2f}.")
    return failed(
        "line_items_sum_to_subtotal",
        f"Line items sum to {line_total:,.2f} but the subtotal is {float(base):,.2f} "
        f"(difference {difference:,.2f}). A line may have been missed or mis-read.",
        severity="warning",
        fields=["subtotal"],
        line_total=round(line_total, 2),
    )


@rule(*MONEY_DOCS, "form")
def vendor_known(context: RuleContext) -> RuleFinding | None:
    """The counterparty must exist in the vendor master file."""
    name = context.value("vendor_name") or context.value("merchant_name") or context.value("applicant_name")
    if not name:
        return None

    match = context.vendor_match
    if match.matched:
        return passed(
            "vendor_known",
            f"Matched vendor {match.vendor['legal_name']} ({match.vendor['vendor_id']}) by {match.method}.",
            vendor_id=match.vendor_id,
            method=match.method,
            score=match.score,
        )

    # Only documents that commit money to a party block on an unknown
    # counterparty. A receipt records a purchase already made, and an onboarding
    # form is submitted *because* the applicant is not registered yet.
    blocks = context.doc_type in context.unknown_vendor_blocks
    suggestions = ", ".join(candidate["legal_name"] for candidate in match.candidates[:3])
    hint = f" Names that look similar (not confirmed): {suggestions}." if suggestions else ""
    consequence = (
        " Register the vendor before this can post."
        if blocks
        else " It will be recorded against an unregistered counterparty."
    )
    return failed(
        "vendor_known",
        f"'{name}' is not in the vendor master file.{hint}{consequence}",
        severity="blocking" if blocks else "warning",
        fields=["vendor_name", "merchant_name", "applicant_name"],
        best_score=match.score,
        candidates=[candidate["legal_name"] for candidate in match.candidates[:3]],
    )


@rule(*MONEY_DOCS)
def vendor_active(context: RuleContext) -> RuleFinding | None:
    """A vendor on hold or blocked must not be paid automatically."""
    vendor = context.vendor
    if not vendor:
        return None
    status = (vendor.get("status") or "active").lower()
    if status == "active":
        return passed("vendor_active", f"Vendor {vendor['legal_name']} is active.")
    return failed(
        "vendor_active",
        f"Vendor {vendor['legal_name']} has master-data status '{status}' and must not be processed automatically.",
        severity="blocking",
        fields=["vendor_name"],
        status=status,
    )


@rule("invoice")
def bank_account_matches_vendor(context: RuleContext) -> RuleFinding | None:
    """Payment details must match the vendor's registered account.

    A changed bank account on an otherwise valid invoice is the signature of
    payment-redirection fraud, so this blocks rather than warns.
    """
    account = context.value("bank_account")
    vendor = context.vendor
    if not account or not vendor:
        return None

    registered = vendor.get("bank_account")
    if not registered:
        return failed(
            "bank_account_matches_vendor",
            f"Vendor {vendor['legal_name']} has no bank account on file to check against.",
            severity="warning",
            fields=["bank_account"],
        )

    if reference_key(account) == reference_key(registered):
        return passed("bank_account_matches_vendor", "Bank account matches the vendor master record.")

    owner = context.master.vendor_for_bank_account(account)
    extra = f" It belongs to {owner['legal_name']}." if owner else " It is not on file for any vendor."
    return failed(
        "bank_account_matches_vendor",
        f"Bank account {account} does not match the account registered for "
        f"{vendor['legal_name']} ({registered}).{extra}",
        severity="blocking",
        fields=["bank_account"],
        registered=registered,
        extracted=str(account),
    )


@rule("invoice")
def purchase_order_required(context: RuleContext) -> RuleFinding | None:
    """Invoices above the threshold must quote a purchase order."""
    total = context.value("total_amount")
    reference = context.value("purchase_order_ref")
    if total is None:
        return None
    if reference:
        return passed("purchase_order_required", f"Invoice quotes purchase order {reference}.")
    if float(total) < context.po_required_above:
        return passed(
            "purchase_order_required",
            f"No purchase order quoted, but the value ({total:,.2f}) is below the "
            f"{context.po_required_above:,.0f} threshold.",
        )
    return failed(
        "purchase_order_required",
        f"No purchase order quoted on an invoice of {total:,.2f}, which is above the "
        f"{context.po_required_above:,.0f} threshold.",
        severity="blocking",
        fields=["purchase_order_ref"],
    )


@rule("invoice")
def po_exists_and_matches(context: RuleContext) -> RuleFinding | None:
    """Three-way match: the quoted PO must exist and belong to this vendor."""
    reference = context.value("purchase_order_ref")
    if not reference:
        return None

    order = context.master.get_purchase_order(reference)
    if order is None:
        return failed(
            "po_exists",
            f"Purchase order {reference} does not exist in the system.",
            severity="blocking",
            fields=["purchase_order_ref"],
        )

    vendor = context.vendor
    if vendor and order["vendor_id"] != vendor["vendor_id"]:
        po_vendor = context.master.get_vendor(order["vendor_id"])
        return failed(
            "po_vendor_matches",
            f"Purchase order {reference} belongs to "
            f"{po_vendor['legal_name'] if po_vendor else order['vendor_id']}, "
            f"not {vendor['legal_name']}.",
            severity="blocking",
            fields=["purchase_order_ref", "vendor_name"],
        )
    return passed("po_exists", f"Purchase order {reference} exists and belongs to this vendor.")


@rule("invoice")
def po_amount_within_tolerance(context: RuleContext) -> RuleFinding | None:
    """The invoiced amount must not overrun the purchase order."""
    reference = context.value("purchase_order_ref")
    total = context.value("total_amount")
    subtotal = context.value("subtotal")
    if not reference or total is None:
        return None

    order = context.master.get_purchase_order(reference)
    if order is None:
        return None

    po_amount = float(order["amount"])
    # Purchase orders are recorded net of tax, so compare against the subtotal
    # when we have it.
    comparable = float(subtotal) if subtotal is not None else float(total)
    difference = comparable - po_amount
    allowed = _tolerance(context, po_amount)

    if abs(difference) <= allowed:
        return passed(
            "po_amount_within_tolerance",
            f"Invoiced {comparable:,.2f} against a purchase order of {po_amount:,.2f} (within tolerance).",
        )
    direction = "over" if difference > 0 else "under"
    return failed(
        "po_amount_within_tolerance",
        f"Invoiced {comparable:,.2f} is {abs(difference):,.2f} {direction} the purchase order "
        f"value of {po_amount:,.2f} ({abs(difference) / po_amount:.1%}), outside the "
        f"{context.amount_tolerance:.0%} tolerance.",
        severity="blocking",
        fields=["total_amount", "subtotal", "purchase_order_ref"],
        po_amount=po_amount,
        invoiced=round(comparable, 2),
        difference=round(difference, 2),
    )


@rule("invoice")
def po_currency_matches(context: RuleContext) -> RuleFinding | None:
    """Currency on the invoice must match the purchase order."""
    reference = context.value("purchase_order_ref")
    currency = context.value("currency")
    if not reference or not currency:
        return None
    order = context.master.get_purchase_order(reference)
    if order is None:
        return None
    if order["currency"] == currency:
        return passed("po_currency_matches", f"Currency {currency} matches the purchase order.")
    return failed(
        "po_currency_matches",
        f"Invoice is in {currency} but purchase order {reference} is in {order['currency']}.",
        severity="blocking",
        fields=["currency"],
    )


@rule(*MONEY_DOCS, "contract", "form")
def duplicate_document(context: RuleContext) -> RuleFinding | None:
    """The same reference must not be posted twice."""
    if context.duplicate_of is None:
        return passed("duplicate_document", "No earlier record with this reference was found.")
    existing = context.duplicate_of
    return failed(
        "duplicate_document",
        f"Already posted as record {existing.get('record_id')} on "
        f"{str(existing.get('posted_at', ''))[:10]} for {existing.get('amount')} "
        f"{existing.get('currency') or ''}. This looks like a re-sent document.",
        severity="blocking",
        fields=["invoice_number", "receipt_number", "po_number", "reference_number"],
        existing_record_id=existing.get("record_id"),
    )


@rule("invoice")
def due_date_after_issue(context: RuleContext) -> RuleFinding | None:
    """A due date before the invoice date is a misread."""
    issued = context.value("invoice_date")
    due = context.value("due_date")
    if not issued or not due:
        return None
    try:
        issued_date, due_date = date.fromisoformat(issued), date.fromisoformat(due)
    except (TypeError, ValueError):
        return None
    if due_date >= issued_date:
        return passed("due_date_after_issue", f"Due date {due} follows the invoice date {issued}.")
    return failed(
        "due_date_after_issue",
        f"Due date {due} is before the invoice date {issued}.",
        severity="warning",
        fields=["due_date", "invoice_date"],
    )


@rule("invoice")
def currency_matches_vendor_default(context: RuleContext) -> RuleFinding | None:
    """Advisory: an unexpected currency for this vendor."""
    currency = context.value("currency")
    vendor = context.vendor
    if not currency or not vendor or not vendor.get("currency"):
        return None
    if currency == vendor["currency"]:
        return passed("currency_matches_vendor_default", f"{currency} is this vendor's usual currency.")
    return failed(
        "currency_matches_vendor_default",
        f"Invoice is in {currency} but {vendor['legal_name']} normally bills in {vendor['currency']}.",
        severity="warning",
        fields=["currency"],
    )
