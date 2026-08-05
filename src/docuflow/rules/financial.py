"""Rules for money-bearing documents: invoices, receipts and purchase orders.

These are the rules that stop a wrong number reaching the ledger, so most of them
are `blocking` rather than advisory.
"""

from __future__ import annotations

from datetime import date

from docuflow.models import RuleFinding
from docuflow.normalize.identifiers import reference_key
from docuflow.rules.context import RuleContext
from docuflow.rules.compare import same_value
from docuflow.rules.engine import failed, passed, rule
from docuflow.rules.repair import Money, explain_difference, money_fixes, value_fix

MONEY_DOCS = ("invoice", "receipt", "purchase_order")

# `find_vendor` methods that mean the name itself was not in doubt. Everything
# else resolved the supplier while disagreeing about how it is spelled, which
# is what `vendor_name_matches_register` exists to report.
_EXACT_VENDOR_METHODS = frozenset({
    "tax id",                      # an exact identifier; the name was not used
    "exact name",
    "name (ignoring legal form)",  # "Co., Ltd." omitted is not a misread
})
# VAT rates and the purchase-order threshold are finance policy and come from
# configuration via RuleContext, not from constants here.
# Absolute currency units of slack when comparing computed and printed totals.
ARITHMETIC_ABSOLUTE_SLACK = 0.02


def _tolerance(context: RuleContext, base: float) -> float:
    """How far a document may differ from a *purchase order*.

    A commercial allowance, and a wide one: 2% of a large order is real money,
    on purpose. Only for comparisons against another document.
    """
    return max(abs(base) * context.amount_tolerance, ARITHMETIC_ABSOLUTE_SLACK)


def _arithmetic_slack(context: RuleContext, base: float) -> float:
    """How far a document may differ from *itself*.

    Rounding, and nothing else. Using the commercial allowance here was a real
    silent error: on a 200,090-baht invoice it allowed 4,001 baht of drift, so a
    total misread as 200,900 was reported as adding up and posted.
    """
    return max(abs(base) * context.arithmetic_tolerance, ARITHMETIC_ABSOLUTE_SLACK)


def _money(context: RuleContext) -> Money:
    """The amounts that have to agree with each other, as they were read.

    Gathered once and handed to the diagnosis so that every arithmetic rule
    reasons about the whole set rather than the pair it happens to compare. A
    rule that only sees the subtotal and the total cannot know which of them is
    wrong; one that also sees the line items and the tax rate usually can.
    """
    amounts = [item.amount for item in context.line_items if item.amount is not None]
    return Money(
        subtotal=_as_float(context.value("subtotal")),
        tax_amount=_as_float(context.value("tax_amount")),
        tax_rate=_as_float(context.value("tax_rate")),
        total_amount=_as_float(context.value("total_amount")),
        line_total=round(sum(amounts), 2) if amounts else None,
    )


def _as_float(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


MONEY_LABELS = {
    "subtotal": "the subtotal",
    "tax_amount": "the tax",
    "tax_rate": "the tax rate",
    "total_amount": "the total",
}


@rule(*MONEY_DOCS, emits=("invoice_totals_consistent",), severity="blocking")
def totals_consistent(context: RuleContext) -> RuleFinding | None:
    """Subtotal + tax must equal the stated total."""
    subtotal = context.value("subtotal")
    tax = context.value("tax_amount")
    total = context.value("total_amount")
    if subtotal is None or total is None:
        return None

    computed = float(subtotal) + float(tax or 0.0)
    difference = abs(computed - float(total))
    if difference <= _arithmetic_slack(context, total):
        return passed("invoice_totals_consistent", f"Totals add up: {subtotal:,.2f} + {tax or 0:,.2f} = {total:,.2f}.")

    # Which of the three is wrong is a different question from whether they
    # agree, and it is the one the reviewer actually has to answer. The line
    # items and the tax rate are extra relations over the same numbers, so where
    # they exist they usually settle it.
    fixes = money_fixes(_money(context), MONEY_LABELS)
    # One clause, not the whole argument: the suggestion box below carries the
    # reasoning, and printing it twice means the reviewer reads it neither time.
    culprit = f" {fixes[0].headline}" if fixes else ""

    return failed(
        "invoice_totals_consistent",
        f"Totals do not add up: {subtotal:,.2f} + {tax or 0:,.2f} = {computed:,.2f}, "
        f"but the document states {total:,.2f} (difference {difference:,.2f}).{culprit}",
        severity="blocking",
        fields=["subtotal", "tax_amount", "total_amount"],
        fixes=fixes,
        computed_total=round(computed, 2),
        stated_total=round(float(total), 2),
        difference=round(difference, 2),
    )


@rule(*MONEY_DOCS, severity="blocking")
def tax_rate_consistent(context: RuleContext) -> RuleFinding | None:
    """The tax amount must match the stated rate applied to the subtotal."""
    subtotal = context.value("subtotal")
    tax = context.value("tax_amount")
    rate = context.value("tax_rate")
    if subtotal is None or tax is None or rate is None or float(subtotal) == 0:
        return None

    expected = float(subtotal) * float(rate) / 100.0
    difference = abs(expected - float(tax))
    if difference <= _arithmetic_slack(context, expected or 1.0):
        return passed("tax_rate_consistent", f"Tax of {tax:,.2f} matches {rate:g}% of {subtotal:,.2f}.")
    fixes = money_fixes(_money(context), MONEY_LABELS)
    if not fixes:
        # Nothing else constrains these three, so the tax is the value this rule
        # is entitled to name - but say so as one relation's answer, not as a
        # settled fact.
        fixes = value_fix(
            "tax_amount", float(tax), round(expected, 2),
            f"{rate:g}% of a subtotal of {subtotal:,.2f} is {expected:,.2f}",
            source="arithmetic", confidence=0.7,
        )
    slip = explain_difference(float(tax), expected)

    return failed(
        "tax_rate_consistent",
        f"Tax of {tax:,.2f} does not match {rate:g}% of {subtotal:,.2f} (expected {expected:,.2f})."
        + (f" Looking at the page: {slip}." if slip else ""),
        severity="blocking",
        fields=["subtotal", "tax_rate", "tax_amount"],
        fixes=fixes,
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
    if difference <= _arithmetic_slack(context, base):
        return passed("line_items_sum_to_subtotal", f"{len(amounts)} line items sum to {line_total:,.2f}.")
    fixes = money_fixes(_money(context), MONEY_LABELS)
    slip = explain_difference(float(base), line_total)

    return failed(
        "line_items_sum_to_subtotal",
        f"Line items sum to {line_total:,.2f} but the subtotal is {float(base):,.2f} "
        f"(difference {difference:,.2f}). A line may have been missed or mis-read."
        + (f" Looking at the page: {slip}." if slip else ""),
        severity="warning",
        fields=["subtotal"],
        fixes=fixes,
        line_total=round(line_total, 2),
    )


@rule(*MONEY_DOCS, "form", severity="warning")
def vendor_name_matches_register(context: RuleContext) -> RuleFinding | None:
    """The name as read must be the name on file, once the vendor is resolved.

    This exists because of two documents that posted while wrong, and it is the
    only signal in the system that could have caught them.

    A Thai invoice read `บริษัท สยามชัพพลาย จำกัด` where the page says
    `สยามซัพพลาย` - **ซ** and **ช** differ by a single loop. Everything held.
    The arithmetic closed, the account matched, `vendor_known` passed, and
    confidence landed on 0.843 against a bar of 0.840. Nothing else in the
    system can see it: grounding cannot catch an OCR misread, because the span
    exists in the OCR output and the OCR output is what was wrong, and two
    channels reading that same output both read `ชัพพลาย`.

    But the register holds the answer. `find_vendor` resolved the document to
    the right supplier *by fuzzy match* and recorded how far it had to stretch
    to do it. **A vendor that only matched fuzzily is a statement that the name
    on the page is not the name on file**, and against an authoritative
    register that is direct evidence of a misread on a critical field.

    A `warning`, not a block, and both halves of that are deliberate. The
    payment is not at risk - money follows the registered bank account, not the
    spelling - so holding every one-stroke difference would be a blocking rule
    wearing a warning's name. What it does instead is join `MISREAD_SIGNALS`,
    where a warning costs a twentieth of the confidence score. That is enough
    to take a document sitting on the threshold below it and no more: three
    unrelated warnings still post, which is the shape a warning is required to
    have.
    """
    name = (
        context.value("vendor_name")
        or context.value("merchant_name")
        or context.value("applicant_name")
    )
    match = context.vendor_match
    if not name or not match.matched:
        # An unresolved vendor is `vendor_known`'s business, not this rule's.
        return None

    # The name the register actually matched on, which is not always the legal
    # name: a supplier registered in English carries a Thai alias, and a Thai
    # invoice matches through the alias. Suggesting `legal_name` there would
    # tell a reviewer that a correct Thai page misread an English name.
    registered = match.matched_name or match.vendor["legal_name"]
    # Anything the register itself would call the same name is the same name.
    # `_core_name` strips the legal form, so "Siam Tech Co., Ltd." against
    # "Siam Tech" is not a misread - it is an invoice omitting a suffix, which
    # is ordinary and must not cost a document its confidence.
    if match.method in _EXACT_VENDOR_METHODS:
        return passed(
            "vendor_name_matches_register",
            f"Vendor name matches the register ({registered}) by {match.method}.",
            method=match.method,
        )

    field = next(
        (key for key in ("vendor_name", "merchant_name", "applicant_name")
         if context.value(key) == name),
        "vendor_name",
    )
    return failed(
        "vendor_name_matches_register",
        f"Read the vendor as {name!r}, but the register spells it {registered!r} "
        f"({match.score:.0%} similar). The document resolved to the right supplier, "
        f"so the payment is not at risk - but a name that only matches this far is "
        f"usually a character misread, and the ledger would record the misreading.",
        severity="warning",
        fields=[field],
        # High confidence for once: the register is authoritative about its own
        # supplier's name, and this fix cannot name a second field.
        fixes=value_fix(
            field, name, registered,
            f"the registered spelling for {match.vendor['vendor_id']}",
            source="master_data", confidence=0.9,
        ),
        registered=registered,
        extracted=str(name),
        score=match.score,
        method=match.method,
    )


@rule(*MONEY_DOCS, "form", severity="blocking")
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
    # A near-miss is worth offering as a correction; a field of three equally
    # plausible names is not, because picking the first would be arbitrary and
    # would look like a finding.
    field = next(
        (key for key in ("vendor_name", "merchant_name", "applicant_name")
         if context.value(key) == name),
        "vendor_name",
    )
    fixes = (
        value_fix(
            field, name, match.candidates[0]["legal_name"],
            f"the closest registered vendor, matching {match.score:.0%} of the name as read"
            f" ({match.candidates[0]['vendor_id']})",
            source="master_data", confidence=round(min(match.score, 0.75), 2),
        )
        if len(match.candidates) == 1 and match.score >= 0.75
        else ()
    )

    return failed(
        "vendor_known",
        f"'{name}' is not in the vendor master file.{hint}{consequence}",
        severity="blocking" if blocks else "warning",
        fields=["vendor_name", "merchant_name", "applicant_name"],
        fixes=fixes,
        best_score=match.score,
        candidates=[candidate["legal_name"] for candidate in match.candidates[:3]],
    )


@rule(*MONEY_DOCS, severity="blocking")
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


@rule("invoice", severity="blocking")
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

    # Offered at low confidence and never for an account that belongs to someone
    # else. A changed account is the signature of payment-redirection fraud: if
    # the document is genuine the master file is stale, and if it is not, the
    # registered account is the only safe one - but either way a person decides,
    # not a suggestion box.
    fixes = (
        ()
        if owner
        else value_fix(
            "bank_account", str(account), registered,
            f"{vendor['legal_name']} is registered against {registered}; confirm the change "
            f"with the vendor by a channel you already had before accepting this",
            source="master_data", confidence=0.35,
        )
    )
    return failed(
        "bank_account_matches_vendor",
        f"Bank account {account} does not match the account registered for "
        f"{vendor['legal_name']} ({registered}).{extra}",
        severity="blocking",
        fields=["bank_account"],
        fixes=fixes,
        registered=registered,
        extracted=str(account),
    )


@rule("invoice", severity="blocking")
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


@rule("invoice", emits=("po_exists", "po_vendor_matches"), severity="blocking")
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


@rule("invoice", severity="blocking")
def po_amount_within_tolerance(context: RuleContext) -> RuleFinding | None:
    """The invoiced amount must not overrun the purchase order.

    The two directions are not the same finding and no longer carry the same
    severity.

    Billing **more** than was ordered is a payment risk: money leaves for
    something nobody agreed to, which is the failure this system exists to
    stop. That stays `blocking`.

    Billing **less** is not. Partial delivery, progress billing and short
    shipments all produce an invoice under its order, all are ordinary, and
    none of them pays anybody too much. Held as `blocking` it stopped
    perfectly good documents - three of them in the demonstration set alone,
    each of which looked like the system misbehaving until the rule was read.
    It is worth reporting, because an invoice far under its order can also mean
    the wrong order was quoted, so it comes back as a `warning`: recorded and
    shown, holding nothing.
    """
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

    overrun = difference > 0
    share = abs(difference) / po_amount if po_amount else 0.0
    if overrun:
        message = (
            f"Invoiced {comparable:,.2f} is {abs(difference):,.2f} over the purchase order "
            f"value of {po_amount:,.2f} ({share:.1%}), outside the "
            f"{context.amount_tolerance:.0%} tolerance."
        )
    else:
        message = (
            f"Invoiced {comparable:,.2f} is {abs(difference):,.2f} under the purchase order "
            f"value of {po_amount:,.2f} ({share:.1%}). Normal for a partial delivery or "
            f"progress billing; worth a look only if this invoice should have been for "
            f"the whole order, or quotes the wrong one."
        )
    return failed(
        "po_amount_within_tolerance",
        message,
        severity="blocking" if overrun else "warning",
        fields=["total_amount", "subtotal", "purchase_order_ref"],
        po_amount=po_amount,
        invoiced=round(comparable, 2),
        difference=round(difference, 2),
        direction="over" if overrun else "under",
    )


@rule("invoice", severity="blocking")
def po_currency_matches(context: RuleContext) -> RuleFinding | None:
    """Currency on the invoice must match the purchase order."""
    reference = context.value("purchase_order_ref")
    currency = context.value("currency")
    if not reference or not currency:
        return None
    order = context.master.get_purchase_order(reference)
    if order is None:
        return None
    if same_value(order["currency"], currency):
        return passed("po_currency_matches", f"Currency {currency} matches the purchase order.")
    return failed(
        "po_currency_matches",
        f"Invoice is in {currency} but purchase order {reference} is in {order['currency']}.",
        severity="blocking",
        fields=["currency"],
        fixes=value_fix(
            "currency", currency, order["currency"],
            f"purchase order {reference} was raised in {order['currency']}",
            source="master_data", confidence=0.7,
        ),
    )


@rule(*MONEY_DOCS, "contract", "form", severity="blocking")
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
    if same_value(currency, vendor["currency"]):
        return passed("currency_matches_vendor_default", f"{currency} is this vendor's usual currency.")
    return failed(
        "currency_matches_vendor_default",
        f"Invoice is in {currency} but {vendor['legal_name']} normally bills in {vendor['currency']}.",
        severity="warning",
        fields=["currency"],
    )
