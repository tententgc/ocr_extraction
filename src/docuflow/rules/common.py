"""Rules that apply to every document type."""

from __future__ import annotations

from datetime import date

from docuflow.models import RuleFinding
from docuflow.normalize.identifiers import thai_tax_id_checksum_ok
from docuflow.rules.context import RuleContext
from docuflow.rules.engine import failed, passed, rule
from docuflow.rules.repair import tax_id_fixes

THAI_TAX_ID_LENGTH = 13


def _check_tax_id(value: str | None, field_name: str) -> RuleFinding | None:
    if not value:
        return None
    digits = "".join(char for char in str(value) if char.isdigit())
    if len(digits) != THAI_TAX_ID_LENGTH:
        # Non-Thai registrations are not checksummable here.
        return passed("tax_id_checksum", "Tax ID is not a 13-digit Thai number; checksum not applicable.")
    if thai_tax_id_checksum_ok(digits):
        return passed("tax_id_checksum", "Thai tax ID checksum is valid.")

    # The check digit is strong enough to locate the damage as well as detect
    # it: exactly one plausible single-digit change makes the number valid often
    # enough to be worth offering, and pointing at the character is what lets a
    # reviewer confirm it against the page in a second.
    fixes = tax_id_fixes(field_name, digits)
    located = f" {fixes[0].reason.capitalize()}." if fixes else ""
    return failed(
        "tax_id_checksum",
        f"Thai tax ID {digits} fails its check-digit test - it was either mis-read "
        f"or is invalid.{located}",
        severity="blocking",
        fields=[field_name],
        fixes=fixes,
        tax_id=digits,
    )


@rule("*", severity="blocking")
def tax_id_checksum(context: RuleContext) -> RuleFinding | None:
    """Validate any Thai tax ID on the document."""
    for field_name in ("vendor_tax_id", "merchant_tax_id", "tax_id"):
        if context.get(field_name) is not None:
            finding = _check_tax_id(context.value(field_name), field_name)
            if finding is not None:
                return finding
    return None


def _date_fields(context: RuleContext) -> list[tuple[str, str]]:
    """`(field_name, iso_value)` for every populated date field."""
    return [
        (spec.name, context.value(spec.name))
        for spec in context.schema.fields
        if spec.type == "date" and context.value(spec.name)
    ]


# Past this far ahead a date is not a document that arrived early, it is a
# misread year. Roughly a quarter: an invoice dated next week is a clerical
# oddity worth a glance, one dated in two years cannot be a real issue date.
IMPOSSIBLE_FUTURE_DAYS = 120


@rule("*", severity="blocking")
def dates_not_in_future(context: RuleContext) -> RuleFinding | None:
    """Issue dates in the future usually mean a misread year.

    Graded rather than flat, and the grading is what makes the rule useful. A
    date a few days ahead is a clock, a timezone or a supplier billing in
    advance - worth mentioning, not worth stopping. A date two years ahead is
    not a date at all: it is a digit that was misread, and the rest of the page
    was read by the same engine under the same conditions.

    Found by an actual silent error. A faded photocopy read `2026-07-17` as
    `2028-07-17` and `PRF-SG-20416` as `PRF.SG20496`, and posted automatically
    at 85% confidence with both values wrong. This rule was the only one that
    fired and it fired as a warning, so nothing stopped. Grounding could not
    help: it confirms a quote exists in the OCR text, and the OCR text was what
    was wrong.
    """
    issue_fields = {"invoice_date", "receipt_date", "po_date", "submitted_date", "document_date"}
    offenders: list[tuple[str, str, int]] = []
    for name, value in _date_fields(context):
        if name not in issue_fields:
            continue
        try:
            parsed = date.fromisoformat(value)
        except (TypeError, ValueError):
            continue
        ahead = (parsed - context.today).days
        if ahead > context.future_date_tolerance_days:
            offenders.append((name, value, ahead))

    if not offenders:
        return passed("dates_not_in_future", "Document dates are not in the future.")

    worst = max(ahead for _, _, ahead in offenders)
    detail = ", ".join(f"{name}={value}" for name, value, _ in offenders)
    if worst > IMPOSSIBLE_FUTURE_DAYS:
        return failed(
            "dates_not_in_future",
            f"Date is {worst // 30} months in the future ({detail}). That is not a document "
            f"issued early, it is a misread year - and the rest of this page was read by the "
            f"same engine.",
            severity="blocking",
            fields=[name for name, _, _ in offenders],
            days_ahead=worst,
        )
    return failed(
        "dates_not_in_future",
        f"Date is {worst} day(s) in the future ({detail}); the year may have been mis-read.",
        severity="warning",
        fields=[name for name, _, _ in offenders],
        days_ahead=worst,
    )
