"""Rules that apply to every document type."""

from __future__ import annotations

from datetime import date

from docuflow.models import RuleFinding
from docuflow.normalize.identifiers import thai_tax_id_checksum_ok
from docuflow.rules.context import RuleContext
from docuflow.rules.engine import failed, passed, rule

THAI_TAX_ID_LENGTH = 13


@rule("*")
def required_fields_present(context: RuleContext) -> RuleFinding | None:
    """Every field the schema marks required must have a value."""
    required = context.schema.required_fields
    if not required:
        return None

    missing = [spec for spec in required if context.value(spec.name) is None]
    if not missing:
        return passed("required_fields_present", f"All {len(required)} required fields were extracted.")
    labels = ", ".join(spec.label for spec in missing)
    return failed(
        "required_fields_present",
        f"Missing required field(s): {labels}.",
        severity="blocking",
        fields=[spec.name for spec in missing],
        missing=[spec.name for spec in missing],
    )


@rule("*")
def critical_fields_grounded(context: RuleContext) -> RuleFinding | None:
    """Money and identity fields must be traceable to text on the page."""
    critical = context.schema.critical_fields
    if not critical:
        return None

    populated = [spec for spec in critical if context.value(spec.name) is not None]
    ungrounded = [spec for spec in populated if not context.is_grounded(spec.name)]
    if not ungrounded:
        return passed(
            "critical_fields_grounded",
            f"All {len(populated)} extracted critical fields are backed by text on the page.",
        )
    labels = ", ".join(spec.label for spec in ungrounded)
    return failed(
        "critical_fields_grounded",
        f"Could not verify against the page text: {labels}. These values may be inferred rather than read.",
        severity="blocking",
        fields=[spec.name for spec in ungrounded],
        ungrounded=[spec.name for spec in ungrounded],
    )


@rule("*")
def low_confidence_fields(context: RuleContext) -> RuleFinding | None:
    """Flag fields the extractor itself is unsure about."""
    threshold = context.field_review_confidence
    weak = [
        item
        for item in context.fields
        if not item.is_empty and item.confidence < threshold
    ]
    if not weak:
        return passed("low_confidence_fields", "No field fell below the review threshold.")
    labels = ", ".join(f"{item.label or item.name} ({item.confidence:.0%})" for item in weak[:5])
    return failed(
        "low_confidence_fields",
        f"{len(weak)} field(s) below the {threshold:.0%} confidence threshold: {labels}.",
        severity="warning",
        fields=[item.name for item in weak],
    )


@rule("*")
def ocr_quality_acceptable(context: RuleContext) -> RuleFinding | None:
    """Warn when the text came from a degraded read."""
    if not context.ocr_degraded:
        return passed("ocr_quality_acceptable", "Document text was recognised cleanly.")
    return failed(
        "ocr_quality_acceptable",
        "Recognition quality was poor or a fallback engine was used; values need checking against the image.",
        severity="warning",
    )


def _check_tax_id(value: str | None, field_name: str) -> RuleFinding | None:
    if not value:
        return None
    digits = "".join(char for char in str(value) if char.isdigit())
    if len(digits) != THAI_TAX_ID_LENGTH:
        # Non-Thai registrations are not checksummable here.
        return passed("tax_id_checksum", "Tax ID is not a 13-digit Thai number; checksum not applicable.")
    if thai_tax_id_checksum_ok(digits):
        return passed("tax_id_checksum", "Thai tax ID checksum is valid.")
    return failed(
        "tax_id_checksum",
        f"Thai tax ID {digits} fails its check-digit test - it was either mis-read or is invalid.",
        severity="blocking",
        fields=[field_name],
        tax_id=digits,
    )


@rule("*")
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


@rule("*")
def dates_not_in_future(context: RuleContext) -> RuleFinding | None:
    """Issue dates in the future usually mean a misread year."""
    issue_fields = {"invoice_date", "receipt_date", "po_date", "submitted_date", "document_date"}
    offenders: list[tuple[str, str]] = []
    for name, value in _date_fields(context):
        if name not in issue_fields:
            continue
        try:
            parsed = date.fromisoformat(value)
        except (TypeError, ValueError):
            continue
        if (parsed - context.today).days > context.future_date_tolerance_days:
            offenders.append((name, value))

    if not offenders:
        return passed("dates_not_in_future", "Document dates are not in the future.")
    detail = ", ".join(f"{name}={value}" for name, value in offenders)
    return failed(
        "dates_not_in_future",
        f"Date is in the future ({detail}); the year may have been mis-read.",
        severity="warning",
        fields=[name for name, _ in offenders],
    )
