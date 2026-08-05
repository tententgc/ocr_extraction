"""Routing policy: post automatically, send to a human, or refuse.

The policy is deliberately conservative. Straight-through processing requires
*all* of: no blocking rule failure, confidence above the threshold, every
required field present, and every critical field grounded in the page text.
Anything less goes to a person, because a wrong value costs more than a review.
"""

from __future__ import annotations

from dataclasses import dataclass

from docuflow.models import Decision, DocumentRun
from docuflow.schemas.registry import DocumentSchema


@dataclass(frozen=True)
class RoutingOutcome:
    decision: Decision
    reasons: tuple[str, ...]

    @property
    def is_auto(self) -> bool:
        return self.decision == "auto_post"


def decide(
    run: DocumentRun,
    schema: DocumentSchema,
    min_confidence: float,
    *,
    rules_evaluated: bool = True,
) -> RoutingOutcome:
    """Choose what happens to this document.

    `rules_evaluated=False` still routes on extraction quality - completeness,
    grounding and confidence are properties of the reading, not of the business
    checks - but says so in the reason, because "no blocking issues" would
    otherwise claim a validation that never ran.
    """
    reasons: list[str] = []

    # --- unprocessable ----------------------------------------------------
    if run.error:
        return RoutingOutcome("rejected", (f"Processing failed: {run.error}",))
    if not run.ocr.pages or run.ocr.char_count < 20:
        return RoutingOutcome("rejected", ("No readable text could be extracted from this file.",))

    populated = [field for field in run.fields if not field.is_empty]
    if not populated:
        return RoutingOutcome("rejected", ("No fields could be extracted from the document.",))

    # --- blocking rule failures -------------------------------------------
    blocking = run.blocking_findings
    if blocking:
        reasons.extend(finding.message for finding in blocking)

    # --- completeness and grounding ---------------------------------------
    missing_required = [
        spec.label for spec in schema.required_fields if run.value(spec.name) is None
    ]
    if missing_required:
        reasons.append(f"Required field(s) not found: {', '.join(missing_required)}.")

    ungrounded_critical = [
        field.label or field.name
        for field in run.fields
        if not field.is_empty and field.name in {spec.name for spec in schema.critical_fields} and not field.grounded
    ]
    if ungrounded_critical:
        reasons.append(
            f"Could not verify these values against the page text: {', '.join(ungrounded_critical)}."
        )

    # --- confidence --------------------------------------------------------
    if run.confidence < min_confidence:
        reasons.append(
            f"Overall confidence {run.confidence:.0%} is below the {min_confidence:.0%} threshold for automatic posting."
        )

    if reasons:
        return RoutingOutcome("needs_review", tuple(reasons))

    # What was actually checked, and no more.
    #
    # This used to read "all required and critical fields are present and
    # verified", which is not what the two checks above do and mis-taught
    # everyone who read it. `critical` is a statement about *consequence* - if
    # this value is present and wrong, somebody is paid the wrong amount - not a
    # demand that it be there. Plenty of legitimate invoices quote no purchase
    # order at all, and whether one is *needed* is a business policy question
    # answered by `purchase_order_required` against `po_required_above`, not by
    # a field's severity.
    verified = _verified_summary(run, schema)
    if not rules_evaluated:
        return RoutingOutcome(
            "auto_post",
            (
                f"Confidence {run.confidence:.0%}; {verified}. Business rules were "
                f"disabled, so nothing was checked against master data or business policy.",
            ),
        )

    return RoutingOutcome(
        "auto_post",
        (f"Confidence {run.confidence:.0%} with no blocking issues; {verified}",),
    )


def _verified_summary(run: DocumentRun, schema: DocumentSchema) -> str:
    """Which fields were checked, counted rather than claimed."""
    critical_names = {spec.name for spec in schema.critical_fields}
    found = [field for field in run.fields if not field.is_empty]
    critical_found = [field for field in found if field.name in critical_names]
    absent = len(critical_names) - len(critical_found)

    summary = (
        f"every required field was found, and {len(critical_found)} critical "
        f"value(s) trace to text on the page"
    )
    # Said out loud, because a reviewer scanning a posted record deserves to
    # know which of the values that matter simply were not on the document.
    if absent:
        missing = sorted(critical_names - {field.name for field in critical_found})
        summary += f" ({absent} not printed on this document: {', '.join(missing)})"
    return summary + "."


def explain(outcome: RoutingOutcome) -> str:
    """One-line summary for the UI and CLI."""
    labels = {
        "auto_post": "Posted automatically",
        "needs_review": "Needs human review",
        "rejected": "Rejected",
    }
    return f"{labels[outcome.decision]}: {outcome.reasons[0] if outcome.reasons else ''}"
