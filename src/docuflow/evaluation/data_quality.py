"""Is the extracted record fit to hand to another system.

Accuracy asks whether a value matches ground truth. This asks something a
production system can actually ask, because production has no ground truth:
**is what came out a well-formed record?**

That distinction is the point of the module. On the corpus you can grade against
the answer sheet; on Monday's post there is no answer sheet, and the only signals
left are the ones below. A quality score that only works where ground truth
exists is a score that stops working exactly where it is needed.

Three questions, in the order they stop being answerable by looking.

**Completeness** - did the fields arrive at all. Reported apart for critical
fields, because `critical` here means *if this is wrong somebody is paid the
wrong amount*, and a blank one is a different problem from a blank description.
An absent field is not automatically a defect: an invoice with no purchase-order
reference is a legitimate invoice, and whether one is *required* is business
policy. So this counts and reports; it does not judge.

**Validity** - is each value the shape its type promises. A date field holding
`14/07/2026` is not a date to anything downstream, an amount holding `"7%"` is
not a number, and both pass through extraction looking fine. This is where the
normalisation layer is measured: every value stored in canonical form, or the
arithmetic rules silently skip and the ledger takes a string.

**Consistency** - do the values agree with each other and with the page.
Grounding says a value was found in text rather than produced from nowhere;
arithmetic says the document agrees with itself. Neither proves a value is
right - an OCR misread is grounded, and two wrong numbers can still add up - but
both catch whole classes of invented value, and they need no answer sheet.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from docuflow.schemas.registry import get_schema

# Canonical forms the rest of the system relies on. A value that reaches the
# ledger in any other shape is one the ledger has to guess about.
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CURRENCY_CODE = re.compile(r"^[A-Z]{3}$")

# Types whose canonical form is a number. `percent` is here because a tax rate
# stored as "7%" reaches `tax_rate_consistent`, which skips in silence rather
# than failing - the quietest way for a check to stop existing.
NUMERIC_TYPES = frozenset({"money", "number", "percent"})


@dataclass(frozen=True)
class QualityIssue:
    """One field that came out in a shape nothing downstream can use."""

    document: str
    field: str
    kind: str
    detail: str

    def to_json(self) -> dict[str, Any]:
        return {
            "document": self.document,
            "field": self.field,
            "kind": self.kind,
            "detail": self.detail,
        }


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or value == []


def _validity_issue(spec, value: Any) -> tuple[str, str] | None:
    """`(kind, detail)` when a value is not the shape its type promises."""
    if spec.type == "date":
        if not (isinstance(value, str) and ISO_DATE.match(value)):
            return "date_not_iso", f"{value!r} is not YYYY-MM-DD"
        return None

    if spec.type in NUMERIC_TYPES:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "not_a_number", f"{value!r} is {type(value).__name__}, not a number"
        if spec.type == "money" and value < 0:
            return "negative_amount", f"{value} is negative"
        if spec.type == "percent" and not 0 <= float(value) <= 100:
            return "percent_out_of_range", f"{value} is outside 0-100"
        return None

    if spec.name == "currency":
        if not (isinstance(value, str) and CURRENCY_CODE.match(value)):
            return "currency_not_a_code", f"{value!r} is not a three-letter code"
        return None

    if spec.choices and isinstance(value, str) and value not in spec.choices:
        return "not_an_allowed_value", f"{value!r} is not one of {', '.join(spec.choices)}"

    return None


def score_document(
    document: str,
    doc_type: str,
    fields: dict[str, Any],
    grounded: dict[str, bool] | None = None,
    findings: Iterable[str] = (),
) -> dict[str, Any]:
    """Completeness, validity and consistency for one extracted record.

    `fields` is the value per field name; `grounded` says which of them were
    traced back to a span of text on the page.
    """
    schema = get_schema(doc_type)
    specs = {spec.name: spec for spec in schema.fields}
    grounded = grounded or {}
    findings = set(findings)

    critical = {spec.name for spec in schema.critical_fields}
    required = {spec.name for spec in schema.fields if spec.required}

    filled = {name for name, spec in specs.items() if not _is_empty(fields.get(name))}
    issues: list[QualityIssue] = []

    for name in sorted(filled):
        problem = _validity_issue(specs[name], fields[name])
        if problem:
            issues.append(QualityIssue(document, name, problem[0], problem[1]))

    # A filled field that no span on the page supports. Not proof of a wrong
    # value - an OCR misread is grounded in the text it misread - but it is the
    # signal that separates "read off the page" from "produced from nowhere".
    ungrounded = sorted(name for name in filled if not grounded.get(name, False))

    # The document agreeing with itself. Independent of any answer sheet, which
    # is what makes it usable in production.
    arithmetic_rules = {
        "invoice_totals_consistent", "line_items_sum_to_subtotal", "tax_rate_consistent",
    }
    broken_arithmetic = sorted(arithmetic_rules & findings)

    return {
        "document": document,
        "doc_type": doc_type,
        "fields_defined": len(specs),
        "fields_filled": len(filled),
        "completeness": round(len(filled) / len(specs), 3) if specs else 0.0,
        "critical_defined": len(critical),
        "critical_filled": len(critical & filled),
        "critical_completeness": (
            round(len(critical & filled) / len(critical), 3) if critical else 1.0
        ),
        "required_missing": sorted(required - filled),
        "invalid": [issue.to_json() for issue in issues],
        "ungrounded": ungrounded,
        "grounded_share": (
            round((len(filled) - len(ungrounded)) / len(filled), 3) if filled else 1.0
        ),
        "broken_arithmetic": broken_arithmetic,
        # One document, one verdict: usable as-is by another system.
        "clean": not issues and not broken_arithmetic and not (required - filled),
    }


def summarize(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Corpus-level data quality.

    Completeness is pooled over fields rather than averaged over documents: a
    receipt with four fields and a contract with fourteen should not count
    equally towards a statement about fields.
    """
    if not scores:
        return {"documents": 0}

    defined = sum(item["fields_defined"] for item in scores)
    filled = sum(item["fields_filled"] for item in scores)
    critical_defined = sum(item["critical_defined"] for item in scores)
    critical_filled = sum(item["critical_filled"] for item in scores)

    kinds: Counter[str] = Counter()
    for item in scores:
        for issue in item["invalid"]:
            kinds[issue["kind"]] += 1

    ungrounded = sum(len(item["ungrounded"]) for item in scores)
    invalid = sum(len(item["invalid"]) for item in scores)
    inconsistent = [item for item in scores if item["broken_arithmetic"]]
    incomplete = [item for item in scores if item["required_missing"]]

    return {
        "documents": len(scores),
        "completeness": round(filled / defined, 3) if defined else 0.0,
        "fields_defined": defined,
        "fields_filled": filled,
        # The half that turns into a payment.
        "critical_completeness": (
            round(critical_filled / critical_defined, 3) if critical_defined else 1.0
        ),
        "critical_defined": critical_defined,
        "critical_filled": critical_filled,
        "documents_missing_a_required_field": len(incomplete),
        "required_missing_documents": [item["document"] for item in incomplete],
        # Values that are not the shape their type promises. Every one of these
        # is a value another system has to guess about.
        "validity": round(1 - invalid / filled, 4) if filled else 1.0,
        "invalid_values": invalid,
        "invalid_by_kind": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
        "grounded_share": round((filled - ungrounded) / filled, 4) if filled else 1.0,
        "ungrounded_values": ungrounded,
        "documents_failing_arithmetic": len(inconsistent),
        "arithmetic_documents": [item["document"] for item in inconsistent],
        "clean_documents": sum(1 for item in scores if item["clean"]),
        "clean_share": round(sum(1 for item in scores if item["clean"]) / len(scores), 3),
        "per_document": scores,
    }
