"""User-authored rules: the model, its variables, and how it is evaluated.

A custom rule is data, not code. It names the document types it applies to, an
optional `when` guard deciding whether it is relevant to the document in front
of it, and an `assert_` condition that has to hold for the document to pass.

`when` matters more than it looks. "Invoices over 50,000 need a PO" is not a
check on every invoice - it is a check that applies *only* to invoices over
50,000, and reporting a pass on a 900-baht invoice would bury the real result.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from typing import Any

from docuflow.models import RuleFinding, Severity
from docuflow.rules.context import RuleContext
from docuflow.rules.expressions import ExpressionError, describe, evaluate, validate
from docuflow.schemas.registry import DOCUMENT_SCHEMAS, get_schema

# Facts about the document that are not schema fields. Exposed by name so a rule
# can say "the vendor is not in the master file" without needing code.
DOCUMENT_FACTS: dict[str, str] = {
    "doc_type": "The kind of document (invoice, receipt, contract, form, purchase_order)",
    "vendor_matched": "true when the counterparty was found in the vendor master file",
    "vendor_status": "Status of the matched vendor (active, on_hold), empty when unmatched",
    "line_item_count": "How many line items were read from the document",
    "duplicate_found": "true when this document matches something already posted",
    "ocr_degraded": "true when the page was recognised poorly or by a fallback engine",
}

# Facts that are always true or false. `present` on one of these is always true
# - False is still a value - so a rule written that way can never fail, which is
# the worst outcome available: a check that reports success without checking.
BOOLEAN_FACTS = frozenset({"vendor_matched", "duplicate_found", "ocr_degraded"})

MAX_RULE_ID = 60
MAX_MESSAGE = 300


class CustomRuleError(ValueError):
    """The rule cannot be stored as written."""


def fields_for(doc_type: str) -> frozenset[str]:
    """Every variable a rule for `doc_type` may reference."""
    schema = get_schema(doc_type)
    names = {spec.name for spec in schema.fields} if schema else set()
    return frozenset(names | set(DOCUMENT_FACTS))


def values_for(context: RuleContext) -> dict[str, Any]:
    """Snapshot of one document, keyed by the names a rule may use."""
    values: dict[str, Any] = {
        spec.name: context.value(spec.name) for spec in context.schema.fields
    }
    vendor = context.vendor
    values.update(
        {
            "doc_type": context.doc_type,
            "vendor_matched": context.vendor_match.matched,
            "vendor_status": (vendor or {}).get("status") or "",
            "line_item_count": len(context.line_items),
            "duplicate_found": context.duplicate_of is not None,
            "ocr_degraded": context.ocr_degraded,
        }
    )
    return values


@dataclass(frozen=True)
class CustomRule:
    """One check a user wrote, ready to evaluate."""

    rule_id: str
    title: str
    message: str
    doc_types: tuple[str, ...]
    assert_: dict[str, Any]
    severity: Severity = "warning"
    when: dict[str, Any] | None = None
    enabled: bool = True
    prompt: str = ""
    created_at: str = ""
    fields: tuple[str, ...] = dataclass_field(default_factory=tuple)

    @property
    def explanation(self) -> str:
        """The compiled logic in words, for the user to check before saving."""
        condition = describe(self.assert_)
        if self.when:
            return f"When {describe(self.when)}, require that {condition}."
        return f"Require that {condition}."

    def applies_to(self, doc_type: str) -> bool:
        return "*" in self.doc_types or doc_type in self.doc_types

    def to_json(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "title": self.title,
            "message": self.message,
            "doc_types": list(self.doc_types),
            "assert": self.assert_,
            "severity": self.severity,
            "when": self.when,
            "enabled": self.enabled,
            "prompt": self.prompt,
            "created_at": self.created_at,
            "fields": list(self.fields),
            "explanation": self.explanation,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "CustomRule":
        return cls(
            rule_id=payload["rule_id"],
            title=payload.get("title", payload["rule_id"]),
            message=payload.get("message", ""),
            doc_types=tuple(payload.get("doc_types") or ("*",)),
            assert_=payload.get("assert") or payload.get("assert_") or {},
            severity=payload.get("severity", "warning"),
            when=payload.get("when"),
            enabled=bool(payload.get("enabled", True)),
            prompt=payload.get("prompt", ""),
            created_at=payload.get("created_at", ""),
            fields=tuple(payload.get("fields") or ()),
        )


def reject_meaningless_tests(expression: Any) -> None:
    """Refuse a condition that cannot discriminate between documents."""
    if not isinstance(expression, dict):
        return
    for key in ("all", "any"):
        if key in expression:
            for branch in expression[key]:
                reject_meaningless_tests(branch)
            return
    if "not" in expression:
        reject_meaningless_tests(expression["not"])
        return
    if expression.get("field") in BOOLEAN_FACTS and expression.get("op") in {"present", "absent"}:
        raise CustomRuleError(
            f"'{expression['field']}' is always true or false, so testing whether it is "
            f"present would never fail. Compare it instead: "
            f"{{\"field\": \"{expression['field']}\", \"op\": \"eq\", \"value\": true}}."
        )


def _normalize_id(raw: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in raw.strip().lower())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    if not cleaned:
        raise CustomRuleError("The rule needs an id.")
    return cleaned[:MAX_RULE_ID]


def build_rule(payload: dict[str, Any], *, prompt: str = "") -> CustomRule:
    """Validate a compiled rule and return it, or raise `CustomRuleError`.

    Every reference is checked against the schemas the rule claims to cover, so
    a rule that could never fire - or could only ever raise - is refused here
    rather than discovered on a document months later.
    """
    from docuflow.rules import all_rules

    doc_types = tuple(payload.get("doc_types") or ())
    if not doc_types:
        raise CustomRuleError("The rule must say which document types it applies to.")
    known_types = set(DOCUMENT_SCHEMAS) | {"*"}
    unknown = [name for name in doc_types if name not in known_types]
    if unknown:
        raise CustomRuleError(
            f"Unknown document type(s): {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(DOCUMENT_SCHEMAS))}."
        )

    rule_id = _normalize_id(str(payload.get("rule_id") or payload.get("title") or ""))
    if any(info.matches(rule_id) for info in all_rules()):
        raise CustomRuleError(
            f"'{rule_id}' is the id of a built-in rule. Choose another name."
        )

    severity = payload.get("severity", "warning")
    if severity not in {"blocking", "warning", "info"}:
        raise CustomRuleError(f"'{severity}' is not a severity. Use blocking, warning or info.")

    message = str(payload.get("message") or "").strip()
    if not message:
        raise CustomRuleError("The rule needs a message explaining a failure to a reviewer.")

    assertion = payload.get("assert") or payload.get("assert_")
    if not assertion:
        raise CustomRuleError("The rule needs a condition to check.")

    # A rule spanning several types may only use variables all of them have.
    covered = set(DOCUMENT_SCHEMAS) if "*" in doc_types else set(doc_types)
    shared = frozenset.intersection(*(fields_for(name) for name in covered))

    try:
        validate(assertion, shared)
        if payload.get("when"):
            validate(payload["when"], shared)
    except ExpressionError as exc:
        raise CustomRuleError(str(exc)) from exc

    reject_meaningless_tests(assertion)

    return CustomRule(
        rule_id=rule_id,
        title=str(payload.get("title") or rule_id).strip()[:120],
        message=message[:MAX_MESSAGE],
        doc_types=doc_types,
        assert_=assertion,
        severity=severity,
        when=payload.get("when") or None,
        enabled=bool(payload.get("enabled", True)),
        prompt=prompt or str(payload.get("prompt") or ""),
        created_at=payload.get("created_at") or datetime.now(timezone.utc).isoformat(),
        fields=tuple(payload.get("fields") or ()),
    )


def run_custom_rule(rule: CustomRule, context: RuleContext) -> RuleFinding | None:
    """Evaluate one user rule. Returns None when it does not apply."""
    values = values_for(context)

    if rule.when is not None and not evaluate(rule.when, values).holds:
        return None

    outcome = evaluate(rule.assert_, values)
    quoted = {name: value for name, value in outcome.observed.items() if value is not None}
    detail = {"custom": True, "observed": quoted, "explanation": rule.explanation}

    if outcome.holds:
        return RuleFinding(
            rule_id=rule.rule_id,
            passed=True,
            severity="info",
            message=f"{rule.title}: satisfied.",
            detail=detail,
        )
    return RuleFinding(
        rule_id=rule.rule_id,
        passed=False,
        severity=rule.severity,
        message=rule.message,
        fields=rule.fields,
        detail=detail,
    )
