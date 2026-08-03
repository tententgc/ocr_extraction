"""Business rule registry and runner.

Rules are plain functions registered against the document types they apply to.
A rule returns a `RuleFinding` (passed or failed) or `None` when it does not
apply to the document in front of it.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable

from docuflow.models import RuleFinding, Severity
from docuflow.rules.context import RuleContext

logger = logging.getLogger(__name__)

RuleFunction = Callable[[RuleContext], RuleFinding | None]

# doc_type (or "*") -> rules
_REGISTRY: dict[str, list[RuleFunction]] = {}


def rule(*doc_types: str) -> Callable[[RuleFunction], RuleFunction]:
    """Register a rule for the given document types (`"*"` means all)."""

    def decorator(function: RuleFunction) -> RuleFunction:
        for doc_type in doc_types or ("*",):
            _REGISTRY.setdefault(doc_type, []).append(function)
        return function

    return decorator


def rules_for(doc_type: str) -> list[RuleFunction]:
    """All rules applicable to a document type, generic ones first."""
    return [*_REGISTRY.get("*", []), *_REGISTRY.get(doc_type, [])]


def run_rules(context: RuleContext) -> tuple[RuleFinding, ...]:
    """Execute every applicable rule, isolating failures."""
    findings: list[RuleFinding] = []
    for function in rules_for(context.doc_type):
        try:
            finding = function(context)
        except Exception as exc:  # noqa: BLE001 - a broken rule must not stop validation
            logger.exception("rule %s raised", getattr(function, "__name__", "?"))
            findings.append(
                RuleFinding(
                    rule_id=getattr(function, "__name__", "unknown_rule"),
                    passed=False,
                    severity="warning",
                    message=f"Rule could not be evaluated: {exc}",
                )
            )
            continue
        if finding is not None:
            findings.append(finding)
    return tuple(findings)


def passed(rule_id: str, message: str, severity: Severity = "info", **detail) -> RuleFinding:
    return RuleFinding(rule_id=rule_id, passed=True, severity=severity, message=message, detail=detail)


def failed(
    rule_id: str,
    message: str,
    severity: Severity = "warning",
    fields: Iterable[str] = (),
    **detail,
) -> RuleFinding:
    return RuleFinding(
        rule_id=rule_id,
        passed=False,
        severity=severity,
        message=message,
        fields=tuple(fields),
        detail=detail,
    )


def load_all_rules() -> None:
    """Import the rule modules so their decorators run."""
    from docuflow.rules import common, contract, financial, form  # noqa: F401
