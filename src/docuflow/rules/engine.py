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

# Emitted in free mode so that "nothing failed" is never confused with
# "nothing was checked". The UI and the stored record both carry it.
FREE_MODE_RULE_ID = "business_rules_disabled"
FREE_MODE_MESSAGE = (
    "Free mode: business rules are disabled, so these values were not checked "
    "against master data, business policy or prior postings."
)


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


def run_rules(
    context: RuleContext,
    *,
    free_mode: bool = False,
    skip: Iterable[str] = (),
) -> tuple[RuleFinding, ...]:
    """Execute every applicable rule, isolating failures.

    `free_mode` evaluates nothing at all and reports that it did so. `skip`
    silences individual rules by id (as shown in the UI) or by function name,
    which is the safer knob: it turns off the checks that do not apply to the
    documents at hand and leaves the rest doing their job.
    """
    if free_mode:
        return (
            RuleFinding(
                rule_id=FREE_MODE_RULE_ID,
                passed=True,
                severity="info",
                message=FREE_MODE_MESSAGE,
            ),
        )

    skipped = frozenset(skip)
    findings: list[RuleFinding] = []
    for function in rules_for(context.doc_type):
        if getattr(function, "__name__", "") in skipped:
            continue
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
        # Rule ids are not always the function name (`po_exists_and_matches`
        # reports as `po_exists`), and the id is what a user reads off the
        # screen, so it has to be honoured too.
        if finding is not None and finding.rule_id not in skipped:
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
