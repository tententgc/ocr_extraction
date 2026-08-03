"""Business rule registry and runner.

Rules are plain functions registered against the document types they apply to.
A rule returns a `RuleFinding` (passed or failed) or `None` when it does not
apply to the document in front of it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dataclass_field
from typing import Callable, Iterable

from docuflow.models import RuleFinding, Severity
from docuflow.rules.context import RuleContext

logger = logging.getLogger(__name__)

RuleFunction = Callable[[RuleContext], RuleFinding | None]


@dataclass(frozen=True)
class RuleInfo:
    """One registered rule, described well enough to list and switch off.

    A rule's id is what appears in the UI and in `DISABLED_RULES`, and it is not
    always the function name: `totals_consistent` reports as
    `invoice_totals_consistent`, and one function can report under more than one
    id depending on which way it failed. Both are declared here so the catalogue
    the UI reads is the truth rather than a guess at it.
    """

    rule_id: str
    function: RuleFunction
    doc_types: tuple[str, ...]
    summary: str
    # Every id this rule can report under, including `rule_id`.
    emits: frozenset[str] = dataclass_field(default_factory=frozenset)

    @property
    def name(self) -> str:
        return getattr(self.function, "__name__", self.rule_id)

    def matches(self, wanted: str) -> bool:
        """Whether `wanted` names this rule, by id, alias or function name."""
        return wanted == self.rule_id or wanted == self.name or wanted in self.emits


# doc_type (or "*") -> rules
_REGISTRY: dict[str, list[RuleInfo]] = {}

# Emitted in free mode so that "nothing failed" is never confused with
# "nothing was checked". The UI and the stored record both carry it.
FREE_MODE_RULE_ID = "business_rules_disabled"
FREE_MODE_MESSAGE = (
    "Free mode: business rules are disabled, so these values were not checked "
    "against master data, business policy or prior postings."
)


def rule(
    *doc_types: str, emits: tuple[str, ...] | None = None
) -> Callable[[RuleFunction], RuleFunction]:
    """Register a rule for the given document types (`"*"` means all).

    `emits` declares the ids the rule reports under when they are not simply the
    function name; the first is the one shown in the UI.
    """

    def decorator(function: RuleFunction) -> RuleFunction:
        reported = emits or (function.__name__,)
        docstring = (function.__doc__ or "").strip().splitlines()
        info = RuleInfo(
            rule_id=reported[0],
            function=function,
            doc_types=doc_types or ("*",),
            summary=docstring[0] if docstring else "",
            emits=frozenset(reported),
        )
        for doc_type in info.doc_types:
            _REGISTRY.setdefault(doc_type, []).append(info)
        return function

    return decorator


def rules_for(doc_type: str) -> list[RuleInfo]:
    """All rules applicable to a document type, generic ones first."""
    return [*_REGISTRY.get("*", []), *_REGISTRY.get(doc_type, [])]


def all_rules() -> list[RuleInfo]:
    """Every registered rule, each listed once, ordered by id."""
    seen: dict[str, RuleInfo] = {}
    for infos in _REGISTRY.values():
        for info in infos:
            seen.setdefault(info.rule_id, info)
    return sorted(seen.values(), key=lambda info: info.rule_id)


def describe_rules(disabled: Iterable[str] = ()) -> list[dict[str, object]]:
    """The rule catalogue the settings screen lists, with current state."""
    off = frozenset(disabled)
    return [
        {
            "rule_id": info.rule_id,
            "function": info.name,
            "doc_types": list(info.doc_types),
            "applies_to_all": "*" in info.doc_types,
            "summary": info.summary,
            "aliases": sorted(info.emits - {info.rule_id}),
            "enabled": not any(info.matches(name) for name in off),
        }
        for info in all_rules()
    ]


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
    for info in rules_for(context.doc_type):
        if any(info.matches(name) for name in skipped):
            continue
        try:
            finding = info.function(context)
        except Exception as exc:  # noqa: BLE001 - a broken rule must not stop validation
            logger.exception("rule %s raised", info.name)
            findings.append(
                RuleFinding(
                    rule_id=info.rule_id,
                    passed=False,
                    severity="warning",
                    message=f"Rule could not be evaluated: {exc}",
                )
            )
            continue
        # A rule can report under an id it did not declare; the id is what a user
        # reads off the screen, so honour it directly too.
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
