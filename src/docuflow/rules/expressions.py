"""A small, total expression language for user-authored rules.

The point of this module is what it *cannot* do. A user describes a check in
their own words and a model turns it into one of these expressions - once, at
authoring time, with the result shown to them before it is saved. From then on
the check is evaluated here, by code that only compares values.

There is no `eval`, no attribute access, no arbitrary regex and no model call at
validation time. That is deliberate: a model asked "is this VAT correct?" on
every document is a model that can be argued out of the right answer by the
document it is reading. Compiling the intent once and executing it
deterministically keeps the check answering to the person who wrote it.

An expression is one of:

    {"all": [expr, ...]}                       every branch holds
    {"any": [expr, ...]}                       at least one branch holds
    {"not": expr}                              the branch does not hold
    {"field": "total_amount", "op": "gt", "value": 50000}
    {"field": "due_date", "op": "gte", "other_field": "invoice_date"}
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

MAX_DEPTH = 8
MAX_BRANCHES = 12

COMPARISONS = frozenset({"eq", "ne", "gt", "gte", "lt", "lte"})
MEMBERSHIPS = frozenset({"in", "not_in"})
TEXT_OPS = frozenset({"contains", "not_contains", "starts_with", "ends_with"})
PRESENCE = frozenset({"present", "absent"})
OPERATORS = COMPARISONS | MEMBERSHIPS | TEXT_OPS | PRESENCE


class ExpressionError(ValueError):
    """The expression is not something this language can represent."""


@dataclass(frozen=True)
class Evaluation:
    """Outcome of one expression against one document."""

    holds: bool
    # Values actually compared, so a failure message can quote them.
    observed: dict[str, Any]


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").strip())
        except ValueError:
            return None
    return None


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


def _ordered(left: Any, right: Any) -> tuple[float, float] | None:
    """Both sides as comparable numbers, or None if they are not comparable.

    Dates are ordered as dates rather than as text, so "2026-1-5" and
    "2026-01-05" do not compare as different values.
    """
    left_date, right_date = _as_date(left), _as_date(right)
    if left_date is not None and right_date is not None:
        return left_date.toordinal(), right_date.toordinal()
    left_number, right_number = _as_number(left), _as_number(right)
    if left_number is not None and right_number is not None:
        return left_number, right_number
    return None


def _equal(left: Any, right: Any) -> bool:
    ordered = _ordered(left, right)
    if ordered is not None:
        return abs(ordered[0] - ordered[1]) < 1e-9
    return str(left).strip().casefold() == str(right).strip().casefold()


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip().casefold()


def _compare(op: str, left: Any, right: Any) -> bool:
    """Apply one operator. Never raises: an incomparable pair is simply false."""
    if op == "present":
        return left is not None and _text(left) != ""
    if op == "absent":
        return left is None or _text(left) == ""

    if op == "eq":
        return _equal(left, right)
    if op == "ne":
        return not _equal(left, right)

    if op in MEMBERSHIPS:
        options = right if isinstance(right, (list, tuple)) else [right]
        found = any(_equal(left, option) for option in options)
        return found if op == "in" else not found

    if op in TEXT_OPS:
        haystack, needle = _text(left), _text(right)
        if op == "contains":
            return needle in haystack
        if op == "not_contains":
            return needle not in haystack
        if op == "starts_with":
            return haystack.startswith(needle)
        return haystack.endswith(needle)

    ordered = _ordered(left, right)
    if ordered is None:
        # Two values that cannot be ordered do not satisfy an ordering test.
        return False
    first, second = ordered
    if op == "gt":
        return first > second
    if op == "gte":
        return first >= second
    if op == "lt":
        return first < second
    return first <= second


def validate(expression: Any, known_fields: frozenset[str], depth: int = 0) -> None:
    """Reject anything the evaluator could not run. Raises `ExpressionError`."""
    if depth > MAX_DEPTH:
        raise ExpressionError(f"The condition nests deeper than {MAX_DEPTH} levels.")
    if not isinstance(expression, dict):
        raise ExpressionError("Each condition must be an object.")

    for key in ("all", "any"):
        if key in expression:
            branches = expression[key]
            if not isinstance(branches, list) or not branches:
                raise ExpressionError(f"'{key}' needs a non-empty list of conditions.")
            if len(branches) > MAX_BRANCHES:
                raise ExpressionError(f"'{key}' has more than {MAX_BRANCHES} conditions.")
            for branch in branches:
                validate(branch, known_fields, depth + 1)
            return

    if "not" in expression:
        validate(expression["not"], known_fields, depth + 1)
        return

    field = expression.get("field")
    if not isinstance(field, str) or not field:
        raise ExpressionError("A condition needs a 'field'.")
    if field not in known_fields:
        raise ExpressionError(
            f"'{field}' is not a field on this document type. "
            f"Available: {', '.join(sorted(known_fields))}."
        )

    op = expression.get("op")
    if op not in OPERATORS:
        raise ExpressionError(
            f"'{op}' is not a supported comparison. Use one of: {', '.join(sorted(OPERATORS))}."
        )

    other = expression.get("other_field")
    if other is not None:
        if op in PRESENCE:
            raise ExpressionError(f"'{op}' compares nothing, so it takes no other field.")
        if other not in known_fields:
            raise ExpressionError(f"'{other}' is not a field on this document type.")
    elif op not in PRESENCE and "value" not in expression:
        raise ExpressionError(f"'{op}' needs a 'value' or an 'other_field' to compare against.")


def evaluate(expression: dict[str, Any], values: dict[str, Any]) -> Evaluation:
    """Run a validated expression against one document's values."""
    observed: dict[str, Any] = {}
    holds = _evaluate(expression, values, observed)
    return Evaluation(holds=holds, observed=observed)


def _evaluate(expression: dict[str, Any], values: dict[str, Any], observed: dict[str, Any]) -> bool:
    if "all" in expression:
        # Evaluated in full rather than short-circuited, so the failure message
        # can quote every value the user's condition mentioned.
        return all([_evaluate(branch, values, observed) for branch in expression["all"]])
    if "any" in expression:
        return any([_evaluate(branch, values, observed) for branch in expression["any"]])
    if "not" in expression:
        return not _evaluate(expression["not"], values, observed)

    field = expression["field"]
    left = values.get(field)
    observed[field] = left

    if "other_field" in expression and expression["other_field"] is not None:
        other = expression["other_field"]
        right = values.get(other)
        observed[other] = right
    else:
        right = expression.get("value")

    return _compare(expression["op"], left, right)


def describe(expression: dict[str, Any]) -> str:
    """Read an expression back as a sentence, so a user can check it."""
    if "all" in expression:
        return " and ".join(describe(branch) for branch in expression["all"])
    if "any" in expression:
        return " or ".join(describe(branch) for branch in expression["any"])
    if "not" in expression:
        return f"not ({describe(expression['not'])})"

    field = expression["field"]
    op = expression["op"]
    if op in PRESENCE:
        return f"{field} is {'present' if op == 'present' else 'missing'}"

    words = {
        "eq": "is", "ne": "is not", "gt": "is more than", "gte": "is at least",
        "lt": "is less than", "lte": "is at most", "in": "is one of",
        "not_in": "is not one of", "contains": "contains",
        "not_contains": "does not contain", "starts_with": "starts with",
        "ends_with": "ends with",
    }
    if expression.get("other_field"):
        return f"{field} {words[op]} {expression['other_field']}"
    return f"{field} {words[op]} {expression.get('value')!r}"
