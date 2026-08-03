"""Value comparison for evaluation.

Field accuracy is measured in the canonical space the downstream system sees, so
"8 July 2026" and "2026-07-08" count as the same answer, while "Mirphol" and
"Mirtphol" do not - a wrong name is wrong however it was produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from docuflow.normalize.text import normalize_for_match
from docuflow.schemas.registry import FieldSpec

# A near-match this close is reported separately: usually an OCR character slip
# rather than a different value.
NEAR_MATCH_THRESHOLD = 90.0
MONEY_EPSILON = 0.01


@dataclass(frozen=True)
class FieldComparison:
    field: str
    expected: Any
    actual: Any
    exact: bool
    near: bool
    missing: bool
    spurious: bool
    similarity: float

    @property
    def correct(self) -> bool:
        return self.exact

    @property
    def status(self) -> str:
        if self.missing:
            return "missing"
        if self.spurious:
            return "spurious"
        if self.exact:
            return "correct"
        return "near" if self.near else "wrong"


def compare_value(expected: Any, actual: Any, spec: FieldSpec | None) -> FieldComparison:
    """Compare one expected/actual pair."""
    name = spec.name if spec else ""
    expected_empty = expected in (None, "", [], {})
    actual_empty = actual in (None, "", [], {})

    if expected_empty and actual_empty:
        return FieldComparison(name, expected, actual, True, True, False, False, 1.0)
    if expected_empty and not actual_empty:
        # The document genuinely has no such value but something was reported.
        return FieldComparison(name, expected, actual, False, False, False, True, 0.0)
    if actual_empty:
        return FieldComparison(name, expected, actual, False, False, True, False, 0.0)

    field_type = spec.type if spec else "string"
    if field_type in {"money", "number", "percent"}:
        try:
            exact = abs(float(expected) - float(actual)) < MONEY_EPSILON
        except (TypeError, ValueError):
            exact = str(expected) == str(actual)
        return FieldComparison(name, expected, actual, exact, exact, False, False, 1.0 if exact else 0.0)

    left, right = normalize_for_match(str(expected)), normalize_for_match(str(actual))
    exact = left == right
    similarity = 100.0 if exact else float(fuzz.ratio(left, right))
    return FieldComparison(
        name, expected, actual, exact, similarity >= NEAR_MATCH_THRESHOLD, False, False, round(similarity / 100, 3)
    )


def compare_document(
    expected: dict[str, Any], actual: dict[str, Any], specs: dict[str, FieldSpec]
) -> list[FieldComparison]:
    """Compare every field named in the ground truth."""
    return [
        compare_value(value, actual.get(name), specs.get(name))
        for name, value in expected.items()
    ]
