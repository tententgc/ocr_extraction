"""Arbitrate between two independent readings of the same page.

Two channels reading the same document is the cheapest reliable accuracy signal
available: when they agree the value is almost always right, and when they
disagree that is precisely the document a human should look at.

The channels are normally two language-model passes — one field-first, one
label-first, preferably on different models. The deterministic label matcher
takes the second slot only when no second model is reachable, so the system
still runs with no credentials at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from docuflow.agents.candidates import Candidate
from docuflow.agents.grounding import find_evidence, value_supported_by_evidence
from docuflow.confidence import score_field
from docuflow.models import ExtractedField
from docuflow.normalize.apply import normalize_field_value
from docuflow.normalize.text import normalize_for_match
from docuflow.schemas.registry import DocumentSchema, FieldSpec


@dataclass(frozen=True)
class _Normalized:
    value: Any
    note: str | None
    raw: str


def _normalize(candidate: Candidate | None, spec: FieldSpec) -> _Normalized | None:
    if candidate is None or candidate.raw_value is None:
        return None
    value, note = normalize_field_value(candidate.raw_value, spec)
    return _Normalized(value=value, note=note, raw=str(candidate.raw_value))


def _values_agree(left: Any, right: Any) -> bool:
    """Type-aware equality between the two channels' normalised values."""
    if left is None or right is None:
        return False
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) < 0.005
    return normalize_for_match(str(left)) == normalize_for_match(str(right))


def _ground(candidate: Candidate | None, ocr_pages: list[tuple[int, str]]) -> tuple[bool, int | None, str]:
    """Check a candidate's quote against the page. Returns (grounded, page, note)."""
    if candidate is None:
        return False, None, ""
    # The label matcher quotes a literal line of the page, so it is grounded by
    # construction; only a model's claim needs verifying.
    if candidate.channel in {"label", "letterhead"}:
        return True, candidate.page, ""

    result = find_evidence(candidate.evidence or "", ocr_pages)
    if not result.grounded:
        return False, None, f"evidence not verified: {result.reason}"
    if not value_supported_by_evidence(candidate.raw_value, candidate.evidence):
        return False, None, "the quoted text does not contain the reported value"
    return True, result.page, ""


def merge_field(
    spec: FieldSpec,
    primary: Candidate | None,
    secondary: Candidate | None,
    ocr_pages: list[tuple[int, str]],
    ocr_quality: float,
    ocr_degraded: bool,
) -> ExtractedField:
    """Combine both readings into one field with an honest confidence."""
    notes: list[str] = []

    primary_norm = _normalize(primary, spec)
    secondary_norm = _normalize(secondary, spec)

    primary_grounded, primary_page, primary_note = _ground(primary, ocr_pages)
    secondary_grounded, secondary_page, secondary_note = _ground(secondary, ocr_pages)

    has_primary = primary_norm is not None and primary_norm.value is not None
    has_secondary = secondary_norm is not None and secondary_norm.value is not None

    for norm, label in ((primary_norm, "first pass"), (secondary_norm, "second pass")):
        if norm is not None and norm.value is None and norm.note:
            notes.append(f"{label}: {norm.note}")

    if not has_primary and not has_secondary:
        return ExtractedField(
            name=spec.name,
            label=spec.label,
            value=None,
            confidence=0.0,
            grounded=False,
            source="llm",
            notes=tuple(notes),
        )

    agree = has_primary and has_secondary and _values_agree(primary_norm.value, secondary_norm.value)

    # --- choose which reading to keep --------------------------------------
    if has_primary and has_secondary and not agree:
        notes.append(
            f"the two readings disagree - first pass read {primary_norm.value!r}, "
            f"second pass read {secondary_norm.value!r}"
        )
        # Prefer whichever one can actually point at the page.
        if primary_grounded or not secondary_grounded:
            chosen, source, evidence, page, grounded = (
                primary_norm, primary, primary.evidence, primary_page, primary_grounded
            )
        else:
            chosen, source, evidence, page, grounded = (
                secondary_norm, secondary, secondary.evidence, secondary_page, secondary_grounded
            )
    elif has_primary:
        chosen, source, evidence, page, grounded = (
            primary_norm, primary, primary.evidence, primary_page, primary_grounded
        )
        if has_secondary and not grounded and secondary_grounded:
            # Agreeing values where only the second pass could quote the page.
            evidence, page, grounded = secondary.evidence, secondary_page, True
    else:
        chosen, source, evidence, page, grounded = (
            secondary_norm, secondary, secondary.evidence, secondary_page, secondary_grounded
        )

    if not grounded:
        notes.append(primary_note or secondary_note or "value could not be verified against the page")

    if chosen.note:
        notes.append(chosen.note)

    positional = source is not None and source.is_positional
    if positional:
        notes.append("no explicit label on the document; taken from the letterhead")

    confidence = score_field(
        both_channels_agree=agree,
        has_primary=has_primary,
        has_secondary=has_secondary,
        grounded=grounded,
        is_critical=spec.critical,
        ocr_quality=ocr_quality,
        ocr_degraded=ocr_degraded,
        had_normalization_note=bool(chosen.note) or positional,
    )

    return ExtractedField(
        name=spec.name,
        label=spec.label,
        value=chosen.value,
        raw_value=chosen.raw,
        evidence=evidence,
        page=page,
        confidence=confidence,
        grounded=grounded,
        source="rule" if source is not None and source.channel in {"label", "letterhead"} else "llm",
        notes=tuple(notes),
    )


def merge_all(
    schema: DocumentSchema,
    primary: dict[str, Candidate],
    secondary: dict[str, Candidate],
    ocr_pages: list[tuple[int, str]],
    ocr_quality: float,
    ocr_degraded: bool,
) -> tuple[ExtractedField, ...]:
    """Merge every field in the schema."""
    return tuple(
        merge_field(
            spec,
            primary.get(spec.name),
            secondary.get(spec.name),
            ocr_pages,
            ocr_quality,
            ocr_degraded,
        )
        for spec in schema.fields
    )
