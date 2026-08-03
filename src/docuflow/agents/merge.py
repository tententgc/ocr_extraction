"""Arbitrate between the rule and LLM extraction channels.

Two independent channels reading the same page is the cheapest reliable accuracy
signal available: when they agree the value is almost always right, and when they
disagree that is precisely the document a human should look at.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from docuflow.agents.grounding import find_evidence, value_supported_by_evidence
from docuflow.agents.llm_extractor import LlmCandidate
from docuflow.agents.rule_extractor import RuleCandidate
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


def _normalize(raw: str | None, spec: FieldSpec) -> _Normalized | None:
    if raw is None:
        return None
    value, note = normalize_field_value(raw, spec)
    if value is None:
        return _Normalized(value=None, note=note, raw=str(raw))
    return _Normalized(value=value, note=note, raw=str(raw))


def _values_agree(left: Any, right: Any) -> bool:
    """Type-aware equality between the two channels' normalised values."""
    if left is None or right is None:
        return False
    if isinstance(left, float) and isinstance(right, float):
        return abs(left - right) < 0.005
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) < 0.005
    return normalize_for_match(str(left)) == normalize_for_match(str(right))


def merge_field(
    spec: FieldSpec,
    rule: RuleCandidate | None,
    llm: LlmCandidate | None,
    ocr_pages: list[tuple[int, str]],
    ocr_quality: float,
    ocr_degraded: bool,
) -> ExtractedField:
    """Combine both channels into one field with an honest confidence."""
    notes: list[str] = []

    rule_norm = _normalize(rule.raw_value if rule else None, spec)
    llm_norm = _normalize(llm.raw_value if llm else None, spec)

    # --- ground the LLM claim -------------------------------------------
    grounded = False
    grounding_page: int | None = None
    if llm is not None:
        result = find_evidence(llm.evidence or "", ocr_pages)
        grounded = result.grounded
        grounding_page = result.page
        if not grounded:
            notes.append(f"evidence not verified: {result.reason}")
        elif not value_supported_by_evidence(llm.raw_value, llm.evidence):
            grounded = False
            notes.append("the quoted text does not contain the reported value")

    has_rule = rule_norm is not None and rule_norm.value is not None
    has_llm = llm_norm is not None and llm_norm.value is not None

    if rule_norm is not None and rule_norm.value is None and rule_norm.note:
        notes.append(f"rule channel: {rule_norm.note}")
    if llm_norm is not None and llm_norm.value is None and llm_norm.note:
        notes.append(f"model channel: {llm_norm.note}")

    if not has_rule and not has_llm:
        return ExtractedField(
            name=spec.name,
            label=spec.label,
            value=None,
            confidence=0.0,
            grounded=False,
            source="llm" if llm is not None else "rule",
            notes=tuple(notes),
        )

    # --- choose the value ------------------------------------------------
    agree = has_rule and has_llm and _values_agree(rule_norm.value, llm_norm.value)

    if has_rule and has_llm and not agree:
        notes.append(
            f"channels disagree - document reader read {llm_norm.value!r}, "
            f"label matcher read {rule_norm.value!r}"
        )
        # The grounded reading wins; otherwise trust the deterministic matcher.
        if grounded:
            chosen, source, raw, evidence, page = llm_norm, "llm", llm_norm.raw, llm.evidence, grounding_page
        else:
            chosen, source, raw, evidence, page = rule_norm, "rule", rule_norm.raw, rule.evidence, rule.page
    elif has_llm:
        chosen, source, raw, evidence = llm_norm, "llm", llm_norm.raw, llm.evidence
        page = grounding_page if grounding_page is not None else (rule.page if rule else None)
        if has_rule:
            # Agreement: keep the rule line as evidence when the model gave none.
            evidence = evidence or rule.evidence
            page = page if page is not None else rule.page
    else:
        chosen, source, raw, evidence, page = rule_norm, "rule", rule_norm.raw, rule.evidence, rule.page
        grounded = True  # a rule candidate is a literal span of the page text

    if agree:
        grounded = True

    if chosen.note:
        notes.append(chosen.note)

    # A value taken from page position rather than a label is weaker evidence.
    positional = source == "rule" and rule is not None and rule.heuristic
    if positional:
        notes.append("no explicit label on the document; taken from the letterhead")

    confidence = score_field(
        both_channels_agree=agree,
        has_llm=has_llm,
        has_rule=has_rule,
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
        raw_value=raw,
        evidence=evidence,
        page=page,
        confidence=confidence,
        grounded=grounded,
        source=source,
        notes=tuple(notes),
    )


def merge_all(
    schema: DocumentSchema,
    rule_candidates: dict[str, RuleCandidate],
    llm_candidates: dict[str, LlmCandidate],
    ocr_pages: list[tuple[int, str]],
    ocr_quality: float,
    ocr_degraded: bool,
) -> tuple[ExtractedField, ...]:
    """Merge every field in the schema."""
    return tuple(
        merge_field(
            spec,
            rule_candidates.get(spec.name),
            llm_candidates.get(spec.name),
            ocr_pages,
            ocr_quality,
            ocr_degraded,
        )
        for spec in schema.fields
    )
