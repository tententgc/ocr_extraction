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
from docuflow.normalize.dates import detect_day_first, parse_date
from docuflow.normalize.text import normalize_for_match
from docuflow.schemas.registry import DocumentSchema, FieldSpec


@dataclass(frozen=True)
class _Normalized:
    value: Any
    note: str | None
    raw: str


def _normalize(
    candidate: Candidate | None, spec: FieldSpec, day_first: bool | None = None
) -> _Normalized | None:
    if candidate is None or candidate.raw_value is None:
        return None
    value, note = normalize_field_value(candidate.raw_value, spec)

    # A numeric date is ambiguous and each model resolves it on its own: for one
    # page, the first pass returned 2019-11-02 and the second 2019-02-11 from
    # the same "11/02/2019". Neither is a reading of the document - they are two
    # guesses at a convention. The page states that convention, so re-read the
    # quote with it and both channels land on the same day by construction.
    if spec.type == "date" and candidate.evidence:
        from_page = parse_date(candidate.evidence, day_first=day_first if day_first is not None else True)
        if from_page is not None and from_page != value:
            return _Normalized(value=from_page, note=note, raw=str(candidate.raw_value))

    return _Normalized(value=value, note=note, raw=str(candidate.raw_value))


# A single token containing another proves nothing - "INV-001" sits inside
# "INV-0012" and "no" inside "notice". Two or more is where phrasing begins.
MIN_SHARED_TOKENS = 2


def _values_agree(left: Any, right: Any) -> bool:
    """Type-aware equality between the two channels' normalised values."""
    if left is None or right is None:
        return False
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) < 0.005

    first, second = normalize_for_match(str(left)), normalize_for_match(str(right))
    if first == second:
        return True

    # Two readings of a passage differ by where each one started and stopped,
    # not by what they say: "laws of Thailand" against "the laws of Thailand",
    # or "the laws of the State of California" against the same clause carrying
    # "without regard to its conflict of law principles". Calling that a
    # disagreement puts a correct answer in front of a human for no reason.
    #
    # Tested on words rather than characters, so the shorter reading has to be
    # made of the longer one's words - "laws of Cambodia" is still a
    # disagreement with "laws of Thailand", and a truncated identifier is never
    # a match because one token is not a phrase.
    first_tokens, second_tokens = first.split(), second.split()
    smaller, larger = (
        (first_tokens, second_tokens)
        if len(first_tokens) <= len(second_tokens)
        else (second_tokens, first_tokens)
    )
    return len(smaller) >= MIN_SHARED_TOKENS and set(smaller) <= set(larger)


@dataclass(frozen=True)
class _Grounding:
    """Two separate questions about one candidate's evidence."""

    # Is the quoted text actually on the page?
    quote_found: bool
    # Does that quote state the value the channel reported?
    value_supported: bool
    page: int | None = None
    note: str = ""

    @property
    def grounded(self) -> bool:
        return self.quote_found and self.value_supported


def _ground(
    candidate: Candidate | None,
    ocr_pages: list[tuple[int, str]],
    spec: FieldSpec,
    value: Any = None,
) -> _Grounding:
    """Check a candidate's quote against the page."""
    if candidate is None:
        return _Grounding(False, False)
    # The label matcher quotes a literal line of the page, so it is grounded by
    # construction; only a model's claim needs verifying.
    if candidate.channel in {"label", "letterhead"}:
        return _Grounding(True, True, candidate.page)

    result = find_evidence(candidate.evidence or "", ocr_pages)
    if not result.grounded:
        return _Grounding(False, False, None, f"evidence not verified: {result.reason}")
    supported = value_supported_by_evidence(
        candidate.raw_value, candidate.evidence, spec.type, value
    )
    return _Grounding(
        True,
        supported,
        result.page,
        "" if supported else "the quoted text does not state the reported value",
    )


def merge_field(
    spec: FieldSpec,
    primary: Candidate | None,
    secondary: Candidate | None,
    ocr_pages: list[tuple[int, str]],
    ocr_quality: float,
    ocr_degraded: bool,
    day_first: bool | None = None,
) -> ExtractedField:
    """Combine both readings into one field with an honest confidence."""
    notes: list[str] = []

    primary_norm = _normalize(primary, spec, day_first)
    secondary_norm = _normalize(secondary, spec, day_first)

    primary_ground = _ground(primary, ocr_pages, spec, primary_norm.value if primary_norm else None)
    secondary_ground = _ground(
        secondary, ocr_pages, spec, secondary_norm.value if secondary_norm else None
    )
    primary_grounded, primary_page, primary_note = (
        primary_ground.grounded, primary_ground.page, primary_ground.note
    )
    secondary_grounded, secondary_page, secondary_note = (
        secondary_ground.grounded, secondary_ground.page, secondary_ground.note
    )

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

    # Two independent readings arriving at the same value, each quoting a line
    # that is really on the page, corroborate each other. That is stronger
    # evidence than one channel's quote containing the value literally, and it
    # is the whole reason for running two channels - a literal containment test
    # must not be able to overrule it. The quotes still have to be real: this
    # relaxes how the value is tied to the page, never whether the page says it.
    corroborated = (
        agree and primary_ground.quote_found and secondary_ground.quote_found
        and not (primary_grounded and secondary_grounded)
    )
    if corroborated:
        primary_grounded = secondary_grounded = True
        primary_page = primary_page or secondary_page
        notes.append(
            "both readings independently reported this value and each quoted the page"
        )

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
    # Read once for the whole document: a page that writes one date day-first
    # writes them all that way, and one unambiguous date settles every other.
    day_first = detect_day_first("\n".join(text for _, text in ocr_pages))
    return tuple(
        merge_field(
            spec,
            primary.get(spec.name),
            secondary.get(spec.name),
            ocr_pages,
            ocr_quality,
            ocr_degraded,
            day_first,
        )
        for spec in schema.fields
    )
