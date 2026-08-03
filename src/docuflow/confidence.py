"""Field and document confidence scoring.

Confidence here is not a model's self-report - those are badly calibrated.
It is computed from things we can verify: whether two independent extraction
channels agree, whether the value is quoted from text that really exists on the
page, and how good the recognition was.
"""

from __future__ import annotations

from docuflow.models import ExtractedField, RuleFinding

# Base scores by evidential situation, before penalties.
AGREEMENT_BOTH_CHANNELS = 0.97
SINGLE_CHANNEL_GROUNDED = 0.86
CHANNELS_DISAGREE = 0.45
SINGLE_CHANNEL_UNGROUNDED = 0.30
# A value nothing on the page states, filled in from elsewhere. Two strengths,
# because they are not the same kind of claim: a counterparty's billing currency
# is a fact in the register, while "this document is written in Thai, so the
# amounts are baht" is a reasonable inference about the document. Both sit below
# AUTO_POST_MIN_CONFIDENCE, so neither can post a document on its own.
FROM_RECORD = 0.84
FROM_INFERENCE = 0.70

# Multipliers applied on top.
DEGRADED_OCR_FACTOR = 0.90
CRITICAL_UNGROUNDED_FACTOR = 0.60
NORMALIZATION_NOTE_FACTOR = 0.85


def _quality_factor(ocr_quality: float) -> float:
    """Recognition quality caps everything.

    Text we could not read well cannot produce a high-confidence value, whether
    that value was read off the page or inferred from it.
    """
    return 0.55 + 0.45 * max(0.0, min(1.0, ocr_quality))


def score_inferred(*, ocr_quality: float, ocr_degraded: bool, from_record: bool) -> float:
    """Confidence for a value filled in rather than read off the page.

    Still scaled by recognition quality: an inference drawn from a page nobody
    could read is worth less than the same inference from a clean one, and the
    signals it rests on - the script, an address, a tax identifier - all come
    out of that same text.
    """
    score = (FROM_RECORD if from_record else FROM_INFERENCE) * _quality_factor(ocr_quality)
    if ocr_degraded:
        score *= DEGRADED_OCR_FACTOR
    return round(max(0.0, min(1.0, score)), 3)


def score_field(
    *,
    both_channels_agree: bool,
    has_primary: bool,
    has_secondary: bool,
    grounded: bool,
    is_critical: bool,
    ocr_quality: float,
    ocr_degraded: bool,
    had_normalization_note: bool = False,
) -> float:
    """Confidence for one field in [0, 1].

    Two channels reading the page and agreeing is the strongest evidence
    available; one channel reading it alone is materially weaker, whether that
    channel is a model or the label matcher.
    """
    if has_primary and has_secondary:
        base = AGREEMENT_BOTH_CHANNELS if both_channels_agree else CHANNELS_DISAGREE
    elif has_primary or has_secondary:
        base = SINGLE_CHANNEL_GROUNDED if grounded else SINGLE_CHANNEL_UNGROUNDED
    else:
        return 0.0

    score = base * _quality_factor(ocr_quality)

    if ocr_degraded:
        score *= DEGRADED_OCR_FACTOR
    if is_critical and not grounded:
        score *= CRITICAL_UNGROUNDED_FACTOR
    if had_normalization_note:
        score *= NORMALIZATION_NOTE_FACTOR

    return round(max(0.0, min(1.0, score)), 3)


def document_confidence(
    fields: tuple[ExtractedField, ...],
    findings: tuple[RuleFinding, ...],
    required_names: frozenset[str],
    critical_names: frozenset[str],
) -> float:
    """Overall confidence, weighted toward the fields that matter.

    Missing required fields and failed rules pull this down hard - the score is
    meant to answer "is it safe to post this without a human?", not "how well did
    the model do on average?".
    """
    if not fields:
        return 0.0

    populated = [field for field in fields if not field.is_empty]
    if not populated:
        return 0.0

    def weight(field: ExtractedField) -> float:
        if field.name in critical_names:
            return 3.0
        if field.name in required_names:
            return 2.0
        return 1.0

    total_weight = sum(weight(field) for field in populated)
    weighted = sum(field.confidence * weight(field) for field in populated)
    score = weighted / total_weight if total_weight else 0.0

    # Every missing required field is a hole in the record.
    present = {field.name for field in populated}
    missing_required = required_names - present
    score *= max(0.35, 1.0 - 0.25 * len(missing_required))

    # Rule failures are direct evidence something is wrong.
    blocking = sum(1 for finding in findings if finding.is_blocking)
    warnings = sum(1 for finding in findings if not finding.passed and finding.severity == "warning")
    if blocking:
        score = min(score, 0.45) * max(0.5, 1.0 - 0.15 * (blocking - 1))
    elif warnings:
        score *= max(0.6, 1.0 - 0.08 * warnings)

    return round(max(0.0, min(1.0, score)), 3)
