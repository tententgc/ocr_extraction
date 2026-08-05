"""Field and document confidence scoring.

Confidence here is not a model's self-report - those are badly calibrated.
It is computed from things we can verify: whether two independent extraction
channels agree, whether the value is quoted from text that really exists on the
page, and how good the recognition was.
"""

from __future__ import annotations

import math

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


# Rule failures that say the *reading* is probably wrong, rather than that the
# *document* is. The distinction is the one this whole module turns on.
#
# A document that does not add up, or whose check digit fails, or that is dated
# next decade, is telling you a character was misread - and the rest of the page
# came off the same engine under the same conditions. That is a confidence
# signal.
#
# A blocked vendor, an expired contract, an invoice above its purchase order:
# every one of those was read perfectly. They are facts about the world that
# stop the document, and they stop it in `routing.decide`, where stopping
# belongs. Feeding them into confidence as well says "we probably misread this",
# which is both false and unhelpful to the reviewer who then cannot tell a
# document we could not read from one we read fine and may not pay.
MISREAD_SIGNALS = frozenset({
    "invoice_totals_consistent",
    "tax_rate_consistent",
    "line_items_sum_to_subtotal",
    "tax_id_checksum",
    "dates_not_in_future",
    "due_date_after_issue",
    "contract_dates_ordered",
    # The only one here that is about a *name* rather than a number, and the
    # only signal in the system that can catch a character misread inside one.
    # Every rule above works by arithmetic disagreeing with itself; this one
    # works because an authoritative register spells the supplier differently
    # from the page. It earned its place on two Thai documents that posted at
    # 0.843 and 0.840 against a 0.840 bar with `ซ` read as `ช`.
    "vendor_name_matches_register",
})

# How far a misread signal moves the score, by how serious the rule itself says
# it is. The gap between these two is the point.
#
# A blocking one takes a fifth. The document is held by routing either way, so
# this is not what stops it - it is what explains to the reviewer that the
# reading is suspect rather than the supplier.
#
# A warning takes a twentieth. A warning must be able to survive the threshold,
# or it is a blocking rule that has not admitted it: at 5% a page read at 95%
# still clears an 84% bar with two warnings against it, and only starts to be
# held once three of them agree something is off. That is the right shape - one
# advisory finding is noise, three are a pattern.
MISREAD_PENALTY = {"blocking": 0.20, "warning": 0.05, "info": 0.0}
# However many of them, the reading is never scored below this on rule evidence
# alone. Below here the fields themselves have to be weak.
MISREAD_FLOOR = 0.5


def document_confidence(
    fields: tuple[ExtractedField, ...],
    findings: tuple[RuleFinding, ...],
    required_names: frozenset[str],
    critical_names: frozenset[str],
) -> float:
    """How sure we are the values were read correctly. Nothing else.

    Deliberately *not* "is it safe to post this". That question is answered in
    `routing.decide`, which weighs this score alongside the blocking findings,
    the missing required fields and the ungrounded critical ones.

    Keeping them apart matters because it was previously one question, and every
    warning quietly became a block: three advisory findings multiplied a
    perfectly-read document from 95% down to 72%, under an 84% threshold, and it
    went to a person who had nothing to correct. A warning that holds documents
    is a blocking rule that has not admitted it.
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

    # Only the failures that are evidence the *reading* is wrong, and weighted by
    # how serious the rule says they are.
    penalty = sum(
        MISREAD_PENALTY.get(finding.severity, 0.0)
        for finding in findings
        if not finding.passed and finding.rule_id in MISREAD_SIGNALS
    )
    if penalty:
        score *= max(MISREAD_FLOOR, 1.0 - penalty)

    # Rounded **down**, not to nearest, and that is the whole point.
    #
    # This number is stored, displayed, and compared against
    # `auto_post_min_confidence` in `routing.decide`. Rounding to nearest lets
    # a document that scored 0.83951 present itself as 0.840 and clear a 0.840
    # bar it did not actually reach - a presentation decision moving money.
    #
    # Half a thousandth is small, and it is not a hypothetical margin: the
    # 2026-08-06 evaluation posted two documents at 0.840 and 0.843 against a
    # 0.840 threshold, so the entire population near the bar sits inside the
    # rounding error.
    #
    # Flooring makes the displayed value a *lower bound* on the true one, so
    # `displayed >= threshold` implies `true >= threshold`. Rounding can then
    # only ever send a document to a person, never past one, which is the
    # direction this system is required to fail in.
    return math.floor(max(0.0, min(1.0, score)) * 1000) / 1000
