"""Evidence grounding - the main defence against hallucinated values.

Every LLM-extracted field must come with a verbatim quote from the page. This
module checks that the quote really exists in the OCR text, and that the value
itself is traceable to it. A field that fails grounding is not deleted, but it is
marked and can never be auto-posted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from rapidfuzz import fuzz

from docuflow.normalize.amounts import parse_number
from docuflow.normalize.dates import parse_date
from docuflow.normalize.text import normalize_for_match, squash

# A quote this similar to page text is considered present despite OCR noise.
FUZZY_MATCH_THRESHOLD = 88.0
# Quotes shorter than this are too generic to prove anything.
MIN_EVIDENCE_CHARS = 3

# Anything that might be a written date: 11/02/2019, 2019-02-11, 8 July 2026,
# ๘ กรกฎาคม ๒๕๖๙. Deliberately loose - `parse_date` decides what is real.
_DATE_TOKEN_RE = re.compile(
    r"\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}"
    r"|[\d๐-๙]{1,2}\s+[^\W\d_]{3,12}\.?\s+[\d๐-๙]{2,4}"
    r"|[^\W\d_]{3,12}\.?\s+[\d๐-๙]{1,2},?\s+[\d๐-๙]{2,4}",
    re.UNICODE,
)
_NUMBER_TOKEN_RE = re.compile(r"[-+]?[\d๐-๙][\d๐-๙,.']*")


@dataclass(frozen=True)
class GroundingResult:
    grounded: bool
    score: float
    page: int | None
    reason: str = ""


def _page_texts(ocr_pages: list[tuple[int, str]]) -> list[tuple[int, str, str]]:
    """Pre-normalise each page once: `(page_number, normalized, squashed)`."""
    return [(number, normalize_for_match(text), squash(text)) for number, text in ocr_pages]


def find_evidence(evidence: str, ocr_pages: list[tuple[int, str]]) -> GroundingResult:
    """Locate a quoted span in the recognised text."""
    if not evidence or len(evidence.strip()) < MIN_EVIDENCE_CHARS:
        return GroundingResult(False, 0.0, None, "evidence quote missing or too short")

    needle_squashed = squash(evidence)
    needle_normalized = normalize_for_match(evidence)
    if not needle_squashed:
        return GroundingResult(False, 0.0, None, "evidence quote has no matchable content")

    pages = _page_texts(ocr_pages)

    # Exact (normalised) containment is the common, cheap case.
    for number, _, page_squashed in pages:
        if needle_squashed in page_squashed:
            return GroundingResult(True, 1.0, number, "")

    # Fall back to fuzzy matching so a single mis-read character does not sink an
    # otherwise correct quote.
    best_score, best_page = 0.0, None
    for number, page_normalized, _ in pages:
        if not page_normalized:
            continue
        score = fuzz.partial_ratio(needle_normalized, page_normalized)
        if score > best_score:
            best_score, best_page = score, number

    if best_score >= FUZZY_MATCH_THRESHOLD:
        return GroundingResult(True, round(best_score / 100.0, 3), best_page, "")
    return GroundingResult(
        False,
        round(best_score / 100.0, 3),
        None,
        f"quoted text was not found on the page (best match {best_score:.0f}%)",
    )


def _dates_in(text: str) -> list[date]:
    """Every date the page text could be stating, however it is written."""
    return [parsed for token in _DATE_TOKEN_RE.findall(text) if (parsed := parse_date(token))]


def _date_supported(value: Any, evidence: str) -> bool | None:
    """Whether `evidence` states this date. None when it states no date at all."""
    wanted = parse_date(value) if not isinstance(value, date) else value
    if wanted is None:
        return None
    found = _dates_in(evidence)
    return (wanted in found) if found else None


def _amount_supported(value: Any, evidence: str) -> bool | None:
    """Whether `evidence` states this amount. None when it states no number."""
    wanted = parse_number(value)
    if wanted is None:
        return None
    found = [parsed for token in _NUMBER_TOKEN_RE.findall(evidence) if (parsed := parse_number(token)) is not None]
    if not found:
        return None
    return any(abs(candidate - wanted) < 0.005 for candidate in found)


def value_supported_by_evidence(
    raw_value: str | None,
    evidence: str | None,
    field_type: str | None = None,
    value: Any = None,
) -> bool:
    """Check the extracted value is really stated by its own quote.

    Catches the failure where a model quotes a real line but reports a number
    that is not in it - the most dangerous kind of error, because the quote looks
    convincing to a reviewer skimming the page.

    Dates and amounts are compared as dates and amounts, not as text. A page
    saying "INVOICE DATE 11/02/2019" does state the date 2019-02-11, and a
    literal substring test calls that a hallucination - which is both wrong and
    the most common thing this check used to report. `value` is the normalised
    reading; `raw_value` is what the model actually wrote down.
    """
    if not evidence:
        return False

    if field_type == "enum":
        # An enum value is a classification, not a transcription. The model
        # picks from a list the schema fixes, so it cannot invent one, and the
        # page will not contain the label: a contract that "shall automatically
        # renew" supports auto_renew=yes without the word "yes" appearing
        # anywhere. Requiring the literal token here is a category error, and it
        # was marking correct readings unverified at 30% confidence.
        return True

    if field_type == "date":
        verdict = _date_supported(value if value is not None else raw_value, evidence)
        if verdict is not None:
            return verdict
    elif field_type in {"money", "number", "percent"}:
        verdict = _amount_supported(value if value is not None else raw_value, evidence)
        if verdict is not None:
            return verdict

    if not raw_value:
        return False
    value_squashed = squash(str(raw_value))
    if not value_squashed:
        return False
    evidence_squashed = squash(evidence)
    if value_squashed in evidence_squashed:
        return True
    # Amounts print with separators the model may drop, and vice versa.
    digits_only_value = "".join(char for char in value_squashed if char.isdigit())
    digits_only_evidence = "".join(char for char in evidence_squashed if char.isdigit())
    if digits_only_value and digits_only_value in digits_only_evidence:
        return True
    return fuzz.partial_ratio(value_squashed, evidence_squashed) >= 92.0
