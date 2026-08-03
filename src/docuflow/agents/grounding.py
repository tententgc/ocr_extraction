"""Evidence grounding - the main defence against hallucinated values.

Every LLM-extracted field must come with a verbatim quote from the page. This
module checks that the quote really exists in the OCR text, and that the value
itself is traceable to it. A field that fails grounding is not deleted, but it is
marked and can never be auto-posted.
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from docuflow.normalize.text import normalize_for_match, squash

# A quote this similar to page text is considered present despite OCR noise.
FUZZY_MATCH_THRESHOLD = 88.0
# Quotes shorter than this are too generic to prove anything.
MIN_EVIDENCE_CHARS = 3


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


def value_supported_by_evidence(raw_value: str | None, evidence: str | None) -> bool:
    """Check the extracted value actually appears inside its own quote.

    Catches the failure where a model quotes a real line but reports a number
    that is not in it - the most dangerous kind of error, because the quote looks
    convincing to a reviewer skimming the page.
    """
    if not raw_value or not evidence:
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
