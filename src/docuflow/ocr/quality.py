"""Heuristic quality scoring for recognised text.

Vision models fail in two recognisable ways: they return almost nothing, or they
fall into a repetition loop. Both are caught here so the engine can escalate to
another provider instead of feeding garbage into extraction.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"\S+")
_MEANINGFUL_RE = re.compile(r"[\w฀-๿]")

# Below this many characters a page is almost certainly a failed read.
MIN_USEFUL_CHARS = 60
# A page with more than this share of repeated 6-grams is looping.
MAX_REPEAT_RATIO = 0.45
MIN_MEANINGFUL_RATIO = 0.55


@dataclass(frozen=True)
class QualityReport:
    score: float
    acceptable: bool
    reasons: tuple[str, ...] = ()


def _repetition_ratio(text: str, window: int = 6) -> float:
    """Share of repeated word n-grams - high means the model is looping."""
    words = _WORD_RE.findall(text)
    if len(words) < window * 2:
        return 0.0
    grams = [" ".join(words[i : i + window]) for i in range(len(words) - window + 1)]
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def _meaningful_ratio(text: str) -> float:
    """Share of non-whitespace characters that are letters or digits."""
    stripped = [char for char in text if not char.isspace()]
    if not stripped:
        return 0.0
    return sum(1 for char in stripped if _MEANINGFUL_RE.match(char)) / len(stripped)


def score_page_text(text: str) -> QualityReport:
    """Score one page of recognised text in [0, 1]."""
    text = (text or "").strip()
    reasons: list[str] = []

    if len(text) < MIN_USEFUL_CHARS:
        return QualityReport(
            score=round(len(text) / MIN_USEFUL_CHARS * 0.4, 3),
            acceptable=False,
            reasons=(f"only {len(text)} characters recognised",),
        )

    repetition = _repetition_ratio(text)
    meaningful = _meaningful_ratio(text)

    # Length contributes on a log curve: 600+ chars is a full page.
    length_score = min(1.0, math.log10(len(text) / 60 + 1) / math.log10(11))
    score = 0.45 * length_score + 0.30 * (1.0 - min(repetition, 1.0)) + 0.25 * meaningful

    if repetition > MAX_REPEAT_RATIO:
        reasons.append(f"repetitive output ({repetition:.0%} duplicated phrases)")
    if meaningful < MIN_MEANINGFUL_RATIO:
        reasons.append(f"mostly symbols ({meaningful:.0%} readable characters)")

    return QualityReport(score=round(min(score, 1.0), 3), acceptable=not reasons, reasons=tuple(reasons))


def score_document_text(pages: list[str]) -> QualityReport:
    """Aggregate page scores; the worst page drags the document down."""
    if not pages:
        return QualityReport(score=0.0, acceptable=False, reasons=("no pages recognised",))

    reports = [score_page_text(page) for page in pages]
    mean = sum(report.score for report in reports) / len(reports)
    worst = min(report.score for report in reports)
    reasons = tuple(reason for report in reports for reason in report.reasons)
    return QualityReport(
        score=round(0.7 * mean + 0.3 * worst, 3),
        acceptable=all(report.acceptable for report in reports),
        reasons=reasons,
    )
