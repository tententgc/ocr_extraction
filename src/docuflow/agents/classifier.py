"""Document type classification: cheap keyword vote, LLM confirmation."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from docuflow.agents.prompts import CLASSIFIER_SYSTEM, classification_prompt
from docuflow.llm.base import ChatMessage, LlmProvider, extract_json
from docuflow.normalize.text import normalize_for_match
from docuflow.schemas.registry import DOCUMENT_SCHEMAS, DocumentSchema, schema_names

logger = logging.getLogger(__name__)

# Keyword agreement this strong makes an LLM call unnecessary.
KEYWORD_CONFIDENT_MARGIN = 3.0


@dataclass(frozen=True)
class Classification:
    doc_type: str
    confidence: float
    method: str
    reason: str = ""


def _shadowed(phrase: str, matched: frozenset[str]) -> bool:
    """True when a longer phrase that also matched already contains this one.

    English keeps its document vocabularies apart: nothing on a receipt says
    "tax invoice". Thai builds terms by concatenation, and the standard heading
    on a retail receipt is `ใบกำกับภาษีอย่างย่อ` - an *abbreviated* tax invoice,
    which is what the Revenue Department calls the slip a till prints. The
    invoice keyword `ใบกำกับภาษี` sits inside it as a substring, so a plain
    `in normalized` test scores every Thai receipt as an invoice on the strength
    of its own receipt heading. The same thing happens a level down, between
    `เลขที่` and `เลขที่ใบเสร็จ`.

    The rule is the general one and it contains no language knowledge: when two
    matched phrases are nested, only the more specific one carries a vote.
    """
    return any(other != phrase and phrase in other for other in matched)


def _present(phrases: Iterable[str], normalized: str) -> frozenset[str]:
    """The given phrases that appear on the page, in normalised form."""
    return frozenset(
        found
        for phrase in phrases
        if phrase and (found := normalize_for_match(phrase)) and found in normalized
    )


def _label_aliases(schema: DocumentSchema) -> tuple[str, ...]:
    return tuple(alias for spec in schema.fields for alias in spec.aliases if alias)


def classify_by_keywords(text: str) -> tuple[str, float, dict[str, float]]:
    """Score each document type by how many of its keywords appear."""
    normalized = normalize_for_match(text)
    types = [name for name in DOCUMENT_SCHEMAS if name != "unknown"]

    # Matched first, across every type, because whether a phrase is the most
    # specific one on the page cannot be answered from inside one schema.
    keywords = {name: _present(DOCUMENT_SCHEMAS[name].keywords, normalized) for name in types}
    labels = {name: _present(_label_aliases(DOCUMENT_SCHEMAS[name]), normalized) for name in types}
    everything = frozenset().union(*keywords.values(), *labels.values()) if types else frozenset()

    scores: dict[str, float] = {}
    for name in types:
        hits = sum(1.0 for phrase in keywords[name] if not _shadowed(phrase, everything))
        # Field labels are weaker evidence than the distinctive keywords.
        label_hits = sum(0.35 for phrase in labels[name] if not _shadowed(phrase, everything))
        scores[name] = hits + min(label_hits, 3.0)

    if not scores or max(scores.values()) == 0:
        return "unknown", 0.0, scores

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_type, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - runner_up
    confidence = min(0.95, 0.5 + margin / 10.0 + min(best_score, 6.0) / 20.0)
    return best_type, round(confidence, 3), scores


def classify(text: str, llm: LlmProvider | None) -> Classification:
    """Determine the document type."""
    keyword_type, keyword_confidence, scores = classify_by_keywords(text)
    ranked = sorted(scores.values(), reverse=True)
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0] if ranked else 0.0

    # Clear keyword winner: no need to spend a model call.
    if keyword_type != "unknown" and margin >= KEYWORD_CONFIDENT_MARGIN:
        return Classification(keyword_type, keyword_confidence, "keywords", f"keyword margin {margin:.1f}")

    if llm is None:
        return Classification(keyword_type, keyword_confidence, "keywords", "no LLM available")

    try:
        reply = llm.complete(
            [
                ChatMessage(role="system", content=CLASSIFIER_SYSTEM),
                ChatMessage(role="user", content=classification_prompt(text)),
            ],
            max_tokens=200,
            temperature=0.0,
        )
        payload = extract_json(reply)
    except Exception as exc:  # noqa: BLE001 - classification must never break the run
        logger.warning("classifier LLM call failed: %s", exc)
        return Classification(keyword_type, keyword_confidence, "keywords", f"LLM failed: {exc}")

    if not isinstance(payload, dict):
        return Classification(keyword_type, keyword_confidence, "keywords", "LLM returned no JSON")

    llm_type = str(payload.get("doc_type", "")).strip().lower()
    if llm_type not in schema_names():
        return Classification(keyword_type, keyword_confidence, "keywords", f"LLM proposed unknown type '{llm_type}'")

    llm_confidence = float(payload.get("confidence") or 0.7)
    reason = str(payload.get("reason", ""))[:200]

    if llm_type == keyword_type:
        return Classification(llm_type, round(min(0.99, max(llm_confidence, keyword_confidence) + 0.05), 3), "keywords+llm", reason)
    # The two channels disagree: trust the model but record lower confidence.
    return Classification(llm_type, round(min(llm_confidence, 0.8), 3), "llm", f"disagrees with keywords ({keyword_type}); {reason}")
