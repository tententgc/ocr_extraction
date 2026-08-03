"""Document type classification: cheap keyword vote, LLM confirmation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from docuflow.agents.prompts import CLASSIFIER_SYSTEM, classification_prompt
from docuflow.llm.base import ChatMessage, LlmProvider, extract_json
from docuflow.normalize.text import normalize_for_match
from docuflow.schemas.registry import DOCUMENT_SCHEMAS, schema_names

logger = logging.getLogger(__name__)

# Keyword agreement this strong makes an LLM call unnecessary.
KEYWORD_CONFIDENT_MARGIN = 3.0


@dataclass(frozen=True)
class Classification:
    doc_type: str
    confidence: float
    method: str
    reason: str = ""


def classify_by_keywords(text: str) -> tuple[str, float, dict[str, float]]:
    """Score each document type by how many of its keywords appear."""
    normalized = normalize_for_match(text)
    scores: dict[str, float] = {}

    for doc_type, schema in DOCUMENT_SCHEMAS.items():
        if doc_type == "unknown":
            continue
        hits = sum(
            1.0 for keyword in schema.keywords if normalize_for_match(keyword) in normalized
        )
        # Field labels are weaker evidence than the distinctive keywords.
        label_hits = sum(
            0.35
            for spec in schema.fields
            for alias in spec.aliases
            if alias and normalize_for_match(alias) in normalized
        )
        scores[doc_type] = hits + min(label_hits, 3.0)

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
