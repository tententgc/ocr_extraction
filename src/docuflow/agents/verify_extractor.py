"""Second, independent extraction pass.

Accuracy here comes from two channels disagreeing, not from either one being
clever. This pass exists to fail *differently* from the primary one, so it:

- reads the page blind, never seeing the primary pass's answers
- is prompted label-first rather than field-first
- runs on a different model whenever one is configured

Same-model, same-prompt self-consistency is nearly worthless: the second run
repeats the first run's mistake and the agreement signal becomes noise.
"""

from __future__ import annotations

import logging

from docuflow.agents.llm_extractor import LlmExtraction, _parse_fields
from docuflow.agents.prompts import VERIFY_SYSTEM, verification_prompt
from docuflow.llm.base import ChatMessage, LlmProvider, extract_json
from docuflow.schemas.registry import DocumentSchema

logger = logging.getLogger(__name__)

MAX_DOCUMENT_CHARS = 24_000
MAX_OUTPUT_TOKENS = 3_000


def verify_with_llm(
    document_text: str, schema: DocumentSchema, llm: LlmProvider | None
) -> LlmExtraction:
    """Re-read the document independently; never raises."""
    if llm is None:
        return LlmExtraction({}, ok=False, note="no verification model configured")
    if not document_text.strip():
        return LlmExtraction({}, ok=False, note="no recognised text to verify against")

    try:
        reply = llm.complete(
            [
                ChatMessage(role="system", content=VERIFY_SYSTEM),
                ChatMessage(
                    role="user",
                    content=verification_prompt(schema, document_text[:MAX_DOCUMENT_CHARS]),
                ),
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 - the run continues on one channel
        logger.warning("verification pass failed: %s", exc)
        return LlmExtraction({}, ok=False, note=f"verification pass failed: {exc}")

    payload = extract_json(reply)
    if payload is None:
        return LlmExtraction({}, ok=False, note="verification reply was not valid JSON")

    candidates = _parse_fields(payload, schema)
    if not candidates:
        return LlmExtraction({}, ok=False, note="verification pass returned no fields")
    return LlmExtraction(candidates, ok=True)
