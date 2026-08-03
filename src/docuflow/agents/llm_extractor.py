"""Schema-driven LLM extraction with mandatory evidence quotes."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from docuflow.agents.prompts import EXTRACTION_SYSTEM, extraction_prompt
from docuflow.llm.base import ChatMessage, LlmProvider, extract_json
from docuflow.schemas.registry import DocumentSchema

logger = logging.getLogger(__name__)

MAX_DOCUMENT_CHARS = 24_000
MAX_OUTPUT_TOKENS = 3_000


@dataclass(frozen=True)
class LlmCandidate:
    field: str
    raw_value: str
    evidence: str | None


@dataclass(frozen=True)
class LlmExtraction:
    candidates: dict[str, LlmCandidate]
    ok: bool
    note: str = ""


def _coerce_scalar(value: object) -> str | None:
    """Flatten the model's value into a string, rejecting empties."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (list, tuple)):
        parts = [part for part in (_coerce_scalar(item) for item in value) if part]
        return ", ".join(parts) or None
    if isinstance(value, dict):
        # Some models nest one level deeper than asked.
        for key in ("value", "text", "amount"):
            if key in value:
                return _coerce_scalar(value[key])
    return None


def _parse_fields(payload: object, schema: DocumentSchema) -> dict[str, LlmCandidate]:
    """Read `{"fields": {name: {value, evidence}}}`, tolerating flatter shapes."""
    if not isinstance(payload, dict):
        return {}

    raw_fields = payload.get("fields")
    if not isinstance(raw_fields, dict):
        # The model may have returned the fields at the top level.
        raw_fields = payload

    known = {spec.name for spec in schema.fields}
    candidates: dict[str, LlmCandidate] = {}

    for name, entry in raw_fields.items():
        if name not in known:
            continue
        if isinstance(entry, dict):
            value = _coerce_scalar(entry.get("value"))
            evidence = _coerce_scalar(entry.get("evidence"))
        else:
            value, evidence = _coerce_scalar(entry), None
        if value is None:
            continue
        candidates[name] = LlmCandidate(field=name, raw_value=value, evidence=evidence)

    return candidates


def extract_with_llm(
    document_text: str, schema: DocumentSchema, llm: LlmProvider | None
) -> LlmExtraction:
    """Run the extraction agent, degrading gracefully on any failure."""
    if llm is None:
        return LlmExtraction({}, ok=False, note="no LLM configured; using rule extraction only")
    if not document_text.strip():
        return LlmExtraction({}, ok=False, note="no recognised text to extract from")

    truncated = document_text[:MAX_DOCUMENT_CHARS]
    try:
        reply = llm.complete(
            [
                ChatMessage(role="system", content=EXTRACTION_SYSTEM),
                ChatMessage(role="user", content=extraction_prompt(schema, truncated)),
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 - fall back to rules rather than fail
        logger.warning("extraction LLM call failed: %s", exc)
        return LlmExtraction({}, ok=False, note=f"LLM extraction failed: {exc}")

    payload = extract_json(reply)
    if payload is None:
        return LlmExtraction({}, ok=False, note="LLM reply was not valid JSON")

    candidates = _parse_fields(payload, schema)
    if not candidates:
        return LlmExtraction({}, ok=False, note="LLM returned no usable fields")
    return LlmExtraction(candidates, ok=True)
