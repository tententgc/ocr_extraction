"""Tools the repair agent uses to fill in what extraction missed.

The reconciler explains a failure to a human. This is the step before that: an
agent that goes back to the page and to the registers and tries to *fix* the
reading, so a document that is actually fine stops arriving in the queue.

Everything here is read-only. The agent proposes; it never writes. A proposal
carries a page quote and is put through exactly the same grounding and
normalisation as the extraction channels, so the agent has no route to a value
the page does not support - it cannot talk a document into passing. What it can
do is find the value that was there all along.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from docuflow.agents.tools import Tool, _string
from docuflow.models import ExtractedField, RuleFinding
from docuflow.normalize.text import collapse_whitespace, normalize_for_match
from docuflow.schemas.registry import DocumentSchema

# How much page text to hand back for one look, so a reply stays readable.
CONTEXT_CHARS = 320
MAX_HITS = 6


@dataclass
class ProposalBox:
    """Collects what the agent proposes during one run."""

    proposals: list[dict[str, Any]]

    def add(self, field: str, value: str, evidence: str, reason: str) -> dict[str, Any]:
        self.proposals.append(
            {"field": field, "value": value, "evidence": evidence, "reason": reason}
        )
        return {
            "recorded": True,
            "field": field,
            "note": "Proposal recorded. It is checked against the page before it is used - "
                    "a quote that is not on the document is discarded.",
        }


def _find_on_page(ocr_text: str) -> Callable[..., dict[str, Any]]:
    """Locate a label or value and return the surrounding text."""

    def run(args: dict[str, Any]) -> dict[str, Any]:
        wanted = collapse_whitespace(str(args.get("query", "")))
        if not wanted:
            return {"error": "Give something to look for."}

        hits: list[dict[str, Any]] = []
        haystack = normalize_for_match(ocr_text)
        needle = normalize_for_match(wanted)
        if needle:
            start = 0
            while len(hits) < MAX_HITS:
                index = haystack.find(needle, start)
                if index < 0:
                    break
                # Map back roughly onto the original text so the quote a
                # proposal carries is the page's own wording.
                ratio = index / max(len(haystack), 1)
                raw_at = int(ratio * len(ocr_text))
                lo = max(0, raw_at - CONTEXT_CHARS // 2)
                hits.append({"context": collapse_whitespace(ocr_text[lo : lo + CONTEXT_CHARS])})
                start = index + max(len(needle), 1)

        return {
            "query": wanted,
            "found": len(hits),
            "matches": hits,
            "note": "" if hits else "Not on the page. Do not report a value the page does not show.",
        }

    return run


def _lines_near(ocr_text: str) -> Callable[..., dict[str, Any]]:
    """Return the lines around a label - where a value usually sits."""

    def run(args: dict[str, Any]) -> dict[str, Any]:
        label = str(args.get("label", ""))
        wanted = normalize_for_match(label)
        if not wanted:
            return {"error": "Give a label to look near."}

        lines = [line for line in ocr_text.splitlines() if line.strip()]
        out: list[dict[str, Any]] = []
        for index, line in enumerate(lines):
            if wanted in normalize_for_match(line):
                out.append(
                    {
                        "line": collapse_whitespace(line),
                        "next_line": collapse_whitespace(lines[index + 1]) if index + 1 < len(lines) else "",
                    }
                )
            if len(out) >= MAX_HITS:
                break
        return {
            "label": label,
            "found": len(out),
            "lines": out,
            "note": "" if out else "No line carries that label. The document may not state this field.",
        }

    return run


def _missing_fields(schema: DocumentSchema, fields: tuple[ExtractedField, ...]) -> Callable[..., dict[str, Any]]:
    """What the schema wants that is still empty."""

    def run(_args: dict[str, Any]) -> dict[str, Any]:
        have = {field.name for field in fields if not field.is_empty}
        rows = [
            {
                "field": spec.name,
                "label": spec.label,
                "type": spec.type,
                "required": spec.required,
                "critical": spec.critical,
                "labelled_like": list(spec.aliases[:6]),
                "choices": list(spec.choices),
            }
            for spec in schema.fields
            if spec.name not in have
        ]
        return {"missing": rows, "count": len(rows)}

    return run


def _failing_checks(findings: tuple[RuleFinding, ...]) -> Callable[..., dict[str, Any]]:
    """What is currently wrong, and which fields each complaint is about."""

    def run(_args: dict[str, Any]) -> dict[str, Any]:
        rows = [
            {
                "rule": finding.rule_id,
                "severity": finding.severity,
                "message": finding.message,
                "fields": list(finding.fields),
                "blocks_posting": finding.is_blocking,
            }
            for finding in findings
            if not finding.passed
        ]
        return {"failing": rows, "count": len(rows)}

    return run


def _current_values(fields: tuple[ExtractedField, ...]) -> Callable[..., dict[str, Any]]:
    """What was read, with how sure the pipeline is and what it quoted."""

    def run(_args: dict[str, Any]) -> dict[str, Any]:
        return {
            "fields": [
                {
                    "field": field.name,
                    "value": field.value,
                    "confidence": field.confidence,
                    "verified_against_page": field.grounded,
                    "quoted": field.evidence,
                }
                for field in fields
                if not field.is_empty
            ]
        }

    return run


def build_repair_tools(
    ocr_text: str,
    schema: DocumentSchema,
    fields: tuple[ExtractedField, ...],
    findings: tuple[RuleFinding, ...],
    box: ProposalBox,
) -> list[Tool]:
    """Read-only tools for repairing one document's extraction."""
    return [
        Tool(
            name="show_extraction",
            description="What was read off this document, with confidence and the quote behind each value. Start here.",
            parameters={"type": "object", "properties": {}},
            run=_current_values(fields),
        ),
        Tool(
            name="show_failing_checks",
            description="Which checks are failing and which fields they concern.",
            parameters={"type": "object", "properties": {}},
            run=_failing_checks(findings),
        ),
        Tool(
            name="show_missing_fields",
            description="Fields the schema expects that are still empty, with the labels they usually carry.",
            parameters={"type": "object", "properties": {}},
            run=_missing_fields(schema, fields),
        ),
        Tool(
            name="find_on_page",
            description="Search the recognised page text. Use this before proposing anything - if it is not here, it is not on the document.",
            parameters={
                "type": "object",
                "properties": {"query": _string("Text or number to look for, e.g. 'Amount Due' or '86,480'")},
                "required": ["query"],
            },
            run=_find_on_page(ocr_text),
        ),
        Tool(
            name="read_near_label",
            description="Return the line carrying a label and the line after it, which is where a value usually sits.",
            parameters={
                "type": "object",
                "properties": {"label": _string("A label as printed, e.g. 'Due Date' or 'เลขที่'")},
                "required": ["label"],
            },
            run=_lines_near(ocr_text),
        ),
        Tool(
            name="propose_value",
            description=(
                "Propose a value for one field. `evidence` must be text copied verbatim from the page - "
                "it is checked, and a proposal whose quote is not on the document is discarded."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "field": _string("Field name from the schema"),
                    "value": _string("The value as printed on the document"),
                    "evidence": _string("Verbatim text from the page that shows this value"),
                    "reason": _string("Why this is the right value, in one sentence"),
                },
                "required": ["field", "value", "evidence"],
            },
            run=lambda args: (
                box.add(
                    str(args.get("field", "")).strip(),
                    str(args.get("value", "")).strip(),
                    str(args.get("evidence", "")).strip(),
                    str(args.get("reason", "")).strip(),
                )
                if all(str(args.get(key, "")).strip() for key in ("field", "value", "evidence"))
                else {"error": "field, value and evidence are all required."}
            ),
        ),
    ]
