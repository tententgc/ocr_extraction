"""Give the agent a chance to fix the reading before a human is asked to.

A document arrives in the review queue for one of two reasons: something is
genuinely wrong with it, or the pipeline missed something that was on the page.
The second kind is waste - a person opens it, finds the due date sitting under a
label the extractor did not recognise, types it in, and approves. That is the
work this stage tries to remove.

The agent gets the failing checks, the current reading, and read-only tools for
going back to the page. It proposes values; it never sets them. Every proposal
is put through the same grounding and normalisation as the extraction channels
and is kept only if:

  * the field is one the schema actually has,
  * the quote it carries is really on the page,
  * that quote states the value, and
  * it does not overwrite a value two channels already agreed on.

So the agent cannot argue a document into passing. The only thing it can do is
find what was already there - which is exactly the case that was wasting a
reviewer's time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dataclass_field
from typing import Any

from docuflow.agents.agent_loop import run_agent
from docuflow.agents.candidates import Candidate
from docuflow.agents.grounding import find_evidence, value_supported_by_evidence
from docuflow.agents.repair_tools import ProposalBox, build_repair_tools
from docuflow.confidence import score_field
from docuflow.llm.base import LlmProvider
from docuflow.models import ExtractedField, RuleFinding
from docuflow.normalize.apply import normalize_field_value
from docuflow.schemas.registry import DocumentSchema

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 8
MAX_OUTPUT_TOKENS = 1_200
# A value the agent recovered is one reading, verified against the page. That is
# the same evidential position as a single grounded channel, and it is scored the
# same way - never as though two channels had agreed.
REPAIR_IS_SINGLE_CHANNEL = True

REPAIR_SYSTEM = """You recover values an extractor missed on a document.

Work in this order:
1. show_extraction and show_failing_checks - see what was read and what is wrong.
2. show_missing_fields - see what is absent.
3. find_on_page / read_near_label - look for those values on the actual page.
4. propose_value - only for values you have seen in the page text.

Rules:
- Never propose a value you have not found with a tool. If the page does not
  state it, say so and propose nothing.
- `evidence` must be copied from the page exactly, not paraphrased.
- Do not "correct" a value that is already right just to make a check pass.
- Proposing nothing is a good outcome when the document is genuinely wrong.

Finish with JSON: {"assessment":"one sentence","proposed":<count>}"""


@dataclass(frozen=True)
class RepairOutcome:
    """What the repair pass changed, and what it tried."""

    ran: bool = False
    fields: tuple[ExtractedField, ...] = ()
    applied: tuple[str, ...] = ()
    rejected: tuple[dict[str, Any], ...] = ()
    assessment: str = ""
    note: str = ""
    tools_used: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.applied)

    def to_json(self) -> dict[str, Any]:
        return {
            "ran": self.ran,
            "applied": list(self.applied),
            "rejected": list(self.rejected),
            "assessment": self.assessment,
            "note": self.note,
            "tools_used": list(self.tools_used),
        }


def _protected(field: ExtractedField | None) -> bool:
    """Whether a field is settled enough that the agent must not touch it.

    Two channels that read the page independently and agreed is stronger
    evidence than one agent looking again, and a human's correction is stronger
    than either.
    """
    if field is None or field.is_empty:
        return False
    if field.source == "human":
        return True
    return field.grounded and field.confidence >= 0.9


def _vet(
    proposal: dict[str, Any],
    schema: DocumentSchema,
    fields: tuple[ExtractedField, ...],
    ocr_pages: list[tuple[int, str]],
    ocr_quality: float,
    ocr_degraded: bool,
) -> tuple[ExtractedField | None, str]:
    """Turn one proposal into a field, or say why it was thrown out."""
    name = str(proposal.get("field") or "").strip()
    spec = schema.field(name)
    if spec is None:
        return None, f"'{name}' is not a field on this document type"

    existing = next((item for item in fields if item.name == name), None)
    if _protected(existing):
        return None, f"{name} was already read and verified; not overwritten"

    raw = str(proposal.get("value") or "").strip()
    evidence = str(proposal.get("evidence") or "").strip()
    if not raw or not evidence:
        return None, f"{name}: a value and a page quote are both required"

    grounding = find_evidence(evidence, ocr_pages)
    if not grounding.grounded:
        return None, f"{name}: the quoted text is not on the page ({grounding.reason})"

    value, note = normalize_field_value(raw, spec)
    if value is None:
        return None, f"{name}: {note or 'the value could not be interpreted'}"

    if not value_supported_by_evidence(raw, evidence, spec.type, value):
        return None, f"{name}: the quoted text does not state {raw!r}"

    return (
        ExtractedField(
            name=name,
            label=spec.label,
            value=value,
            raw_value=raw,
            evidence=evidence,
            page=grounding.page,
            confidence=score_field(
                both_channels_agree=False,
                has_primary=True,
                has_secondary=False,
                grounded=True,
                is_critical=spec.critical,
                ocr_quality=ocr_quality,
                ocr_degraded=ocr_degraded,
                had_normalization_note=bool(note),
            ),
            grounded=True,
            source="agent",
            notes=tuple(
                filter(
                    None,
                    (
                        "recovered from the page by the repair agent"
                        + (f": {proposal['reason']}" if proposal.get("reason") else ""),
                        note,
                    ),
                )
            ),
        ),
        "",
    )


def repair_extraction(
    *,
    llm: LlmProvider | None,
    schema: DocumentSchema,
    fields: tuple[ExtractedField, ...],
    findings: tuple[RuleFinding, ...],
    ocr_pages: list[tuple[int, str]],
    ocr_text: str,
    ocr_quality: float,
    ocr_degraded: bool,
) -> RepairOutcome:
    """Try to recover values the extractor missed. Never raises."""
    if llm is None:
        return RepairOutcome(note="no agent model configured")

    failing = [finding for finding in findings if not finding.passed]
    missing = [spec for spec in schema.fields if not any(
        f.name == spec.name and not f.is_empty for f in fields
    )]
    if not failing and not missing:
        return RepairOutcome(note="nothing to repair")

    box = ProposalBox(proposals=[])
    tools = build_repair_tools(ocr_text, schema, fields, findings, box)

    request = (
        f"Document type: {schema.doc_type}\n"
        f"{len(failing)} check(s) failing, {len(missing)} field(s) empty.\n\n"
        "Find the values this document actually states."
    )

    try:
        result = run_agent(
            llm,
            REPAIR_SYSTEM,
            request,
            tools,
            max_iterations=MAX_ITERATIONS,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
    except Exception as exc:  # noqa: BLE001 - a repair attempt must not fail the run
        logger.warning("repair pass failed: %s", exc)
        return RepairOutcome(ran=True, note=f"repair pass failed: {exc}")

    updated = list(fields)
    applied: list[str] = []
    rejected: list[dict[str, Any]] = []

    for proposal in box.proposals:
        candidate, reason = _vet(proposal, schema, tuple(updated), ocr_pages, ocr_quality, ocr_degraded)
        if candidate is None:
            rejected.append({"field": proposal.get("field"), "reason": reason})
            continue
        index = next((i for i, item in enumerate(updated) if item.name == candidate.name), None)
        if index is None:
            updated.append(candidate)
        else:
            updated[index] = candidate
        applied.append(candidate.name)

    payload = result.payload if isinstance(result.payload, dict) else {}
    return RepairOutcome(
        ran=True,
        fields=tuple(updated),
        applied=tuple(applied),
        rejected=tuple(rejected),
        assessment=str(payload.get("assessment", ""))[:400],
        note=result.note if hasattr(result, "note") else "",
        tools_used=tuple(dict.fromkeys(call.name for call in result.calls)),
    )
