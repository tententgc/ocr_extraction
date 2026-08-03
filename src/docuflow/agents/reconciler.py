"""Discrepancy agent: investigates failures with tools and explains them.

Runs only when a check has already failed, so it costs nothing on the happy
path. It may look things up freely, but any correction it proposes is re-grounded
against the page text before a reviewer ever sees it - the agent assists a human,
it never changes a value itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from docuflow.agents.agent_loop import AgentResult, ToolInvocation, run_agent
from docuflow.agents.grounding import find_evidence, value_supported_by_evidence
from docuflow.agents.prompts import RECONCILER_SYSTEM, reconciliation_prompt
from docuflow.agents.tools import Tool, build_toolbox
from docuflow.llm.base import LlmProvider
from docuflow.masterdata.repository import MasterDataRepository
from docuflow.models import DocumentRun
from docuflow.store.records import RecordStore

logger = logging.getLogger(__name__)

VALID_ACTIONS = frozenset({"post", "review", "reject"})
VALID_ROOT_CAUSES = frozenset({"ocr_error", "document_error", "master_data_gap", "policy_exception", "none"})
MAX_CORRECTIONS = 6
# Generous, because reasoning models spend part of this budget thinking.
MAX_TOKENS = 3000


@dataclass(frozen=True)
class SuggestedCorrection:
    field: str
    suggested_value: str
    evidence: str
    reason: str
    grounded: bool


@dataclass(frozen=True)
class Reconciliation:
    assessment: str = ""
    root_cause: str = "none"
    recommended_action: str = "review"
    reviewer_note: str = ""
    corrections: tuple[SuggestedCorrection, ...] = ()
    tool_calls: tuple[ToolInvocation, ...] = ()
    ran: bool = False
    note: str = ""

    @property
    def tools_used(self) -> list[str]:
        return [call.name for call in self.tool_calls]


def _fields_summary(run: DocumentRun) -> str:
    lines = []
    for item in run.fields:
        if item.is_empty:
            continue
        flag = "" if item.grounded else "  [UNVERIFIED]"
        lines.append(f"- {item.name} = {item.value!r} ({item.confidence:.0%}){flag}")
    missing = [item.name for item in run.fields if item.is_empty]
    if missing:
        lines.append(f"- not found: {', '.join(missing)}")
    return "\n".join(lines) or "(nothing extracted)"


def _findings_summary(run: DocumentRun) -> str:
    lines = [
        f"- [{finding.severity.upper()}] {finding.rule_id}: {finding.message}"
        for finding in run.findings
        if not finding.passed
    ]
    return "\n".join(lines) or "(no rule failures)"


def should_run(run: DocumentRun) -> bool:
    """Only worth calling when there is something to explain."""
    return bool(run.failed_findings)


def _fallback(run: DocumentRun, note: str) -> Reconciliation:
    """Without a model, surface the rule messages directly."""
    blocking = run.blocking_findings or run.failed_findings
    return Reconciliation(
        assessment=" ".join(finding.message for finding in blocking[:3]),
        recommended_action="review" if run.blocking_findings else "post",
        reviewer_note="No agent model available; showing the rule results directly.",
        ran=False,
        note=note,
    )


def _parse(result: AgentResult, run: DocumentRun) -> Reconciliation:
    """Validate the agent's JSON and re-ground every proposed correction."""
    payload = result.payload or {}
    known_fields = {item.name for item in run.fields}
    pages = [(page.page_number, page.text) for page in run.ocr.pages]

    corrections: list[SuggestedCorrection] = []
    for entry in (payload.get("corrections") or [])[:MAX_CORRECTIONS]:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("field", "")).strip()
        value = entry.get("suggested_value")
        if name not in known_fields or value in (None, ""):
            continue
        evidence = str(entry.get("evidence") or "")
        # Only a correction the page actually supports may be offered.
        grounded = find_evidence(evidence, pages).grounded and value_supported_by_evidence(
            str(value), evidence
        )
        corrections.append(
            SuggestedCorrection(
                field=name,
                suggested_value=str(value),
                evidence=evidence,
                reason=str(entry.get("reason", ""))[:300],
                grounded=grounded,
            )
        )

    action = str(payload.get("recommended_action", "review")).strip().lower()
    root_cause = str(payload.get("root_cause", "none")).strip().lower()
    assessment = str(payload.get("assessment", "")).strip()

    if not assessment:
        assessment = " ".join(finding.message for finding in run.failed_findings[:2])

    return Reconciliation(
        assessment=assessment[:1200],
        root_cause=root_cause if root_cause in VALID_ROOT_CAUSES else "none",
        recommended_action=action if action in VALID_ACTIONS else "review",
        reviewer_note=str(payload.get("reviewer_note", ""))[:400],
        corrections=tuple(corrections),
        tool_calls=result.calls,
        ran=True,
        note=result.note,
    )


def reconcile(
    run: DocumentRun,
    llm: LlmProvider | None,
    master: MasterDataRepository | None = None,
    records: RecordStore | None = None,
) -> Reconciliation:
    """Investigate the failures and report what a reviewer needs to know."""
    if not should_run(run):
        return Reconciliation(ran=False, note="no discrepancies to reconcile")
    if llm is None:
        return _fallback(run, "no agent model configured")

    tools: list[Tool] = []
    if master is not None and records is not None:
        tools = build_toolbox(
            ocr_text=run.ocr.full_text,
            master=master,
            records=records,
            doc_type=run.doc_type,
            document_id=run.document_id,
        )

    result = run_agent(
        llm,
        system=RECONCILER_SYSTEM,
        user=reconciliation_prompt(run.doc_type, _fields_summary(run), _findings_summary(run)),
        tools=tools,
        max_tokens=MAX_TOKENS,
    )

    if not result.ok and result.payload is None:
        fallback = _fallback(run, result.note or "agent produced no usable answer")
        return Reconciliation(
            assessment=fallback.assessment,
            root_cause=fallback.root_cause,
            recommended_action=fallback.recommended_action,
            reviewer_note=fallback.reviewer_note,
            tool_calls=result.calls,
            ran=bool(result.calls),
            note=fallback.note,
        )

    return _parse(result, run)
