"""Try a rule against documents already processed, before it is saved.

Reading a compiled rule back in words catches most mistakes. This catches the
rest. A rule that reports "12 of 12 pass" on a corpus the author knows contains
bad documents is inverted or vacuous, and a rule that fires on everything is
usually missing its `when` guard. Both are obvious in a table of real results
and easy to miss in a sentence.

Nothing here writes anything: the stored runs are replayed, not reprocessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from docuflow.config import Settings
from docuflow.masterdata.repository import MasterDataRepository
from docuflow.models import DocumentRun
from docuflow.rules.context import RuleContext
from docuflow.rules.custom import CustomRule, run_custom_rule
from docuflow.schemas.registry import get_schema
from docuflow.store.documents import DocumentStore
from docuflow.store.records import RecordStore, reference_field_for

MAX_DOCUMENTS = 40
MAX_EXAMPLES = 6
# An agent rule costs a model call per document, so its trial run is short. Four
# is enough to catch a plan that fails everything or decides nothing, which is
# what a preview is for.
MAX_AGENT_DOCUMENTS = 4


@dataclass(frozen=True)
class PreviewOutcome:
    """What a rule would have done to the documents already on file."""

    considered: int
    passed: int
    failed: int
    not_applicable: int
    examples: tuple[dict[str, Any], ...]
    note: str = ""
    # Held because the agent could not decide, which is neither a pass nor a
    # failure of the document and must not be reported as either.
    unsure: int = 0
    agent: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "considered": self.considered,
            "passed": self.passed,
            "failed": self.failed,
            "unsure": self.unsure,
            "not_applicable": self.not_applicable,
            "agent": self.agent,
            "examples": list(self.examples),
            "note": self.note or self._verdict(),
        }

    def _verdict(self) -> str:
        """A plain reading of the numbers, including the suspicious ones."""
        if self.considered == 0:
            return "No documents of this type have been processed yet, so there is nothing to try it on."
        evaluated = self.passed + self.failed + self.unsure
        if evaluated == 0:
            return (
                f"The rule did not apply to any of the {self.considered} document(s) checked - "
                "its 'when' condition never matched."
            )
        return self._reading(evaluated) + self._cap_note()

    def _reading(self, evaluated: int) -> str:
        if self.unsure == evaluated:
            return (
                f"The agent could not decide on any of the {evaluated} document(s) it looked at, "
                "so all of them would be held for a person. Check the plan can be carried out "
                "with the tools it was given."
            )
        undecided = f" {self.unsure} could not be decided." if self.unsure else ""
        if self.failed == 0:
            return f"All {self.passed} document(s) it applied to would pass.{undecided}"
        if self.passed == 0:
            return (
                f"Every one of the {evaluated} document(s) it applied to would FAIL. "
                f"Worth checking the condition is the right way round.{undecided}"
            )
        return f"{self.failed} of {evaluated} document(s) it applied to would fail.{undecided}"

    def _cap_note(self) -> str:
        if not self.agent or self.considered < MAX_AGENT_DOCUMENTS:
            return ""
        return (
            f" (An agent rule costs a model call per document, so the trial stops at "
            f"{MAX_AGENT_DOCUMENTS}.)"
        )


def _context_for(run: DocumentRun, settings: Settings) -> RuleContext:
    """Rebuild the validation context for a stored run, without re-processing."""
    master = MasterDataRepository(settings.db_path)
    records = RecordStore(settings.db_path)

    vendor_name = run.value("vendor_name") or run.value("merchant_name") or run.value("applicant_name")
    tax_id = run.value("vendor_tax_id") or run.value("merchant_tax_id") or run.value("tax_id")
    vendor_match = master.find_vendor(
        str(vendor_name) if vendor_name else None, str(tax_id) if tax_id else None
    )
    reference = run.value(reference_field_for(run.doc_type))
    duplicate = records.find_duplicate(
        run.doc_type,
        str(reference) if reference else None,
        vendor_match.vendor_id,
        exclude_document_id=run.document_id,
    )
    return RuleContext(
        document_id=run.document_id,
        doc_type=run.doc_type,
        schema=get_schema(run.doc_type),
        fields=run.fields,
        line_items=run.line_items,
        master=master,
        vendor_match=vendor_match,
        ocr_text=run.ocr.full_text,
        ocr_degraded=run.ocr.degraded,
        today=date.today(),
        duplicate_of=duplicate,
        **RuleContext.policy_from(settings),
    )


def preview_rule(
    rule: CustomRule,
    settings: Settings,
    limit: int = MAX_DOCUMENTS,
    runner: Any = None,
) -> PreviewOutcome:
    """Replay `rule` over recent documents of the types it covers."""
    documents = DocumentStore(settings.db_path)
    # Listing gives summaries; the rule needs the extracted fields, so the runs
    # that could possibly match are loaded in full and the rest never touched.
    summaries = [
        row for row in documents.list(limit=limit) if rule.applies_to(row["doc_type"])
    ]
    runs = [
        run
        for run in (documents.get(row["document_id"]) for row in summaries)
        if run is not None and run.fields
    ]
    if rule.is_agent:
        runs = runs[:MAX_AGENT_DOCUMENTS]

    passed = failed = unsure = not_applicable = 0
    examples: list[dict[str, Any]] = []

    for run in runs:
        try:
            finding = run_custom_rule(rule, _context_for(run, settings), runner)
        except Exception as exc:  # noqa: BLE001 - a draft rule is allowed to be wrong
            failed += 1
            examples.append(
                {
                    "document_id": run.document_id,
                    "filename": run.source.filename,
                    "outcome": "error",
                    "detail": str(exc),
                }
            )
            continue

        if finding is None:
            not_applicable += 1
            continue

        if finding.passed:
            passed += 1
        elif finding.detail.get("verdict") == "unsure":
            unsure += 1
        else:
            failed += 1

        if len(examples) < MAX_EXAMPLES:
            verdict = finding.detail.get("verdict")
            examples.append(
                {
                    "document_id": run.document_id,
                    "filename": run.source.filename,
                    "doc_type": run.doc_type,
                    # An agent's "unsure" is a fail in the tally - the document is
                    # held - but showing it as one would hide the difference
                    # between a plan that finds problems and a plan that cannot
                    # reach an answer, which is the thing a preview must reveal.
                    "outcome": verdict or ("pass" if finding.passed else "fail"),
                    "observed": finding.detail.get("observed", {}),
                    "reason": finding.detail.get("reason", ""),
                    "tools_used": finding.detail.get("tools_used", []),
                }
            )

    return PreviewOutcome(
        considered=len(runs),
        passed=passed,
        failed=failed,
        unsure=unsure,
        not_applicable=not_applicable,
        agent=rule.is_agent,
        examples=tuple(examples),
    )
