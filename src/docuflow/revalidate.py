"""Judge a document again under the current policy, without reading it again.

A policy is not fixed. Someone disables a rule that never applied to this
company, writes a new one, changes the auto-post threshold, or registers the
vendor that was blocking half the queue. Every document already held is now
being judged by a policy that no longer exists.

Re-evaluation replays only the second half of the pipeline - master-data
lookup, rules, confidence, routing - against the stored run. The reading is not
repeated: no render, no OCR, no model call. The values on the page did not
change; what changed is what we decide about them. That makes re-checking the
whole queue a local operation measured in milliseconds rather than a re-run.

Two rules keep this safe:

* A document a *person* rejected is never posted by a re-evaluation. The
  machine may reconsider it, but it lands back in the queue for a human, because
  a policy change is not consent to overturn someone's decision.
* A document already posted is refused outright. It is in the ledger; the way
  to change a posted record is a credit note, not a re-run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

from docuflow.agents.rule_agent import runner_for
from docuflow.config import Settings
from docuflow.confidence import document_confidence
from docuflow.masterdata.repository import MasterDataRepository, VendorMatch
from docuflow.models import DocumentRun, TraceStep
from docuflow.routing import decide
from docuflow.rules import RuleContext, run_rules
from docuflow.schemas.registry import DocumentSchema, get_schema
from docuflow.store.custom_rules import CustomRuleStore
from docuflow.store.documents import DocumentStore
from docuflow.store.records import RecordStore, reference_field_for

logger = logging.getLogger(__name__)


class RevalidationError(RuntimeError):
    """The document could not be re-evaluated."""


@dataclass(frozen=True)
class Revalidated:
    """A run with fresh findings and confidence. Nothing has been persisted."""

    run: DocumentRun
    schema: DocumentSchema
    vendor_match: VendorMatch
    duplicate_of: dict[str, Any] | None


def revalidate(settings: Settings, run: DocumentRun) -> Revalidated:
    """Re-run master-data lookup, rules and confidence over an existing run.

    Shared by human review and by re-evaluation so a corrected document and a
    re-checked one are judged by exactly the same code. It decides nothing and
    writes nothing - the caller does that.
    """
    master = MasterDataRepository(settings.db_path)
    records = RecordStore(settings.db_path)
    schema = get_schema(run.doc_type)

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

    context = RuleContext(
        document_id=run.document_id,
        doc_type=run.doc_type,
        schema=schema,
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
    custom = CustomRuleStore(settings.db_path).active_for(run.doc_type)
    findings = run_rules(
        context,
        free_mode=settings.rules_free_mode,
        builtins_disabled=settings.builtin_rules_disabled,
        skip=settings.skipped_rules,
        custom=custom,
        severity=settings.rule_severity_overrides,
        # Resolved only when one of this user's rules actually needs a model.
        agent_runner=runner_for(settings) if any(rule.is_agent for rule in custom) else None,
    )
    run = run.evolve(findings=findings)
    run = run.evolve(
        confidence=document_confidence(
            run.fields,
            findings,
            frozenset(spec.name for spec in schema.required_fields),
            frozenset(spec.name for spec in schema.critical_fields),
        )
    )
    return Revalidated(run, schema, vendor_match, duplicate)


# --------------------------------------------------------------------------- #
# The user-facing operation                                                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Reevaluation:
    document_id: str
    before: dict[str, Any]
    after: dict[str, Any]
    record_id: str | None
    message: str

    @property
    def changed(self) -> bool:
        return self.before["decision"] != self.after["decision"]

    def to_json(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "before": self.before,
            "after": self.after,
            "record_id": self.record_id,
            "changed": self.changed,
            "message": self.message,
        }


def _snapshot(run: DocumentRun) -> dict[str, Any]:
    return {
        "decision": run.decision,
        "confidence": round(run.confidence, 3),
        "blocking": sorted(finding.rule_id for finding in run.blocking_findings),
        "failing": sorted(finding.rule_id for finding in run.failed_findings),
    }


def _rejected_by_a_person(run: DocumentRun) -> bool:
    """Did a reviewer reject this, as opposed to the pipeline refusing it?

    A reviewer's rejection is recorded as a failed `review` step; an unreadable
    file is refused at routing and carries no such step.
    """
    return any(step.step == "review" and step.status == "error" for step in run.trace)


def reevaluate(settings: Settings, document_id: str, actor: str = "reviewer") -> Reevaluation:
    """Re-judge one stored document under the policy in force right now."""
    documents = DocumentStore(settings.db_path)
    records = RecordStore(settings.db_path)

    run = documents.get(document_id)
    if run is None:
        raise RevalidationError(f"Document {document_id} was not found.")
    if run.record_id:
        raise RevalidationError(
            f"{document_id} is already posted as {run.record_id}. A posted record is "
            f"corrected downstream, not by re-running the check."
        )

    before = _snapshot(run)
    was_rejected_by_a_person = _rejected_by_a_person(run)

    result = revalidate(settings, run)
    run = result.run
    outcome = decide(
        run,
        result.schema,
        settings.auto_post_min_confidence,
        rules_evaluated=not settings.rules_free_mode,
    )
    decision, reasons = outcome.decision, outcome.reasons

    # A person said no to this document. The new policy may disagree, but it
    # gets to ask for another look, not to post over the top of them.
    if decision == "auto_post" and was_rejected_by_a_person:
        decision = "needs_review"
        reasons = (
            "This document now clears every check, but it was rejected by a reviewer. "
            "It is back in the queue for a person to confirm rather than posting itself.",
            *reasons,
        )

    run = run.evolve(decision=decision, decision_reasons=reasons)
    after = _snapshot(run)
    run = run.with_step(_trace(before, after, settings))
    documents.save(run)

    record_id = None
    if decision == "auto_post":
        record_id, run = _post(records, documents, run, result.vendor_match, actor)

    records.log(
        "document_reevaluated",
        actor=actor,
        document_id=document_id,
        record_id=record_id,
        before=before,
        after=after,
    )
    return Reevaluation(document_id, before, after, record_id, _message(before, after, record_id))


def _trace(before: dict[str, Any], after: dict[str, Any], settings: Settings) -> TraceStep:
    moved = before["decision"] != after["decision"]
    return TraceStep(
        step="reevaluate",
        status="ok" if moved else "skipped",
        summary=(
            f"Re-checked under the current policy: "
            f"{before['decision'].replace('_', ' ')} → {after['decision'].replace('_', ' ')} "
            f"({before['confidence']:.0%} → {after['confidence']:.0%})"
            if moved
            else f"Re-checked under the current policy: unchanged ({after['decision'].replace('_', ' ')})"
        ),
        detail={
            "before": before,
            "after": after,
            # What the document was judged against, so a surprising outcome can be
            # traced to the policy that produced it rather than guessed at.
            "policy": {
                "auto_post_min_confidence": settings.auto_post_min_confidence,
                "free_mode": settings.rules_free_mode,
                "disabled_rules": sorted(settings.skipped_rules),
            },
        },
    )


def _post(
    records: RecordStore,
    documents: DocumentStore,
    run: DocumentRun,
    vendor_match: VendorMatch,
    actor: str,
) -> tuple[str | None, DocumentRun]:
    """Post a document the new policy cleared, exactly as the pipeline would."""
    try:
        posted = records.post_record(run, vendor_match.vendor_id, actor=actor)
    except Exception as exc:  # noqa: BLE001 - a posting failure must never lose the run
        logger.exception("posting failed for %s after re-evaluation", run.document_id)
        run = run.evolve(decision="needs_review").with_step(
            TraceStep(step="post", status="error", summary=f"Posting failed: {exc}")
        )
        documents.save(run)
        return None, run

    run = run.evolve(record_id=posted["record_id"]).with_step(
        TraceStep(
            step="post",
            summary=f"Posted to the ledger as {posted['record_id']} after re-evaluation",
            detail=posted,
        )
    )
    documents.save(run)
    return posted["record_id"], run


def _message(before: dict[str, Any], after: dict[str, Any], record_id: str | None) -> str:
    if record_id:
        return f"Cleared by the current policy and posted as {record_id}."
    if before["decision"] == after["decision"]:
        return f"No change — still {after['decision'].replace('_', ' ')}."
    return (
        f"{before['decision'].replace('_', ' ')} → {after['decision'].replace('_', ' ')} "
        f"under the current policy."
    )


@dataclass(frozen=True)
class BulkReevaluation:
    results: tuple[Reevaluation, ...]
    skipped: tuple[dict[str, str], ...]

    def to_json(self) -> dict[str, Any]:
        changed = [result for result in self.results if result.changed]
        return {
            "checked": len(self.results),
            "changed": len(changed),
            "posted": [result.record_id for result in self.results if result.record_id],
            "skipped": list(self.skipped),
            "results": [result.to_json() for result in self.results],
            "summary": _bulk_summary(len(self.results), changed, self.skipped),
        }


def _bulk_summary(checked: int, changed: list[Reevaluation], skipped: tuple[dict[str, str], ...]) -> str:
    posted = sum(1 for result in changed if result.record_id)
    parts = [f"Re-checked {checked} document(s)"]
    if changed:
        parts.append(f"{len(changed)} changed")
    if posted:
        parts.append(f"{posted} posted")
    if skipped:
        parts.append(f"{len(skipped)} skipped")
    if not changed and not skipped:
        parts.append("nothing changed under the current policy")
    return ", ".join(parts) + "."


# A re-check of everything would include documents already in the ledger, which
# cannot be re-judged. The queue is what a policy change is actually about.
DEFAULT_SCOPE = "needs_review"
MAX_BULK = 500


def reevaluate_many(
    settings: Settings,
    decision: str | None = DEFAULT_SCOPE,
    actor: str = "reviewer",
    limit: int = MAX_BULK,
) -> BulkReevaluation:
    """Re-judge every document with the given decision under the current policy."""
    documents = DocumentStore(settings.db_path)
    results: list[Reevaluation] = []
    skipped: list[dict[str, str]] = []

    for row in documents.list(decision=decision, limit=limit):
        try:
            results.append(reevaluate(settings, row["document_id"], actor=actor))
        except RevalidationError as exc:
            skipped.append({"document_id": row["document_id"], "reason": str(exc)})
        except Exception as exc:  # noqa: BLE001 - one bad document must not stop the batch
            logger.exception("re-evaluation failed for %s", row["document_id"])
            skipped.append({"document_id": row["document_id"], "reason": str(exc)})

    return BulkReevaluation(tuple(results), tuple(skipped))
