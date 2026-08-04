"""Human review: apply corrections, re-validate, then post or reject.

A reviewer's edit is not simply written through to the ledger. The corrected
values are re-normalised and the full rule set runs again, so an edit that fixes
one problem but creates another is still caught.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from docuflow.config import Settings
from docuflow.models import DocumentRun, ExtractedField, TraceStep
from docuflow.normalize.apply import normalize_field_value
from docuflow.revalidate import revalidate
from docuflow.routing import decide
from docuflow.schemas.registry import get_schema
from docuflow.store.corrections import CorrectionStore
from docuflow.store.documents import DocumentStore
from docuflow.store.records import RecordPostingError, RecordStore

logger = logging.getLogger(__name__)

# A value a human typed and confirmed is the most reliable source available.
HUMAN_CONFIDENCE = 1.0

# Shown when an edit clears every check but nobody has approved the document.
READY_TO_POST = (
    "Every check passes with the corrected values. The document stays here until "
    "someone approves it, or until it is re-checked against the current policy."
)


class ReviewError(RuntimeError):
    """The review could not be applied."""


@dataclass(frozen=True)
class ReviewOutcome:
    run: DocumentRun
    record_id: str | None
    applied: tuple[str, ...]
    rejected_edits: tuple[str, ...]
    message: str


def _apply_edits(
    run: DocumentRun, edits: dict[str, Any], schema
) -> tuple[DocumentRun, list[str], list[str], list[tuple[str, Any, Any]]]:
    """Normalise and apply reviewer edits, returning what was accepted."""
    applied: list[str] = []
    rejected: list[str] = []
    history: list[tuple[str, Any, Any]] = []
    fields = list(run.fields)

    for name, raw in edits.items():
        spec = schema.field(name)
        if spec is None:
            rejected.append(f"{name}: not a field of this document type")
            continue

        if raw is None or (isinstance(raw, str) and not raw.strip()):
            value, note = None, None
        else:
            value, note = normalize_field_value(raw, spec)
            if value is None:
                rejected.append(f"{name}: {note or 'value could not be interpreted'}")
                continue

        index = next((i for i, field in enumerate(fields) if field.name == name), None)
        previous = fields[index].value if index is not None else None
        corrected = ExtractedField(
            name=name,
            label=spec.label,
            value=value,
            raw_value=None if raw is None else str(raw),
            evidence=fields[index].evidence if index is not None else None,
            page=fields[index].page if index is not None else None,
            confidence=HUMAN_CONFIDENCE if value is not None else 0.0,
            grounded=True,  # a human confirmed it against the image
            source="human",
            notes=("corrected by reviewer",),
        )
        if index is None:
            fields.append(corrected)
        else:
            fields[index] = corrected
        applied.append(name)
        history.append((name, previous, value))

    return run.with_fields(tuple(fields)), applied, rejected, history


def _save(
    run: DocumentRun,
    schema,
    settings: Settings,
    documents: DocumentStore,
    records: RecordStore,
    applied: list[str],
    rejected_edits: list[str],
    reviewer: str,
    document_id: str,
) -> ReviewOutcome:
    """Keep the corrected values. Do not move the document.

    Saving is an edit, not a verdict. The rules re-run so the reviewer can see
    what their correction fixed - that is the point of saving before deciding -
    but the routing decision is left exactly where it was.

    Recomputing it here was the older behaviour and it was wrong in a specific
    way: a correction that cleared every check silently moved the document out
    of "needs review" and labelled it as posted, without anybody approving
    anything and without a record existing. A save that changes a document's
    disposition is not a save.
    """
    outcome = decide(
        run, schema, settings.auto_post_min_confidence, rules_evaluated=not settings.rules_free_mode
    )
    cleared = outcome.decision == "auto_post"

    # The stored reasons must describe the state the document is actually in.
    # When the edit fixed everything, the routing reasons would read like an
    # approval, so say what is true instead: it is ready, and waiting.
    reasons = (READY_TO_POST,) if cleared else outcome.reasons
    # What still holds it comes from routing, not from a rule count: a document
    # can be held with every rule passing - a required field missing, a critical
    # value unverified, confidence short of the threshold - and "0 checks
    # failing" next to a held document reads like a contradiction.
    still = outcome.reasons[0] if outcome.reasons else "still held"
    summary = (
        f"{len(applied)} correction(s) saved by {reviewer}; "
        + ("every check now passes" if cleared else f"still held — {still[:120]}")
        + f" (confidence {run.confidence:.0%})"
    )

    run = run.evolve(decision_reasons=reasons).with_step(
        TraceStep(
            step="review",
            status="ok" if cleared else "warning",
            summary=summary,
            detail={"corrections": applied, "would_route_to": outcome.decision},
        )
    )
    documents.save(run)
    records.log("review_saved", actor=reviewer, document_id=document_id, edits=applied)
    return ReviewOutcome(
        run,
        None,
        tuple(applied),
        tuple(rejected_edits),
        "Saved. Every check now passes — approve it to post." if cleared else "Saved.",
    )


def apply_review(
    settings: Settings,
    document_id: str,
    edits: dict[str, Any] | None = None,
    action: str = "approve",
    reviewer: str = "reviewer",
    note: str = "",
) -> ReviewOutcome:
    """Apply a reviewer's decision to a document."""
    if action not in {"approve", "reject", "save"}:
        raise ReviewError(f"Unknown review action '{action}'.")

    documents = DocumentStore(settings.db_path)
    records = RecordStore(settings.db_path)
    corrections = CorrectionStore(settings.db_path)

    run = documents.get(document_id)
    if run is None:
        raise ReviewError(f"Document {document_id} was not found.")
    if run.record_id:
        raise ReviewError(f"Document {document_id} has already been posted as {run.record_id}.")

    schema = get_schema(run.doc_type)
    run, applied, rejected_edits, history = _apply_edits(run, edits or {}, schema)

    # --- re-validate with the corrected values ----------------------------
    # The same code path a policy re-check uses, so a document corrected by a
    # person and one re-judged by a policy change are never assessed by two
    # subtly different rule runs.
    revalidated = revalidate(settings, run)
    run, vendor_match, findings = revalidated.run, revalidated.vendor_match, revalidated.run.findings

    for name, previous, value in history:
        corrections.record(
            document_id=document_id,
            doc_type=run.doc_type,
            field_name=name,
            old_value=previous,
            new_value=value,
            vendor_key=vendor_match.vendor_id,
            reviewer=reviewer,
        )

    # --- carry out the decision -------------------------------------------
    if action == "reject":
        run = run.evolve(
            decision="rejected",
            decision_reasons=(note or f"Rejected by {reviewer}.",),
        ).with_step(TraceStep(step="review", status="error", summary=f"Rejected by {reviewer}. {note}".strip()))
        documents.save(run)
        records.log("document_rejected", actor=reviewer, document_id=document_id, note=note, edits=applied)
        return ReviewOutcome(run, None, tuple(applied), tuple(rejected_edits), "Document rejected.")

    if action == "save":
        return _save(run, schema, settings, documents, records, applied, rejected_edits, reviewer, document_id)

    # approve -> post, but never past a blocking rule that is still failing
    still_blocking = [finding for finding in findings if finding.is_blocking]
    if still_blocking:
        run = run.evolve(
            decision="needs_review",
            decision_reasons=tuple(finding.message for finding in still_blocking),
        ).with_step(
            TraceStep(
                step="review",
                status="warning",
                summary=f"Approval blocked: {len(still_blocking)} check(s) still failing",
                detail={"blocking": [finding.rule_id for finding in still_blocking]},
            )
        )
        documents.save(run)
        records.log(
            "approval_blocked",
            actor=reviewer,
            document_id=document_id,
            blocking=[finding.rule_id for finding in still_blocking],
        )
        raise ReviewError(
            "Cannot post: "
            + " ".join(finding.message for finding in still_blocking[:2])
            + " Correct these values or reject the document."
        )

    run = run.evolve(decision="auto_post")
    try:
        posted = records.post_record(run, vendor_match.vendor_id, actor=reviewer, force=True)
    except RecordPostingError as exc:
        raise ReviewError(str(exc)) from exc

    run = run.evolve(record_id=posted["record_id"], decision_reasons=(f"Approved by {reviewer}.",)).with_step(
        TraceStep(
            step="review",
            summary=f"Approved by {reviewer} with {len(applied)} correction(s); posted as {posted['record_id']}",
            detail={"corrections": applied, **posted},
        )
    )
    documents.save(run)
    return ReviewOutcome(
        run, posted["record_id"], tuple(applied), tuple(rejected_edits), f"Posted as {posted['record_id']}."
    )
