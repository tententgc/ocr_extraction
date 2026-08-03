"""The orchestrator: bytes in, a routed and recorded document out.

Stages run in a fixed order and each one appends to the run's trace, so the UI
can show exactly what happened and why. Every stage is defensive: a failure
degrades the run and routes it to a human rather than raising.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from docuflow.agents.candidates import from_llm, from_rule
from docuflow.agents.classifier import classify
from docuflow.agents.llm_extractor import extract_with_llm
from docuflow.agents.merge import merge_all
from docuflow.agents.verify_extractor import verify_with_llm
from docuflow.agents.reconciler import Reconciliation, reconcile
from docuflow.agents.rule_extractor import detect_currency_from_text, extract_with_rules, find_line_items
from docuflow.config import Settings
from docuflow.confidence import document_confidence, score_derived
from docuflow.ingest.store import StoredBlob, store_bytes
from docuflow.llm.base import LlmProvider
from docuflow.llm.factory import resolve_llm, resolve_verify_llm
from docuflow.masterdata.repository import MasterDataRepository, VendorMatch
from docuflow.models import (
    DocumentRun,
    ExtractedField,
    LineItem,
    OcrResult,
    SourceInfo,
    TraceStep,
)
from docuflow.normalize.amounts import parse_money, parse_number
from docuflow.ocr.render import UnsupportedDocumentError, render_document
from docuflow.ocr.engine import run_ocr
from docuflow.ocr.quality import score_document_text
from docuflow.routing import decide
from docuflow.rules import RuleContext, run_rules
from docuflow.schemas.registry import DocumentSchema, get_schema
from docuflow.store.documents import DocumentStore
from docuflow.store.records import RecordStore, reference_field_for

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str], None]


@dataclass(frozen=True)
class PipelineResult:
    run: DocumentRun
    reconciliation: Reconciliation
    vendor_match: VendorMatch
    duplicate_of: dict[str, Any] | None


class Pipeline:
    """Runs a document end to end."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.master = MasterDataRepository(settings.db_path)
        self.documents = DocumentStore(settings.db_path)
        self.records = RecordStore(settings.db_path)

    # --- entry points ------------------------------------------------------
    def process_bytes(
        self,
        content: bytes,
        filename: str,
        source_kind: str = "upload",
        origin: str | None = None,
        content_type: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> PipelineResult:
        """Full pipeline for an in-memory document."""
        blob = store_bytes(content, filename, self.settings.blob_dir)
        source = SourceInfo(
            kind=source_kind,
            filename=blob.filename,
            origin=origin,
            content_type=content_type,
            byte_size=blob.byte_size,
            sha256=blob.sha256,
        )
        return self._run(blob, source, on_progress)

    def process_file(
        self, path: Path, source_kind: str = "path", on_progress: ProgressCallback | None = None
    ) -> PipelineResult:
        return self.process_bytes(
            path.read_bytes(), path.name, source_kind=source_kind, origin=str(path), on_progress=on_progress
        )

    # --- the pipeline ------------------------------------------------------
    def _run(
        self, blob: StoredBlob, source: SourceInfo, on_progress: ProgressCallback | None
    ) -> PipelineResult:
        document_id = f"DOC-{uuid.uuid4().hex[:10].upper()}"
        run = DocumentRun(document_id=document_id, source=source, blob_path=str(blob.path))

        def progress(step: str, message: str) -> None:
            if on_progress:
                on_progress(step, message)

        llm, llm_note = resolve_llm(self.settings)

        run, images = self._stage_render(run, blob, progress)
        if run.error:
            # Nothing downstream can run, so route it out here rather than
            # leaving it at the default "needs review".
            run = run.evolve(decision="rejected", decision_reasons=(run.error,))
            return self._finish(
                run, get_schema("unknown"), VendorMatch(None, 0.0, "not evaluated"), None, Reconciliation()
            )

        run = self._stage_ocr(run, images, progress)
        ocr_report = score_document_text([page.text for page in run.ocr.pages])
        pages = [(page.page_number, page.text) for page in run.ocr.pages]

        run, schema = self._stage_classify(run, llm, llm_note, progress)
        run = self._stage_extract(run, schema, pages, ocr_report.score, llm, progress)
        run = self._stage_line_items(run, schema, pages, progress)

        run, vendor_match, duplicate = self._stage_validate(run, schema, pages, progress)
        run = self._stage_route(run, schema, progress)
        reconciliation = self._stage_reconcile(run, llm, progress)

        return self._finish(run, schema, vendor_match, duplicate, reconciliation)

    # --- stages ------------------------------------------------------------
    def _stage_render(
        self, run: DocumentRun, blob: StoredBlob, progress: ProgressCallback
    ) -> tuple[DocumentRun, list]:
        progress("render", "Rendering document pages")
        started = time.perf_counter()
        try:
            images = render_document(
                blob.path,
                self.settings.page_dir / run.document_id,
                max_pages=self.settings.ocr_max_pages,
                target_dim=self.settings.ocr_image_dim,
            )
        except UnsupportedDocumentError as exc:
            return (
                run.evolve(error=str(exc)).with_step(
                    TraceStep(step="render", status="error", summary=str(exc))
                ),
                [],
            )
        except Exception as exc:  # noqa: BLE001 - corrupt files must not crash the API
            logger.exception("render failed for %s", blob.path)
            message = f"Could not open this file: {exc}"
            return (
                run.evolve(error=message).with_step(
                    TraceStep(step="render", status="error", summary=message)
                ),
                [],
            )

        return (
            run.evolve(page_images=tuple(str(image.path) for image in images)).with_step(
                TraceStep(
                    step="render",
                    summary=f"Rendered {len(images)} page(s)",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    detail={"pages": len(images)},
                )
            ),
            images,
        )

    def _stage_ocr(self, run: DocumentRun, images: list, progress: ProgressCallback) -> DocumentRun:
        progress("ocr", "Reading the document with Typhoon OCR 1.5")
        result: OcrResult = run_ocr(images, self.settings)
        report = score_document_text([page.text for page in result.pages])
        status = "warning" if result.degraded else "ok"
        return run.evolve(ocr=result).with_step(
            TraceStep(
                step="ocr",
                status=status,
                summary=f"{result.provider} read {result.char_count:,} characters "
                f"from {len(result.pages)} page(s) (quality {report.score:.0%})",
                duration_ms=result.duration_ms,
                detail={
                    "provider": result.provider,
                    "quality": report.score,
                    "degraded": result.degraded,
                    "notes": list(result.notes)[:5],
                },
            )
        )

    def _stage_classify(
        self, run: DocumentRun, llm: LlmProvider | None, llm_note: str, progress: ProgressCallback
    ) -> tuple[DocumentRun, DocumentSchema]:
        progress("classify", "Identifying the document type")
        started = time.perf_counter()
        classification = classify(run.ocr.full_text, llm)
        schema = get_schema(classification.doc_type)
        summary = f"{schema.display_name} ({classification.confidence:.0%}) via {classification.method}"
        return (
            run.evolve(
                doc_type=classification.doc_type, doc_type_confidence=classification.confidence
            ).with_step(
                TraceStep(
                    step="classify",
                    summary=summary,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    detail={
                        "doc_type": classification.doc_type,
                        "method": classification.method,
                        "reason": classification.reason,
                        "llm": llm.name if llm else "rules-only",
                        "llm_note": llm_note,
                    },
                )
            ),
            schema,
        )

    def _stage_extract(
        self,
        run: DocumentRun,
        schema: DocumentSchema,
        pages: list[tuple[int, str]],
        ocr_quality: float,
        llm: LlmProvider | None,
        progress: ProgressCallback,
    ) -> DocumentRun:
        progress("extract", f"Extracting {len(schema.fields)} fields")
        started = time.perf_counter()

        # Channel 1: field-first extraction.
        primary_result = extract_with_llm(run.ocr.full_text, schema, llm)
        primary = {
            name: from_llm(name, candidate.raw_value, candidate.evidence, channel="model")
            for name, candidate in primary_result.candidates.items()
        }

        # Channel 2: an independent second reading, label-first and preferably
        # on a different model. Falls back to the deterministic label matcher
        # when no second model is reachable, so a keyless run still gets a
        # cross-check rather than silently dropping to one channel.
        verify_llm, verify_note = resolve_verify_llm(self.settings, llm)
        secondary_result = verify_with_llm(run.ocr.full_text, schema, verify_llm)
        if secondary_result.ok:
            secondary = {
                name: from_llm(name, candidate.raw_value, candidate.evidence, channel="verify")
                for name, candidate in secondary_result.candidates.items()
            }
            channel_name = getattr(verify_llm, "model", None) or getattr(verify_llm, "name", "model")
        else:
            secondary = {
                name: from_rule(candidate)
                for name, candidate in extract_with_rules(pages, schema).items()
            }
            channel_name = "label matcher"
            verify_note = verify_note or secondary_result.note

        fields = merge_all(schema, primary, secondary, pages, ocr_quality, run.ocr.degraded)
        fields = self._derive_currency(fields, schema, pages, ocr_quality, run.ocr.degraded)

        populated = [field for field in fields if not field.is_empty]
        agreed = [field for field in populated if field.grounded and field.confidence >= 0.9]
        status = "ok" if primary_result.ok else "warning"
        return run.with_fields(fields).with_step(
            TraceStep(
                step="extract",
                status=status,
                summary=f"{len(populated)}/{len(schema.fields)} fields extracted, "
                f"{len(agreed)} confirmed by both readings (second reading: {channel_name})"
                + ("" if primary_result.ok else f" ({primary_result.note})"),
                duration_ms=int((time.perf_counter() - started) * 1000),
                detail={
                    "primary_fields": sorted(primary),
                    "secondary_fields": sorted(secondary),
                    "secondary_channel": channel_name,
                    "primary_ok": primary_result.ok,
                    "note": "; ".join(filter(None, [primary_result.note, verify_note])),
                },
            )
        )

    def _derive_currency(
        self,
        fields: tuple[ExtractedField, ...],
        schema: DocumentSchema,
        pages: list[tuple[int, str]],
        ocr_quality: float,
        ocr_degraded: bool,
    ) -> tuple[ExtractedField, ...]:
        """Fall back to a document-wide currency scan when the field is absent.

        Many Thai invoices never print a currency code - the amounts are simply
        Baht. Marking that as `derived` is more honest than an ungrounded guess
        or an empty field the downstream ledger cannot use.
        """
        spec = schema.field("currency")
        if spec is None:
            return fields

        current = next((item for item in fields if item.name == "currency"), None)
        if current is not None and not current.is_empty and current.grounded:
            return fields

        detected = detect_currency_from_text(pages)
        if not detected or detected not in spec.choices:
            return fields

        replacement = ExtractedField(
            name="currency",
            label=spec.label,
            value=detected,
            raw_value=detected,
            evidence=None,
            confidence=score_derived(
                ocr_quality=ocr_quality, ocr_degraded=ocr_degraded, is_critical=spec.critical
            ),
            grounded=False,
            source="derived",
            notes=("currency was not printed as a code; inferred from the amounts and language on the page",),
        )
        return tuple(replacement if item.name == "currency" else item for item in fields)

    def _stage_line_items(
        self, run: DocumentRun, schema: DocumentSchema, pages: list[tuple[int, str]], progress: ProgressCallback
    ) -> DocumentRun:
        if not schema.has_line_items:
            return run
        progress("line_items", "Reading the line item table")
        rows = find_line_items(pages)
        items = tuple(
            LineItem(
                description=str(row.get("description", ""))[:300],
                quantity=parse_number(row.get("quantity")),
                unit_price=parse_money(row.get("unit_price")),
                amount=parse_money(row.get("amount")),
                confidence=0.8,
            )
            for row in rows
        )
        return run.evolve(line_items=items).with_step(
            TraceStep(
                step="line_items",
                status="ok" if items else "skipped",
                summary=f"{len(items)} line item(s) read" if items else "no line item table found",
                detail={"count": len(items)},
            )
        )

    def _stage_validate(
        self, run: DocumentRun, schema: DocumentSchema, pages: list[tuple[int, str]], progress: ProgressCallback
    ) -> tuple[DocumentRun, VendorMatch, dict[str, Any] | None]:
        progress("validate", "Cross-checking against business rules and master data")
        started = time.perf_counter()

        vendor_name = run.value("vendor_name") or run.value("merchant_name") or run.value("applicant_name")
        tax_id = run.value("vendor_tax_id") or run.value("merchant_tax_id") or run.value("tax_id")
        vendor_match = self.master.find_vendor(
            str(vendor_name) if vendor_name else None, str(tax_id) if tax_id else None
        )

        reference = run.value(reference_field_for(run.doc_type))
        duplicate = self.records.find_duplicate(
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
            master=self.master,
            vendor_match=vendor_match,
            ocr_text=run.ocr.full_text,
            ocr_degraded=run.ocr.degraded,
            today=date.today(),
            duplicate_of=duplicate,
            **RuleContext.policy_from(self.settings),
        )
        free_mode = self.settings.rules_free_mode
        skipped = self.settings.skipped_rules
        findings = run_rules(context, free_mode=free_mode, skip=skipped)

        failed = [finding for finding in findings if not finding.passed]
        blocking = [finding for finding in failed if finding.severity == "blocking"]
        status = "error" if blocking else ("warning" if failed else "ok")
        if free_mode:
            summary = "Free mode: no business rule was evaluated"
        else:
            summary = (
                f"{len(findings) - len(failed)}/{len(findings)} checks passed"
                + (f", {len(blocking)} blocking" if blocking else "")
                + (f", {len(skipped)} disabled" if skipped else "")
            )

        return (
            run.evolve(findings=findings).with_step(
                TraceStep(
                    step="validate",
                    status=status,
                    summary=summary,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    detail={
                        "vendor": vendor_match.vendor["legal_name"] if vendor_match.vendor else None,
                        "vendor_match_method": vendor_match.method,
                        "failed": [finding.rule_id for finding in failed],
                        "blocking": [finding.rule_id for finding in blocking],
                        "free_mode": free_mode,
                        "disabled_rules": sorted(skipped),
                    },
                )
            ),
            vendor_match,
            duplicate,
        )

    def _stage_route(self, run: DocumentRun, schema: DocumentSchema, progress: ProgressCallback) -> DocumentRun:
        progress("route", "Deciding whether this can post automatically")
        confidence = document_confidence(
            run.fields,
            run.findings,
            frozenset(spec.name for spec in schema.required_fields),
            frozenset(spec.name for spec in schema.critical_fields),
        )
        run = run.evolve(confidence=confidence)
        outcome = decide(
            run,
            schema,
            self.settings.auto_post_min_confidence,
            rules_evaluated=not self.settings.rules_free_mode,
        )
        status = {"auto_post": "ok", "needs_review": "warning", "rejected": "error"}[outcome.decision]
        return run.evolve(decision=outcome.decision, decision_reasons=outcome.reasons).with_step(
            TraceStep(
                step="route",
                status=status,
                summary=f"{outcome.decision.replace('_', ' ')} (confidence {confidence:.0%})",
                detail={"reasons": list(outcome.reasons)},
            )
        )

    def _stage_reconcile(
        self, run: DocumentRun, llm: LlmProvider | None, progress: ProgressCallback
    ) -> Reconciliation:
        if not run.failed_findings:
            return Reconciliation(ran=False, note="nothing to reconcile")
        progress("reconcile", "Investigating the discrepancies")
        return reconcile(run, llm, master=self.master, records=self.records)

    # --- persistence -------------------------------------------------------
    def _finish(
        self,
        run: DocumentRun,
        schema: DocumentSchema,
        vendor_match: VendorMatch,
        duplicate: dict[str, Any] | None,
        reconciliation: Reconciliation,
    ) -> PipelineResult:
        if reconciliation.ran or reconciliation.assessment:
            run = run.with_step(
                TraceStep(
                    step="reconcile",
                    status="warning",
                    summary=reconciliation.assessment[:200] or "discrepancy analysis complete",
                    detail={
                        # The full text lives here; `summary` is the trace one-liner.
                        "assessment": reconciliation.assessment,
                        "root_cause": reconciliation.root_cause,
                        "recommended_action": reconciliation.recommended_action,
                        "reviewer_note": reconciliation.reviewer_note,
                        "tools_used": reconciliation.tools_used,
                        "tool_calls": [
                            {
                                "name": call.name,
                                "arguments": call.arguments,
                                "result": call.result,
                            }
                            for call in reconciliation.tool_calls
                        ],
                        "corrections": [
                            {
                                "field": correction.field,
                                "suggested_value": correction.suggested_value,
                                "reason": correction.reason,
                                "grounded": correction.grounded,
                            }
                            for correction in reconciliation.corrections
                        ],
                    },
                )
            )

        self.documents.save(run)
        self.records.log(
            "document_processed",
            document_id=run.document_id,
            doc_type=run.doc_type,
            decision=run.decision,
            confidence=run.confidence,
            source=run.source.kind,
        )

        # Straight-through posting happens only here, and only for cleared runs.
        if run.decision == "auto_post":
            try:
                posted = self.records.post_record(run, vendor_match.vendor_id, actor="system")
                run = run.evolve(record_id=posted["record_id"]).with_step(
                    TraceStep(
                        step="post",
                        summary=f"Posted to the ledger as {posted['record_id']}",
                        detail=posted,
                    )
                )
                self.documents.save(run)
            except Exception as exc:  # noqa: BLE001 - posting failure must not lose the run
                logger.exception("posting failed for %s", run.document_id)
                run = run.evolve(decision="needs_review").with_step(
                    TraceStep(step="post", status="error", summary=f"Posting failed: {exc}")
                )
                self.documents.save(run)

        return PipelineResult(
            run=run, reconciliation=reconciliation, vendor_match=vendor_match, duplicate_of=duplicate
        )
