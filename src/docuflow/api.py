"""HTTP API and web platform.

Pipeline endpoints are declared with `def` rather than `async def` so FastAPI
runs them in its worker threadpool - a document takes tens of seconds and must
not block the event loop.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from docuflow.archive import describe_archive, rebuild as rebuild_archive
from docuflow.bootstrap import bootstrap
from docuflow.config import Settings, get_settings
from docuflow.models_registry import list_models
from docuflow.runtime import SelectionError, SelectionStore
from docuflow.ingest.fetch import FetchError, fetch_url
from docuflow.jobs import MAX_BATCH_FILES, JobQueue
from docuflow.llm.factory import describe_llms, resolve_llm
from docuflow.masterdata.repository import MasterDataRepository
from docuflow.ocr.factory import describe_providers
from docuflow.ocr.render import is_supported
from docuflow.pipeline import Pipeline, PipelineResult
from docuflow.policy import PolicyError, PolicyStore
from docuflow.rules import describe_rules
from docuflow.rules.authoring import available_fields, compile_rule
from docuflow.rules.custom import CustomRuleError, build_rule
from docuflow.rules.preview import preview_rule
from docuflow.store.custom_rules import CustomRuleStore
from docuflow.review import ReviewError, apply_review
from docuflow.samples.catalog import SAMPLE_DOCUMENTS
from docuflow.schemas.registry import DOCUMENT_SCHEMAS
from docuflow.store.corrections import CorrectionStore
from docuflow.store.documents import DocumentStore
from docuflow.store.records import RecordStore
from docuflow.warmup import warm_ollama_in_background

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"

settings = get_settings()
# Runtime model selection; the UI can switch engines without a restart.
selection = SelectionStore(settings)
# Business policy the user changed from the settings screen. Created after the
# database exists, because unlike the model picker it is persisted.
policy: PolicyStore | None = None
# Batch processing. One pool for the process, sized to the configured engine.
queue = JobQueue(lambda: active_settings())


def active_settings() -> Settings:
    """Configuration with the current model selection and policy applied."""
    chosen = selection.settings()
    return policy.settings(chosen) if policy is not None else chosen


def active_queue() -> JobQueue:
    return queue


def active_policy() -> PolicyStore:
    if policy is None:  # pragma: no cover - only before startup completes
        raise RuntimeError("the policy store is not ready yet")
    return policy


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Create the database, seed master data and render the demo corpus."""
    global policy
    bootstrap(settings, with_samples=True)
    policy = PolicyStore(settings)
    # Loads the local models in the background so the first upload does not wait
    # on a cold vision model. Startup is not held up and a failure is harmless.
    warm_ollama_in_background(active_settings())
    yield
    queue.shutdown()


app = FastAPI(
    title="DocuFlow",
    version="1.0.0",
    description="Unstructured documents in, validated downstream records out.",
    lifespan=lifespan,
)


def _pipeline() -> Pipeline:
    """A pipeline bound to whichever models are currently selected."""
    return Pipeline(active_settings())


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class UrlIngestRequest(BaseModel):
    url: str = Field(..., description="Public http(s) URL of a PDF or image")


class ReviewRequest(BaseModel):
    edits: dict[str, Any] = Field(default_factory=dict)
    action: str = Field(default="approve", pattern="^(approve|reject|save)$")
    reviewer: str = "reviewer"
    note: str = ""


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _serialize_result(result: PipelineResult) -> dict[str, Any]:
    run = result.run
    return {
        "document": run.model_dump(mode="json"),
        "vendor_match": {
            "matched": result.vendor_match.matched,
            "vendor": result.vendor_match.vendor,
            "method": result.vendor_match.method,
            "score": result.vendor_match.score,
            "candidates": [candidate["legal_name"] for candidate in result.vendor_match.candidates],
        },
        "duplicate_of": result.duplicate_of,
        "reconciliation": {
            "ran": result.reconciliation.ran,
            "assessment": result.reconciliation.assessment,
            "root_cause": result.reconciliation.root_cause,
            "recommended_action": result.reconciliation.recommended_action,
            "reviewer_note": result.reconciliation.reviewer_note,
            "corrections": [
                {
                    "field": correction.field,
                    "suggested_value": correction.suggested_value,
                    "evidence": correction.evidence,
                    "reason": correction.reason,
                    "grounded": correction.grounded,
                }
                for correction in result.reconciliation.corrections
            ],
        },
    }


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/app.js", include_in_schema=False)
def app_js() -> FileResponse:
    return FileResponse(WEB_DIR / "app.js", media_type="application/javascript")


@app.get("/styles.css", include_in_schema=False)
def styles_css() -> FileResponse:
    return FileResponse(WEB_DIR / "styles.css", media_type="text/css")


# ---------------------------------------------------------------------------
# Status and reference data
# ---------------------------------------------------------------------------


@app.get("/api/status")
def status() -> dict[str, Any]:
    current = active_settings()
    return {
        "ocr_providers": describe_providers(current),
        "llm_providers": describe_llms(current),
        "selection": selection.describe(),
        "thresholds": {
            "auto_post_min_confidence": current.auto_post_min_confidence,
            "field_review_confidence": current.field_review_confidence,
            "amount_tolerance": current.amount_tolerance,
        },
        "policy": {
            "po_required_above": current.po_required_above,
            "expected_vat_rates": list(current.vat_rates),
            "contract_expiry_warning_days": current.contract_expiry_warning_days,
        },
        "rules": {
            "free_mode": current.rules_free_mode,
            "disabled": sorted(current.skipped_rules),
        },
        "url_fetch_enabled": current.allow_url_fetch,
        "max_document_bytes": current.max_document_bytes,
        "stats": DocumentStore(settings.db_path).stats(),
    }


class PolicyRequest(BaseModel):
    """Every field optional: the screen sends only what the user changed."""

    rules_free_mode: bool | None = None
    disabled_rules: list[str] | None = Field(
        default=None, description="Rule ids to switch off; replaces the current set"
    )
    auto_post_min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rule_severity: dict[str, str] | None = Field(
        default=None, description="rule id -> blocking | warning | info"
    )


def _settings_payload() -> dict[str, Any]:
    store = active_policy()
    current = active_settings()
    return {
        "policy": store.describe(selection.settings()),
        "rules": describe_rules(current.skipped_rules, store.severity_overrides()),
        "tunable": {
            # Read-only here: these are declared in .env because changing them
            # mid-flight would silently reinterpret documents already posted.
            "field_review_confidence": current.field_review_confidence,
            "amount_tolerance": current.amount_tolerance,
            "po_required_above": current.po_required_above,
            "expected_vat_rates": list(current.vat_rates),
            "unknown_vendor_blocks": sorted(current.vendor_blocking_types),
        },
    }


@app.get("/api/settings")
def get_settings_view() -> dict[str, Any]:
    """Current business policy and the full rule catalogue."""
    return _settings_payload()


@app.post("/api/settings")
def update_settings(request: PolicyRequest) -> dict[str, Any]:
    """Change the policy. Unknown rule names are rejected, not ignored."""
    try:
        active_policy().update(
            rules_free_mode=request.rules_free_mode,
            disabled_rules=request.disabled_rules,
            auto_post_min_confidence=request.auto_post_min_confidence,
            rule_severity=request.rule_severity,
        )
    except PolicyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _settings_payload()


@app.post("/api/settings/reset")
def reset_settings() -> dict[str, Any]:
    """Return to the policy declared in configuration."""
    active_policy().reset()
    return _settings_payload()


# ---------------------------------------------------------------------------
# Custom rules, written from a description
# ---------------------------------------------------------------------------


class RuleDraftRequest(BaseModel):
    prompt: str = Field(..., description="The check, in plain language")
    doc_types: list[str] = Field(default_factory=lambda: ["invoice"])


class RuleSaveRequest(BaseModel):
    rule: dict[str, Any] = Field(..., description="A compiled rule from /api/rules/draft")
    prompt: str = ""
    replace: bool = False


def _rule_store() -> CustomRuleStore:
    return CustomRuleStore(settings.db_path)


@app.get("/api/rules/custom")
def list_custom_rules() -> dict[str, Any]:
    return {
        "rules": [rule.to_json() for rule in _rule_store().list_rules()],
        "doc_types": sorted(DOCUMENT_SCHEMAS),
    }


@app.get("/api/rules/fields")
def rule_fields(doc_types: str = Query(default="invoice")) -> dict[str, Any]:
    """Variables a rule for these document types may reference."""
    wanted = tuple(part.strip() for part in doc_types.split(",") if part.strip())
    return {"doc_types": list(wanted), "fields": available_fields(wanted or ("invoice",))}


@app.post("/api/rules/draft")
def draft_rule(request: RuleDraftRequest) -> dict[str, Any]:
    """Compile a description into a rule. Nothing is saved until confirmed.

    The model runs here and only here. What it returns is data, shown back to
    the author in words before they accept it, and evaluated afterwards by the
    expression interpreter rather than by a model.
    """
    current = active_settings()
    llm, note = resolve_llm(current)
    result = compile_rule(request.prompt, tuple(request.doc_types), llm)
    if not result.ok:
        raise HTTPException(status_code=422, detail=result.error or note or "Could not compile the rule.")
    return {
        "rule": result.rule.to_json(),
        "explanation": result.rule.explanation,
        # Reading it back catches most mistakes; trying it on real documents
        # catches an inverted condition, which reads perfectly well.
        "preview": preview_rule(result.rule, current).to_json(),
    }


@app.post("/api/rules/preview")
def preview_custom_rule(request: RuleSaveRequest) -> dict[str, Any]:
    """Replay a rule over documents already processed. Writes nothing."""
    try:
        rule = build_rule(request.rule, prompt=request.prompt)
    except CustomRuleError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"rule": rule.to_json(), "preview": preview_rule(rule, active_settings()).to_json()}


@app.post("/api/rules/custom")
def save_custom_rule(request: RuleSaveRequest) -> dict[str, Any]:
    """Store a compiled rule after the author has read it back."""
    try:
        rule = build_rule(request.rule, prompt=request.prompt)
        _rule_store().save(rule, replace=request.replace)
    except CustomRuleError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"rule": rule.to_json()}


@app.post("/api/rules/custom/{rule_id}/enabled")
def toggle_custom_rule(rule_id: str, enabled: bool = Query(...)) -> dict[str, Any]:
    try:
        rule = _rule_store().set_enabled(rule_id, enabled)
    except CustomRuleError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"rule": rule.to_json()}


@app.delete("/api/rules/custom/{rule_id}")
def delete_custom_rule(rule_id: str) -> dict[str, Any]:
    if not _rule_store().delete(rule_id):
        raise HTTPException(status_code=404, detail=f"There is no rule called '{rule_id}'.")
    return {"deleted": rule_id}


@app.get("/api/archive")
def archive() -> dict[str, Any]:
    """Documents on disk, shelved by type and month."""
    return describe_archive(settings)


@app.post("/api/archive/rebuild")
def archive_rebuild() -> dict[str, Any]:
    """File every document already processed. Safe to run repeatedly."""
    return {**rebuild_archive(settings), "archive": describe_archive(settings)}


@app.get("/api/models")
def models() -> dict[str, Any]:
    """Every selectable OCR and agent model, with live availability."""
    return {**list_models(active_settings()), "selection": selection.describe()}


class ModelSelectionRequest(BaseModel):
    ocr_key: str | None = Field(default=None, description='"<provider>:<model>"')
    llm_key: str | None = Field(default=None, description='"<provider>:<model>"')


@app.post("/api/models/select")
def select_models(request: ModelSelectionRequest) -> dict[str, Any]:
    """Switch the OCR and/or agent model for subsequent documents."""
    try:
        selection.select(ocr_key=request.ocr_key, llm_key=request.llm_key)
    except SelectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    current = active_settings()
    describe = selection.describe()

    # Report immediately whether the new choice can actually run.
    ocr_row = next(
        (row for row in describe_providers(current) if row["key"] == describe["ocr_provider"]), None
    )
    llm_row = next(
        (row for row in describe_llms(current) if row["key"] == describe["llm_provider"]), None
    )
    return {
        "selection": describe,
        "ocr_available": bool(ocr_row and ocr_row["available"]),
        "ocr_reason": (ocr_row or {}).get("reason", ""),
        "llm_available": bool(llm_row and llm_row["available"]),
        "llm_reason": (llm_row or {}).get("reason", ""),
    }


@app.post("/api/models/reset")
def reset_models() -> dict[str, Any]:
    """Return to the models declared in configuration."""
    selection.reset()
    return {"selection": selection.describe()}


@app.get("/api/schemas")
def schemas() -> dict[str, Any]:
    return {
        doc_type: {
            "display_name": schema.display_name,
            "description": schema.description,
            "record_kind": schema.record_kind,
            "fields": [
                {
                    "name": spec.name,
                    "label": spec.label,
                    "type": spec.type,
                    "required": spec.required,
                    "critical": spec.critical,
                    "description": spec.description,
                    "choices": list(spec.choices),
                }
                for spec in schema.fields
            ],
        }
        for doc_type, schema in DOCUMENT_SCHEMAS.items()
    }


@app.get("/api/samples")
def samples() -> list[dict[str, Any]]:
    available = {path.stem: path for path in settings.sample_dir.glob("*")}
    result = []
    for doc in SAMPLE_DOCUMENTS:
        scan = settings.sample_dir / f"{doc.doc_id}.jpg"
        pdf = settings.sample_dir / f"{doc.doc_id}.pdf"
        path = scan if scan.exists() else pdf
        result.append(
            {
                "doc_id": doc.doc_id,
                "doc_type": doc.doc_type,
                "title": doc.title,
                "demonstrates": doc.demonstrates,
                "language": doc.language,
                "scanned": doc.degrade > 0,
                "available": path.exists(),
                "filename": path.name if path.exists() else None,
            }
        )
    return result


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


@app.post("/api/documents/upload")
def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > settings.max_document_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File is {len(content) / 1e6:.1f} MB, above the "
            f"{settings.max_document_bytes / 1e6:.0f} MB limit.",
        )
    result = _pipeline().process_bytes(
        content,
        file.filename or "upload",
        source_kind="upload",
        content_type=file.content_type,
    )
    return _serialize_result(result)


# ---------------------------------------------------------------------------
# Batch upload
# ---------------------------------------------------------------------------


@app.post("/api/documents/batch")
def upload_batch(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Accept many documents at once and queue them.

    Returns as soon as the files are on disk. Twenty documents is ten minutes of
    work, which is not something to hold a request open for - the browser polls
    /api/jobs instead.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"{len(files)} files at once; the limit is {MAX_BATCH_FILES}.",
        )

    queue = active_queue()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    inbox = settings.data_dir / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)

    for upload in files:
        name = upload.filename or "upload"
        content = upload.file.read()
        if not content:
            rejected.append({"filename": name, "reason": "the file is empty"})
            continue
        if len(content) > settings.max_document_bytes:
            rejected.append({
                "filename": name,
                "reason": f"{len(content) / 1e6:.1f} MB is above the "
                          f"{settings.max_document_bytes / 1e6:.0f} MB limit",
            })
            continue
        if not is_supported(Path(name)):
            rejected.append({"filename": name, "reason": "unsupported file type"})
            continue

        # Written under a unique name: two files called "invoice.pdf" in one
        # batch are two different documents.
        staged = inbox / f"{uuid.uuid4().hex[:8]}-{Path(name).name}"
        staged.write_bytes(content)
        accepted.append(queue.submit(staged, name).to_json())

    if not accepted and rejected:
        raise HTTPException(status_code=400, detail=rejected[0]["reason"])
    return {"queued": accepted, "rejected": rejected, "summary": queue.summary()}


@app.get("/api/jobs")
def list_jobs(limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
    """Queue state, polled by the browser while a batch runs."""
    queue = active_queue()
    return {
        "jobs": [job.to_json() for job in queue.list_jobs(limit)],
        "summary": queue.summary(),
    }


@app.post("/api/jobs/clear")
def clear_jobs() -> dict[str, Any]:
    """Remove finished jobs from the list. The documents themselves stay."""
    return {"cleared": active_queue().clear_finished(), "summary": active_queue().summary()}


@app.post("/api/documents/url")
def ingest_url(request: UrlIngestRequest) -> dict[str, Any]:
    if not settings.allow_url_fetch:
        raise HTTPException(status_code=403, detail="URL ingestion is disabled by configuration.")
    try:
        fetched = fetch_url(request.url, settings.max_document_bytes)
    except FetchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = _pipeline().process_bytes(
        fetched.content,
        fetched.filename,
        source_kind="url",
        origin=fetched.final_url,
        content_type=fetched.content_type,
    )
    return _serialize_result(result)


@app.post("/api/documents/sample/{doc_id}")
def ingest_sample(doc_id: str) -> dict[str, Any]:
    scan = settings.sample_dir / f"{doc_id}.jpg"
    pdf = settings.sample_dir / f"{doc_id}.pdf"
    path = scan if scan.exists() else pdf
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Sample '{doc_id}' has not been generated.")
    return _serialize_result(_pipeline().process_file(path, source_kind="sample"))


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


@app.get("/api/documents")
def list_documents(
    decision: str | None = Query(default=None),
    doc_type: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict[str, Any]]:
    return DocumentStore(settings.db_path).list(decision=decision, doc_type=doc_type, limit=limit)


@app.get("/api/documents/{document_id}")
def get_document(document_id: str) -> dict[str, Any]:
    store = DocumentStore(settings.db_path)
    run = store.get(document_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} was not found.")

    master = MasterDataRepository(settings.db_path)
    vendor_name = run.value("vendor_name") or run.value("merchant_name") or run.value("applicant_name")
    tax_id = run.value("vendor_tax_id") or run.value("merchant_tax_id") or run.value("tax_id")
    match = master.find_vendor(str(vendor_name) if vendor_name else None, str(tax_id) if tax_id else None)

    return {
        "document": run.model_dump(mode="json"),
        "vendor_match": {
            "matched": match.matched,
            "vendor": match.vendor,
            "method": match.method,
            "score": match.score,
            "candidates": [candidate["legal_name"] for candidate in match.candidates],
        },
        "audit": RecordStore(settings.db_path).audit_for(document_id),
        "corrections": CorrectionStore(settings.db_path).for_document(document_id),
    }


@app.get("/api/documents/{document_id}/page/{page_number}")
def get_page_image(document_id: str, page_number: int) -> FileResponse:
    run = DocumentStore(settings.db_path).get(document_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} was not found.")
    if not 1 <= page_number <= len(run.page_images):
        raise HTTPException(status_code=404, detail=f"Page {page_number} does not exist.")

    path = Path(run.page_images[page_number - 1])
    # Only ever serve from inside the configured page directory.
    try:
        path.resolve().relative_to(settings.page_dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Refusing to serve a file outside the page store.") from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail="The rendered page is no longer on disk.")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/documents/{document_id}/review")
def review_document(document_id: str, request: ReviewRequest) -> dict[str, Any]:
    try:
        outcome = apply_review(
            active_settings(),
            document_id,
            edits=request.edits,
            action=request.action,
            reviewer=request.reviewer,
            note=request.note,
        )
    except ReviewError as exc:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    return {
        "document": outcome.run.model_dump(mode="json"),
        "record_id": outcome.record_id,
        "applied": list(outcome.applied),
        "rejected_edits": list(outcome.rejected_edits),
        "message": outcome.message,
    }


# ---------------------------------------------------------------------------
# Records and master data
# ---------------------------------------------------------------------------


@app.get("/api/records")
def list_records(
    record_kind: str | None = Query(default=None), limit: int = Query(default=200, ge=1, le=1000)
) -> list[dict[str, Any]]:
    return RecordStore(settings.db_path).list(record_kind=record_kind, limit=limit)


@app.get("/api/records/summary")
def records_summary() -> dict[str, Any]:
    """The ledger split by record kind, with the shape of each one."""
    store = RecordStore(settings.db_path)
    kinds = store.summary_by_kind()
    return {
        "kinds": [
            {**kind, "top_counterparties": store.top_counterparties(kind["record_kind"], limit=5)}
            for kind in kinds
        ],
        "total_records": sum(kind["count"] for kind in kinds),
        # What each kind of record means, so the page can say it rather than
        # showing a word from the database and leaving the reader to guess.
        "labels": {
            "payable": {
                "title": "Payables",
                "blurb": "Supplier invoices accepted for payment.",
                "source": "invoice",
            },
            "expense": {
                "title": "Expenses",
                "blurb": "Money already spent, evidenced by a receipt.",
                "source": "receipt",
            },
            "commitment": {
                "title": "Commitments",
                "blurb": "Purchase orders — spend committed but not yet billed.",
                "source": "purchase_order",
            },
            "contract": {
                "title": "Contracts",
                "blurb": "Agreements on file, with their term and value.",
                "source": "contract",
            },
            "submission": {
                "title": "Submissions",
                "blurb": "Onboarding forms and applications received.",
                "source": "form",
            },
            "generic": {
                "title": "Other",
                "blurb": "Documents that did not match a known type.",
                "source": "unknown",
            },
        },
    }


@app.get("/api/records/{record_id}")
def get_record(record_id: str) -> dict[str, Any]:
    record = RecordStore(settings.db_path).get(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Record {record_id} was not found.")
    return record


@app.get("/api/master/vendors")
def list_vendors() -> list[dict[str, Any]]:
    return MasterDataRepository(settings.db_path).list_vendors()


class NewVendorRequest(BaseModel):
    legal_name: str = Field(..., min_length=1)
    tax_id: str | None = None
    bank_account: str | None = None
    aliases: str = ""
    status: str = Field(default="active", pattern="^(active|on_hold|blocked)$")
    payment_terms: str | None = None
    currency: str = "THB"


@app.post("/api/master/vendors", status_code=201)
def add_vendor(request: NewVendorRequest) -> dict[str, Any]:
    """Register a vendor so its documents stop being blocked as unknown."""
    try:
        return MasterDataRepository(settings.db_path).add_vendor(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/master/purchase-orders")
def list_purchase_orders() -> list[dict[str, Any]]:
    return MasterDataRepository(settings.db_path).list_purchase_orders()


@app.get("/api/master/contracts")
def list_contracts() -> list[dict[str, Any]]:
    return MasterDataRepository(settings.db_path).list_contracts()


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    documents = DocumentStore(settings.db_path)
    records = RecordStore(settings.db_path)
    corrections = CorrectionStore(settings.db_path)
    stats = documents.stats()
    return {
        **stats,
        "totals_by_currency": records.totals_by_currency(),
        "corrections": corrections.total(),
        "error_hotspots": corrections.error_hotspots(),
    }
