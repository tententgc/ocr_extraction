"""Core domain models shared by every pipeline stage.

All models are frozen: a stage never mutates the object it was handed, it returns
a new one. That keeps the pipeline trace an accurate, replayable history.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DocumentType = Literal["invoice", "receipt", "purchase_order", "contract", "form", "unknown"]
Severity = Literal["blocking", "warning", "info"]
Decision = Literal["auto_post", "needs_review", "rejected"]
FieldSource = Literal["llm", "rule", "derived", "human", "master_data"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Frozen(BaseModel):
    """Immutable base: `.evolve()` returns a modified copy, never mutates."""

    model_config = ConfigDict(frozen=True)

    def evolve(self, **changes: Any):
        return self.model_copy(update=changes)


class OcrPage(Frozen):
    """One page of recognised text."""

    page_number: int
    text: str
    provider: str
    image_path: str | None = None
    duration_ms: int = 0
    warnings: tuple[str, ...] = ()


class OcrResult(Frozen):
    """Recognised text for a whole document."""

    pages: tuple[OcrPage, ...] = ()
    provider: str = "none"
    duration_ms: int = 0
    degraded: bool = False
    notes: tuple[str, ...] = ()

    @property
    def full_text(self) -> str:
        return "\n\n".join(
            f"<!-- page {page.page_number} -->\n{page.text}" for page in self.pages
        )

    @property
    def char_count(self) -> int:
        return sum(len(page.text) for page in self.pages)


class ExtractedField(Frozen):
    """A single extracted value plus everything needed to trust or reject it.

    `evidence` is a verbatim span the model claims to have read the value from.
    `grounded` records whether that span was actually found in the OCR text -
    an ungrounded field is treated as a hallucination and never auto-posted.
    """

    name: str
    label: str = ""
    value: Any = None
    raw_value: str | None = None
    evidence: str | None = None
    page: int | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    grounded: bool = False
    source: FieldSource = "llm"
    notes: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return self.value in (None, "", [], {})

    def with_note(self, note: str) -> ExtractedField:
        return self.evolve(notes=self.notes + (note,))


class LineItem(Frozen):
    """A row from an invoice / PO table."""

    description: str = ""
    quantity: float | None = None
    unit_price: float | None = None
    amount: float | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SuggestedFix(Frozen):
    """A concrete correction a reviewer can accept or ignore.

    A failing rule knows something is wrong. Knowing which of the values on the
    page is the wrong one, and what it should say instead, is a further step -
    and it is the step that turns a reviewer's five minutes of re-reading into a
    glance and a click.

    A suggestion is never applied on its own. It is offered, a person accepts
    it, and the full rule set runs again over what they accepted, exactly as if
    they had typed the value themselves. `confidence` is what separates a value
    two independent relations agree on from one a single relation implies, and
    the UI orders and words them by it.
    """

    field: str
    current: Any = None
    suggested: Any = None
    # The whole argument, shown inside the suggestion box.
    reason: str = ""
    # One clause of it, for the failure message above the box. Without this the
    # rule's message and the suggestion print the same three sentences one after
    # the other, and the reviewer reads neither.
    headline: str = ""
    # Where the replacement came from: arithmetic on the document's own numbers,
    # a record we already hold, something read off the page, or the agent.
    source: Literal["arithmetic", "master_data", "prior_record", "page", "policy", "agent"] = (
        "arithmetic"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    def to_json(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "current": self.current,
            "suggested": self.suggested,
            "reason": self.reason,
            "headline": self.headline,
            "source": self.source,
            "confidence": round(self.confidence, 2),
        }


class RuleFinding(Frozen):
    """Outcome of one business rule."""

    rule_id: str
    passed: bool
    severity: Severity
    message: str
    fields: tuple[str, ...] = ()
    detail: dict[str, Any] = Field(default_factory=dict)
    # What to do about it, when the rule can work that out. Empty is the honest
    # answer for a check that can tell something is wrong but not what is right.
    fixes: tuple[SuggestedFix, ...] = ()

    @property
    def is_blocking(self) -> bool:
        return not self.passed and self.severity == "blocking"


class TraceStep(Frozen):
    """One agent/pipeline step, surfaced in the UI so the run is auditable."""

    step: str
    status: Literal["ok", "warning", "error", "skipped"] = "ok"
    summary: str = ""
    duration_ms: int = 0
    detail: dict[str, Any] = Field(default_factory=dict)


class SourceInfo(Frozen):
    """Where the document came from."""

    kind: Literal["upload", "url", "sample", "path"] = "upload"
    filename: str = "document"
    origin: str | None = None
    content_type: str | None = None
    byte_size: int = 0
    sha256: str = ""


class DocumentRun(Frozen):
    """Everything known about one document after (part of) the pipeline ran."""

    document_id: str
    source: SourceInfo
    created_at: datetime = Field(default_factory=_utc_now)

    blob_path: str | None = None
    page_images: tuple[str, ...] = ()

    doc_type: DocumentType = "unknown"
    doc_type_confidence: float = 0.0

    ocr: OcrResult = Field(default_factory=OcrResult)
    fields: tuple[ExtractedField, ...] = ()
    line_items: tuple[LineItem, ...] = ()
    findings: tuple[RuleFinding, ...] = ()
    trace: tuple[TraceStep, ...] = ()

    confidence: float = 0.0
    decision: Decision = "needs_review"
    decision_reasons: tuple[str, ...] = ()
    record_id: str | None = None

    error: str | None = None

    # --- convenience accessors --------------------------------------------
    def field(self, name: str) -> ExtractedField | None:
        return next((f for f in self.fields if f.name == name), None)

    def value(self, name: str, default: Any = None) -> Any:
        found = self.field(name)
        return default if found is None or found.is_empty else found.value

    def field_map(self) -> dict[str, Any]:
        return {f.name: f.value for f in self.fields}

    @property
    def blocking_findings(self) -> tuple[RuleFinding, ...]:
        return tuple(f for f in self.findings if f.is_blocking)

    @property
    def failed_findings(self) -> tuple[RuleFinding, ...]:
        return tuple(f for f in self.findings if not f.passed)

    @property
    def suggested_fixes(self) -> tuple[SuggestedFix, ...]:
        """Every correction the failing checks propose, best per field.

        Gathered across rules rather than per rule, because two checks failing
        on the same misread number should put one suggestion in front of the
        reviewer, not two that say the same thing at different confidences.
        """
        from docuflow.rules.repair import best_of

        return best_of(fix for finding in self.failed_findings for fix in finding.fixes)

    def with_step(self, step: TraceStep) -> DocumentRun:
        return self.evolve(trace=self.trace + (step,))

    def with_fields(self, fields: tuple[ExtractedField, ...]) -> DocumentRun:
        return self.evolve(fields=fields)
