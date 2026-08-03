"""Everything a business rule is allowed to look at."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from docuflow.masterdata.repository import MasterDataRepository, VendorMatch
from docuflow.models import ExtractedField, LineItem
from docuflow.schemas.registry import DocumentSchema


@dataclass(frozen=True)
class RuleContext:
    """Read-only view of one document under validation."""

    document_id: str
    doc_type: str
    schema: DocumentSchema
    fields: tuple[ExtractedField, ...]
    line_items: tuple[LineItem, ...]
    master: MasterDataRepository
    vendor_match: VendorMatch
    ocr_text: str
    ocr_degraded: bool
    amount_tolerance: float
    today: date
    # Populated by the records repository: prior postings that may collide.
    duplicate_of: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str) -> ExtractedField | None:
        return next((item for item in self.fields if item.name == name), None)

    def value(self, name: str, default: Any = None) -> Any:
        found = self.get(name)
        return default if found is None or found.is_empty else found.value

    def is_grounded(self, name: str) -> bool:
        found = self.get(name)
        return bool(found and found.grounded and not found.is_empty)

    def confidence(self, name: str) -> float:
        found = self.get(name)
        return found.confidence if found else 0.0

    @property
    def vendor(self) -> dict[str, Any] | None:
        return self.vendor_match.vendor
