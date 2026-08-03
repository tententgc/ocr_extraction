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
    today: date

    # --- finance policy, supplied from configuration ------------------------
    # These are business decisions, not code. Rules read them from here so the
    # thresholds can be changed in .env without editing a rule module.
    amount_tolerance: float = 0.02
    po_required_above: float = 50_000.0
    expected_vat_rates: tuple[float, ...] = (0.0, 7.0)
    contract_expiry_warning_days: int = 90
    max_contract_years: int = 25
    future_date_tolerance_days: int = 2
    field_review_confidence: float = 0.75
    # Document types where an unregistered counterparty blocks posting.
    # Invoices and purchase orders commit money to a party, so an unknown one
    # must stop. A receipt records money already spent at a merchant who was
    # never going to be an AP vendor, and an onboarding form exists precisely
    # because the applicant is *not* registered yet - blocking either is wrong.
    unknown_vendor_blocks: frozenset[str] = frozenset({"invoice", "purchase_order"})
    # Populated by the records repository: prior postings that may collide.
    duplicate_of: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def policy_from(cls, settings) -> dict[str, Any]:
        """The policy keyword arguments carried by `settings`."""
        return {
            "amount_tolerance": settings.amount_tolerance,
            "po_required_above": settings.po_required_above,
            "expected_vat_rates": settings.vat_rates,
            "contract_expiry_warning_days": settings.contract_expiry_warning_days,
            "max_contract_years": settings.max_contract_years,
            "future_date_tolerance_days": settings.future_date_tolerance_days,
            "field_review_confidence": settings.field_review_confidence,
            "unknown_vendor_blocks": settings.vendor_blocking_types,
        }

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
