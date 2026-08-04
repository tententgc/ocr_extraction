"""Turn a document into the master record it is asking for.

A vendor onboarding form exists to create a vendor. A purchase order creates a
commitment. A signed contract belongs in the contract register. Until now the
system could only ever check documents *against* master data, so a reviewer
looking at an invoice from an unregistered supplier had to read the details off
the screen, retype them somewhere else, come back and re-check.

The values are already extracted, already grounded in quotes from the page.
This offers them, with their evidence, as a record a person can create in one
step.

**A person creates it. Never the pipeline, never the agent.** That is the whole
security argument for `vendor_known`: if the system registered unknown suppliers
by itself, an invoice from a company that does not exist would create the
company and then pass the check written to catch it. So this module proposes,
shows its working, and waits.

What it creates is also marked. `source_document_id` records that this row came
off a page we read rather than out of the ERP, because those are not equally
trustworthy and a register that cannot tell them apart is worse than one that
can.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from docuflow.config import Settings
from docuflow.masterdata.importer import BUILDERS
from docuflow.masterdata.repository import MasterDataRepository
from docuflow.models import DocumentRun
from docuflow.normalize.identifiers import thai_tax_id_checksum_ok
from docuflow.store.documents import DocumentStore
from docuflow.store.records import RecordStore

logger = logging.getLogger(__name__)

KINDS = ("vendors", "purchase_orders", "contracts")


class RegistrationError(RuntimeError):
    """The document cannot become this kind of master record."""


# Which extracted field feeds which master column, per document type. A receipt
# calls the counterparty `merchant_name`; an onboarding form calls it
# `applicant_name`. Same column, three names, and this is the one place that
# knows it.
VENDOR_FROM: dict[str, dict[str, str]] = {
    "invoice": {
        "legal_name": "vendor_name",
        "tax_id": "vendor_tax_id",
        "bank_account": "bank_account",
        "currency": "currency",
        "payment_terms": "payment_terms",
    },
    "receipt": {
        "legal_name": "merchant_name",
        "tax_id": "merchant_tax_id",
        "currency": "currency",
    },
    "purchase_order": {
        "legal_name": "vendor_name",
        "currency": "currency",
    },
    # The onboarding form is the strongest case: it exists to be turned into
    # exactly this record.
    "form": {
        "legal_name": "applicant_name",
        "tax_id": "tax_id",
        "bank_account": "bank_account",
    },
}

PURCHASE_ORDER_FROM = {
    "po_number": "po_number",
    "po_date": "po_date",
    "amount": "total_amount",
    "currency": "currency",
}

CONTRACT_FROM = {
    "contract_number": "contract_number",
    "party_a": "party_a",
    "party_b": "party_b",
    "effective_date": "effective_date",
    "expiry_date": "expiry_date",
    "value": "contract_value",
    "currency": "currency",
}

# Which master record each document type can become.
OFFERS: dict[str, tuple[str, ...]] = {
    "invoice": ("vendors",),
    "receipt": ("vendors",),
    "form": ("vendors",),
    "purchase_order": ("vendors", "purchase_orders"),
    "contract": ("contracts",),
}


@dataclass(frozen=True)
class Proposal:
    """What this document says a master record should be, and how it knows."""

    kind: str
    title: str
    values: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, str] = field(default_factory=dict)
    ungrounded: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    blocked: str = ""

    @property
    def available(self) -> bool:
        return not self.blocked

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "values": self.values,
            "evidence": self.evidence,
            "ungrounded": list(self.ungrounded),
            "warnings": list(self.warnings),
            "blocked": self.blocked,
            "available": self.available,
        }


def _collect(run: DocumentRun, mapping: dict[str, str]) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """Read the mapped fields off the run, keeping each one's evidence."""
    values: dict[str, Any] = {}
    evidence: dict[str, str] = {}
    ungrounded: list[str] = []
    by_name = {item.name: item for item in run.fields}

    for column, field_name in mapping.items():
        item = by_name.get(field_name)
        if item is None or item.is_empty:
            continue
        values[column] = item.value
        if item.evidence:
            evidence[column] = item.evidence
        if not item.grounded:
            ungrounded.append(column)
    return values, evidence, ungrounded


def _vendor_proposal(run: DocumentRun, master: MasterDataRepository) -> Proposal:
    mapping = VENDOR_FROM.get(run.doc_type)
    if not mapping:
        return Proposal("vendors", "Register this counterparty", blocked="This document type has no counterparty to register.")

    values, evidence, ungrounded = _collect(run, mapping)
    name = values.get("legal_name")
    if not name:
        return Proposal("vendors", "Register this counterparty",
                        blocked="No counterparty name was read from this document.")

    existing = master.find_vendor(str(name), str(values.get("tax_id") or "") or None)
    if existing.matched:
        return Proposal(
            "vendors", "Register this counterparty",
            blocked=f"Already registered as {existing.vendor['legal_name']} ({existing.vendor_id}).",
        )

    warnings = []
    tax_id = str(values.get("tax_id") or "")
    if tax_id and len(tax_id) == 13 and tax_id.isdigit() and not thai_tax_id_checksum_ok(tax_id):
        warnings.append(f"The tax ID {tax_id} fails its check digit — confirm it before registering.")
    if "bank_account" in ungrounded:
        # The value a fraudulent invoice most wants into the master file.
        warnings.append(
            "The bank account could not be verified against the page text. Check it against "
            "the document image, and against something other than this document, before saving it."
        )
    elif values.get("bank_account"):
        warnings.append(
            "Registering the bank account from an invoice is how invoice fraud succeeds. "
            "Confirm it by a channel this document did not arrive through."
        )
    if existing.candidates:
        similar = ", ".join(c["legal_name"] for c in existing.candidates[:3])
        warnings.append(f"Similar names already registered (not a match): {similar}.")

    return Proposal(
        kind="vendors",
        title="Register this counterparty",
        values=values,
        evidence=evidence,
        ungrounded=tuple(ungrounded),
        warnings=tuple(warnings),
    )


def _purchase_order_proposal(run: DocumentRun, master: MasterDataRepository) -> Proposal:
    values, evidence, ungrounded = _collect(run, PURCHASE_ORDER_FROM)
    number = values.get("po_number")
    if not number:
        return Proposal("purchase_orders", "Add to the purchase order register",
                        blocked="No purchase order number was read from this document.")
    if master.get_purchase_order(str(number)):
        return Proposal("purchase_orders", "Add to the purchase order register",
                        blocked=f"{number} is already in the purchase order register.")

    vendor_name = run.value("vendor_name")
    match = master.find_vendor(str(vendor_name) if vendor_name else None, None)
    if not match.matched:
        return Proposal(
            "purchase_orders", "Add to the purchase order register",
            blocked=f"{vendor_name or 'The vendor'} is not registered yet. Register the counterparty first.",
        )

    values["vendor_id"] = match.vendor_id
    return Proposal(
        kind="purchase_orders",
        title="Add to the purchase order register",
        values=values,
        evidence=evidence,
        ungrounded=tuple(ungrounded),
    )


def _contract_proposal(run: DocumentRun, master: MasterDataRepository) -> Proposal:
    values, evidence, ungrounded = _collect(run, CONTRACT_FROM)
    number = values.get("contract_number")
    if not number:
        return Proposal("contracts", "Add to the contract register",
                        blocked="No contract number was read from this document.")
    if master.get_contract(str(number)):
        return Proposal("contracts", "Add to the contract register",
                        blocked=f"{number} is already in the contract register.")

    party_b = values.get("party_b")
    match = master.find_vendor(str(party_b) if party_b else None, None)
    if match.matched:
        values["vendor_id"] = match.vendor_id
    return Proposal(
        kind="contracts",
        title="Add to the contract register",
        values=values,
        evidence=evidence,
        ungrounded=tuple(ungrounded),
    )


_PROPOSERS = {
    "vendors": _vendor_proposal,
    "purchase_orders": _purchase_order_proposal,
    "contracts": _contract_proposal,
}


def proposals(run: DocumentRun, master: MasterDataRepository) -> list[Proposal]:
    """Every master record this document could become, offered or blocked."""
    return [_PROPOSERS[kind](run, master) for kind in OFFERS.get(run.doc_type, ())]


def register(
    settings: Settings,
    document_id: str,
    kind: str,
    overrides: dict[str, Any] | None = None,
    actor: str = "reviewer",
) -> dict[str, Any]:
    """Create the master record a person has confirmed, from the document's values."""
    if kind not in KINDS:
        raise RegistrationError(f"Cannot register a {kind!r}. Expected one of {', '.join(KINDS)}.")

    documents = DocumentStore(settings.db_path)
    master = MasterDataRepository(settings.db_path)
    run = documents.get(document_id)
    if run is None:
        raise RegistrationError(f"Document {document_id} was not found.")
    if kind not in OFFERS.get(run.doc_type, ()):
        raise RegistrationError(f"A {run.doc_type.replace('_', ' ')} cannot become a {kind[:-1]} record.")

    proposal = _PROPOSERS[kind](run, master)
    if proposal.blocked:
        raise RegistrationError(proposal.blocked)

    # What the reviewer edited on screen wins over what was read, and the whole
    # lot goes through the CSV importer's validators - a value typed here and a
    # value imported from a file must be held to the same standard.
    merged = {**proposal.values, **{k: v for k, v in (overrides or {}).items() if v not in (None, "")}}
    try:
        row = BUILDERS[kind]({key: _as_text(value) for key, value in merged.items()}, master)
    except ValueError as exc:
        raise RegistrationError(str(exc)) from exc

    row["source_document_id"] = document_id
    upsert = {"vendors": master.upsert_vendor, "purchase_orders": master.upsert_purchase_order,
              "contracts": master.upsert_contract}[kind]
    outcome = upsert(row)

    RecordStore(settings.db_path).log(
        "master_record_created",
        actor=actor,
        document_id=document_id,
        kind=kind,
        outcome=outcome,
        values={key: str(value) for key, value in row.items() if value is not None},
    )
    return {
        "kind": kind,
        "outcome": outcome,
        "row": row,
        "message": f"{_label(kind, row)} added to the master file from this document.",
    }


def _as_text(value: Any) -> str:
    """The builders parse text, because that is what a CSV cell is."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _label(kind: str, row: dict[str, Any]) -> str:
    return str(
        row.get("legal_name") or row.get("po_number") or row.get("contract_number") or kind
    )
