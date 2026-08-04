"""Load vendors, purchase orders and contracts from CSV.

Master data belongs to the ERP, the procurement system and the contract
register - not here. This system holds a copy so it has something to check
documents against, and a CSV export is the lowest common denominator every one
of those systems can produce.

Two decisions shape the whole module.

**Headers are matched loosely, values are validated strictly.** A file exported
from SAP says "Vendor Name"; one typed by hand says "name". Refusing either
teaches the user nothing, so headers are normalised and aliased. What is *in*
the columns gets no such generosity: a date that does not parse, an amount that
is not a number, a status outside the allowed set - each is refused, by row
number, with the value that caused it. Master data is the thing every rule is
checked against; a silently mangled row here mis-judges every document that
mentions it, forever.

**A bad row is skipped, not fatal.** A hundred-row file with two bad rows
imports ninety-eight and reports the two. Aborting the lot punishes the user for
the exporter's mistakes.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from docuflow.masterdata.repository import MasterDataRepository
from docuflow.normalize.amounts import parse_money
from docuflow.normalize.dates import parse_date
from docuflow.normalize.identifiers import parse_tax_id

logger = logging.getLogger(__name__)

KINDS = ("vendors", "purchase_orders", "contracts")
VENDOR_STATUSES = ("active", "on_hold", "blocked")
PO_STATUSES = ("open", "closed", "cancelled")
MAX_ROWS = 20_000


class MasterImportError(ValueError):
    """The file itself could not be read. Individual bad rows do not raise."""


@dataclass(frozen=True)
class RowError:
    row: int
    message: str

    def to_json(self) -> dict[str, Any]:
        return {"row": self.row, "message": self.message}


@dataclass(frozen=True)
class ImportReport:
    kind: str
    created: int = 0
    updated: int = 0
    errors: tuple[RowError, ...] = field(default_factory=tuple)

    @property
    def imported(self) -> int:
        return self.created + self.updated

    @property
    def summary(self) -> str:
        parts = []
        if self.created:
            parts.append(f"{self.created} added")
        if self.updated:
            parts.append(f"{self.updated} updated")
        if self.errors:
            parts.append(f"{len(self.errors)} row(s) skipped")
        label = self.kind.replace("_", " ")
        return f"{label}: " + (", ".join(parts) if parts else "nothing to import") + "."

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "created": self.created,
            "updated": self.updated,
            "imported": self.imported,
            "errors": [error.to_json() for error in self.errors],
            "summary": self.summary,
        }


# --------------------------------------------------------------------------- #
# Headers                                                                       #
# --------------------------------------------------------------------------- #

def _normalise_header(name: str) -> str:
    return name.strip().lower().lstrip("﻿").replace("-", "_").replace(" ", "_")


# What other systems call the same column. The left side is what we store.
HEADER_ALIASES: dict[str, tuple[str, ...]] = {
    "legal_name": ("name", "vendor_name", "supplier_name", "vendor", "supplier", "company", "company_name"),
    "tax_id": ("tax_number", "taxid", "vat_number", "vat_id", "tin", "tax_registration_number"),
    "bank_account": ("account", "account_number", "bank_account_number", "iban"),
    "aliases": ("alias", "other_names", "trading_names", "also_known_as"),
    "payment_terms": ("terms", "payment_term"),
    "vendor_id": ("supplier_id", "vendor_code", "supplier_code", "vendor_no", "supplier_no"),
    "po_number": ("po", "po_no", "purchase_order", "purchase_order_number", "order_number"),
    "po_date": ("date", "order_date", "issue_date"),
    "amount": ("value", "total", "po_amount", "net_value", "total_amount"),
    "description": ("desc", "item", "details", "text"),
    "contract_number": ("contract", "contract_no", "agreement_number", "agreement_no"),
    "effective_date": ("start_date", "valid_from", "from"),
    "expiry_date": ("end_date", "valid_to", "to", "expiration_date"),
    "party_a": ("buyer", "client", "customer"),
    "party_b": ("supplier", "counterparty", "provider"),
    "currency": ("ccy", "curr", "currency_code"),
    "status": ("state",),
}
_ALIAS_LOOKUP = {alias: canonical for canonical, aliases in HEADER_ALIASES.items() for alias in aliases}


def _canonical(header: str) -> str:
    normalised = _normalise_header(header)
    return _ALIAS_LOOKUP.get(normalised, normalised)


def read_rows(content: bytes | str) -> list[dict[str, str]]:
    """Decode and parse the file. Raises only when the file is unusable."""
    text = content.decode("utf-8-sig", errors="replace") if isinstance(content, bytes) else content
    if not text.strip():
        raise MasterImportError("The file is empty.")

    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel  # a single column has no delimiter to find

    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        raise MasterImportError("The file has no header row.")

    rows = []
    for index, row in enumerate(reader):
        if index >= MAX_ROWS:
            raise MasterImportError(f"Refusing to import more than {MAX_ROWS:,} rows in one file.")
        rows.append({_canonical(key): (value or "").strip() for key, value in row.items() if key})
    if not rows:
        raise MasterImportError("The file has a header but no rows.")
    return rows


# --------------------------------------------------------------------------- #
# Per-kind row builders                                                         #
# --------------------------------------------------------------------------- #

def _require(row: dict[str, str], column: str) -> str:
    value = row.get(column, "")
    if not value:
        raise ValueError(f"'{column}' is required and is empty")
    return value


def _optional_date(row: dict[str, str], column: str) -> str | None:
    raw = row.get(column, "")
    if not raw:
        return None
    # Master data is typed by people and exported by systems, so both
    # 2026-07-31 and 31/07/2026 arrive; an unparseable one is refused rather
    # than stored as text no rule can compare.
    parsed = parse_date(raw)
    if parsed is None:
        raise ValueError(f"'{column}' is not a date I can read: {raw!r}")
    return parsed


def _optional_amount(row: dict[str, str], column: str) -> float | None:
    raw = row.get(column, "")
    if not raw:
        return None
    amount = parse_money(raw)
    if amount is None:
        raise ValueError(f"'{column}' is not a number: {raw!r}")
    return amount


def _choice(row: dict[str, str], column: str, allowed: tuple[str, ...], default: str) -> str:
    raw = (row.get(column, "") or default).strip().lower().replace(" ", "_").replace("-", "_")
    if raw not in allowed:
        raise ValueError(f"'{column}' must be one of {', '.join(allowed)}, not {raw!r}")
    return raw


def _vendor_row(row: dict[str, str], master: MasterDataRepository) -> dict[str, Any]:
    tax_id = row.get("tax_id", "")
    return {
        "legal_name": _require(row, "legal_name"),
        # Kept as given when it is not a Thai 13-digit number: foreign suppliers
        # have tax numbers too, and refusing them would exclude every one.
        "tax_id": (parse_tax_id(tax_id) or tax_id) if tax_id else None,
        "bank_account": row.get("bank_account") or None,
        "aliases": row.get("aliases", ""),
        "status": _choice(row, "status", VENDOR_STATUSES, "active"),
        "payment_terms": row.get("payment_terms") or None,
        "currency": (row.get("currency") or "THB").upper(),
        "vendor_id": row.get("vendor_id") or None,
    }


def _resolve_vendor(row: dict[str, str], master: MasterDataRepository, required: bool) -> str | None:
    """Accept a vendor code, a name or a tax number - whichever the export has.

    A reference that resolves to nothing is refused rather than stored: a
    purchase order pointing at a vendor who is not in the file would make
    `po_exists` fail for a reason that has nothing to do with the document.
    """
    code = row.get("vendor_id", "")
    if code:
        if master.get_vendor(code) is None:
            raise ValueError(f"vendor_id {code!r} is not in the vendor master file")
        return code

    name, tax_id = row.get("legal_name") or row.get("party_b", ""), row.get("tax_id", "")
    if name or tax_id:
        match = master.find_vendor(name or None, tax_id or None)
        if match.matched:
            return match.vendor_id
        if required:
            raise ValueError(f"no vendor in the master file matches {name or tax_id!r}")
    if required:
        raise ValueError("'vendor_id' is required (or a vendor name that matches the master file)")
    return None


def _purchase_order_row(row: dict[str, str], master: MasterDataRepository) -> dict[str, Any]:
    return {
        "po_number": _require(row, "po_number"),
        "vendor_id": _resolve_vendor(row, master, required=True),
        "po_date": _optional_date(row, "po_date"),
        "currency": (row.get("currency") or "THB").upper(),
        "amount": _optional_amount(row, "amount") or 0.0,
        "status": _choice(row, "status", PO_STATUSES, "open"),
        "description": row.get("description") or None,
    }


def _contract_row(row: dict[str, str], master: MasterDataRepository) -> dict[str, Any]:
    effective = _optional_date(row, "effective_date")
    expiry = _optional_date(row, "expiry_date")
    if effective and expiry and expiry < effective:
        raise ValueError(f"expiry_date {expiry} is before effective_date {effective}")
    return {
        "contract_number": _require(row, "contract_number"),
        "vendor_id": _resolve_vendor(row, master, required=False),
        "party_a": row.get("party_a") or None,
        "party_b": row.get("party_b") or None,
        "effective_date": effective,
        "expiry_date": expiry,
        "value": _optional_amount(row, "value") or _optional_amount(row, "amount"),
        "currency": (row.get("currency") or "THB").upper(),
    }


BUILDERS: dict[str, Callable[[dict[str, str], MasterDataRepository], dict[str, Any]]] = {
    "vendors": _vendor_row,
    "purchase_orders": _purchase_order_row,
    "contracts": _contract_row,
}
UPSERTS = {
    "vendors": "upsert_vendor",
    "purchase_orders": "upsert_purchase_order",
    "contracts": "upsert_contract",
}

TEMPLATES: dict[str, str] = {
    "vendors": "legal_name,tax_id,bank_account,aliases,status,payment_terms,currency\n"
               '"Prosoft CRM Co., Ltd.",0105539000123,1234567890,'
               '"Prosoft;บริษัท โปรซอฟท์ ซีอาร์เอ็ม จำกัด",active,Net 30,THB\n',
    "purchase_orders": "po_number,vendor_id,po_date,amount,currency,status,description\n"
                       "PO-2026-0001,V-1001,2026-07-01,187000.00,THB,open,Laptops and docking stations\n",
    "contracts": "contract_number,vendor_id,party_a,party_b,effective_date,expiry_date,value,currency\n"
                 'CT-2026-001,V-1001,Your Company PCL,"Prosoft CRM Co., Ltd.",2026-01-01,2026-12-31,368000.00,THB\n',
}


def import_rows(
    master: MasterDataRepository, kind: str, rows: Iterable[dict[str, str]]
) -> ImportReport:
    """Validate and upsert every row, collecting failures rather than raising."""
    if kind not in BUILDERS:
        raise MasterImportError(f"Unknown master data type {kind!r}. Expected one of {', '.join(KINDS)}.")

    build = BUILDERS[kind]
    upsert = getattr(master, UPSERTS[kind])
    created = updated = 0
    errors: list[RowError] = []

    for index, row in enumerate(rows, start=2):   # row 1 is the header, as the user sees it
        try:
            outcome = upsert(build(row, master))
        except ValueError as exc:
            errors.append(RowError(index, str(exc)))
            continue
        except Exception as exc:  # noqa: BLE001 - one bad row must not stop the file
            logger.exception("row %s of a %s import failed", index, kind)
            errors.append(RowError(index, str(exc)))
            continue
        if outcome == "created":
            created += 1
        else:
            updated += 1

    return ImportReport(kind=kind, created=created, updated=updated, errors=tuple(errors))


def import_csv(master: MasterDataRepository, kind: str, content: bytes | str) -> ImportReport:
    """Read a CSV and load it into the master file."""
    return import_rows(master, kind, read_rows(content))
