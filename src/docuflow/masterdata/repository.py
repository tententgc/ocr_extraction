"""Master-data lookups with tolerant vendor matching.

Vendor names on documents rarely match the master file exactly - OCR noise,
abbreviations, Thai/English variants and missing legal suffixes are all normal.
Matching therefore runs a ladder from strongest to weakest evidence and reports
which rung it landed on, so the confidence downstream is honest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from docuflow.db import connect, rows_to_dicts, row_to_dict, transaction
from docuflow.normalize.identifiers import parse_tax_id, reference_key
from docuflow.normalize.text import normalize_for_match

# Fuzzy name similarity at or above this is treated as the same vendor.
NAME_MATCH_THRESHOLD = 86.0
# Below this, no match is reported at all.
NAME_SUGGEST_THRESHOLD = 70.0

# Legal-form suffixes that carry no identifying information.
_NOISE_TOKENS = (
    "co ltd", "co , ltd", "company limited", "limited", "ltd", "public company limited",
    "pcl", "plc", "inc", "gmbh", "corporation", "corp", "บริษัท", "จำกัด", "มหาชน",
)


def _core_name(name: str) -> str:
    """Strip legal-form noise so "Siam Tech Supply Co., Ltd." ~ "Siam Tech Supply"."""
    normalized = normalize_for_match(name)
    for token in _NOISE_TOKENS:
        normalized = normalized.replace(normalize_for_match(token), " ")
    return " ".join(normalized.split())


@dataclass(frozen=True)
class VendorMatch:
    """Result of resolving a document's vendor against the master file."""

    vendor: dict[str, Any] | None
    score: float
    method: str
    candidates: tuple[dict[str, Any], ...] = ()

    @property
    def matched(self) -> bool:
        return self.vendor is not None

    @property
    def vendor_id(self) -> str | None:
        return self.vendor.get("vendor_id") if self.vendor else None


class MasterDataRepository:
    """Read/write access to vendors, purchase orders and contracts."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    # --- seeding -----------------------------------------------------------
    def seed(self, data: dict[str, tuple[dict[str, Any], ...]]) -> dict[str, int]:
        """Insert master data, leaving existing rows untouched."""
        counts: dict[str, int] = {}
        with transaction(self.db_path) as connection:
            for vendor in data.get("vendors", ()):
                connection.execute(
                    """INSERT OR IGNORE INTO vendors
                       (vendor_id, legal_name, aliases, tax_id, bank_account, status, payment_terms, currency)
                       VALUES (:vendor_id, :legal_name, :aliases, :tax_id, :bank_account, :status,
                               :payment_terms, :currency)""",
                    vendor,
                )
            counts["vendors"] = len(data.get("vendors", ()))

            for order in data.get("purchase_orders", ()):
                connection.execute(
                    """INSERT OR IGNORE INTO purchase_orders
                       (po_number, vendor_id, po_date, currency, amount, status, description)
                       VALUES (:po_number, :vendor_id, :po_date, :currency, :amount, :status, :description)""",
                    order,
                )
            counts["purchase_orders"] = len(data.get("purchase_orders", ()))

            for contract in data.get("contracts", ()):
                connection.execute(
                    """INSERT OR IGNORE INTO contracts
                       (contract_number, vendor_id, party_a, party_b, effective_date, expiry_date, value, currency)
                       VALUES (:contract_number, :vendor_id, :party_a, :party_b, :effective_date,
                               :expiry_date, :value, :currency)""",
                    contract,
                )
            counts["contracts"] = len(data.get("contracts", ()))
        return counts

    # --- writes ------------------------------------------------------------
    def add_vendor(
        self,
        legal_name: str,
        tax_id: str | None = None,
        bank_account: str | None = None,
        aliases: str = "",
        status: str = "active",
        payment_terms: str | None = None,
        currency: str = "THB",
        vendor_id: str | None = None,
    ) -> dict[str, Any]:
        """Register a vendor, so documents from it stop being blocked.

        Without this the only way to add a counterparty was to edit the sample
        catalogue and re-seed, which is not something a finance user can do.
        """
        if not legal_name.strip():
            raise ValueError("A vendor needs a legal name.")

        existing = self.find_vendor(legal_name, tax_id)
        if existing.matched and existing.score >= 0.97:
            raise ValueError(
                f"'{legal_name}' already exists as {existing.vendor['legal_name']} "
                f"({existing.vendor_id})."
            )

        identifier = vendor_id or self._next_vendor_id()
        row = {
            "vendor_id": identifier,
            "legal_name": legal_name.strip(),
            "aliases": aliases,
            "tax_id": parse_tax_id(tax_id) if tax_id else None,
            "bank_account": bank_account,
            "status": status,
            "payment_terms": payment_terms,
            "currency": currency,
        }
        with transaction(self.db_path) as connection:
            connection.execute(
                """INSERT INTO vendors
                   (vendor_id, legal_name, aliases, tax_id, bank_account, status, payment_terms, currency)
                   VALUES (:vendor_id, :legal_name, :aliases, :tax_id, :bank_account, :status,
                           :payment_terms, :currency)""",
                row,
            )
        return row

    # --- bulk load ---------------------------------------------------------
    # Import is an update of a copy, not an insert of new facts: the ERP is the
    # source, and re-exporting it tomorrow must correct yesterday's rows rather
    # than duplicate them. So these upsert on the natural key, and unlike
    # `add_vendor` they do not refuse an existing record - that refusal exists
    # to stop a person registering a duplicate by hand.

    def upsert_vendor(self, row: dict[str, Any]) -> str:
        """Insert or update one vendor. Returns "created" or "updated"."""
        if not str(row.get("legal_name", "")).strip():
            raise ValueError("'legal_name' is required and is empty")

        existing = None
        if row.get("vendor_id"):
            existing = self.get_vendor(str(row["vendor_id"]))
        if existing is None:
            match = self.find_vendor(row.get("legal_name"), row.get("tax_id"))
            # Only an identity-grade match may overwrite an existing record. A
            # fuzzy name hit is a suggestion, and quietly merging two suppliers
            # whose names look alike is worse than creating a second row.
            existing = match.vendor if match.matched and match.score >= 0.97 else None

        values = {
            "vendor_id": (existing or {}).get("vendor_id") or row.get("vendor_id") or self._next_vendor_id(),
            "legal_name": str(row["legal_name"]).strip(),
            "aliases": row.get("aliases") or "",
            "tax_id": row.get("tax_id"),
            "bank_account": row.get("bank_account"),
            "status": row.get("status") or "active",
            "payment_terms": row.get("payment_terms"),
            "currency": row.get("currency") or "THB",
            # Only set when the record came off a document; a CSV from the ERP
            # leaves it null, which is how the two are told apart later.
            "source_document_id": row.get("source_document_id"),
        }
        with transaction(self.db_path) as connection:
            connection.execute(
                """INSERT INTO vendors
                   (vendor_id, legal_name, aliases, tax_id, bank_account, status, payment_terms,
                    currency, source_document_id)
                   VALUES (:vendor_id, :legal_name, :aliases, :tax_id, :bank_account, :status,
                           :payment_terms, :currency, :source_document_id)
                   ON CONFLICT(vendor_id) DO UPDATE SET
                       legal_name = excluded.legal_name, aliases = excluded.aliases,
                       tax_id = excluded.tax_id, bank_account = excluded.bank_account,
                       status = excluded.status, payment_terms = excluded.payment_terms,
                       currency = excluded.currency,
                       source_document_id = excluded.source_document_id""",
                values,
            )
        return "updated" if existing else "created"

    def upsert_purchase_order(self, row: dict[str, Any]) -> str:
        existing = self.get_purchase_order(row.get("po_number"))
        with transaction(self.db_path) as connection:
            connection.execute(
                """INSERT INTO purchase_orders
                   (po_number, vendor_id, po_date, currency, amount, status, description,
                    source_document_id)
                   VALUES (:po_number, :vendor_id, :po_date, :currency, :amount, :status,
                           :description, :source_document_id)
                   ON CONFLICT(po_number) DO UPDATE SET
                       vendor_id = excluded.vendor_id, po_date = excluded.po_date,
                       currency = excluded.currency, amount = excluded.amount,
                       status = excluded.status, description = excluded.description,
                       source_document_id = excluded.source_document_id""",
                {"source_document_id": None, **row},
            )
        return "updated" if existing else "created"

    def upsert_contract(self, row: dict[str, Any]) -> str:
        existing = self.get_contract(row.get("contract_number"))
        with transaction(self.db_path) as connection:
            connection.execute(
                """INSERT INTO contracts
                   (contract_number, vendor_id, party_a, party_b, effective_date,
                    expiry_date, value, currency, source_document_id)
                   VALUES (:contract_number, :vendor_id, :party_a, :party_b, :effective_date,
                           :expiry_date, :value, :currency, :source_document_id)
                   ON CONFLICT(contract_number) DO UPDATE SET
                       vendor_id = excluded.vendor_id, party_a = excluded.party_a,
                       party_b = excluded.party_b, effective_date = excluded.effective_date,
                       expiry_date = excluded.expiry_date, value = excluded.value,
                       currency = excluded.currency,
                       source_document_id = excluded.source_document_id""",
                {"source_document_id": None, **row},
            )
        return "updated" if existing else "created"

    def clear(self) -> dict[str, int]:
        """Empty the master file. Posted records keep their own copy of what they matched."""
        counts = {}
        with transaction(self.db_path) as connection:
            for table in ("purchase_orders", "contracts", "vendors"):
                counts[table] = connection.execute(f"DELETE FROM {table}").rowcount
        return counts

    def is_empty(self) -> bool:
        return not (self.list_vendors() or self.list_purchase_orders() or self.list_contracts())

    def _next_vendor_id(self) -> str:
        """Next id in the V-#### series."""
        numbers = [
            int(vendor["vendor_id"].split("-")[-1])
            for vendor in self.list_vendors()
            if vendor["vendor_id"].startswith("V-") and vendor["vendor_id"].split("-")[-1].isdigit()
        ]
        return f"V-{max(numbers, default=1000) + 1}"

    # --- reads -------------------------------------------------------------
    def list_vendors(self) -> list[dict[str, Any]]:
        rows = connect(self.db_path).execute("SELECT * FROM vendors ORDER BY legal_name").fetchall()
        return rows_to_dicts(rows)

    def list_purchase_orders(self) -> list[dict[str, Any]]:
        rows = connect(self.db_path).execute("SELECT * FROM purchase_orders ORDER BY po_number").fetchall()
        return rows_to_dicts(rows)

    def list_contracts(self) -> list[dict[str, Any]]:
        rows = connect(self.db_path).execute("SELECT * FROM contracts ORDER BY contract_number").fetchall()
        return rows_to_dicts(rows)

    def get_vendor(self, vendor_id: str) -> dict[str, Any] | None:
        row = connect(self.db_path).execute(
            "SELECT * FROM vendors WHERE vendor_id = ?", (vendor_id,)
        ).fetchone()
        return row_to_dict(row)

    def get_purchase_order(self, po_number: str | None) -> dict[str, Any] | None:
        if not po_number:
            return None
        key = reference_key(po_number)
        for order in self.list_purchase_orders():
            if reference_key(order["po_number"]) == key:
                return order
        return None

    def get_contract(self, contract_number: str | None) -> dict[str, Any] | None:
        if not contract_number:
            return None
        key = reference_key(contract_number)
        for contract in self.list_contracts():
            if reference_key(contract["contract_number"]) == key:
                return contract
        return None

    # --- vendor resolution -------------------------------------------------
    def find_vendor(self, name: str | None, tax_id: str | None = None) -> VendorMatch:
        """Resolve a document's vendor to a master record."""
        vendors = self.list_vendors()
        if not vendors:
            return VendorMatch(None, 0.0, "no master data")

        # 1. Tax ID is an exact identifier - strongest possible evidence.
        normalized_tax = parse_tax_id(tax_id) if tax_id else None
        if normalized_tax:
            for vendor in vendors:
                if vendor.get("tax_id") and parse_tax_id(vendor["tax_id"]) == normalized_tax:
                    return VendorMatch(vendor, 1.0, "tax id")

        if not name:
            return VendorMatch(None, 0.0, "no vendor name extracted")

        target_normalized = normalize_for_match(name)
        target_core = _core_name(name)

        # 2. Exact legal name or a registered alias.
        for vendor in vendors:
            names = [vendor["legal_name"], *(vendor.get("aliases") or "").split(";")]
            for candidate in names:
                if candidate and normalize_for_match(candidate) == target_normalized:
                    return VendorMatch(vendor, 1.0, "exact name")

        # 3. Same name once legal-form noise is removed.
        for vendor in vendors:
            names = [vendor["legal_name"], *(vendor.get("aliases") or "").split(";")]
            if any(candidate and _core_name(candidate) == target_core for candidate in names):
                return VendorMatch(vendor, 0.97, "name (ignoring legal form)")

        # 4. Fuzzy - handles OCR damage such as "แฟชิลิตี้" for "แฟซิลิตี้".
        scored: list[tuple[float, dict[str, Any]]] = []
        for vendor in vendors:
            names = [vendor["legal_name"], *(vendor.get("aliases") or "").split(";")]
            best = max(
                (
                    max(
                        fuzz.WRatio(target_normalized, normalize_for_match(candidate)),
                        fuzz.WRatio(target_core, _core_name(candidate)),
                    )
                    for candidate in names
                    if candidate
                ),
                default=0.0,
            )
            scored.append((best, vendor))

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_vendor = scored[0]
        candidates = tuple(vendor for score, vendor in scored[:3] if score >= NAME_SUGGEST_THRESHOLD)

        if best_score >= NAME_MATCH_THRESHOLD:
            return VendorMatch(best_vendor, round(best_score / 100.0, 3), "fuzzy name", candidates)
        return VendorMatch(None, round(best_score / 100.0, 3), "no match", candidates)

    def vendor_for_bank_account(self, account: str | None) -> dict[str, Any] | None:
        """Which vendor, if any, owns this bank account."""
        if not account:
            return None
        key = reference_key(account)
        for vendor in self.list_vendors():
            if vendor.get("bank_account") and reference_key(vendor["bank_account"]) == key:
                return vendor
        return None
