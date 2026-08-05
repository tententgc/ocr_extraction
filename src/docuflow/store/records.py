"""The downstream system of record.

This is the only place a value crosses from "extracted" into "posted". Every
write goes through `post_record`, which refuses documents that were not cleared
for posting and writes an audit entry for anything it does.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from docuflow.db import connect, dump_json, rows_to_dicts, row_to_dict, transaction
from docuflow.models import Decision, DocumentRun
from docuflow.normalize.identifiers import reference_key

# Where a document lands when the record it produced is deleted out from under
# it. Not `auto_post`: that is a claim about a ledger entry which no longer
# exists. Not `rejected` either - nobody rejected it, and rejection is a
# person's verdict that a policy change is not allowed to overturn.
RELEASED_DECISION: Decision = "needs_review"

# Which extracted field is the natural key for each document type.
REFERENCE_FIELDS = {
    "invoice": "invoice_number",
    "receipt": "receipt_number",
    "purchase_order": "po_number",
    "contract": "contract_number",
    "form": "reference_number",
    "unknown": "reference_number",
}

# Which extracted fields carry the headline amount and dates per type.
AMOUNT_FIELDS = {
    "invoice": "total_amount",
    "receipt": "total_amount",
    "purchase_order": "total_amount",
    "contract": "contract_value",
    "form": "requested_amount",
    "unknown": "total_amount",
}
ISSUE_DATE_FIELDS = {
    "invoice": "invoice_date",
    "receipt": "receipt_date",
    "purchase_order": "po_date",
    "contract": "effective_date",
    "form": "submitted_date",
    "unknown": "document_date",
}
COUNTERPARTY_FIELDS = {
    "invoice": "vendor_name",
    "receipt": "merchant_name",
    "purchase_order": "vendor_name",
    "contract": "party_b",
    "form": "applicant_name",
    "unknown": "counterparty",
}


def reference_field_for(doc_type: str) -> str:
    return REFERENCE_FIELDS.get(doc_type, "reference_number")


class RecordPostingError(RuntimeError):
    """Raised when a document is not eligible to post."""


class RecordStore:
    """Posted records plus the audit trail."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    # --- audit -------------------------------------------------------------
    def log(
        self,
        action: str,
        actor: str = "system",
        document_id: str | None = None,
        record_id: str | None = None,
        **detail: Any,
    ) -> None:
        with transaction(self.db_path) as connection:
            connection.execute(
                """INSERT INTO audit_log (document_id, record_id, actor, action, detail, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    document_id,
                    record_id,
                    actor,
                    action,
                    dump_json(detail),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def audit_for(self, document_id: str) -> list[dict[str, Any]]:
        rows = connect(self.db_path).execute(
            "SELECT * FROM audit_log WHERE document_id = ? ORDER BY id ASC", (document_id,)
        ).fetchall()
        return rows_to_dicts(rows)

    # --- duplicates --------------------------------------------------------
    def find_duplicate(
        self, doc_type: str, reference: str | None, vendor_id: str | None, exclude_document_id: str | None = None
    ) -> dict[str, Any] | None:
        """An already-posted record with the same reference for the same party."""
        if not reference:
            return None

        key = reference_key(reference)
        rows = connect(self.db_path).execute(
            "SELECT * FROM records WHERE record_kind IS NOT NULL AND status != 'reversed'"
        ).fetchall()

        for row in rows_to_dicts(rows):
            if exclude_document_id and row["document_id"] == exclude_document_id:
                continue
            if reference_key(row.get("reference")) != key:
                continue
            # Same reference from a different vendor is a coincidence, not a
            # duplicate - suppliers number their own invoices independently.
            if vendor_id and row.get("vendor_id") and row["vendor_id"] != vendor_id:
                continue
            return row
        return None

    # --- posting -----------------------------------------------------------
    def post_record(
        self,
        run: DocumentRun,
        vendor_id: str | None,
        actor: str = "system",
        force: bool = False,
    ) -> dict[str, Any]:
        """Write the extracted document into the downstream record table.

        `force` is what a human reviewer's approval sets; without it, only runs
        the routing policy cleared for `auto_post` may pass.
        """
        if run.decision != "auto_post" and not force:
            raise RecordPostingError(
                f"Document {run.document_id} is '{run.decision}' and has not been approved for posting."
            )

        from docuflow.schemas.registry import get_schema

        schema = get_schema(run.doc_type)
        reference = run.value(reference_field_for(run.doc_type))
        amount = run.value(AMOUNT_FIELDS.get(run.doc_type, "total_amount"))
        issue_date = run.value(ISSUE_DATE_FIELDS.get(run.doc_type, "document_date"))
        counterparty = run.value(COUNTERPARTY_FIELDS.get(run.doc_type, "counterparty"))

        record_id = f"REC-{uuid.uuid4().hex[:10].upper()}"
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "fields": run.field_map(),
            "line_items": [item.model_dump(mode="json") for item in run.line_items],
            "confidence": run.confidence,
            "decision": run.decision,
            "posted_via": "reviewer" if force else "straight_through",
            "evidence": {
                item.name: {"evidence": item.evidence, "page": item.page, "confidence": item.confidence}
                for item in run.fields
                if not item.is_empty
            },
        }

        with transaction(self.db_path) as connection:
            connection.execute(
                """INSERT INTO records
                   (record_id, document_id, record_kind, reference, vendor_id, counterparty,
                    amount, currency, issue_date, due_date, status, posted_by, posted_at, payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'posted', ?, ?, ?)""",
                (
                    record_id,
                    run.document_id,
                    schema.record_kind,
                    str(reference) if reference else None,
                    vendor_id,
                    str(counterparty) if counterparty else None,
                    float(amount) if isinstance(amount, (int, float)) else None,
                    run.value("currency"),
                    str(issue_date) if issue_date else None,
                    run.value("due_date"),
                    actor,
                    now,
                    dump_json(payload),
                ),
            )
            connection.execute(
                "UPDATE documents SET record_id = ? WHERE document_id = ?", (record_id, run.document_id)
            )

        self.log(
            "record_posted",
            actor=actor,
            document_id=run.document_id,
            record_id=record_id,
            reference=reference,
            amount=amount,
            currency=run.value("currency"),
            forced=force,
        )
        return {"record_id": record_id, "posted_at": now, "reference": reference, "amount": amount}

    # --- reads -------------------------------------------------------------
    def get(self, record_id: str) -> dict[str, Any] | None:
        row = connect(self.db_path).execute(
            "SELECT * FROM records WHERE record_id = ?", (record_id,)
        ).fetchone()
        return row_to_dict(row)

    def list(
        self, record_kind: str | None = None, limit: int = 200, offset: int = 0
    ) -> list[dict[str, Any]]:
        where = "WHERE record_kind = ?" if record_kind else ""
        params = (record_kind, limit, offset) if record_kind else (limit, offset)
        rows = connect(self.db_path).execute(
            f"SELECT * FROM records {where} ORDER BY posted_at DESC LIMIT ? OFFSET ?", params
        ).fetchall()
        return rows_to_dicts(rows)

    def count(self, record_kind: str | None = None) -> int:
        where = "WHERE record_kind = ?" if record_kind else ""
        params = (record_kind,) if record_kind else ()
        row = connect(self.db_path).execute(
            f"SELECT COUNT(*) AS total FROM records {where}", params
        ).fetchone()
        return int(row["total"]) if row else 0

    def delete(self, record_id: str, actor: str = "reviewer") -> dict[str, Any]:
        """Remove a posted record and release the document that produced it.

        A production ledger reverses rather than deletes - you post a credit note
        and both entries stand. This deletes, because the value here is being
        able to clear a test run, and the audit log keeps what happened either
        way. The document is unlinked rather than removed: the page it came from
        is still the evidence for the entry that used to exist.

        The document also stops claiming it posted. `auto_post` is a statement
        about a ledger entry, and once that entry is gone the queue would render
        "Posted automatically" beside an empty record column - two contradictory
        claims in one row, where the believable one is the one saying nobody
        needs to look. It returns as `needs_review`, which is what it now is.
        """
        record = self.get(record_id)
        if record is None:
            raise RecordPostingError(f"Record {record_id} was not found.")

        with transaction(self.db_path) as connection:
            connection.execute("DELETE FROM records WHERE record_id = ?", (record_id,))
            # The stored run carries its own copy of record_id and decision, and
            # everything that refuses to touch a posted document reads *those*.
            # Clearing only the columns would leave the document permanently
            # frozen, and moving only the columns would leave the payload -
            # which is what the document detail view reloads from - still posted.
            connection.execute(
                """UPDATE documents
                      SET record_id = NULL,
                          decision  = ?,
                          payload   = json_set(
                              payload,
                              '$.record_id', json('null'),
                              '$.decision', ?
                          )
                    WHERE document_id = ?""",
                (RELEASED_DECISION, RELEASED_DECISION, record["document_id"]),
            )
        self.log(
            "record_deleted",
            actor=actor,
            document_id=record["document_id"],
            record_id=record_id,
            reference=record.get("reference"),
            counterparty=record.get("counterparty"),
            amount=record.get("amount"),
            currency=record.get("currency"),
        )
        return {
            "record_id": record_id,
            "document_id": record["document_id"],
            "message": (
                f"Deleted {record_id}. {record['document_id']} is back in the queue and "
                f"can be approved again or removed."
            ),
        }

    def totals_by_currency(self) -> list[dict[str, Any]]:
        rows = connect(self.db_path).execute(
            """SELECT currency, record_kind, COUNT(*) AS count, SUM(amount) AS total
               FROM records WHERE status = 'posted' GROUP BY currency, record_kind"""
        ).fetchall()
        return rows_to_dicts(rows)

    def summary_by_kind(self) -> list[dict[str, Any]]:
        """One row per record kind, for the ledger overview.

        Amounts are grouped by currency as well as by kind: adding baht to euros
        produces a number that means nothing, and a ledger view that does it is
        worse than one that shows two rows.
        """
        rows = connect(self.db_path).execute(
            """SELECT record_kind, currency,
                      COUNT(*)              AS count,
                      SUM(amount)           AS total,
                      MIN(issue_date)       AS earliest,
                      MAX(issue_date)       AS latest,
                      COUNT(DISTINCT vendor_id)    AS counterparties,
                      SUM(CASE WHEN posted_by = 'system' THEN 1 ELSE 0 END) AS auto_posted
               FROM records
               WHERE status = 'posted'
               GROUP BY record_kind, currency
               ORDER BY record_kind, currency"""
        ).fetchall()

        by_kind: dict[str, dict[str, Any]] = {}
        for row in rows_to_dicts(rows):
            kind = row["record_kind"]
            entry = by_kind.setdefault(
                kind,
                {
                    "record_kind": kind,
                    "count": 0,
                    "auto_posted": 0,
                    "counterparties": 0,
                    "earliest": None,
                    "latest": None,
                    "totals": [],
                },
            )
            entry["count"] += row["count"]
            entry["auto_posted"] += row["auto_posted"] or 0
            entry["counterparties"] = max(entry["counterparties"], row["counterparties"] or 0)
            for key, pick in (("earliest", min), ("latest", max)):
                value = row[key]
                if value:
                    entry[key] = value if entry[key] is None else pick(entry[key], value)
            if row["total"] is not None:
                entry["totals"].append(
                    {"currency": row["currency"] or "-", "total": row["total"], "count": row["count"]}
                )
        return sorted(by_kind.values(), key=lambda item: -item["count"])

    def top_counterparties(self, record_kind: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        """Who this kind of record is mostly with."""
        where = "WHERE status = 'posted'" + (" AND record_kind = ?" if record_kind else "")
        params: tuple[Any, ...] = (record_kind, limit) if record_kind else (limit,)
        rows = connect(self.db_path).execute(
            f"""SELECT counterparty, currency, COUNT(*) AS count, SUM(amount) AS total
                FROM records {where}
                GROUP BY counterparty, currency
                ORDER BY count DESC, total DESC
                LIMIT ?""",
            params,
        ).fetchall()
        return rows_to_dicts(rows)
