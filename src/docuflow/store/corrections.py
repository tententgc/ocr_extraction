"""Reviewer corrections - the feedback loop that makes the system improve.

Every edit a reviewer makes is recorded against the document type and vendor.
The aggregated view shows which fields the extractor gets wrong most often,
which is what tells you where to spend the next improvement effort.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from docuflow.db import connect, rows_to_dicts, transaction


class CorrectionStore:
    """Records of human edits to extracted values."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def record(
        self,
        document_id: str,
        doc_type: str,
        field_name: str,
        old_value: Any,
        new_value: Any,
        vendor_key: str | None = None,
        reviewer: str = "reviewer",
    ) -> None:
        with transaction(self.db_path) as connection:
            connection.execute(
                """INSERT INTO corrections
                   (document_id, doc_type, vendor_key, field_name, old_value, new_value, reviewer, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    document_id,
                    doc_type,
                    vendor_key,
                    field_name,
                    None if old_value is None else str(old_value),
                    None if new_value is None else str(new_value),
                    reviewer,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def for_document(self, document_id: str) -> list[dict[str, Any]]:
        rows = connect(self.db_path).execute(
            "SELECT * FROM corrections WHERE document_id = ? ORDER BY id ASC", (document_id,)
        ).fetchall()
        return rows_to_dicts(rows)

    def error_hotspots(self, limit: int = 15) -> list[dict[str, Any]]:
        """Fields most often corrected - where accuracy work pays off most."""
        rows = connect(self.db_path).execute(
            """SELECT doc_type, field_name, COUNT(*) AS corrections
               FROM corrections GROUP BY doc_type, field_name
               ORDER BY corrections DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return rows_to_dicts(rows)

    def total(self) -> int:
        row = connect(self.db_path).execute("SELECT COUNT(*) AS total FROM corrections").fetchone()
        return int(row["total"]) if row else 0
