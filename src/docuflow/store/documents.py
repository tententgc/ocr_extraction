"""Persistence for processed documents.

The full `DocumentRun` - OCR text, fields, evidence, findings and trace - is
stored as JSON so any decision can be reconstructed and audited later.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from docuflow.db import connect, dump_json, load_json, rows_to_dicts, row_to_dict, transaction
from docuflow.models import DocumentRun


class DocumentStore:
    """Read/write access to processed documents."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def save(self, run: DocumentRun) -> None:
        """Insert or replace the stored run."""
        payload = run.model_dump(mode="json")
        with transaction(self.db_path) as connection:
            connection.execute(
                """INSERT INTO documents
                   (document_id, sha256, filename, source_kind, origin, content_type, byte_size,
                    doc_type, decision, confidence, record_id, created_at, payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(document_id) DO UPDATE SET
                       doc_type   = excluded.doc_type,
                       decision   = excluded.decision,
                       confidence = excluded.confidence,
                       record_id  = excluded.record_id,
                       payload    = excluded.payload""",
                (
                    run.document_id,
                    run.source.sha256,
                    run.source.filename,
                    run.source.kind,
                    run.source.origin,
                    run.source.content_type,
                    run.source.byte_size,
                    run.doc_type,
                    run.decision,
                    run.confidence,
                    run.record_id,
                    run.created_at.isoformat(),
                    dump_json(payload),
                ),
            )

    def get(self, document_id: str) -> DocumentRun | None:
        row = connect(self.db_path).execute(
            "SELECT payload FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        if row is None:
            return None
        return DocumentRun.model_validate(load_json(row["payload"]))

    def summary(self, document_id: str) -> dict[str, Any] | None:
        row = connect(self.db_path).execute(
            """SELECT document_id, filename, source_kind, origin, doc_type, decision,
                      confidence, record_id, created_at, byte_size
               FROM documents WHERE document_id = ?""",
            (document_id,),
        ).fetchone()
        return row_to_dict(row)

    @staticmethod
    def _filters(decision: str | None, doc_type: str | None) -> tuple[str, list[Any]]:
        clauses, params = [], []
        if decision:
            clauses.append("decision = ?")
            params.append(decision)
        if doc_type:
            clauses.append("doc_type = ?")
            params.append(doc_type)
        return (f"WHERE {' AND '.join(clauses)}" if clauses else ""), params

    def list(
        self,
        decision: str | None = None,
        doc_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        where, params = self._filters(decision, doc_type)
        rows = connect(self.db_path).execute(
            f"""SELECT document_id, filename, source_kind, origin, doc_type, decision,
                       confidence, record_id, created_at, byte_size
                FROM documents {where} ORDER BY created_at DESC LIMIT ? OFFSET ?""",
            (*params, limit, offset),
        ).fetchall()
        return rows_to_dicts(rows)

    def count(self, decision: str | None = None, doc_type: str | None = None) -> int:
        """How many match, regardless of the page being shown."""
        where, params = self._filters(decision, doc_type)
        row = connect(self.db_path).execute(
            f"SELECT COUNT(*) AS total FROM documents {where}", params
        ).fetchone()
        return int(row["total"]) if row else 0

    def find_by_hash(self, sha256: str) -> dict[str, Any] | None:
        row = connect(self.db_path).execute(
            """SELECT document_id, filename, doc_type, decision, record_id, created_at
               FROM documents WHERE sha256 = ? ORDER BY created_at DESC LIMIT 1""",
            (sha256,),
        ).fetchone()
        return row_to_dict(row)

    def counts_by_decision(self) -> dict[str, int]:
        rows = connect(self.db_path).execute(
            "SELECT decision, COUNT(*) AS total FROM documents GROUP BY decision"
        ).fetchall()
        return {row["decision"]: row["total"] for row in rows}

    def stats(self) -> dict[str, Any]:
        """Headline numbers for the dashboard."""
        connection = connect(self.db_path)
        counts = self.counts_by_decision()
        total = sum(counts.values())
        row = connection.execute(
            "SELECT AVG(confidence) AS average FROM documents"
        ).fetchone()
        by_type = connection.execute(
            "SELECT doc_type, COUNT(*) AS total FROM documents GROUP BY doc_type ORDER BY total DESC"
        ).fetchall()
        return {
            "total_documents": total,
            "by_decision": counts,
            "by_type": {item["doc_type"]: item["total"] for item in by_type},
            "auto_post_rate": round(counts.get("auto_post", 0) / total, 3) if total else 0.0,
            "average_confidence": round(row["average"] or 0.0, 3),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
