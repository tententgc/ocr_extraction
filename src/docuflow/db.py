"""SQLite storage: master data, processed documents, posted records, audit trail.

The `records` table stands in for the downstream system (ERP / ledger). Nothing
reaches it except through `records.repository.post_record`, which is what makes
the "wrong values are costly" requirement enforceable in one place.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_local = threading.local()

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS vendors (
    vendor_id      TEXT PRIMARY KEY,
    legal_name     TEXT NOT NULL,
    aliases        TEXT NOT NULL DEFAULT '',
    tax_id         TEXT,
    bank_account   TEXT,
    status         TEXT NOT NULL DEFAULT 'active',
    payment_terms  TEXT,
    currency       TEXT NOT NULL DEFAULT 'THB'
);

CREATE TABLE IF NOT EXISTS purchase_orders (
    po_number    TEXT PRIMARY KEY,
    vendor_id    TEXT NOT NULL,
    po_date      TEXT,
    currency     TEXT NOT NULL DEFAULT 'THB',
    amount       REAL NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'open',
    description  TEXT
);

CREATE TABLE IF NOT EXISTS contracts (
    contract_number TEXT PRIMARY KEY,
    vendor_id       TEXT,
    party_a         TEXT,
    party_b         TEXT,
    effective_date  TEXT,
    expiry_date     TEXT,
    value           REAL,
    currency        TEXT NOT NULL DEFAULT 'THB'
);

CREATE TABLE IF NOT EXISTS documents (
    document_id   TEXT PRIMARY KEY,
    sha256        TEXT NOT NULL,
    filename      TEXT NOT NULL,
    source_kind   TEXT NOT NULL,
    origin        TEXT,
    content_type  TEXT,
    byte_size     INTEGER NOT NULL DEFAULT 0,
    doc_type      TEXT NOT NULL DEFAULT 'unknown',
    decision      TEXT NOT NULL DEFAULT 'needs_review',
    confidence    REAL NOT NULL DEFAULT 0,
    record_id     TEXT,
    created_at    TEXT NOT NULL,
    payload       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_sha ON documents(sha256);
CREATE INDEX IF NOT EXISTS idx_documents_decision ON documents(decision);

CREATE TABLE IF NOT EXISTS records (
    record_id     TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL,
    record_kind   TEXT NOT NULL,
    reference     TEXT,
    vendor_id     TEXT,
    counterparty  TEXT,
    amount        REAL,
    currency      TEXT,
    issue_date    TEXT,
    due_date      TEXT,
    status        TEXT NOT NULL DEFAULT 'posted',
    posted_by     TEXT NOT NULL DEFAULT 'system',
    posted_at     TEXT NOT NULL,
    payload       TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);
CREATE INDEX IF NOT EXISTS idx_records_reference ON records(record_kind, reference);
CREATE INDEX IF NOT EXISTS idx_records_vendor ON records(vendor_id);

CREATE TABLE IF NOT EXISTS audit_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  TEXT,
    record_id    TEXT,
    actor        TEXT NOT NULL,
    action       TEXT NOT NULL,
    detail       TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_document ON audit_log(document_id);

CREATE TABLE IF NOT EXISTS corrections (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  TEXT NOT NULL,
    doc_type     TEXT NOT NULL,
    vendor_key   TEXT,
    field_name   TEXT NOT NULL,
    old_value    TEXT,
    new_value    TEXT,
    reviewer     TEXT NOT NULL DEFAULT 'reviewer',
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_corrections_field ON corrections(doc_type, field_name);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Per-thread connection (SQLite objects are not shareable across threads)."""
    key = f"conn_{db_path}"
    existing = getattr(_local, key, None)
    if existing is not None:
        return existing

    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    setattr(_local, key, connection)
    return connection


def init_db(db_path: Path) -> None:
    """Create tables if they do not exist."""
    connection = connect(db_path)
    connection.executescript(SCHEMA)
    connection.commit()


@contextmanager
def transaction(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Run a unit of work, rolling back on any error."""
    connection = connect(db_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def load_json(value: str | None) -> Any:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}
