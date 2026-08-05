"""Getting the ledger back out, in the two shapes it is actually asked for.

A posted record is the point of the whole system, and until now the only way to
read one back was the paginated API or opening SQLite. Both are fine for a
program and wrong for the two people who ask: someone reconciling in Excel, and
someone loading the ledger into another system.

So two formats, and they are not variations on one thing:

**CSV** is for a person. The nested payload is flattened into columns, amounts
stay numeric so a spreadsheet can sum them, and the file carries a UTF-8 BOM -
without it Excel on Windows reads Thai counterparty names as mojibake, which is
most of this register.

**JSONL** is for a machine. One record per line, payload intact, nothing
flattened or rounded. A consumer that wants the line items or the evidence
still has them.

Both stream. The ledger is the table that grows without limit, and an export
that builds the whole answer in memory first is one that stops working on the
day it finally matters.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import date
from typing import Any, Iterable, Iterator

from docuflow.db import load_json
from docuflow.store.records import RecordStore

# Excel on Windows assumes the system code page unless a file says otherwise,
# and renders every Thai name as mojibake. Three bytes fix it. UTF-8 without a
# BOM is the better format and the wrong choice here: the person opening this
# file is double-clicking it, not configuring an import dialog.
BOM = "﻿"

FORMATS = ("csv", "jsonl")

# The columns a reconciliation actually uses, in the order it reads them.
# `payload` is deliberately absent - a JSON blob in a spreadsheet cell helps
# nobody, and the fields worth having are pulled out below.
CSV_COLUMNS: tuple[str, ...] = (
    "record_id",
    "posted_at",
    "record_kind",
    "reference",
    "counterparty",
    "vendor_id",
    "amount",
    "currency",
    "issue_date",
    "due_date",
    "status",
    "posted_by",
    "document_id",
    "source_file",
    "purchase_order_ref",
    "subtotal",
    "tax_amount",
    "total_amount",
    "confidence",
    "decision",
    "posted_via",
    "line_item_count",
)

# Which extracted field carries the buyer's own reference, per document type.
_PAYLOAD_FIELDS = ("purchase_order_ref", "subtotal", "tax_amount", "total_amount")


class ExportError(ValueError):
    """The request cannot be answered as asked."""


def _payload_of(row: dict[str, Any]) -> dict[str, Any]:
    """The stored payload as a dict, whatever state it is in.

    A record whose payload failed to parse is still a record: it was posted,
    it has an amount, and leaving it out of an export would understate the
    ledger. The flattened columns come back empty and the row survives.
    """
    raw = row.get("payload")
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return load_json(raw)
    except Exception:  # noqa: BLE001 - a bad payload must not lose the row
        return {}


def flatten(row: dict[str, Any]) -> dict[str, Any]:
    """One ledger row as flat, spreadsheet-shaped values."""
    payload = _payload_of(row)
    fields = payload.get("fields") or {}
    flat: dict[str, Any] = {column: row.get(column) for column in CSV_COLUMNS}

    # Joined on from the document by `list_with_source` - the record's own
    # payload holds the extracted values, not the file they were read from.
    flat["source_file"] = row.get("source_file") or ""
    for name in _PAYLOAD_FIELDS:
        flat[name] = fields.get(name)
    flat["confidence"] = payload.get("confidence")
    flat["decision"] = payload.get("decision") or ""
    flat["posted_via"] = payload.get("posted_via") or ""
    flat["line_item_count"] = len(payload.get("line_items") or ())
    return flat


def _within(row: dict[str, Any], since: str | None, until: str | None) -> bool:
    """Filter on the posting date, not the document's own date.

    An export for "what did we post in July" is a question about the ledger.
    An invoice dated in June can be posted in July, and a reconciliation is
    looking for the second.
    """
    posted = str(row.get("posted_at") or "")[:10]
    if since and posted < since:
        return False
    if until and posted > until:
        return False
    return True


def _validate_date(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ExportError(f"{label} must be a date like 2026-07-01, not {value!r}") from exc


def validate_range(since: str | None, until: str | None) -> tuple[str | None, str | None]:
    """Both bounds, normalised - or `ExportError` now rather than mid-stream."""
    since = _validate_date(since, "--from")
    until = _validate_date(until, "--to")
    if since and until and since > until:
        raise ExportError(f"--from {since} is after --to {until}")
    return since, until


def iter_records(
    store: RecordStore,
    record_kind: str | None = None,
    since: str | None = None,
    until: str | None = None,
    batch: int = 500,
) -> Iterator[dict[str, Any]]:
    """Every matching record, read a page at a time.

    Paging here is an implementation detail rather than a limit the caller has
    to know about - the API's `per_page` cap is a courtesy to the browser, and
    an export that inherited it would silently stop at 200 rows.

    Callers that need the arguments checked before the first row is pulled
    should use `export`, or `validate_range` directly: this is a generator, so
    nothing in its body runs until it is iterated.
    """
    since, until = validate_range(since, until)

    offset = 0
    while True:
        rows = store.list_with_source(record_kind=record_kind, limit=batch, offset=offset)
        if not rows:
            return
        for row in rows:
            if _within(row, since, until):
                yield row
        if len(rows) < batch:
            return
        offset += batch


def to_csv(rows: Iterable[dict[str, Any]]) -> Iterator[str]:
    """Flat rows, header first, BOM before the header."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")

    writer.writeheader()
    yield BOM + buffer.getvalue()
    buffer.seek(0), buffer.truncate(0)

    for row in rows:
        writer.writerow(flatten(row))
        yield buffer.getvalue()
        buffer.seek(0), buffer.truncate(0)


def to_jsonl(rows: Iterable[dict[str, Any]]) -> Iterator[str]:
    """One record per line, payload parsed rather than a string of JSON.

    The stored payload is text in SQLite. Emitting it as-is would nest a JSON
    document inside a JSON document as an escaped string, and every consumer
    would have to parse twice and discover that for themselves.
    """
    for row in rows:
        record = dict(row)
        record["payload"] = _payload_of(row)
        yield json.dumps(record, ensure_ascii=False, default=str) + "\n"


def export(
    store: RecordStore,
    fmt: str = "csv",
    record_kind: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> Iterator[str]:
    """The ledger, streamed in the requested format.

    Arguments are checked here, before any generator is built, and this is not
    a stylistic choice. `iter_records` is a generator, so validation inside it
    does not run until something pulls the first row - which for the HTTP
    endpoint is after the response has been handed to the server and the
    caller's `except ExportError` has long since been passed. A mistyped
    `--from July` came back as 200 and a complete export of everything, which
    is the worst possible answer to a typo in a date filter.
    """
    if fmt not in FORMATS:
        raise ExportError(f"format must be one of {', '.join(FORMATS)}, not {fmt!r}")
    since, until = validate_range(since, until)
    rows = iter_records(store, record_kind=record_kind, since=since, until=until)
    return to_csv(rows) if fmt == "csv" else to_jsonl(rows)


def suggested_filename(fmt: str, record_kind: str | None = None) -> str:
    stem = f"docuflow-{record_kind or 'ledger'}-{date.today().isoformat()}"
    return f"{stem}.{fmt}"
