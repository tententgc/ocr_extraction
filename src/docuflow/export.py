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
from docuflow.schemas.registry import DOCUMENT_SCHEMAS
from docuflow.store.records import RecordStore

# Excel on Windows assumes the system code page unless a file says otherwise,
# and renders every Thai name as mojibake. Three bytes fix it. UTF-8 without a
# BOM is the better format and the wrong choice here: the person opening this
# file is double-clicking it, not configuring an import dialog.
BOM = "﻿"

FORMATS = ("csv", "jsonl")

# What every record has, whatever kind of document produced it: who, how much,
# when, and where it came from. These mean the same thing on an invoice, a
# receipt and a contract, which is what makes them safe to put in one sheet.
#
# `payload` is deliberately absent - a JSON blob in a spreadsheet cell helps
# nobody, and the fields worth having are added per type below.
LEDGER_COLUMNS: tuple[str, ...] = (
    "record_id",
    "posted_at",
    "doc_type",
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
    "confidence",
    "decision",
    "posted_via",
    "line_item_count",
)


def columns_for(doc_type: str | None) -> tuple[str, ...]:
    """The header for this export.

    The three document types share almost nothing. An invoice has
    `purchase_order_ref`, `subtotal` and `tax_amount`; a contract has
    `party_a`, `party_b`, `contract_value` and two dates that are not the same
    two dates; a receipt has a merchant and an expense category. Across all
    three, `currency` is the only extracted field in common.

    So a single-type export gets that type's own fields appended, and a mixed
    export gets the ledger columns alone. The alternative - a union of every
    schema - produces a sheet that is mostly empty cells and asks the reader to
    work out which columns apply to which row.
    """
    if doc_type is None:
        return LEDGER_COLUMNS
    schema = DOCUMENT_SCHEMAS.get(doc_type)
    if schema is None:
        return LEDGER_COLUMNS
    # `currency` already carries the ledger's own value; a second column of the
    # same name would silently win or lose depending on dict ordering.
    extra = tuple(f.name for f in schema.fields if f.name not in LEDGER_COLUMNS)
    return LEDGER_COLUMNS + extra


# Kept for callers that want the plain ledger shape without asking for a type.
CSV_COLUMNS = LEDGER_COLUMNS


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


def flatten(row: dict[str, Any], columns: tuple[str, ...] | None = None) -> dict[str, Any]:
    """One ledger row as flat, spreadsheet-shaped values.

    Anything in `columns` that is not a ledger column is looked up in the
    document's extracted fields, so the same function serves an invoice export
    and a contract export without knowing which is which.
    """
    columns = columns or LEDGER_COLUMNS
    payload = _payload_of(row)
    fields = payload.get("fields") or {}
    flat: dict[str, Any] = {column: row.get(column) for column in columns}

    # Joined on from the document by `list_with_source` - the record's own
    # payload holds the extracted values, not the file they were read from.
    flat["source_file"] = row.get("source_file") or ""
    flat["doc_type"] = row.get("doc_type") or ""
    flat["confidence"] = payload.get("confidence")
    flat["decision"] = payload.get("decision") or ""
    flat["posted_via"] = payload.get("posted_via") or ""
    flat["line_item_count"] = len(payload.get("line_items") or ())

    for column in columns:
        if column not in LEDGER_COLUMNS:
            flat[column] = fields.get(column)
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


# A document type is what the page is; a record kind is what the ledger calls
# the entry it became. They are not the same word and never have been:
_KIND_OF = {"invoice": "payable", "receipt": "expense", "contract": "contract"}


def validate_doc_type(doc_type: str | None) -> str | None:
    """Refuse a type nobody can produce, rather than exporting an empty file.

    Asking for `--doc-type payable` is an easy mistake - it is the word the
    ledger uses - and silently returning a header row and nothing else reads
    as "there are no invoices", which is a far worse answer than an error.
    """
    if doc_type is None:
        return None
    known = sorted(DOCUMENT_SCHEMAS)
    if doc_type in known:
        return doc_type

    produced_by = {kind: doc for doc, kind in _KIND_OF.items()}
    if doc_type in produced_by:
        raise ExportError(
            f"{doc_type!r} is a record kind, not a document type. "
            f"{produced_by[doc_type]} documents post as {doc_type} records - "
            f"use doc_type={produced_by[doc_type]}, or record_kind={doc_type}."
        )
    raise ExportError(f"unknown document type {doc_type!r}; try one of: {', '.join(known)}")


def iter_records(
    store: RecordStore,
    record_kind: str | None = None,
    since: str | None = None,
    until: str | None = None,
    doc_type: str | None = None,
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
    doc_type = validate_doc_type(doc_type)

    offset = 0
    while True:
        rows = store.list_with_source(
            record_kind=record_kind, doc_type=doc_type, limit=batch, offset=offset
        )
        if not rows:
            return
        for row in rows:
            if _within(row, since, until):
                yield row
        if len(rows) < batch:
            return
        offset += batch


def to_csv(rows: Iterable[dict[str, Any]], doc_type: str | None = None) -> Iterator[str]:
    """Flat rows, header first, BOM before the header.

    The header depends on the type being exported: one type gets its own
    schema's fields, a mixed export gets the columns that mean the same thing
    on all of them.
    """
    columns = columns_for(doc_type)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore")

    writer.writeheader()
    yield BOM + buffer.getvalue()
    buffer.seek(0), buffer.truncate(0)

    for row in rows:
        writer.writerow(flatten(row, columns))
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
    doc_type: str | None = None,
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
    doc_type = validate_doc_type(doc_type)
    rows = iter_records(
        store, record_kind=record_kind, since=since, until=until, doc_type=doc_type
    )
    return to_csv(rows, doc_type) if fmt == "csv" else to_jsonl(rows)


def suggested_filename(
    fmt: str, record_kind: str | None = None, doc_type: str | None = None
) -> str:
    """Named for what is in it, so three exports in a folder stay apart."""
    what = doc_type or record_kind or "ledger"
    return f"docuflow-{what}-{date.today().isoformat()}.{fmt}"
