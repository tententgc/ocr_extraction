"""Getting the ledger back out.

Two formats for two readers, and the tests are about the differences between
them: a spreadsheet needs the nesting flattened and its encoding announced, a
downstream system needs the nesting intact. Anything that is true of both is
tested once.
"""

from __future__ import annotations

import csv
import io
import json

import pytest

from docuflow.export import (
    BOM,
    LEDGER_COLUMNS,
    ExportError,
    columns_for,
    export,
    flatten,
    iter_records,
    suggested_filename,
)
from docuflow.store.documents import DocumentStore
from docuflow.store.records import RecordStore

GOOD_INVOICE = {
    "invoice_number": "INV-2026-0731",
    "vendor_name": "Siam Tech Supply Co., Ltd.",
    "vendor_tax_id": "0105543000129",
    "invoice_date": "2026-07-31",
    "due_date": "2026-08-30",
    "currency": "THB",
    "subtotal": 187000.0,
    "tax_rate": 7.0,
    "tax_amount": 13090.0,
    "total_amount": 200090.0,
    "purchase_order_ref": "PO-2026-0412",
}


@pytest.fixture
def records(settings) -> RecordStore:
    return RecordStore(settings.db_path)


@pytest.fixture
def documents(settings) -> DocumentStore:
    return DocumentStore(settings.db_path)


def _post(documents, records, make_run, fields, document_id):
    """A record has a foreign key to its document, so the document goes first."""
    run = make_run(fields).evolve(document_id=document_id, decision="auto_post")
    documents.save(run)
    return records.post_record(run, "V-1001")


@pytest.fixture
def posted(settings, master, records, documents, make_run):
    """Three posted records, one of them with a Thai counterparty."""
    return [
        _post(documents, records, make_run,
              dict(GOOD_INVOICE, invoice_number=f"INV-EXPORT-{index}", vendor_name=name),
              f"DOC-EXPORT-{index}")
        for index, name in enumerate(
            ("Siam Tech Supply Co., Ltd.", "บริษัท ไทยพัฒนา วิศวกรรม จำกัด", "Northwind Logistics")
        )
    ]


def _read_csv(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text.lstrip(BOM))))


class TestCsvIsForAPerson:
    def test_it_starts_with_a_bom(self, records, posted) -> None:
        """Without it, Excel on Windows renders every Thai name as mojibake."""
        text = "".join(export(records, "csv"))
        assert text.startswith(BOM), (
            "Excel assumes the system code page unless the file says otherwise, "
            "and most counterparties in this register are Thai"
        )

    def test_thai_survives_the_round_trip(self, records, posted) -> None:
        text = "".join(export(records, "csv"))
        assert "บริษัท ไทยพัฒนา วิศวกรรม จำกัด" in text

    def test_every_record_becomes_one_row(self, records, posted) -> None:
        rows = _read_csv("".join(export(records, "csv")))
        assert len(rows) == len(posted)

    def test_the_payload_is_flattened_not_pasted_in(self, records, posted) -> None:
        """A JSON blob in a spreadsheet cell helps nobody."""
        rows = _read_csv("".join(export(records, "csv", doc_type="invoice")))
        assert "payload" not in rows[0]
        assert rows[0]["purchase_order_ref"] == "PO-2026-0412"
        assert rows[0]["source_file"], "the file it came from is how a person finds it again"

    def test_amounts_stay_numeric(self, records, posted) -> None:
        """So the column can be summed without cleaning it first."""
        rows = _read_csv("".join(export(records, "csv", doc_type="invoice")))
        assert float(rows[0]["amount"]) == pytest.approx(200090.0)
        assert float(rows[0]["subtotal"]) == pytest.approx(187000.0)

    def test_the_header_is_the_declared_column_order(self, records, posted) -> None:
        rows = _read_csv("".join(export(records, "csv")))
        assert tuple(rows[0].keys()) == LEDGER_COLUMNS

    def test_a_mixed_export_says_which_type_each_row_is(self, records, posted) -> None:
        """Without it, one sheet of invoices and contracts is unreadable."""
        rows = _read_csv("".join(export(records, "csv")))
        assert all(row["doc_type"] for row in rows)


class TestJsonlIsForASystem:
    def test_one_record_per_line(self, records, posted) -> None:
        lines = "".join(export(records, "jsonl")).strip().splitlines()
        assert len(lines) == len(posted)
        for line in lines:
            json.loads(line)

    def test_the_payload_is_an_object_not_a_string_of_json(self, records, posted) -> None:
        """Otherwise every consumer parses twice and finds that out the hard way."""
        first = json.loads("".join(export(records, "jsonl")).splitlines()[0])
        assert isinstance(first["payload"], dict)
        assert first["payload"]["fields"]["invoice_number"].startswith("INV-EXPORT-")

    def test_nothing_is_flattened_away(self, records, posted) -> None:
        first = json.loads("".join(export(records, "jsonl")).splitlines()[0])
        assert "line_items" in first["payload"]
        assert "evidence" in first["payload"], "the evidence is why the value is trusted"

    def test_thai_is_not_escaped_into_ascii(self, records, posted) -> None:
        text = "".join(export(records, "jsonl"))
        assert "บริษัท" in text and "\\u0e1a" not in text


class TestFilters:
    def test_posting_date_bounds(self, records, posted) -> None:
        assert len(list(iter_records(records, since="2000-01-01"))) == len(posted)
        assert list(iter_records(records, until="2000-01-01")) == []

    def test_a_bad_date_is_refused_rather_than_ignored(self, records) -> None:
        """Silently exporting everything is the wrong answer to a typo."""
        with pytest.raises(ExportError, match="2026-07"):
            list(iter_records(records, since="July"))
        with pytest.raises(ExportError, match="must be a date"):
            list(iter_records(records, until="31/07/2026"))

    def test_a_backwards_range_is_refused(self, records) -> None:
        with pytest.raises(ExportError, match="is after"):
            list(iter_records(records, since="2026-08-01", until="2026-07-01"))

    def test_an_unknown_format_is_refused(self, records) -> None:
        with pytest.raises(ExportError, match="format must be"):
            list(export(records, "xlsx"))

    def test_a_bad_date_is_refused_before_a_single_row_is_read(self, records, posted) -> None:
        """The bug this exists for.

        `iter_records` is a generator, so validating inside it does not run
        until something pulls the first row - which for the HTTP endpoint is
        after the response has been handed to the server and the caller's
        `except ExportError` has already returned. `--from July` came back 200
        with a complete export of everything.

        Calling `export` must raise on the way in, not on the way out.
        """
        with pytest.raises(ExportError):
            export(records, "csv", since="July")   # no list(), no iteration

        with pytest.raises(ExportError):
            export(records, "jsonl", since="2026-08-01", until="2026-07-01")


class TestItStreamsPastThePageSize:
    def test_more_records_than_one_batch(
        self, settings, master, records, documents, make_run
    ) -> None:
        """The API caps `per_page`; an export that inherited it would stop short."""
        for index in range(7):
            _post(documents, records, make_run,
                  dict(GOOD_INVOICE, invoice_number=f"INV-BATCH-{index}"),
                  f"DOC-BATCH-{index}")

        assert len(list(iter_records(records, batch=2))) == 7

    def test_the_header_arrives_before_the_rows_are_read(self, records, posted) -> None:
        """Proves it is a stream, not a list built first and yielded after."""
        chunks = export(records, "csv")
        assert next(chunks).startswith(BOM)


class TestOddRows:
    def test_a_record_whose_payload_will_not_parse_still_exports(
        self, settings, records, posted
    ) -> None:
        """It was posted and it has an amount; dropping it understates the ledger."""
        from docuflow.db import transaction

        with transaction(settings.db_path) as connection:
            connection.execute(
                "UPDATE records SET payload = ? WHERE record_id = ?",
                ("{not json", posted[0]["record_id"]),
            )

        rows = _read_csv("".join(export(records, "csv", doc_type="invoice")))
        assert len(rows) == len(posted)
        damaged = next(r for r in rows if r["record_id"] == posted[0]["record_id"])
        assert damaged["amount"], "the ledger columns survive a bad payload"
        assert damaged["purchase_order_ref"] == "", "the flattened ones come back empty"

    def test_flatten_covers_every_declared_column(self, records, posted) -> None:
        row = records.list_with_source(limit=1)[0]
        assert set(flatten(row)) == set(LEDGER_COLUMNS)
        invoice_columns = columns_for("invoice")
        assert set(flatten(row, invoice_columns)) == set(invoice_columns)


class TestEachDocumentTypeHasItsOwnShape:
    """The three schemas share almost nothing.

    An invoice has `purchase_order_ref` and `subtotal`; a contract has
    `party_a`, `party_b` and `contract_value`; a receipt has a merchant and an
    expense category. Exporting all of them under one invoice-shaped header
    gives contracts a row of empty cells and drops the fields that matter.
    """

    def test_an_invoice_export_carries_invoice_fields(self) -> None:
        columns = columns_for("invoice")
        assert {"purchase_order_ref", "subtotal", "tax_amount", "bank_account"} <= set(columns)
        assert "party_a" not in columns

    def test_a_contract_export_carries_contract_fields(self) -> None:
        columns = columns_for("contract")
        assert {"party_a", "party_b", "contract_value", "expiry_date"} <= set(columns)
        assert "purchase_order_ref" not in columns

    def test_a_receipt_export_carries_receipt_fields(self) -> None:
        columns = columns_for("receipt")
        assert {"merchant_name", "expense_category", "payment_method"} <= set(columns)
        assert "party_a" not in columns

    def test_a_mixed_export_falls_back_to_what_they_share(self) -> None:
        """Not a union of every schema - that is a sheet of mostly empty cells."""
        assert columns_for(None) == LEDGER_COLUMNS
        for name in ("purchase_order_ref", "party_a", "merchant_name"):
            assert name not in LEDGER_COLUMNS

    def test_the_ledger_columns_are_never_duplicated_by_a_schema_field(self) -> None:
        """`currency` is on every schema and already a ledger column."""
        for doc_type in ("invoice", "receipt", "contract"):
            columns = columns_for(doc_type)
            assert len(columns) == len(set(columns)), f"{doc_type} repeats a column"

    def test_every_type_keeps_the_shared_columns_first(self) -> None:
        for doc_type in ("invoice", "receipt", "contract"):
            assert columns_for(doc_type)[: len(LEDGER_COLUMNS)] == LEDGER_COLUMNS


class TestChoosingTheType:
    def test_only_that_type_is_exported(self, records, posted) -> None:
        rows = _read_csv("".join(export(records, "csv", doc_type="invoice")))
        assert rows and all(row["doc_type"] == "invoice" for row in rows)

    def test_a_type_with_nothing_posted_yields_a_header_and_no_rows(
        self, records, posted
    ) -> None:
        """Empty is a real answer here - the type exists, nothing of it posted."""
        text = "".join(export(records, "csv", doc_type="contract"))
        assert _read_csv(text) == []
        assert "party_a" in text, "the header still describes what a contract would hold"

    def test_a_record_kind_passed_as_a_doc_type_is_caught(self, records) -> None:
        """`payable` is the ledger's word for an invoice, and an easy mistake.

        Exporting nothing would read as "we have no invoices", which is worse
        than being told the argument was wrong.
        """
        with pytest.raises(ExportError, match="record kind, not a document type"):
            export(records, "csv", doc_type="payable")

    def test_an_unknown_type_lists_the_real_ones(self, records) -> None:
        with pytest.raises(ExportError, match="unknown document type"):
            export(records, "csv", doc_type="quotation")

    def test_jsonl_needs_no_header_so_the_type_only_filters(self, records, posted) -> None:
        lines = "".join(export(records, "jsonl", doc_type="invoice")).strip().splitlines()
        assert lines
        for line in lines:
            assert json.loads(line)["doc_type"] == "invoice"


def test_the_filename_says_what_and_when() -> None:
    assert suggested_filename("csv").startswith("docuflow-ledger-")
    assert suggested_filename("jsonl", "invoice").endswith(".jsonl")
    assert "contract" in suggested_filename("csv", doc_type="contract")
    assert "invoice" in suggested_filename("csv", doc_type="invoice", record_kind="payable"), (
        "the document type is what was asked for, so it names the file"
    )
