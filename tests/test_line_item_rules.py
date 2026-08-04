"""Rules about the rows, and the tools that stop an agent guessing.

An invoice *is* its line items, and until now a rule could see only how many
there were. "No single line over 50,000" - the most ordinary request in accounts
payable - could not be written at all: not as an expression, which had no way to
look inside a row, and not as an agent rule, which had no tool that could read
one.
"""

from __future__ import annotations

import pytest

from docuflow.models import LineItem
from docuflow.rules.custom import CustomRuleError, build_rule, run_custom_rule
from docuflow.rules.expressions import ExpressionError, describe, evaluate, validate
from docuflow.rules.expressions import ITEM_FIELDS, LINE_ITEMS

INVOICE_FIELDS = frozenset({"total_amount", "line_item_count", "currency"})

ROWS = [
    {"description": "Server rack", "quantity": 2.0, "unit_price": 30000.0, "amount": 60000.0},
    {"description": "Cabling", "quantity": 10.0, "unit_price": 1200.0, "amount": 12000.0},
    {"description": "", "quantity": 1.0, "unit_price": 500.0, "amount": 500.0},
]


def values(rows=ROWS, **extra):
    return {"total_amount": 72_500.0, "line_item_count": len(rows), LINE_ITEMS: rows, **extra}


class TestTheLanguage:
    def test_a_row_condition_is_checked_against_row_names(self):
        validate({"any_item": {"field": "amount", "op": "gt", "value": 50_000}}, INVOICE_FIELDS)

    def test_a_document_field_is_out_of_reach_inside_a_row(self):
        """`total_amount` is not a property of one line, and reading like it is
        would quietly compare the wrong thing."""
        with pytest.raises(ExpressionError, match="not a field"):
            validate({"any_item": {"field": "total_amount", "op": "gt", "value": 1}},
                     INVOICE_FIELDS)

    def test_a_row_name_is_out_of_reach_outside_one(self):
        with pytest.raises(ExpressionError, match="not a field"):
            validate({"field": "unit_price", "op": "gt", "value": 1}, INVOICE_FIELDS)

    def test_rows_do_not_nest(self):
        with pytest.raises(ExpressionError, match="no line items of its own"):
            validate({"any_item": {"every_item": {"field": "amount", "op": "present"}}},
                     INVOICE_FIELDS)

    def test_it_reads_back_as_a_sentence(self):
        assert describe({"any_item": {"field": "amount", "op": "gt", "value": 50_000}}) == \
            "any line item has amount is more than 50000"
        assert describe({"every_item": {"field": "description", "op": "present"}}) == \
            "every line item has description is present"

    def test_the_row_fields_are_the_ones_the_extractor_produces(self):
        assert ITEM_FIELDS == {"description", "quantity", "unit_price", "amount"}


class TestEvaluation:
    def test_any_item_finds_the_row_that_matches(self):
        outcome = evaluate({"any_item": {"field": "amount", "op": "gt", "value": 50_000}}, values())
        assert outcome.holds is True
        assert outcome.observed["line_items_matching"][0]["description"] == "Server rack"

    def test_any_item_is_false_when_no_row_matches(self):
        outcome = evaluate({"any_item": {"field": "amount", "op": "gt", "value": 99_000}}, values())
        assert outcome.holds is False

    def test_every_item_names_the_rows_that_failed(self):
        outcome = evaluate({"every_item": {"field": "description", "op": "present"}}, values())
        assert outcome.holds is False
        failing = outcome.observed["line_items_failing"]
        assert len(failing) == 1 and failing[0]["amount"] == 500.0

    def test_every_item_holds_when_all_rows_pass(self):
        assert evaluate({"every_item": {"field": "amount", "op": "present"}}, values()).holds

    def test_a_document_with_no_rows(self):
        """`any` over nothing is false; `every` over nothing is true. The row
        count is recorded either way, so the reviewer can see it was nought."""
        empty = values(rows=[])
        assert evaluate({"any_item": {"field": "amount", "op": "gt", "value": 1}}, empty).holds is False
        outcome = evaluate({"every_item": {"field": "description", "op": "present"}}, empty)
        assert outcome.holds is True
        assert outcome.observed["line_item_count"] == 0

    def test_one_rows_values_are_not_reported_as_the_documents(self):
        """Each row is evaluated in its own scope."""
        outcome = evaluate({"any_item": {"field": "amount", "op": "gt", "value": 50_000}}, values())
        assert "amount" not in outcome.observed


class TestAgainstADocument:
    def _context(self, master, make_run, rows):
        from tests.test_rules import GOOD_INVOICE, build_context
        from dataclasses import replace

        context = build_context(master, GOOD_INVOICE, make_run=make_run)
        return replace(context, line_items=tuple(
            LineItem(description=r["description"], quantity=r["quantity"],
                     unit_price=r["unit_price"], amount=r["amount"]) for r in rows))

    def test_the_rule_a_finance_team_actually_asks_for(self, master, make_run):
        rule = build_rule({
            "rule_id": "no_line_over_50k",
            "message": "A single line item exceeds 50,000 and needs a second signature.",
            "doc_types": ["invoice"], "severity": "blocking",
            "assert": {"not": {"any_item": {"field": "amount", "op": "gt", "value": 50_000}}},
        })
        assert "any line item has amount is more than 50000" in rule.explanation

        finding = run_custom_rule(rule, self._context(master, make_run, ROWS))
        assert finding.passed is False
        assert finding.detail["observed"]["line_items_matching"][0]["amount"] == 60000.0

    def test_it_passes_a_document_whose_rows_are_all_small(self, master, make_run):
        rule = build_rule({
            "rule_id": "no_line_over_50k", "message": "x",
            "doc_types": ["invoice"], "severity": "blocking",
            "assert": {"not": {"any_item": {"field": "amount", "op": "gt", "value": 50_000}}},
        })
        small = [{**row, "amount": 100.0} for row in ROWS]
        assert run_custom_rule(rule, self._context(master, make_run, small)).passed is True

    def test_a_rule_naming_a_row_field_at_the_top_level_is_refused_when_saved(self):
        with pytest.raises(CustomRuleError, match="not a field"):
            build_rule({
                "rule_id": "wrong_level", "message": "x", "doc_types": ["invoice"],
                "assert": {"field": "unit_price", "op": "lt", "value": 10},
            })


class TestConnectionsAreBounded:
    """Found while running this branch's tests: the connection cache was keyed by
    path and never evicted, so a process touching many databases held every one
    of them open - four descriptors apiece in WAL mode. A server uses one path
    and is unaffected; the evaluation harness, which isolates each run in its
    own database, is exactly the case that ran out."""

    def test_one_thread_holds_a_bounded_number_of_databases(self, tmp_path):
        from docuflow.db import MAX_CACHED_CONNECTIONS, _local, connect, init_db

        for index in range(MAX_CACHED_CONNECTIONS * 3):
            path = tmp_path / f"db{index}" / "d.db"
            init_db(path)
            connect(path).execute("SELECT 1").fetchone()

        assert len(_local.connections) == MAX_CACHED_CONNECTIONS

    def test_the_one_in_use_survives_and_still_works(self, tmp_path):
        from docuflow.db import MAX_CACHED_CONNECTIONS, connect, init_db

        first = tmp_path / "first" / "d.db"
        init_db(first)
        connect(first).execute("CREATE TABLE t (x INTEGER)")

        for index in range(MAX_CACHED_CONNECTIONS * 2):
            other = tmp_path / f"other{index}" / "d.db"
            init_db(other)
            connect(other).execute("SELECT 1").fetchone()

        # Evicted, so this reopens - and the data written through the old handle
        # is still there, because closing a connection is not losing a database.
        connect(first).execute("INSERT INTO t VALUES (1)")
        assert connect(first).execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1

    def test_a_connection_mid_transaction_is_not_closed_underneath_its_caller(self, tmp_path):
        from docuflow.db import MAX_CACHED_CONNECTIONS, connect, init_db, transaction

        held = tmp_path / "held" / "d.db"
        init_db(held)
        connect(held).execute("CREATE TABLE t (x INTEGER)")

        with transaction(held) as connection:
            connection.execute("INSERT INTO t VALUES (1)")
            for index in range(MAX_CACHED_CONNECTIONS * 2):
                other = tmp_path / f"busy{index}" / "d.db"
                init_db(other)
                connect(other).execute("SELECT 1").fetchone()
            # Still open, still inside the transaction it started.
            connection.execute("INSERT INTO t VALUES (2)")

        assert connect(held).execute("SELECT COUNT(*) FROM t").fetchone()[0] == 2
