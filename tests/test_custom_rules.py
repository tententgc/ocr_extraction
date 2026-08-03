"""User-authored rules: the expression language, its guards, and evaluation."""

from __future__ import annotations

import pytest

from docuflow.rules import run_rules
from docuflow.rules.custom import CustomRuleError, build_rule, run_custom_rule
from docuflow.rules.expressions import ExpressionError, describe, evaluate, validate
from docuflow.store.custom_rules import CustomRuleStore

INVOICE_FIELDS = frozenset({"total_amount", "purchase_order_ref", "invoice_date", "due_date",
                            "currency", "vendor_matched", "doc_type"})


def rule(**overrides):
    payload = {
        "rule_id": "high_value_needs_po",
        "message": "Invoices above 50,000 must quote a purchase order.",
        "doc_types": ["invoice"],
        "severity": "blocking",
        "when": {"field": "total_amount", "op": "gt", "value": 50_000},
        "assert": {"field": "purchase_order_ref", "op": "present"},
    }
    payload.update(overrides)
    return build_rule(payload)


class TestExpressionSafety:
    """What the language refuses is the point of it."""

    def test_an_unknown_field_is_refused(self):
        with pytest.raises(ExpressionError, match="not a field"):
            validate({"field": "profit_margin", "op": "gt", "value": 1}, INVOICE_FIELDS)

    def test_an_unknown_operator_is_refused(self):
        with pytest.raises(ExpressionError, match="not a supported comparison"):
            validate({"field": "total_amount", "op": "exec", "value": 1}, INVOICE_FIELDS)

    def test_a_comparison_with_nothing_to_compare_is_refused(self):
        with pytest.raises(ExpressionError, match="needs a 'value'"):
            validate({"field": "total_amount", "op": "gt"}, INVOICE_FIELDS)

    def test_runaway_nesting_is_refused(self):
        deep = {"field": "total_amount", "op": "gt", "value": 1}
        for _ in range(12):
            deep = {"not": deep}
        with pytest.raises(ExpressionError, match="nests deeper"):
            validate(deep, INVOICE_FIELDS)

    def test_an_empty_branch_is_refused(self):
        with pytest.raises(ExpressionError, match="non-empty"):
            validate({"all": []}, INVOICE_FIELDS)


class TestEvaluation:
    def test_numbers_survive_thousands_separators(self):
        """OCR'd amounts arrive as text more often than not."""
        expression = {"field": "total_amount", "op": "gt", "value": 50_000}
        assert evaluate(expression, {"total_amount": "200,090.00"}).holds is True
        assert evaluate(expression, {"total_amount": "1,090.00"}).holds is False

    def test_dates_are_ordered_as_dates(self):
        expression = {"field": "due_date", "op": "gte", "other_field": "invoice_date"}
        assert evaluate(expression, {"due_date": "2026-08-07", "invoice_date": "2026-07-08"}).holds
        assert not evaluate(expression, {"due_date": "2026-06-01", "invoice_date": "2026-07-08"}).holds

    def test_incomparable_values_do_not_satisfy_an_ordering(self):
        """A missing amount must not read as "greater than 50,000"."""
        expression = {"field": "total_amount", "op": "gt", "value": 50_000}
        assert evaluate(expression, {"total_amount": None}).holds is False
        assert evaluate(expression, {"total_amount": "n/a"}).holds is False

    def test_observed_values_are_collected_for_the_message(self):
        expression = {"all": [
            {"field": "total_amount", "op": "gt", "value": 1},
            {"field": "currency", "op": "eq", "value": "THB"},
        ]}
        outcome = evaluate(expression, {"total_amount": 90, "currency": "THB"})
        assert outcome.observed == {"total_amount": 90, "currency": "THB"}

    def test_a_condition_reads_back_as_a_sentence(self):
        """The author approves the sentence, so it has to match the logic."""
        assert describe({"field": "total_amount", "op": "gte", "value": 100}) == \
            "total_amount is at least 100"
        assert describe({"not": {"field": "currency", "op": "present"}}) == \
            "not (currency is present)"


class TestRuleConstruction:
    def test_a_rule_cannot_shadow_a_built_in(self):
        with pytest.raises(CustomRuleError, match="built-in rule"):
            rule(rule_id="vendor_known")

    def test_a_rule_needs_a_message_for_the_reviewer(self):
        with pytest.raises(CustomRuleError, match="message"):
            rule(message="")

    def test_an_unknown_document_type_is_refused(self):
        with pytest.raises(CustomRuleError, match="Unknown document type"):
            rule(doc_types=["purchase_invoice"])

    def test_a_test_that_can_never_fail_is_refused(self):
        """`vendor_matched present` is true even when it is False - a check that
        reports success without checking."""
        with pytest.raises(CustomRuleError, match="never fail"):
            rule(assert_=None, **{"assert": {"field": "vendor_matched", "op": "present"}})

    def test_a_multi_type_rule_may_only_use_shared_fields(self):
        with pytest.raises(CustomRuleError, match="not a field"):
            rule(doc_types=["invoice", "contract"],
                 **{"assert": {"field": "purchase_order_ref", "op": "present"}})

    def test_the_id_is_normalised(self):
        assert rule(rule_id="High Value / Needs PO!").rule_id == "high_value_needs_po"


class TestAgainstDocuments:
    def test_the_when_guard_keeps_the_rule_off_irrelevant_documents(self, master, make_run):
        from tests.test_rules import GOOD_INVOICE, build_context

        small = build_context(master, {**GOOD_INVOICE, "total_amount": 900.0,
                                       "purchase_order_ref": None}, make_run=make_run)
        assert run_custom_rule(rule(), small) is None

    def test_it_fails_the_document_it_was_written_for(self, master, make_run):
        from tests.test_rules import GOOD_INVOICE, build_context

        context = build_context(master, {**GOOD_INVOICE, "purchase_order_ref": None},
                                make_run=make_run)
        finding = run_custom_rule(rule(), context)
        assert finding is not None and finding.passed is False
        assert finding.severity == "blocking"
        assert finding.detail["custom"] is True

    def test_it_passes_a_compliant_document(self, master, make_run):
        from tests.test_rules import GOOD_INVOICE, build_context

        finding = run_custom_rule(rule(), build_context(master, GOOD_INVOICE, make_run=make_run))
        assert finding is not None and finding.passed is True

    def test_the_engine_runs_it_beside_the_built_ins(self, master, make_run):
        from tests.test_rules import GOOD_INVOICE, build_context, result_for

        context = build_context(master, {**GOOD_INVOICE, "purchase_order_ref": None},
                                make_run=make_run)
        findings = run_rules(context, custom=[rule()])
        assert result_for(findings, "high_value_needs_po").passed is False
        assert result_for(findings, "invoice_totals_consistent") is not None

    def test_a_broken_rule_does_not_stop_the_others(self, master, make_run):
        from tests.test_rules import GOOD_INVOICE, build_context, result_for

        class Exploding:
            rule_id = "boom"
            def applies_to(self, _doc_type):
                raise RuntimeError("bad rule")

        context = build_context(master, GOOD_INVOICE, make_run=make_run)
        findings = run_rules(context, custom=[Exploding()])
        assert result_for(findings, "invoice_totals_consistent").passed is True

    def test_a_disabled_custom_rule_does_not_run(self, master, make_run):
        from tests.test_rules import GOOD_INVOICE, build_context, result_for

        context = build_context(master, {**GOOD_INVOICE, "purchase_order_ref": None},
                                make_run=make_run)
        findings = run_rules(context, custom=[rule()], skip={"high_value_needs_po"})
        assert result_for(findings, "high_value_needs_po") is None


class TestStorage:
    def test_a_saved_rule_comes_back(self, settings):
        store = CustomRuleStore(settings.db_path)
        store.save(rule())
        assert [r.rule_id for r in store.list_rules()] == ["high_value_needs_po"]

    def test_saving_twice_is_refused_unless_asked(self, settings):
        store = CustomRuleStore(settings.db_path)
        store.save(rule())
        with pytest.raises(CustomRuleError, match="already exists"):
            store.save(rule())
        store.save(rule(severity="warning"), replace=True)
        assert store.get("high_value_needs_po").severity == "warning"

    def test_only_enabled_rules_reach_the_engine(self, settings):
        store = CustomRuleStore(settings.db_path)
        store.save(rule())
        assert len(store.active_for("invoice")) == 1
        store.set_enabled("high_value_needs_po", False)
        assert store.active_for("invoice") == []

    def test_rules_are_filtered_by_document_type(self, settings):
        store = CustomRuleStore(settings.db_path)
        store.save(rule())
        assert store.active_for("receipt") == []

    def test_deleting_a_rule_that_is_not_there_reports_so(self, settings):
        assert CustomRuleStore(settings.db_path).delete("nothing") is False
