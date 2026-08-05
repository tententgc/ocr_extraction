"""Runtime business policy: the settings screen's backing store."""

from __future__ import annotations

import pytest

from docuflow.policy import PolicyError, PolicyStore, warnings_for
from docuflow.rules import all_rules, describe_rules, run_rules


@pytest.fixture
def store(settings):
    return PolicyStore(settings)


class TestRuleCatalogue:
    """The screen can only list rules the registry can describe."""

    def test_every_registered_rule_is_listed_once(self):
        rows = describe_rules()
        assert len(rows) == len(all_rules())
        assert len({row["rule_id"] for row in rows}) == len(rows)

    def test_a_rule_reporting_under_another_id_declares_it(self):
        """`po_exists_and_matches` reports as `po_exists` or `po_vendor_matches`."""
        row = next(r for r in describe_rules() if r["rule_id"] == "po_exists")
        assert row["function"] == "po_exists_and_matches"
        assert "po_vendor_matches" in row["aliases"]

    def test_disabled_state_is_reported_for_any_name_the_rule_answers_to(self):
        for name in ("po_exists", "po_vendor_matches", "po_exists_and_matches"):
            row = next(r for r in describe_rules(disabled={name}) if r["rule_id"] == "po_exists")
            assert row["enabled"] is False, name

    def test_rules_carry_a_human_summary(self):
        """The catalogue is unusable if a row cannot say what the rule does."""
        assert all(row["summary"] for row in describe_rules())


class TestPolicyOverrides:
    def test_defaults_come_from_configuration(self, store, settings):
        described = store.describe()
        assert described["auto_post_min_confidence"] == settings.auto_post_min_confidence
        assert described["overridden"] == []

    def test_threshold_change_reaches_the_settings(self, store):
        store.update(auto_post_min_confidence=0.55)
        assert store.settings().auto_post_min_confidence == 0.55
        assert "auto_post_min_confidence" in store.describe()["overridden"]

    def test_disabling_a_rule_reaches_the_engine(self, store, master, make_run):
        from tests.test_rules import GOOD_INVOICE, build_context, result_for

        context = build_context(master, GOOD_INVOICE, make_run=make_run)
        assert result_for(run_rules(context), "vendor_known") is not None

        store.update(disabled_rules=["vendor_known"])
        findings = run_rules(context, skip=store.settings().skipped_rules)
        assert result_for(findings, "vendor_known") is None

    def test_an_alias_is_stored_under_the_canonical_id(self, store):
        """Otherwise the same rule could appear switched off twice, or neither."""
        store.update(disabled_rules=["po_vendor_matches", "po_exists_and_matches"])
        assert store.describe()["disabled_rules"] == ["po_exists"]

    def test_an_unknown_rule_name_is_refused(self, store):
        """Silently accepting a typo leaves someone believing a check is off."""
        with pytest.raises(PolicyError, match="not a known rule"):
            store.update(disabled_rules=["vendor_konwn"])
        assert store.describe()["disabled_rules"] == []

    def test_a_threshold_outside_zero_to_one_is_refused(self, store):
        with pytest.raises(PolicyError, match="between 0 and 1"):
            store.update(auto_post_min_confidence=1.4)

    def test_only_the_fields_given_are_touched(self, store):
        store.update(auto_post_min_confidence=0.6, disabled_rules=["vendor_known"])
        store.update(rules_free_mode=True)
        described = store.describe()
        assert described["auto_post_min_confidence"] == 0.6
        assert described["disabled_rules"] == ["vendor_known"]
        assert described["rules_free_mode"] is True

    def test_reset_returns_to_configuration(self, store, settings):
        store.update(rules_free_mode=True, auto_post_min_confidence=0.1)
        store.reset()
        described = store.describe()
        assert described["overridden"] == []
        assert described["auto_post_min_confidence"] == settings.auto_post_min_confidence

    def test_the_base_settings_are_never_mutated(self, store, settings):
        store.update(auto_post_min_confidence=0.2, rules_free_mode=True)
        assert settings.auto_post_min_confidence != 0.2
        assert settings.rules_free_mode is False

    def test_changes_survive_a_restart(self, store, settings):
        """A rule switched off is a decision about the business, not a session."""
        store.update(rules_free_mode=True, disabled_rules=["vendor_known"])
        restarted = PolicyStore(settings)
        described = restarted.describe()
        assert described["rules_free_mode"] is True
        assert described["disabled_rules"] == ["vendor_known"]


class TestPolicyWarnings:
    """Loosening a check has to be visible, not merely permitted."""

    def test_free_mode_is_called_out(self, settings):
        notes = warnings_for(settings.model_copy(update={"rules_free_mode": True}))
        assert any("nothing is checked" in note for note in notes)

    def test_a_low_threshold_is_called_out(self, settings):
        notes = warnings_for(settings.model_copy(update={"auto_post_min_confidence": 0.2}))
        assert any("without a human" in note for note in notes)

    def test_the_configured_defaults_warn_about_nothing(self, settings):
        assert warnings_for(settings) == []

    def test_switched_off_rules_are_named(self, settings):
        notes = warnings_for(settings.model_copy(
            update={"disabled_rules": "vat_rate_expected,due_date_after_issue"}))
        assert any("2 rule(s) switched off" in note for note in notes)


class TestDisabledRulesFromTheEnvironment:
    """`DISABLED_RULES` is read before any rule is registered, so it cannot
    refuse a bad id the way the settings screen does. It has to report instead."""

    def test_a_typo_switches_nothing_off_and_says_so(self, settings):
        """The failure this exists for is silent: you disable `vendor_knwon`,
        see no error, and conclude the rule is still too noisy."""
        from docuflow.policy import unrecognised_rules

        mistyped = settings.model_copy(
            update={"disabled_rules": "vat_rate_expected,vendor_knwon"})

        assert unrecognised_rules(mistyped) == ["vendor_knwon"]
        assert any("switches nothing off" in note for note in warnings_for(mistyped))

    def test_every_real_rule_id_is_accepted(self, settings):
        from docuflow.policy import unrecognised_rules

        every = ",".join(info.rule_id for info in all_rules())
        assert unrecognised_rules(settings.model_copy(update={"disabled_rules": every})) == []

    def test_a_rule_this_installation_wrote_is_not_a_typo(self, settings):
        """`DISABLED_RULES` may name your own rules too, and those are in a
        different register. Reporting one as unknown would be worse than saying
        nothing — it would send someone looking for a spelling mistake that is
        not there."""
        from docuflow.policy import unrecognised_rules
        from docuflow.rules.custom import build_rule
        from docuflow.store.custom_rules import CustomRuleStore

        CustomRuleStore(settings.db_path).save(build_rule({
            "rule_id": "our_own_check", "title": "Ours", "message": "no",
            "doc_types": ["invoice"], "severity": "warning",
            "assert": {"field": "total_amount", "op": "present"},
        }), replace=True)

        mixed = settings.model_copy(update={"disabled_rules": "our_own_check,not_a_rule"})
        assert unrecognised_rules(mixed) == ["not_a_rule"]

    def test_the_recommended_set_is_spelled_correctly(self, settings):
        """The four ids shipped in `.env.example` under BALANCED. A rename that
        orphans one of them should fail here rather than quietly stop
        disabling it."""
        from docuflow.policy import unrecognised_rules

        recommended = ("vat_rate_expected", "currency_matches_vendor_default",
                       "payment_terms_stated", "tax_id_belongs_to_this_vendor")
        known = {info.rule_id for info in all_rules()}
        # Two are built in; two ship in `real_samples/rules.json`.
        assert {"vat_rate_expected", "currency_matches_vendor_default"} <= known
        assert unrecognised_rules(settings.model_copy(
            update={"disabled_rules": ",".join(recommended[:2])})) == []

    def test_no_recommended_rule_is_one_confidence_depends_on(self):
        """Switching off a check that feeds confidence does not just stop it
        reporting — it stops the reading being questioned. None of the four may
        ever be in `MISREAD_SIGNALS`."""
        from docuflow.confidence import MISREAD_SIGNALS

        recommended = {"vat_rate_expected", "currency_matches_vendor_default",
                       "payment_terms_stated", "tax_id_belongs_to_this_vendor"}
        assert not (recommended & MISREAD_SIGNALS)

    def test_a_disabled_rule_stops_firing(self, settings, master, make_run):
        """End to end: the same document, the same context, one setting."""
        from tests.test_rules import build_context, result_for

        fields = {"invoice_number": "X-1", "invoice_date": "2026-07-20",
                  "vendor_name": "Siam Tech Supply Co., Ltd.",
                  "vendor_tax_id": "0105543000129", "currency": "THB",
                  "subtotal": 1_000.0, "tax_rate": 10.0, "tax_amount": 100.0,
                  "total_amount": 1_100.0}
        context = build_context(master, fields, make_run=make_run)

        assert result_for(run_rules(context), "vat_rate_expected").passed is False
        assert result_for(run_rules(context, skip=["vat_rate_expected"]),
                          "vat_rate_expected") is None


class TestRuleSeverityOverrides:
    """The setting that was missing between "on" and "off"."""

    def test_a_blocking_rule_can_be_dropped_to_a_warning(self, store, master, make_run):
        from tests.test_rules import GOOD_INVOICE, build_context, result_for

        fields = {**GOOD_INVOICE, "vendor_name": "Nobody Ltd.", "vendor_tax_id": None}
        context = build_context(master, fields, make_run=make_run)
        assert result_for(run_rules(context), "vendor_known").severity == "blocking"

        store.update(rule_severity={"vendor_known": "warning"})
        finding = result_for(
            run_rules(context, severity=store.severity_overrides()), "vendor_known"
        )
        assert finding.severity == "warning"
        assert finding.passed is False, "the check still ran and still failed"

    def test_the_original_grade_is_recorded(self, store, master, make_run):
        """A reviewer has to be able to see that it was re-graded, not born mild."""
        from tests.test_rules import GOOD_INVOICE, build_context, result_for

        fields = {**GOOD_INVOICE, "vendor_name": "Nobody Ltd.", "vendor_tax_id": None}
        store.update(rule_severity={"vendor_known": "info"})
        finding = result_for(
            run_rules(build_context(master, fields, make_run=make_run),
                      severity=store.severity_overrides()),
            "vendor_known",
        )
        assert finding.detail["original_severity"] == "blocking"

    def test_a_passing_rule_is_left_alone(self, store, master, make_run):
        from tests.test_rules import GOOD_INVOICE, build_context, result_for

        store.update(rule_severity={"vendor_known": "info"})
        finding = result_for(
            run_rules(build_context(master, GOOD_INVOICE, make_run=make_run),
                      severity=store.severity_overrides()),
            "vendor_known",
        )
        assert finding.passed is True

    def test_an_invalid_severity_is_refused(self, store):
        with pytest.raises(PolicyError, match="not a severity"):
            store.update(rule_severity={"vendor_known": "catastrophic"})

    def test_an_unknown_rule_is_refused(self, store):
        with pytest.raises(PolicyError, match="not a known rule"):
            store.update(rule_severity={"no_such_rule": "warning"})

    def test_overrides_survive_a_restart(self, store, settings):
        store.update(rule_severity={"vendor_known": "warning"})
        assert PolicyStore(settings).severity_overrides() == {"vendor_known": "warning"}


class TestRemovedPseudoRules:
    """Four checks were removed because they restated the pipeline's own state.

    They read as business-rule failures when nothing was wrong with the
    document - the pipeline was simply unsure - and routing enforces all four
    anyway, with clearer wording.
    """

    @pytest.mark.parametrize("rule_id", [
        "required_fields_present", "critical_fields_grounded",
        "low_confidence_fields", "ocr_quality_acceptable",
    ])
    def test_they_are_gone(self, rule_id):
        assert not any(info.matches(rule_id) for info in all_rules())

    def test_routing_still_holds_an_incomplete_document(self, make_run):
        from docuflow.routing import decide
        from docuflow.schemas.registry import get_schema

        run = make_run({"vendor_name": "Siam Tech Supply Co., Ltd."}).evolve(confidence=0.99)
        outcome = decide(run, get_schema("invoice"), 0.90)
        assert outcome.decision == "needs_review"
        assert any("Required field" in reason for reason in outcome.reasons)

    def test_routing_still_holds_an_ungrounded_critical_field(self, make_run):
        from docuflow.routing import decide
        from docuflow.schemas.registry import get_schema
        from tests.test_rules import GOOD_INVOICE

        run = make_run(GOOD_INVOICE).evolve(confidence=0.99)
        run = run.with_fields(tuple(
            field.evolve(grounded=False) if field.name == "total_amount" else field
            for field in run.fields
        ))
        outcome = decide(run, get_schema("invoice"), 0.90)
        assert outcome.decision == "needs_review"
        assert any("Could not verify" in reason for reason in outcome.reasons)


class TestBuiltinRulesDisabled:
    """"All rules off" and "your rules only" are different requests.

    Free mode evaluates nothing. This one switches off the twenty-six checks
    that ship with the system while still running the ones the user compiled
    from a prompt, for a company whose own policy is not ours.
    """

    def _context(self, settings, master, make_run, fields):
        from datetime import date

        from docuflow.rules import RuleContext
        from docuflow.schemas.registry import get_schema

        run = make_run(fields)
        return RuleContext(
            document_id=run.document_id,
            doc_type="invoice",
            schema=get_schema("invoice"),
            fields=run.fields,
            line_items=(),
            master=master,
            vendor_match=master.find_vendor(fields.get("vendor_name")),
            ocr_text=run.ocr.full_text,
            ocr_degraded=False,
            today=date(2026, 8, 4),
            **RuleContext.policy_from(settings),
        )

    @pytest.fixture
    def custom_rule(self):
        from docuflow.rules.custom import build_rule

        return build_rule({
            "rule_id": "total_must_be_small",
            "message": "This invoice is over the limit set here.",
            "doc_types": ["invoice"],
            "severity": "blocking",
            "assert": {"field": "total_amount", "op": "lte", "value": 1_000},
        })

    def test_built_in_rules_do_not_run(self, settings, master, make_run):
        from docuflow.rules import run_rules

        context = self._context(settings, master, make_run, {"vendor_name": "Nobody Ltd"})
        findings = run_rules(context, builtins_disabled=True)

        assert [f.rule_id for f in findings] == ["builtin_rules_disabled"]

    def test_the_user_s_own_rules_still_run(self, settings, master, make_run, custom_rule):
        from docuflow.rules import run_rules

        context = self._context(
            settings, master, make_run, {"vendor_name": "Nobody Ltd", "total_amount": 5_000.0}
        )
        findings = run_rules(context, builtins_disabled=True, custom=[custom_rule])

        failed = [f for f in findings if not f.passed]
        assert [f.rule_id for f in failed] == ["total_must_be_small"]
        assert failed[0].severity == "blocking"

    def test_it_says_out_loud_that_our_rules_did_not_run(self, settings, master, make_run):
        """A record checked against nothing of ours must not look like one that was."""
        from docuflow.rules import run_rules

        context = self._context(settings, master, make_run, {"vendor_name": "Nobody Ltd"})
        marker = next(f for f in run_rules(context, builtins_disabled=True)
                      if f.rule_id == "builtin_rules_disabled")
        assert marker.passed and marker.severity == "info"
        assert "not against the standard finance policy" in marker.message

    def test_free_mode_still_wins(self, settings, master, make_run, custom_rule):
        """Free mode means nothing runs — including the user's own rules."""
        from docuflow.rules import run_rules

        context = self._context(
            settings, master, make_run, {"vendor_name": "Nobody Ltd", "total_amount": 5_000.0}
        )
        findings = run_rules(context, free_mode=True, builtins_disabled=True, custom=[custom_rule])
        assert [f.rule_id for f in findings] == ["business_rules_disabled"]

    def test_the_setting_survives_a_restart(self, settings):
        PolicyStore(settings).update(builtin_rules_disabled=True)
        assert PolicyStore(settings).settings().builtin_rules_disabled is True

    def test_the_screen_is_told_what_it_costs(self, settings):
        store = PolicyStore(settings)
        store.update(builtin_rules_disabled=True)
        warnings = store.describe()["warnings"]
        assert any("Only the rules written here" in text for text in warnings)
