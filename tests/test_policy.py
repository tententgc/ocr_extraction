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
