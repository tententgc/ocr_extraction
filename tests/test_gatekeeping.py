"""Did the gate open for the right documents, and did each rule earn its place.

The corpus states its expectation in every filename - `_True` should post,
`_False` should be held - so this compares two decisions rather than grading the
system against itself.

What the tests here mostly enforce is that the two ways of being wrong are never
added together. A missed defect is a wrong payment; a false alarm is a wasted
minute. A single "accuracy" number lets a system trade one for the other and
report an improvement.
"""

from __future__ import annotations

import pytest

from docuflow.evaluation.gatekeeping import Outcome, RuleScore, score_rules, summarize


def outcome(name: str, should_post: bool, posted: bool, expected=(), failed=()) -> Outcome:
    return Outcome(
        document=name, should_post=should_post, posted=posted, confidence=0.9,
        expected_findings=tuple(expected), failed_findings=tuple(failed),
    )


class TestTheTwoWaysOfBeingWrong:
    def test_posting_something_that_should_have_been_held_is_a_missed_defect(self):
        item = outcome("bad.pdf", should_post=False, posted=True)
        assert item.missed_defect is True
        assert item.false_alarm is False
        assert item.correct is False

    def test_holding_something_that_should_have_posted_is_a_false_alarm(self):
        item = outcome("fine.pdf", should_post=True, posted=False)
        assert item.false_alarm is True
        assert item.missed_defect is False

    def test_they_are_reported_separately_and_never_summed(self):
        """One wrong payment and one wasted minute are not two of anything."""
        report = summarize([
            outcome("a.pdf", should_post=False, posted=True),
            outcome("b.pdf", should_post=True, posted=False),
        ], [])

        assert report["missed_defects"] == 1
        assert report["false_alarms"] == 1
        assert "errors" not in report, "there is no combined figure to hide behind"

    def test_each_rate_is_over_its_own_population(self):
        """A missed-defect rate over the whole corpus falls simply by adding
        documents that were never going to be held."""
        report = summarize([
            outcome("bad.pdf", should_post=False, posted=True),
            outcome("held.pdf", should_post=False, posted=False),
            *[outcome(f"fine{i}.pdf", should_post=True, posted=True) for i in range(8)],
        ], [])

        assert report["should_hold"] == 2
        assert report["missed_defect_rate"] == 0.5, "one of the two that mattered"
        assert report["false_alarm_rate"] == 0.0

    def test_the_offending_documents_are_named(self):
        report = summarize([
            outcome("slipped.pdf", should_post=False, posted=True),
            outcome("nagged.pdf", should_post=True, posted=False),
        ], [])
        assert report["missed_defect_documents"] == ["slipped.pdf"]
        assert report["false_alarm_documents"] == ["nagged.pdf"]


class TestPerRule:
    def test_a_rule_that_fires_where_expected_has_full_recall(self):
        scores = score_rules([
            outcome("a.pdf", False, False, expected=["vendor_active"], failed=["vendor_active"]),
            outcome("b.pdf", True, True),
        ], ["vendor_active"])
        rule = next(item for item in scores if item.rule_id == "vendor_active")

        assert rule.recall == 1.0
        assert rule.precision == 1.0
        assert rule.silent_when_not_expected == 1

    def test_a_rule_that_stayed_silent_when_it_should_have_fired_is_named(self):
        """The dangerous direction, and the document is recorded so it can be
        looked at rather than inferred from a percentage."""
        scores = score_rules([
            outcome("a.pdf", False, True, expected=["duplicate_document"], failed=[]),
        ], ["duplicate_document"])
        rule = scores[0]

        assert rule.recall == 0.0
        assert rule.silent_when_expected == 1
        assert rule.missed_on == ["a.pdf"]

    def test_a_rule_firing_where_nothing_was_planted_is_recorded_not_hidden(self):
        scores = score_rules([
            outcome("a.pdf", True, False, expected=[], failed=["vat_rate_expected"]),
        ], ["vat_rate_expected"])
        rule = scores[0]

        assert rule.fired_unexpectedly == 1
        assert rule.fired_unexpectedly_on == ["a.pdf"]
        assert rule.tested is False, "nothing in the corpus claims it should fire"

    def test_a_rule_no_document_exercises_is_untested_not_passing(self):
        """Reporting an unexercised rule as 100% is how a corpus flatters
        itself. Untested is the honest word."""
        scores = score_rules([outcome("a.pdf", True, True)], ["po_currency_matches"])
        rule = scores[0]

        assert rule.tested is False
        assert rule.recall is None

    def test_a_rule_the_corpus_names_but_the_registry_does_not_is_still_scored(self):
        """Usually a rule that was renamed, and silently dropping it is exactly
        how that goes unnoticed."""
        scores = score_rules([
            outcome("a.pdf", False, False, expected=["renamed_rule"], failed=[]),
        ], ["vendor_active"])

        assert any(item.rule_id == "renamed_rule" for item in scores)

    def test_the_summary_separates_tested_from_untested(self):
        report = summarize(
            [outcome("a.pdf", False, False, expected=["vendor_active"], failed=["vendor_active"])],
            score_rules(
                [outcome("a.pdf", False, False, expected=["vendor_active"], failed=["vendor_active"])],
                ["vendor_active", "po_exists"],
            ),
        )

        assert report["rules_tested"] == 1
        assert report["rules_untested"] == 1
        assert report["rules_with_full_recall"] == 1
        assert report["rules_that_missed"] == []


class TestTheHeadline:
    def test_a_perfect_run(self):
        report = summarize([
            outcome("fine.pdf", should_post=True, posted=True),
            outcome("bad.pdf", should_post=False, posted=False),
        ], [])
        assert report["decision_accuracy"] == 1.0
        assert report["missed_defects"] == 0
        assert report["false_alarms"] == 0

    def test_holding_everything_is_not_a_good_score(self):
        """The failure mode a decision-accuracy figure has to be able to see."""
        report = summarize(
            [outcome(f"fine{i}.pdf", should_post=True, posted=False) for i in range(5)]
            + [outcome("bad.pdf", should_post=False, posted=False)],
            [],
        )
        assert report["missed_defects"] == 0
        assert report["false_alarm_rate"] == 1.0
        assert report["decision_accuracy"] < 0.2

    def test_an_empty_corpus_is_not_a_division_by_zero(self):
        assert summarize([], []) == {"documents": 0}
