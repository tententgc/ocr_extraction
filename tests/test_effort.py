"""Time saved, and the ways that number gets inflated.

The brief asks for accuracy, coverage and time saved. The first two are
measured. The third is a model of what people were doing before, and a model
presented like a measurement is how a project reports a saving nobody can find
afterwards - so what is tested here is mostly the arithmetic refusing to
flatter itself.
"""

from __future__ import annotations

from docuflow.evaluation.effort import (
    ASSISTED_REVIEW_MINUTES,
    CROSS_CHECK_MINUTES,
    KEYING_MINUTES,
    SILENT_ERROR_MINUTES,
    effort_saved,
)


class TestWhatItCosts:
    def test_a_document_that_posts_costs_nothing(self):
        effort = effort_saved(documents=1, auto_posted=1, silent_errors=0, machine_seconds=60)
        assert effort.assisted_minutes == 0.0
        assert effort.manual_minutes == KEYING_MINUTES + CROSS_CHECK_MINUTES

    def test_a_held_document_is_not_free(self):
        """Counting a review as free is how the number gets inflated.

        The reviewer still opens it, still reads the values, still decides. What
        the system saved them is the keying and the cross-checking, not the
        judgement.
        """
        effort = effort_saved(documents=1, auto_posted=0, silent_errors=0, machine_seconds=60)
        assert effort.assisted_minutes == ASSISTED_REVIEW_MINUTES
        assert effort.saved_minutes < effort.manual_minutes

    def test_nothing_automated_still_saves_something(self):
        """Even holding everything, the values and the checks are on screen."""
        effort = effort_saved(documents=10, auto_posted=0, silent_errors=0, machine_seconds=600)
        assert 0 < effort.saved_share < 1

    def test_everything_automated_and_nothing_wrong_saves_all_of_it(self):
        effort = effort_saved(documents=10, auto_posted=10, silent_errors=0, machine_seconds=600)
        assert effort.saved_share == 1.0


class TestItCannotFlatterItself:
    def test_a_posted_error_is_charged_against_the_saving(self):
        """A system that posts everything must not look efficient.

        Somebody finds the wrong record at reconciliation, traces it and raises
        a correction, and that costs far more than the review it skipped.
        """
        clean = effort_saved(documents=20, auto_posted=20, silent_errors=0, machine_seconds=600)
        sloppy = effort_saved(documents=20, auto_posted=20, silent_errors=1, machine_seconds=600)

        assert sloppy.saved_minutes < clean.saved_minutes
        assert sloppy.rework_minutes == SILENT_ERROR_MINUTES

    def test_enough_silent_errors_wipe_the_saving_out(self):
        """The arithmetic has to be able to say the system made things worse."""
        effort = effort_saved(documents=20, auto_posted=20, silent_errors=5, machine_seconds=600)
        assert effort.saved_minutes < 0
        assert effort.saved_share < 0

    def test_machine_time_is_reported_but_never_subtracted(self):
        """A model reading a page is not a person waiting for it."""
        quick = effort_saved(documents=5, auto_posted=3, silent_errors=0, machine_seconds=10)
        slow = effort_saved(documents=5, auto_posted=3, silent_errors=0, machine_seconds=10_000)

        assert quick.saved_minutes == slow.saved_minutes
        assert slow.machine_minutes > quick.machine_minutes

    def test_an_empty_corpus_is_not_a_division_by_zero(self):
        effort = effort_saved(documents=0, auto_posted=0, silent_errors=0, machine_seconds=0)
        assert effort.saved_share == 0.0
        assert effort.to_json()["days_per_thousand_documents"] == 0.0


class TestTheAssumptionsTravelWithTheAnswer:
    def test_every_number_it_rests_on_is_in_the_output(self):
        report = effort_saved(documents=4, auto_posted=2, silent_errors=0,
                              machine_seconds=120).to_json()
        assumptions = report["assumptions"]

        assert assumptions["keying_minutes_per_document"] == KEYING_MINUTES
        assert assumptions["cross_check_minutes_per_document"] == CROSS_CHECK_MINUTES
        assert assumptions["assisted_review_minutes_per_held_document"] == ASSISTED_REVIEW_MINUTES
        assert assumptions["minutes_to_clean_up_one_silent_error"] == SILENT_ERROR_MINUTES

    def test_it_scales_to_a_number_a_manager_can_use(self):
        report = effort_saved(documents=10, auto_posted=7, silent_errors=0,
                              machine_seconds=600).to_json()
        assert report["manual_days_per_thousand_documents"] > report["days_per_thousand_documents"]
