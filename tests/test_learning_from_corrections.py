"""The feedback loop, now that something reads it back.

Corrections were recorded and reported to a person for a long time before
anything in the pipeline consulted them. These tests are about the shape of
what it learns, because the wrong shape is actively dangerous.

**Trust, never values.** A correction says what the right answer was on one
document. Carrying that value forward would put one invoice's total on another,
which is the exact failure this system exists to prevent. So the tests assert
that confidence moves and that no extracted value ever does.

**A pattern, not an incident.** Three edits to one field on one document is one
person fixing one page; the same field wrong across three documents is a layout
that cannot be read.

**Recoverable.** Distrust sends documents to review, review produces
corrections. Without a window that loop only tightens, and a vendor read badly
once would be distrusted forever.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from docuflow.db import transaction
from docuflow.store.corrections import (
    MIN_DOCUMENTS,
    UNRELIABLE_FIELD_FACTOR,
    WINDOW_DAYS,
    CorrectionStore,
)

VENDOR = "V-1001"


@pytest.fixture
def corrections(settings) -> CorrectionStore:
    return CorrectionStore(settings.db_path)


def _correct(
    store: CorrectionStore,
    document_id: str,
    field: str = "total_amount",
    vendor: str | None = VENDOR,
    doc_type: str = "invoice",
) -> None:
    store.record(document_id, doc_type, field, "1.00", "2.00", vendor_key=vendor)


def _age(settings, days: int) -> None:
    """Push every recorded correction back in time."""
    when = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with transaction(settings.db_path) as connection:
        connection.execute("UPDATE corrections SET created_at = ?", (when,))


class TestItLearnsAPatternNotAnIncident:
    def test_one_document_is_not_a_pattern(self, corrections) -> None:
        _correct(corrections, "DOC-1")
        assert corrections.unreliable_fields("invoice", VENDOR) == {}

    def test_repeated_edits_to_one_document_are_still_one_document(self, corrections) -> None:
        """One person fixing one page, three times. That is not a layout problem."""
        for _ in range(5):
            _correct(corrections, "DOC-1")
        assert corrections.unreliable_fields("invoice", VENDOR) == {}, (
            "counting edits rather than documents turns one bad page into a "
            "permanent verdict about the vendor"
        )

    def test_the_same_field_across_enough_documents_is(self, corrections) -> None:
        for index in range(MIN_DOCUMENTS):
            _correct(corrections, f"DOC-{index}")
        assert corrections.unreliable_fields("invoice", VENDOR) == {
            "total_amount": MIN_DOCUMENTS
        }

    def test_it_is_scoped_to_the_vendor(self, corrections) -> None:
        """One supplier's layout problem is not every supplier's."""
        for index in range(MIN_DOCUMENTS):
            _correct(corrections, f"DOC-{index}")
        assert corrections.unreliable_fields("invoice", "V-9999") == {}

    def test_it_is_scoped_to_the_document_type(self, corrections) -> None:
        for index in range(MIN_DOCUMENTS):
            _correct(corrections, f"DOC-{index}")
        assert corrections.unreliable_fields("receipt", VENDOR) == {}

    def test_an_unknown_vendor_teaches_nothing(self, corrections) -> None:
        """It cannot be attributed to a layout, so it must not be applied to one."""
        for index in range(MIN_DOCUMENTS + 2):
            _correct(corrections, f"DOC-{index}", vendor=None)
        assert corrections.unreliable_fields("invoice", None) == {}


class TestItRecovers:
    def test_old_corrections_age_out(self, settings, corrections) -> None:
        for index in range(MIN_DOCUMENTS):
            _correct(corrections, f"DOC-{index}")
        assert corrections.unreliable_fields("invoice", VENDOR)

        _age(settings, WINDOW_DAYS + 30)
        assert corrections.unreliable_fields("invoice", VENDOR) == {}, (
            "without a window, distrust sends documents to review, review "
            "produces corrections, and the count climbs on its own"
        )

    def test_corrections_just_inside_the_window_still_count(self, settings, corrections) -> None:
        for index in range(MIN_DOCUMENTS):
            _correct(corrections, f"DOC-{index}")
        _age(settings, WINDOW_DAYS - 5)
        assert corrections.unreliable_fields("invoice", VENDOR)


class TestWhatThePipelineDoesWithIt:
    """The important half: what it is allowed to change."""

    def _run_with_history(self, settings, make_run, corrections, fields: dict):
        from docuflow.pipeline import Pipeline

        for index in range(MIN_DOCUMENTS):
            _correct(corrections, f"DOC-HIST-{index}")
        pipeline = Pipeline(settings)
        return pipeline._distrust_proven_fields(make_run(fields), VENDOR)

    def test_confidence_drops_for_the_proven_field(self, settings, master, make_run, corrections):
        original = make_run({"total_amount": 200090.0})
        before = original.field("total_amount").confidence
        after = self._run_with_history(
            settings, make_run, corrections, {"total_amount": 200090.0}
        ).field("total_amount")

        assert after.confidence < before
        assert after.confidence == pytest.approx(round(before * UNRELIABLE_FIELD_FACTOR, 4))

    def test_the_value_itself_is_never_touched(self, settings, master, make_run, corrections):
        """The whole point. A correction is evidence about trust, not a value.

        Carrying a past answer forward would put one invoice's total on the
        next invoice, which is precisely the silent error the system exists to
        prevent.
        """
        after = self._run_with_history(
            settings, make_run, corrections, {"total_amount": 200090.0}
        ).field("total_amount")
        assert after.value == 200090.0

    def test_other_fields_are_left_alone(self, settings, master, make_run, corrections):
        fields = {"total_amount": 200090.0, "invoice_number": "INV-1"}
        original = make_run(fields)
        after = self._run_with_history(settings, make_run, corrections, fields)
        assert after.field("invoice_number").confidence == pytest.approx(
            original.field("invoice_number").confidence
        )

    def test_the_reviewer_is_told_why(self, settings, master, make_run, corrections):
        """A score that moved without explanation is a score nobody can act on."""
        after = self._run_with_history(
            settings, make_run, corrections, {"total_amount": 200090.0}
        ).field("total_amount")
        assert any("corrected this field" in note for note in after.notes), after.notes

    def test_it_shows_up_in_the_trace(self, settings, master, make_run, corrections):
        run = self._run_with_history(settings, make_run, corrections, {"total_amount": 200090.0})
        steps = [step for step in run.trace if "correction history" in step.summary]
        assert steps and steps[0].detail["fields"] == {"total_amount": MIN_DOCUMENTS}

    def test_no_history_changes_nothing(self, settings, master, make_run):
        from docuflow.pipeline import Pipeline

        original = make_run({"total_amount": 200090.0})
        after = Pipeline(settings)._distrust_proven_fields(original, VENDOR)
        assert after is original, "a vendor with no history must cost nothing at all"

    def test_an_unidentified_vendor_changes_nothing(self, settings, master, make_run):
        from docuflow.pipeline import Pipeline

        original = make_run({"total_amount": 200090.0})
        assert Pipeline(settings)._distrust_proven_fields(original, None) is original
