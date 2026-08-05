"""What each severity is allowed to do.

Three words are only worth having if they behave differently, and for a while
two of them did not. A warning reduced confidence by 8% each, so three advisory
findings took a perfectly-read document from 95% to 72% - under an 84%
threshold - and sent it to a person who had nothing to correct. That is a
blocking rule that has not admitted it, and it is how a human-in-the-loop
system ends up with a human in every loop.

The split this file pins:

  blocking  stops the document. Routing holds it; a person corrects and
            re-checks, or rejects.
  warning   recorded, shown, carried into the record. Does not stop anything.
  info      recorded.

And, on a separate axis from severity: a failure that is evidence the *reading*
is wrong lowers confidence, because confidence is about the reading. A failure
that is evidence the *world* is wrong does not, because it isn't.
"""

from __future__ import annotations

import pytest

from docuflow.confidence import MISREAD_PENALTY, MISREAD_SIGNALS, document_confidence
from docuflow.models import ExtractedField, RuleFinding

REQUIRED = frozenset({"total_amount", "invoice_number"})
CRITICAL = frozenset({"total_amount"})


def read_well() -> tuple[ExtractedField, ...]:
    """A document every field of which was read cleanly and grounded."""
    return tuple(
        ExtractedField(name=name, value="x", confidence=0.95, grounded=True)
        for name in ("total_amount", "invoice_number", "vendor_name", "currency")
    )


def finding(rule_id: str, severity: str = "warning") -> RuleFinding:
    return RuleFinding(rule_id=rule_id, passed=False, severity=severity, message="")


def score(*findings: RuleFinding) -> float:
    return document_confidence(read_well(), findings, REQUIRED, CRITICAL)


class TestConfidenceIsAboutTheReading:
    def test_a_clean_reading_scores_on_its_fields(self):
        assert score() == pytest.approx(0.95, abs=0.01)

    def test_a_blocked_vendor_does_not_make_the_reading_doubtful(self):
        """It was read perfectly. It just cannot be paid.

        Scoring it down says "we probably misread this", which is false, and
        leaves the reviewer unable to tell a document we could not read from one
        we read fine and may not pay.
        """
        assert score(finding("vendor_active", "blocking")) == score()

    @pytest.mark.parametrize("rule_id", [
        "vendor_known", "po_exists", "duplicate_document", "vat_rate_expected",
        "contract_expiring_soon", "po_amount_within_tolerance",
    ])
    def test_facts_about_the_world_leave_confidence_alone(self, rule_id):
        assert score(finding(rule_id, "blocking")) == score()

    def test_a_document_that_does_not_add_up_is_a_doubtful_reading(self):
        """The arithmetic disagreeing means a character was misread, and the
        rest of the page came off the same engine."""
        assert score(finding("invoice_totals_consistent", "blocking")) < score()


class TestAWarningMustBeAbleToSurviveTheThreshold:
    THRESHOLD = 0.84

    def test_one_advisory_finding_still_clears_the_bar(self):
        assert score(finding("line_items_sum_to_subtotal")) > self.THRESHOLD

    def test_two_still_clear_it(self):
        assert score(
            finding("line_items_sum_to_subtotal"), finding("due_date_after_issue")
        ) > self.THRESHOLD

    def test_three_agreeing_that_something_is_off_do_not(self):
        """One advisory finding is noise; three are a pattern."""
        assert score(
            finding("line_items_sum_to_subtotal"),
            finding("due_date_after_issue"),
            finding("dates_not_in_future"),
        ) < self.THRESHOLD

    def test_a_warning_costs_a_quarter_of_what_a_blocking_one_does(self):
        assert MISREAD_PENALTY["warning"] * 4 == pytest.approx(MISREAD_PENALTY["blocking"])
        assert MISREAD_PENALTY["info"] == 0.0

    def test_rule_evidence_alone_never_sinks_a_well_read_page(self):
        from docuflow.confidence import MISREAD_FLOOR

        everything = [finding(rule_id, "blocking") for rule_id in MISREAD_SIGNALS]
        assert score(*everything) >= 0.95 * MISREAD_FLOOR - 0.01


class TestRoutingIsWhatStopsADocument:
    # Everything the invoice schema needs, so that what holds the document is
    # the finding under test and not a missing field.
    COMPLETE = {
        "invoice_number": "INV-1", "invoice_date": "2026-07-01",
        "vendor_name": "Siam Tech Supply Co., Ltd.", "total_amount": 107.0,
        "subtotal": 100.0, "tax_amount": 7.0, "purchase_order_ref": "PO-2026-0412",
        "bank_account": "1234567890",
    }

    def _run(self, make_run, findings, confidence=0.95):
        run = make_run(self.COMPLETE)
        return run.evolve(
            # Every critical field verified against the page, which is the other
            # thing routing insists on before it will post anything.
            fields=tuple(item.evolve(grounded=True) for item in run.fields),
            findings=tuple(findings),
            confidence=confidence,
        )

    def test_a_warning_does_not_hold_a_document(self, make_run):
        from docuflow.routing import decide
        from docuflow.schemas.registry import get_schema

        run = self._run(make_run, [finding("vat_rate_expected", "warning")])
        outcome = decide(run, get_schema("invoice"), 0.84)
        assert outcome.decision == "auto_post"

    def test_a_blocking_finding_does(self, make_run):
        from docuflow.routing import decide
        from docuflow.schemas.registry import get_schema

        run = self._run(make_run, [finding("vendor_active", "blocking")])
        outcome = decide(run, get_schema("invoice"), 0.84)
        assert outcome.decision == "needs_review"

    def test_a_warning_is_still_recorded_on_the_document(self, make_run):
        """Not holding it is not the same as hiding it."""
        run = self._run(make_run, [finding("vat_rate_expected", "warning")])
        assert [item.rule_id for item in run.failed_findings] == ["vat_rate_expected"]


class TestTheCatalogueTellsTheTruth:
    """A rule's declared severity is what the settings screen shows and what a
    person re-grades. It has to match what the rule actually emits, or the
    screen is describing a system that does not exist."""

    def test_every_rule_declares_the_severity_it_emits(self, master, make_run):
        from tests.test_rules import GOOD_INVOICE, build_context
        from docuflow.rules import load_all_rules
        from docuflow.rules.engine import all_rules

        load_all_rules()
        context = build_context(master, GOOD_INVOICE, make_run=make_run)

        mismatched = []
        for info in all_rules():
            try:
                result = info.function(context)
            except Exception:                     # not applicable to this document
                continue
            if result is None or result.passed:
                continue
            if result.rule_id == info.rule_id and result.severity != info.severity:
                mismatched.append((info.rule_id, info.severity, result.severity))

        assert not mismatched, f"declared severity differs from emitted: {mismatched}"

    def test_the_catalogue_says_what_each_severity_does(self):
        from docuflow.rules import load_all_rules
        from docuflow.rules.engine import describe_rules

        load_all_rules()
        for entry in describe_rules():
            assert entry["severity_means"], f"{entry['rule_id']} explains nothing"
            assert entry["affects_confidence"] == (entry["rule_id"] in MISREAD_SIGNALS)


class TestRoundingCannotPromoteADocument:
    """The score is stored, displayed, *and* compared against the auto-post
    threshold. Rounding to nearest let a document that scored 0.83951 present
    itself as 0.840 and clear a 0.840 bar it never reached - a presentation
    decision moving money.

    Half a thousandth is small and it is not a hypothetical margin: the
    2026-08-06 evaluation posted two documents at 0.840 and 0.843 against a
    0.840 threshold, so the whole population near the bar sits inside the
    rounding error.
    """

    def _score_of(self, raw: float) -> float:
        """Drive `document_confidence` to a chosen raw score.

        A clean reading scores its mean field confidence, so setting every
        field to `raw` sets the document score to `raw`.
        """
        fields = tuple(
            ExtractedField(name=name, value="x", confidence=raw, grounded=True)
            for name in ("total_amount", "invoice_number", "vendor_name")
        )
        return document_confidence(fields, (), REQUIRED, CRITICAL)

    def test_a_score_below_the_bar_stays_below_it(self):
        assert self._score_of(0.83951) < 0.84
        assert self._score_of(0.83999) < 0.84

    def test_a_score_that_reaches_the_bar_still_clears_it(self):
        """Flooring must not hold documents that genuinely qualified."""
        assert self._score_of(0.84) >= 0.84
        assert self._score_of(0.8409) >= 0.84

    def test_the_displayed_value_never_overstates_the_true_one(self):
        """`displayed >= threshold` has to imply `true >= threshold`, or the
        rounding is deciding the routing."""
        for raw in (0.5001, 0.7499, 0.83951, 0.9009, 0.9999):
            assert self._score_of(raw) <= raw + 1e-9, raw

    def test_it_is_still_three_decimals(self):
        """Flooring is not an excuse to leak float noise into the UI."""
        value = self._score_of(0.834567)
        assert value == round(value, 3)
