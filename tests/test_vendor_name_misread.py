"""The rule written for two documents that posted while wrong.

Both were Thai, both were the same failure, and the evaluation on 2026-08-06
found them rather than confirming them:

    บริษัท สยามซัพพลาย จำกัด    read as    บริษัท สยามชัพพลาย จำกัด
    บริษัท ซันไรส์ ...          read as    บริษัท ชันไรส์ ...

`ซ` (U+0E0B) and `ช` (U+0E0A) differ by one loop. Nothing stopped them. The
arithmetic closed, the bank account matched, `vendor_known` passed, and
confidence landed on **0.843 and 0.840 against a bar of 0.840**.

Nothing else in the system could have. Grounding cannot catch an OCR misread -
the span exists in the OCR output, and the OCR output is what was wrong - and
two channels reading that same output both read `ชัพพลาย`. Every other member
of `MISREAD_SIGNALS` works by arithmetic disagreeing with itself, and a name
has no arithmetic.

But `find_vendor` had already resolved the supplier **by fuzzy match** and
recorded how far it stretched to do it. The register is authoritative about how
its own supplier is spelled, so a fuzzy resolution is a statement that the page
and the register disagree about a critical field.

Three things this file holds, and each is a way the rule could be wrong:

  * it fires when a name only matched fuzzily;
  * it stays silent when the register itself would call it the same name -
    an omitted "Co., Ltd." is not a misread, and firing there would tax every
    ordinary invoice;
  * it is a **warning**, so it holds nothing by itself. What stops the document
    is its membership of `MISREAD_SIGNALS`, and that costs a twentieth - enough
    to take a document sitting on the threshold below it, and no more.

No model is called anywhere here.
"""

from __future__ import annotations

from datetime import date

import pytest

from docuflow.confidence import MISREAD_SIGNALS
from docuflow.masterdata.repository import MasterDataRepository
from docuflow.rules import run_rules

from tests.test_rules import GOOD_INVOICE, build_context, result_for

RULE = "vendor_name_matches_register"

# The register's spelling for the Thai vendor in the working corpus.
THAI_VENDOR = "บริษัท สยามเทค ซัพพลาย จำกัด"
# What OCR returned on document 25: one loop different.
THAI_MISREAD = "บริษัท สยามเทค ชัพพลาย จำกัด"


def _finding(master, make_run, fields: dict):
    return result_for(run_rules(build_context(master, fields, make_run=make_run)), RULE)


class TestTheDocumentsThatPostedWhileWrong:
    """The live cases, against the corpus's own register."""

    @pytest.fixture
    def register(self, settings) -> MasterDataRepository:
        from docuflow.samples.real_world import master_rows

        repository = MasterDataRepository(settings.db_path)
        repository.seed(master_rows())
        return repository

    def test_a_one_loop_thai_misread_is_reported(self, register, make_run) -> None:
        fields = {**GOOD_INVOICE, "vendor_name": THAI_MISREAD, "vendor_tax_id": None}
        finding = _finding(register, make_run, fields)
        assert finding is not None and not finding.passed, (
            "a vendor resolved only by fuzzy match is the one signal that can "
            "catch a character misread inside a name"
        )

    def test_the_register_spelling_is_offered_as_the_fix(self, register, make_run) -> None:
        """The register is authoritative about its own supplier's name, and this
        suggestion cannot name a second field - so it is offered with
        confidence, unlike most."""
        fields = {**GOOD_INVOICE, "vendor_name": THAI_MISREAD, "vendor_tax_id": None}
        finding = _finding(register, make_run, fields)
        assert finding.fixes, "the correct spelling is known and must be offered"
        assert finding.fixes[0].suggested == THAI_VENDOR

    def test_the_correct_thai_name_passes(self, register, make_run) -> None:
        fields = {**GOOD_INVOICE, "vendor_name": THAI_VENDOR, "vendor_tax_id": None}
        finding = _finding(register, make_run, fields)
        assert finding is not None and finding.passed


class TestWhatMustNotFire:
    """A rule that taxes ordinary invoices is worse than no rule."""

    def test_an_exact_name_passes(self, master, make_run) -> None:
        assert _finding(master, make_run, GOOD_INVOICE).passed

    def test_an_omitted_legal_form_is_not_a_misread(self, master, make_run) -> None:
        """"Siam Tech Supply" for "Siam Tech Supply Co., Ltd." is an invoice
        being terse. Firing here would put a warning on a large share of real
        documents and teach reviewers to ignore the chip."""
        fields = {**GOOD_INVOICE, "vendor_name": "Siam Tech Supply"}
        finding = _finding(master, make_run, fields)
        assert finding is not None and finding.passed, finding.message

    def test_a_tax_id_match_does_not_judge_the_name(self, master, make_run) -> None:
        """A tax ID is an exact identifier and resolves the vendor without the
        name being used at all, so there is no name comparison to make."""
        fields = {**GOOD_INVOICE, "vendor_name": "S T Supply"}
        finding = _finding(master, make_run, fields)
        assert finding is not None and finding.passed
        assert finding.detail.get("method") == "tax id"

    def test_an_unknown_vendor_is_not_this_rule_s_business(self, master, make_run) -> None:
        """`vendor_known` blocks it. Two findings about one problem is how a
        review queue becomes noise."""
        fields = {**GOOD_INVOICE, "vendor_name": "Entirely Different Ltd.", "vendor_tax_id": None}
        assert _finding(master, make_run, fields) is None

    def test_no_vendor_name_at_all(self, master, make_run) -> None:
        fields = {**GOOD_INVOICE, "vendor_name": None, "vendor_tax_id": None}
        assert _finding(master, make_run, fields) is None


class TestItHoldsNothingByItself:
    """A warning that holds documents is a blocking rule that has not admitted
    it, and it is how a human-in-the-loop system ends up with a human in every
    loop."""

    def test_it_is_a_warning(self, master, make_run) -> None:
        fields = {**GOOD_INVOICE, "vendor_name": "Siam Tekh Suply Co., Ltd.", "vendor_tax_id": None}
        finding = _finding(master, make_run, fields)
        assert finding is not None and not finding.passed
        assert finding.severity == "warning"

    def test_a_warning_does_not_block_the_document(self, master, make_run) -> None:
        from docuflow.routing import decide

        fields = {**GOOD_INVOICE, "vendor_name": "Siam Tekh Suply Co., Ltd.", "vendor_tax_id": None}
        findings = run_rules(build_context(master, fields, make_run=make_run))
        blocking = [f for f in findings if not f.passed and f.severity == "blocking"]
        assert not blocking, f"nothing here may block: {[f.rule_id for f in blocking]}"


class TestItIsWhatStopsTheDocument:
    """Not by blocking - by saying the reading is suspect."""

    def test_it_counts_as_a_misread_signal(self) -> None:
        assert RULE in MISREAD_SIGNALS, (
            "without this the rule is a note nobody acts on: it is a warning, "
            "so it holds nothing, and the confidence penalty is the entire "
            "mechanism by which it stops a document"
        )

    def test_a_document_on_the_threshold_falls_below_it(self) -> None:
        """The arithmetic that closes the two live cases.

        Both posted at 0.843 and 0.840 against a 0.840 bar. A warning-severity
        misread signal costs a twentieth, which is the difference.
        """
        from docuflow.confidence import MISREAD_PENALTY

        penalty = MISREAD_PENALTY["warning"]
        for observed in (0.843, 0.840):
            assert observed * (1 - penalty) < 0.840, (
                f"{observed} must fall below the 84% bar, not sit on it"
            )

    def test_three_unrelated_warnings_still_post(self) -> None:
        """The other half of the invariant. A warning has to be able to survive
        the threshold or it is a block in disguise: a page read at 95% clears an
        84% bar with two warnings against it, and is only held once three of
        them agree something is off."""
        from docuflow.confidence import MISREAD_PENALTY

        penalty = MISREAD_PENALTY["warning"]
        assert 0.95 * (1 - 2 * penalty) >= 0.84
        assert 0.95 * (1 - 3 * penalty) < 0.84
