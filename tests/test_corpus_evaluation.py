"""The corpus harness, and the two ways it can report a lie.

`docs/EVALUATION.md` quotes this harness. A number in a document nobody can
reproduce is a number nobody should believe, so the harness is the part that
has to be trustworthy before any figure it prints means anything.

Two of these tests exist because the failure was found in a real run and both
move the headline metrics in the flattering direction, which is the direction
nobody checks.

**A rule the corpus expects but the machine disabled.** `Settings` fills what
it was not given from a `.env`. This checkout has seven rules off and two of
them are rules corpus documents declare, so an unguarded run reports two
**missed defects** - the metric this system exists to drive to zero - where
nothing was missed and nothing was even asked.

**A transcript that was copied rather than read.** The text-layer fast path
returns the PDF's own text, and `reference_text` is that same text taken from
that same place. Character error rate is then zero by construction. Pooled with
the pages an engine actually recognised, it reports an OCR accuracy the OCR
never earned - and half this corpus arrives as digital PDF.

No model is called anywhere here.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from docuflow.config import Settings
from docuflow.evaluation.corpus import (
    TEXT_LAYER_PROVIDER,
    CorpusDocument,
    enable_rules,
    expected_but_disabled,
    load_manifest,
)
from docuflow.evaluation.report import print_corpus_report

CORPUS = Path("real_samples")


def _document(order: int, expects: tuple[str, ...] = (), posts: bool = True) -> CorpusDocument:
    return CorpusDocument(
        order=order,
        file=f"{order}_sample_{posts}.pdf",
        doc_type="invoice",
        posts=posts,
        ground_truth={"total_amount": 1000.0},
        expected_findings=expects,
        reference_text="TAX INVOICE",
        arrival="pdf",
        language="en",
    )


class TestTheManifest:
    """A corpus that scored nothing would print a perfect report."""

    def test_a_missing_manifest_raises_rather_than_scoring_nothing(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="real-samples"):
            load_manifest(tmp_path)

    def test_an_empty_manifest_raises(self, tmp_path) -> None:
        (tmp_path / "ground_truth.json").write_text('{"documents": []}', encoding="utf-8")
        with pytest.raises(ValueError, match="no documents"):
            load_manifest(tmp_path)

    @pytest.mark.skipif(not CORPUS.exists(), reason="the corpus is not generated here")
    def test_documents_come_back_in_run_order(self) -> None:
        """Document 15 is a duplicate of document 1 and cannot be detected
        until document 1 has posted. Alphabetical order puts 15 before 2."""
        orders = [document.order for document in load_manifest(CORPUS)]
        assert orders == sorted(orders)

    @pytest.mark.skipif(not CORPUS.exists(), reason="the corpus is not generated here")
    def test_the_expected_outcome_in_the_filename_matches_the_manifest(self) -> None:
        """The suffix is the claim the corpus makes before the system is asked.
        A manifest that disagrees with it grades against the wrong answer."""
        for document in load_manifest(CORPUS):
            stem = Path(document.file).stem
            assert stem.endswith("_True") or stem.endswith("_False"), document.file
            assert document.posts is stem.endswith("_True"), document.file


class TestARuleThatWasNeverAsked:
    """The dangerous one: it moves `missed_defects`, which means wrong payments."""

    def test_a_declared_rule_that_is_disabled_is_reported(self) -> None:
        settings = Settings(disabled_rules="po_amount_within_tolerance,vat_rate_expected")
        documents = (_document(1, ("po_amount_within_tolerance",)), _document(2))
        assert expected_but_disabled(settings, documents) == ["po_amount_within_tolerance"]

    def test_a_disabled_rule_no_document_declares_is_not_a_conflict(self) -> None:
        """Switching off a check nothing in the corpus tests is a configuration
        choice, not a hole in the measurement."""
        settings = Settings(disabled_rules="payment_terms_stated")
        assert expected_but_disabled(settings, (_document(1, ("vendor_known",)),)) == []

    def test_nothing_disabled_is_no_conflict(self) -> None:
        settings = Settings(disabled_rules="")
        assert expected_but_disabled(settings, (_document(1, ("vendor_known",)),)) == []

    def test_enabling_leaves_the_other_disabled_rules_alone(self) -> None:
        """Re-enabling what the corpus needs must not quietly switch on
        everything else the operator turned off for their own reasons."""
        settings = Settings(disabled_rules="a_rule,b_rule,c_rule")
        after = enable_rules(settings, ["b_rule"])
        assert after.skipped_rules == frozenset({"a_rule", "c_rule"})

    def test_enabling_nothing_returns_the_same_settings(self) -> None:
        settings = Settings(disabled_rules="a_rule")
        assert enable_rules(settings, []) is settings

    def test_the_conflict_is_resolved_by_enabling(self) -> None:
        settings = Settings(disabled_rules="po_amount_within_tolerance")
        documents = (_document(1, ("po_amount_within_tolerance",)),)
        after = enable_rules(settings, expected_but_disabled(settings, documents))
        assert expected_but_disabled(after, documents) == []

    @pytest.mark.skipif(not CORPUS.exists(), reason="the corpus is not generated here")
    def test_it_catches_this_checkout(self) -> None:
        """Not hypothetical. Run against the real corpus with the real
        `.env` this returns two rules, and both belong to documents whose whole
        purpose is to be held."""
        settings = Settings()
        conflict = expected_but_disabled(settings, load_manifest(CORPUS))
        assert conflict == sorted(conflict)
        for rule_id in conflict:
            assert rule_id in settings.skipped_rules


class TestOcrScoredOnWhatWasActuallyRead:
    """A transcript lifted from a text layer is not evidence about OCR."""

    def test_the_text_layer_provider_is_the_one_the_engine_reports(self) -> None:
        """The constant is a string match against `run.ocr.provider`, so a
        rename in the OCR layer silently stops excluding anything."""
        from docuflow.ocr.fallbacks import PdfTextLayerOcr

        assert PdfTextLayerOcr.name == TEXT_LAYER_PROVIDER

    def test_the_recognised_figure_leads_and_the_diluted_one_is_labelled(self) -> None:
        report = {
            "setup": {},
            "summary": {
                "documents": 2, "wall_clock_seconds": 1.0, "type_accuracy": 1.0,
                "field_accuracy": 1.0, "field_accuracy_with_near": 1.0,
                "fields_total": 2, "fields_correct": 2, "fields_near": 0,
                "fields_missing": 0, "auto_posted": 1, "auto_post_rate": 0.5,
                "silent_errors": 0, "silent_error_rate": 0.0,
                "silent_error_documents": [], "auto_post_precision": 1.0,
                "planted_defects": 0, "planted_defects_caught": 0,
                "defect_recall": 1.0, "average_confidence": 0.9,
            },
            # Pooled: dragged to near-perfect by the copied transcript.
            "ocr": {
                "documents": 2, "character_accuracy": 0.995,
                "character_error_rate": 0.005, "value_recall": 1.0,
                "values_found": 4, "values_expected": 4, "worst_documents": [],
            },
            # What an engine actually had to read.
            "ocr_recognised": {
                "documents": 1, "character_accuracy": 0.99,
                "character_error_rate": 0.01, "value_recall": 1.0,
                "values_found": 2, "values_expected": 2, "worst_documents": [],
            },
            "ocr_from_text_layer": 1,
            "documents": [],
        }
        out = io.StringIO()
        with redirect_stdout(out):
            print_corpus_report(report)
        printed = out.getvalue()

        assert "99.0%" in printed, "the recognised figure has to be the one on the line"
        assert "1 PDF(s) served from their own text" in printed
        assert "zero by" in printed and "construction" in printed
        assert "99.5%" in printed, "the diluted figure is still shown, but labelled"

    def test_with_nothing_recognised_the_section_says_so(self) -> None:
        """An all-PDF corpus measures no OCR at all, and must not print a
        character accuracy as though it had."""
        report = {
            "setup": {},
            "summary": {
                "documents": 1, "wall_clock_seconds": 1.0, "type_accuracy": 1.0,
                "field_accuracy": 1.0, "field_accuracy_with_near": 1.0,
                "fields_total": 1, "fields_correct": 1, "fields_near": 0,
                "fields_missing": 0, "auto_posted": 1, "auto_post_rate": 1.0,
                "silent_errors": 0, "silent_error_rate": 0.0,
                "silent_error_documents": [], "auto_post_precision": 1.0,
                "planted_defects": 0, "planted_defects_caught": 0,
                "defect_recall": 1.0, "average_confidence": 0.9,
            },
            "ocr": {
                "documents": 1, "character_accuracy": 1.0, "character_error_rate": 0.0,
                "value_recall": 1.0, "values_found": 2, "values_expected": 2,
                "worst_documents": [],
            },
            "ocr_recognised": {"documents": 0},
            "ocr_from_text_layer": 1,
            "documents": [],
        }
        out = io.StringIO()
        with redirect_stdout(out):
            print_corpus_report(report)
        printed = out.getvalue()
        assert "Character accuracy" not in printed
        assert "1 PDF(s) served from their own text" in printed


class TestTheReportSaysWhatItRan:
    """A report that does not name its configuration describes a system
    nobody else has."""

    def _report(self, **setup) -> str:
        base = {
            "corpus": "real_samples", "master_loaded": {"vendors": 13},
            "rules_loaded": [], "auto_post_threshold": 0.84,
            "disabled_rules": [], "rules_re_enabled": [],
            "ocr_provider": "ollama", "llm_provider": "typhoon",
            "llm_verify_provider": "claude",
        }
        report = {
            "setup": {**base, **setup},
            "summary": {
                "documents": 1, "wall_clock_seconds": 1.0, "type_accuracy": 1.0,
                "field_accuracy": 1.0, "field_accuracy_with_near": 1.0,
                "fields_total": 1, "fields_correct": 1, "fields_near": 0,
                "fields_missing": 0, "auto_posted": 1, "auto_post_rate": 1.0,
                "silent_errors": 0, "silent_error_rate": 0.0,
                "silent_error_documents": [], "auto_post_precision": 1.0,
                "planted_defects": 0, "planted_defects_caught": 0,
                "defect_recall": 1.0, "average_confidence": 0.9,
            },
            "documents": [],
        }
        out = io.StringIO()
        with redirect_stdout(out):
            print_corpus_report(report)
        return out.getvalue()

    def test_it_names_the_providers_behind_the_numbers(self) -> None:
        printed = self._report()
        assert "ollama" in printed and "typhoon" in printed and "claude" in printed

    def test_it_prints_the_threshold(self) -> None:
        assert "84%" in self._report()

    def test_re_enabled_rules_are_called_out_not_buried(self) -> None:
        printed = self._report(
            disabled_rules=["vat_rate_expected"],
            rules_re_enabled=["po_amount_within_tolerance"],
        )
        assert "re-enabled for this run" in printed
        assert "po_amount_within_tolerance" in printed
        assert "missed defect where nothing was missed" in printed

    def test_a_clean_run_does_not_mention_re_enabling(self) -> None:
        assert "re-enabled for this run" not in self._report()


class TestTheTwoErrorsAreNeverSummed:
    """One is a wrong payment, the other a wasted minute."""

    def test_they_are_printed_apart_with_their_consequence(self) -> None:
        report = {
            "setup": {},
            "summary": {
                "documents": 4, "wall_clock_seconds": 1.0, "type_accuracy": 1.0,
                "field_accuracy": 1.0, "field_accuracy_with_near": 1.0,
                "fields_total": 4, "fields_correct": 4, "fields_near": 0,
                "fields_missing": 0, "auto_posted": 2, "auto_post_rate": 0.5,
                "silent_errors": 0, "silent_error_rate": 0.0,
                "silent_error_documents": [], "auto_post_precision": 1.0,
                "planted_defects": 2, "planted_defects_caught": 1,
                "defect_recall": 0.5, "average_confidence": 0.9,
            },
            "gatekeeping": {
                "documents": 4, "decision_accuracy": 0.5,
                "should_post": 2, "should_hold": 2,
                "missed_defects": 1, "missed_defect_rate": 0.5,
                "missed_defect_documents": ["8_over_po_False.pdf"],
                "false_alarms": 1, "false_alarm_rate": 0.5,
                "false_alarm_documents": ["19_two_column_True.pdf"],
                "rules_tested": 2, "rules_untested": 24,
                "rules_with_full_recall": 1, "rules_that_missed": [],
            },
            "documents": [],
        }
        out = io.StringIO()
        with redirect_stdout(out):
            print_corpus_report(report)
        printed = out.getvalue()

        assert "a wrong payment each" in printed
        assert "a wasted review each" in printed
        assert "8_over_po_False.pdf" in printed
        assert "19_two_column_True.pdf" in printed
        # Never a combined "2 errors" line.
        assert "untested, not passing" in printed


@pytest.mark.skipif(not CORPUS.exists(), reason="the corpus is not generated here")
class TestTheCorpusItself:
    """The corpus has to be gradeable before a grade against it means anything."""

    def test_every_manifest_entry_is_on_disk(self) -> None:
        missing = [d.file for d in load_manifest(CORPUS) if not (CORPUS / d.file).exists()]
        assert not missing, f"in the manifest but not on disk: {missing}"

    def test_every_document_carries_ground_truth(self) -> None:
        for document in load_manifest(CORPUS):
            assert document.ground_truth, f"{document.file} has nothing to be graded against"

    def test_the_reference_text_is_present_for_scoring_ocr(self) -> None:
        """Taken from the PDF's text layer before the page was photographed.
        Without it there is nothing to compute a character error rate against."""
        for document in load_manifest(CORPUS):
            assert document.reference_text.strip(), f"{document.file} has no reference text"

    def test_a_document_held_without_a_declared_rule_is_reported(self) -> None:
        """A `_False` document naming no rule grades as a missed defect against
        whatever the system does, with nothing to explain it.

        `3_invoice_photo_on_a_desk_False.jpg` is the live case: it quotes
        `PO-2026-0545`, the only closed purchase order in the register, so the
        defect is real - but its `demonstrates` text describes the photograph
        and `expected_findings` is empty, so whichever rule catches it is scored
        as having fired unexpectedly and defect recall never counts it.

        The check is that the harness *says so* rather than that the corpus is
        already perfect. A run that graded silently against this would report a
        rule as over-firing when the rule is right.
        """
        from docuflow.evaluation import integrity

        report = integrity.check(load_manifest(CORPUS))
        assert "held_without_a_declared_rule" in report

    def test_the_corpus_measures_both_kinds_of_mistake(self) -> None:
        """Only defect cases measures missed defects and cannot see false
        alarms; only clean cases is the reverse."""
        documents = load_manifest(CORPUS)
        assert sum(1 for d in documents if d.posts) >= 5
        assert sum(1 for d in documents if not d.posts) >= 5

    def test_the_master_data_the_corpus_needs_is_beside_it(self) -> None:
        for kind in ("vendors", "purchase_orders", "contracts"):
            assert (CORPUS / "master" / f"{kind}.csv").exists(), kind

    def test_the_rules_written_for_this_corpus_are_valid(self) -> None:
        """A rule set that fails to build is one the run loads none of, and the
        documents it exists to catch then post."""
        from docuflow.rules.custom import build_rule

        path = CORPUS / "rules.json"
        if not path.exists():
            pytest.skip("this corpus ships no rules")
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("rules") if isinstance(payload, dict) else payload
        assert entries, "rules.json is present but empty"
        for entry in entries:
            build_rule(entry)
