"""Precision and recall, and the failures plain accuracy cannot see.

Both of these were found in this corpus rather than imagined.

**A classifier with one answer.** Twenty-four of thirty documents are invoices,
so answering "invoice" every time scores 80% and has never recognised a receipt.
Every Thai receipt here was read as an invoice once, because a Thai till prints
`ใบกำกับภาษีอย่างย่อ` - an *abbreviated* tax invoice, a receipt - and that
contains the invoice keyword whole. Accuracy said 80%; recall for `receipt` said
zero, which is the number that describes what happened.

**A value invented for a field the document does not have.** Accuracy computed
over the fields ground truth names never looks at it, because there is nothing
to compare it against. It is the more dangerous kind of wrong: a confident value
for a field nobody filled in reads downstream exactly like a correct one.
"""

from __future__ import annotations

import pytest

from docuflow.evaluation.classification import (
    ClassScore,
    confusion,
    score_labels,
    summarize_fields,
    summarize_labels,
)


class TestOneMistakeIsTwoFacts:
    def test_a_wrong_answer_is_a_miss_and_a_claim(self) -> None:
        """Reading a receipt as an invoice costs `receipt` a recall point and
        `invoice` a precision point. Counting it once hides half of it."""
        scores = {score.label: score for score in score_labels([("receipt", "invoice")])}
        assert scores["receipt"].false_negative == 1
        assert scores["invoice"].false_positive == 1

    def test_a_right_answer_is_one_fact(self) -> None:
        scores = {score.label: score for score in score_labels([("invoice", "invoice")])}
        assert scores["invoice"].true_positive == 1
        assert scores["invoice"].false_positive == 0


class TestTheClassifierThatOnlyKnowsOneAnswer:
    """The case the whole module exists for."""

    PAIRS = [("invoice", "invoice")] * 8 + [("receipt", "invoice")] * 2

    def test_accuracy_alone_calls_it_eighty_per_cent(self) -> None:
        assert summarize_labels(self.PAIRS)["accuracy"] == 0.8

    def test_recall_for_the_ignored_class_is_zero(self) -> None:
        result = summarize_labels(self.PAIRS)
        receipt = next(c for c in result["classes"] if c["label"] == "receipt")
        assert receipt["recall"] == 0.0
        assert receipt["support"] == 2
        assert receipt["predicted"] == 0

    def test_macro_averages_refuse_to_hide_it(self) -> None:
        """Macro is the one that notices. Micro would report 80% three times.

        The first version of this module returned `None` for the F1 of a class
        that was never predicted, which dropped it out of the macro average and
        scored this classifier 0.89 - better than its accuracy, for never having
        recognised a receipt.
        """
        result = summarize_labels(self.PAIRS)
        assert result["macro_recall"] == 0.5
        assert result["macro_f1"] is not None and result["macro_f1"] < 0.8

    def test_the_ignored_class_scores_zero_rather_than_nothing(self) -> None:
        result = summarize_labels(self.PAIRS)
        receipt = next(c for c in result["classes"] if c["label"] == "receipt")
        assert receipt["f1"] == 0.0, "a class that exists and is never found scores 0"

    def test_the_confusion_matrix_says_what_became_what(self) -> None:
        assert confusion(self.PAIRS)["receipt"] == {"invoice": 2}


class TestAClassThatWasNeverPredicted:
    def test_precision_is_none_not_one(self) -> None:
        """Calling it perfect lets an ignored class raise the average, which is
        precisely the failure being measured."""
        assert ClassScore("receipt", 0, 0, 3).precision is None

    def test_a_class_that_does_not_occur_has_no_recall(self) -> None:
        assert ClassScore("contract", 0, 2, 0).recall is None

    def test_macro_skips_classes_the_corpus_never_contains(self) -> None:
        """A type nothing in the corpus is has nothing to be right about, and
        averaging it in would make the score depend on the label vocabulary."""
        result = summarize_labels([("invoice", "invoice"), ("invoice", "receipt")])
        receipt = next(c for c in result["classes"] if c["label"] == "receipt")
        assert receipt["support"] == 0
        assert result["macro_recall"] == 0.5


class TestExtractionPrecisionAndRecall:
    def _documents(self, *statuses: str) -> list[dict]:
        return [{
            "comparisons": [
                {"field": f"field_{index}", "status": status}
                for index, status in enumerate(statuses)
            ]
        }]

    def test_a_perfect_run(self) -> None:
        result = summarize_fields(self._documents("correct", "correct"))
        assert result["precision"] == 1.0 and result["recall"] == 1.0

    def test_a_missing_value_costs_recall_but_not_precision(self) -> None:
        """Leaving work for a person is the safe way to be wrong: nothing false
        was produced, so nothing downstream can act on it."""
        result = summarize_fields(self._documents("correct", "missing"))
        assert result["precision"] == 1.0
        assert result["recall"] == 0.5

    def test_an_invented_value_costs_precision_but_not_recall(self) -> None:
        """The case accuracy over ground-truth fields cannot see at all."""
        result = summarize_fields(self._documents("correct", "spurious"))
        assert result["precision"] == 0.5
        assert result["recall"] == 1.0
        assert result["spurious"] == 1

    def test_a_wrong_value_costs_both(self) -> None:
        """An answer was produced that should not have been, and the answer that
        should have been produced was not."""
        result = summarize_fields(self._documents("correct", "wrong"))
        assert result["precision"] == 0.5
        assert result["recall"] == 0.5

    def test_a_near_miss_gets_no_partial_credit(self) -> None:
        """`RSW-2026-1188` read as `RSW-2026-1I88` is a different invoice number
        to everything downstream, however close it looks to a person."""
        result = summarize_fields(self._documents("near"))
        assert result["correct"] == 0
        assert result["wrong"] == 1
        assert result["precision"] == 0.0

    def test_the_weakest_fields_come_first(self) -> None:
        documents = [{
            "comparisons": [
                {"field": "total_amount", "status": "correct"},
                {"field": "bank_account", "status": "wrong"},
            ]
        }]
        result = summarize_fields(documents)
        assert result["per_field"][0]["field"] == "bank_account"

    def test_nothing_to_score(self) -> None:
        assert summarize_fields([])["values"] == 0


def test_an_empty_corpus_scores_nothing_rather_than_everything() -> None:
    assert summarize_labels([]) == {"documents": 0}
