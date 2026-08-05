"""Precision and recall, for the two things this system predicts.

Plain accuracy is the wrong summary for both of them and hides opposite
failures.

**Document type.** Twenty-four of thirty documents are invoices, so a classifier
that answers "invoice" every single time scores 80% accuracy and has never once
recognised a receipt. Per-class precision and recall make that impossible to
report as success: recall for `receipt` would be zero. It is not hypothetical -
every Thai receipt in this corpus was read as an invoice once, because the
keyword for an abbreviated tax invoice contains the keyword for a tax invoice.

**Extracted fields.** Field accuracy over the fields ground truth names cannot
see a value the system *invented*: a field the document does not have, filled in
anyway, is never compared against anything. That is a false positive and it is
the more dangerous kind, because a confidently wrong value reads exactly like a
correct one downstream.

So both get the same treatment:

  * **precision** - of what was answered, how much was right. Low precision
    means the system makes things up.
  * **recall** - of what was there, how much was found. Low recall means the
    system leaves work for a person.
  * **F1** - the two combined, for a single number that neither can game.

Macro and micro are both reported. Macro averages the classes equally and is the
one that notices a rare class being ignored; micro pools every decision and is
dominated by whichever class is largest. Quoting only micro is how "80% accurate"
gets said about a classifier that only knows one answer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ClassScore:
    """One class's record: what it caught, missed and invented."""

    label: str
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def support(self) -> int:
        """How many documents genuinely are this class."""
        return self.true_positive + self.false_negative

    @property
    def predicted(self) -> int:
        return self.true_positive + self.false_positive

    @property
    def precision(self) -> float | None:
        """None, not 1.0, when the class was never predicted.

        A class nothing was ever assigned to has no precision to report, and
        calling that a perfect score lets an ignored class raise the average."""
        if not self.predicted:
            return None
        return self.true_positive / self.predicted

    @property
    def recall(self) -> float | None:
        if not self.support:
            return None
        return self.true_positive / self.support

    @property
    def f1(self) -> float | None:
        """Zero, not None, for a class that exists and was never found.

        `precision` is None when nothing was ever predicted as this class, and
        letting that propagate into F1 excluded the class from the macro
        average - so a classifier that ignored `receipt` entirely scored 0.89
        macro F1, which is the exact failure this module was written to make
        impossible. Recall is 0 there, and F1 is 0 whenever recall is 0.
        """
        recall = self.recall
        if recall is None:
            return None
        if recall == 0:
            return 0.0
        precision = self.precision
        if precision is None or precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "support": self.support,
            "predicted": self.predicted,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": None if self.precision is None else round(self.precision, 3),
            "recall": None if self.recall is None else round(self.recall, 3),
            "f1": None if self.f1 is None else round(self.f1, 3),
        }


def score_labels(pairs: Iterable[tuple[str, str]]) -> list[ClassScore]:
    """Per-class scores from `(expected, predicted)` pairs."""
    pairs = list(pairs)
    labels = sorted({label for pair in pairs for label in pair})
    counts: dict[str, Counter[str]] = {
        label: Counter({"tp": 0, "fp": 0, "fn": 0}) for label in labels
    }

    for expected, predicted in pairs:
        if expected == predicted:
            counts[expected]["tp"] += 1
            continue
        # One mistake is two facts: the right class was missed, and the wrong
        # class was claimed.
        counts[expected]["fn"] += 1
        counts[predicted]["fp"] += 1

    return [
        ClassScore(
            label=label,
            true_positive=counts[label]["tp"],
            false_positive=counts[label]["fp"],
            false_negative=counts[label]["fn"],
        )
        for label in labels
    ]


def confusion(pairs: Iterable[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """`expected -> predicted -> count`, for reading what turned into what."""
    matrix: dict[str, dict[str, int]] = {}
    for expected, predicted in pairs:
        matrix.setdefault(expected, {})
        matrix[expected][predicted] = matrix[expected].get(predicted, 0) + 1
    return matrix


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_labels(pairs: Iterable[tuple[str, str]]) -> dict[str, Any]:
    """Accuracy, per-class scores, and both averages."""
    pairs = list(pairs)
    if not pairs:
        return {"documents": 0}

    scores = score_labels(pairs)
    correct = sum(1 for expected, predicted in pairs if expected == predicted)

    # Macro averages only over classes that exist in the corpus. A class the
    # corpus never contains has nothing to be right or wrong about.
    present = [score for score in scores if score.support]
    macro_precision = _mean([s.precision for s in present if s.precision is not None])
    macro_recall = _mean([s.recall for s in present if s.recall is not None])
    macro_f1 = _mean([s.f1 for s in present if s.f1 is not None])

    return {
        "documents": len(pairs),
        # Micro-averaged precision, recall and F1 all equal accuracy in a
        # single-label problem, so it is reported once under its honest name.
        "accuracy": round(correct / len(pairs), 3),
        "macro_precision": None if macro_precision is None else round(macro_precision, 3),
        "macro_recall": None if macro_recall is None else round(macro_recall, 3),
        "macro_f1": None if macro_f1 is None else round(macro_f1, 3),
        "classes": [score.to_json() for score in scores],
        "confusion": confusion(pairs),
    }


def summarize_fields(documents: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Extraction as precision and recall rather than a single accuracy.

    Each field comparison lands in one of four buckets:

      * `correct` - answered, and right. A true positive.
      * `wrong` / `near` - answered, and not right. Counted as a false positive
        *and* a false negative: a value was produced that should not have been,
        and the value that should have been produced was not. A near-miss is not
        given partial credit here - `RSW-2026-1188` read as `RSW-2026-1I88` is a
        different invoice number to anything downstream.
      * `missing` - there and not found. A false negative only; the system left
        work for a person rather than inventing an answer, which is the safe way
        to be wrong.
      * `spurious` - not there and answered anyway. A false positive only, and
        the one plain accuracy cannot see at all.
    """
    documents = list(documents)
    counts = Counter()
    per_field: dict[str, Counter[str]] = {}

    for document in documents:
        for comparison in document.get("comparisons", ()):
            status = comparison["status"]
            counts[status] += 1
            per_field.setdefault(comparison["field"], Counter())[status] += 1

    correct = counts["correct"]
    wrong = counts["wrong"] + counts["near"]
    missing = counts["missing"]
    spurious = counts["spurious"]

    answered = correct + wrong + spurious
    present = correct + wrong + missing

    def _scores(tp: int, fp: int, fn: int) -> dict[str, float | None]:
        """None only where nothing was measured; 0.0 where everything failed.

        A field with one wrong value has precision 0 and recall 0, and returning
        None for its F1 sorted it to the *end* of "weakest fields" - the worst
        field in the corpus printed last, or not at all.
        """
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        if precision is None or recall is None:
            f1 = None
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return {
            "precision": None if precision is None else round(precision, 3),
            "recall": None if recall is None else round(recall, 3),
            "f1": None if f1 is None else round(f1, 3),
        }

    fields = []
    for name, tally in sorted(per_field.items()):
        field_wrong = tally["wrong"] + tally["near"]
        fields.append({
            "field": name,
            "total": sum(tally.values()),
            "correct": tally["correct"],
            "wrong": field_wrong,
            "missing": tally["missing"],
            "spurious": tally["spurious"],
            **_scores(
                tally["correct"],
                field_wrong + tally["spurious"],
                field_wrong + tally["missing"],
            ),
        })

    return {
        "values": correct + wrong + missing + spurious,
        "answered": answered,
        "present": present,
        "correct": correct,
        "wrong": wrong,
        "missing": missing,
        # Answered a field the document does not have. Invisible to accuracy
        # computed over ground-truth fields alone.
        "spurious": spurious,
        **_scores(correct, wrong + spurious, wrong + missing),
        "per_field": sorted(fields, key=lambda item: (item["f1"] is None, item["f1"] or 0)),
    }
