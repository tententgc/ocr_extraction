"""Does the gate open for the right documents, and does each rule earn its place.

Two questions the accuracy figures cannot answer.

**Did the document go the right way?** Every file in the corpus carries its
expected outcome in its name - `_True` should post, `_False` should be held - so
this is a straight comparison of two decisions rather than the system being
graded against itself.

The two ways of being wrong are not equivalent and are never summed here:

  * a **missed defect** is a wrong payment. It is the metric this system exists
    to drive to zero and one of them outweighs a great many of the other kind;
  * a **false alarm** is a valid document held for a person who finds nothing to
    correct. Cheap once. Fifty of them and nobody reads the review queue, at
    which point the missed defects stop being caught either - so it is not a
    number to shrug at, it is the one that decides whether the first number
    keeps meaning anything.

**Did each rule fire for the reason it exists?** A corpus that only counts
decisions cannot tell a rule that works from one that is silent while a
different rule happens to catch the same document. Each document declares which
rules its defect should trip, so every rule gets a confusion matrix: fired when
expected, fired when not, silent when expected, silent when not.

A rule with no expectations anywhere in the corpus is reported as untested
rather than as passing. Untested is the honest word for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# Firing on a document that did not declare it is not automatically wrong - a
# document can have a real problem the author did not think to plant. These are
# the rules whose failure is a statement about the *reading* rather than about
# the document, so an unexpected one is worth surfacing separately rather than
# counting as a false alarm against the rule.
READING_RULES = frozenset({
    "invoice_totals_consistent", "tax_rate_consistent", "line_items_sum_to_subtotal",
    "tax_id_checksum", "dates_not_in_future", "due_date_after_issue",
})


@dataclass(frozen=True)
class Outcome:
    """What one document was expected to do, and what it did."""

    document: str
    should_post: bool
    posted: bool
    confidence: float
    expected_findings: tuple[str, ...] = ()
    failed_findings: tuple[str, ...] = ()

    @property
    def correct(self) -> bool:
        return self.should_post == self.posted

    @property
    def missed_defect(self) -> bool:
        """Posted something that should have been held."""
        return self.posted and not self.should_post

    @property
    def false_alarm(self) -> bool:
        """Held something that should have posted."""
        return self.should_post and not self.posted

    def to_json(self) -> dict[str, Any]:
        return {
            "document": self.document,
            "should_post": self.should_post,
            "posted": self.posted,
            "correct": self.correct,
            "missed_defect": self.missed_defect,
            "false_alarm": self.false_alarm,
            "confidence": round(self.confidence, 3),
            "expected_findings": list(self.expected_findings),
            "failed_findings": list(self.failed_findings),
        }


@dataclass
class RuleScore:
    """One rule's record across the corpus."""

    rule_id: str
    fired_when_expected: int = 0        # true positive
    silent_when_expected: int = 0       # false negative - the dangerous one
    fired_unexpectedly: int = 0         # false positive, or an unplanted real problem
    silent_when_not_expected: int = 0   # true negative
    missed_on: list[str] = field(default_factory=list)
    fired_unexpectedly_on: list[str] = field(default_factory=list)

    @property
    def expected_total(self) -> int:
        return self.fired_when_expected + self.silent_when_expected

    @property
    def tested(self) -> bool:
        """Whether any document in the corpus claims this rule should fire."""
        return self.expected_total > 0

    @property
    def recall(self) -> float | None:
        """Of the documents that should trip it, how many did."""
        if not self.tested:
            return None
        return self.fired_when_expected / self.expected_total

    @property
    def precision(self) -> float | None:
        fired = self.fired_when_expected + self.fired_unexpectedly
        if not fired:
            return None
        return self.fired_when_expected / fired

    def to_json(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "tested": self.tested,
            "recall": None if self.recall is None else round(self.recall, 3),
            "precision": None if self.precision is None else round(self.precision, 3),
            "fired_when_expected": self.fired_when_expected,
            "silent_when_expected": self.silent_when_expected,
            "fired_unexpectedly": self.fired_unexpectedly,
            "silent_when_not_expected": self.silent_when_not_expected,
            "missed_on": self.missed_on,
            "fired_unexpectedly_on": self.fired_unexpectedly_on,
            "reading_rule": self.rule_id in READING_RULES,
        }


def score_rules(outcomes: Iterable[Outcome], known_rules: Iterable[str]) -> list[RuleScore]:
    """A confusion matrix per rule, over the whole corpus."""
    outcomes = list(outcomes)
    scores = {rule_id: RuleScore(rule_id=rule_id) for rule_id in known_rules}

    for outcome in outcomes:
        expected = set(outcome.expected_findings)
        fired = set(outcome.failed_findings)
        # A rule the corpus expects but the registry has never heard of is still
        # worth scoring - it is usually a rule that was renamed.
        for rule_id in expected | fired:
            scores.setdefault(rule_id, RuleScore(rule_id=rule_id))

        for rule_id, score in scores.items():
            if rule_id in expected and rule_id in fired:
                score.fired_when_expected += 1
            elif rule_id in expected:
                score.silent_when_expected += 1
                score.missed_on.append(outcome.document)
            elif rule_id in fired:
                score.fired_unexpectedly += 1
                score.fired_unexpectedly_on.append(outcome.document)
            else:
                score.silent_when_not_expected += 1

    return sorted(scores.values(), key=lambda score: (not score.tested, score.rule_id))


def summarize(outcomes: Iterable[Outcome], rules: Iterable[RuleScore]) -> dict[str, Any]:
    """The gate's record, and the rules behind it."""
    outcomes = list(outcomes)
    rules = list(rules)
    if not outcomes:
        return {"documents": 0}

    should_post = [item for item in outcomes if item.should_post]
    should_hold = [item for item in outcomes if not item.should_post]
    missed = [item for item in outcomes if item.missed_defect]
    alarms = [item for item in outcomes if item.false_alarm]

    tested = [rule for rule in rules if rule.tested]
    perfect = [rule for rule in tested if rule.recall == 1.0]
    leaky = [rule for rule in tested if rule.recall is not None and rule.recall < 1.0]

    return {
        "documents": len(outcomes),
        "decision_accuracy": round(
            sum(1 for item in outcomes if item.correct) / len(outcomes), 3
        ),
        # Kept apart on purpose. Summing them would let a system trade a wrong
        # payment for a saved review and call it an improvement.
        "should_post": len(should_post),
        "should_hold": len(should_hold),
        "missed_defects": len(missed),
        "missed_defect_rate": round(len(missed) / len(should_hold), 3) if should_hold else 0.0,
        "missed_defect_documents": [item.document for item in missed],
        "false_alarms": len(alarms),
        "false_alarm_rate": round(len(alarms) / len(should_post), 3) if should_post else 0.0,
        "false_alarm_documents": [item.document for item in alarms],
        "rules_tested": len(tested),
        "rules_untested": len(rules) - len(tested),
        "rules_with_full_recall": len(perfect),
        "rules_that_missed": [rule.to_json() for rule in leaky],
        "per_rule": [rule.to_json() for rule in rules],
        "per_document": [item.to_json() for item in outcomes],
    }
