"""How much of a person's day this actually gives back.

Accuracy and coverage are measurable properties of the system. Time saved is
not: it is a claim about what people were doing before, and the honest version
of it has the assumption printed next to the answer rather than buried in a
spreadsheet.

So this module does two things and refuses a third. It states, in one place and
in the open, how long each part of the manual process takes. It computes what
follows from those numbers and what the run actually did. It does not pretend
the result is measured - `assumptions` travels with every figure, and the report
prints it.

The model of the manual process:

  * every document is **keyed** by hand, whatever else happens to it;
  * every document is **cross-checked** against the vendor file, the purchase
    order and prior postings, by hand;
  * a document the system holds still costs a **review**, but a shorter one than
    doing it from scratch, because the values, the evidence and the failing
    checks are already in front of the reviewer;
  * a document the system posts costs nothing.

The third bullet is where an inflated number would come from, and it is the one
worth arguing about. A held document is not free, and counting it as free is how
automation projects report savings nobody can find afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

# Minutes, per document, for the manual process. Deliberately conservative:
# these are the numbers a sceptical finance manager would accept, not the ones
# that make the answer large.
KEYING_MINUTES = 4.0
CROSS_CHECK_MINUTES = 3.5
# What a review costs when the extraction, the evidence and the failing checks
# are already on screen. Not zero, and not the full manual cost either.
ASSISTED_REVIEW_MINUTES = 1.5
# A document that posted wrong costs more than the one it saved: somebody finds
# it at reconciliation, traces it, raises a correction. Counted against the
# saving so a system that posts everything cannot look efficient.
SILENT_ERROR_MINUTES = 45.0

WORKING_MINUTES_PER_DAY = 7.5 * 60


@dataclass(frozen=True)
class Effort:
    """What the corpus would have cost by hand, and what it cost here."""

    documents: int
    manual_minutes: float
    assisted_minutes: float
    machine_minutes: float
    rework_minutes: float

    @property
    def saved_minutes(self) -> float:
        return self.manual_minutes - self.assisted_minutes

    @property
    def saved_share(self) -> float:
        return self.saved_minutes / self.manual_minutes if self.manual_minutes else 0.0

    @property
    def minutes_per_document(self) -> float:
        return self.assisted_minutes / self.documents if self.documents else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "manual_minutes": round(self.manual_minutes, 1),
            "assisted_minutes": round(self.assisted_minutes, 1),
            "machine_minutes": round(self.machine_minutes, 1),
            "rework_minutes": round(self.rework_minutes, 1),
            "saved_minutes": round(self.saved_minutes, 1),
            "saved_share": round(self.saved_share, 3),
            "minutes_per_document": round(self.minutes_per_document, 2),
            "days_per_thousand_documents": round(
                (self.assisted_minutes / self.documents * 1000) / WORKING_MINUTES_PER_DAY, 1
            ) if self.documents else 0.0,
            "manual_days_per_thousand_documents": round(
                (self.manual_minutes / self.documents * 1000) / WORKING_MINUTES_PER_DAY, 1
            ) if self.documents else 0.0,
            "assumptions": assumptions(),
        }


def assumptions() -> dict[str, float]:
    """Every number this rests on, so the figure can be argued with."""
    return {
        "keying_minutes_per_document": KEYING_MINUTES,
        "cross_check_minutes_per_document": CROSS_CHECK_MINUTES,
        "assisted_review_minutes_per_held_document": ASSISTED_REVIEW_MINUTES,
        "minutes_to_clean_up_one_silent_error": SILENT_ERROR_MINUTES,
    }


def effort_saved(
    documents: int, auto_posted: int, silent_errors: int, machine_seconds: float
) -> Effort:
    """What the corpus cost by hand against what it cost through here.

    `machine_seconds` is wall-clock and is reported separately rather than
    subtracted: a document being read by a model is not a person waiting for it.
    In a batch it is not on anyone's critical path at all.
    """
    documents = max(0, documents)
    held = max(0, documents - auto_posted)

    manual = documents * (KEYING_MINUTES + CROSS_CHECK_MINUTES)
    # Held documents cost a shortened review; posted ones cost nothing; and
    # anything posted wrong costs the clean-up, charged here rather than
    # quietly left out.
    rework = silent_errors * SILENT_ERROR_MINUTES
    assisted = held * ASSISTED_REVIEW_MINUTES + rework

    return Effort(
        documents=documents,
        manual_minutes=manual,
        assisted_minutes=assisted,
        machine_minutes=machine_seconds / 60.0,
        rework_minutes=rework,
    )
