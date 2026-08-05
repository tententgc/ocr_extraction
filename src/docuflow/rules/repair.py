"""Which value is wrong, and what it should say instead.

A rule that fails knows something does not add up. That is not the same as
knowing what to do about it, and the gap between the two is where a reviewer's
time goes. "Totals do not add up: 68,000 + 4,760 = 72,760, but the document
states 27,760" tells someone there is a problem and leaves them to work out
which of the three numbers to trust, re-read the page, and retype it.

This module closes that gap in two steps.

**Which field.** The money on an invoice is tied together by more than one
relation - the lines sum to the subtotal, the tax is a rate applied to the
subtotal, the subtotal and tax make the total. A single misread value breaks
every relation it appears in and leaves the others intact, so the relations that
still hold identify the value that broke them. Where exactly one field explains
every failure, this says so; where two or more would, it says that instead,
because naming one at random is worse than admitting the page is ambiguous.

**Which digit.** Once the arithmetic says what the value should have been, the
difference between that and what was read is usually not random. 27,760 against
72,760 is a transposition. 1,070 against 7,070 is the single confusion that OCR
makes most often. 45.00 against 4,500 is a decimal point. Naming the slip turns
a correction a reviewer has to verify from scratch into one they can confirm by
glancing at the page, which is the difference between a fix that gets accepted
and one that gets ignored.

Nothing here decides anything. It produces suggestions, a person accepts them,
and the full rule set runs again on what they accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from docuflow.models import SuggestedFix

# Two amounts this close are the same amount, rounded differently. Deliberately
# tight: this is asking "is this the same number", not "is this within the
# tolerance the business allows". The relative term exists only to absorb
# per-line rounding on a long invoice - at 72,760 it is about seven baht, which
# leaves a ninety-baht misread visible. An earlier version used half a percent
# and swallowed exactly that error, reporting the document as consistent.
MONEY_SLACK = 0.02
MONEY_RELATIVE_SLACK = 0.0001

# Digit pairs a scan genuinely confuses, chosen from the shapes: each pair
# differs by a stroke that print can lose or a scan can invent. Kept short on
# purpose - every pair added here is another coincidence the tax-ID repair below
# can mistake for evidence.
CONFUSABLE_DIGITS: frozenset[frozenset[str]] = frozenset(
    frozenset(pair) for pair in ("17", "08", "38", "39", "56", "69", "89", "68", "35", "27")
)


@dataclass(frozen=True)
class Blame:
    """The single field that explains a broken set of relations."""

    field: str
    current: float | None
    suggested: float
    # The relations that agreed this field should hold `suggested`.
    because: tuple[str, ...] = ()
    # Relations that do not mention this field at all, and hold. These are the
    # real evidence: they are what rules out the other fields being the misread
    # one, and without any of them the accusation rests on nothing but the
    # absence of an alternative.
    corroborated_by: tuple[str, ...] = ()

    @property
    def certain(self) -> bool:
        return bool(self.corroborated_by)


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= max(MONEY_SLACK, abs(right) * MONEY_RELATIVE_SLACK)


def _digits(amount: float) -> str:
    """An amount as the digits a person would read off the page."""
    return f"{abs(amount):.2f}".replace(".", "")


def explain_difference(stated: float, expected: float) -> str | None:
    """Name the slip that turns `expected` into `stated`, if there is one.

    Returns None when the two numbers are simply different, which is the honest
    answer for a value that was not misread but was wrong on the document.
    """
    if _close(stated, expected):
        return None

    read, want = _digits(stated), _digits(expected)

    # A decimal point in the wrong place. Checked before digit comparison,
    # because 4500.00 and 45.00 share no digit positions at all.
    for factor, wording in ((10, "one place"), (100, "two places")):
        if _close(stated, expected * factor):
            return f"the decimal point is {wording} to the right of where it balances"
        if _close(stated * factor, expected):
            return f"the decimal point is {wording} to the left of where it balances"

    if len(read) == len(want):
        differing = [index for index, (a, b) in enumerate(zip(read, want)) if a != b]

        if len(differing) == 1:
            position = differing[0]
            got, should = read[position], want[position]
            confusable = frozenset({got, should}) in CONFUSABLE_DIGITS
            return (
                f"one digit differs: '{got}' was read where '{should}' balances"
                + (f", and {got}/{should} is a pair OCR routinely confuses" if confusable else "")
            )

        if len(differing) == 2:
            first, second = differing
            if read[first] == want[second] and read[second] == want[first]:
                swapped = "adjacent " if second == first + 1 else ""
                return (
                    f"two {swapped}digits are transposed: '{read[first]}{read[second]}' "
                    f"was read where '{want[first]}{want[second]}' balances"
                )

    if len(read) == len(want) + 1 and want in (read[1:], read[:-1]):
        return "there is one digit too many"
    if len(want) == len(read) + 1 and read in (want[1:], want[:-1]):
        return "a digit is missing"

    return None


# --------------------------------------------------------------------------- #
# The money on an invoice                                                       #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Money:
    """The amounts that have to agree with each other, as they were read."""

    subtotal: float | None = None
    tax_amount: float | None = None
    tax_rate: float | None = None
    total_amount: float | None = None
    line_total: float | None = None

    def get(self, name: str) -> float | None:
        return getattr(self, name)


FIELDS = ("subtotal", "tax_amount", "tax_rate", "total_amount")


@dataclass(frozen=True)
class Relation:
    """One thing that has to be true of a document's amounts.

    `terms` is every field the relation mentions. A relation is *active* only
    when the document supplies all of them: a missing tax rate leaves the tax
    amount unexplained rather than assumed to be zero, which matters because
    most receipts have no tax line at all and none of them are broken.
    """

    name: str
    terms: tuple[str, ...]
    # Given the document, what this relation says `field` should hold.
    solve: Any
    # Given the document, whether it currently holds.
    holds: Any
    # How the answer reads in a sentence, per field.
    wording: dict[str, str]

    def active(self, money: Money) -> bool:
        return all(money.get(term) is not None for term in self.terms)


def _relations() -> tuple[Relation, ...]:
    """The arithmetic every money document has to satisfy."""
    return (
        Relation(
            name="sum",
            terms=("subtotal", "tax_amount", "total_amount"),
            solve=lambda money, field: {
                "subtotal": money.total_amount - money.tax_amount,
                "tax_amount": money.total_amount - money.subtotal,
                "total_amount": money.subtotal + money.tax_amount,
            }[field],
            holds=lambda money: _close(money.subtotal + money.tax_amount, money.total_amount),
            wording={
                "subtotal": "the total less the tax",
                "tax_amount": "the total less the subtotal",
                "total_amount": "the subtotal plus the tax",
            },
        ),
        Relation(
            name="rate",
            terms=("subtotal", "tax_amount", "tax_rate"),
            solve=lambda money, field: {
                "subtotal": money.tax_amount * 100.0 / money.tax_rate if money.tax_rate else None,
                "tax_amount": money.subtotal * money.tax_rate / 100.0,
                "tax_rate": money.tax_amount * 100.0 / money.subtotal if money.subtotal else None,
            }[field],
            holds=lambda money: _close(money.subtotal * money.tax_rate / 100.0, money.tax_amount),
            wording={
                "subtotal": "the tax divided by the rate",
                "tax_amount": "the rate applied to the subtotal",
                "tax_rate": "the tax as a share of the subtotal",
            },
        ),
        Relation(
            name="lines",
            terms=("subtotal", "line_total"),
            solve=lambda money, field: money.line_total,
            holds=lambda money: _close(money.line_total, money.subtotal),
            wording={"subtotal": "the line items added up"},
        ),
    )


def _plausible(field: str, value: float) -> bool:
    """Whether a proposed replacement could be a real value at all.

    Solving a relation backwards happily produces a tax of minus forty thousand.
    Ruling those out is not cosmetic - it is what lets an invoice with only one
    active relation still name its culprit, because two of the three candidates
    are arithmetic artefacts rather than readings anyone could have made.
    """
    if value < 0:
        return False
    if field == "tax_rate":
        return value <= 100.0
    return True


def blame(money: Money) -> Blame | None:
    """The one field whose correction would make everything agree.

    Returns None when nothing is broken, and None when more than one field would
    explain it equally well. The second case is the important one: an invoice
    where either the subtotal or the total could be the misread value is a page
    a person has to look at, and inventing a preference between them would put a
    wrong number in front of them wearing a suggestion's confidence.
    """
    active = [relation for relation in _relations() if relation.active(money)]
    broken = [relation for relation in active if not relation.holds(money)]
    if not broken:
        return None

    culprits: list[Blame] = []
    for field in FIELDS:
        if money.get(field) is None:
            # A value the document never stated cannot be the misread one. It
            # can be missing, which is a different finding with a different fix.
            continue

        mentions = [relation for relation in active if field in relation.terms]
        others = [relation for relation in active if field not in relation.terms]
        if not mentions:
            continue

        answers = [(relation.solve(money, field), relation) for relation in mentions]
        if any(value is None for value, _ in answers):
            continue

        # Every relation that mentions this field has to name the same
        # replacement; where they disagree, this field is not the single cause.
        target = answers[0][0]
        if not all(_close(value, target) for value, _ in answers):
            continue
        if _close(money.get(field), target) or not _plausible(field, target):
            continue

        # And every relation that does *not* mention it has to already hold,
        # or something else is wrong too and one change will not settle it.
        if not all(relation.holds(money) for relation in others):
            continue

        culprits.append(
            Blame(
                field=field,
                current=money.get(field),
                suggested=round(target, 2),
                because=tuple(relation.wording[field] for _, relation in answers),
                corroborated_by=tuple(relation.name for relation in others),
            )
        )

    return culprits[0] if len(culprits) == 1 else None


def money_fixes(money: Money, labels: dict[str, str] | None = None) -> tuple[SuggestedFix, ...]:
    """A reviewer-ready correction for a document whose amounts do not add up."""
    found = blame(money)
    if found is None:
        return ()

    names = labels or {}
    label = names.get(found.field, found.field.replace("_", " "))
    evidence = " and ".join(found.because)
    corroboration = {
        "lines": "the line items add up to the subtotal",
        "rate": "the tax matches the stated rate",
        "sum": "the subtotal and tax make the total",
    }
    standing = [corroboration[name] for name in found.corroborated_by if name in corroboration]

    reason = f"{evidence} gives {found.suggested:,.2f}"
    if standing:
        # The corroboration is what makes this an accusation rather than a
        # preference: it is the reason the *other* values are not the culprit.
        reason = (
            f"{label} is the only value that explains it - "
            f"{' and '.join(standing)}, so those are right. {evidence} "
            f"gives {found.suggested:,.2f}"
        )

    slip = explain_difference(found.current, found.suggested) if found.current is not None else None
    if slip:
        reason += f". Looking at the page: {slip}"

    headline = (
        f"{label.capitalize()} is the only value that explains it; "
        f"{found.suggested:,.2f} would balance."
        if standing
        else f"{label.capitalize()} may be the misread value; {found.suggested:,.2f} would balance."
    )

    return (
        SuggestedFix(
            field=found.field,
            current=found.current,
            suggested=found.suggested,
            reason=reason,
            headline=headline,
            source="arithmetic",
            # A relation standing untouched is a different kind of evidence from
            # the mere absence of an alternative, and the confidence has to say
            # so or the reviewer cannot tell a derivation from a guess.
            confidence=0.92 if found.certain else 0.7,
        ),
    )


# --------------------------------------------------------------------------- #
# Identifiers                                                                   #
# --------------------------------------------------------------------------- #

def tax_id_fixes(field: str, digits: str) -> tuple[SuggestedFix, ...]:
    """A digit that would make a failing Thai tax ID pass its own check.

    Worth being precise about what this can and cannot do. The check digit
    detects a single-character error reliably; it does not *identify* one. A
    thirteen-digit number has a valid single-digit repair at most positions, and
    the repair this finds is often a different valid tax ID rather than the one
    actually printed on the page.

    That still earns its place, because a reviewer re-reading a thirteen-digit
    number needs somewhere to start looking and "digit nine is suspect" is a
    better start than none. It does not earn the authority of the arithmetic
    suggestions, so it is worded as one possible reading, carries a low
    confidence, and is offered only when a single stroke-plausible repair exists.
    """
    from docuflow.normalize.identifiers import thai_tax_id_checksum_ok

    if len(digits) != 13 or not digits.isdigit() or thai_tax_id_checksum_ok(digits):
        return ()

    # Any thirteen-digit number has a single-digit repair at most positions -
    # the check digit alone can always be rewritten to match whatever body it
    # follows. So passing the checksum is not, by itself, evidence about which
    # character the scan got wrong. What narrows it is the shape of the digits:
    # a repair that swaps a pair OCR genuinely confuses is a reading someone
    # could have made, and the rest are arithmetic coincidences.
    confusable: list[tuple[int, str, str]] = []
    for position in range(13):
        for replacement in "0123456789":
            if frozenset({digits[position], replacement}) not in CONFUSABLE_DIGITS:
                continue
            candidate = digits[:position] + replacement + digits[position + 1:]
            if thai_tax_id_checksum_ok(candidate):
                confusable.append((position, replacement, candidate))

    # Only when it is the only such reading. Two plausible repairs is a page for
    # a person to look at, not a suggestion to accept.
    if len(confusable) != 1:
        return ()

    position, replacement, candidate = confusable[0]
    return (
        SuggestedFix(
            field=field,
            current=digits,
            suggested=candidate,
            reason=(
                f"one reading that would pass the check digit: digit {position + 1} as "
                f"'{replacement}' rather than '{digits[position]}', a pair scans routinely "
                f"confuse. Other single-digit readings also pass, so check this against the "
                f"page rather than accepting it"
            ),
            source="page",
            confidence=0.55,
        ),
    )


def value_fix(
    field: str,
    current: Any,
    suggested: Any,
    reason: str,
    source: str = "master_data",
    confidence: float = 0.8,
) -> tuple[SuggestedFix, ...]:
    """A correction taken from a record we already hold.

    Nothing is suggested when the value is already right, which keeps a rule
    from proposing a change that would do nothing.
    """
    if suggested is None or str(suggested).strip() == "":
        return ()
    if current is not None and str(current).strip().casefold() == str(suggested).strip().casefold():
        return ()
    return (
        SuggestedFix(
            field=field,
            current=current,
            suggested=suggested,
            reason=reason,
            source=source,
            confidence=confidence,
        ),
    )


def best_of(fixes: Iterable[SuggestedFix]) -> tuple[SuggestedFix, ...]:
    """One suggestion per field, the most confident one, in a stable order."""
    strongest: dict[str, SuggestedFix] = {}
    for fix in fixes:
        held = strongest.get(fix.field)
        if held is None or fix.confidence > held.confidence:
            strongest[fix.field] = fix
    return tuple(sorted(strongest.values(), key=lambda fix: (-fix.confidence, fix.field)))
