"""How well the page was actually read.

Every other number in the evaluation is downstream of this one. A field that
extraction got wrong because the OCR text never contained the right characters
is not an extraction problem, and reporting it as one sends you off to tune the
wrong stage.

Two measurements, because they answer different questions.

**Character error rate** is the classic one: edit distance between what the
engine returned and what is printed, over the length of what is printed. It
describes the transcript as a whole and is the number to watch when comparing
engines.

**Value recall** is the one that matters here. Of the values this system has to
find - the total, the invoice number, the tax ID, the bank account - how many
survived into the OCR text at all? A page can score a respectable CER while
mangling precisely the digits that turn into a payment, because the digits are a
vanishing fraction of the characters. CER weights a misread word in an address
the same as a misread total. This does not.

The reference is exact rather than estimated: the corpus is drawn here, so the
PDF's text layer is the string the renderer was given, captured before the page
was ever photographed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from docuflow.normalize.text import normalize_for_match

# HTML tags and the `<!-- page 1 -->` markers the OCR layer emits. Structure the
# engine is right to return, and not text either document claims is on the page.
_TAG = re.compile(r"<!--.*?-->|</?[a-zA-Z][^>]*>", re.DOTALL)

# Values shorter than this are not worth scoring for recall - a currency code or
# a one-digit quantity matches by accident too often to mean anything.
MIN_VALUE_LENGTH = 4

# Amounts are printed with separators the reference and the transcript need not
# agree about: 1,500.00 and 1500.00 are the same number on the page.
_MONEY_NOISE = str.maketrans({",": None, " ": None, " ": None})


@dataclass(frozen=True)
class OcrScore:
    """What one document's transcript got right."""

    document: str
    reference_chars: int
    edit_distance: int
    values_expected: int
    values_found: int
    values_missing: tuple[str, ...] = ()

    @property
    def character_error_rate(self) -> float:
        if not self.reference_chars:
            return 0.0
        return min(1.0, self.edit_distance / self.reference_chars)

    @property
    def character_accuracy(self) -> float:
        return 1.0 - self.character_error_rate

    @property
    def value_recall(self) -> float:
        if not self.values_expected:
            return 1.0
        return self.values_found / self.values_expected

    def to_json(self) -> dict[str, Any]:
        return {
            "document": self.document,
            "character_accuracy": round(self.character_accuracy, 4),
            "character_error_rate": round(self.character_error_rate, 4),
            "reference_chars": self.reference_chars,
            "edit_distance": self.edit_distance,
            "value_recall": round(self.value_recall, 4),
            "values_expected": self.values_expected,
            "values_found": self.values_found,
            "values_missing": list(self.values_missing),
        }


def _collapse(text: str) -> str:
    """The words on the page, with everything structural taken out.

    Two things are removed and both would otherwise be scored as errors while
    being entirely correct.

    **Markup.** Typhoon OCR returns structure - HTML tables, page markers -
    because a total that stays inside a table cell is a total that can still be
    found later. That is the feature. Compared against a PDF's plain text layer,
    every `<td>` is half a dozen characters of "error" for output that is doing
    its job, and a table-heavy invoice would score worse than a page of prose
    the model read badly.

    **Whitespace.** OCR breaks lines where it saw columns, the text layer where
    the renderer did. Scoring that measures a disagreement about layout.

    Tags become a space rather than nothing, or `<td>A</td><td>B</td>` collapses
    into `AB` and invents a word that is in neither document.
    """
    without_markup = _TAG.sub(" ", text)
    return " ".join(without_markup.split())


def _edit_distance(left: str, right: str) -> int:
    """Levenshtein, on two rows rather than a full matrix.

    A page of text against a page of text is a few thousand characters each
    way; the full matrix is tens of millions of cells and megabytes of nothing.
    """
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, source in enumerate(left, start=1):
        current = [i]
        for j, target in enumerate(right, start=1):
            current.append(min(
                previous[j] + 1,                                  # deletion
                current[j - 1] + 1,                               # insertion
                previous[j - 1] + (source != target),             # substitution
            ))
        previous = current
    return previous[-1]


def _searchable(text: str) -> str:
    """The transcript, flattened so a value can be looked for in it."""
    return normalize_for_match(_collapse(text)).translate(_MONEY_NOISE)


def _significant(value: Any) -> int:
    """How many characters of a value actually identify it.

    Judged on the compact form, not on any of the padded ones. A tax rate of 7.0
    prints as "7.00", which is four characters and none of them distinguishing -
    scoring the length of that form lets a single digit into a recall figure
    where it will match by accident on almost any page.
    """
    if isinstance(value, bool) or value is None:
        return 0
    compact = f"{float(value):g}" if isinstance(value, (int, float)) else str(value)
    return sum(1 for char in compact if char.isalnum())


def _printed_forms(value: Any) -> tuple[str, ...]:
    """How a value might legitimately appear on the page.

    An amount stored as 1500.0 is printed as "1,500.00"; a rate stored as 7.0 is
    printed as "7%". Searching for the stored form alone would report the OCR as
    having lost values that are plainly there.
    """
    if isinstance(value, bool) or value is None:
        return ()
    if isinstance(value, (int, float)):
        number = float(value)
        forms = {f"{number:.2f}", f"{number:,.2f}", f"{number:g}"}
        if number.is_integer():
            forms.add(str(int(number)))
        return tuple(forms)
    return (str(value),)


def score_document(reference: str, transcript: str, values: dict[str, Any],
                   document: str = "") -> OcrScore:
    """Score one transcript against the text that was printed."""
    reference_flat = _collapse(reference)
    transcript_flat = _collapse(transcript)

    haystack = _searchable(transcript)
    expected = 0
    found = 0
    missing: list[str] = []
    for name, value in values.items():
        if _significant(value) < MIN_VALUE_LENGTH:
            continue
        forms = [_searchable(form) for form in _printed_forms(value)]
        if not any(forms):
            continue
        expected += 1
        if any(form and form in haystack for form in forms):
            found += 1
        else:
            missing.append(name)

    return OcrScore(
        document=document,
        reference_chars=len(reference_flat),
        edit_distance=_edit_distance(reference_flat, transcript_flat),
        values_expected=expected,
        values_found=found,
        values_missing=tuple(missing),
    )


def summarize(scores: list[OcrScore]) -> dict[str, Any]:
    """Corpus-level OCR quality.

    Character accuracy is pooled over characters rather than averaged over
    documents: a one-line receipt and a full contract should not count equally
    towards a figure about characters. Value recall is pooled over values for
    the same reason.
    """
    if not scores:
        return {"documents": 0}

    reference_chars = sum(score.reference_chars for score in scores)
    edits = sum(score.edit_distance for score in scores)
    expected = sum(score.values_expected for score in scores)
    found = sum(score.values_found for score in scores)

    worst = sorted(scores, key=lambda score: score.character_accuracy)[:5]
    return {
        "documents": len(scores),
        "character_accuracy": round(1 - (edits / reference_chars), 4) if reference_chars else 1.0,
        "character_error_rate": round(edits / reference_chars, 4) if reference_chars else 0.0,
        "reference_chars": reference_chars,
        "edit_distance": edits,
        "value_recall": round(found / expected, 4) if expected else 1.0,
        "values_expected": expected,
        "values_found": found,
        "worst_documents": [score.to_json() for score in worst],
        "per_document": [score.to_json() for score in scores],
    }
