"""Finding where one document ends and the next begins.

A scanner produces one PDF from a stack of paper. Treating that file as one
document extracts one invoice number, one total, one vendor - and every other
invoice in the stack is silently never recorded. Nothing downstream can notice:
the run looks like an ordinary single invoice that posted cleanly.

That asymmetry decides the design.

**Missing a boundary is a silent error.** An invoice that was never recorded is
indistinguishable from one that never arrived, and it is exactly the failure
this system exists to measure.

**Splitting where there is no boundary is a loud one.** Page two of an invoice
becomes a fragment with no header, no invoice number and no total, fails
`purchase_order_required` and half the arithmetic, and lands in the review queue
where somebody sees it in ten seconds.

So this module is willing to split and reluctant to merge, and where it cannot
tell it says so rather than guessing - an ambiguous file goes to a person with
the reason attached, which is cheaper than either mistake.

Deterministic on purpose. A boundary is a fact about the text - a reference
number that changed, a header that reappeared - and a model asked per page adds
latency, cost and non-determinism to a question that regular expressions answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from docuflow.normalize.text import normalize_for_match
from docuflow.schemas.registry import DOCUMENT_SCHEMAS

# How far into a page a heading has to appear to count as that page's heading.
# A "TAX INVOICE" in the footer of a terms-and-conditions page is not the start
# of anything; the same words across the top are.
HEADING_WINDOW_CHARS = 400

# Reference numbers: the strongest signal there is. Two pages carrying two
# different invoice numbers are two invoices, whatever else the layout does.
_REFERENCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:invoice|tax\s*invoice|receipt|contract|agreement|credit\s*note)"
        r"\s*(?:no|number|#|:)[.:\s#]*([A-Z0-9][A-Z0-9\-/]{3,})",
        re.IGNORECASE,
    ),
    re.compile(r"(?:เลขที่|เลขที่ใบเสร็จ|เลขที่ใบกำกับภาษี)\s*[:：]?\s*([A-Z0-9][A-Z0-9\-/]{3,})"),
)

# "Page 2 of 5" and its Thai equivalent - an explicit statement that this page
# is a continuation, which outranks a heading that happens to repeat.
_CONTINUATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bpage\s+(\d+)\s*(?:of|/)\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"หน้า\s*(?:ที่)?\s*(\d+)\s*(?:จาก|/)\s*(\d+)"),
)


@dataclass(frozen=True)
class Segment:
    """One document inside the file, as a page range."""

    start_page: int
    end_page: int
    reason: str
    doc_type: str = "unknown"

    @property
    def pages(self) -> int:
        return self.end_page - self.start_page + 1


@dataclass(frozen=True)
class SplitReport:
    """What was found, and how sure it is."""

    segments: tuple[Segment, ...]
    # True when the file holds more than one document.
    is_multi: bool
    # Set when the boundaries could not be decided confidently. The caller is
    # expected to route the file to a person rather than pick an answer.
    uncertain: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)


def _first_reference(text: str) -> str | None:
    """The document number this page states, if it states one."""
    for pattern in _REFERENCE_PATTERNS:
        match = pattern.search(text)
        if match:
            token = match.group(1).strip().rstrip(".,;:")
            # A bare year or a page number is not a document reference.
            if len(token) >= 4 and not token.isdigit():
                return token.upper()
    return None


def _continuation(text: str) -> tuple[int, int] | None:
    """`(page, of)` when the page says which of how many it is."""
    for pattern in _CONTINUATION_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return int(match.group(1)), int(match.group(2))
            except ValueError:
                continue
    return None


def _heading_type(text: str) -> str | None:
    """The document type this page announces at its top, if any.

    Only the most specific matched phrase counts, for the same reason the
    classifier works that way: a Thai till prints `ใบกำกับภาษีอย่างย่อ`, an
    *abbreviated* tax invoice, which contains the invoice keyword whole.
    """
    head = normalize_for_match(text[:HEADING_WINDOW_CHARS])
    if not head:
        return None
    best_type, best_len = None, 0
    for name, schema in DOCUMENT_SCHEMAS.items():
        if name == "unknown":
            continue
        for phrase in schema.keywords:
            normalized = normalize_for_match(phrase)
            if normalized and normalized in head and len(normalized) > best_len:
                best_type, best_len = name, len(normalized)
    return best_type


def detect(pages: list[tuple[int, str]]) -> SplitReport:
    """Where the documents in this file begin and end.

    `pages` is `(page_number, recognised_text)` in order - the same shape the
    pipeline already carries after reading.
    """
    if not pages:
        return SplitReport(segments=(), is_multi=False)
    if len(pages) == 1:
        return SplitReport(
            segments=(Segment(1, 1, "single page", _heading_type(pages[0][1]) or "unknown"),),
            is_multi=False,
        )

    starts: list[tuple[int, str, str]] = [(pages[0][0], "first page", _heading_type(pages[0][1]) or "unknown")]
    notes: list[str] = []
    current_reference = _first_reference(pages[0][1])
    unreadable = 0

    for number, text in pages[1:]:
        if len(text.strip()) < 40:
            # Nothing to reason about. Kept with whatever came before rather
            # than guessed at, and counted - a file of blank pages is not a
            # file this function should be answering about.
            unreadable += 1
            continue

        continuation = _continuation(text)
        reference = _first_reference(text)
        heading = _heading_type(text)

        # An explicit "page 2 of 5" outranks everything: the page states that
        # it belongs to something that started earlier.
        if continuation and continuation[0] > 1:
            if reference and current_reference and reference != current_reference:
                notes.append(
                    f"page {number} says it is page {continuation[0]} of {continuation[1]} but "
                    f"quotes {reference} where the previous page quoted {current_reference}"
                )
            continue

        # A reference number that changed is the strongest signal there is.
        if reference and current_reference and reference != current_reference:
            starts.append((number, f"reference changed to {reference}", heading or "unknown"))
            current_reference = reference
            continue

        # The same reference is the same document, whatever else the page shows.
        # A continuation page often repeats the invoice number in its own header
        # block, and `_heading_type` sees the word "Invoice" in "Invoice No." -
        # so without this the document splits from itself.
        if reference and reference == current_reference:
            continue

        # A heading over a reference we had not seen before. This is the case
        # where the previous page carried no readable number at all: an invoice
        # number appearing for the first time under a heading is a new document
        # far more often than it is page two of one.
        if heading and reference and not current_reference:
            starts.append((number, f"{heading} heading with its own reference", heading))
            current_reference = reference
            continue

        if reference and not current_reference:
            current_reference = reference

    segments: list[Segment] = []
    for index, (start, reason, doc_type) in enumerate(starts):
        end = starts[index + 1][0] - 1 if index + 1 < len(starts) else pages[-1][0]
        segments.append(Segment(start, end, reason, doc_type))

    if unreadable:
        notes.append(
            f"{unreadable} page(s) held too little text to place; they stayed with "
            f"the document before them"
        )

    # Every page unreadable but the first means the file was never really read,
    # and any answer here would be invented.
    uncertain = unreadable > 0 and unreadable >= (len(pages) - 1)
    if uncertain:
        notes.append("too little text was recognised to decide where documents begin")

    return SplitReport(
        segments=tuple(segments),
        is_multi=len(segments) > 1,
        uncertain=uncertain,
        notes=tuple(notes),
    )


def describe(report: SplitReport) -> str:
    """One line per document, for a trace or a CLI."""
    if not report.segments:
        return "no pages"
    if not report.is_multi:
        return f"one document, {report.segments[0].pages} page(s)"
    parts = [
        f"pages {segment.start_page}-{segment.end_page} ({segment.doc_type}, {segment.reason})"
        for segment in report.segments
    ]
    return f"{len(report.segments)} documents: " + "; ".join(parts)
