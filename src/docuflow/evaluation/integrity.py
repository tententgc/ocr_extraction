"""Is the corpus gradeable, checked before anything is graded against it.

On this project the corpus has been wrong more often than the pipeline. A VAT
that was correct when it was meant to be wrong, two contracts with no body text,
four documents claiming fields their layout never drew, two real defects
labelled clean. Every one looked like a system fault until it was read closely,
and the expensive part was always the time spent tuning a stage that was fine.

So the run states what it can prove about its own answer sheet, first, in the
same report as the numbers.

The check that finds most of it: **does the page actually print this value?**
`reference_text` is the PDF's own text layer, captured before the page was ever
photographed - it is not an approximation of what was drawn, it is the string
the renderer was handed. A ground-truth value that is not in it was never on the
page, so extraction is being graded against something no reader could have
found, and the field is reported as missing on every engine anyone ever tries.

Nothing here is silently corrected. A harness that quietly drops the fields it
cannot verify is a harness that reports a better number every time the corpus
gets worse. It lists them, and a person decides whether the ground truth is
wrong or the layout is.

No model, no OCR, no API key: this reads the manifest and nothing else.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from docuflow.evaluation.ocr_accuracy import (
    MIN_VALUE_LENGTH,
    _printed_forms,
    _searchable,
    _significant,
)

# Below this a text layer is too thin to prove anything either way, and every
# value would be reported as undrawn. A photograph has no text layer at all,
# which is why the reference is taken from the PDF before it is photographed -
# a short one means that capture went wrong.
MIN_REFERENCE_CHARS = 200


def undrawn_values(document: Any) -> list[tuple[str, Any]]:
    """Ground-truth values this document's page does not print.

    Uses the same matching as OCR value recall - the printed forms of a date,
    the separator-free form of an amount - so a value counted as found there is
    counted as drawn here.
    """
    reference = (document.reference_text or "").strip()
    if len(reference) < MIN_REFERENCE_CHARS:
        return []

    haystack = _searchable(reference)
    undrawn: list[tuple[str, Any]] = []
    for name, value in (document.ground_truth or {}).items():
        if _significant(value) < MIN_VALUE_LENGTH:
            continue
        forms = [_searchable(form) for form in _printed_forms(value)]
        if not any(forms):
            continue
        if not any(form and form in haystack for form in forms):
            undrawn.append((name, value))
    return undrawn


def check(documents: Iterable[Any]) -> dict[str, Any]:
    """Everything provable about the corpus without running anything."""
    documents = list(documents)
    if not documents:
        return {"documents": 0}

    undrawn: list[dict[str, Any]] = []
    by_field: Counter[str] = Counter()
    thin_reference: list[str] = []
    ungradeable: list[str] = []
    undeclared_defects: list[str] = []

    for document in documents:
        reference = (document.reference_text or "").strip()
        if len(reference) < MIN_REFERENCE_CHARS:
            thin_reference.append(document.file)
        if not document.ground_truth:
            ungradeable.append(document.file)
        # A document that must be held while naming no rule grades as a missed
        # defect against whatever the system does, with nothing to explain it.
        if not document.posts and not document.expected_findings:
            undeclared_defects.append(document.file)

        for name, value in undrawn_values(document):
            undrawn.append({"document": document.file, "field": name, "value": value})
            by_field[name] += 1

    gradeable = sum(
        1
        for document in documents
        for name, value in (document.ground_truth or {}).items()
        if _significant(value) >= MIN_VALUE_LENGTH
    )

    return {
        "documents": len(documents),
        "gradeable_values": gradeable,
        # Claimed by the answer sheet, absent from the page. Each one is a
        # field that reports as missing on every engine anyone ever tries.
        "undrawn_values": len(undrawn),
        "undrawn_share": round(len(undrawn) / gradeable, 4) if gradeable else 0.0,
        "undrawn_documents": sorted({item["document"] for item in undrawn}),
        "undrawn_by_field": dict(by_field.most_common()),
        "undrawn": undrawn,
        "thin_reference": thin_reference,
        "documents_without_ground_truth": ungradeable,
        "held_without_a_declared_rule": undeclared_defects,
        "clean": not undrawn and not ungradeable and not thin_reference,
    }
