"""Reviewer corrections - the feedback loop that makes the system improve.

Every edit a reviewer makes is recorded against the document type and vendor.
The aggregated view shows which fields the extractor gets wrong most often,
which is what tells you where to spend the next improvement effort.

That much was true and only half the sentence: the corrections were recorded
and reported to a person, and nothing ever read them back into the pipeline.
`unreliable_fields` closes it.

**What is learned is trust, not values.** A correction says what the right
answer was on one document; it says nothing about what the next one should say,
and feeding a past value forward as a hint would put one invoice's total on
another. What does generalise is that this vendor's layout, in this field, has
repeatedly defeated the extractor - so confidence in that field drops, the
document stops auto-posting, and a person sees it. The system becomes careful
exactly where it has been shown to be wrong, and nowhere else.

**Counted by document, not by edit.** Three edits to one field on one document
is one person fixing one page. The same field wrong on three different
documents is a layout that cannot be read. Only the second generalises.

**Aged out.** A raw count can only rise, so a field read badly last year would
be distrusted forever - and since distrust sends documents to review, and
review produces corrections, it would climb on its own. A window is what lets
a vendor recover by simply being read correctly for a while.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from docuflow.db import connect, rows_to_dicts, transaction

# How many separate documents must show the same field corrected before it is
# treated as a pattern. Two is a coincidence and acting on it would lengthen
# the review queue without making a single payment safer.
MIN_DOCUMENTS = 3

# How far back to look. Without this a count only rises, so a field that was
# read badly last year would be distrusted forever - and since distrust sends
# documents to review, and review produces corrections, it would climb on its
# own. Ageing out is what lets a vendor recover by being read correctly.
WINDOW_DAYS = 180

# How much of a field's confidence a proven-unreliable field loses. Enough to
# drop it below the auto-post threshold on its own, not enough to make it look
# like the value was never read.
UNRELIABLE_FIELD_FACTOR = 0.6


class CorrectionStore:
    """Records of human edits to extracted values."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def record(
        self,
        document_id: str,
        doc_type: str,
        field_name: str,
        old_value: Any,
        new_value: Any,
        vendor_key: str | None = None,
        reviewer: str = "reviewer",
    ) -> None:
        with transaction(self.db_path) as connection:
            connection.execute(
                """INSERT INTO corrections
                   (document_id, doc_type, vendor_key, field_name, old_value, new_value, reviewer, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    document_id,
                    doc_type,
                    vendor_key,
                    field_name,
                    None if old_value is None else str(old_value),
                    None if new_value is None else str(new_value),
                    reviewer,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def for_document(self, document_id: str) -> list[dict[str, Any]]:
        rows = connect(self.db_path).execute(
            "SELECT * FROM corrections WHERE document_id = ? ORDER BY id ASC", (document_id,)
        ).fetchall()
        return rows_to_dicts(rows)

    def error_hotspots(self, limit: int = 15) -> list[dict[str, Any]]:
        """Fields most often corrected - where accuracy work pays off most."""
        rows = connect(self.db_path).execute(
            """SELECT doc_type, field_name, COUNT(*) AS corrections
               FROM corrections GROUP BY doc_type, field_name
               ORDER BY corrections DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return rows_to_dicts(rows)

    def total(self) -> int:
        row = connect(self.db_path).execute("SELECT COUNT(*) AS total FROM corrections").fetchone()
        return int(row["total"]) if row else 0

    def unreliable_fields(
        self,
        doc_type: str,
        vendor_key: str | None,
        window_days: int = WINDOW_DAYS,
        minimum: int = MIN_DOCUMENTS,
    ) -> dict[str, int]:
        """Fields of this vendor's documents that reviewers keep having to fix.

        `{field_name: how many separate documents}`, counting only corrections
        inside the recency window.

        **Separate documents, not corrections.** Three edits to one field on
        one document is one person fixing one page; the same field wrong on
        three different documents is a layout this extractor cannot read. Only
        the second generalises, and `COUNT(DISTINCT document_id)` is the whole
        difference between them.

        **Only recent ones.** A raw count can only rise, so a field that was
        wrong last year would be distrusted forever - and worse, distrust sends
        documents to review, review produces corrections, and the count climbs
        on its own. Ageing out lets a vendor recover by simply being read
        correctly for a while.
        """
        if not vendor_key:
            # Nothing to key on. A correction against an unknown vendor still
            # counts in the hotspot report, but it cannot be attributed to a
            # layout, and applying it to every vendor of this type would spread
            # one supplier's problem across all of them.
            return {}

        cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
        rows = connect(self.db_path).execute(
            """SELECT field_name, COUNT(DISTINCT document_id) AS documents
                 FROM corrections
                WHERE doc_type = ? AND vendor_key = ? AND created_at >= ?
             GROUP BY field_name
               HAVING documents >= ?""",
            (doc_type, vendor_key, cutoff, minimum),
        ).fetchall()
        return {str(row["field_name"]): int(row["documents"]) for row in rows}
