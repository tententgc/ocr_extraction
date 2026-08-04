"""What the reading looks like, one document type at a time.

The records ledger answers "what did we post". This answers the question the
person improving the system actually asks: for *this* kind of document, what
holds it back - a business rule, a field the extractor keeps missing, or a
value a reviewer keeps having to correct?

Everything is grouped by document type on purpose. An invoice held for a
missing purchase order and a contract held for expiring next week are different
problems with different fixes, and one blended average across all types hides
precisely the thing worth seeing.

Aggregation runs inside SQLite over the stored run payload, so counting a rule
never means parsing a full `DocumentRun` in Python. That scan is proportional to
the corpus: at demo and departmental volumes it is far cheaper than the parse,
and a corpus large enough to feel it wants findings materialised into their own
table rather than a faster query here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docuflow.db import connect, rows_to_dicts
from docuflow.schemas.registry import get_schema
from docuflow.store.corrections import CorrectionStore

# How many rules / fields / hotspots each type shows before the tail is cut.
TOP_N = 6
# Correction hotspots are stored flat and split per type here, so enough rows
# have to be read for every type to get its share rather than the busiest few.
HOTSPOT_SCAN = 100


class InsightStore:
    """Read-only aggregates over processed documents, split by document type."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    # ---------------------------------------------------------------- volume

    def totals_by_type(self) -> list[dict[str, Any]]:
        """Volume, confidence and the period covered, per type."""
        rows = connect(self.db_path).execute(
            """SELECT doc_type,
                      COUNT(*)                    AS total,
                      AVG(confidence)             AS average_confidence,
                      substr(MIN(created_at), 1, 10) AS earliest,
                      substr(MAX(created_at), 1, 10) AS latest
               FROM documents
               GROUP BY doc_type
               ORDER BY total DESC"""
        ).fetchall()
        return rows_to_dicts(rows)

    def decisions_by_type(self) -> dict[str, dict[str, int]]:
        """How each type was routed: posted on its own, held, or rejected."""
        rows = connect(self.db_path).execute(
            """SELECT doc_type, decision, COUNT(*) AS total
               FROM documents GROUP BY doc_type, decision"""
        ).fetchall()
        decisions: dict[str, dict[str, int]] = {}
        for row in rows:
            decisions.setdefault(row["doc_type"], {})[row["decision"]] = row["total"]
        return decisions

    # ----------------------------------------------------------------- rules

    def failing_rules_by_type(self) -> dict[str, list[dict[str, Any]]]:
        """Which rules fail, per type, counted in documents rather than findings.

        A document that trips the same rule twice is still one held document,
        which is what a person deciding where to spend effort wants to know.
        """
        rows = connect(self.db_path).execute(
            """SELECT d.doc_type                              AS doc_type,
                      json_extract(f.value, '$.rule_id')      AS rule_id,
                      json_extract(f.value, '$.severity')     AS severity,
                      COUNT(DISTINCT d.document_id)           AS documents
               FROM documents d, json_each(d.payload, '$.findings') f
               WHERE json_extract(f.value, '$.passed') = 0
               GROUP BY d.doc_type, rule_id, severity
               ORDER BY documents DESC, rule_id"""
        ).fetchall()
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows_to_dicts(rows):
            grouped.setdefault(row.pop("doc_type"), []).append(row)
        return grouped

    def blocked_documents_by_type(self) -> dict[str, int]:
        """Documents held by at least one blocking failure, per type.

        Counted separately rather than summed from `failing_rules_by_type`:
        one document can trip two blocking rules, and adding those up would
        report more blocked documents than exist.
        """
        rows = connect(self.db_path).execute(
            """SELECT d.doc_type, COUNT(DISTINCT d.document_id) AS blocked
               FROM documents d, json_each(d.payload, '$.findings') f
               WHERE json_extract(f.value, '$.passed') = 0
                 AND json_extract(f.value, '$.severity') = 'blocking'
               GROUP BY d.doc_type"""
        ).fetchall()
        return {row["doc_type"]: row["blocked"] for row in rows}

    # ---------------------------------------------------------------- fields

    def field_gaps_by_type(self) -> dict[str, list[dict[str, Any]]]:
        """Fields that came back empty, or filled but not verifiable on the page.

        These are the two failures worth separating: nothing was read, versus
        something was read that no quote on the page supports. The first is an
        extraction gap, the second is a trust problem.
        """
        rows = connect(self.db_path).execute(
            """SELECT d.doc_type                          AS doc_type,
                      json_extract(x.value, '$.name')     AS field_name,
                      COUNT(*)                            AS seen,
                      SUM(CASE WHEN json_extract(x.value, '$.value') IS NULL
                                 OR json_extract(x.value, '$.value') = ''
                               THEN 1 ELSE 0 END)         AS missing,
                      SUM(CASE WHEN json_extract(x.value, '$.value') IS NOT NULL
                                AND json_extract(x.value, '$.value') <> ''
                                AND json_extract(x.value, '$.grounded') = 0
                               THEN 1 ELSE 0 END)         AS unverified
               FROM documents d, json_each(d.payload, '$.fields') x
               GROUP BY d.doc_type, field_name"""
        ).fetchall()

        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows_to_dicts(rows):
            doc_type = row.pop("doc_type")
            if not (row["missing"] or row["unverified"]):
                continue  # nothing wrong with this field; it does not belong on a gap list
            spec = get_schema(doc_type).field(row["field_name"])
            grouped.setdefault(doc_type, []).append(
                {
                    **row,
                    "label": spec.label if spec else row["field_name"],
                    # A missing optional note is not the same problem as a missing
                    # total, so the page can rank and mark them differently.
                    "required": bool(spec and spec.required),
                    "critical": bool(spec and spec.critical),
                }
            )
        for gaps in grouped.values():
            gaps.sort(key=lambda gap: (-gap["critical"], -gap["required"], -(gap["missing"] + gap["unverified"])))
        return grouped

    # ------------------------------------------------------------- assembled

    def describe_types(self, limit: int = TOP_N) -> list[dict[str, Any]]:
        """One entry per document type seen, with its own summary and problems."""
        decisions = self.decisions_by_type()
        failing = self.failing_rules_by_type()
        blocked = self.blocked_documents_by_type()
        gaps = self.field_gaps_by_type()
        corrections = _corrections_by_type(
            CorrectionStore(self.db_path).error_hotspots(limit=HOTSPOT_SCAN)
        )

        described = []
        for row in self.totals_by_type():
            doc_type = row["doc_type"]
            schema = get_schema(doc_type)
            routed = decisions.get(doc_type, {})
            auto_posted = routed.get("auto_post", 0)
            described.append(
                {
                    **row,
                    "title": schema.display_name,
                    "blurb": schema.description,
                    "record_kind": schema.record_kind,
                    "average_confidence": round(row["average_confidence"] or 0.0, 3),
                    "decisions": routed,
                    "auto_posted": auto_posted,
                    "held": row["total"] - auto_posted,
                    "auto_post_rate": round(auto_posted / row["total"], 3) if row["total"] else 0.0,
                    "blocked": blocked.get(doc_type, 0),
                    "failing_rules": failing.get(doc_type, [])[:limit],
                    "field_gaps": gaps.get(doc_type, [])[:limit],
                    "corrections": corrections.get(doc_type, [])[:limit],
                }
            )
        return described


def _corrections_by_type(hotspots: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Reshape the flat (doc_type, field) hotspot list into per-type lists."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for spot in hotspots:
        grouped.setdefault(spot["doc_type"], []).append(
            {"field_name": spot["field_name"], "corrections": spot["corrections"]}
        )
    return grouped
