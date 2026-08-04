"""Run the pipeline over the ground-truth corpus and measure accuracy.

The headline number is not field accuracy - it is the **silent error rate**: the
share of documents that posted automatically while carrying a wrong critical
value. That is the only metric that maps directly onto "wrong values flowing
downstream are costly".
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from docuflow.bootstrap import bootstrap
from docuflow.config import Settings
from docuflow.evaluation.compare import compare_document
from docuflow.evaluation.cost import document_cost, summarize_cost
from docuflow.pipeline import Pipeline
from docuflow.samples.catalog import SAMPLE_DOCUMENTS
from docuflow.schemas.registry import get_schema

logger = logging.getLogger(__name__)


def _isolated_settings(settings: Settings, label: str = "run") -> Settings:
    """A settings copy pointing at a throwaway data directory.

    Evaluation must not post into the demo ledger, and must start with an empty
    records table so duplicate detection does not fire on its own output.

    The directory name is unique per call: a benchmark evaluates several engines
    in sequence, and a shared path would let one run delete another's workspace
    (and two concurrent evaluations would corrupt each other outright).
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-")[:40] or "run"
    eval_dir = settings.data_dir.parent / "eval" / "out" / f"{slug}-{uuid.uuid4().hex[:8]}"
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    return settings.model_copy(update={"data_dir": eval_dir})


def seed_prerequisite(pipeline, sample, processed: dict[str, Any]) -> None:
    """Force-post the record a relative defect depends on.

    `inv-duplicate` is only a duplicate if `inv-clean-classic` already posted.
    A cautious engine that held the predecessor for review would otherwise be
    scored as having *missed* the duplicate, penalising it for being careful.
    """
    required = getattr(sample, "requires_posted", None)
    if not required:
        return

    earlier = processed.get(required)
    if earlier is None or earlier.run.record_id:
        return

    try:
        pipeline.records.post_record(
            earlier.run, earlier.vendor_match.vendor_id, actor="evaluation", force=True
        )
    except Exception as exc:  # noqa: BLE001 - seeding is best-effort
        logger.warning("could not seed prerequisite %s: %s", required, exc)


def _sample_path(settings: Settings, doc_id: str) -> Path | None:
    scan = settings.sample_dir / f"{doc_id}.jpg"
    pdf = settings.sample_dir / f"{doc_id}.pdf"
    if scan.exists():
        return scan
    return pdf if pdf.exists() else None


def run_evaluation(settings: Settings, limit: int = 0) -> dict[str, Any]:
    """Process the whole corpus and return a structured accuracy report."""
    bootstrap(settings, with_samples=True)

    eval_settings = _isolated_settings(settings)
    bootstrap(eval_settings, with_samples=False)
    pipeline = Pipeline(eval_settings)

    documents = list(SAMPLE_DOCUMENTS)
    if limit:
        documents = documents[:limit]

    results: list[dict[str, Any]] = []
    processed: dict[str, Any] = {}
    started = time.perf_counter()

    for sample in documents:
        path = _sample_path(settings, sample.doc_id)
        if path is None:
            logger.warning("sample %s has not been rendered; skipping", sample.doc_id)
            continue

        logger.info("evaluating %s", sample.doc_id)
        seed_prerequisite(pipeline, sample, processed)
        outcome = pipeline.process_file(path, source_kind="sample")
        processed[sample.doc_id] = outcome
        run = outcome.run
        schema = get_schema(sample.doc_type)
        specs = {spec.name: spec for spec in schema.fields}

        comparisons = compare_document(sample.ground_truth, run.field_map(), specs)
        critical_names = {spec.name for spec in schema.critical_fields}

        wrong = [c for c in comparisons if c.status in {"wrong", "near", "spurious"}]
        missing = [c for c in comparisons if c.status == "missing"]
        wrong_critical = [c for c in wrong if c.field in critical_names]

        fired = {finding.rule_id for finding in run.failed_findings}
        expected_findings = set(sample.expected_findings)

        results.append(
            {
                "doc_id": sample.doc_id,
                "doc_type": sample.doc_type,
                "classified_as": run.doc_type,
                "type_correct": run.doc_type == sample.doc_type,
                "decision": run.decision,
                "confidence": run.confidence,
                "ocr_provider": run.ocr.provider,
                "ocr_ms": run.ocr.duration_ms,
                "cost": document_cost(run),
                "fields_total": len(comparisons),
                "fields_correct": sum(1 for c in comparisons if c.correct),
                "fields_near": sum(1 for c in comparisons if c.status == "near"),
                "fields_wrong": len(wrong),
                "fields_missing": len(missing),
                "wrong_critical": [c.field for c in wrong_critical],
                # The dangerous case: posted without a human, carrying a bad value.
                "silent_error": run.decision == "auto_post" and bool(wrong_critical),
                "expected_findings": sorted(expected_findings),
                "findings_fired": sorted(fired),
                "expected_findings_caught": sorted(expected_findings & fired),
                "expected_findings_missed": sorted(expected_findings - fired),
                "comparisons": [
                    {
                        "field": c.field,
                        "expected": c.expected,
                        "actual": c.actual,
                        "status": c.status,
                        "similarity": c.similarity,
                    }
                    for c in comparisons
                ],
            }
        )

    return _summarize(results, round(time.perf_counter() - started, 1))


def _summarize(results: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    """Aggregate per-document results into the report."""
    if not results:
        return {"documents": [], "summary": {"documents": 0}, "field_breakdown": {}}

    total_fields = sum(item["fields_total"] for item in results)
    correct_fields = sum(item["fields_correct"] for item in results)
    near_fields = sum(item["fields_near"] for item in results)
    missing_fields = sum(item["fields_missing"] for item in results)

    auto_posted = [item for item in results if item["decision"] == "auto_post"]
    silent_errors = [item for item in results if item["silent_error"]]

    expected_total = sum(len(item["expected_findings"]) for item in results)
    caught_total = sum(len(item["expected_findings_caught"]) for item in results)

    # Per-field accuracy across the corpus, worst first.
    breakdown: dict[str, dict[str, int]] = {}
    for item in results:
        for comparison in item["comparisons"]:
            entry = breakdown.setdefault(
                comparison["field"], {"total": 0, "correct": 0, "wrong": 0, "missing": 0, "near": 0}
            )
            entry["total"] += 1
            status = comparison["status"]
            if status == "correct":
                entry["correct"] += 1
            elif status == "missing":
                entry["missing"] += 1
            elif status == "near":
                entry["near"] += 1
            else:
                entry["wrong"] += 1

    for entry in breakdown.values():
        entry["accuracy"] = round(entry["correct"] / entry["total"], 3) if entry["total"] else 0.0

    return {
        "summary": {
            "documents": len(results),
            "elapsed_seconds": elapsed,
            "type_accuracy": round(sum(1 for i in results if i["type_correct"]) / len(results), 3),
            "field_accuracy": round(correct_fields / total_fields, 3) if total_fields else 0.0,
            "field_accuracy_with_near": round((correct_fields + near_fields) / total_fields, 3) if total_fields else 0.0,
            "fields_total": total_fields,
            "fields_correct": correct_fields,
            "fields_near": near_fields,
            "fields_missing": missing_fields,
            "auto_post_rate": round(len(auto_posted) / len(results), 3),
            "auto_posted": len(auto_posted),
            # The number that matters: bad values that reached the ledger.
            "silent_error_rate": round(len(silent_errors) / len(results), 3),
            "silent_errors": len(silent_errors),
            "silent_error_documents": [item["doc_id"] for item in silent_errors],
            "auto_post_precision": round(1 - len(silent_errors) / len(auto_posted), 3) if auto_posted else 1.0,
            "planted_defects": expected_total,
            "planted_defects_caught": caught_total,
            "defect_recall": round(caught_total / expected_total, 3) if expected_total else 1.0,
            "average_confidence": round(sum(i["confidence"] for i in results) / len(results), 3),
        },
        # What it took, beside what it got right: an optimisation is only worth
        # having if the numbers above it did not move. Empty on a report written
        # before timings were collected, and the section is then not printed.
        "cost": summarize_cost([item["cost"] for item in results if item.get("cost")]),
        "field_breakdown": dict(sorted(breakdown.items(), key=lambda kv: kv[1]["accuracy"])),
        "documents": results,
    }


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
