"""Compare OCR engines and agent models on the ground-truth corpus.

Model choice should be an evidence-based decision, not a guess. This runs the
same documents through several engines and reports field accuracy, silent error
rate and cost in wall-clock seconds per page.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docuflow.bootstrap import bootstrap
from docuflow.config import Settings
from docuflow.evaluation.harness import _isolated_settings, _sample_path, _summarize
from docuflow.evaluation.compare import compare_document
from docuflow.pipeline import Pipeline
from docuflow.samples.catalog import SAMPLE_DOCUMENTS
from docuflow.schemas.registry import get_schema

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Combination:
    """One (OCR engine, agent model) pair to measure."""

    label: str
    ocr_provider: str
    ocr_model: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None

    def apply(self, settings: Settings) -> Settings:
        updates: dict[str, Any] = {"ocr_provider": self.ocr_provider}
        field = {
            "ollama": "ocr_ollama_model",
            "typhoon_api": "ocr_cloud_model",
            "local": "ocr_local_model",
            "openai": "ocr_openai_model",
            "claude": "ocr_anthropic_model",
        }.get(self.ocr_provider)
        if field and self.ocr_model:
            updates[field] = self.ocr_model

        if self.llm_provider:
            updates["llm_provider"] = self.llm_provider
            llm_field = {
                "ollama": "llm_ollama_model",
                "typhoon": "llm_cloud_model",
                "local": "llm_local_model",
                "openai": "llm_openai_model",
                "claude": "llm_anthropic_model",
            }.get(self.llm_provider)
            if llm_field and self.llm_model:
                updates[llm_field] = self.llm_model
        return settings.model_copy(update=updates)


def parse_combination(spec: str) -> Combination:
    """Parse `"ocr=ollama:model,llm=typhoon:model"` into a Combination."""
    ocr_provider, ocr_model, llm_provider, llm_model = "pdftext", None, None, None
    for part in spec.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        kind, value = part.split("=", 1)
        provider, _, model = value.partition(":")
        if kind.strip() == "ocr":
            ocr_provider, ocr_model = provider, model or None
        elif kind.strip() == "llm":
            llm_provider, llm_model = provider, model or None
    label = f"ocr={ocr_provider}{':' + ocr_model if ocr_model else ''}"
    if llm_provider:
        label += f" llm={llm_provider}{':' + llm_model if llm_model else ''}"
    return Combination(label, ocr_provider, ocr_model, llm_provider, llm_model)


def run_one(settings: Settings, combination: Combination, limit: int = 0) -> dict[str, Any]:
    """Evaluate a single combination and return its report."""
    eval_settings = _isolated_settings(settings)
    eval_settings = combination.apply(eval_settings)
    bootstrap(eval_settings, with_samples=False)
    pipeline = Pipeline(eval_settings)

    documents = list(SAMPLE_DOCUMENTS)[: limit or None]
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    pages_read = 0

    for sample in documents:
        path = _sample_path(settings, sample.doc_id)
        if path is None:
            continue
        outcome = pipeline.process_file(path, source_kind="sample")
        run = outcome.run
        schema = get_schema(sample.doc_type)
        specs = {spec.name: spec for spec in schema.fields}
        comparisons = compare_document(sample.ground_truth, run.field_map(), specs)
        critical = {spec.name for spec in schema.critical_fields}
        wrong = [c for c in comparisons if c.status in {"wrong", "near", "spurious"}]
        pages_read += len(run.ocr.pages)

        fired = {finding.rule_id for finding in run.failed_findings}
        expected = set(sample.expected_findings)
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
                "fields_total": len(comparisons),
                "fields_correct": sum(1 for c in comparisons if c.correct),
                "fields_near": sum(1 for c in comparisons if c.status == "near"),
                "fields_wrong": len(wrong),
                "fields_missing": sum(1 for c in comparisons if c.status == "missing"),
                "wrong_critical": [c.field for c in wrong if c.field in critical],
                "silent_error": run.decision == "auto_post"
                and any(c.field in critical for c in wrong),
                "expected_findings": sorted(expected),
                "findings_fired": sorted(fired),
                "expected_findings_caught": sorted(expected & fired),
                "expected_findings_missed": sorted(expected - fired),
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

    elapsed = round(time.perf_counter() - started, 1)
    report = _summarize(results, elapsed)
    report["combination"] = {
        "label": combination.label,
        "ocr_provider": combination.ocr_provider,
        "ocr_model": combination.ocr_model,
        "llm_provider": combination.llm_provider,
        "llm_model": combination.llm_model,
    }
    report["summary"]["pages"] = pages_read
    report["summary"]["seconds_per_page"] = round(elapsed / pages_read, 1) if pages_read else 0.0
    return report


def run_benchmark(settings: Settings, combinations: list[Combination], limit: int = 0) -> dict[str, Any]:
    """Evaluate every combination and rank them."""
    bootstrap(settings, with_samples=True)

    reports: list[dict[str, Any]] = []
    for combination in combinations:
        logger.info("benchmarking %s", combination.label)
        try:
            reports.append(run_one(settings, combination, limit=limit))
        except Exception as exc:  # noqa: BLE001 - one bad engine must not stop the sweep
            logger.exception("combination %s failed", combination.label)
            reports.append(
                {
                    "combination": {"label": combination.label},
                    "error": str(exc),
                    "summary": {"documents": 0},
                }
            )

    ranked = sorted(
        (report for report in reports if report["summary"].get("documents")),
        key=lambda report: (
            report["summary"]["silent_error_rate"],
            -report["summary"]["field_accuracy"],
            report["summary"]["seconds_per_page"],
        ),
    )
    return {"runs": reports, "ranking": [report["combination"]["label"] for report in ranked]}


def print_benchmark(result: dict[str, Any]) -> None:
    """Render the comparison table."""
    runs = result.get("runs", [])
    print("\n" + "=" * 96)
    print("MODEL BENCHMARK")
    print("=" * 96)
    print(f"{'COMBINATION':40} {'FIELD ACC':>9} {'NEAR':>6} {'SILENT':>7} {'AUTO':>6} {'DEFECTS':>8} {'S/PAGE':>7}")
    print("-" * 96)
    for report in runs:
        label = report["combination"]["label"][:39]
        if report.get("error"):
            print(f"{label:40} {'failed: ' + report['error'][:44]}")
            continue
        summary = report["summary"]
        print(
            f"{label:40} {summary['field_accuracy']:>8.1%} "
            f"{summary['field_accuracy_with_near']:>6.0%} "
            f"{summary['silent_error_rate']:>7.1%} "
            f"{summary['auto_post_rate']:>6.0%} "
            f"{summary['defect_recall']:>8.0%} "
            f"{summary['seconds_per_page']:>7.1f}"
        )

    if result.get("ranking"):
        print("\nRanked by: fewest silent errors, then field accuracy, then speed")
        for position, label in enumerate(result["ranking"], start=1):
            print(f"  {position}. {label}")
    print()
