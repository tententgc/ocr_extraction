"""Run the working corpus and score it four ways.

`harness.py` scores the thirteen catalogue samples in `data/samples`. This runs
`real_samples/` - the documents that arrive the way documents actually arrive,
whose expected outcome is written in the filename before the system is asked.

The distinction is not academic. Every number in `docs/EVALUATION.md` is from
this corpus, and until this module existed the command the document told you to
run (`docuflow evaluate`) measured a different set of documents and printed a
different report. A published figure nobody can reproduce is a figure nobody
should believe, including the person who published it.

Four scores, because they fail independently and each one hides the others.

**Fields** - was the value read correctly, against ground truth the corpus
carries.

**OCR** - was the *page* read at all, against `reference_text` taken from the
PDF's text layer before the page was photographed. A field wrong because the
characters never arrived is not an extraction problem, and reporting it as one
sends you to tune the wrong stage.

**Gatekeeping** - did the document go the right way, and did each rule fire for
the reason it exists. Missed defects and false alarms are counted apart and
never summed: one is a wrong payment, the other a wasted minute.

**Effort** - modelled, not measured, with its assumptions attached.

Documents run in the order their filenames give, because document 15 is a
duplicate of document 1 and cannot be detected until document 1 has posted.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docuflow.bootstrap import bootstrap
from docuflow.config import Settings
from docuflow.evaluation import (
    classification,
    data_quality,
    gatekeeping,
    integrity,
    ocr_accuracy,
)
from docuflow.evaluation.compare import compare_document
from docuflow.evaluation.cost import document_cost, summarize_cost
from docuflow.evaluation.effort import effort_saved
from docuflow.masterdata import MasterDataRepository
from docuflow.pipeline import Pipeline
from docuflow.schemas.registry import get_schema

logger = logging.getLogger(__name__)

MANIFEST = "ground_truth.json"
MASTER_DIR = "master"
RULES_FILE = "rules.json"

# The provider that reads a PDF's own text layer instead of recognising the
# page. Its transcript and the corpus's `reference_text` are the same string
# taken from the same place, so its character error rate is zero by
# construction and says nothing about how well anything reads. Pooled in with
# the rest it drags corpus CER towards zero and the report claims an OCR
# accuracy the OCR never earned - so scores from it are reported apart.
TEXT_LAYER_PROVIDER = "pdf-text-layer"

# The master files, in the order they have to load: a purchase order names a
# vendor, and a row pointing at a vendor who is not loaded yet is refused.
MASTER_KINDS = ("vendors", "purchase_orders", "contracts")


@dataclass(frozen=True)
class CorpusDocument:
    """One file in the corpus, with everything known about it beforehand."""

    order: int
    file: str
    doc_type: str
    posts: bool
    ground_truth: dict[str, Any]
    expected_findings: tuple[str, ...]
    reference_text: str
    arrival: str
    language: str

    @property
    def name(self) -> str:
        return self.file


def load_manifest(corpus_dir: Path) -> tuple[CorpusDocument, ...]:
    """The corpus as it declares itself, in run order.

    Raises rather than returning an empty list: an evaluation that silently
    scored nothing would print a perfect report.
    """
    path = corpus_dir / MANIFEST
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist - run `docuflow real-samples --out {corpus_dir}` first"
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("documents") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} holds no documents")

    documents = [
        CorpusDocument(
            order=int(entry["order"]),
            file=str(entry["file"]),
            doc_type=str(entry["doc_type"]),
            posts=bool(entry["posts"]),
            ground_truth=dict(entry.get("ground_truth") or {}),
            expected_findings=tuple(entry.get("expected_findings") or ()),
            reference_text=str(entry.get("reference_text") or ""),
            arrival=str(entry.get("arrival") or "pdf"),
            language=str(entry.get("language") or "en"),
        )
        for entry in entries
    ]
    return tuple(sorted(documents, key=lambda document: document.order))


def _isolated_settings(settings: Settings, label: str = "corpus") -> Settings:
    """A throwaway data directory, unique per call.

    Evaluation must not post into the demo ledger, and must start with an empty
    records table or the duplicate rule fires on the evaluation's own output.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-")[:40] or "corpus"
    directory = settings.data_dir.parent / "eval" / "out" / f"{slug}-{uuid.uuid4().hex[:8]}"
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return settings.model_copy(update={"data_dir": directory})


def load_master(settings: Settings, corpus_dir: Path) -> dict[str, int]:
    """Load the corpus's own vendor, purchase-order and contract exports.

    Most of these documents mean nothing without it: an invoice quoting
    `PO-2026-0501` is unverifiable, not correct, when no such order exists.
    """
    from docuflow.masterdata.importer import import_csv

    directory = corpus_dir / MASTER_DIR
    repository = MasterDataRepository(settings.db_path)
    loaded: dict[str, int] = {}
    for kind in MASTER_KINDS:
        path = directory / f"{kind}.csv"
        if not path.exists():
            logger.warning("no %s in %s; documents needing it cannot reconcile", path.name, directory)
            continue
        report = import_csv(repository, kind, path.read_bytes())
        loaded[kind] = report.imported
        for error in report.errors:
            logger.warning("%s row %s: %s", path.name, error.row, error.message)
    return loaded


def load_rules(settings: Settings, corpus_dir: Path) -> list[str]:
    """Load the rules written for this corpus, if it ships any.

    Skipping this is the trap the demo set documents: the documents those rules
    exist to catch post cleanly, and it reads as the system failing rather than
    as the system not having been told.
    """
    from docuflow.rules.custom import CustomRuleError, build_rule
    from docuflow.store.custom_rules import CustomRuleStore

    path = corpus_dir / RULES_FILE
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("rules") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a list of rules or an object with a 'rules' list")

    store = CustomRuleStore(settings.db_path)
    loaded: list[str] = []
    for entry in entries:
        try:
            rule = build_rule(entry)
        except CustomRuleError as exc:
            raise ValueError(f"{path}: {entry.get('rule_id', '?')}: {exc}") from exc
        store.save(rule, replace=True)
        loaded.append(rule.rule_id)
    return loaded


def expected_but_disabled(settings: Settings, documents: tuple[CorpusDocument, ...]) -> list[str]:
    """Rules the corpus says should fire that this machine has switched off.

    `Settings` fills anything it was not given from a `.env`, so an evaluation
    inherits whichever checks the machine that ran it happens to have disabled.
    That is not a detail: this checkout has seven off, and four of them are
    rules corpus documents declare. Left alone, the run reports those documents
    as **missed defects** - the metric this system exists to drive to zero -
    when nothing was missed and nothing was even asked.

    A number that moves with the operator's `.env` is not a measurement.
    """
    declared = {rule for document in documents for rule in document.expected_findings}
    return sorted(declared & settings.skipped_rules)


def enable_rules(settings: Settings, rule_ids: list[str]) -> Settings:
    """A settings copy with those rules switched back on."""
    if not rule_ids:
        return settings
    remaining = sorted(settings.skipped_rules - set(rule_ids))
    return settings.model_copy(update={"disabled_rules": ",".join(remaining)})


def known_rule_ids(settings: Settings) -> list[str]:
    """Every rule that could have fired: built-in plus whatever was loaded."""
    from docuflow.rules import all_rules
    from docuflow.store.custom_rules import CustomRuleStore

    builtin = [info.rule_id for info in all_rules()]
    custom = [rule.rule_id for rule in CustomRuleStore(settings.db_path).list_rules()]
    return sorted(set(builtin) | set(custom))


def _transcript(run: Any) -> str:
    """Everything the OCR returned for this document, as one string."""
    return "\n".join(page.text for page in run.ocr.pages)


def run_corpus(
    settings: Settings,
    corpus_dir: Path,
    limit: int = 0,
    label: str = "corpus",
    respect_disabled: bool = False,
) -> dict[str, Any]:
    """Process the corpus and return the four-part report.

    `respect_disabled` runs with the machine's `.env` exactly as it stands,
    which measures this installation rather than the system. The default
    switches back on the rules the corpus declares - see `expected_but_disabled`
    - and records in `setup` which ones it had to.
    """
    documents = load_manifest(corpus_dir)
    if limit:
        documents = documents[:limit]

    # Read the answer sheet before grading against it. On this project the
    # corpus has been wrong more often than the pipeline.
    corpus_check = integrity.check(documents)
    if corpus_check.get("undrawn_values"):
        logger.warning(
            "%d ground-truth value(s) are not printed on their page; extraction "
            "is graded against something no reader could find: %s",
            corpus_check["undrawn_values"],
            ", ".join(f"{k} x{v}" for k, v in corpus_check["undrawn_by_field"].items()),
        )

    conflicting = expected_but_disabled(settings, documents)
    re_enabled: list[str] = []
    if conflicting and not respect_disabled:
        settings = enable_rules(settings, conflicting)
        re_enabled = conflicting
        logger.warning(
            "re-enabled %d rule(s) this corpus expects but .env had disabled: %s",
            len(conflicting),
            ", ".join(conflicting),
        )
    elif conflicting:
        logger.warning(
            "%d rule(s) this corpus expects are disabled and will be reported as "
            "missed defects: %s",
            len(conflicting),
            ", ".join(conflicting),
        )

    eval_settings = _isolated_settings(settings, label)
    bootstrap(eval_settings, with_samples=False)
    master_counts = load_master(eval_settings, corpus_dir)
    rule_ids = load_rules(eval_settings, corpus_dir)
    pipeline = Pipeline(eval_settings)

    results: list[dict[str, Any]] = []
    ocr_scores: list[ocr_accuracy.OcrScore] = []
    # The subset an OCR engine actually recognised - see TEXT_LAYER_PROVIDER.
    recognised_scores: list[ocr_accuracy.OcrScore] = []
    outcomes: list[gatekeeping.Outcome] = []
    # `(expected, predicted)` document type, for per-class precision and recall.
    type_pairs: list[tuple[str, str]] = []
    quality_scores: list[dict[str, Any]] = []
    started = time.perf_counter()

    for document in documents:
        path = corpus_dir / document.file
        if not path.exists():
            logger.warning("%s is in the manifest but not on disk; skipping", document.file)
            continue

        logger.info("evaluating %s", document.file)
        began = time.perf_counter()
        outcome = pipeline.process_file(path, source_kind="sample")
        run = outcome.run
        elapsed = round((time.perf_counter() - began) * 1000)

        schema = get_schema(document.doc_type)
        specs = {spec.name: spec for spec in schema.fields}
        critical = {spec.name for spec in schema.critical_fields}

        comparisons = compare_document(document.ground_truth, run.field_map(), specs)
        wrong = [item for item in comparisons if item.status in {"wrong", "near", "spurious"}]
        wrong_critical = [item for item in wrong if item.field in critical]
        posted = run.decision == "auto_post"

        score = ocr_accuracy.score_document(
            reference=document.reference_text,
            transcript=_transcript(run),
            values=document.ground_truth,
            document=document.file,
        )
        ocr_scores.append(score)
        if run.ocr.provider != TEXT_LAYER_PROVIDER:
            recognised_scores.append(score)
        fired = tuple(sorted({finding.rule_id for finding in run.failed_findings}))
        type_pairs.append((document.doc_type, run.doc_type))
        quality_scores.append(
            data_quality.score_document(
                document=document.file,
                doc_type=run.doc_type,
                fields=run.field_map(),
                grounded={field.name: field.grounded for field in run.fields},
                findings=fired,
            )
        )
        outcomes.append(
            gatekeeping.Outcome(
                document=document.file,
                should_post=document.posts,
                posted=posted,
                confidence=run.confidence,
                expected_findings=document.expected_findings,
                failed_findings=fired,
            )
        )

        results.append(
            {
                "order": document.order,
                "document": document.file,
                "arrival": document.arrival,
                "language": document.language,
                "doc_type": document.doc_type,
                "classified_as": run.doc_type,
                "type_correct": run.doc_type == document.doc_type,
                "should_post": document.posts,
                "decision": run.decision,
                "posted": posted,
                "confidence": round(run.confidence, 4),
                "ocr_provider": run.ocr.provider,
                "ocr_ms": run.ocr.duration_ms,
                "elapsed_ms": elapsed,
                "cost": document_cost(run),
                "fields_total": len(comparisons),
                "fields_correct": sum(1 for item in comparisons if item.correct),
                "fields_near": sum(1 for item in comparisons if item.status == "near"),
                "fields_wrong": len(wrong),
                "fields_missing": sum(1 for item in comparisons if item.status == "missing"),
                "wrong_critical": [item.field for item in wrong_critical],
                # Posted without a person, carrying a wrong critical value.
                "silent_error": posted and bool(wrong_critical),
                "expected_findings": list(document.expected_findings),
                "findings_fired": list(fired),
                "comparisons": [
                    {
                        "field": item.field,
                        "expected": item.expected,
                        "actual": item.actual,
                        "status": item.status,
                        "similarity": item.similarity,
                    }
                    for item in comparisons
                ],
            }
        )

    wall_clock = round(time.perf_counter() - started, 1)
    rules = gatekeeping.score_rules(outcomes, known_rule_ids(eval_settings))
    return _summarize(
        results,
        ocr_scores,
        recognised_scores,
        outcomes,
        rules,
        type_pairs,
        quality_scores,
        corpus_check,
        wall_clock,
        setup={
            "corpus": str(corpus_dir),
            "master_loaded": master_counts,
            "rules_loaded": rule_ids,
            "auto_post_threshold": settings.auto_post_min_confidence,
            # The rules switched off on the machine that produced the numbers.
            # A report that does not say which checks were not running is a
            # report about a system nobody else has.
            "disabled_rules": sorted(settings.skipped_rules),
            # Switched back on for this run because the corpus declares them.
            # Empty is the healthy answer; anything here means the operator's
            # `.env` and the corpus disagreed about what is being tested.
            "rules_re_enabled": re_enabled,
            "ocr_provider": settings.ocr_provider,
            "llm_provider": settings.llm_provider,
            "llm_verify_provider": settings.llm_verify_provider,
        },
    )


def _summarize(
    results: list[dict[str, Any]],
    ocr_scores: list[ocr_accuracy.OcrScore],
    recognised_scores: list[ocr_accuracy.OcrScore],
    outcomes: list[gatekeeping.Outcome],
    rules: list[gatekeeping.RuleScore],
    type_pairs: list[tuple[str, str]],
    quality_scores: list[dict[str, Any]],
    corpus_check: dict[str, Any],
    wall_clock: float,
    setup: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate the run into the report `docs/EVALUATION.md` quotes."""
    if not results:
        return {"setup": setup, "summary": {"documents": 0}, "documents": []}

    total_fields = sum(item["fields_total"] for item in results)
    correct_fields = sum(item["fields_correct"] for item in results)
    near_fields = sum(item["fields_near"] for item in results)

    auto_posted = [item for item in results if item["posted"]]
    silent = [item for item in results if item["silent_error"]]

    expected_total = sum(len(item["expected_findings"]) for item in results)
    caught_total = sum(
        len(set(item["expected_findings"]) & set(item["findings_fired"])) for item in results
    )

    breakdown: dict[str, dict[str, Any]] = {}
    for item in results:
        for comparison in item["comparisons"]:
            entry = breakdown.setdefault(
                comparison["field"],
                {"total": 0, "correct": 0, "near": 0, "wrong": 0, "missing": 0},
            )
            entry["total"] += 1
            status = comparison["status"]
            entry[status if status in entry else "wrong"] += 1
    for entry in breakdown.values():
        entry["accuracy"] = round(entry["correct"] / entry["total"], 3) if entry["total"] else 0.0

    return {
        "setup": setup,
        # What the answer sheet can be shown to be wrong about, printed before
        # the numbers it produced.
        "corpus_integrity": corpus_check,
        "summary": {
            "documents": len(results),
            "wall_clock_seconds": wall_clock,
            "type_accuracy": round(
                sum(1 for item in results if item["type_correct"]) / len(results), 3
            ),
            "field_accuracy": round(correct_fields / total_fields, 3) if total_fields else 0.0,
            "field_accuracy_with_near": (
                round((correct_fields + near_fields) / total_fields, 3) if total_fields else 0.0
            ),
            "fields_total": total_fields,
            "fields_correct": correct_fields,
            "fields_near": near_fields,
            "fields_missing": sum(item["fields_missing"] for item in results),
            "auto_posted": len(auto_posted),
            "auto_post_rate": round(len(auto_posted) / len(results), 3),
            # The number that matters: wrong values that reached the ledger with
            # nobody in the way.
            "silent_errors": len(silent),
            "silent_error_rate": round(len(silent) / len(results), 3),
            "silent_error_documents": [item["document"] for item in silent],
            "auto_post_precision": (
                round(1 - len(silent) / len(auto_posted), 3) if auto_posted else 1.0
            ),
            "planted_defects": expected_total,
            "planted_defects_caught": caught_total,
            "defect_recall": round(caught_total / expected_total, 3) if expected_total else 1.0,
            "average_confidence": round(
                sum(item["confidence"] for item in results) / len(results), 3
            ),
        },
        # Everything read, including PDFs whose text layer was lifted straight
        # off the page. Kept because value recall is still meaningful there.
        "ocr": ocr_accuracy.summarize(ocr_scores),
        # Only the pages an engine had to recognise. This is the figure to quote
        # as OCR accuracy; the one above is diluted by transcripts that were
        # copied rather than read.
        "ocr_recognised": ocr_accuracy.summarize(recognised_scores),
        "ocr_from_text_layer": len(ocr_scores) - len(recognised_scores),
        # Per-class precision and recall, so a classifier that answers
        # "invoice" every time cannot report 80% and call it working.
        "classification": classification.summarize_labels(type_pairs),
        # Extraction as precision and recall. Accuracy over ground-truth fields
        # alone cannot see a value that was invented for a field the document
        # does not have; `spurious` is exactly that case.
        "extraction": classification.summarize_fields(results),
        # Is the record fit to hand to another system - answerable without any
        # answer sheet, which is what makes it usable in production.
        "data_quality": data_quality.summarize(quality_scores),
        "gatekeeping": gatekeeping.summarize(outcomes, rules),
        "effort": effort_saved(
            documents=len(results),
            auto_posted=len(auto_posted),
            silent_errors=len(silent),
            machine_seconds=wall_clock,
        ).to_json(),
        "cost": summarize_cost([item["cost"] for item in results if item.get("cost")]),
        "field_breakdown": dict(sorted(breakdown.items(), key=lambda kv: kv[1]["accuracy"])),
        "documents": results,
    }
