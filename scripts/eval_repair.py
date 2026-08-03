"""Does the repair pass actually help? Same corpus, same settings, one variable.

Each arm gets its own database so duplicate detection does not see the other
arm's postings, and both arms read the same rendered documents with the same
engines. The only difference is REPAIR_DISABLED.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import time

from docuflow.bootstrap import bootstrap
from docuflow.config import Settings
from docuflow.pipeline import Pipeline
from docuflow.samples.catalog import sample_by_id
from docuflow.samples.generator import render_pdf

DOC_IDS = [
    "inv-clean-classic", "inv-bad-math", "inv-thai-scan", "inv-unknown-vendor",
    "inv-duplicate", "inv-eur-over-po", "contract-facility",
    "form-vendor-onboarding", "receipt-fuel", "po-orient-paper",
]

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "eval-repair.json")
corpus = pathlib.Path(tempfile.mkdtemp())

# Render once; both arms read the identical files.
documents = {doc_id: render_pdf(sample_by_id(doc_id), corpus / f"{doc_id}.pdf") for doc_id in DOC_IDS}
print(f"rendered {len(documents)} documents", flush=True)


def run_arm(name: str, repair_disabled: bool) -> list[dict]:
    data_dir = pathlib.Path(tempfile.mkdtemp())
    settings = Settings(
        data_dir=data_dir,
        rules_free_mode=False,      # the whole point is whether rules pass
        disabled_rules="",
        repair_disabled=repair_disabled,
    )
    settings.ensure_directories()
    bootstrap(settings, with_samples=False)
    pipeline = Pipeline(settings)

    rows = []
    for doc_id in DOC_IDS:
        started = time.perf_counter()
        run = pipeline.process_file(documents[doc_id]).run
        repair_step = next((s for s in run.trace if s.step == "repair"), None)
        rows.append({
            "doc_id": doc_id,
            "doc_type": run.doc_type,
            "decision": run.decision,
            "confidence": run.confidence,
            "blocking": sorted(f.rule_id for f in run.blocking_findings),
            "failing": sorted(f.rule_id for f in run.findings if not f.passed),
            "fields_filled": sum(1 for f in run.fields if not f.is_empty),
            "recovered": sorted(f.name for f in run.fields if f.source == "agent"),
            "repair_summary": repair_step.summary if repair_step else "",
            "repair_rejected": (repair_step.detail.get("rejected", []) if repair_step else []),
            "seconds": round(time.perf_counter() - started, 1),
        })
        print(f"  [{name}] {doc_id:24} {run.decision:13} {run.confidence:.0%} "
              f"blocking={len(run.blocking_findings)} "
              f"recovered={rows[-1]['recovered']}", flush=True)
    return rows


print("\n=== arm A: repair OFF ===", flush=True)
baseline = run_arm("off", repair_disabled=True)
print("\n=== arm B: repair ON ===", flush=True)
repaired = run_arm("on", repair_disabled=False)

OUT.write_text(json.dumps({"baseline": baseline, "repaired": repaired}, indent=2, ensure_ascii=False))
print(f"\nwritten to {OUT}", flush=True)
