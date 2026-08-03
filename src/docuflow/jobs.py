"""A processing queue for batches of documents.

Uploading twenty files should not mean twenty requests held open for ten minutes
each. Files are accepted immediately, written to disk, and worked through by a
small pool while the browser polls for progress.

How small the pool is matters more than it looks. A local OCR model is a single
resident model on one machine: running two documents at once makes Ollama swap
models between them, and the pair finishes slower than they would one after the
other - measured earlier at 168s cold against ~10s warm for the same page. So
local engines get one worker, and only a cloud engine, where the concurrency
lives on someone else's hardware, gets several.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from docuflow.config import Settings

logger = logging.getLogger(__name__)

# Cloud OCR is rate-limited by the provider, not by this machine.
CLOUD_WORKERS = 4
LOCAL_WORKERS = 1
# Batches are capped so one upload cannot fill the disk or the queue.
MAX_BATCH_FILES = 50

JobStatus = str  # queued | running | done | failed


def worker_count(settings: Settings) -> int:
    """How many documents to process at once for the configured engine."""
    engine = settings.resolved_ocr_provider()
    if engine in {"ollama", "local"}:
        return LOCAL_WORKERS
    if engine in {"pdftext", "tesseract"}:
        # No model to contend over; these are pure CPU and finish in milliseconds.
        return CLOUD_WORKERS
    return CLOUD_WORKERS


@dataclass(frozen=True)
class Job:
    """One document waiting for, or going through, the pipeline."""

    job_id: str
    filename: str
    status: JobStatus = "queued"
    queued_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    stage: str = ""
    detail: str = ""
    document_id: str | None = None
    decision: str | None = None
    confidence: float | None = None
    record_id: str | None = None
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "status": self.status,
            "stage": self.stage,
            "detail": self.detail,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "document_id": self.document_id,
            "decision": self.decision,
            "confidence": self.confidence,
            "record_id": self.record_id,
            "error": self.error,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobQueue:
    """In-process queue with a bounded worker pool.

    Deliberately not a distributed queue: this is one server running one demo,
    and a Redis dependency would buy nothing a thread pool does not already do.
    What it does guarantee is that the browser never waits on a batch, that
    progress is visible per file, and that a document failing takes only itself
    down.
    """

    def __init__(self, settings_factory: Callable[[], Settings], workers: int | None = None) -> None:
        self._settings_factory = settings_factory
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._workers = workers
        self._pool: ThreadPoolExecutor | None = None

    # --- lifecycle ---------------------------------------------------------
    def _executor(self) -> ThreadPoolExecutor:
        if self._pool is None:
            count = self._workers or worker_count(self._settings_factory())
            self._pool = ThreadPoolExecutor(max_workers=count, thread_name_prefix="docuflow-job")
            logger.info("document queue started with %s worker(s)", count)
        return self._pool

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    # --- submission --------------------------------------------------------
    def submit(self, path: Path, filename: str) -> Job:
        """Queue one already-saved file. Returns immediately."""
        job = Job(job_id=f"JOB-{uuid.uuid4().hex[:8].upper()}", filename=filename, queued_at=_now())
        with self._lock:
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
        self._executor().submit(self._run, job.job_id, path)
        return job

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            existing = self._jobs.get(job_id)
            if existing is not None:
                self._jobs[job_id] = replace(existing, **changes)

    def _run(self, job_id: str, path: Path) -> None:
        """Process one document. Never raises - a failure belongs to its job."""
        from docuflow.pipeline import Pipeline

        self._update(job_id, status="running", started_at=_now(), stage="render")
        try:
            settings = self._settings_factory()
            pipeline = Pipeline(settings)

            def progress(stage: str, message: str) -> None:
                self._update(job_id, stage=stage, detail=message)

            result = pipeline.process_file(path, source_kind="upload", on_progress=progress)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
            logger.exception("job %s failed", job_id)
            self._update(job_id, status="failed", finished_at=_now(), stage="", error=str(exc))
            return

        run = result.run
        self._update(
            job_id,
            status="done",
            finished_at=_now(),
            stage="",
            detail="",
            document_id=run.document_id,
            decision=run.decision,
            confidence=run.confidence,
            record_id=run.record_id,
            error=run.error or "",
        )

    # --- reads -------------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 100) -> list[Job]:
        """Most recently queued first."""
        with self._lock:
            ids = self._order[-limit:][::-1]
            return [self._jobs[job_id] for job_id in ids if job_id in self._jobs]

    def summary(self) -> dict[str, Any]:
        jobs = self.list_jobs(limit=10_000)
        counts: dict[str, int] = {}
        for job in jobs:
            counts[job.status] = counts.get(job.status, 0) + 1
        return {
            "workers": self._workers or worker_count(self._settings_factory()),
            "queued": counts.get("queued", 0),
            "running": counts.get("running", 0),
            "done": counts.get("done", 0),
            "failed": counts.get("failed", 0),
            "active": counts.get("queued", 0) + counts.get("running", 0),
        }

    def clear_finished(self) -> int:
        """Drop completed jobs from the list. Documents are untouched."""
        with self._lock:
            finished = [
                job_id for job_id in self._order
                if self._jobs.get(job_id) and self._jobs[job_id].status in {"done", "failed"}
            ]
            for job_id in finished:
                self._jobs.pop(job_id, None)
                self._order.remove(job_id)
            return len(finished)
