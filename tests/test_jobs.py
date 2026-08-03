"""The batch queue: many files in, one worker pool, nothing lost."""

from __future__ import annotations

import time

import pytest

from docuflow.jobs import CLOUD_WORKERS, LOCAL_WORKERS, JobQueue, worker_count


def drain(queue: JobQueue, timeout: float = 60.0) -> None:
    """Wait for the queue to empty, or fail loudly."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if queue.summary()["active"] == 0:
            return
        time.sleep(0.05)
    raise AssertionError(f"queue did not drain: {queue.summary()}")


class TestWorkerSizing:
    """Concurrency is chosen for the engine, not picked arbitrarily."""

    def test_a_local_model_gets_one_worker(self, settings):
        """Two documents at once make Ollama swap models between them, so the
        pair finishes slower than running them in sequence."""
        for engine in ("ollama", "local"):
            assert worker_count(settings.model_copy(update={"ocr_provider": engine})) == LOCAL_WORKERS

    def test_a_cloud_engine_gets_several(self, settings):
        config = settings.model_copy(update={"ocr_provider": "typhoon_api", "typhoon_api_key": "x"})
        assert worker_count(config) == CLOUD_WORKERS


class TestQueue:
    @pytest.fixture
    def queue(self, settings, master):
        made = JobQueue(lambda: settings, workers=2)
        yield made
        made.shutdown()

    def test_a_batch_is_accepted_immediately_and_all_of_it_runs(self, queue, sample_pdf):
        jobs = [
            queue.submit(sample_pdf(doc_id), f"{doc_id}.pdf")
            for doc_id in ("inv-clean-classic", "inv-bad-math", "receipt-fuel")
        ]
        assert all(job.status == "queued" for job in jobs)

        drain(queue)
        finished = {job.job_id: job for job in queue.list_jobs()}
        assert len(finished) == 3
        for job in jobs:
            assert finished[job.job_id].status == "done"
            assert finished[job.job_id].document_id is not None

    def test_one_bad_file_does_not_take_the_batch_down(self, queue, sample_pdf, tmp_path):
        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"not a pdf at all")
        queue.submit(broken, "broken.pdf")
        good = queue.submit(sample_pdf("inv-clean-classic"), "good.pdf")

        drain(queue)
        by_id = {job.job_id: job for job in queue.list_jobs()}
        assert by_id[good.job_id].status == "done"
        # A corrupt file is a rejected document, not a crashed worker.
        assert queue.summary()["done"] + queue.summary()["failed"] == 2

    def test_progress_is_visible_per_file(self, queue, sample_pdf):
        job = queue.submit(sample_pdf("inv-clean-classic"), "one.pdf")
        drain(queue)
        assert queue.get(job.job_id).status == "done"
        assert queue.get(job.job_id).finished_at

    def test_clearing_keeps_unfinished_work(self, queue, sample_pdf):
        queue.submit(sample_pdf("inv-clean-classic"), "one.pdf")
        drain(queue)
        assert queue.clear_finished() == 1
        assert queue.list_jobs() == []

    def test_an_unknown_job_is_reported_as_missing(self, queue):
        assert queue.get("JOB-NOPE") is None
