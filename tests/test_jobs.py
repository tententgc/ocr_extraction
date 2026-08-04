"""The batch queue: many files in, one worker pool, nothing lost."""

from __future__ import annotations

import time

import pytest

from docuflow.jobs import CLOUD_WORKERS, LOCAL_WORKERS, Job, JobQueue, worker_count


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

    def test_switching_engine_resizes_the_pool_between_batches(self, settings, master, sample_pdf):
        """The engine is picked from the screen, long after the pool was sized."""
        chosen = {"ocr_provider": "ollama"}
        queue = JobQueue(lambda: settings.model_copy(update=chosen))
        try:
            queue._executor()
            assert queue._pool_size == LOCAL_WORKERS

            chosen["ocr_provider"] = "pdftext"      # no model to contend over
            queue.submit(sample_pdf("inv-clean-classic"), "one.pdf")
            assert queue._pool_size == CLOUD_WORKERS
            drain(queue)
        finally:
            queue.shutdown()

    def test_a_running_batch_keeps_the_pool_it_started_with(self, settings):
        """Replacing a pool with work in it would strand the work."""
        queue = JobQueue(lambda: settings.model_copy(update={"ocr_provider": "pdftext"}))
        try:
            queue._executor()
            queue._jobs["JOB-BUSY"] = Job(job_id="JOB-BUSY", filename="x", status="running")
            queue._order.append("JOB-BUSY")
            before = queue._pool
            queue._resize_if_idle()
            assert queue._pool is before
        finally:
            queue.shutdown()


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

    def test_a_staged_file_keeps_the_name_it_was_given(self, queue, sample_pdf, tmp_path, settings):
        """An upload is written under a unique name; the document is not called that."""
        from docuflow.store.documents import DocumentStore

        staged = tmp_path / "a1b2c3d4-Invoice July.pdf"
        staged.write_bytes(sample_pdf("inv-clean-classic").read_bytes())

        job = queue.submit(staged, "Invoice July.pdf", source_kind="upload")
        drain(queue)

        run = DocumentStore(settings.db_path).get(queue.get(job.job_id).document_id)
        # Blob storage sanitises spaces out; the staging prefix is what must go.
        assert run.source.filename == "Invoice_July.pdf"
        assert "a1b2c3d4" not in run.source.filename
        assert run.source.kind == "upload"

    def test_where_a_document_came_from_travels_with_the_job(self, queue, sample_pdf, settings):
        """A sample queued rather than run inline is still a sample, not an upload."""
        from docuflow.store.documents import DocumentStore

        job = queue.submit(sample_pdf("inv-clean-classic"), "inv-clean-classic.pdf",
                           source_kind="sample", origin="https://example.test/inv.pdf")
        drain(queue)

        run = DocumentStore(settings.db_path).get(queue.get(job.job_id).document_id)
        assert run.source.kind == "sample"
        assert run.source.origin == "https://example.test/inv.pdf"


class TestNothingIsRefusedForBeingSecond:
    """The bug this replaced: a single upload ran inline behind a global lock.

    Dropping one file was pleasant - the result appeared by itself. Dropping a
    second while the first was still reading got "Still processing the previous
    document" and the file was discarded. Ten minutes of OCR is a long time to be
    unable to hand over the next document.
    """

    def test_files_dropped_one_after_another_all_arrive(self, client, sample_pdf):
        queued = []
        for name in ("inv-clean-classic", "inv-bad-math", "receipt-fuel"):
            response = client.post(
                "/api/documents/batch",
                files={"files": (f"{name}.pdf", sample_pdf(name).read_bytes(), "application/pdf")},
            )
            assert response.status_code == 200, response.text
            queued.extend(response.json()["queued"])

        assert len(queued) == 3
        assert len({job["job_id"] for job in queued}) == 3, "each one is its own job"

    def test_a_sample_can_be_queued_instead_of_waited_for(self, client):
        payload = client.post(
            "/api/documents/sample/inv-clean-classic?queue=true", json={}
        ).json()
        assert len(payload["queued"]) == 1
        assert payload["queued"][0]["status"] in {"queued", "running", "done"}

    def test_the_synchronous_form_still_answers_with_the_result(self, client):
        """Queueing is what the browser wants; an API caller may still just ask."""
        payload = client.post("/api/documents/sample/inv-clean-classic").json()
        assert "document" in payload and "queued" not in payload

    def test_a_bad_url_is_still_reported_to_the_caller_not_to_a_job(self, client):
        """A typo belongs in the reply to the request that made it."""
        response = client.post(
            "/api/documents/url?queue=true", json={"url": "http://127.0.0.1/nope.pdf"}
        )
        assert response.status_code == 400
