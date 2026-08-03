"""Command line interface, exercised without models or network."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import pytest

from docuflow import cli
from docuflow.samples.catalog import sample_by_id
from docuflow.samples.generator import render_pdf


@pytest.fixture(autouse=True)
def isolated_cli(settings, monkeypatch):
    """Point every CLI command at the temporary settings."""
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    return settings


@pytest.fixture
def invoice_pdf(tmp_path) -> Path:
    return render_pdf(sample_by_id("inv-clean-classic"), tmp_path / "invoice.pdf")


def run(argv: list[str]) -> tuple[int, str]:
    """Invoke the CLI, capturing stdout."""
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(io.StringIO()):
        code = cli.main(argv)
    return code, out.getvalue()


class TestInit:
    def test_init_creates_the_database_and_master_data(self, settings):
        code, output = run(["init", "--no-samples"])
        assert code == 0
        payload = json.loads(output)
        assert payload["vendors"] == 5
        assert payload["purchase_orders"] == 4
        assert settings.db_path.exists()

    def test_init_is_repeatable(self):
        assert run(["init", "--no-samples"])[0] == 0
        assert run(["init", "--no-samples"])[0] == 0


class TestStatus:
    def test_status_lists_engines_and_thresholds(self):
        code, output = run(["status"])
        assert code == 0
        assert "OCR providers" in output
        assert "LLM providers" in output
        assert "Auto-post at" in output


class TestModels:
    def test_models_lists_both_kinds(self):
        code, output = run(["models"])
        assert code == 0
        assert "OCR ENGINES" in output
        assert "AGENT MODELS" in output
        # The no-model options are always available.
        assert "pdftext" in output
        assert "rules" in output


class TestProcess:
    def test_process_extracts_and_prints_a_verdict(self, invoice_pdf):
        code, output = run(["process", str(invoice_pdf), "--quiet"])
        assert code == 0
        assert "INV-2026-0731" in output or "REVIEW" in output or "POSTED" in output

    def test_process_json_emits_the_full_run(self, invoice_pdf):
        code, output = run(["process", str(invoice_pdf), "--json", "--quiet"])
        assert code == 0
        payload = json.loads(output)
        assert payload["doc_type"] == "invoice"
        assert payload["fields"]
        assert "trace" in payload

    def test_missing_file_is_reported(self, tmp_path):
        code, _ = run(["process", str(tmp_path / "nope.pdf"), "--quiet"])
        assert code == 1

    def test_a_private_url_is_refused(self):
        code, _ = run(["process", "http://127.0.0.1/secret.pdf", "--quiet"])
        assert code == 1


class TestRecords:
    def test_records_reports_an_empty_ledger(self):
        run(["init", "--no-samples"])
        code, output = run(["records"])
        assert code == 0
        assert "No records posted yet" in output

    def test_records_lists_a_posted_document(self, settings, make_run):
        from docuflow.store.documents import DocumentStore
        from docuflow.store.records import RecordStore

        run(["init", "--no-samples"])
        document = make_run(
            {
                "invoice_number": "INV-CLI-1",
                "vendor_name": "Siam Tech Supply Co., Ltd.",
                "total_amount": 1234.50,
                "currency": "THB",
                "invoice_date": "2026-07-08",
            }
        ).evolve(decision="auto_post")
        DocumentStore(settings.db_path).save(document)
        RecordStore(settings.db_path).post_record(document, "V-1001")

        code, output = run(["records"])
        assert code == 0
        assert "INV-CLI-1" in output
        assert "1,234.50" in output


class TestDemo:
    def test_demo_processes_the_corpus_and_summarises(self):
        code, output = run(["demo", "--limit", "2", "--quiet", "--reset"])
        assert code == 0
        assert "Straight-through" in output
        assert "DECISION" in output


class TestArgumentParsing:
    def test_no_command_is_an_error(self):
        with pytest.raises(SystemExit):
            cli.main([])

    def test_unknown_command_is_an_error(self):
        with pytest.raises(SystemExit):
            cli.main(["frobnicate"])

    @pytest.mark.parametrize(
        "command",
        [["init"], ["status"], ["process", "x"], ["demo"], ["records"], ["serve"], ["evaluate"], ["models"], ["benchmark"]],
    )
    def test_every_command_is_registered(self, command):
        parser_commands = {"init", "status", "process", "demo", "records", "serve", "evaluate", "models", "benchmark"}
        assert command[0] in parser_commands
