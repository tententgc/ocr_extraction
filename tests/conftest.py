"""Shared fixtures. Tests never touch the demo database or call a model."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from docuflow.config import Settings
from docuflow.db import init_db
from docuflow.masterdata.repository import MasterDataRepository
from docuflow.models import DocumentRun, ExtractedField, OcrPage, OcrResult, SourceInfo
from docuflow.samples.catalog import master_data_rows
from docuflow.schemas.registry import get_schema


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Isolated settings backed by a temporary data directory.

    `Settings` reads the developer's `.env`, so anything the suite asserts on
    has to be pinned here or the result depends on how one machine happens to be
    configured. That covers credentials, and equally the rule-execution switches:
    a developer exploring with RULES_FREE_MODE=true would otherwise see every
    validation test "pass" while nothing was being validated at all.
    """
    config = Settings(
        data_dir=tmp_path,
        ocr_provider="pdftext",
        llm_provider="rules",
        typhoon_api_key="",
        openai_api_key="",
        anthropic_api_key="",
        auto_post_min_confidence=0.90,
        amount_tolerance=0.02,
        rules_free_mode=False,
        disabled_rules="",
    )
    config.ensure_directories()
    init_db(config.db_path)
    return config


@pytest.fixture
def master(settings: Settings) -> MasterDataRepository:
    repository = MasterDataRepository(settings.db_path)
    repository.seed(master_data_rows())
    return repository


@pytest.fixture
def ocr_pages() -> list[tuple[int, str]]:
    """A small, realistic invoice page as recognised text."""
    return [
        (
            1,
            "Siam Tech Supply Co., Ltd.\n"
            "199/22 Ratchadaphisek Road, Din Daeng, Bangkok 10400\n"
            "Tax ID: 0105543000129\n"
            "TAX INVOICE\n"
            "Invoice No.: INV-2026-0731\n"
            "Invoice Date: 8 July 2026\n"
            "Due Date: 7 August 2026\n"
            "PO Reference: PO-2026-0412\n"
            "Bill To\n"
            "Mirtphol Industries PCL\n"
            "Subtotal 187,000.00 THB\n"
            "VAT 7% 13,090.00 THB\n"
            "Amount Due 200,090.00 THB\n"
            "Payment Terms: Net 30\n"
            "Remit to Account No: 1234567890\n",
        )
    ]


def build_run(
    fields: dict[str, object],
    doc_type: str = "invoice",
    text: str = "",
    grounded: bool = True,
    confidence: float = 0.95,
) -> DocumentRun:
    """Construct a DocumentRun with the given field values."""
    schema = get_schema(doc_type)
    specs = {spec.name: spec for spec in schema.fields}
    extracted = tuple(
        ExtractedField(
            name=name,
            label=specs[name].label if name in specs else name,
            value=value,
            evidence=str(value),
            confidence=confidence if value is not None else 0.0,
            grounded=grounded,
            source="llm",
        )
        for name, value in fields.items()
    )
    return DocumentRun(
        document_id="DOC-TEST",
        source=SourceInfo(filename="test.pdf", sha256="0" * 64, byte_size=1),
        doc_type=doc_type,
        fields=extracted,
        ocr=OcrResult(
            pages=(OcrPage(page_number=1, text=text or "sample text " * 20, provider="test"),),
            provider="test",
        ),
    )


@pytest.fixture
def make_run():
    return build_run


@pytest.fixture
def today() -> date:
    return date(2026, 8, 3)


@pytest.fixture
def sample_pdf(settings, tmp_path):
    """Render one document from the demo corpus to a PDF."""
    from docuflow.samples.catalog import sample_by_id
    from docuflow.samples.generator import render_pdf

    def _render(doc_id: str):
        doc = sample_by_id(doc_id)
        assert doc is not None, f"unknown sample {doc_id}"
        return render_pdf(doc, tmp_path / f"{doc_id}.pdf")

    return _render
