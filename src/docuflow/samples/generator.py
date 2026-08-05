"""Render the demo corpus to files plus a ground-truth manifest."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from reportlab.lib.pagesizes import A4

from docuflow.samples.catalog import SAMPLE_DOCUMENTS, SampleDoc, master_data_rows
from docuflow.samples.degrade import save_as_scan
from docuflow.samples.layouts import LAYOUT_RENDERERS
from docuflow.samples.sheet import Sheet

logger = logging.getLogger(__name__)

# 80mm thermal paper, tall enough for a fuel receipt.
RECEIPT_PAGE = (226.0, 420.0)
# 72dpi * 2.5 = 180dpi: what a phone photo or an office scanner produces, and
# enough for OCR without making the corpus enormous.
IMAGE_SCALE = 2.5
THAI_LAYOUTS = frozenset({"thai_tax", "thai_modern", "thai_receipt", "thai_contract"})
# Layouts drawn on the thermal roll rather than A4.
NARROW_LAYOUTS = frozenset({"narrow", "thai_receipt"})


def _page_size(doc: SampleDoc) -> tuple[float, float]:
    return RECEIPT_PAGE if doc.layout in NARROW_LAYOUTS else A4


def _needs_thai_font(doc: SampleDoc) -> bool:
    if doc.language == "th" or doc.layout in THAI_LAYOUTS:
        return True
    # Thai text can also appear in an otherwise-English document.
    blob = json.dumps(doc.ground_truth, ensure_ascii=False) + json.dumps(list(doc.line_items), ensure_ascii=False)
    return any("฀" <= char <= "๿" for char in blob)


def render_pdf(doc: SampleDoc, out_path: Path) -> Path:
    """Draw one sample document to a PDF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    margin = 14.0 if doc.layout in NARROW_LAYOUTS else 48.0
    sheet = Sheet.create(
        str(out_path),
        page_size=_page_size(doc),
        margin=margin,
        thai=_needs_thai_font(doc),
    )
    renderer = LAYOUT_RENDERERS.get(doc.layout)
    if renderer is None:
        raise ValueError(f"No renderer for layout '{doc.layout}'")
    renderer(sheet, doc)
    sheet.save()
    return out_path


def _pdf_to_scan(pdf_path: Path, out_path: Path, intensity: float, seed: int) -> Path:
    """Rasterise page 1 and apply scan damage."""
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(pdf_path))
    try:
        image = document[0].render(scale=2.0).to_pil()
    finally:
        document.close()
    save_as_scan(image, out_path, intensity, seed=seed)
    return out_path


def _pdf_to_image(pdf_path: Path, out_path: Path, scale: float = IMAGE_SCALE) -> Path:
    """Rasterise page 1 at print resolution, undamaged."""
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(pdf_path))
    try:
        document[0].render(scale=scale).to_pil().save(out_path, format="PNG")
    finally:
        document.close()
    return out_path


def generate_corpus(out_dir: Path, include_clean_pdfs: bool = True) -> list[dict]:
    """Render every sample and return the ground-truth manifest.

    Each document is written twice: as the PDF it was drawn as, and as a PNG of
    its first page. Photographs and scans are how most documents actually
    arrive, and the two paths through the system are not the same one - a PDF
    can carry a text layer, an image never can, so the image is the only version
    that puts the OCR model under real load.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for index, doc in enumerate(SAMPLE_DOCUMENTS):
        pdf_path = out_dir / f"{doc.doc_id}.pdf"
        render_pdf(doc, pdf_path)

        if doc.degrade > 0:
            artifact = out_dir / f"{doc.doc_id}.jpg"
            _pdf_to_scan(pdf_path, artifact, doc.degrade, seed=index * 7 + 3)
            if not include_clean_pdfs:
                pdf_path.unlink(missing_ok=True)
        else:
            artifact = pdf_path

        image_path = _pdf_to_image(pdf_path, out_dir / f"{doc.doc_id}.png") if pdf_path.exists() else None

        entry = asdict(doc)
        entry["file"] = artifact.name
        entry["path"] = str(artifact)
        if image_path is not None:
            entry["image"] = image_path.name
        manifest.append(entry)
        logger.info("rendered %s -> %s", doc.doc_id, artifact.name)

    (out_dir / "ground_truth.json").write_text(
        json.dumps({"documents": manifest, "master_data": master_data_rows()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_manifest(out_dir: Path) -> list[dict]:
    """Read a previously generated manifest."""
    path = out_dir / "ground_truth.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("documents", [])
