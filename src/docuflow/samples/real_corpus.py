"""Write the working corpus to disk: master CSVs, documents, ground truth.

Each document is rendered once, in the single form it is meant to arrive in.
That is the difference from the demo corpus, which writes every document as both
a PDF and a PNG so the two paths can be compared. Here the arrival *is* the test:
document 3 is a photograph and only a photograph, because what it exists to
prove is that a photograph gets read correctly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from docuflow.samples.degrade import photocopy, photograph
from docuflow.samples.generator import render_pdf
from docuflow.samples.real_world import ALL_DOCUMENTS, CSV_COLUMNS, master_rows, to_csv

logger = logging.getLogger(__name__)

# 72dpi * 2.5 = 180dpi, which is what an office scanner produces.
PAGE_SCALE = 2.5
# A phone photographs at higher resolution than it needs to, and downsampling
# afterwards would hide exactly the softness the file is here to carry.
PHOTO_SCALE = 3.0

MASTER_DIR = "master"


def _render_page(pdf_path: Path, scale: float):
    """Rasterise page one of a rendered PDF."""
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(pdf_path))
    try:
        return document[0].render(scale=scale).to_pil()
    finally:
        document.close()


def page_text(pdf_path: Path) -> str:
    """Exactly what is printed on the page, from the PDF's own text layer.

    This is the reference OCR is scored against, and it is trustworthy for a
    reason worth stating: the corpus is *drawn* here, so the text layer is not
    an approximation of the page - it is the string the renderer was given. A
    photograph of that page has no text layer at all, which is precisely why
    the reference has to be taken before the page is photographed.
    """
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(pdf_path))
    try:
        page = document[0]
        text_page = page.get_textpage()
        try:
            return (text_page.get_text_range() or "").strip()
        finally:
            text_page.close()
    finally:
        document.close()


def write_master_csvs(out_dir: Path) -> dict[str, Path]:
    """The vendor, purchase-order and contract files an ERP would export."""
    directory = out_dir / MASTER_DIR
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for kind in CSV_COLUMNS:
        path = directory / f"{kind}.csv"
        path.write_text(to_csv(kind), encoding="utf-8")
        written[kind] = path
    return written


def generate(out_dir: Path) -> list[dict[str, Any]]:
    """Render every document and return the ground-truth manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_master_csvs(out_dir)

    manifest: list[dict[str, Any]] = []
    for index, (slug, arrival, posts, doc) in enumerate(ALL_DOCUMENTS):
        # The expected outcome goes in the filename. A corpus that states what
        # should happen before the system is asked is one you can grade against;
        # one that does not ends up graded against itself.
        name = f"{index + 1}_{slug}_{posts}"

        # Rendered to a scratch PDF first in every case: the layouts draw
        # vector pages, and a photograph is a picture of a printed one.
        pdf_path = out_dir / f"{name}.pdf"
        render_pdf(doc, pdf_path)

        # Taken now, while there is still a text layer to take it from.
        reference = page_text(pdf_path)

        if arrival == "pdf":
            artifact = pdf_path
        elif arrival == "png":
            artifact = out_dir / f"{name}.png"
            _render_page(pdf_path, PAGE_SCALE).save(artifact, format="PNG")
            pdf_path.unlink()
        elif arrival == "photo":
            artifact = out_dir / f"{name}.jpg"
            photograph(_render_page(pdf_path, PHOTO_SCALE), seed=index * 13 + 5).save(
                artifact, format="JPEG", quality=72
            )
            pdf_path.unlink()
        elif arrival == "photocopy":
            artifact = out_dir / f"{name}.jpg"
            photocopy(_render_page(pdf_path, PAGE_SCALE), seed=index * 13 + 5).save(
                artifact, format="JPEG", quality=62
            )
            pdf_path.unlink()
        else:
            raise ValueError(f"Unknown arrival {arrival!r} for {name}")

        entry = asdict(doc)
        entry["file"] = artifact.name
        entry["arrival"] = arrival
        entry["order"] = index + 1
        entry["posts"] = posts
        entry["reference_text"] = reference
        manifest.append(entry)
        logger.info("wrote %s (%s)", artifact.name, arrival)

    (out_dir / "ground_truth.json").write_text(
        json.dumps(
            {
                "documents": manifest,
                "master_data": {kind: list(rows) for kind, rows in master_rows().items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(_readme(manifest), encoding="utf-8")
    return manifest


ARRIVAL_WORDS = {
    "pdf": "digital PDF",
    "png": "scanned page",
    "photo": "phone photograph",
    "photocopy": "faded photocopy",
}


def _readme(manifest: list[dict[str, Any]]) -> str:
    """A corpus you cannot read is a corpus nobody maintains."""
    lines = [
        "# Working corpus",
        "",
        "Documents to run the pipeline against, with the master data they",
        "reconcile to. Generated by `docuflow real-samples`; edit",
        "`src/docuflow/samples/real_world.py` rather than these files.",
        "",
        "`master/` holds the vendor, purchase-order and contract exports. Load",
        "them first — most of these documents mean nothing without them:",
        "",
        "```bash",
        "docuflow reset --confirm",
        "docuflow load-master real_samples/master",
        "```",
        "",
        "The filename ends in the expected outcome. `_True` means nothing is",
        "wrong with the document and holding it is a false alarm; `_False` means",
        "something is wrong and posting it is a silent error. The evaluation",
        "compares that against what actually happened.",
        "",
        "| # | File | Arrives as | Should post | What it is for | Expected to trip |",
        "|---|------|-----------|---|----------------|------------------|",
    ]
    for entry in manifest:
        findings = ", ".join(f"`{name}`" for name in entry.get("expected_findings", ())) or "—"
        lines.append(
            f"| {entry['order']} | `{entry['file']}` | {ARRIVAL_WORDS[entry['arrival']]} "
            f"| {'yes' if entry['posts'] else 'no'} | {entry['demonstrates']} | {findings} |"
        )
    lines.append("")
    lines.append(
        "The duplicate submission can only be detected once the original has "
        "posted, so run these in order."
    )
    lines.append("")
    return "\n".join(lines)
