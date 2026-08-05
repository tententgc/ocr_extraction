"""Five documents for a live demonstration, and the register they reconcile against.

Different job from `real_world.py`. That corpus exists to *measure* the system:
thirty documents, several of them ambiguous on purpose, because a corpus that
only contains easy cases reports a number nobody should believe.

This set exists to be *watched*. Every outcome here is unarguable, and each
document fails - or passes - for one reason that can be pointed at in a sentence:

    1  clean            PDF    posts     nothing is wrong with it
    2  bank account     PNG    held      the page says an account the register does not
    3  VAT wrong        PDF    held      7% of the subtotal is not the tax charged
    4  Thai, correct    PNG    posts     Buddhist dates and Thai labels, all consistent
    5  over the PO      PDF    held      the invoice bills more than the order allows

Two of the five pass, three fail, and the three fail in three different *kinds*
of way: one is a fact about the world (the register), one is the document
contradicting itself (arithmetic), one needs a lookup and a sum before anyone
can tell (the agent rule). A demo where every failure is caught by the same
mechanism teaches that the system has one mechanism.

Both formats appear because they are not the same path. A PDF carries a text
layer; a PNG carries none, so the OCR model is the only thing standing between
the page and the extraction. Demonstrating on PDFs alone demonstrates the easy
half.

The register below is deliberately small - four vendors, three purchase orders -
so a viewer can hold all of it in their head while a document is checked against
it. It is written out as the same CSVs an ERP would export, and the same rows are
mirrored in `notebooks/build_live.py` so these five files behave identically
whether they go through `docuflow` or through the Colab notebook.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from docuflow.samples.catalog import SampleDoc
from docuflow.samples.generator import render_pdf

logger = logging.getLogger(__name__)

BUYER = "Mirtphol Industries PCL"
MASTER_DIR = "master"
# 72dpi × 2.5 = 180dpi, what an office scanner produces.
PAGE_SCALE = 2.5

CSV_COLUMNS: dict[str, tuple[str, ...]] = {
    "vendors": ("vendor_id", "legal_name", "aliases", "tax_id", "bank_account",
                "status", "payment_terms", "currency"),
    "purchase_orders": ("po_number", "vendor_id", "po_date", "currency", "amount",
                        "status", "description"),
    "contracts": ("contract_number", "vendor_id", "party_a", "party_b",
                  "effective_date", "expiry_date", "value", "currency"),
}

# --------------------------------------------------------------------------- #
# The register                                                                  #
# --------------------------------------------------------------------------- #

VENDORS: tuple[dict[str, Any], ...] = (
    {
        "vendor_id": "V-9001",
        "legal_name": "Siam Tech Supply Co., Ltd.",
        "aliases": "Siam Tech;Siam Tech Supply;บริษัท สยามเทค ซัพพลาย จำกัด",
        "tax_id": "0105543000129",
        "bank_account": "1234567890",
        "status": "active", "payment_terms": "Net 30", "currency": "THB",
    },
    {
        "vendor_id": "V-9002",
        "legal_name": "Rungruang Steel Works Co., Ltd.",
        "aliases": "Rungruang Steel;บริษัท รุ่งเรือง สตีล เวิร์ค จำกัด",
        "tax_id": "0105552000110",
        "bank_account": "3120045678",
        "status": "active", "payment_terms": "Net 60", "currency": "THB",
    },
    {
        "vendor_id": "V-9003",
        "legal_name": "บริษัท ไทยพัฒนา วิศวกรรม จำกัด",
        "aliases": "Thai Pattana Engineering;ไทยพัฒนา วิศวกรรม",
        "tax_id": "0105548000117",
        "bank_account": "4450091122",
        "status": "active", "payment_terms": "Net 30", "currency": "THB",
    },
    {
        "vendor_id": "V-9004",
        "legal_name": "Northwind Logistics Ltd.",
        "aliases": "Northwind;Northwind Logistics",
        "tax_id": "0105561071237",
        "bank_account": "9876543210",
        "status": "active", "payment_terms": "Net 45", "currency": "THB",
    },
)

PURCHASE_ORDERS: tuple[dict[str, Any], ...] = (
    # Amounts are before tax. A purchase order records what was agreed; VAT is
    # added when the invoice is raised, so comparing an invoice *total* against
    # a PO amount reports every correct invoice as 7% over.
    {"po_number": "PO-2026-9101", "vendor_id": "V-9001", "po_date": "2026-06-02",
     "currency": "THB", "amount": 135_000.00, "status": "open",
     "description": "UPS units and installation"},
    {"po_number": "PO-2026-9102", "vendor_id": "V-9002", "po_date": "2026-06-20",
     "currency": "THB", "amount": 96_000.00, "status": "open",
     "description": "Steel plate, cut to size"},
    {"po_number": "PO-2026-9103", "vendor_id": "V-9004", "po_date": "2026-07-01",
     "currency": "THB", "amount": 38_000.00, "status": "open",
     "description": "Inbound freight, July"},
    {"po_number": "PO-2026-9104", "vendor_id": "V-9003", "po_date": "2026-06-28",
     "currency": "THB", "amount": 60_000.00, "status": "open",
     "description": "งานติดตั้งระบบไฟฟ้า อาคาร B"},
)

CONTRACTS: tuple[dict[str, Any], ...] = ()

# --------------------------------------------------------------------------- #
# The documents                                                                 #
# --------------------------------------------------------------------------- #

# (slug, arrival format, expected to post, document)
DEMO_DOCUMENTS: tuple[tuple[str, str, bool, SampleDoc], ...] = (
    (
        "invoice_clean", "pdf", True,
        SampleDoc(
            doc_id="demo_invoice_clean",
            doc_type="invoice", layout="classic",
            title="Siam Tech invoice against PO-2026-9101",
            demonstrates=(
                "Nothing is wrong with it. The vendor is registered, the bank account "
                "matches the register, the arithmetic closes, and the amount sits inside "
                "the purchase order. Posts without a person."
            ),
            ground_truth={
                "invoice_number": "STS-2026-0318", "invoice_date": "2026-07-24",
                "due_date": "2026-08-23", "vendor_name": "Siam Tech Supply Co., Ltd.",
                "vendor_tax_id": "0105543000129", "buyer_name": BUYER,
                "purchase_order_ref": "PO-2026-9101", "currency": "THB",
                "subtotal": 135_000.00, "tax_rate": 7.0, "tax_amount": 9_450.00,
                "total_amount": 144_450.00, "payment_terms": "Net 30",
                "bank_account": "1234567890",
            },
            line_items=(
                {"description": "UPS 3kVA rack mount", "quantity": 12,
                 "unit_price": 9_500.00, "amount": 114_000.00},
                {"description": "Installation and commissioning", "quantity": 1,
                 "unit_price": 21_000.00, "amount": 21_000.00},
            ),
        ),
    ),
    (
        "invoice_bank_account_changed", "png", False,
        SampleDoc(
            doc_id="demo_invoice_bank_account_changed",
            doc_type="invoice", layout="modern",
            title="Rungruang Steel invoice paying to an account nobody registered",
            demonstrates=(
                "Every number on this page is correct and it must still not be paid. "
                "The remittance account is 5590112233; the register holds 3120045678 "
                "for this vendor. Reading the document more carefully cannot catch it - "
                "only comparing it against something outside the page can."
            ),
            ground_truth={
                "invoice_number": "RSW-2026-1188", "invoice_date": "2026-07-18",
                "due_date": "2026-09-16", "vendor_name": "Rungruang Steel Works Co., Ltd.",
                "vendor_tax_id": "0105552000110", "buyer_name": BUYER,
                "purchase_order_ref": "PO-2026-9102", "currency": "THB",
                "subtotal": 96_000.00, "tax_rate": 7.0, "tax_amount": 6_720.00,
                "total_amount": 102_720.00, "payment_terms": "Net 60",
                # The whole defect, in one field.
                "bank_account": "5590112233",
            },
            line_items=(
                {"description": "Hot-rolled steel plate 12mm x 1500 x 6000", "quantity": 4,
                 "unit_price": 18_500.00, "amount": 74_000.00},
                {"description": "Cutting and edge preparation", "quantity": 4,
                 "unit_price": 5_500.00, "amount": 22_000.00},
            ),
            expected_findings=("bank_account_matches_vendor",),
        ),
    ),
    (
        "invoice_vat_wrong", "pdf", False,
        SampleDoc(
            doc_id="demo_invoice_vat_wrong",
            doc_type="invoice", layout="classic",
            title="Northwind invoice whose VAT is not 7% of its own subtotal",
            demonstrates=(
                "The page states 7%, a subtotal of 38,000.00 and tax of 3,420.00. "
                "Seven percent of 38,000 is 2,660. The document contradicts itself, "
                "and no lookup is needed to know it - the arithmetic is recomputed "
                "rather than read."
            ),
            ground_truth={
                "invoice_number": "NWL-2026-0442", "invoice_date": "2026-07-21",
                "due_date": "2026-09-04", "vendor_name": "Northwind Logistics Ltd.",
                "vendor_tax_id": "0105561071237", "buyer_name": BUYER,
                "purchase_order_ref": "PO-2026-9103", "currency": "THB",
                "subtotal": 38_000.00, "tax_rate": 7.0,
                # 9% of the subtotal, printed under a 7% label.
                "tax_amount": 3_420.00, "total_amount": 41_420.00,
                "payment_terms": "Net 45", "bank_account": "9876543210",
            },
            line_items=(
                {"description": "Inbound freight, Laem Chabang to Bangkok", "quantity": 8,
                 "unit_price": 3_500.00, "amount": 28_000.00},
                {"description": "Customs clearance handling", "quantity": 2,
                 "unit_price": 5_000.00, "amount": 10_000.00},
            ),
            expected_findings=("tax_rate_consistent",),
        ),
    ),
    (
        "invoice_thai_correct", "png", True,
        SampleDoc(
            doc_id="demo_invoice_thai_correct",
            doc_type="invoice", layout="thai_modern",
            language="th",
            title="ใบกำกับภาษี — ไทยพัฒนา วิศวกรรม",
            demonstrates=(
                "Thai labels, a Buddhist-era date, and nothing wrong with it. This is "
                "the case that fails quietly when it fails: the digits are Arabic and "
                "come back from OCR perfectly, so the only question is whether anything "
                "understood that จำนวนเงินทั้งสิ้น is the total and that 2569 is 2026."
            ),
            ground_truth={
                "invoice_number": "TPE-2569-0077", "invoice_date": "2026-07-15",
                "due_date": "2026-08-14",
                "vendor_name": "บริษัท ไทยพัฒนา วิศวกรรม จำกัด",
                "vendor_tax_id": "0105548000117", "buyer_name": BUYER,
                "purchase_order_ref": "PO-2026-9104", "currency": "THB",
                "subtotal": 60_000.00, "tax_rate": 7.0, "tax_amount": 4_200.00,
                "total_amount": 64_200.00, "payment_terms": "Net 30",
                "bank_account": "4450091122",
            },
            line_items=(
                {"description": "งานติดตั้งระบบไฟฟ้า อาคาร B", "quantity": 1,
                 "unit_price": 45_000.00, "amount": 45_000.00},
                {"description": "ค่าแรงช่างเทคนิค", "quantity": 5,
                 "unit_price": 3_000.00, "amount": 15_000.00},
            ),
        ),
    ),
    (
        "invoice_over_purchase_order", "pdf", False,
        SampleDoc(
            doc_id="demo_invoice_over_purchase_order",
            doc_type="invoice", layout="modern",
            title="Siam Tech invoice billing more than PO-2026-9101 allows",
            demonstrates=(
                "The invoice adds up, the vendor is registered and the bank account is "
                "right. It bills 168,000.00 against a purchase order for 135,000.00. "
                "Nothing on the page is wrong; the defect only exists once the order has "
                "been fetched and the two numbers compared, which is the one check here "
                "that needs an agent rather than a condition."
            ),
            ground_truth={
                "invoice_number": "STS-2026-0402", "invoice_date": "2026-07-29",
                "due_date": "2026-08-28", "vendor_name": "Siam Tech Supply Co., Ltd.",
                "vendor_tax_id": "0105543000129", "buyer_name": BUYER,
                "purchase_order_ref": "PO-2026-9101", "currency": "THB",
                "subtotal": 168_000.00, "tax_rate": 7.0, "tax_amount": 11_760.00,
                "total_amount": 179_760.00, "payment_terms": "Net 30",
                "bank_account": "1234567890",
            },
            line_items=(
                {"description": "UPS 3kVA rack mount", "quantity": 16,
                 "unit_price": 9_500.00, "amount": 152_000.00},
                {"description": "Installation and commissioning", "quantity": 1,
                 "unit_price": 16_000.00, "amount": 16_000.00},
            ),
            expected_findings=("invoice_within_purchase_order",),
        ),
    ),
)


def master_rows() -> dict[str, tuple[dict[str, Any], ...]]:
    return {"vendors": VENDORS, "purchase_orders": PURCHASE_ORDERS, "contracts": CONTRACTS}


def to_csv(kind: str) -> str:
    """One master table as the CSV an ERP would export."""
    if kind not in CSV_COLUMNS:
        raise ValueError(f"Unknown master data type {kind!r}.")
    columns = CSV_COLUMNS[kind]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in master_rows()[kind]:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buffer.getvalue()


def _render_page(pdf_path: Path, scale: float = PAGE_SCALE):
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(pdf_path))
    try:
        return document[0].render(scale=scale).to_pil()
    finally:
        document.close()


def _page_text(pdf_path: Path) -> str:
    """What is printed on the page, taken from the PDF's own text layer.

    Read before the page becomes a PNG, because a PNG has no text layer and
    this is what OCR is scored against.
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


def generate(out_dir: Path) -> list[dict[str, Any]]:
    """Render the five documents, their register, and a ground-truth manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)

    master = out_dir / MASTER_DIR
    master.mkdir(parents=True, exist_ok=True)
    for kind in CSV_COLUMNS:
        # A header with no rows is not an empty table, it is a file the loader
        # refuses - and a refusal in the middle of a demo reads as a fault.
        # Nothing here signs contracts, so that file simply is not written.
        if not master_rows()[kind]:
            (master / f"{kind}.csv").unlink(missing_ok=True)
            continue
        (master / f"{kind}.csv").write_text(to_csv(kind), encoding="utf-8")

    manifest: list[dict[str, Any]] = []
    for index, (slug, arrival, posts, doc) in enumerate(DEMO_DOCUMENTS, start=1):
        # The expected outcome is in the filename, decided before the system is
        # asked. A demo whose answers are only known after the run is a demo
        # that can be talked into agreeing with whatever happened.
        name = f"{index}_{slug}_{'POST' if posts else 'HOLD'}"

        pdf_path = out_dir / f"{name}.pdf"
        render_pdf(doc, pdf_path)
        reference = _page_text(pdf_path)

        if arrival == "png":
            artifact = out_dir / f"{name}.png"
            _render_page(pdf_path).save(artifact, format="PNG")
            pdf_path.unlink()
        elif arrival == "pdf":
            artifact = pdf_path
        else:
            raise ValueError(f"Unknown arrival {arrival!r} for {name}")

        entry = asdict(doc)
        entry.update({"file": artifact.name, "expected_post": posts,
                      "arrival": arrival, "reference_text": reference})
        manifest.append(entry)
        logger.info("rendered %s -> %s", doc.doc_id, artifact.name)

    (out_dir / "ground_truth.json").write_text(
        json.dumps({"documents": manifest, "master_data": master_rows()},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
