"""Five documents for a live demonstration, and the register they reconcile against.

Different job from `real_world.py`. That corpus exists to *measure* the system:
thirty documents, several of them ambiguous on purpose, because a corpus that
only contains easy cases reports a number nobody should believe.

This set exists to be *watched*. Every outcome here is unarguable, and each
document fails - or passes - for one reason that can be pointed at in a sentence:

    01 PASS  clean              PDF   nothing is wrong with it
    02 PASS  Thai, correct      PNG   Buddhist dates and Thai labels, all consistent
    03 PASS  no purchase order  PDF   absent is not wrong, and it is under the threshold
    04 PASS  fuel receipt       PNG   a different document type, so the classifier works
    05 PASS  photographed       JPG   tilted and unevenly lit, and still read correctly
    06 FAIL  bank account       PNG   the page pays an account the register does not hold
    07 FAIL  VAT wrong          PDF   7% of the subtotal is not the tax charged
    08 FAIL  over the PO        PDF   bills more than the order allows
    09 FAIL  unknown vendor     PDF   nobody in the register issued this
    10 FAIL  total transposed   PNG   two digits swapped in the one number that matters

Five pass, five fail, and the five failures fail in five different *kinds* of
way: two are the document contradicting itself (07, 10), two are facts about the
world that no amount of careful reading finds (06, 09), and one needs a lookup
and a sum before anyone can tell (08). A demo where every failure is caught by
the same mechanism teaches that the system has one mechanism.

The passes are not five copies of each other either. 03 posts *because* a field
is missing and that is allowed; 04 is a different schema; 05 is the arrival
format that actually puts OCR and preprocessing under load.

Both formats appear because they are not the same path. A PDF carries a text
layer; a PNG carries none, so the OCR model is the only thing standing between
the page and the extraction. Demonstrating on PDFs alone demonstrates the easy
half.

The register below is deliberately small - six vendors, five purchase orders -
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
import re
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
# A phone shoots at more resolution than it needs; downsampling afterwards
# would hide exactly the softness the file exists to carry.
PHOTO_SCALE = 3.0

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
    {
        "vendor_id": "V-9005",
        "legal_name": "บริษัท ปตท. น้ำมันและการค้าปลีก จำกัด",
        "aliases": "PTT Oil and Retail;PTT OR;ปตท. น้ำมันและการค้าปลีก",
        "tax_id": "0105554000770",
        "bank_account": "2210556677",
        "status": "active", "payment_terms": "Immediate", "currency": "THB",
    },
    {
        "vendor_id": "V-9006",
        "legal_name": "Chao Phraya Print & Pack Co., Ltd.",
        "aliases": "Chao Phraya Print;CP Print & Pack",
        "tax_id": "0105559000188",
        "bank_account": "6680033445",
        "status": "active", "payment_terms": "Net 30", "currency": "THB",
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
    {"po_number": "PO-2026-9105", "vendor_id": "V-9006", "po_date": "2026-07-05",
     "currency": "THB", "amount": 128_000.00, "status": "open",
     "description": "Printed cartons and labels"},
    {"po_number": "PO-2026-9106", "vendor_id": "V-9004", "po_date": "2026-07-19",
     "currency": "THB", "amount": 35_000.00, "status": "open",
     "description": "Inbound freight, late July"},
    # The pair the agent rule decides. These two orders exist to differ in one
    # column - `status` - because that column is the whole demonstration: no
    # built-in rule reads it, so a closed order is invisible to all 26 of them
    # and only the rule a person wrote in a sentence catches it.
    # Both amounts sit within `AMOUNT_TOLERANCE` of what their invoice bills,
    # and that is not decoration. `po_amount_within_tolerance` tests
    # `abs(difference)`, so it flags an *underrun* as loudly as an overrun -
    # billing 48,000 against a 72,000 order is 33% under and blocks. Either of
    # these arriving with a second, unintended defect would ruin the one thing
    # the pair is for: being identical everywhere except the status column.
    {"po_number": "PO-2026-9107", "vendor_id": "V-9001", "po_date": "2026-07-08",
     "currency": "THB", "amount": 48_900.00, "status": "open",
     "description": "Rack rails and cable management, phase 1"},
    {"po_number": "PO-2026-9108", "vendor_id": "V-9006", "po_date": "2026-05-14",
     "currency": "THB", "amount": 21_300.00, "status": "closed",
     "description": "Printed manuals, spring run - closed 2026-06-30"},
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
    # ── 6-10 ──────────────────────────────────────────────────────────────
    (
        "invoice_no_purchase_order", "pdf", True,
        SampleDoc(
            doc_id="demo_invoice_no_purchase_order",
            doc_type="invoice", layout="two_column",
            title="Chao Phraya Print invoice quoting no purchase order",
            demonstrates=(
                "There is no PO reference anywhere on this page, and it posts. "
                "`critical` is about consequence, not presence: it means a wrong "
                "value here pays the wrong amount, not that the field must exist. "
                "Whether an order is *required* is business policy, answered by a "
                "threshold, and 42,000.00 is under it."
            ),
            ground_truth={
                "invoice_number": "CPP-2026-0655", "invoice_date": "2026-07-11",
                "due_date": "2026-08-10",
                "vendor_name": "Chao Phraya Print & Pack Co., Ltd.",
                "vendor_tax_id": "0105559000188", "buyer_name": BUYER,
                "currency": "THB",
                "subtotal": 42_000.00, "tax_rate": 7.0, "tax_amount": 2_940.00,
                "total_amount": 44_940.00, "payment_terms": "Net 30",
                "bank_account": "6680033445",
            },
            line_items=(
                {"description": "Printed carton, 3-ply, B-flute", "quantity": 3_000,
                 "unit_price": 12.00, "amount": 36_000.00},
                {"description": "Label printing, 4-colour", "quantity": 1,
                 "unit_price": 6_000.00, "amount": 6_000.00},
            ),
        ),
    ),
    (
        "receipt_fuel_correct", "png", True,
        SampleDoc(
            doc_id="demo_receipt_fuel_correct",
            doc_type="receipt", layout="narrow",
            title="PTT fuel receipt, nothing wrong with it",
            demonstrates=(
                "A different document type, so the classifier has something to do. "
                "A till receipt has no purchase order, no payment terms and no bank "
                "account, and none of that holds it - the rules that need those "
                "fields abstain rather than fail, which is not the same thing."
            ),
            ground_truth={
                "receipt_number": "PTT-880451", "receipt_date": "2026-07-28",
                "merchant_name": "บริษัท ปตท. น้ำมันและการค้าปลีก จำกัด",
                "merchant_tax_id": "0105554000770", "currency": "THB",
                "subtotal": 2_000.00, "tax_amount": 140.00,
                "total_amount": 2_140.00,
                "payment_method": "cash", "expense_category": "fuel",
            },
            line_items=(
                {"description": "Diesel B7", "quantity": 50,
                 "unit_price": 40.00, "amount": 2_000.00},
            ),
        ),
    ),
    (
        "invoice_photographed", "photo", True,
        SampleDoc(
            doc_id="demo_invoice_photographed",
            doc_type="invoice", layout="modern",
            title="Northwind invoice photographed on a desk",
            demonstrates=(
                "The same kind of correct invoice as document 1, arriving the way "
                "most documents actually arrive: a phone photograph, tilted and "
                "unevenly lit. Preprocessing has something to do here, and this is "
                "the only document in the set where it does."
            ),
            ground_truth={
                "invoice_number": "NWL-2026-0518", "invoice_date": "2026-07-26",
                "due_date": "2026-09-09", "vendor_name": "Northwind Logistics Ltd.",
                "vendor_tax_id": "0105561071237", "buyer_name": BUYER,
                "purchase_order_ref": "PO-2026-9106", "currency": "THB",
                "subtotal": 35_000.00, "tax_rate": 7.0, "tax_amount": 2_450.00,
                "total_amount": 37_450.00, "payment_terms": "Net 45",
                "bank_account": "9876543210",
            },
            line_items=(
                {"description": "Inbound freight, Laem Chabang to Bangkok", "quantity": 7,
                 "unit_price": 3_500.00, "amount": 24_500.00},
                {"description": "Customs clearance handling", "quantity": 2,
                 "unit_price": 5_250.00, "amount": 10_500.00},
            ),
        ),
    ),
    (
        "invoice_unknown_vendor", "pdf", False,
        SampleDoc(
            doc_id="demo_invoice_unknown_vendor",
            doc_type="invoice", layout="classic",
            title="Invoice from a company nobody has registered",
            demonstrates=(
                "Every number adds up and the page is perfectly legible. Nobody in "
                "the register is Sathorn Metalworks, and its tax id belongs to no "
                "vendor either - so there is nothing to check the bank account "
                "against. An unrecognised counterparty is not a small problem; it is "
                "the shape most invoice fraud arrives in."
            ),
            ground_truth={
                "invoice_number": "SMW-2026-0091", "invoice_date": "2026-07-22",
                "due_date": "2026-08-21",
                "vendor_name": "Sathorn Metalworks Co., Ltd.",
                "vendor_tax_id": "0105566000451", "buyer_name": BUYER,
                "currency": "THB",
                "subtotal": 42_000.00, "tax_rate": 7.0, "tax_amount": 2_940.00,
                "total_amount": 44_940.00, "payment_terms": "Net 30",
                "bank_account": "7730099001",
            },
            line_items=(
                {"description": "Fabricated steel bracket, galvanised", "quantity": 200,
                 "unit_price": 160.00, "amount": 32_000.00},
                {"description": "Powder coating", "quantity": 200,
                 "unit_price": 50.00, "amount": 10_000.00},
            ),
            expected_findings=("vendor_known",),
        ),
    ),
    (
        "invoice_total_transposed", "png", False,
        SampleDoc(
            doc_id="demo_invoice_total_transposed",
            doc_type="invoice", layout="modern",
            title="Chao Phraya Print invoice whose total has two digits swapped",
            demonstrates=(
                "The line items, the subtotal and the VAT all agree on 128,000.00 "
                "and 8,960.00. The total should read 136,960.00 and reads "
                "139,660.00. Three relations tie those numbers together, one value "
                "breaks all of them, and the others stand - which is how the finding "
                "can name the total rather than shrug at the page."
            ),
            ground_truth={
                "invoice_number": "CPP-2026-0702", "invoice_date": "2026-07-27",
                "due_date": "2026-08-26",
                "vendor_name": "Chao Phraya Print & Pack Co., Ltd.",
                "vendor_tax_id": "0105559000188", "buyer_name": BUYER,
                "purchase_order_ref": "PO-2026-9105", "currency": "THB",
                "subtotal": 128_000.00, "tax_rate": 7.0, "tax_amount": 8_960.00,
                # 136,960.00 with the 6 and the 9 swapped.
                "total_amount": 139_660.00, "payment_terms": "Net 30",
                "bank_account": "6680033445",
            },
            line_items=(
                {"description": "Printed carton, 3-ply, B-flute", "quantity": 8_000,
                 "unit_price": 14.00, "amount": 112_000.00},
                {"description": "Label printing, 4-colour", "quantity": 1,
                 "unit_price": 16_000.00, "amount": 16_000.00},
            ),
            expected_findings=("invoice_totals_consistent",),
        ),
    ),
    # ---------------------------------------------------------------------- #
    # The pair the AI-written rule decides.                                    #
    #                                                                          #
    # These two are deliberately almost the same invoice. Both are arithmetically
    # clean, both quote a real purchase order belonging to their own vendor, both
    # sit inside the ordered amount, and both pay an account the register holds.
    # Every one of the 26 built-in rules passes them identically.
    #
    # They differ in one column of one row of the register - `status` - and no
    # built-in rule reads it. What separates them is a rule somebody wrote as a
    # sentence, which a model turned into a plan and a tool list, and which the
    # engine then enforces. Put side by side they answer the question the demo
    # is actually for: what does the agent add that the fixed rules do not.
    # ---------------------------------------------------------------------- #
    (
        "invoice_purchase_order_open", "pdf", True,
        SampleDoc(
            doc_id="demo_invoice_purchase_order_open",
            doc_type="invoice", layout="classic",
            title="Siam Tech invoice against a purchase order that is still open",
            demonstrates=(
                "The pass half of the agent-rule pair. PO-2026-9107 is open and was "
                "raised for 48,900.00; this invoice bills 48,000.00 against it. The agent "
                "looks the order up, reads the status, compares the amounts and answers "
                "pass - on evidence it fetched, not on anything printed on the page."
            ),
            ground_truth={
                "invoice_number": "STS-2026-0431", "invoice_date": "2026-07-28",
                "due_date": "2026-08-27", "vendor_name": "Siam Tech Supply Co., Ltd.",
                "vendor_tax_id": "0105543000129", "buyer_name": BUYER,
                "purchase_order_ref": "PO-2026-9107", "currency": "THB",
                "subtotal": 48_000.00, "tax_rate": 7.0, "tax_amount": 3_360.00,
                "total_amount": 51_360.00, "payment_terms": "Net 30",
                "bank_account": "1234567890",
            },
            line_items=(
                {"description": "Rack rail set, 42U", "quantity": 24,
                 "unit_price": 1_500.00, "amount": 36_000.00},
                {"description": "Cable management arms", "quantity": 24,
                 "unit_price": 500.00, "amount": 12_000.00},
            ),
        ),
    ),
    (
        "invoice_purchase_order_closed", "pdf", False,
        SampleDoc(
            doc_id="demo_invoice_purchase_order_closed",
            doc_type="invoice", layout="classic",
            title="Chao Phraya Print invoice against a purchase order that was closed",
            demonstrates=(
                "The block half of the pair, and the only document in the set that no "
                "built-in rule objects to. PO-2026-9108 was closed on 2026-06-30; this "
                "invoice is dated a month later and bills 21,000.00 against it, which is "
                "inside the 21,300.00 that was ordered. Arithmetic, vendor, bank "
                "account and amount are all correct. The status is not printed on the "
                "invoice and no fixed rule reads it - the agent rule is the only thing "
                "in the system that stops this being paid."
            ),
            ground_truth={
                "invoice_number": "CPP-2026-0815", "invoice_date": "2026-07-30",
                "due_date": "2026-08-29",
                "vendor_name": "Chao Phraya Print & Pack Co., Ltd.",
                "vendor_tax_id": "0105559000188", "buyer_name": BUYER,
                "purchase_order_ref": "PO-2026-9108", "currency": "THB",
                "subtotal": 21_000.00, "tax_rate": 7.0, "tax_amount": 1_470.00,
                "total_amount": 22_470.00, "payment_terms": "Net 30",
                "bank_account": "6680033445",
            },
            line_items=(
                {"description": "Operator manual, perfect bound", "quantity": 1_500,
                 "unit_price": 12.00, "amount": 18_000.00},
                {"description": "Setup and plate charge", "quantity": 1,
                 "unit_price": 3_000.00, "amount": 3_000.00},
            ),
            expected_findings=("purchase_order_still_has_room",),
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


# Anything at the top of the directory that looks like a document this module
# rendered on a previous run.
_GENERATED = re.compile(r"^\d+_.*\.(pdf|png|jpg)$")


def generate(out_dir: Path) -> list[dict[str, Any]]:
    """Render the ten documents, their register, and a ground-truth manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear what a previous run left. Renaming or reordering the set otherwise
    # leaves the old files sitting beside the new ones, and a demonstration
    # folder holding two answers for the same document is worse than one holding
    # none. Only the generated pattern is touched - README.md and master/ stay.
    for stale in out_dir.iterdir():
        if stale.is_file() and _GENERATED.match(stale.name):
            stale.unlink()

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

    # Passes first, then failures, each group keeping the order it is declared
    # in. Numbering them as they happen to appear in the source interleaves the
    # two, and a directory listing is the first thing anyone looks at.
    ordered = sorted(DEMO_DOCUMENTS, key=lambda row: not row[2])

    manifest: list[dict[str, Any]] = []
    for index, (slug, arrival, posts, doc) in enumerate(ordered, start=1):
        # The verdict comes *before* the description, so a directory listing
        # reads as five PASS then five FAIL rather than as ten filenames whose
        # last word has to be hunted for. It is decided before the system is
        # asked: a set whose answers are only known after the run is a set that
        # can be talked into agreeing with whatever happened.
        name = f"{index:02d}_{'PASS' if posts else 'FAIL'}_{slug}"

        pdf_path = out_dir / f"{name}.pdf"
        render_pdf(doc, pdf_path)
        reference = _page_text(pdf_path)

        if arrival == "png":
            artifact = out_dir / f"{name}.png"
            _render_page(pdf_path).save(artifact, format="PNG")
            pdf_path.unlink()
        elif arrival == "photo":
            # A phone snap: tilted, unevenly lit, softened. The only document in
            # the set that gives preprocessing anything to do.
            from docuflow.samples.degrade import photograph

            artifact = out_dir / f"{name}.jpg"
            photograph(_render_page(pdf_path, PHOTO_SCALE), seed=index * 13 + 5).save(
                artifact, format="JPEG", quality=72
            )
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
