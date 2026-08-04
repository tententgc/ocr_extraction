"""A working corpus: master data as CSV, and documents that arrive like real ones.

The demo corpus in `catalog.py` exists to prove the pipeline runs. This one
exists to prove it holds up, and the difference is in how the documents arrive.
Half of these are photographs and photocopies rather than digital PDFs, because
that is what an accounts-payable inbox actually contains and because a clean
render never puts the OCR model - or the preprocessing in front of it - under
any load at all.

The master data here *extends* the demo master data rather than replacing it, so
one clean database serves both corpora and neither one's documents stop
reconciling when the other is loaded.

Every document declares what it is for. A corpus where you cannot say why a file
is in it decays into a pile nobody dares delete from, and half of it ends up
testing the same path.
"""

from __future__ import annotations

import csv
import io
from typing import Any, Literal

from docuflow.samples.catalog import BUYER, SampleDoc

# How a document reaches us. `pdf` is the digital original, `png` a clean
# rasterisation, `photo` a phone picture taken at an angle under one lamp, and
# `photocopy` a faded third-generation copy.
Arrival = Literal["pdf", "png", "photo", "photocopy"]


EXTRA_VENDORS: tuple[dict[str, Any], ...] = (
    {
        "vendor_id": "V-1006",
        "legal_name": "Rungruang Steel Works Co., Ltd.",
        "aliases": "Rungruang Steel;RSW;บริษัท รุ่งเรือง สตีล เวิร์ค จำกัด",
        "tax_id": "0105552000110",
        "bank_account": "3120045678",
        "status": "active",
        "payment_terms": "Net 60",
        "currency": "THB",
    },
    {
        "vendor_id": "V-1007",
        "legal_name": "Pacific Rim Freight Pte. Ltd.",
        "aliases": "Pacific Rim;PRF Singapore",
        "tax_id": "SG198800123K",
        "bank_account": "SG21DBSS0110001234567",
        "status": "active",
        "payment_terms": "Net 30",
        "currency": "SGD",
    },
    {
        "vendor_id": "V-1008",
        "legal_name": "Thai Beverage Supply Co., Ltd.",
        "aliases": "Thai Beverage Supply;TBS;บริษัท ไทยเบเวอเรจ ซัพพลาย จำกัด",
        "tax_id": "0105563000221",
        "bank_account": "7710028899",
        "status": "active",
        "payment_terms": "Net 15",
        "currency": "THB",
    },
    {
        "vendor_id": "V-1009",
        "legal_name": "Meridian Consulting LLP",
        "aliases": "Meridian;Meridian Consulting",
        "tax_id": "GB428765190",
        "bank_account": "GB29NWBK60161331926819",
        "status": "active",
        "payment_terms": "Net 30",
        "currency": "USD",
    },
    {
        # Blocked, so an otherwise perfect invoice from them must still stop.
        "vendor_id": "V-1010",
        "legal_name": "Krungthep Print House Co., Ltd.",
        "aliases": "Krungthep Print;KPH",
        "tax_id": "0105547000336",
        "bank_account": "2260031122",
        "status": "blocked",
        "payment_terms": "Net 30",
        "currency": "THB",
    },
    {
        "vendor_id": "V-1011",
        "legal_name": "Sunrise Electrical Co., Ltd.",
        "aliases": "Sunrise Electrical;บริษัท ซันไรส์ อิเล็คทริคอล จำกัด",
        "tax_id": "0105571000448",
        "bank_account": "8890014455",
        "status": "active",
        "payment_terms": "Net 45",
        "currency": "THB",
    },
    {
        "vendor_id": "V-1012",
        "legal_name": "Nakhon Chemicals Co., Ltd.",
        "aliases": "Nakhon Chemicals;NCC",
        "tax_id": "0105539000551",
        "bank_account": "6640077331",
        "status": "on_hold",
        "payment_terms": "Net 30",
        "currency": "THB",
    },
)


EXTRA_PURCHASE_ORDERS: tuple[dict[str, Any], ...] = (
    {
        "po_number": "PO-2026-0501", "vendor_id": "V-1006", "po_date": "2026-06-22",
        "currency": "THB", "amount": 415_000.00, "status": "open",
        "description": "Hot-rolled steel plate, 12mm",
    },
    {
        "po_number": "PO-2026-0512", "vendor_id": "V-1007", "po_date": "2026-06-28",
        "currency": "SGD", "amount": 18_500.00, "status": "open",
        "description": "Sea freight, Singapore to Laem Chabang",
    },
    {
        "po_number": "PO-2026-0523", "vendor_id": "V-1008", "po_date": "2026-07-02",
        "currency": "THB", "amount": 42_000.00, "status": "open",
        "description": "Pantry and canteen supplies, Q3",
    },
    {
        "po_number": "PO-2026-0534", "vendor_id": "V-1009", "po_date": "2026-07-06",
        "currency": "USD", "amount": 36_000.00, "status": "open",
        "description": "Operations advisory, phase 2",
    },
    {
        # Already closed: an invoice arriving against it is late, not fraudulent,
        # but it is not something to post without a person looking.
        "po_number": "PO-2026-0545", "vendor_id": "V-1011", "po_date": "2026-05-14",
        "currency": "THB", "amount": 128_000.00, "status": "closed",
        "description": "Low-voltage switchgear panels",
    },
    {
        "po_number": "PO-2026-0556", "vendor_id": "V-1012", "po_date": "2026-07-09",
        "currency": "THB", "amount": 76_500.00, "status": "open",
        "description": "Industrial solvents, drum x 20",
    },
)


EXTRA_CONTRACTS: tuple[dict[str, Any], ...] = (
    {
        "contract_number": "CT-2026-011", "vendor_id": "V-1009",
        "party_a": BUYER, "party_b": "Meridian Consulting LLP",
        "effective_date": "2026-03-01", "expiry_date": "2027-02-28",
        "value": 144_000.00, "currency": "USD",
    },
    {
        # Ends inside the 90-day warning window from the corpus date.
        "contract_number": "CT-2026-014", "vendor_id": "V-1006",
        "party_a": BUYER, "party_b": "Rungruang Steel Works Co., Ltd.",
        "effective_date": "2026-02-01", "expiry_date": "2026-09-30",
        "value": 1_250_000.00, "currency": "THB",
    },
    {
        "contract_number": "CT-2025-031", "vendor_id": "V-1011",
        "party_a": BUYER, "party_b": "Sunrise Electrical Co., Ltd.",
        "effective_date": "2025-01-01", "expiry_date": "2026-06-30",
        "value": 480_000.00, "currency": "THB",
    },
)


# --------------------------------------------------------------------------- #
# The documents                                                                 #
# --------------------------------------------------------------------------- #
# Numbered by what each one is for, so the corpus reads as a list of cases
# rather than as a directory of files.

REAL_DOCUMENTS: tuple[tuple[str, Arrival, SampleDoc], ...] = (
    (
        "1_invoice_clean_matches_po", "pdf",
        SampleDoc(
            doc_id="1_invoice_clean_matches_po",
            doc_type="invoice", layout="classic",
            title="Rungruang Steel invoice against PO-2026-0501",
            demonstrates="Everything reconciles: PO, vendor, bank account, arithmetic. Should post without a person.",
            ground_truth={
                "invoice_number": "RSW-2026-1188", "invoice_date": "2026-07-14",
                "due_date": "2026-09-12", "vendor_name": "Rungruang Steel Works Co., Ltd.",
                "vendor_tax_id": "0105552000110", "buyer_name": BUYER,
                "purchase_order_ref": "PO-2026-0501", "currency": "THB",
                "subtotal": 415_000.00, "tax_rate": 7.0, "tax_amount": 29_050.00,
                "total_amount": 444_050.00, "payment_terms": "Net 60",
                "bank_account": "3120045678",
            },
            line_items=(
                {"description": "Hot-rolled steel plate 12mm x 1500 x 6000", "quantity": 20, "unit_price": 18_500.00, "amount": 370_000.00},
                {"description": "Cutting and edge preparation", "quantity": 20, "unit_price": 2_250.00, "amount": 45_000.00},
            ),
        ),
    ),
    (
        "2_invoice_total_transposed", "pdf",
        SampleDoc(
            doc_id="2_invoice_total_transposed",
            doc_type="invoice", layout="modern",
            title="Thai Beverage invoice whose total has two digits swapped",
            demonstrates="Subtotal, VAT and the line items all agree on 44,940.00; the total reads 49,440.00. The suggestion should name the total and call the transposition.",
            ground_truth={
                "invoice_number": "TBS-7741", "invoice_date": "2026-07-16",
                "due_date": "2026-07-31", "vendor_name": "Thai Beverage Supply Co., Ltd.",
                "vendor_tax_id": "0105563000221", "buyer_name": BUYER,
                "purchase_order_ref": "PO-2026-0523", "currency": "THB",
                "subtotal": 42_000.00, "tax_rate": 7.0, "tax_amount": 2_940.00,
                "total_amount": 49_440.00, "payment_terms": "Net 15",
                "bank_account": "7710028899",
            },
            line_items=(
                {"description": "Drinking water 20L returnable", "quantity": 300, "unit_price": 95.00, "amount": 28_500.00},
                {"description": "Instant coffee, catering tin", "quantity": 45, "unit_price": 300.00, "amount": 13_500.00},
            ),
            expected_findings=("invoice_totals_consistent",),
        ),
    ),
    (
        "3_invoice_photo_on_a_desk", "photo",
        SampleDoc(
            doc_id="3_invoice_photo_on_a_desk",
            doc_type="invoice", layout="classic",
            title="Sunrise Electrical invoice photographed at an angle",
            demonstrates="A phone photo: skewed, lit from one side, slightly soft. Preprocessing has to straighten and relight it before OCR sees it.",
            ground_truth={
                "invoice_number": "SE-2026-0442", "invoice_date": "2026-07-18",
                "due_date": "2026-09-01", "vendor_name": "Sunrise Electrical Co., Ltd.",
                "vendor_tax_id": "0105571000448", "buyer_name": BUYER,
                "purchase_order_ref": "PO-2026-0545", "currency": "THB",
                "subtotal": 128_000.00, "tax_rate": 7.0, "tax_amount": 8_960.00,
                "total_amount": 136_960.00, "payment_terms": "Net 45",
                "bank_account": "8890014455",
            },
            line_items=(
                {"description": "LV switchgear panel, 8-way", "quantity": 4, "unit_price": 26_000.00, "amount": 104_000.00},
                {"description": "Installation and commissioning", "quantity": 1, "unit_price": 24_000.00, "amount": 24_000.00},
            ),
        ),
    ),
    (
        "4_invoice_vat_miscalculated", "pdf",
        SampleDoc(
            doc_id="4_invoice_vat_miscalculated",
            doc_type="invoice", layout="dense",
            title="Nakhon Chemicals invoice where the VAT does not match the rate",
            demonstrates="States 7% but charges 5,355.00 on 76,500.00. The tax is the only value the arithmetic can blame, and the vendor is on hold as well.",
            ground_truth={
                "invoice_number": "NCC-30187", "invoice_date": "2026-07-20",
                "due_date": "2026-08-19", "vendor_name": "Nakhon Chemicals Co., Ltd.",
                "vendor_tax_id": "0105539000551", "buyer_name": BUYER,
                "purchase_order_ref": "PO-2026-0556", "currency": "THB",
                "subtotal": 76_500.00, "tax_rate": 7.0, "tax_amount": 5_355.00,
                "total_amount": 81_855.00, "payment_terms": "Net 30",
                "bank_account": "6640077331",
            },
            line_items=(
                {"description": "Isopropyl alcohol 99%, 200L drum", "quantity": 12, "unit_price": 4_250.00, "amount": 51_000.00},
                {"description": "Acetone technical grade, 200L drum", "quantity": 6, "unit_price": 4_250.00, "amount": 25_500.00},
            ),
            expected_findings=("tax_rate_consistent", "vendor_active"),
        ),
    ),
    (
        "5_invoice_bank_account_changed", "pdf",
        SampleDoc(
            doc_id="5_invoice_bank_account_changed",
            doc_type="invoice", layout="classic",
            title="Rungruang Steel invoice quoting an account that is not theirs",
            demonstrates="The signature of payment-redirection fraud: a correct invoice with new remittance details. Must block, and must not offer a one-click fix.",
            ground_truth={
                "invoice_number": "RSW-2026-1203", "invoice_date": "2026-07-21",
                "due_date": "2026-09-19", "vendor_name": "Rungruang Steel Works Co., Ltd.",
                "vendor_tax_id": "0105552000110", "buyer_name": BUYER,
                "currency": "THB", "subtotal": 96_000.00, "tax_rate": 7.0,
                "tax_amount": 6_720.00, "total_amount": 102_720.00,
                "payment_terms": "Net 60", "bank_account": "0455512399",
            },
            line_items=(
                {"description": "Steel angle bar 50x50x5mm", "quantity": 240, "unit_price": 400.00, "amount": 96_000.00},
            ),
            expected_findings=("bank_account_matches_vendor",),
        ),
    ),
    (
        "6_invoice_unknown_vendor", "png",
        SampleDoc(
            doc_id="6_invoice_unknown_vendor",
            doc_type="invoice", layout="minimal",
            title="Invoice from a company that is not in the vendor file",
            demonstrates="Nothing about the document is wrong; the counterparty is simply unregistered. Blocks, and the agent must not invent a match.",
            ground_truth={
                "invoice_number": "AX-2026-556", "invoice_date": "2026-07-22",
                "due_date": "2026-08-21", "vendor_name": "Apex Industrial Fasteners Ltd.",
                "vendor_tax_id": "0105568000661", "buyer_name": BUYER,
                "currency": "THB", "subtotal": 31_400.00, "tax_rate": 7.0,
                "tax_amount": 2_198.00, "total_amount": 33_598.00,
                "payment_terms": "Net 30",
            },
            line_items=(
                {"description": "Stainless bolt M12 x 60, box of 100", "quantity": 40, "unit_price": 785.00, "amount": 31_400.00},
            ),
            expected_findings=("vendor_known",),
        ),
    ),
    (
        "7_invoice_from_a_blocked_vendor", "pdf",
        SampleDoc(
            doc_id="7_invoice_from_a_blocked_vendor",
            doc_type="invoice", layout="two_column",
            title="Krungthep Print invoice - vendor is blocked in the master file",
            demonstrates="A clean, correct invoice that must still not pay automatically, because the block is a decision someone made deliberately.",
            ground_truth={
                "invoice_number": "KPH-2026-0912", "invoice_date": "2026-07-23",
                "due_date": "2026-08-22", "vendor_name": "Krungthep Print House Co., Ltd.",
                "vendor_tax_id": "0105547000336", "buyer_name": BUYER,
                "currency": "THB", "subtotal": 18_600.00, "tax_rate": 7.0,
                "tax_amount": 1_302.00, "total_amount": 19_902.00,
                "payment_terms": "Net 30", "bank_account": "2260031122",
            },
            line_items=(
                {"description": "Annual report, 500 copies, full colour", "quantity": 500, "unit_price": 37.20, "amount": 18_600.00},
            ),
            expected_findings=("vendor_active",),
        ),
    ),
    (
        "8_invoice_over_the_purchase_order", "pdf",
        SampleDoc(
            doc_id="8_invoice_over_the_purchase_order",
            doc_type="invoice", layout="classic",
            title="Meridian invoice 22% above the purchase order it quotes",
            demonstrates="Arithmetic is perfect; the commercial overrun is the problem. Tests that the 2% purchase-order allowance is still a commercial one.",
            ground_truth={
                "invoice_number": "MC-2026-0774", "invoice_date": "2026-07-24",
                "due_date": "2026-08-23", "vendor_name": "Meridian Consulting LLP",
                "vendor_tax_id": "GB428765190", "buyer_name": BUYER,
                "purchase_order_ref": "PO-2026-0534", "currency": "USD",
                "subtotal": 44_000.00, "tax_rate": 0.0, "tax_amount": 0.00,
                "total_amount": 44_000.00, "payment_terms": "Net 30",
                "bank_account": "GB29NWBK60161331926819",
            },
            line_items=(
                {"description": "Operations advisory, phase 2 - senior partner", "quantity": 110, "unit_price": 280.00, "amount": 30_800.00},
                {"description": "Operations advisory, phase 2 - analyst", "quantity": 132, "unit_price": 100.00, "amount": 13_200.00},
            ),
            expected_findings=("po_amount_within_tolerance",),
        ),
    ),
    (
        "9_invoice_thai_buddhist_dates", "png",
        SampleDoc(
            doc_id="9_invoice_thai_buddhist_dates",
            doc_type="invoice", layout="thai_tax", language="th",
            title="ใบกำกับภาษี from Thai Beverage with Buddhist-era dates",
            demonstrates="Thai layout, Thai month names, dates in BE 2569. Everything reconciles once the calendar is converted.",
            ground_truth={
                "invoice_number": "TBS-7802", "invoice_date": "2026-07-25",
                "due_date": "2026-08-09", "vendor_name": "บริษัท ไทยเบเวอเรจ ซัพพลาย จำกัด",
                "vendor_tax_id": "0105563000221", "buyer_name": BUYER,
                "currency": "THB", "subtotal": 24_000.00, "tax_rate": 7.0,
                "tax_amount": 1_680.00, "total_amount": 25_680.00,
                "payment_terms": "Net 15", "bank_account": "7710028899",
            },
            line_items=(
                {"description": "น้ำดื่ม 20 ลิตร", "quantity": 200, "unit_price": 95.00, "amount": 19_000.00},
                {"description": "กาแฟสำเร็จรูป", "quantity": 20, "unit_price": 250.00, "amount": 5_000.00},
            ),
        ),
    ),
    (
        "10_invoice_no_po_above_threshold", "pdf",
        SampleDoc(
            doc_id="10_invoice_no_po_above_threshold",
            doc_type="invoice", layout="minimal",
            title="Sunrise Electrical invoice for 214,000 with no purchase order",
            demonstrates="Above the 50,000 threshold with nothing to match against. Blocks on policy rather than on anything being wrong.",
            ground_truth={
                "invoice_number": "SE-2026-0471", "invoice_date": "2026-07-26",
                "due_date": "2026-09-09", "vendor_name": "Sunrise Electrical Co., Ltd.",
                "vendor_tax_id": "0105571000448", "buyer_name": BUYER,
                "currency": "THB", "subtotal": 200_000.00, "tax_rate": 7.0,
                "tax_amount": 14_000.00, "total_amount": 214_000.00,
                "payment_terms": "Net 45", "bank_account": "8890014455",
            },
            line_items=(
                {"description": "Emergency cable replacement, plant 2", "quantity": 1, "unit_price": 200_000.00, "amount": 200_000.00},
            ),
            expected_findings=("purchase_order_required",),
        ),
    ),
    (
        "11_receipt_photographed", "photo",
        SampleDoc(
            doc_id="11_receipt_photographed",
            doc_type="receipt", layout="narrow",
            title="Thermal fuel receipt, photographed on a desk",
            demonstrates="80mm thermal paper, low contrast, photographed. The hardest OCR case in the corpus and the one preprocessing matters most for.",
            ground_truth={
                "receipt_number": "PTT-556123", "receipt_date": "2026-07-19",
                "merchant_name": "PTT Station Rama IV", "currency": "THB",
                "subtotal": 1_869.16, "tax_rate": 7.0, "tax_amount": 130.84,
                "total_amount": 2_000.00, "payment_method": "Corporate card",
            },
            line_items=(
                {"description": "Diesel B7", "quantity": 61.35, "unit_price": 32.60, "amount": 2_000.00},
            ),
        ),
    ),
    (
        "12_contract_expiring_soon", "pdf",
        SampleDoc(
            doc_id="12_contract_expiring_soon",
            doc_type="contract", layout="prose",
            title="Rungruang Steel supply agreement ending in under 90 days",
            demonstrates="Nothing is wrong with it; it simply needs renewing before it lapses. A warning, not a block.",
            ground_truth={
                "contract_number": "CT-2026-014", "party_a": BUYER,
                "party_b": "Rungruang Steel Works Co., Ltd.",
                "effective_date": "2026-02-01", "expiry_date": "2026-09-30",
                "auto_renew": "no", "contract_value": 1_250_000.00, "currency": "THB",
            },
            expected_findings=("contract_expiring_soon",),
        ),
    ),
    (
        "13_contract_already_expired", "photocopy",
        SampleDoc(
            doc_id="13_contract_already_expired",
            doc_type="contract", layout="prose",
            title="Sunrise Electrical maintenance agreement that ended in June",
            demonstrates="A faded photocopy of a contract that has already lapsed. Tests the contrast correction and the expiry check together.",
            ground_truth={
                "contract_number": "CT-2025-031", "party_a": BUYER,
                "party_b": "Sunrise Electrical Co., Ltd.",
                "effective_date": "2025-01-01", "expiry_date": "2026-06-30",
                "auto_renew": "no", "contract_value": 480_000.00, "currency": "THB",
            },
            expected_findings=("contract_expired",),
        ),
    ),
    (
        "14_form_vendor_onboarding", "png",
        SampleDoc(
            doc_id="14_form_vendor_onboarding",
            doc_type="form", layout="form",
            title="New supplier registration form",
            demonstrates="An unregistered counterparty is the point of the document, so an unknown vendor must not block it.",
            ground_truth={
                "form_type": "Vendor onboarding", "applicant_name": "Apex Industrial Fasteners Ltd.",
                "tax_id": "0105568000661", "submitted_date": "2026-07-22",
                "contact_email": "accounts@apexfasteners.co.th",
                "contact_phone": "02-115-8890", "bank_account": "1180099221",
            },
        ),
    ),
    (
        "15_invoice_duplicate_submission", "pdf",
        SampleDoc(
            doc_id="15_invoice_duplicate_submission",
            doc_type="invoice", layout="classic",
            title="RSW-2026-1188 sent a second time",
            demonstrates="Identical to document 1. Only detectable against what has already been posted, so it needs document 1 to have gone through first.",
            ground_truth={
                "invoice_number": "RSW-2026-1188", "invoice_date": "2026-07-14",
                "due_date": "2026-09-12", "vendor_name": "Rungruang Steel Works Co., Ltd.",
                "vendor_tax_id": "0105552000110", "buyer_name": BUYER,
                "purchase_order_ref": "PO-2026-0501", "currency": "THB",
                "subtotal": 415_000.00, "tax_rate": 7.0, "tax_amount": 29_050.00,
                "total_amount": 444_050.00, "payment_terms": "Net 60",
                "bank_account": "3120045678",
            },
            line_items=(
                {"description": "Hot-rolled steel plate 12mm x 1500 x 6000", "quantity": 20, "unit_price": 18_500.00, "amount": 370_000.00},
                {"description": "Cutting and edge preparation", "quantity": 20, "unit_price": 2_250.00, "amount": 45_000.00},
            ),
            expected_findings=("duplicate_document",),
            requires_posted="1_invoice_clean_matches_po",
        ),
    ),
    (
        "16_invoice_line_items_do_not_sum", "photocopy",
        SampleDoc(
            doc_id="16_invoice_line_items_do_not_sum",
            doc_type="invoice", layout="dense",
            title="Pacific Rim freight invoice with a line missing from the table",
            demonstrates="The stated subtotal is right and the visible lines are 4,300 short. A faded copy, so it also tests whether the missing line was misread rather than absent.",
            ground_truth={
                "invoice_number": "PRF-SG-20416", "invoice_date": "2026-07-17",
                "due_date": "2026-08-16", "vendor_name": "Pacific Rim Freight Pte. Ltd.",
                "buyer_name": BUYER, "purchase_order_ref": "PO-2026-0512",
                "currency": "SGD", "subtotal": 18_500.00, "tax_rate": 0.0,
                "tax_amount": 0.00, "total_amount": 18_500.00,
                "payment_terms": "Net 30", "bank_account": "SG21DBSS0110001234567",
            },
            line_items=(
                {"description": "Ocean freight 40ft HC", "quantity": 2, "unit_price": 5_600.00, "amount": 11_200.00},
                {"description": "Terminal handling, origin", "quantity": 2, "unit_price": 1_500.00, "amount": 3_000.00},
            ),
            expected_findings=("line_items_sum_to_subtotal",),
        ),
    ),
    (
        "17_invoice_currency_disagrees_with_po", "pdf",
        SampleDoc(
            doc_id="17_invoice_currency_disagrees_with_po",
            doc_type="invoice", layout="modern",
            title="Pacific Rim invoice billed in USD against an SGD order",
            demonstrates="Same numbers, wrong currency. Paying it as printed overpays by the exchange rate, and the fix is on the purchase order.",
            ground_truth={
                "invoice_number": "PRF-SG-20455", "invoice_date": "2026-07-27",
                "due_date": "2026-08-26", "vendor_name": "Pacific Rim Freight Pte. Ltd.",
                "buyer_name": BUYER, "purchase_order_ref": "PO-2026-0512",
                "currency": "USD", "subtotal": 18_500.00, "tax_rate": 0.0,
                "tax_amount": 0.00, "total_amount": 18_500.00,
                "payment_terms": "Net 30", "bank_account": "SG21DBSS0110001234567",
            },
            line_items=(
                {"description": "Ocean freight 40ft HC", "quantity": 2, "unit_price": 7_150.00, "amount": 14_300.00},
                {"description": "Terminal handling, origin", "quantity": 2, "unit_price": 2_100.00, "amount": 4_200.00},
            ),
            expected_findings=("po_currency_matches",),
        ),
    ),
    (
        "18_invoice_unlabelled_amounts", "photo",
        SampleDoc(
            doc_id="18_invoice_unlabelled_amounts",
            doc_type="invoice", layout="minimal",
            title="Invoice with no field labels at all, photographed",
            demonstrates="Three bare numbers in a column and a company name. Nothing for a label matcher to anchor on, so the two extraction channels have to disagree honestly rather than both guessing.",
            ground_truth={
                "invoice_number": "1188-B", "invoice_date": "2026-07-28",
                "vendor_name": "Thai Beverage Supply Co., Ltd.",
                "vendor_tax_id": "0105563000221", "buyer_name": BUYER,
                "currency": "THB", "subtotal": 9_800.00, "tax_rate": 7.0,
                "tax_amount": 686.00, "total_amount": 10_486.00,
            },
            line_items=(
                {"description": "Canteen supplies", "quantity": 1, "unit_price": 9_800.00, "amount": 9_800.00},
            ),
        ),
    ),
)


# --------------------------------------------------------------------------- #
# Master data as CSV                                                            #
# --------------------------------------------------------------------------- #

CSV_COLUMNS: dict[str, tuple[str, ...]] = {
    "vendors": ("vendor_id", "legal_name", "aliases", "tax_id", "bank_account",
                "status", "payment_terms", "currency"),
    "purchase_orders": ("po_number", "vendor_id", "po_date", "currency", "amount",
                        "status", "description"),
    "contracts": ("contract_number", "vendor_id", "party_a", "party_b",
                  "effective_date", "expiry_date", "value", "currency"),
}


def master_rows() -> dict[str, tuple[dict[str, Any], ...]]:
    """Every master record, demo corpus and this one together.

    Combined rather than kept apart because they load into one database and a
    document from either corpus has to find its counterparty there.
    """
    from docuflow.samples.catalog import CONTRACTS, PURCHASE_ORDERS, VENDORS

    return {
        "vendors": VENDORS + EXTRA_VENDORS,
        "purchase_orders": PURCHASE_ORDERS + EXTRA_PURCHASE_ORDERS,
        "contracts": CONTRACTS + EXTRA_CONTRACTS,
    }


def to_csv(kind: str) -> str:
    """One master-data table as the CSV an ERP would export."""
    if kind not in CSV_COLUMNS:
        raise ValueError(f"Unknown master data type {kind!r}.")
    columns = CSV_COLUMNS[kind]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in master_rows()[kind]:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buffer.getvalue()
