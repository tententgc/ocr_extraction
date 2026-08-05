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
        # A fuel station a company buys from every week is a registered
        # counterparty, not a stranger. Leaving it out made every fuel receipt
        # trip `vendor_known` for a reason that was about the master file
        # rather than about the receipt.
        "vendor_id": "V-1013",
        "legal_name": "PTT Station Rama IV",
        "aliases": "PTT Rama IV;PTT Oil and Retail;บริษัท ปตท. น้ำมันและการค้าปลีก จำกัด",
        "tax_id": "0105554000770",
        "bank_account": "0140099887",
        "status": "active",
        "payment_terms": "Corporate card",
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
    # --- for the Thai documents -------------------------------------------
    # Both are stated net of VAT, like every other order here: a PO records what
    # was agreed, and the tax is added when the invoice is raised.
    {
        "po_number": "PO-2026-0567", "vendor_id": "V-1001", "po_date": "2026-07-05",
        "currency": "THB", "amount": 135_000.00, "status": "open",
        "description": "UPS units and installation, plant 1",
    },
    {
        "po_number": "PO-2026-0578", "vendor_id": "V-1006", "po_date": "2026-07-01",
        "currency": "THB", "amount": 88_000.00, "status": "open",
        "description": "Hot-rolled plate and forming, 6mm",
    },
)


EXTRA_CONTRACTS: tuple[dict[str, Any], ...] = (
    {
        # For `contract_without_a_stated_value`. A framework agreement commits
        # no total, so the register holds none either - the document is not
        # missing a value, the arrangement has one to miss.
        "contract_number": "CT-2026-052", "vendor_id": "V-1002",
        "party_a": BUYER, "party_b": "Northwind Logistics Ltd.",
        "effective_date": "2026-06-01", "expiry_date": "2027-05-31",
        "value": None, "currency": "THB",
    },
    {
        # For `contract_dates_out_of_order`. Registered with the dates in the
        # right order, so the disagreement the document creates is only ever
        # about the order, not about being unknown.
        "contract_number": "CT-2026-041", "vendor_id": "V-1009",
        "party_a": BUYER, "party_b": "Meridian Consulting LLP",
        "effective_date": "2027-04-30", "expiry_date": "2027-09-01",
        "value": 96_000.00, "currency": "USD",
    },
    {
        # For `contract_naming_one_party_twice`.
        "contract_number": "CT-2026-046", "vendor_id": "V-1004",
        "party_a": BUYER, "party_b": "Chao Phraya Facility Services Co., Ltd",
        "effective_date": "2026-05-01", "expiry_date": "2027-04-30",
        "value": 210_000.00, "currency": "THB",
    },
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
    {
        # Registered under the Thai legal name, because that is the name the
        # signed agreement carries. The master file is not the place to quietly
        # translate a counterparty into English so our matcher has an easier
        # time - if the Thai name cannot be matched, that is the bug.
        "contract_number": "CT-2026-027", "vendor_id": "V-1004",
        "party_a": BUYER, "party_b": "บริษัท เจ้าพระยา แฟซิลิตี้ จำกัด",
        "effective_date": "2026-06-01", "expiry_date": "2027-05-31",
        "value": 420_000.00, "currency": "THB",
    },
)


# --------------------------------------------------------------------------- #
# The documents                                                                 #
# --------------------------------------------------------------------------- #
# `(slug, arrival, posts, document)`, written to disk as
# `<n>_<slug>_<True|False>.<ext>`.
#
# `posts` is the expected outcome, and putting it in the filename is the point:
# the corpus states what should happen before the system is asked, so the
# evaluation compares two decisions rather than grading the system against
# itself. In a directory listing you can see what the corpus expects.
#
#   True   nothing is wrong with it. Holding it is a false alarm that costs a
#          review, and enough of those are what make people stop reading the
#          queue.
#   False  something is wrong. Posting it is a silent error, which is the
#          failure this whole system exists to prevent.
#
# The two are not symmetric and the evaluation does not treat them as such: a
# missed defect is a wrong payment, a false alarm is a wasted minute.

REAL_DOCUMENTS: tuple[tuple[str, Arrival, bool, SampleDoc], ...] = (
    (
        "invoice_clean_matches_po", "pdf", True,
        SampleDoc(
            doc_id="invoice_clean_matches_po",
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
        "invoice_total_transposed", "pdf", False,
        SampleDoc(
            doc_id="invoice_total_transposed",
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
        "invoice_photo_on_a_desk", "photo", False,
        SampleDoc(
            doc_id="invoice_photo_on_a_desk",
            doc_type="invoice", layout="classic",
            title="Sunrise Electrical invoice photographed at an angle",
            demonstrates="A phone photo: skewed, lit from one side, slightly soft. Preprocessing has to straighten and relight it before OCR sees it. It also quotes PO-2026-0545, the only closed order in the register, which no built-in rule reads.",
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
            # The defect is the closed purchase order, not the photograph -
            # nothing about the document is wrong except the order it quotes,
            # and `status` is a column no built-in rule reads. It is caught by
            # `purchase_order_still_has_room` from `rules.json`, so skipping
            # `docuflow rules load` posts this document.
            #
            # Left undeclared it was worse than untested: whichever rule caught
            # it scored as having fired unexpectedly, and defect recall could
            # not count a defect the corpus never named.
            expected_findings=("purchase_order_still_has_room",),
        ),
    ),
    (
        "invoice_vat_miscalculated", "pdf", False,
        SampleDoc(
            doc_id="invoice_vat_miscalculated",
            doc_type="invoice", layout="dense",
            title="Nakhon Chemicals invoice where the VAT does not match the rate",
            demonstrates="States 7% but charges 4,590.00 on 76,500.00, which is 6%. The tax is the only value the arithmetic can blame, and the vendor is on hold as well.",
            ground_truth={
                "invoice_number": "NCC-30187", "invoice_date": "2026-07-20",
                "due_date": "2026-08-19", "vendor_name": "Nakhon Chemicals Co., Ltd.",
                "vendor_tax_id": "0105539000551", "buyer_name": BUYER,
                "currency": "THB",
                # 7% of 76,500 is 5,355. The document charges 4,590, and the
                # total it prints is consistent with the wrong tax rather than
                # with the rate it states - so the tax is the single value that
                # explains both failures, which is what the diagnosis should say.
                "subtotal": 76_500.00, "tax_rate": 7.0, "tax_amount": 4_590.00,
                "total_amount": 81_090.00, "payment_terms": "Net 30",
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
        "invoice_bank_account_changed", "pdf", False,
        SampleDoc(
            doc_id="invoice_bank_account_changed",
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
        "invoice_unknown_vendor", "png", False,
        SampleDoc(
            doc_id="invoice_unknown_vendor",
            doc_type="invoice", layout="minimal",
            title="Invoice from a company that is not in the vendor file",
            demonstrates="Nothing about the document is wrong; the counterparty is simply unregistered. Blocks, and the agent must not invent a match.",
            ground_truth={
                "invoice_number": "AX-2026-556", "invoice_date": "2026-07-22",
                "due_date": "2026-08-21", "vendor_name": "Apex Industrial Fasteners Ltd.",
                "buyer_name": BUYER,
                "currency": "THB", "subtotal": 31_400.00, "tax_rate": 7.0,
                "tax_amount": 2_198.00, "total_amount": 33_598.00,
            },
            line_items=(
                {"description": "Stainless bolt M12 x 60, box of 100", "quantity": 40, "unit_price": 785.00, "amount": 31_400.00},
            ),
            expected_findings=("vendor_known",),
        ),
    ),
    (
        "invoice_from_a_blocked_vendor", "pdf", False,
        SampleDoc(
            doc_id="invoice_from_a_blocked_vendor",
            doc_type="invoice", layout="two_column",
            title="Krungthep Print invoice - vendor is blocked in the master file",
            demonstrates="A clean, correct invoice that must still not pay automatically, because the block is a decision someone made deliberately.",
            ground_truth={
                "invoice_number": "KPH-2026-0912", "invoice_date": "2026-07-23",
                "vendor_name": "Krungthep Print House Co., Ltd.",
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
        "invoice_over_the_purchase_order", "pdf", False,
        SampleDoc(
            doc_id="invoice_over_the_purchase_order",
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
        "invoice_thai_buddhist_dates", "png", True,
        SampleDoc(
            doc_id="invoice_thai_buddhist_dates",
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
        "invoice_no_po_above_threshold", "pdf", False,
        SampleDoc(
            doc_id="invoice_no_po_above_threshold",
            doc_type="invoice", layout="minimal",
            title="Sunrise Electrical invoice for 214,000 with no purchase order",
            demonstrates="Above the 50,000 threshold with nothing to match against. Blocks on policy rather than on anything being wrong.",
            ground_truth={
                "invoice_number": "SE-2026-0471", "invoice_date": "2026-07-26",
                "due_date": "2026-09-09", "vendor_name": "Sunrise Electrical Co., Ltd.",
                "buyer_name": BUYER,
                "currency": "THB", "subtotal": 200_000.00, "tax_rate": 7.0,
                "tax_amount": 14_000.00, "total_amount": 214_000.00,
                "bank_account": "8890014455",
            },
            line_items=(
                {"description": "Emergency cable replacement, plant 2", "quantity": 1, "unit_price": 200_000.00, "amount": 200_000.00},
            ),
            expected_findings=("purchase_order_required",),
        ),
    ),
    (
        "receipt_photographed", "photo", True,
        SampleDoc(
            doc_id="receipt_photographed",
            doc_type="receipt", layout="narrow",
            title="Thermal fuel receipt, photographed on a desk",
            demonstrates="80mm thermal paper, low contrast, photographed. The hardest OCR case in the corpus and the one preprocessing matters most for.",
            ground_truth={
                "receipt_number": "PTT-556123", "receipt_date": "2026-07-19",
                "merchant_name": "PTT Station Rama IV", "currency": "THB",
                "subtotal": 1_869.16, "tax_rate": 7.0, "tax_amount": 130.84,
                "total_amount": 2_000.00,
            },
            line_items=(
                {"description": "Diesel B7", "quantity": 61.35, "unit_price": 32.60, "amount": 2_000.00},
            ),
        ),
    ),
    (
        "contract_expiring_soon", "pdf", True,
        SampleDoc(
            doc_id="contract_expiring_soon",
            doc_type="contract", layout="prose",
            title="Rungruang Steel supply agreement ending in under 90 days",
            demonstrates="Nothing is wrong with it; it simply needs renewing before it lapses. A warning, not a block.",
            ground_truth={
                "contract_number": "CT-2026-014", "party_a": BUYER,
                "party_b": "Rungruang Steel Works Co., Ltd.",
                "effective_date": "2026-02-01", "expiry_date": "2026-09-30",
                "auto_renew": "no", "contract_value": 1_250_000.00, "currency": "THB",
            },
            body=(
                "This Steel Supply Agreement (the \"Agreement\") is made and entered into as of "
                "1 February 2026 (the \"Effective Date\") by and between Mirtphol Industries PCL, "
                "a public company limited organised under the laws of Thailand (\"Party A\"), and "
                "Rungruang Steel Works Co., Ltd., a company limited organised under the laws of "
                "Thailand (\"Party B\").",
                "1. SCOPE OF SUPPLY. Party B shall supply hot-rolled and cold-rolled steel plate, "
                "angle bar and structural sections to Party A's Rayong fabrication yard, against "
                "purchase orders raised from time to time under this Agreement.",
                "2. TERM. This Agreement shall commence on the Effective Date and shall continue "
                "in force until 30 September 2026. It shall NOT renew automatically. Any "
                "continuation beyond the expiry date requires a new written agreement executed by "
                "both parties.",
                "3. CONTRACT VALUE. The total contract value for the term is THB 1,250,000.00 "
                "(one million two hundred fifty thousand Baht), drawn down against purchase "
                "orders and invoiced on delivery, with payment due within sixty (60) days of "
                "receipt of a valid tax invoice.",
                "4. PRICE ADJUSTMENT. Prices are fixed for the term save for movements in the "
                "published mill base price exceeding five per cent (5%), in which case either "
                "party may request a review on thirty (30) days' written notice.",
                "5. QUALITY. All material shall be supplied with mill test certificates. Material "
                "failing to conform shall be replaced at Party B's cost within fourteen (14) days.",
                "6. GOVERNING LAW. This Agreement shall be governed by and construed in "
                "accordance with the laws of the Kingdom of Thailand.",
            ),
            expected_findings=("contract_expiring_soon",),
        ),
    ),
    (
        "contract_already_expired", "photocopy", False,
        SampleDoc(
            doc_id="contract_already_expired",
            doc_type="contract", layout="prose",
            title="Sunrise Electrical maintenance agreement that ended in June",
            demonstrates="A faded photocopy of a contract that has already lapsed. Tests the contrast correction and the expiry check together.",
            ground_truth={
                "contract_number": "CT-2025-031", "party_a": BUYER,
                "party_b": "Sunrise Electrical Co., Ltd.",
                "effective_date": "2025-01-01", "expiry_date": "2026-06-30",
                "auto_renew": "no", "contract_value": 480_000.00, "currency": "THB",
            },
            body=(
                "This Electrical Maintenance Agreement (the \"Agreement\") is made and entered "
                "into as of 1 January 2025 (the \"Effective Date\") by and between Mirtphol "
                "Industries PCL, a public company limited organised under the laws of Thailand "
                "(\"Party A\"), and Sunrise Electrical Co., Ltd., a company limited organised "
                "under the laws of Thailand (\"Party B\").",
                "1. SCOPE OF SERVICES. Party B shall provide scheduled preventive maintenance, "
                "thermographic inspection and emergency call-out cover for all low-voltage "
                "switchgear and distribution equipment at Party A's Bangkok headquarters.",
                "2. TERM. This Agreement shall commence on the Effective Date and shall continue "
                "in force until 30 June 2026. It shall NOT renew automatically and shall lapse on "
                "the expiry date unless replaced by a new written agreement.",
                "3. CONTRACT VALUE. The total contract value is THB 480,000.00 (four hundred "
                "eighty thousand Baht) for the full term, invoiced quarterly in advance with "
                "payment due within forty-five (45) days of receipt of a valid tax invoice.",
                "4. RESPONSE TIMES. Party B shall attend site within four (4) hours of an "
                "emergency call-out during business hours and within eight (8) hours otherwise.",
                "5. SAFETY. Party B's personnel shall hold current electrical safety "
                "certification and shall comply with Party A's permit-to-work procedures at all "
                "times while on site.",
                "6. GOVERNING LAW. This Agreement shall be governed by and construed in "
                "accordance with the laws of the Kingdom of Thailand.",
            ),
            expected_findings=("contract_expired",),
        ),
    ),
    (
        "form_vendor_onboarding", "png", True,
        SampleDoc(
            doc_id="form_vendor_onboarding",
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
        "invoice_duplicate_submission", "pdf", False,
        SampleDoc(
            doc_id="invoice_duplicate_submission",
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
        "invoice_line_items_do_not_sum", "photocopy", False,
        SampleDoc(
            doc_id="invoice_line_items_do_not_sum",
            doc_type="invoice", layout="dense",
            title="Pacific Rim freight invoice with a line missing from the table",
            demonstrates="The stated subtotal is right and the visible lines are 4,300 short. A faded copy, so it also tests whether the missing line was misread rather than absent.",
            ground_truth={
                "invoice_number": "PRF-SG-20416", "invoice_date": "2026-07-17",
                "due_date": "2026-08-16", "vendor_name": "Pacific Rim Freight Pte. Ltd.",
                "buyer_name": BUYER,
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
        "invoice_currency_disagrees_with_po", "pdf", False,
        SampleDoc(
            doc_id="invoice_currency_disagrees_with_po",
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
        "invoice_unlabelled_amounts", "photo", True,
        SampleDoc(
            doc_id="invoice_unlabelled_amounts",
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
# Structural variation: the same correct document, laid out differently          #
# --------------------------------------------------------------------------- #
# Every one of these is a perfectly ordinary invoice or receipt that reconciles
# in every way - the arithmetic adds up, the vendor is registered and active,
# the purchase order is open and covers it, the bank account is the one on file.
# All of them must post.
#
# What varies is only the shape: a different renderer, a different label
# vocabulary, values in a different order, a table that is not a table. That
# variation is the point. A pipeline tuned on one house style scores beautifully
# on documents that look like the ones it was tuned on, and the failure it hides
# is the expensive one - not a wrong value, but a valid document held because it
# was laid out unfamiliarly. Fifty of those and nobody reads the review queue.
#
# So this half of the corpus measures the false-alarm rate, which is the number
# the accuracy figures cannot see.

VARIED_DOCUMENTS: tuple[tuple[str, Arrival, bool, SampleDoc], ...] = (
    (
        "invoice_two_column_layout", "pdf", True,
        SampleDoc(
            doc_id="invoice_two_column_layout",
            doc_type="invoice", layout="two_column",
            title="Thai Beverage invoice in a two-column layout",
            demonstrates="Correct in every way; the header sits in two columns instead of one. Must post.",
            ground_truth={
                "invoice_number": "TBS-7855", "invoice_date": "2026-07-29",
                "vendor_name": "Thai Beverage Supply Co., Ltd.",
                "vendor_tax_id": "0105563000221", "buyer_name": BUYER,
                "currency": "THB",
                "subtotal": 38_000.00, "tax_rate": 7.0, "tax_amount": 2_660.00,
                "total_amount": 40_660.00, "payment_terms": "Net 15",
                "bank_account": "7710028899",
            },
            line_items=(
                {"description": "Drinking water 20L returnable", "quantity": 280, "unit_price": 95.00, "amount": 26_600.00},
                {"description": "Instant coffee, catering tin", "quantity": 38, "unit_price": 300.00, "amount": 11_400.00},
            ),
        ),
    ),
    (
        "invoice_dense_no_whitespace", "pdf", True,
        SampleDoc(
            doc_id="invoice_dense_no_whitespace",
            doc_type="invoice", layout="dense",
            title="Rungruang Steel invoice typeset with no whitespace",
            demonstrates="Correct, and packed edge to edge. Tests that density alone does not hold a document.",
            # The dense layout prints no purchase-order line, so the ground truth
            # must not claim one - and the total therefore has to sit below the
            # 50,000 threshold, or `purchase_order_required` is right to block
            # and the document is not the clean case it is here to be.
            # Declaring fields the chosen layout never draws was the authoring
            # error that made four of these look like false alarms.
            ground_truth={
                "invoice_number": "RSW-2026-1240", "invoice_date": "2026-07-29",
                "due_date": "2026-09-27", "vendor_name": "Rungruang Steel Works Co., Ltd.",
                "vendor_tax_id": "0105552000110", "buyer_name": BUYER,
                "currency": "THB",
                "subtotal": 28_000.00, "tax_rate": 7.0, "tax_amount": 1_960.00,
                "total_amount": 29_960.00, "bank_account": "3120045678",
            },
            line_items=(
                {"description": "Steel channel 100x50x5mm", "quantity": 70, "unit_price": 400.00, "amount": 28_000.00},
            ),
        ),
    ),
    (
        "invoice_minimal_unlabelled", "pdf", True,
        SampleDoc(
            doc_id="invoice_minimal_unlabelled",
            doc_type="invoice", layout="minimal",
            title="Sunrise Electrical invoice with almost no labels",
            demonstrates="Correct, but the amounts are a bare column with no words beside them. The label-first channel has nothing to anchor on.",
            # No payment terms and no tax ID: the minimal layout draws neither,
            # and claiming them in the ground truth invents fields the page does
            # not have.
            ground_truth={
                "invoice_number": "SE-2026-0488", "invoice_date": "2026-07-29",
                "due_date": "2026-09-12", "vendor_name": "Sunrise Electrical Co., Ltd.",
                "buyer_name": BUYER,
                "currency": "THB", "subtotal": 22_000.00, "tax_rate": 7.0,
                "tax_amount": 1_540.00, "total_amount": 23_540.00,
                "bank_account": "8890014455",
            },
            line_items=(
                {"description": "Distribution board, 12-way", "quantity": 4, "unit_price": 5_500.00, "amount": 22_000.00},
            ),
        ),
    ),
    (
        "invoice_thai_layout_correct", "png", True,
        SampleDoc(
            doc_id="invoice_thai_layout_correct",
            doc_type="invoice", layout="thai_tax", language="th",
            title="ใบกำกับภาษี that is entirely correct",
            demonstrates="Thai layout, Thai month names, Buddhist-era dates, and nothing wrong with it. Must post.",
            ground_truth={
                "invoice_number": "TBS-7860", "invoice_date": "2026-07-30",
                "due_date": "2026-08-14", "vendor_name": "บริษัท ไทยเบเวอเรจ ซัพพลาย จำกัด",
                "vendor_tax_id": "0105563000221", "buyer_name": BUYER,
                "currency": "THB", "subtotal": 16_000.00, "tax_rate": 7.0,
                "tax_amount": 1_120.00, "total_amount": 17_120.00,
                "payment_terms": "Net 15", "bank_account": "7710028899",
            },
            line_items=(
                {"description": "น้ำดื่ม 20 ลิตร", "quantity": 150, "unit_price": 95.00, "amount": 14_250.00},
                {"description": "กาแฟสำเร็จรูป", "quantity": 7, "unit_price": 250.00, "amount": 1_750.00},
            ),
        ),
    ),
    (
        "invoice_modern_scanned", "png", True,
        SampleDoc(
            doc_id="invoice_modern_scanned",
            doc_type="invoice", layout="modern",
            title="Meridian invoice, modern layout, scanned rather than digital",
            demonstrates="Correct, zero-rated, in USD, and arriving as an image with no text layer.",
            ground_truth={
                "invoice_number": "MC-2026-0801", "invoice_date": "2026-07-30",
                "due_date": "2026-08-29", "vendor_name": "Meridian Consulting LLP",
                "vendor_tax_id": "GB428765190", "buyer_name": BUYER,
                # The full order value. Invoicing 30,000 against a 36,000 order
                # is a part delivery, which `po_amount_within_tolerance` is
                # right to stop - and this document is here to be the clean
                # case, not to smuggle in a second defect.
                "purchase_order_ref": "PO-2026-0534", "currency": "USD",
                "subtotal": 36_000.00, "tax_rate": 0.0, "tax_amount": 0.00,
                "total_amount": 36_000.00, "payment_terms": "Net 30",
                "bank_account": "GB29NWBK60161331926819",
            },
            line_items=(
                {"description": "Operations advisory, phase 2 - senior partner", "quantity": 90, "unit_price": 280.00, "amount": 25_200.00},
                {"description": "Operations advisory, phase 2 - analyst", "quantity": 108, "unit_price": 100.00, "amount": 10_800.00},
            ),
        ),
    ),
    (
        "receipt_clean_thermal", "pdf", True,
        SampleDoc(
            doc_id="receipt_clean_thermal",
            doc_type="receipt", layout="narrow",
            title="Thermal receipt, 80mm, clean and correct",
            demonstrates="The same shape as the photographed receipt, without the photograph. Isolates layout from image quality.",
            ground_truth={
                "receipt_number": "PTT-556890", "receipt_date": "2026-07-28",
                "merchant_name": "PTT Station Rama IV", "currency": "THB",
                "subtotal": 1_401.87, "tax_rate": 7.0, "tax_amount": 98.13,
                "total_amount": 1_500.00,
            },
            # Net, not gross. A line item priced at the VAT-inclusive total makes
            # the rows disagree with the subtotal by exactly the tax, and
            # `line_items_sum_to_subtotal` is right to say so.
            line_items=(
                {"description": "Diesel B7", "quantity": 43.00, "unit_price": 32.60, "amount": 1_401.87},
            ),
        ),
    ),
)


# --------------------------------------------------------------------------- #
# Thai documents                                                                #
# --------------------------------------------------------------------------- #
# Thai was in the corpus already, but only in one shape: two invoices on the
# `thai_tax` layout, which prints ๒๕,๖๘๐.๐๐ throughout. That is what a
# government form looks like. It is not what a supplier's accounting package
# emits, and the gap matters more than it sounds.
#
# Thai numerals are a *recognition* problem: either the OCR returns ๙ or it does
# not, and either way the failure is loud. Arabic digits under Thai labels are a
# *comprehension* problem - the numbers come back perfectly and the only question
# is whether anything understood that จำนวนเงินทั้งสิ้น is the total. An
# English-trained extractor gets that wrong quietly, which is the expensive way.
#
# So these six hold the language fixed and vary what is being asked of it:
#
#   25  Thai labels, Arabic digits, Buddhist-era dates. The common real case.
#   26  the same, with the VAT miscalculated. Do the rules fire in Thai, and
#       does the suggestion name the right field?
#   27  a receipt, not an invoice. Can the classifier tell ใบเสร็จ from
#       ใบกำกับภาษี on a page that says ภาษีมูลค่าเพิ่ม like both of them do?
#   28  a contract in Thai prose. No arithmetic anywhere to catch a misread date.
#   29  a supplier who is genuinely not in the master file, written หจก. rather
#       than บริษัท. Is an unknown Thai vendor caught, or fuzzy-matched onto a
#       registered one that merely reads alike?
#   30  a registered vendor with no tax ID printed. The match has to be made on
#       the Thai name alone, which is the single place a Thai vendor is most
#       likely to be lost.

THAI_DOCUMENTS: tuple[tuple[str, Arrival, bool, SampleDoc], ...] = (
    (
        "invoice_thai_arabic_digits", "png", True,
        SampleDoc(
            doc_id="invoice_thai_arabic_digits",
            doc_type="invoice", layout="thai_modern", language="th",
            title="ใบกำกับภาษี from Siam Tech with Arabic numerals",
            demonstrates="Thai labels throughout, amounts in 135,000.00 rather than ๑๓๕,๐๐๐.๐๐, dates as 24 กรกฎาคม 2569. The ordinary Thai business invoice, and the shape the corpus was missing.",
            ground_truth={
                "invoice_number": "STS-2026-0318", "invoice_date": "2026-07-24",
                "due_date": "2026-08-23", "vendor_name": "บริษัท สยามเทค ซัพพลาย จำกัด",
                "vendor_tax_id": "0105543000129", "buyer_name": BUYER,
                "purchase_order_ref": "PO-2026-0567", "currency": "THB",
                "subtotal": 135_000.00, "tax_rate": 7.0, "tax_amount": 9_450.00,
                "total_amount": 144_450.00, "payment_terms": "Net 30",
                "bank_account": "1234567890",
            },
            line_items=(
                {"description": "เครื่องสำรองไฟ UPS 3kVA", "quantity": 12, "unit_price": 9_500.00, "amount": 114_000.00},
                {"description": "ติดตั้งและทดสอบระบบ", "quantity": 1, "unit_price": 21_000.00, "amount": 21_000.00},
            ),
        ),
    ),
    (
        "invoice_thai_vat_miscalculated", "pdf", False,
        SampleDoc(
            doc_id="invoice_thai_vat_miscalculated",
            doc_type="invoice", layout="thai_modern", language="th",
            title="ใบกำกับภาษี stating 7% and charging 6%",
            demonstrates="Says ภาษีมูลค่าเพิ่ม 7% and charges 5,280.00 on 88,000.00, which is 6%. The total agrees with the tax as printed, so only the rate is contradicted - and the tax is the only value the arithmetic can blame.",
            ground_truth={
                "invoice_number": "RSW-2026-1240", "invoice_date": "2026-07-27",
                "due_date": "2026-09-25", "vendor_name": "บริษัท รุ่งเรือง สตีล เวิร์ค จำกัด",
                "vendor_tax_id": "0105552000110", "buyer_name": BUYER,
                "purchase_order_ref": "PO-2026-0578", "currency": "THB",
                "subtotal": 88_000.00, "tax_rate": 7.0, "tax_amount": 5_280.00,
                "total_amount": 93_280.00, "payment_terms": "Net 60",
                "bank_account": "3120045678",
            },
            line_items=(
                {"description": "เหล็กแผ่นรีดร้อน หนา 6 มม.", "quantity": 4, "unit_price": 16_000.00, "amount": 64_000.00},
                {"description": "งานตัดและพับขึ้นรูป", "quantity": 1, "unit_price": 24_000.00, "amount": 24_000.00},
            ),
            expected_findings=("tax_rate_consistent",),
        ),
    ),
    (
        "receipt_thai_fuel_station", "photo", True,
        SampleDoc(
            doc_id="receipt_thai_fuel_station",
            doc_type="receipt", layout="thai_receipt", language="th",
            title="ใบเสร็จรับเงิน from a PTT station",
            demonstrates="A Thai receipt, photographed. It has to be classified as a receipt rather than an invoice from ใบเสร็จ alone, on a page that also says ภาษีมูลค่าเพิ่ม - and the counterparty has to reach merchant_name, not vendor_name.",
            # `tax_rate` and `payment_method` are deliberately absent, and both
            # for the reason this corpus keeps relearning: ground truth must be
            # something the system was actually asked for, in the form it was
            # asked for it.
            #
            # The receipt schema has no `tax_rate` field at all - the page
            # prints ภาษีมูลค่าเพิ่ม 7%, but nothing is looking for it, so
            # declaring it scores a miss against a question never posed.
            #
            # `payment_method` is an enum: a classification, not a
            # transcription. The page says บัตรเครดิตองค์กร and the right
            # answer is a category, so asserting the printed Thai would mark a
            # correct classification wrong.
            ground_truth={
                "receipt_number": "PTT-880451", "receipt_date": "2026-07-28",
                "merchant_name": "บริษัท ปตท. น้ำมันและการค้าปลีก จำกัด",
                "merchant_tax_id": "0105554000770", "currency": "THB",
                "subtotal": 2_000.00, "tax_amount": 140.00, "total_amount": 2_140.00,
            },
            # Net of VAT. Pricing the line at the pump's inclusive figure makes
            # the rows disagree with the subtotal by exactly the tax, and
            # `line_items_sum_to_subtotal` is right to say so - a lesson this
            # corpus has already learned once, on the English PTT receipt.
            line_items=(
                {"description": "น้ำมันดีเซล B7", "quantity": 50.00, "unit_price": 40.00, "amount": 2_000.00},
            ),
        ),
    ),
    (
        "contract_thai_service", "pdf", True,
        SampleDoc(
            doc_id="contract_thai_service",
            doc_type="contract", layout="thai_contract", language="th",
            title="สัญญาจ้างบริการ with Chao Phraya Facility",
            demonstrates="A contract in Thai prose. Every field is a phrase in running text and there is no arithmetic anywhere, so a misread Buddhist-era date has nothing at all to contradict it.",
            # Only what the page states in a form something can read off it.
            # `contract_type` is nowhere on the document, and the notice period
            # is written หกสิบวัน rather than 60 - claiming either would be
            # scoring the system against ground truth its own renderer never
            # drew, which this corpus has already done once and paid for.
            ground_truth={
                "contract_number": "CT-2026-027",
                "contract_title": "สัญญาจ้างบริการทำความสะอาดและบำรุงรักษาอาคาร",
                "party_a": BUYER, "party_b": "บริษัท เจ้าพระยา แฟซิลิตี้ จำกัด",
                "effective_date": "2026-06-01", "expiry_date": "2027-05-31",
                "contract_value": 420_000.00, "currency": "THB",
                "payment_terms": "ชำระรายเดือนภายใน 30 วัน",
                "auto_renew": "no", "governing_law": "กฎหมายไทย",
            },
            body=(
                "ข้อ 1. ขอบเขตของงาน ผู้รับจ้างตกลงให้บริการทำความสะอาดพื้นที่สำนักงานและโรงงาน "
                "รวมถึงงานบำรุงรักษาระบบปรับอากาศและระบบไฟฟ้าส่องสว่างภายในอาคารของผู้ว่าจ้าง "
                "ตามมาตรฐานและความถี่ที่กำหนดไว้ในเอกสารแนบท้ายสัญญา",
                "ข้อ 2. ระยะเวลาของสัญญา สัญญานี้มีกำหนดระยะเวลาหนึ่งปี นับตั้งแต่วันที่มีผลบังคับใช้ "
                "หากฝ่ายใดประสงค์จะบอกเลิกสัญญา ต้องแจ้งเป็นหนังสือให้อีกฝ่ายทราบล่วงหน้าไม่น้อยกว่าหกสิบวัน "
                "สัญญานี้ไม่ต่ออายุโดยอัตโนมัติ",
                "ข้อ 3. ค่าจ้างและการชำระเงิน ผู้ว่าจ้างตกลงชำระค่าจ้างเป็นรายเดือน "
                "โดยผู้รับจ้างจะออกใบแจ้งหนี้ในวันทำการสุดท้ายของแต่ละเดือน "
                "และผู้ว่าจ้างจะชำระภายในสามสิบวันนับแต่วันที่ได้รับใบแจ้งหนี้ที่ถูกต้อง",
                "ข้อ 4. กฎหมายที่ใช้บังคับ สัญญานี้อยู่ภายใต้บังคับแห่งกฎหมายไทย "
                "ข้อพิพาทใดอันเกิดจากสัญญานี้ให้อยู่ในเขตอำนาจของศาลไทย",
            ),
        ),
    ),
    (
        "invoice_thai_unknown_vendor", "png", False,
        SampleDoc(
            doc_id="invoice_thai_unknown_vendor",
            doc_type="invoice", layout="thai_modern", language="th",
            title="ใบแจ้งหนี้ from หจก. สมชายการช่าง, who is not registered",
            demonstrates="A limited partnership (หจก.) rather than a company (บริษัท), and nobody has onboarded them. Everything on the page reconciles; the only thing wrong is that we have no record of who they are. The risk is the opposite of a miss - a fuzzy match onto a registered Thai vendor that merely reads alike.",
            ground_truth={
                "invoice_number": "SCK-2569-0142", "invoice_date": "2026-07-29",
                "due_date": "2026-08-28", "vendor_name": "หจก. สมชายการช่าง",
                # Checksum-valid, so `tax_id_checksum` passes and the unknown
                # vendor is the only thing this document is testing.
                "vendor_tax_id": "0105566000991", "buyer_name": BUYER,
                "currency": "THB", "subtotal": 42_000.00, "tax_rate": 7.0,
                "tax_amount": 2_940.00, "total_amount": 44_940.00,
                "payment_terms": "Net 30", "bank_account": "6650071234",
            },
            line_items=(
                {"description": "งานเชื่อมโครงเหล็กหลังคาโรงงาน", "quantity": 1, "unit_price": 42_000.00, "amount": 42_000.00},
            ),
            expected_findings=("vendor_known",),
        ),
    ),
    (
        "invoice_thai_no_tax_id", "png", True,
        SampleDoc(
            doc_id="invoice_thai_no_tax_id",
            doc_type="invoice", layout="thai_modern", language="th",
            title="ใบกำกับภาษี from Sunrise Electrical with no tax ID printed",
            demonstrates="A registered vendor named only in Thai, with a phone number where the tax ID usually goes. The exact identifier is gone, so the match has to be made on บริษัท ซันไรส์ อิเล็คทริคอล จำกัด against an English legal name and a Thai alias - and if Thai vendors are lost anywhere, it is here.",
            ground_truth={
                "invoice_number": "SE-2026-0498", "invoice_date": "2026-07-31",
                "due_date": "2026-09-14", "vendor_name": "บริษัท ซันไรส์ อิเล็คทริคอล จำกัด",
                "buyer_name": BUYER, "currency": "THB",
                "subtotal": 36_000.00, "tax_rate": 7.0, "tax_amount": 2_520.00,
                "total_amount": 38_520.00, "payment_terms": "Net 45",
                "bank_account": "8890014455",
            },
            line_items=(
                {"description": "หลอดไฟ LED 18W", "quantity": 300, "unit_price": 90.00, "amount": 27_000.00},
                {"description": "ค่าแรงติดตั้ง", "quantity": 1, "unit_price": 9_000.00, "amount": 9_000.00},
            ),
        ),
    ),
)



# --------------------------------------------------------------------------- #
# Coverage: rules the corpus above never exercised                              #
# --------------------------------------------------------------------------- #
#
# The 2026-08-06 evaluation reported twenty-two of thirty-four rules as
# **untested** rather than as passing, and untested is the honest word: a rule
# no document trips has never been shown to work, and the first time it matters
# will be on a real document with real money behind it.
#
# One defect each, and each names the rule it is for. Appended rather than
# interleaved so the numbering of every document above stays put - the README,
# the tests and `docs/EVALUATION.md` all quote documents by number.

COVERAGE_DOCUMENTS: tuple[tuple[str, Arrival, bool, SampleDoc], ...] = (
    (
        "invoice_dated_next_year", "pdf", False,
        SampleDoc(
            doc_id="invoice_dated_next_year",
            doc_type="invoice", layout="classic",
            title="Siam Tech invoice dated well into the future",
            demonstrates="Dated 2027-03-14. Past 120 days a future date stops being a typo and starts being a document that should not exist yet, so it blocks rather than warns.",
            ground_truth={
                "invoice_number": "STS-2027-0031", "invoice_date": "2027-03-14",
                "due_date": "2027-04-13", "vendor_name": "Siam Tech Supply Co., Ltd.",
                "vendor_tax_id": "0105543000129", "buyer_name": BUYER,
                "currency": "THB", "subtotal": 42_000.00, "tax_rate": 7.0,
                "tax_amount": 2_940.00, "total_amount": 44_940.00,
                "payment_terms": "Net 30", "bank_account": "1234567890",
            },
            line_items=(
                {"description": "Rack-mount patch panel, 48 port", "quantity": 20, "unit_price": 2_100.00, "amount": 42_000.00},
            ),
            expected_findings=("dates_not_in_future",),
        ),
    ),
    (
        "invoice_quotes_an_unknown_purchase_order", "pdf", False,
        SampleDoc(
            doc_id="invoice_quotes_an_unknown_purchase_order",
            doc_type="invoice", layout="classic",
            title="Northwind invoice against an order nobody raised",
            demonstrates="Quotes PO-2026-9999, which is not in the register at all. Different from quoting no order: this one claims an authority that does not exist.",
            ground_truth={
                "invoice_number": "NWL-2026-0741", "invoice_date": "2026-07-24",
                "due_date": "2026-08-23", "vendor_name": "Northwind Logistics Ltd.",
                "vendor_tax_id": "0105561071237", "buyer_name": BUYER,
                "purchase_order_ref": "PO-2026-9999", "currency": "THB",
                "subtotal": 31_000.00, "tax_rate": 7.0, "tax_amount": 2_170.00,
                "total_amount": 33_170.00, "payment_terms": "Net 30",
                "bank_account": "9876543210",
            },
            line_items=(
                {"description": "Inbound freight, August", "quantity": 1, "unit_price": 31_000.00, "amount": 31_000.00},
            ),
            expected_findings=("po_exists",),
        ),
    ),
    (
        "invoice_with_a_broken_tax_id", "pdf", False,
        SampleDoc(
            doc_id="invoice_with_a_broken_tax_id",
            doc_type="invoice", layout="classic",
            title="Rungruang Steel invoice whose tax ID fails its check digit",
            demonstrates="Thirteen digits that are not a valid Thai tax ID. The vendor still resolves by name, so this isolates the checksum from vendor matching.",
            ground_truth={
                "invoice_number": "RSW-2026-1204", "invoice_date": "2026-07-21",
                "due_date": "2026-09-19", "vendor_name": "Rungruang Steel Works Co., Ltd.",
                "vendor_tax_id": "0105552000119", "buyer_name": BUYER,
                # Deliberately under `po_required_above` (50,000), so the
                # checksum is the only thing wrong with the document. A second
                # finding by accident makes a corpus document test two things
                # and prove neither.
                "currency": "THB", "subtotal": 40_000.00, "tax_rate": 7.0,
                "tax_amount": 2_800.00, "total_amount": 42_800.00,
                "payment_terms": "Net 60", "bank_account": "3120045678",
            },
            line_items=(
                {"description": "Cold-rolled sheet 3mm", "quantity": 10, "unit_price": 4_000.00, "amount": 40_000.00},
            ),
            expected_findings=("tax_id_checksum",),
        ),
    ),
    (
        "invoice_due_before_it_was_issued", "pdf", True,
        SampleDoc(
            doc_id="invoice_due_before_it_was_issued",
            doc_type="invoice", layout="classic",
            title="Siam Tech invoice whose due date precedes its issue date",
            demonstrates="Due 2026-07-01 on an invoice issued 2026-07-20. A warning, because the ordinary cause is a misread date rather than a bad invoice - and a warning must not hold a document on its own.",
            ground_truth={
                "invoice_number": "STS-2026-0522", "invoice_date": "2026-07-20",
                "due_date": "2026-07-01", "vendor_name": "Siam Tech Supply Co., Ltd.",
                "vendor_tax_id": "0105543000129", "buyer_name": BUYER,
                "currency": "THB", "subtotal": 18_000.00, "tax_rate": 7.0,
                "tax_amount": 1_260.00, "total_amount": 19_260.00,
                "payment_terms": "Net 30", "bank_account": "1234567890",
            },
            line_items=(
                {"description": "Fibre patch lead, 3m", "quantity": 60, "unit_price": 300.00, "amount": 18_000.00},
            ),
            expected_findings=("due_date_after_issue",),
        ),
    ),
    (
        "invoice_vat_at_an_unlisted_rate", "pdf", True,
        SampleDoc(
            doc_id="invoice_vat_at_an_unlisted_rate",
            doc_type="invoice", layout="classic",
            title="Northwind invoice charging a VAT rate we do not expect",
            demonstrates="Charges 10%, which is arithmetically consistent and simply not a rate Thailand uses. A warning: an unexpected rate is a question for a person, not a wrong document.",
            ground_truth={
                "invoice_number": "NWL-2026-0802", "invoice_date": "2026-07-25",
                "due_date": "2026-08-24", "vendor_name": "Northwind Logistics Ltd.",
                "vendor_tax_id": "0105561071237", "buyer_name": BUYER,
                "currency": "THB", "subtotal": 20_000.00, "tax_rate": 10.0,
                "tax_amount": 2_000.00, "total_amount": 22_000.00,
                "payment_terms": "Net 30", "bank_account": "9876543210",
            },
            line_items=(
                {"description": "Customs brokerage, August", "quantity": 1, "unit_price": 20_000.00, "amount": 20_000.00},
            ),
            expected_findings=("vat_rate_expected",),
        ),
    ),
    (
        "contract_dates_out_of_order", "pdf", False,
        SampleDoc(
            doc_id="contract_dates_out_of_order",
            doc_type="contract", layout="prose",
            title="Meridian agreement that expires before it starts",
            demonstrates="Effective 2026-09-01, expiring 2026-04-30. Either date could be the misread one, so it blocks and names both rather than guessing.",
            ground_truth={
                "contract_number": "CT-2026-041", "party_a": BUYER,
                "party_b": "Meridian Consulting LLP",
                "effective_date": "2027-09-01", "expiry_date": "2027-04-30",
                "auto_renew": "no", "contract_value": 96_000.00, "currency": "USD",
            },
            body=(
                "This Advisory Services Agreement is made as of 1 September 2027 (the "
                "\"Effective Date\") between Mirtphol Industries PCL (\"Party A\") and Meridian "
                "Consulting LLP (\"Party B\").",
                "1. TERM. This Agreement commences on the Effective Date and expires on "
                "30 April 2027. It shall not renew automatically.",
                "2. CONTRACT VALUE. The total value for the term is USD 96,000.00.",
                "3. SCOPE. Party B shall provide operations advisory services to Party A on a "
                "time-and-materials basis against statements of work agreed in writing.",
            ),
            # Both, and both are right: a contract whose own dates are out
            # of order also disagrees with the register that holds them in
            # order. Declaring one would score the other as an over-fire.
            expected_findings=("contract_dates_ordered", "contract_matches_master"),
        ),
    ),
    (
        "contract_naming_one_party_twice", "pdf", False,
        SampleDoc(
            doc_id="contract_naming_one_party_twice",
            doc_type="contract", layout="prose",
            title="Agreement where both parties are the buyer",
            demonstrates="Party A and Party B are the same organisation. Nobody contracts with themselves, so one side was misread - and which side is not knowable from the page.",
            ground_truth={
                "contract_number": "CT-2026-046", "party_a": BUYER, "party_b": BUYER,
                "effective_date": "2026-05-01", "expiry_date": "2027-04-30",
                "auto_renew": "no", "contract_value": 210_000.00, "currency": "THB",
            },
            body=(
                "This Services Agreement is made as of 1 May 2026 between Mirtphol Industries "
                "PCL (\"Party A\") and Mirtphol Industries PCL (\"Party B\").",
                "1. TERM. This Agreement commences on 1 May 2026 and expires on 30 April 2027.",
                "2. CONTRACT VALUE. The total value for the term is THB 210,000.00.",
                "3. SCOPE. Party B shall provide maintenance services to Party A at the Rayong "
                "fabrication yard on a scheduled basis.",
            ),
            # As above: naming the buyer as both parties disagrees with the
            # register about `party_b` as well as with itself.
            expected_findings=("contract_parties_distinct", "contract_matches_master"),
        ),
    ),
    (
        "contract_disagreeing_with_the_register", "pdf", False,
        SampleDoc(
            doc_id="contract_disagreeing_with_the_register",
            doc_type="contract", layout="prose",
            title="CT-2026-011 with a value the register does not hold",
            demonstrates="The contract number is an exact key into the register, and everything it keys to disagrees: the value is 144,000 on file and 244,000 on the page.",
            ground_truth={
                "contract_number": "CT-2026-011", "party_a": BUYER,
                "party_b": "Meridian Consulting LLP",
                "effective_date": "2026-03-01", "expiry_date": "2027-02-28",
                "auto_renew": "no", "contract_value": 244_000.00, "currency": "USD",
            },
            body=(
                "This Consulting Agreement is made as of 1 March 2026 (the \"Effective Date\") "
                "between Mirtphol Industries PCL (\"Party A\") and Meridian Consulting LLP "
                "(\"Party B\"), and is registered under number CT-2026-011.",
                "1. TERM. This Agreement commences on the Effective Date and continues until "
                "28 February 2027. It shall not renew automatically.",
                "2. CONTRACT VALUE. The total value for the term is USD 244,000.00, payable "
                "monthly in arrears against approved statements of work.",
                "3. SCOPE. Party B shall provide operations advisory and process improvement "
                "services to Party A's manufacturing division.",
            ),
            expected_findings=("contract_matches_master",),
        ),
    ),
    (
        "form_reusing_a_registered_bank_account", "png", False,
        SampleDoc(
            doc_id="form_reusing_a_registered_bank_account",
            doc_type="form", layout="form",
            title="Onboarding form quoting an account already on file for someone else",
            demonstrates="A new supplier asking to be paid into Siam Tech's registered account. The signature of an invoice-redirection attempt, and the one form defect that blocks.",
            ground_truth={
                "form_type": "Vendor onboarding", "applicant_name": "Bangkok Precision Tools Ltd.",
                "tax_id": "0105566000419", "submitted_date": "2026-07-28",
                "contact_email": "accounts@bkkprecision.co.th",
                "contact_phone": "02-118-4420", "bank_account": "1234567890",
            },
            expected_findings=("bank_account_not_reused",),
        ),
    ),
    (
        "form_with_an_unusable_email", "png", True,
        SampleDoc(
            doc_id="form_with_an_unusable_email",
            doc_type="form", layout="form",
            title="Onboarding form whose contact address cannot be written to",
            demonstrates="`accounts@bkkfittings` has no domain suffix. A warning: nothing about the supplier is wrong, but nobody can reach them, and a form that blocks on a typo is a form nobody submits.",
            ground_truth={
                "form_type": "Vendor onboarding", "applicant_name": "Bangkok Fittings Co., Ltd.",
                "tax_id": "0105569000118", "submitted_date": "2026-07-29",
                "contact_email": "accounts@bkkfittings",
                "contact_phone": "02-119-7730", "bank_account": "2280046610",
            },
            expected_findings=("contact_email_valid",),
        ),
    ),
    (
        "invoice_thai_vendor_name_misread", "png", True,
        SampleDoc(
            doc_id="invoice_thai_vendor_name_misread",
            doc_type="invoice", layout="thai_tax", language="th",
            title="Thai invoice naming the vendor with one stroke wrong",
            demonstrates="ซ printed where the register has ช. The corpus document for the failure that produced two silent errors on 2026-08-06: everything reconciles, the vendor still resolves, and only the register knows the name is misspelt.",
            ground_truth={
                "invoice_number": "SNR-2026-0318", "invoice_date": "2026-07-23",
                "due_date": "2026-09-06", "vendor_name": "บริษัท ชันไรส์ อิเล็คทริคอล จำกัด",
                "buyer_name": BUYER, "currency": "THB",
                "subtotal": 45_000.00, "tax_rate": 7.0, "tax_amount": 3_150.00,
                "total_amount": 48_150.00, "payment_terms": "Net 45",
                "bank_account": "8890014455",
            },
            line_items=(
                {"description": "ตู้ควบคุมไฟฟ้าแรงต่ำ", "quantity": 3, "unit_price": 15_000.00, "amount": 45_000.00},
            ),
            expected_findings=("vendor_name_matches_register",),
        ),
    ),
    (
        "invoice_billed_in_the_wrong_currency_for_the_vendor", "pdf", True,
        SampleDoc(
            doc_id="invoice_billed_in_the_wrong_currency_for_the_vendor",
            doc_type="invoice", layout="classic",
            title="Meridian invoice in THB against a vendor registered in USD",
            demonstrates="Meridian is registered to bill in USD and this one is in THB. A warning: suppliers do change billing currency, and a rule that blocked would be wrong more often than right.",
            ground_truth={
                "invoice_number": "MC-2026-0611", "invoice_date": "2026-07-26",
                "due_date": "2026-08-25", "vendor_name": "Meridian Consulting LLP",
                "vendor_tax_id": "GB428765190", "buyer_name": BUYER,
                "currency": "THB", "subtotal": 30_000.00, "tax_rate": 0.0,
                "tax_amount": 0.00, "total_amount": 30_000.00,
                "payment_terms": "Net 30", "bank_account": "GB29NWBK60161331926819",
            },
            line_items=(
                {"description": "Operations advisory, July", "quantity": 1, "unit_price": 30_000.00, "amount": 30_000.00},
            ),
            expected_findings=("currency_matches_vendor_default",),
        ),
    ),
    (
        "contract_without_a_stated_value", "pdf", True,
        SampleDoc(
            doc_id="contract_without_a_stated_value",
            doc_type="contract", layout="prose",
            title="Framework agreement that names no total value",
            demonstrates="A genuine framework agreement: the value is whatever is ordered under it. A warning rather than a block, because this is a real way for a contract to be written.",
            ground_truth={
                "contract_number": "CT-2026-052", "party_a": BUYER,
                "party_b": "Northwind Logistics Ltd.",
                "effective_date": "2026-06-01", "expiry_date": "2027-05-31",
                "auto_renew": "no", "currency": "THB",
            },
            body=(
                "This Framework Logistics Agreement is made as of 1 June 2026 (the \"Effective "
                "Date\") between Mirtphol Industries PCL (\"Party A\") and Northwind Logistics "
                "Ltd. (\"Party B\").",
                "1. TERM. This Agreement commences on the Effective Date and expires on "
                "31 May 2027. It shall not renew automatically.",
                "2. CHARGES. No total value is committed under this Agreement. Party B shall "
                "invoice against the rate card at Schedule 1 for services actually ordered.",
                "3. SCOPE. Party B shall provide inbound and outbound freight forwarding to "
                "Party A on a call-off basis.",
            ),
            expected_findings=("contract_value_present",),
        ),
    ),
    (
        "form_from_a_vendor_already_on_file", "png", True,
        SampleDoc(
            doc_id="form_from_a_vendor_already_on_file",
            doc_type="form", layout="form",
            title="Onboarding form for a supplier already registered",
            demonstrates="Siam Tech is already V-1001. Usually a duplicate submission rather than fraud, so a warning - but it needs a person, because the alternative is two records for one supplier.",
            ground_truth={
                "form_type": "Vendor onboarding", "applicant_name": "Siam Tech Supply Co., Ltd.",
                "tax_id": "0105543000129", "submitted_date": "2026-07-30",
                "contact_email": "accounts@siamtech.co.th",
                "contact_phone": "02-115-2200", "bank_account": "1234567890",
            },
            expected_findings=("applicant_already_registered",),
        ),
    ),
    (
        "form_missing_its_payment_details", "png", True,
        SampleDoc(
            doc_id="form_missing_its_payment_details",
            doc_type="form", layout="form",
            title="Onboarding form with no bank account and an unusable phone number",
            demonstrates="Two warnings on one form, both about reaching or paying the supplier rather than about whether to trust them. Neither blocks: an incomplete form is chased, not refused.",
            ground_truth={
                "form_type": "Vendor onboarding", "applicant_name": "Lanna Toolworks Co., Ltd.",
                "tax_id": "0105567000218", "submitted_date": "2026-07-31",
                "contact_email": "finance@lannatoolworks.co.th",
                "contact_phone": "12",
            },
            expected_findings=("bank_details_complete", "contact_phone_valid"),
        ),
    ),
    (
        "invoice_in_a_currency_we_do_not_trade", "pdf", False,
        SampleDoc(
            doc_id="invoice_in_a_currency_we_do_not_trade",
            doc_type="invoice", layout="classic",
            title="Pacific Rim invoice denominated in JPY",
            demonstrates="JPY is not one of the four currencies this business settles in. A rule written here rather than built in, because which currencies a company trades is a business fact, not a property of invoices.",
            ground_truth={
                "invoice_number": "PRF-JP-30188", "invoice_date": "2026-07-27",
                "due_date": "2026-08-26", "vendor_name": "Pacific Rim Freight Pte. Ltd.",
                "vendor_tax_id": "SG198800123K", "buyer_name": BUYER,
                "currency": "JPY", "subtotal": 42_000.00, "tax_rate": 0.0,
                "tax_amount": 0.00, "total_amount": 42_000.00,
                "payment_terms": "Net 30", "bank_account": "SG21DBSS0110001234567",
            },
            line_items=(
                {"description": "Air freight, Osaka to Bangkok", "quantity": 1, "unit_price": 42_000.00, "amount": 42_000.00},
            ),
            expected_findings=("currency_is_one_we_trade_in",),
        ),
    ),
    (
        "invoice_with_a_line_over_the_ceiling", "pdf", False,
        SampleDoc(
            doc_id="invoice_with_a_line_over_the_ceiling",
            doc_type="invoice", layout="classic",
            title="Rungruang Steel invoice with a single 460,000 line",
            demonstrates="One line above the 400,000 ceiling this business set. Nothing is arithmetically wrong; the limit is a policy about how much may move on one line without a person.",
            ground_truth={
                "invoice_number": "RSW-2026-1290", "invoice_date": "2026-07-28",
                "due_date": "2026-09-26", "vendor_name": "Rungruang Steel Works Co., Ltd.",
                "vendor_tax_id": "0105552000110", "buyer_name": BUYER,
                "purchase_order_ref": "PO-2026-0501", "currency": "THB",
                "subtotal": 460_000.00, "tax_rate": 7.0, "tax_amount": 32_200.00,
                "total_amount": 492_200.00, "payment_terms": "Net 60",
                "bank_account": "3120045678",
            },
            line_items=(
                {"description": "Hot-rolled plate 12mm, full coil", "quantity": 1, "unit_price": 460_000.00, "amount": 460_000.00},
            ),
            expected_findings=("no_single_line_over_400k",),
        ),
    ),
    (
        "invoice_with_an_undescribed_line", "pdf", True,
        SampleDoc(
            doc_id="invoice_with_an_undescribed_line",
            doc_type="invoice", layout="classic",
            title="Siam Tech invoice with a line that says nothing",
            demonstrates="A charge with an amount and no description. Cannot be coded to an account, so somebody has to ask - a warning, because the invoice may still be perfectly good.",
            ground_truth={
                "invoice_number": "STS-2026-0655", "invoice_date": "2026-07-29",
                "due_date": "2026-08-28", "vendor_name": "Siam Tech Supply Co., Ltd.",
                "vendor_tax_id": "0105543000129", "buyer_name": BUYER,
                "currency": "THB", "subtotal": 26_000.00, "tax_rate": 7.0,
                "tax_amount": 1_820.00, "total_amount": 27_820.00,
                "payment_terms": "Net 30", "bank_account": "1234567890",
            },
            line_items=(
                {"description": "Network switch, 24 port", "quantity": 2, "unit_price": 9_000.00, "amount": 18_000.00},
                {"description": "", "quantity": 1, "unit_price": 8_000.00, "amount": 8_000.00},
            ),
            expected_findings=("every_line_item_described",),
        ),
    ),
    (
        "invoice_stating_no_payment_terms", "pdf", True,
        SampleDoc(
            doc_id="invoice_stating_no_payment_terms",
            doc_type="invoice", layout="classic",
            title="Northwind invoice that never says when it is due",
            demonstrates="No payment terms anywhere on the page. The ledger can still hold it; nobody can schedule it. A warning that exists because this business asked for it.",
            ground_truth={
                "invoice_number": "NWL-2026-0915", "invoice_date": "2026-07-30",
                "vendor_name": "Northwind Logistics Ltd.",
                "vendor_tax_id": "0105561071237", "buyer_name": BUYER,
                "currency": "THB", "subtotal": 12_000.00, "tax_rate": 7.0,
                "tax_amount": 840.00, "total_amount": 12_840.00,
                "bank_account": "9876543210",
            },
            line_items=(
                {"description": "Courier, July", "quantity": 1, "unit_price": 12_000.00, "amount": 12_000.00},
            ),
            expected_findings=("payment_terms_stated",),
        ),
    ),
    (
        "invoice_large_with_no_buyer_named", "pdf", True,
        SampleDoc(
            doc_id="invoice_large_with_no_buyer_named",
            doc_type="invoice", layout="classic",
            title="A 214,000 invoice addressed to nobody in particular",
            demonstrates="Above 100,000 and naming no buyer. Below that threshold this business does not care; above it, an invoice that does not say who owes the money is one to look at.",
            ground_truth={
                "invoice_number": "STS-2026-0701", "invoice_date": "2026-07-31",
                "due_date": "2026-08-30", "vendor_name": "Siam Tech Supply Co., Ltd.",
                "vendor_tax_id": "0105543000129", "currency": "THB",
                "purchase_order_ref": "PO-2026-0412",
                "subtotal": 200_000.00, "tax_rate": 7.0, "tax_amount": 14_000.00,
                "total_amount": 214_000.00, "payment_terms": "Net 30",
                "bank_account": "1234567890",
            },
            line_items=(
                {"description": "Data centre rack build-out", "quantity": 1, "unit_price": 200_000.00, "amount": 200_000.00},
            ),
            expected_findings=("large_invoice_names_a_buyer",),
        ),
    ),
    (
        "invoice_quoting_another_vendors_tax_id", "pdf", False,
        SampleDoc(
            doc_id="invoice_quoting_another_vendors_tax_id",
            doc_type="invoice", layout="classic",
            title="Siam Tech invoice carrying Northwind's tax ID",
            demonstrates="A valid Thai tax ID, correct checksum, registered - to a different supplier. `find_vendor` takes the tax ID as the strongest evidence there is and resolves to Northwind, so the bank account then belongs to the wrong vendor too. Both findings are correct and the second is the consequence of the first.",
            ground_truth={
                "invoice_number": "STS-2026-0733", "invoice_date": "2026-07-31",
                "due_date": "2026-08-30", "vendor_name": "Siam Tech Supply Co., Ltd.",
                "vendor_tax_id": "0105561071237", "buyer_name": BUYER,
                "currency": "THB", "subtotal": 15_000.00, "tax_rate": 7.0,
                "tax_amount": 1_050.00, "total_amount": 16_050.00,
                "payment_terms": "Net 30", "bank_account": "1234567890",
            },
            line_items=(
                {"description": "Cable tray, 3m section", "quantity": 30, "unit_price": 500.00, "amount": 15_000.00},
            ),
            # Both, in order of cause. A tax ID is an exact identifier, so
            # quoting someone else's does not merely mislabel the document -
            # it resolves the whole document to the other supplier, and the
            # account on the page then matches nobody.
            expected_findings=("tax_id_belongs_to_this_vendor", "bank_account_matches_vendor"),
        ),
    ),
)


ALL_DOCUMENTS: tuple[tuple[str, Arrival, bool, SampleDoc], ...] = (
    REAL_DOCUMENTS + VARIED_DOCUMENTS + THAI_DOCUMENTS + COVERAGE_DOCUMENTS
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
