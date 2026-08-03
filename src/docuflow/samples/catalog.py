"""The demo corpus: varied documents plus the master data they reconcile against.

Each entry carries its own ground truth, so the same catalogue drives the demo,
the regression tests and the accuracy report. Several documents contain planted
defects (arithmetic errors, unknown vendors, duplicates, stale contracts) - those
exist to prove the validation layer actually catches them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

LayoutStyle = Literal["classic", "modern", "thai_tax", "dense", "prose", "form", "narrow"]


@dataclass(frozen=True)
class SampleDoc:
    """One demo document: how to render it and what the right answer is."""

    doc_id: str
    doc_type: str
    layout: LayoutStyle
    title: str
    # What this sample is meant to demonstrate, shown in the UI.
    demonstrates: str
    ground_truth: dict[str, Any]
    line_items: tuple[dict[str, Any], ...] = ()
    body: tuple[str, ...] = ()
    # Simulated scan damage: 0 = clean digital PDF, 1 = badly photocopied.
    degrade: float = 0.0
    language: Literal["en", "th"] = "en"
    # Rule IDs this document is expected to trip.
    expected_findings: tuple[str, ...] = ()
    # A defect that only exists relative to an earlier posting: this document
    # can only be a duplicate if `requires_posted` reached the ledger first.
    # Evaluation seeds that record so engine comparisons stay fair.
    requires_posted: str | None = None


VENDORS: tuple[dict[str, Any], ...] = (
    {
        "vendor_id": "V-1001",
        "legal_name": "Siam Tech Supply Co., Ltd.",
        "aliases": "Siam Tech;Siam Tech Supply;บริษัท สยามเทค ซัพพลาย จำกัด",
        "tax_id": "0105543000129",
        "bank_account": "1234567890",
        "status": "active",
        "payment_terms": "Net 30",
        "currency": "THB",
    },
    {
        "vendor_id": "V-1002",
        "legal_name": "Northwind Logistics Ltd.",
        "aliases": "Northwind;Northwind Logistics",
        "tax_id": "0105561071237",
        "bank_account": "9876543210",
        "status": "active",
        "payment_terms": "Net 45",
        "currency": "THB",
    },
    {
        "vendor_id": "V-1003",
        "legal_name": "Bluepeak Software GmbH",
        "aliases": "Bluepeak;Bluepeak Software",
        "tax_id": "DE811907980",
        "bank_account": "DE89370400440532013000",
        "status": "active",
        "payment_terms": "Net 30",
        "currency": "EUR",
    },
    {
        "vendor_id": "V-1004",
        "legal_name": "Chao Phraya Facility Services Co., Ltd.",
        "aliases": "Chao Phraya Facility;CPFS;บริษัท เจ้าพระยา แฟซิลิตี้ จำกัด",
        "tax_id": "0994000165111",
        "bank_account": "5550001111",
        "status": "active",
        "payment_terms": "Net 15",
        "currency": "THB",
    },
    {
        "vendor_id": "V-1005",
        "legal_name": "Orient Paper & Print Co., Ltd.",
        "aliases": "Orient Paper;Orient Print",
        "tax_id": "0105539000117",
        "bank_account": "4440002222",
        "status": "on_hold",
        "payment_terms": "Net 30",
        "currency": "THB",
    },
)

PURCHASE_ORDERS: tuple[dict[str, Any], ...] = (
    {
        "po_number": "PO-2026-0412",
        "vendor_id": "V-1001",
        "po_date": "2026-06-02",
        "currency": "THB",
        "amount": 187_000.00,
        "status": "open",
        "description": "Dell Latitude laptops and docking stations",
    },
    {
        "po_number": "PO-2026-0455",
        "vendor_id": "V-1002",
        "po_date": "2026-06-18",
        "currency": "THB",
        "amount": 64_000.00,
        "status": "open",
        "description": "Inbound freight, June",
    },
    {
        "po_number": "PO-2026-0470",
        "vendor_id": "V-1003",
        "po_date": "2026-07-01",
        "currency": "EUR",
        "amount": 24_000.00,
        "status": "open",
        "description": "Annual platform licence renewal",
    },
    {
        "po_number": "PO-2026-0488",
        "vendor_id": "V-1004",
        "po_date": "2026-07-05",
        "currency": "THB",
        "amount": 92_000.00,
        "status": "open",
        "description": "Quarterly cleaning and facility services",
    },
)

CONTRACTS: tuple[dict[str, Any], ...] = (
    {
        "contract_number": "CT-2025-018",
        "vendor_id": "V-1004",
        "party_a": "Mirtphol Industries PCL",
        "party_b": "Chao Phraya Facility Services Co., Ltd.",
        "effective_date": "2025-08-01",
        "expiry_date": "2026-07-31",
        "value": 368_000.00,
        "currency": "THB",
    },
    {
        "contract_number": "CT-2026-004",
        "vendor_id": "V-1003",
        "party_a": "Mirtphol Industries PCL",
        "party_b": "Bluepeak Software GmbH",
        "effective_date": "2026-01-15",
        "expiry_date": "2027-01-14",
        "value": 24_000.00,
        "currency": "EUR",
    },
)

BUYER = "Mirtphol Industries PCL"
BUYER_ADDRESS = "88/9 Rama IV Road, Khlong Toei, Bangkok 10110, Thailand"


SAMPLE_DOCUMENTS: tuple[SampleDoc, ...] = (
    # ---- 1. Clean English invoice, everything reconciles --------------------
    SampleDoc(
        doc_id="inv-clean-classic",
        doc_type="invoice",
        layout="classic",
        title="Siam Tech invoice matching PO-2026-0412",
        demonstrates="Happy path: every field grounded, PO matches, posts automatically.",
        ground_truth={
            "invoice_number": "INV-2026-0731",
            "invoice_date": "2026-07-08",
            "due_date": "2026-08-07",
            "vendor_name": "Siam Tech Supply Co., Ltd.",
            "vendor_tax_id": "0105543000129",
            "buyer_name": BUYER,
            "purchase_order_ref": "PO-2026-0412",
            "currency": "THB",
            "subtotal": 187_000.00,
            "tax_rate": 7.0,
            "tax_amount": 13_090.00,
            "total_amount": 200_090.00,
            "payment_terms": "Net 30",
            "bank_account": "1234567890",
        },
        line_items=(
            {"description": "Dell Latitude 5450 laptop", "quantity": 5, "unit_price": 32_000.00, "amount": 160_000.00},
            {"description": "USB-C docking station", "quantity": 5, "unit_price": 5_400.00, "amount": 27_000.00},
        ),
    ),
    # ---- 2. Arithmetic error: subtotal + VAT != total -----------------------
    SampleDoc(
        doc_id="inv-bad-math",
        doc_type="invoice",
        layout="modern",
        title="Northwind freight invoice with a transposed total",
        demonstrates="Document states a total of 86,480.00 but 64,000.00 + 7% VAT is 68,480.00 - an 18,000 error, blocked before posting.",
        ground_truth={
            "invoice_number": "NW-88214",
            "invoice_date": "2026-07-11",
            "due_date": "2026-08-25",
            "vendor_name": "Northwind Logistics Ltd.",
            "vendor_tax_id": "0105561071237",
            "buyer_name": BUYER,
            "purchase_order_ref": "PO-2026-0455",
            "currency": "THB",
            "subtotal": 64_000.00,
            "tax_rate": 7.0,
            "tax_amount": 4_480.00,
            # Correct value is 68,480.00 - the document states 86,480.00.
            "total_amount": 86_480.00,
            "payment_terms": "Net 45",
            "bank_account": "9876543210",
        },
        line_items=(
            {"description": "Inbound freight - Laem Chabang to Bangkok", "quantity": 8, "unit_price": 6_500.00, "amount": 52_000.00},
            {"description": "Customs handling fee", "quantity": 4, "unit_price": 3_000.00, "amount": 12_000.00},
        ),
        expected_findings=("invoice_totals_consistent",),
    ),
    # ---- 3. Thai tax invoice, Buddhist-era dates, degraded scan -------------
    SampleDoc(
        doc_id="inv-thai-scan",
        doc_type="invoice",
        layout="thai_tax",
        language="th",
        degrade=0.55,
        title="ใบกำกับภาษี - เจ้าพระยา แฟซิลิตี้ (สแกน)",
        demonstrates="Thai-language scanned tax invoice with Buddhist-era dates and Thai numerals.",
        ground_truth={
            "invoice_number": "CP-2569-0142",
            "invoice_date": "2026-07-15",
            "due_date": "2026-07-30",
            "vendor_name": "บริษัท เจ้าพระยา แฟซิลิตี้ จำกัด",
            "vendor_tax_id": "0994000165111",
            "buyer_name": BUYER,
            "purchase_order_ref": "PO-2026-0488",
            "currency": "THB",
            "subtotal": 92_000.00,
            "tax_rate": 7.0,
            "tax_amount": 6_440.00,
            "total_amount": 98_440.00,
            "payment_terms": "Net 15",
            "bank_account": "5550001111",
        },
        line_items=(
            {"description": "บริการทำความสะอาด ไตรมาส 3/2569", "quantity": 1, "unit_price": 92_000.00, "amount": 92_000.00},
        ),
    ),
    # ---- 4. Unknown vendor + bank account mismatch (fraud pattern) ----------
    SampleDoc(
        doc_id="inv-unknown-vendor",
        doc_type="invoice",
        layout="dense",
        degrade=0.25,
        title="Invoice from a vendor not in the master file",
        demonstrates="Unregistered vendor and an unrecognised bank account - the classic invoice-fraud pattern.",
        ground_truth={
            "invoice_number": "SS-4471",
            "invoice_date": "2026-07-19",
            "due_date": "2026-07-26",
            "vendor_name": "Summit Solutions Trading",
            "vendor_tax_id": "0105562099887",
            "buyer_name": BUYER,
            "purchase_order_ref": None,
            "currency": "THB",
            "subtotal": 145_000.00,
            "tax_rate": 7.0,
            "tax_amount": 10_150.00,
            "total_amount": 155_150.00,
            "payment_terms": "Due on receipt",
            "bank_account": "7778889990",
        },
        line_items=(
            {"description": "IT consulting services - July", "quantity": 1, "unit_price": 145_000.00, "amount": 145_000.00},
        ),
        expected_findings=("vendor_known", "purchase_order_required"),
    ),
    # ---- 5. Duplicate of sample 1 (re-sent invoice) -------------------------
    SampleDoc(
        doc_id="inv-duplicate",
        doc_type="invoice",
        layout="modern",
        title="The same Siam Tech invoice sent a second time",
        demonstrates="Duplicate detection: identical invoice number and vendor as a record already posted.",
        ground_truth={
            "invoice_number": "INV-2026-0731",
            "invoice_date": "2026-07-08",
            "due_date": "2026-08-07",
            "vendor_name": "Siam Tech Supply Co., Ltd.",
            "vendor_tax_id": "0105543000129",
            "buyer_name": BUYER,
            "purchase_order_ref": "PO-2026-0412",
            "currency": "THB",
            "subtotal": 187_000.00,
            "tax_rate": 7.0,
            "tax_amount": 13_090.00,
            "total_amount": 200_090.00,
            "payment_terms": "Net 30",
            "bank_account": "1234567890",
        },
        line_items=(
            {"description": "Dell Latitude 5450 laptop", "quantity": 5, "unit_price": 32_000.00, "amount": 160_000.00},
            {"description": "USB-C docking station", "quantity": 5, "unit_price": 5_400.00, "amount": 27_000.00},
        ),
        expected_findings=("duplicate_document",),
        requires_posted="inv-clean-classic",
    ),
    # ---- 6. Foreign currency invoice over PO amount ------------------------
    SampleDoc(
        doc_id="inv-eur-over-po",
        doc_type="invoice",
        layout="classic",
        title="EUR software renewal billed above the PO",
        demonstrates="Foreign currency plus a PO overrun outside tolerance - needs a human decision.",
        ground_truth={
            "invoice_number": "BP-2026-1187",
            "invoice_date": "2026-07-03",
            "due_date": "2026-08-02",
            "vendor_name": "Bluepeak Software GmbH",
            "vendor_tax_id": "DE811907980",
            "buyer_name": BUYER,
            "purchase_order_ref": "PO-2026-0470",
            "currency": "EUR",
            "subtotal": 27_500.00,
            "tax_rate": 0.0,
            "tax_amount": 0.00,
            "total_amount": 27_500.00,
            "payment_terms": "Net 30",
            "bank_account": "DE89370400440532013000",
        },
        line_items=(
            {"description": "Bluepeak Platform - annual licence, 250 seats", "quantity": 1, "unit_price": 24_000.00, "amount": 24_000.00},
            {"description": "Additional 40 seats (mid-term)", "quantity": 40, "unit_price": 87.50, "amount": 3_500.00},
        ),
        expected_findings=("po_amount_within_tolerance",),
    ),
    # ---- 7. Service contract -----------------------------------------------
    SampleDoc(
        doc_id="contract-facility",
        doc_type="contract",
        layout="prose",
        title="Facility services agreement (renewal window open)",
        demonstrates="Multi-page prose contract: parties, dates, value and notice period pulled from paragraphs.",
        ground_truth={
            "contract_number": "CT-2025-018",
            "contract_title": "Facility Services Agreement",
            "contract_type": "service",
            "party_a": "Mirtphol Industries PCL",
            "party_b": "Chao Phraya Facility Services Co., Ltd.",
            "effective_date": "2025-08-01",
            "expiry_date": "2026-07-31",
            "contract_value": 368_000.00,
            "currency": "THB",
            "notice_period_days": 60,
            "auto_renew": "yes",
            "governing_law": "Thailand",
            "payment_terms": "Quarterly in arrears, Net 15",
        },
        body=(
            "This Facility Services Agreement (the \"Agreement\") is made and entered into as of "
            "1 August 2025 (the \"Effective Date\") by and between Mirtphol Industries PCL, a public "
            "company limited organised under the laws of Thailand with its registered office at "
            f"{BUYER_ADDRESS} (\"Party A\"), and Chao Phraya Facility Services Co., Ltd., a company "
            "limited organised under the laws of Thailand (\"Party B\").",
            "1. SCOPE OF SERVICES. Party B shall provide cleaning, waste management, landscaping and "
            "general facility maintenance services at Party A's Bangkok headquarters and its Rayong "
            "warehouse, in accordance with the service levels set out in Schedule 1.",
            "2. TERM. This Agreement shall commence on the Effective Date and shall continue in force "
            "until 31 July 2026, unless terminated earlier in accordance with Clause 6. The Agreement "
            "shall renew automatically for successive periods of twelve (12) months unless either "
            "party gives written notice of non-renewal not less than sixty (60) days before expiry.",
            "3. CONTRACT VALUE. In consideration of the Services, Party A shall pay Party B a total "
            "contract value of THB 368,000.00 (three hundred sixty-eight thousand Baht) for the "
            "initial term, invoiced quarterly in arrears with payment due within fifteen (15) days "
            "of receipt of a valid tax invoice.",
            "4. CONFIDENTIALITY. Each party shall keep confidential all information disclosed by the "
            "other party in connection with this Agreement and shall not disclose it to any third "
            "party without prior written consent, save as required by law.",
            "5. INSURANCE. Party B shall maintain public liability insurance of not less than THB "
            "10,000,000 per occurrence throughout the Term and shall provide certificates of "
            "insurance to Party A upon request.",
            "6. TERMINATION. Either party may terminate this Agreement for material breach that "
            "remains unremedied for thirty (30) days after written notice. Party A may terminate for "
            "convenience upon sixty (60) days' written notice.",
            "7. GOVERNING LAW. This Agreement shall be governed by and construed in accordance with "
            "the laws of Thailand, and the parties submit to the exclusive jurisdiction of the courts "
            "of Bangkok.",
        ),
        expected_findings=("contract_expiring_soon",),
    ),
    # ---- 8. Vendor onboarding form -----------------------------------------
    SampleDoc(
        doc_id="form-vendor-onboarding",
        doc_type="form",
        layout="form",
        degrade=0.35,
        title="Vendor onboarding form (hand-filled, scanned)",
        demonstrates="Semi-structured form with checkboxes and an invalid tax-ID checksum.",
        ground_truth={
            "form_type": "Vendor Onboarding",
            "reference_number": "VOF-2026-0219",
            "submitted_date": "2026-07-21",
            "applicant_name": "Andaman Marine Supply Co., Ltd.",
            # Deliberately fails the Thai check-digit test.
            "tax_id": "0105545000124",
            "contact_email": "accounts@andamanmarine.co.th",
            "contact_phone": "+66 2 555 0188",
            "address": "212 Sathorn Road, Sathorn, Bangkok 10120",
            "bank_account": "3216549870",
            "requested_amount": None,
            "signed_by": "Kittipong Sae-Lim",
        },
        expected_findings=("tax_id_checksum",),
    ),
    # ---- 9. Receipt ---------------------------------------------------------
    SampleDoc(
        doc_id="receipt-fuel",
        doc_type="receipt",
        layout="narrow",
        degrade=0.4,
        title="Fuel receipt (thermal print)",
        demonstrates="Small, low-contrast thermal receipt - the hardest OCR case in the corpus.",
        ground_truth={
            "receipt_number": "R-770412",
            "receipt_date": "2026-07-17",
            "merchant_name": "PTT Station Rama IV",
            "merchant_tax_id": "0107544000108",
            "currency": "THB",
            "subtotal": 1_401.87,
            "tax_amount": 98.13,
            "total_amount": 1_500.00,
            "payment_method": "credit_card",
            "expense_category": "Fuel",
        },
        line_items=(
            {"description": "Diesel B7", "quantity": 40.0, "unit_price": 37.50, "amount": 1_500.00},
        ),
    ),
    # ---- 10. Purchase order -------------------------------------------------
    SampleDoc(
        doc_id="po-orient-paper",
        doc_type="purchase_order",
        layout="classic",
        title="Purchase order raised against an on-hold vendor",
        demonstrates="Commitment document where the vendor's master record is on hold.",
        ground_truth={
            "po_number": "PO-2026-0501",
            "po_date": "2026-07-22",
            "vendor_name": "Orient Paper & Print Co., Ltd.",
            "buyer_name": BUYER,
            "currency": "THB",
            "total_amount": 48_150.00,
            "delivery_date": "2026-08-12",
        },
        line_items=(
            {"description": "A4 copier paper, 80gsm (carton of 5 reams)", "quantity": 150, "unit_price": 285.00, "amount": 42_750.00},
            {"description": "Letterhead print run", "quantity": 1, "unit_price": 5_400.00, "amount": 5_400.00},
        ),
        expected_findings=("vendor_active",),
    ),
)


def master_data_rows() -> dict[str, tuple[dict[str, Any], ...]]:
    """Master data seeded into the demo database."""
    return {"vendors": VENDORS, "purchase_orders": PURCHASE_ORDERS, "contracts": CONTRACTS}


def sample_by_id(doc_id: str) -> SampleDoc | None:
    return next((doc for doc in SAMPLE_DOCUMENTS if doc.doc_id == doc_id), None)
