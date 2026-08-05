"""Field schemas per document type.

One declaration drives four things at once: the extraction prompt, the regex
fallback extractor, the normalizer, and the review UI. Adding a new document
type means adding a `DocumentSchema` here - no other file needs to change.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FieldType = Literal["string", "text", "number", "money", "date", "percent", "tax_id", "enum"]


class FieldSpec(BaseModel):
    """Definition of one field the business needs off a document."""

    model_config = ConfigDict(frozen=True)

    name: str
    label: str
    type: FieldType = "string"
    required: bool = False
    # Money-bearing or identity fields: a wrong value here is expensive, so they
    # get stricter grounding and always block auto-post when unverified.
    critical: bool = False
    description: str = ""
    # Label variants seen on real documents, used by the regex fallback and as
    # hints in the extraction prompt. Thai + English.
    aliases: tuple[str, ...] = ()
    choices: tuple[str, ...] = ()
    # What each choice actually looks like on a document. A receipt says
    # "PAID BY VISA **** 4412", never "credit_card"; a fuel receipt says
    # "Diesel B7", never "fuel". Without these the value is rejected as "not one
    # of cash, credit_card, ..." - which is the schema failing to describe the
    # documents, reported as if the reading were wrong.
    #
    # They live here so the vocabulary of a document type is one editable table
    # rather than conditionals spread through the normalizer.
    choice_aliases: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @property
    def prompt_line(self) -> str:
        bits = [f'"{self.name}"', f"({self.type})", self.description or self.label]
        if self.choices:
            bits.append(f"one of: {', '.join(self.choices)}")
        if self.aliases:
            bits.append(f"labelled like: {', '.join(self.aliases[:6])}")
        return " - ".join(b for b in bits if b)


class DocumentSchema(BaseModel):
    """A document type and the fields extracted from it."""

    model_config = ConfigDict(frozen=True)

    doc_type: str
    display_name: str
    description: str
    fields: tuple[FieldSpec, ...]
    has_line_items: bool = False
    # Which downstream record table this posts into.
    record_kind: str = "generic"
    # Distinctive keywords used by the cheap pre-classifier.
    keywords: tuple[str, ...] = ()

    def field(self, name: str) -> FieldSpec | None:
        return next((f for f in self.fields if f.name == name), None)

    @property
    def required_fields(self) -> tuple[FieldSpec, ...]:
        return tuple(f for f in self.fields if f.required)

    @property
    def critical_fields(self) -> tuple[FieldSpec, ...]:
        return tuple(f for f in self.fields if f.critical)


_INVOICE = DocumentSchema(
    doc_type="invoice",
    display_name="Invoice / Tax Invoice",
    description="A supplier bill requesting payment.",
    record_kind="payable",
    has_line_items=True,
    keywords=(
        "invoice", "tax invoice", "bill to", "amount due", "vat", "subtotal",
        "ใบแจ้งหนี้", "ใบกำกับภาษี", "ยอดรวม", "ภาษีมูลค่าเพิ่ม",
    ),
    fields=(
        FieldSpec(
            name="invoice_number", label="Invoice No.", required=True, critical=True,
            description="The supplier's unique invoice identifier",
            aliases=("Invoice No", "Invoice #", "Tax Invoice No", "เลขที่ใบแจ้งหนี้", "เลขที่"),
        ),
        FieldSpec(
            name="invoice_date", label="Invoice Date", type="date", required=True,
            description="Date the invoice was issued",
            aliases=("Invoice Date", "Date", "Issued", "วันที่", "ลงวันที่"),
        ),
        FieldSpec(
            name="due_date", label="Due Date", type="date",
            description="Date payment is due",
            # Deliberately no "Terms": that labels payment terms ("Net 30"),
            # which is text, not a date.
            aliases=("Due Date", "Payment Due", "ครบกำหนด", "กำหนดชำระ"),
        ),
        FieldSpec(
            name="vendor_name", label="Vendor", required=True, critical=True,
            description="Legal name of the company issuing the invoice (the seller, not the buyer)",
            aliases=("From", "Seller", "Supplier", "Vendor", "ผู้ขาย", "ผู้ให้บริการ"),
        ),
        FieldSpec(
            name="vendor_tax_id", label="Vendor Tax ID", type="tax_id",
            description="Seller's tax identification number (13 digits in Thailand)",
            aliases=("Tax ID", "TIN", "VAT No", "เลขประจำตัวผู้เสียภาษี"),
        ),
        FieldSpec(
            name="buyer_name", label="Bill To",
            description="The company being billed (the buyer)",
            aliases=("Bill To", "To", "Customer", "ผู้ซื้อ", "ลูกค้า"),
        ),
        FieldSpec(
            name="purchase_order_ref", label="PO Reference", critical=True,
            description="Purchase order number this invoice bills against",
            aliases=("PO", "P.O.", "PO No", "Purchase Order", "ใบสั่งซื้อ", "เลขที่ PO"),
        ),
        FieldSpec(
            name="currency", label="Currency", type="enum",
            description="ISO currency code of the amounts",
            choices=("THB", "USD", "EUR", "JPY", "SGD", "GBP", "CNY"),
            aliases=("Currency", "สกุลเงิน"),
        ),
        FieldSpec(
            name="subtotal", label="Subtotal", type="money", critical=True,
            description="Total before tax",
            aliases=("Subtotal", "Sub-total", "Amount before VAT", "รวมเป็นเงิน", "ราคารวม"),
        ),
        FieldSpec(
            name="tax_rate", label="Tax Rate", type="percent",
            description="VAT/GST percentage applied",
            aliases=("VAT", "VAT 7%", "GST", "Tax Rate", "ภาษีมูลค่าเพิ่ม"),
        ),
        FieldSpec(
            name="tax_amount", label="Tax Amount", type="money", critical=True,
            description="Total tax charged",
            aliases=("VAT", "Tax", "VAT Amount", "ภาษีมูลค่าเพิ่ม", "ภาษี"),
        ),
        FieldSpec(
            name="total_amount", label="Total", type="money", required=True, critical=True,
            description="Grand total payable including tax",
            aliases=("Total", "Grand Total", "Amount Due", "Total Due", "ยอดรวมทั้งสิ้น", "รวมทั้งสิ้น", "จำนวนเงินรวม"),
        ),
        FieldSpec(
            name="payment_terms", label="Payment Terms",
            description="Agreed payment terms, e.g. Net 30",
            aliases=("Terms", "Payment Terms", "เงื่อนไขการชำระเงิน"),
        ),
        FieldSpec(
            name="bank_account", label="Bank Account", critical=True,
            description="Bank account number given for payment",
            aliases=("Account No", "Bank Account", "A/C", "เลขที่บัญชี"),
        ),
    ),
)

_RECEIPT = DocumentSchema(
    doc_type="receipt",
    display_name="Receipt",
    description="Proof that a payment was already made.",
    record_kind="expense",
    has_line_items=True,
    keywords=(
        "receipt", "paid", "cash", "change",
        "ใบเสร็จ", "ใบเสร็จรับเงิน", "ใบรับเงิน", "เงินสด", "เงินทอน",
        # The Revenue Department's term for the slip a till prints: an
        # *abbreviated* tax invoice. It is a receipt, and it is the heading on
        # most Thai retail receipts - so it belongs here and not with invoices,
        # even though `ใบกำกับภาษี` is a substring of it.
        "ใบกำกับภาษีอย่างย่อ",
    ),
    fields=(
        FieldSpec(name="receipt_number", label="Receipt No.", required=True, critical=True,
                  aliases=("Receipt No", "No.", "เลขที่ใบเสร็จ")),
        FieldSpec(name="receipt_date", label="Date", type="date", required=True,
                  aliases=("Date", "วันที่")),
        FieldSpec(name="merchant_name", label="Merchant", required=True, critical=True,
                  aliases=("Merchant", "Store", "ร้าน", "ผู้ขาย")),
        FieldSpec(name="merchant_tax_id", label="Tax ID", type="tax_id",
                  aliases=("Tax ID", "เลขประจำตัวผู้เสียภาษี")),
        FieldSpec(name="currency", label="Currency", type="enum",
                  choices=("THB", "USD", "EUR", "JPY", "SGD", "GBP", "CNY")),
        FieldSpec(name="subtotal", label="Subtotal", type="money", critical=True,
                  aliases=("Subtotal", "รวมเป็นเงิน")),
        FieldSpec(name="tax_amount", label="Tax", type="money", critical=True,
                  aliases=("VAT", "Tax", "ภาษี")),
        FieldSpec(name="total_amount", label="Total", type="money", required=True, critical=True,
                  aliases=("Total", "Grand Total", "รวมทั้งสิ้น")),
        FieldSpec(name="payment_method", label="Payment Method", type="enum",
                  choices=("cash", "credit_card", "bank_transfer", "qr_promptpay", "other"),
                  choice_aliases={
                      "cash": ("cash", "เงินสด", "ชำระเงินสด"),
                      "credit_card": ("visa", "mastercard", "master card", "amex",
                                      "american express", "jcb", "unionpay", "debit",
                                      "credit", "card", "บัตรเครดิต", "บัตร"),
                      "bank_transfer": ("transfer", "wire", "bank", "โอน", "โอนเงิน"),
                      "qr_promptpay": ("promptpay", "prompt pay", "qr", "พร้อมเพย์"),
                  },
                  aliases=("Payment", "Paid by", "ชำระโดย")),
        FieldSpec(name="expense_category", label="Category", type="enum",
                  choices=("fuel", "travel", "meals", "office_supplies", "utilities",
                           "maintenance", "professional_services", "other"),
                  choice_aliases={
                      "fuel": ("diesel", "gasoline", "petrol", "gasohol", "benzine",
                               "b7", "e20", "e85", "ngv", "lpg", "น้ำมัน", "ดีเซล", "เบนซิน"),
                      "travel": ("taxi", "grab", "flight", "airfare", "hotel", "toll",
                                 "parking", "เดินทาง", "แท็กซี่", "ที่พัก"),
                      "meals": ("food", "restaurant", "catering", "coffee", "beverage",
                                "อาหาร", "เครื่องดื่ม"),
                      "office_supplies": ("stationery", "paper", "printer", "toner",
                                          "เครื่องเขียน", "กระดาษ"),
                      "utilities": ("electricity", "water", "internet", "telephone",
                                    "mobile", "ค่าไฟ", "ค่าน้ำ", "อินเทอร์เน็ต"),
                      "maintenance": ("repair", "service", "cleaning", "ซ่อม", "บำรุงรักษา"),
                      "professional_services": ("consulting", "legal", "audit", "advisory",
                                                "ที่ปรึกษา", "ตรวจสอบบัญชี"),
                  },
                  description="Best-guess expense category for accounting"),
    ),
)

_PURCHASE_ORDER = DocumentSchema(
    doc_type="purchase_order",
    display_name="Purchase Order",
    description="A buyer's order committing to purchase goods or services.",
    record_kind="commitment",
    has_line_items=True,
    keywords=("purchase order", "p.o.", "po number", "ship to", "ใบสั่งซื้อ"),
    fields=(
        FieldSpec(name="po_number", label="PO Number", required=True, critical=True,
                  aliases=("PO No", "Purchase Order No", "เลขที่ใบสั่งซื้อ")),
        FieldSpec(name="po_date", label="PO Date", type="date", required=True,
                  aliases=("Date", "Order Date", "วันที่")),
        FieldSpec(name="vendor_name", label="Vendor", required=True, critical=True,
                  aliases=("Vendor", "Supplier", "ผู้ขาย")),
        FieldSpec(name="buyer_name", label="Buyer", aliases=("Buyer", "Ship To", "ผู้ซื้อ")),
        FieldSpec(name="currency", label="Currency", type="enum",
                  choices=("THB", "USD", "EUR", "JPY", "SGD", "GBP", "CNY")),
        FieldSpec(name="total_amount", label="Total", type="money", required=True, critical=True,
                  aliases=("Total", "Order Total", "รวมทั้งสิ้น")),
        FieldSpec(name="delivery_date", label="Delivery Date", type="date",
                  aliases=("Delivery", "Required By", "กำหนดส่ง")),
    ),
)

_CONTRACT = DocumentSchema(
    doc_type="contract",
    display_name="Contract / Agreement",
    description="A signed agreement between two parties.",
    record_kind="contract",
    keywords=(
        "agreement", "contract", "party", "hereby", "term of this agreement",
        "witnesseth", "governing law", "สัญญา", "คู่สัญญา", "ข้อตกลง",
    ),
    fields=(
        FieldSpec(name="contract_number", label="Contract No.", critical=True,
                  aliases=("Contract No", "Agreement No", "Ref", "เลขที่สัญญา")),
        FieldSpec(name="contract_title", label="Title", required=True,
                  description="The title of the agreement"),
        FieldSpec(name="contract_type", label="Type", type="enum",
                  choices=("service", "supply", "lease", "nda", "employment", "license", "other")),
        FieldSpec(name="party_a", label="Party A", required=True, critical=True,
                  description="First contracting party, usually the client or buyer",
                  aliases=("Party A", "Client", "Customer", "ผู้ว่าจ้าง")),
        FieldSpec(name="party_b", label="Party B", required=True, critical=True,
                  description="Second contracting party, usually the supplier",
                  aliases=("Party B", "Contractor", "Supplier", "ผู้รับจ้าง")),
        FieldSpec(name="effective_date", label="Effective Date", type="date", required=True, critical=True,
                  aliases=("Effective Date", "Commencement", "Start Date", "วันที่มีผลบังคับ")),
        FieldSpec(name="expiry_date", label="Expiry Date", type="date", critical=True,
                  aliases=("Expiry", "End Date", "Termination Date", "วันสิ้นสุด")),
        FieldSpec(name="contract_value", label="Contract Value", type="money", critical=True,
                  description="Total monetary value of the agreement",
                  aliases=("Contract Value", "Total Fee", "Consideration", "มูลค่าสัญญา")),
        FieldSpec(name="currency", label="Currency", type="enum",
                  choices=("THB", "USD", "EUR", "JPY", "SGD", "GBP", "CNY")),
        FieldSpec(name="payment_terms", label="Payment Terms", type="text",
                  aliases=("Payment Terms", "เงื่อนไขการชำระเงิน")),
        FieldSpec(name="notice_period_days", label="Notice Period (days)", type="number",
                  description="Days of notice required to terminate"),
        FieldSpec(name="auto_renew", label="Auto Renew", type="enum", choices=("yes", "no"),
                  description="Whether the contract renews automatically"),
        FieldSpec(name="governing_law", label="Governing Law",
                  aliases=("Governing Law", "Jurisdiction", "กฎหมายที่ใช้บังคับ")),
    ),
)

_FORM = DocumentSchema(
    doc_type="form",
    display_name="Application / Onboarding Form",
    description="A filled-in business form such as vendor onboarding or a claim.",
    record_kind="submission",
    keywords=(
        "application", "form", "please complete", "signature", "date of birth",
        "registration", "แบบฟอร์ม", "ใบสมัคร", "ลงชื่อ",
    ),
    fields=(
        FieldSpec(name="form_type", label="Form Type", required=True,
                  description="What kind of form this is, e.g. vendor onboarding, expense claim"),
        FieldSpec(name="reference_number", label="Reference No.", critical=True,
                  aliases=("Ref", "Reference", "Form No", "เลขที่")),
        FieldSpec(name="submitted_date", label="Submitted Date", type="date",
                  aliases=("Date", "Submitted", "วันที่")),
        FieldSpec(name="applicant_name", label="Applicant / Company", required=True, critical=True,
                  aliases=("Name", "Company Name", "Applicant", "ชื่อ", "ชื่อบริษัท")),
        FieldSpec(name="tax_id", label="Tax ID", type="tax_id",
                  aliases=("Tax registration number", "Tax registration", "VAT registration",
                           "Registration number", "Tax ID", "Registration No",
                           "เลขประจำตัวผู้เสียภาษี", "เลขทะเบียนนิติบุคคล")),
        FieldSpec(name="contact_email", label="Email",
                  aliases=("Email", "E-mail", "อีเมล")),
        FieldSpec(name="contact_phone", label="Phone",
                  aliases=("Phone", "Tel", "Mobile", "โทรศัพท์")),
        FieldSpec(name="address", label="Address", type="text",
                  aliases=("Address", "ที่อยู่")),
        FieldSpec(name="bank_account", label="Bank Account", critical=True,
                  aliases=("Account No", "Bank Account", "เลขที่บัญชี")),
        FieldSpec(name="requested_amount", label="Requested Amount", type="money",
                  description="Amount requested, if the form asks for money"),
        FieldSpec(name="signed_by", label="Signed By",
                  aliases=("Signature", "Authorised by", "ลงชื่อ")),
    ),
)

_UNKNOWN = DocumentSchema(
    doc_type="unknown",
    display_name="Unclassified Document",
    description="Document type could not be determined; only generic fields are extracted.",
    record_kind="generic",
    fields=(
        FieldSpec(name="title", label="Title", description="Best-guess document title"),
        FieldSpec(name="document_date", label="Date", type="date"),
        FieldSpec(name="counterparty", label="Counterparty",
                  description="The main organisation named in the document"),
        FieldSpec(name="reference_number", label="Reference No."),
        FieldSpec(name="total_amount", label="Amount", type="money",
                  description="Any headline monetary amount"),
        FieldSpec(name="currency", label="Currency", type="enum",
                  choices=("THB", "USD", "EUR", "JPY", "SGD", "GBP", "CNY")),
        FieldSpec(name="summary", label="Summary", type="text",
                  description="One or two sentences describing what this document is"),
    ),
)

DOCUMENT_SCHEMAS: dict[str, DocumentSchema] = {
    schema.doc_type: schema
    for schema in (_INVOICE, _RECEIPT, _PURCHASE_ORDER, _CONTRACT, _FORM, _UNKNOWN)
}


def get_schema(doc_type: str) -> DocumentSchema:
    """Look up a schema, falling back to the generic one."""
    return DOCUMENT_SCHEMAS.get(doc_type, _UNKNOWN)


def schema_names() -> tuple[str, ...]:
    """Classifiable document types (excludes `unknown`)."""
    return tuple(name for name in DOCUMENT_SCHEMAS if name != "unknown")
