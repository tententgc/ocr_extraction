"""Per-style page layouts for the demo corpus.

Deliberately inconsistent: different label wording, different date formats,
different table shapes and two languages. That variety is the point - it is what
makes the extraction problem realistic.
"""

from __future__ import annotations

from reportlab.lib.colors import HexColor, white

from docuflow.samples.catalog import BUYER, BUYER_ADDRESS, SampleDoc
from docuflow.samples.formatting import (
    date_long_en,
    date_slash,
    date_thai,
    date_us,
    money,
    money_thai,
    quantity,
    to_thai_digits,
)
from docuflow.samples.sheet import BLACK, GREY, LIGHT, Sheet

ACCENT = HexColor("#1F3A5F")
SOFT = HexColor("#F1F4F8")

VENDOR_ADDRESSES = {
    "Siam Tech Supply Co., Ltd.": "199/22 Ratchadaphisek Road, Din Daeng, Bangkok 10400",
    "Northwind Logistics Ltd.": "45 Bangna-Trad Km.19, Bang Phli, Samut Prakan 10540",
    "Bluepeak Software GmbH": "Rosenthaler Strasse 40, 10178 Berlin, Germany",
    "บริษัท เจ้าพระยา แฟซิลิตี้ จำกัด": "77/8 ถนนพระราม 3 แขวงบางโพงพาง เขตยานนาวา กรุงเทพฯ 10120",
    "Summit Solutions Trading": "Unit 8B, 331 Sukhumvit Soi 22, Khlong Toei, Bangkok 10110",
    "Orient Paper & Print Co., Ltd.": "1200 Phetkasem Road, Bang Khae, Bangkok 10160",
}


def _vendor_address(name: str | None) -> str:
    return VENDOR_ADDRESSES.get(name or "", "Bangkok, Thailand")


def _items_or_default(doc: SampleDoc) -> tuple[dict, ...]:
    if doc.line_items:
        return doc.line_items
    total = doc.ground_truth.get("total_amount")
    return ({"description": "Services rendered", "quantity": 1, "unit_price": total, "amount": total},)


# ---------------------------------------------------------------------------
# Invoice styles
# ---------------------------------------------------------------------------


def draw_classic_invoice(sheet: Sheet, doc: SampleDoc) -> None:
    """Traditional layout: vendor top-left, meta top-right, totals bottom-right."""
    gt = doc.ground_truth
    is_po = doc.doc_type == "purchase_order"
    top = sheet.y

    sheet.text(gt.get("vendor_name", ""), size=14, bold=True)
    sheet.text(_vendor_address(gt.get("vendor_name")), size=8.5, color=GREY)
    if gt.get("vendor_tax_id"):
        sheet.text(f"Tax ID: {gt['vendor_tax_id']}", size=8.5, color=GREY)

    title = "PURCHASE ORDER" if is_po else "TAX INVOICE"
    sheet.at(title, sheet.right, top, size=20, bold=True, align="right", color=ACCENT)

    meta_y = top - 30
    rows = (
        [("PO Number", gt.get("po_number")), ("PO Date", date_long_en(gt.get("po_date")))]
        if is_po
        else [
            ("Invoice No.", gt.get("invoice_number")),
            ("Invoice Date", date_long_en(gt.get("invoice_date"))),
            ("Due Date", date_long_en(gt.get("due_date"))),
        ]
    )
    if gt.get("purchase_order_ref"):
        rows.append(("PO Reference", gt["purchase_order_ref"]))
    if is_po and gt.get("delivery_date"):
        rows.append(("Required By", date_long_en(gt["delivery_date"])))

    for label, value in rows:
        sheet.at(f"{label}:", sheet.right - 130, meta_y, size=8.5, color=GREY, align="left")
        sheet.at(str(value or ""), sheet.right, meta_y, size=9.5, bold=True, align="right")
        meta_y -= 14

    sheet.y = min(sheet.y, meta_y) - 14
    sheet.rule()
    sheet.down(6)

    sheet.text("Bill To" if not is_po else "Ship To", size=8.5, bold=True, color=GREY)
    sheet.text(gt.get("buyer_name", BUYER), size=10.5, bold=True)
    sheet.text(BUYER_ADDRESS, size=8.5, color=GREY)
    sheet.down(14)

    _draw_items_table(sheet, doc, currency=gt.get("currency", "THB"))
    _draw_totals_block(sheet, doc)

    sheet.down(18)
    if gt.get("payment_terms"):
        sheet.text(f"Payment Terms: {gt['payment_terms']}", size=8.5, color=GREY)
    if gt.get("bank_account"):
        sheet.text(f"Remit to Account No: {gt['bank_account']}", size=8.5, color=GREY)


def draw_modern_invoice(sheet: Sheet, doc: SampleDoc) -> None:
    """Banded header, boxed totals, different label wording ('Amount Due')."""
    gt = doc.ground_truth

    sheet.band(56, ACCENT)
    sheet.at(gt.get("vendor_name", ""), sheet.left + 14, sheet.y - 24, size=14, bold=True, color=white)
    sheet.at(_vendor_address(gt.get("vendor_name")), sheet.left + 14, sheet.y - 40, size=8, color=white)
    sheet.at("INVOICE", sheet.right - 14, sheet.y - 30, size=17, bold=True, align="right", color=white)
    sheet.down(56 + 20)

    left_x, right_x = sheet.left, sheet.left + sheet.content_width / 2 + 10
    block_y = sheet.y

    sheet.at("BILLED TO", left_x, block_y, size=7.5, bold=True, color=GREY)
    sheet.at(gt.get("buyer_name", BUYER), left_x, block_y - 14, size=10, bold=True)
    sheet.at(BUYER_ADDRESS, left_x, block_y - 27, size=8, color=GREY)

    pairs = [
        ("Invoice #", gt.get("invoice_number")),
        ("Issued", date_us(gt.get("invoice_date"))),
        ("Payment Due", date_us(gt.get("due_date"))),
        ("PO", gt.get("purchase_order_ref") or "-"),
        ("Supplier Tax ID", gt.get("vendor_tax_id") or "-"),
    ]
    row_y = block_y
    for label, value in pairs:
        sheet.at(label, right_x, row_y, size=8, color=GREY)
        sheet.at(str(value or ""), sheet.right, row_y, size=9, bold=True, align="right")
        row_y -= 13

    sheet.y = min(block_y - 45, row_y) - 16
    _draw_items_table(sheet, doc, currency=gt.get("currency", "THB"), shaded=True)

    # Totals sit in a tinted panel on the right.
    panel_width = 230.0
    panel_x = sheet.right - panel_width
    rows = _totals_rows(doc)
    panel_height = 16 * len(rows) + 16
    sheet.box(panel_x, sheet.y - panel_height, panel_width, panel_height, stroke=LIGHT, fill=SOFT)

    row_y = sheet.y - 20
    currency = gt.get("currency", "THB")
    for index, (label, value) in enumerate(rows):
        emphasise = index == len(rows) - 1
        sheet.at(label, panel_x + 12, row_y, size=9 if not emphasise else 10, bold=emphasise, color=BLACK if emphasise else GREY)
        sheet.at(
            f"{money(value)} {currency}",
            sheet.right - 12,
            row_y,
            size=9 if not emphasise else 11.5,
            bold=emphasise,
            align="right",
        )
        row_y -= 16
    sheet.y = sheet.y - panel_height - 22

    if gt.get("payment_terms"):
        sheet.text(f"Terms: {gt['payment_terms']}", size=8.5, color=GREY)
    if gt.get("bank_account"):
        sheet.text(f"Bank Account: {gt['bank_account']}", size=8.5, color=GREY)


def draw_thai_tax_invoice(sheet: Sheet, doc: SampleDoc) -> None:
    """Thai-language tax invoice: Thai labels, Buddhist era, Thai numerals."""
    gt = doc.ground_truth
    top = sheet.y

    sheet.text(gt.get("vendor_name", ""), size=13, bold=True)
    sheet.text(_vendor_address(gt.get("vendor_name")), size=8.5, color=GREY)
    sheet.text(f"เลขประจำตัวผู้เสียภาษี {gt.get('vendor_tax_id', '')}", size=8.5, color=GREY)

    sheet.at("ใบกำกับภาษี / ใบแจ้งหนี้", sheet.right, top, size=15, bold=True, align="right", color=ACCENT)
    sheet.at("(ต้นฉบับ)", sheet.right, top - 17, size=8.5, align="right", color=GREY)

    meta_y = top - 36
    for label, value in (
        ("เลขที่", gt.get("invoice_number")),
        ("วันที่", date_thai(gt.get("invoice_date"))),
        ("ครบกำหนดชำระ", date_thai(gt.get("due_date"))),
        ("อ้างอิงใบสั่งซื้อ", gt.get("purchase_order_ref")),
    ):
        if not value:
            continue
        sheet.at(f"{label}:", sheet.right - 150, meta_y, size=8.5, color=GREY)
        sheet.at(str(value), sheet.right, meta_y, size=9.5, bold=True, align="right")
        meta_y -= 14

    sheet.y = min(sheet.y, meta_y) - 12
    sheet.rule()
    sheet.down(6)

    sheet.text("ลูกค้า / ผู้ซื้อ", size=8.5, bold=True, color=GREY)
    sheet.text(gt.get("buyer_name", BUYER), size=10.5, bold=True)
    sheet.text(BUYER_ADDRESS, size=8.5, color=GREY)
    sheet.down(14)

    _draw_items_table(
        sheet,
        doc,
        currency="",
        headers=("รายการ", "จำนวน", "ราคาต่อหน่วย", "จำนวนเงิน"),
        thai_numerals=True,
    )

    row_y = sheet.y
    for label, value, bold in (
        ("รวมเป็นเงิน", gt.get("subtotal"), False),
        (f"ภาษีมูลค่าเพิ่ม {to_thai_digits('7')}%", gt.get("tax_amount"), False),
        ("จำนวนเงินรวมทั้งสิ้น", gt.get("total_amount"), True),
    ):
        sheet.at(label, sheet.right - 200, row_y, size=9.5 if not bold else 10.5, bold=bold, color=GREY if not bold else BLACK)
        sheet.at(money_thai(value), sheet.right, row_y, size=9.5 if not bold else 11.5, bold=bold, align="right")
        row_y -= 17
    sheet.y = row_y - 14

    sheet.text(f"เงื่อนไขการชำระเงิน: {gt.get('payment_terms', '')}", size=8.5, color=GREY)
    sheet.text(f"เลขที่บัญชีธนาคาร: {gt.get('bank_account', '')}", size=8.5, color=GREY)
    sheet.down(24)
    sheet.at("ผู้รับเงิน ..............................", sheet.right - 170, sheet.y, size=9)


def draw_dense_invoice(sheet: Sheet, doc: SampleDoc) -> None:
    """Cramped, borderless, inline labels - the layout OCR struggles with most."""
    gt = doc.ground_truth

    sheet.text(gt.get("vendor_name", ""), size=11, bold=True)
    sheet.text(_vendor_address(gt.get("vendor_name")), size=7.5)
    sheet.text(f"TIN {gt.get('vendor_tax_id', '')}   Tel 02-555-0100", size=7.5)
    sheet.down(4)

    sheet.text(
        f"INVOICE  {gt.get('invoice_number', '')}   Dated {date_slash(gt.get('invoice_date'))}   "
        f"Terms {gt.get('payment_terms', '')}   Due {date_slash(gt.get('due_date'))}",
        size=8,
        bold=True,
    )
    sheet.text(f"To: {gt.get('buyer_name', BUYER)}, {BUYER_ADDRESS}", size=7.5)
    sheet.down(6)

    currency = gt.get("currency", "THB")
    for index, item in enumerate(_items_or_default(doc), start=1):
        if sheet.needs_page(60.0):
            sheet.new_page()
        sheet.text(
            f"{index}. {item['description']}  x{quantity(item.get('quantity'))} @ "
            f"{money(item.get('unit_price'))} = {money(item.get('amount'))}",
            size=7.5,
        )
    sheet.down(6)

    sheet.text(
        f"Sub-total {money(gt.get('subtotal'))} {currency}   "
        f"VAT {money(gt.get('tax_rate'), 0)}% {money(gt.get('tax_amount'))}   "
        f"TOTAL DUE {money(gt.get('total_amount'))} {currency}",
        size=8.5,
        bold=True,
    )
    sheet.down(4)
    sheet.text(f"Please remit to a/c {gt.get('bank_account', '')} within terms.", size=7.5)


# ---------------------------------------------------------------------------
# Non-invoice styles
# ---------------------------------------------------------------------------


def draw_contract(sheet: Sheet, doc: SampleDoc) -> None:
    """Multi-page prose agreement with a signature block."""
    gt = doc.ground_truth

    sheet.at(gt.get("contract_title", "AGREEMENT").upper(), sheet.width / 2, sheet.y, size=14, bold=True, align="center")
    sheet.down(20)
    if gt.get("contract_number"):
        sheet.at(f"Contract No. {gt['contract_number']}", sheet.width / 2, sheet.y, size=9.5, align="center", color=GREY)
        sheet.down(24)
    sheet.rule()
    sheet.down(10)

    for paragraph in doc.body:
        sheet.paragraph(paragraph, size=9.5, leading=14)
        sheet.down(8)

    if sheet.needs_page(150):
        sheet.new_page()
    sheet.down(20)
    sheet.rule()
    sheet.down(14)
    sheet.text("IN WITNESS WHEREOF, the parties have executed this Agreement as of the Effective Date.", size=9.5)
    sheet.down(28)

    column_y = sheet.y
    for x, party, signer in (
        (sheet.left, gt.get("party_a", ""), "Somchai Wattanakul, Chief Financial Officer"),
        (sheet.left + sheet.content_width / 2 + 10, gt.get("party_b", ""), "Naruemon Pichai, Managing Director"),
    ):
        sheet.at("_________________________", x, column_y, size=9.5)
        sheet.at(party, x, column_y - 16, size=9, bold=True)
        sheet.at(signer, x, column_y - 29, size=8, color=GREY)
    sheet.y = column_y - 44


def draw_form(sheet: Sheet, doc: SampleDoc) -> None:
    """Label/value form with checkboxes and a signature line."""
    gt = doc.ground_truth

    sheet.band(34, SOFT)
    sheet.at(BUYER, sheet.left + 12, sheet.y - 14, size=10, bold=True, color=ACCENT)
    sheet.at("Procurement - Supplier Registration", sheet.left + 12, sheet.y - 26, size=8, color=GREY)
    sheet.at(f"Ref: {gt.get('reference_number', '')}", sheet.right - 12, sheet.y - 20, size=9, bold=True, align="right")
    sheet.down(34 + 18)

    sheet.text(gt.get("form_type", "Application Form").upper() + " FORM", size=13, bold=True)
    sheet.down(6)
    sheet.text(f"Date submitted: {date_slash(gt.get('submitted_date'))}", size=9, color=GREY)
    sheet.down(10)

    sheet.text("Registration type (tick one):", size=8.5, bold=True, color=GREY)
    options = (("New supplier", True), ("Update existing", False), ("Reactivation", False))
    box_y = sheet.y - 2
    x = sheet.left
    for label, checked in options:
        sheet.checkbox(x, box_y, checked)
        sheet.at(label, x + 13, box_y + 0.5, size=8.5)
        x += 130
    sheet.down(24)

    fields = (
        ("Company / Applicant name", gt.get("applicant_name")),
        ("Tax registration number", gt.get("tax_id")),
        ("Registered address", gt.get("address")),
        ("Contact e-mail", gt.get("contact_email")),
        ("Contact telephone", gt.get("contact_phone")),
        ("Bank account number", gt.get("bank_account")),
    )
    for label, value in fields:
        row_y = sheet.y
        sheet.at(f"{label}", sheet.left, row_y, size=8, color=GREY)
        sheet.at(str(value or ""), sheet.left + 4, row_y - 15, size=10)
        sheet.canvas.setStrokeColor(LIGHT)
        sheet.canvas.setLineWidth(0.7)
        sheet.canvas.line(sheet.left, row_y - 20, sheet.right, row_y - 20)
        sheet.down(34)

    sheet.down(10)
    sheet.text("Declaration", size=8.5, bold=True, color=GREY)
    sheet.paragraph(
        "I confirm that the information given above is true and complete, and that the bank account "
        "provided belongs to the applicant company. I understand that payment will only be made to "
        "the registered account.",
        size=8.5,
        leading=12,
    )
    sheet.down(26)
    signature_y = sheet.y
    sheet.at("_________________________", sheet.left, signature_y, size=9.5)
    sheet.at(f"Signed: {gt.get('signed_by', '')}", sheet.left, signature_y - 15, size=9)
    sheet.at("_________________________", sheet.left + 260, signature_y, size=9.5)
    sheet.at(f"Date: {date_slash(gt.get('submitted_date'))}", sheet.left + 260, signature_y - 15, size=9)
    sheet.y = signature_y - 30


def draw_receipt(sheet: Sheet, doc: SampleDoc) -> None:
    """Narrow thermal-printer receipt."""
    gt = doc.ground_truth
    centre = sheet.width / 2

    sheet.at(gt.get("merchant_name", ""), centre, sheet.y, size=10, bold=True, align="center")
    sheet.down(14)
    sheet.at("Rama IV Rd, Khlong Toei, Bangkok", centre, sheet.y, size=6.5, align="center", color=GREY)
    sheet.down(10)
    sheet.at(f"TAX ID {gt.get('merchant_tax_id', '')}", centre, sheet.y, size=6.5, align="center", color=GREY)
    sheet.down(12)
    sheet.at("- " * 26, centre, sheet.y, size=6, align="center", color=GREY)
    sheet.down(12)

    sheet.text(f"RECEIPT {gt.get('receipt_number', '')}", size=7.5, bold=True)
    sheet.text(f"{date_slash(gt.get('receipt_date'))}  14:07", size=7)
    sheet.down(6)
    sheet.at("- " * 26, centre, sheet.y, size=6, align="center", color=GREY)
    sheet.down(12)

    for item in _items_or_default(doc):
        sheet.text(str(item["description"]), size=7.5)
        sheet.at(
            f"{quantity(item.get('quantity'))} L x {money(item.get('unit_price'))}",
            sheet.left,
            sheet.y,
            size=7,
            color=GREY,
        )
        sheet.at(money(item.get("amount")), sheet.right, sheet.y, size=7.5, align="right")
        sheet.down(13)

    sheet.at("- " * 26, centre, sheet.y, size=6, align="center", color=GREY)
    sheet.down(12)
    for label, value, bold in (
        ("Sub-total", gt.get("subtotal"), False),
        ("VAT 7%", gt.get("tax_amount"), False),
        ("TOTAL", gt.get("total_amount"), True),
    ):
        sheet.at(label, sheet.left, sheet.y, size=7.5 if not bold else 9, bold=bold)
        sheet.at(money(value), sheet.right, sheet.y, size=7.5 if not bold else 9, bold=bold, align="right")
        sheet.down(13)

    sheet.down(4)
    sheet.at("PAID BY VISA ****4412", centre, sheet.y, size=7, align="center")
    sheet.down(14)
    sheet.at("THANK YOU / ขอบคุณค่ะ", centre, sheet.y, size=7, align="center", color=GREY)


# ---------------------------------------------------------------------------
# Shared table + totals helpers
# ---------------------------------------------------------------------------


def draw_two_column_invoice(sheet: Sheet, doc: SampleDoc) -> None:
    """Three company names on one page, only one of which is the supplier.

    Bill-to and ship-to sit side by side, and payment is directed to a factoring
    house under its own name and its own account number. Every real invoice-fraud
    attempt looks exactly like this, and so does a great many legitimate ones -
    the reader has to know that the issuer is the letterhead, the payee block is
    not a party to the transaction, and the account is the one to check.
    """
    gt = doc.ground_truth
    top = sheet.y

    sheet.text(gt.get("vendor_name", ""), size=13, bold=True)
    sheet.text(_vendor_address(gt.get("vendor_name")), size=8, color=GREY)
    if gt.get("vendor_tax_id"):
        sheet.text(f"Tax ID {gt['vendor_tax_id']}", size=8, color=GREY)

    sheet.at("INVOICE", sheet.right, top, size=18, bold=True, align="right", color=ACCENT)
    sheet.at(str(gt.get("invoice_number", "")), sheet.right, top - 20, size=10.5, bold=True, align="right")
    sheet.at(date_us(gt.get("invoice_date")), sheet.right, top - 36, size=9, align="right", color=GREY)
    sheet.at(f"Terms {gt.get('payment_terms', '')}", sheet.right, top - 50, size=9, align="right", color=GREY)

    sheet.y = min(sheet.y, top - 64)
    sheet.rule()
    sheet.down(8)

    # Two address blocks, side by side, same shape and same size. Nothing about
    # the typography says which one is the counterparty.
    column = sheet.content_width / 2
    block_top = sheet.y
    for offset, (heading, name, address) in enumerate((
        ("BILL TO", gt.get("buyer_name", BUYER), BUYER_ADDRESS),
        ("SHIP TO", gt.get("ship_to_name", BUYER), gt.get("ship_to_address", BUYER_ADDRESS)),
    )):
        x = sheet.left + offset * column
        sheet.at(heading, x, block_top, size=7.5, bold=True, color=GREY)
        sheet.at(str(name), x, block_top - 14, size=10, bold=True)
        sheet.at(str(address), x, block_top - 27, size=8, color=GREY)
    sheet.y = block_top - 48

    _draw_items_table(sheet, doc, currency=gt.get("currency", "THB"), shaded=True)
    _draw_totals_block(sheet, doc)

    sheet.down(16)
    payee = gt.get("remit_to_name")
    if payee:
        sheet.box(sheet.left, sheet.y - 58, sheet.content_width, 58, fill=SOFT)
        sheet.at("PAYMENT INSTRUCTIONS", sheet.left + 12, sheet.y - 16, size=7.5, bold=True, color=GREY)
        sheet.at(f"Payable to: {payee}", sheet.left + 12, sheet.y - 31, size=9.5, bold=True)
        sheet.at(
            f"Account No: {gt.get('bank_account', '')}   ·   {gt.get('remit_to_note', '')}",
            sheet.left + 12, sheet.y - 46, size=8.5, color=GREY,
        )
        sheet.down(70)
    elif gt.get("bank_account"):
        sheet.text(f"Remit to Account No: {gt['bank_account']}", size=8.5, color=GREY)


def draw_minimal_invoice(sheet: Sheet, doc: SampleDoc) -> None:
    """Almost nothing is labelled. The values are there; the words are not.

    A designer's invoice: the number sits under the logo with no caption, the
    date is a bare line, and the only word near the total is the currency. Blur
    tests the OCR; this tests whether the reader can place a value with no label
    beside it - which is the harder and more common problem.
    """
    gt = doc.ground_truth
    sheet.down(30)

    sheet.text(gt.get("vendor_name", ""), size=22, bold=True)
    sheet.down(2)
    sheet.text(str(gt.get("invoice_number", "")), size=11, color=GREY)
    sheet.text(date_slash(gt.get("invoice_date")), size=11, color=GREY)
    sheet.down(26)

    sheet.text(str(gt.get("buyer_name", BUYER)), size=10.5)
    sheet.text(BUYER_ADDRESS, size=8.5, color=GREY)
    sheet.down(30)

    # No column headers at all - just rows of description and amount.
    for item in _items_or_default(doc):
        row_y = sheet.y
        sheet.at(str(item["description"]), sheet.left, row_y, size=10)
        sheet.at(money(item.get("amount")), sheet.right, row_y, size=10, align="right")
        sheet.down(20)

    sheet.down(6)
    sheet.rule(color=BLACK, thickness=0.8, gap=14)

    currency = gt.get("currency", "THB")
    for label, value in (
        ("", gt.get("subtotal")),
        (f"+{money(gt.get('tax_rate'), 0)}%", gt.get("tax_amount")),
    ):
        if value is None:
            continue
        row_y = sheet.y
        if label:
            sheet.at(label, sheet.right - 120, row_y, size=9.5, color=GREY, align="right")
        sheet.at(money(value), sheet.right, row_y, size=9.5, align="right")
        sheet.down(17)

    sheet.down(4)
    sheet.at(currency, sheet.right - 120, sheet.y, size=12, color=GREY, align="right")
    sheet.at(money(gt.get("total_amount")), sheet.right, sheet.y, size=16, bold=True, align="right")
    sheet.down(40)

    if gt.get("due_date"):
        sheet.text(date_slash(gt["due_date"]), size=9.5, color=GREY)
    if gt.get("bank_account"):
        sheet.text(str(gt["bank_account"]), size=9.5, color=GREY)


def _draw_items_table(
    sheet: Sheet,
    doc: SampleDoc,
    currency: str = "THB",
    headers: tuple[str, str, str, str] = ("Description", "Qty", "Unit Price", "Amount"),
    shaded: bool = False,
    thai_numerals: bool = False,
) -> None:
    items = _items_or_default(doc)
    columns = (sheet.left + 4, sheet.right - 250, sheet.right - 150, sheet.right)

    header_y = sheet.y
    if shaded:
        sheet.band(18, SOFT)
    sheet.at(headers[0], columns[0], header_y - 12, size=8, bold=True, color=GREY)
    for index, header in enumerate(headers[1:], start=1):
        sheet.at(header, columns[index], header_y - 12, size=8, bold=True, color=GREY, align="right")
    sheet.y = header_y - 22
    if not shaded:
        sheet.rule(color=GREY, thickness=0.5, gap=8)

    fmt = money_thai if thai_numerals else money
    qty_fmt = (lambda value: to_thai_digits(quantity(value))) if thai_numerals else quantity

    for item in items:
        # A long order runs onto a second page, taking its totals with it. Any
        # reader that stops after page one gets a plausible, wrong answer -
        # which is worse than failing.
        if sheet.needs_page(120.0):
            sheet.new_page()
            sheet.text(f"{doc.ground_truth.get('invoice_number') or doc.ground_truth.get('po_number', '')}"
                       f"  ·  continued", size=8, color=GREY)
            sheet.down(6)
            header_y = sheet.y
            sheet.at(headers[0], columns[0], header_y - 12, size=8, bold=True, color=GREY)
            for index, header in enumerate(headers[1:], start=1):
                sheet.at(header, columns[index], header_y - 12, size=8, bold=True, color=GREY, align="right")
            sheet.y = header_y - 22
            sheet.rule(color=GREY, thickness=0.5, gap=8)

        row_y = sheet.y
        sheet.at(str(item["description"]), columns[0], row_y, size=9)
        sheet.at(qty_fmt(item.get("quantity")), columns[1], row_y, size=9, align="right")
        sheet.at(fmt(item.get("unit_price")), columns[2], row_y, size=9, align="right")
        sheet.at(fmt(item.get("amount")), columns[3], row_y, size=9, align="right")
        sheet.down(17)

    sheet.rule(color=LIGHT, gap=10)


def _totals_rows(doc: SampleDoc) -> list[tuple[str, float | None]]:
    gt = doc.ground_truth
    rows: list[tuple[str, float | None]] = []
    if gt.get("subtotal") is not None:
        rows.append(("Subtotal", gt["subtotal"]))
    if gt.get("tax_amount") is not None:
        rate = gt.get("tax_rate")
        label = f"VAT {money(rate, 0)}%" if rate else "VAT"
        rows.append((label, gt["tax_amount"]))
    rows.append(("Amount Due", gt.get("total_amount")))
    return rows


def _draw_totals_block(sheet: Sheet, doc: SampleDoc) -> None:
    currency = doc.ground_truth.get("currency", "THB")
    row_y = sheet.y
    rows = _totals_rows(doc)
    for index, (label, value) in enumerate(rows):
        emphasise = index == len(rows) - 1
        sheet.at(label, sheet.right - 200, row_y, size=9.5 if not emphasise else 10.5, bold=emphasise, color=GREY if not emphasise else BLACK)
        sheet.at(
            f"{money(value)} {currency}".strip(),
            sheet.right,
            row_y,
            size=9.5 if not emphasise else 11.5,
            bold=emphasise,
            align="right",
        )
        row_y -= 17
    sheet.y = row_y - 6


def _thai_renderers() -> dict:
    """Imported here rather than at module scope: `layouts_th` reuses the table
    and totals helpers defined above, so a top-level import is a cycle."""
    from docuflow.samples.layouts_th import (
        draw_thai_contract,
        draw_thai_modern_invoice,
        draw_thai_receipt,
    )

    return {
        "thai_modern": draw_thai_modern_invoice,
        "thai_receipt": draw_thai_receipt,
        "thai_contract": draw_thai_contract,
    }


LAYOUT_RENDERERS = {
    "classic": draw_classic_invoice,
    "modern": draw_modern_invoice,
    "thai_tax": draw_thai_tax_invoice,
    **_thai_renderers(),
    "dense": draw_dense_invoice,
    "two_column": draw_two_column_invoice,
    "minimal": draw_minimal_invoice,
    "prose": draw_contract,
    "form": draw_form,
    "narrow": draw_receipt,
}
