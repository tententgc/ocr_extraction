"""Thai-language page layouts.

Kept apart from `layouts.py` because they are one coherent group and because
Thai documents are where this corpus asks its own questions - the English
layouts vary the *shape* of a page, these vary the language it is written in,
and the two fail for different reasons.

Three of them, and the split between the first two is the important one:

* `thai_tax` (still in `layouts.py`) prints ๒๕,๖๘๐.๐๐ - Thai numerals, the way a
  government form or a temple receipt is printed.
* `thai_modern` prints 25,680.00 under the same Thai labels, which is what a
  supplier's accounting package emits and what almost every real Thai business
  invoice looks like.

Thai numerals test recognition: either ๙ comes back or it does not, and the
failure is loud. Arabic digits under Thai labels test comprehension: the numbers
come back perfectly and the only question is whether anything understood which
one is the total. That failure is silent, and it is the one worth testing.
"""

from __future__ import annotations

from docuflow.samples.catalog import BUYER, BUYER_ADDRESS, SampleDoc
from docuflow.samples.formatting import date_thai, money, quantity
from docuflow.samples.layouts import (
    ACCENT,
    GREY,
    SOFT,
    _draw_items_table,
    _items_or_default,
    _vendor_address,
)
from docuflow.samples.sheet import BLACK, Sheet


def _draw_letterhead(sheet: Sheet, gt: dict) -> None:
    """Vendor block on the left, document title on the right, in a tinted band.

    A vendor with no tax ID prints a phone number in its place - which is what a
    Thai letterhead does, and it removes the exact identifier that makes vendor
    resolution easy, leaving only the Thai name to match on.
    """
    top = sheet.y
    sheet.band(56, SOFT)
    sheet.at(gt.get("vendor_name", ""), sheet.left + 8, top - 18, size=13, bold=True)
    sheet.at(_vendor_address(gt.get("vendor_name")), sheet.left + 8, top - 32, size=8, color=GREY)
    sheet.at(
        f"เลขประจำตัวผู้เสียภาษี {gt['vendor_tax_id']}" if gt.get("vendor_tax_id")
        else "โทร. 02-118-4400",
        sheet.left + 8, top - 44, size=8, color=GREY,
    )
    sheet.at("ใบกำกับภาษี / ใบแจ้งหนี้", sheet.right - 8, top - 20, size=14, bold=True,
             align="right", color=ACCENT)
    sheet.at("TAX INVOICE (ORIGINAL)", sheet.right - 8, top - 34, size=7.5, align="right", color=GREY)
    sheet.y = top - 74


def _draw_parties_and_meta(sheet: Sheet, gt: dict) -> None:
    """Who it is for on the left, the document's own numbers on the right.

    Buddhist-era years in Arabic digits: "25 กรกฎาคม 2569".
    """
    block_y = sheet.y
    sheet.at("ลูกค้า", sheet.left, block_y, size=8, bold=True, color=GREY)
    sheet.at(gt.get("buyer_name", BUYER), sheet.left, block_y - 15, size=10.5, bold=True)
    sheet.at(BUYER_ADDRESS, sheet.left, block_y - 28, size=8, color=GREY)

    meta_y = block_y
    for label, value in (
        ("เลขที่เอกสาร", gt.get("invoice_number")),
        ("วันที่ออก", date_thai(gt.get("invoice_date"), thai_digits=False)),
        ("กำหนดชำระ", date_thai(gt.get("due_date"), thai_digits=False)),
        ("ใบสั่งซื้อเลขที่", gt.get("purchase_order_ref")),
    ):
        if not value:
            continue
        sheet.at(str(label), sheet.right - 165, meta_y, size=8, color=GREY)
        sheet.at(str(value), sheet.right, meta_y, size=9.5, bold=True, align="right")
        meta_y -= 14

    sheet.y = min(block_y - 44, meta_y) - 10
    sheet.rule()
    sheet.down(8)


def draw_thai_modern_invoice(sheet: Sheet, doc: SampleDoc) -> None:
    """Thai labels, **Arabic numerals**, Buddhist-era years spelled out.

    The overwhelmingly common shape of a real Thai tax invoice, and the one the
    corpus was missing: `thai_tax` prints ๒๕,๖๘๐.๐๐ throughout, which is what a
    government form or a temple receipt looks like, not what a supplier's
    accounting package emits. A business invoice is written in Thai and totalled
    in 25,680.00.

    Keeping both matters because they fail differently. Thai numerals test
    whether the digits survive OCR at all; Arabic digits under Thai labels test
    whether the *label* was understood — the harder question, and the one an
    English-trained extractor gets wrong quietly.
    """
    gt = doc.ground_truth
    _draw_letterhead(sheet, gt)
    _draw_parties_and_meta(sheet, gt)

    _draw_items_table(
        sheet, doc, currency="",
        headers=("รายละเอียด", "จำนวน", "ราคา/หน่วย", "รวมเงิน"),
        shaded=True,
    )

    row_y = sheet.y
    rate = gt.get("tax_rate")
    for label, value, bold in (
        ("มูลค่าสินค้า/บริการ", gt.get("subtotal"), False),
        (f"ภาษีมูลค่าเพิ่ม {money(rate, 0)}%" if rate else "ภาษีมูลค่าเพิ่ม", gt.get("tax_amount"), False),
        ("จำนวนเงินทั้งสิ้น (บาท)", gt.get("total_amount"), True),
    ):
        if value is None:
            continue
        sheet.at(label, sheet.right - 210, row_y, size=9 if not bold else 10.5, bold=bold,
                 color=GREY if not bold else BLACK)
        sheet.at(money(value), sheet.right, row_y, size=9.5 if not bold else 12, bold=bold, align="right")
        row_y -= 17
    sheet.y = row_y - 16

    if gt.get("payment_terms"):
        sheet.text(f"เงื่อนไขการชำระเงิน  {gt['payment_terms']}", size=8.5, color=GREY)
    if gt.get("bank_account"):
        sheet.text(f"ชำระเข้าบัญชีเลขที่  {gt['bank_account']}", size=8.5, color=GREY)
    sheet.down(26)
    sheet.at("ผู้รับเงิน ______________________", sheet.right - 180, sheet.y, size=8.5, color=GREY)


def draw_thai_receipt(sheet: Sheet, doc: SampleDoc) -> None:
    """ใบเสร็จรับเงิน on the narrow thermal roll, Arabic digits.

    A receipt names its counterparty `merchant_name`, not `vendor_name`, and the
    Thai word is ผู้ขาย or ร้าน rather than ผู้จำหน่าย. Whether the extractor
    reaches the right schema field from the Thai label is exactly what this is
    for - and the classifier has to pick `receipt` over `invoice` from ใบเสร็จ
    alone, on a page that also says ภาษีมูลค่าเพิ่ม like every invoice does.
    """
    gt = doc.ground_truth
    centre = sheet.width / 2

    sheet.at(gt.get("merchant_name", ""), centre, sheet.y, size=9.5, bold=True, align="center")
    sheet.down(13)
    sheet.at("ถนนพระราม 4 แขวงคลองเตย กรุงเทพฯ", centre, sheet.y, size=6.5, align="center", color=GREY)
    sheet.down(10)
    if gt.get("merchant_tax_id"):
        sheet.at(f"เลขประจำตัวผู้เสียภาษี {gt['merchant_tax_id']}", centre, sheet.y,
                 size=6.5, align="center", color=GREY)
        sheet.down(11)
    sheet.at("ใบเสร็จรับเงิน / ใบกำกับภาษีอย่างย่อ", centre, sheet.y, size=7.5, bold=True, align="center")
    sheet.down(12)
    sheet.at("- " * 24, centre, sheet.y, size=6, align="center", color=GREY)
    sheet.down(12)

    sheet.text(f"เลขที่ {gt.get('receipt_number', '')}", size=7.5, bold=True)
    sheet.text(f"วันที่ {date_thai(gt.get('receipt_date'), thai_digits=False)}  14:07 น.", size=7)
    sheet.down(6)
    sheet.at("- " * 24, centre, sheet.y, size=6, align="center", color=GREY)
    sheet.down(12)

    for item in _items_or_default(doc):
        sheet.text(str(item["description"]), size=7.5)
        sheet.at(f"{quantity(item.get('quantity'))} x {money(item.get('unit_price'))}",
                 sheet.left, sheet.y, size=7, color=GREY)
        sheet.at(money(item.get("amount")), sheet.right, sheet.y, size=7.5, align="right")
        sheet.down(13)

    sheet.at("- " * 24, centre, sheet.y, size=6, align="center", color=GREY)
    sheet.down(12)
    rate = gt.get("tax_rate")
    for label, value, bold in (
        ("มูลค่าสินค้า", gt.get("subtotal"), False),
        (f"ภาษีมูลค่าเพิ่ม {money(rate, 0)}%" if rate else "ภาษีมูลค่าเพิ่ม", gt.get("tax_amount"), False),
        ("รวมทั้งสิ้น", gt.get("total_amount"), True),
    ):
        if value is None:
            continue
        sheet.at(label, sheet.left, sheet.y, size=7.5 if not bold else 9, bold=bold)
        sheet.at(money(value), sheet.right, sheet.y, size=7.5 if not bold else 9, bold=bold, align="right")
        sheet.down(13)

    sheet.down(4)
    if gt.get("payment_method"):
        sheet.at(f"ชำระโดย {gt['payment_method']}", centre, sheet.y, size=7, align="center")
        sheet.down(14)
    sheet.at("ขอบคุณที่ใช้บริการ", centre, sheet.y, size=7, align="center", color=GREY)


def draw_thai_contract(sheet: Sheet, doc: SampleDoc) -> None:
    """สัญญา: Thai prose, Buddhist-era dates, two signature blocks.

    A contract has no arithmetic to fall back on. Every field is a phrase in
    running text - who the parties are, when it starts, when it ends - so this
    is the layout with the least redundancy in the corpus and the one where a
    misread Thai date has nothing to contradict it.
    """
    gt = doc.ground_truth

    sheet.at(gt.get("contract_title", "สัญญา"), sheet.width / 2, sheet.y, size=14, bold=True, align="center")
    sheet.down(20)
    if gt.get("contract_number"):
        sheet.at(f"เลขที่สัญญา {gt['contract_number']}", sheet.width / 2, sheet.y,
                 size=9.5, align="center", color=GREY)
        sheet.down(22)
    sheet.rule()
    sheet.down(12)

    sheet.paragraph(
        f"สัญญาฉบับนี้ทำขึ้นระหว่าง {gt.get('party_a', BUYER)} ซึ่งต่อไปในสัญญานี้เรียกว่า "
        f"“ผู้ว่าจ้าง” ฝ่ายหนึ่ง กับ {gt.get('party_b', '')} ซึ่งต่อไปในสัญญานี้เรียกว่า "
        f"“ผู้รับจ้าง” อีกฝ่ายหนึ่ง",
        size=9.5, leading=15,
    )
    sheet.down(10)

    for paragraph in doc.body:
        sheet.paragraph(paragraph, size=9.5, leading=15)
        sheet.down(8)

    sheet.down(6)
    for label, value in (
        ("วันที่มีผลบังคับใช้", date_thai(gt.get("effective_date"), thai_digits=False)),
        ("วันสิ้นสุดสัญญา", date_thai(gt.get("expiry_date"), thai_digits=False)),
        ("มูลค่าสัญญา", f"{money(gt.get('contract_value'))} {gt.get('currency', 'บาท')}"
         if gt.get("contract_value") is not None else ""),
        ("เงื่อนไขการชำระเงิน", gt.get("payment_terms")),
        ("กฎหมายที่ใช้บังคับ", gt.get("governing_law")),
    ):
        if not value:
            continue
        sheet.at(f"{label}", sheet.left, sheet.y, size=9, color=GREY)
        sheet.at(str(value), sheet.left + 150, sheet.y, size=9.5, bold=True)
        sheet.down(16)

    if sheet.needs_page(150):
        sheet.new_page()
    sheet.down(16)
    sheet.rule()
    sheet.down(14)
    sheet.text("คู่สัญญาทั้งสองฝ่ายได้อ่านและเข้าใจข้อความในสัญญานี้โดยตลอดแล้ว จึงลงลายมือชื่อไว้เป็นสำคัญ", size=9.5)
    sheet.down(28)

    column_y = sheet.y
    for x, party, signer in (
        (sheet.left, gt.get("party_a", ""), "สมชาย วัฒนกุล  ประธานเจ้าหน้าที่ฝ่ายการเงิน"),
        (sheet.left + sheet.content_width / 2 + 10, gt.get("party_b", ""), "นฤมล พิชัย  กรรมการผู้จัดการ"),
    ):
        sheet.at("_________________________", x, column_y, size=9.5)
        sheet.at(str(party), x, column_y - 16, size=9, bold=True)
        sheet.at(signer, x, column_y - 29, size=8, color=GREY)
    sheet.y = column_y - 44


