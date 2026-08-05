"""Generate the self-contained miniature notebook.

The third notebook, and the only one that needs nothing. `docuflow_walkthrough`
is the reference and `docuflow_explained` walks the real pipeline — both of them
require the repository, an install and two API keys before a single cell can
run, which is a long way to travel before you learn anything.

This one carries the whole system inside itself. Every stage is reimplemented in
a few dozen lines of standard-library Python: master data, two independent
readers, merging, grounding, confidence, rules with severity tiers, routing, and
a ledger. Nothing is imported from `docuflow`. Open it in Colab, press Run all,
and watch a document travel from arrival to posted record.

The point is not that the miniature is good — a regex is not Typhoon OCR. The
point is that the *shape* is identical, and the shape is what is hard to see
when every stage is behind an API call. The failures it demonstrates are the
real ones this project has actually hit: the Thai receipt whose own heading
classifies it as an invoice, the two tolerances that must never be one number,
and the invoice that is perfect on the page and must still not be paid.

Written in Thai, because the people it is for read Thai.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("notebooks/docuflow_miniature.ipynb")

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text: str) -> None:
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    })


# ================================================================= title ====
md(r"""
# DocuFlow ย่อส่วน — เอกสารหนึ่งใบ ตั้งแต่มาถึงจนเป็นรายการบัญชี

Notebook นี้**เขียนโค้ดทั้งระบบไว้ในตัวเอง** ไม่ clone อะไร ไม่ต้องใส่ API key ไม่ต้องติดตั้งอะไรเพิ่ม — กด `Runtime → Run all` แล้วดูมันทำงานตั้งแต่ต้นจนจบ

เป็นระบบเดียวกับของจริง แค่ย่อส่วน: ขั้นตอนเหมือนกัน หลักการเหมือนกัน โค้ดสั้นพอที่จะอ่านจบทั้งหมด

---

## ประโยคเดียวที่สรุปทั้งระบบ

> ## โมเดล**อ่าน** · โค้ด**ตัดสิน**

ทุกอย่างที่เหลือตามมาจากประโยคนี้ โมเดลภาษาเก่งเรื่องหาตัวเลขบนหน้ากระดาษที่ไม่เคยเห็นมาก่อน แต่**ผิดคน**ที่จะถามว่าใบนี้จ่ายได้ไหม เพราะเอกสารเองพูดโน้มน้าวมันได้

ดังนั้น: โมเดลสกัดข้อมูล · **โค้ดเท่านั้นที่ตัดสินใจ**

---

## แผนที่

```
  ①            ②           ③            ④
เอกสาร ──▶ อ่านตัวอักษร ──▶ แยกประเภท ──▶ ┌── ช่อง A: อ่านจากป้ายกำกับ ──┐
 มาถึง      (OCR)         invoice?      │                              │
                          receipt?      └── ช่อง B: อ่านจากโครงสร้าง ──┘
                                                      │
                                                      ▼  ⑤
                                          ┌───────────────────────┐
                                          │  รวม + ตรวจว่าอยู่บนหน้า │
                                          └───────────────────────┘
                                                      ▼  ⑥
                                          ┌───────────────────────┐
                                          │      กฎธุรกิจ          │
                                          └───────────────────────┘
                                                      ▼  ⑦
                                          ┌───────────────────────┐
                                          │  ผ่าน หรือ ส่งให้คนดู    │
                                          └───────────────────────┘
```

| ขั้น | ถามอะไร | ใครตอบ |
|---|---|---|
| ① เอกสาร | หน้านี้เขียนว่าอะไร | — |
| ② OCR | ตัวอักษรบนหน้าคืออะไร | โมเดล OCR |
| ③ แยกประเภท | ใบแจ้งหนี้ ใบเสร็จ หรือสัญญา | คำสำคัญ |
| ④ อ่าน ×2 | ค่าแต่ละช่องอยู่ตรงไหน | **สองช่องทางที่ไม่คุยกัน** |
| ⑤ รวม | ทั้งสองตรงกันไหม · ค่านี้อยู่บนหน้าจริงไหม | โค้ด |
| ⑥ กฎ | เอกสารสมเหตุสมผลไหม · ตรงกับที่เราบันทึกไว้ไหม | โค้ด |
| ⑦ ตัดสิน | โพสต์เลย หรือหยุดให้คนดู | **โค้ด เสมอ** |

---

## สองความคิดที่ต้องจำก่อนเริ่ม

**อ่านสองครั้งด้วยวิธีที่ต่างกัน** — โมเดลตัวเดียวบอกตัวเลขมา คือ*คำอ้าง* · สองตัวที่ไม่เคยเห็นคำตอบของกันแล้วตอบตรงกัน คือ*หลักฐาน* ในระบบจริงวัดได้ว่า ช่องที่ตรงกันถูก **97%** ช่องที่ขัดกันถูก **45%** — ช่องว่างตรงนั้นคือเหตุผลที่ต้องเรียกครั้งที่สอง

**ต้องชี้ได้ว่าอ่านมาจากตรงไหน** (grounding) — ทุกค่าต้องอ้างอิงข้อความที่อยู่บนหน้าจริง ค่าที่โมเดลสร้างขึ้นมาเองแต่หาบนหน้าไม่เจอ ถือว่า**แต่งขึ้น** ไม่ว่าจะมั่นใจแค่ไหน
""")

# =========================================================== master data ====
md(r"""
## ① ข้อมูลหลัก (master data) — สิ่งที่เรา "ตกลงไว้แล้ว"

ก่อนดูเอกสาร ต้องมีของให้เทียบก่อน

เอกสารบอก**สิ่งที่คนอื่นอ้าง** · master data บอก**สิ่งที่เราตกลงไว้แล้ว** — ถ้าไม่มีอันหลัง ระบบตรวจได้แค่ว่าเอกสารขัดกับตัวเองไหม (บวกเลขถูกไหม) แต่ตรวจไม่ได้เลยว่าใบนี้**ควรมีอยู่หรือเปล่า**

สามทะเบียน แกนเวลาต่างกัน:

| | ตอบอะไร | อายุ |
|---|---|---|
| **ผู้ขาย** | คนนี้คือใคร ยังค้าด้วยอยู่ไหม | สถานะ ไม่มีวันหมด |
| **ใบสั่งซื้อ** | ของชิ้นนี้เคยสั่งไหม เท่าไหร่ | **ใช้แล้วหมดไป** → ปิด |
| **สัญญา** | ช่วงเวลานี้ผูกพันกันอยู่ไหม | **มีวันเริ่ม–วันจบ** |
""")

code(r'''
# ── ข้อมูลหลัก ─────────────────────────────────────────────────────────────
# ในระบบจริงมาจาก CSV ที่ ERP ส่งออกมา ที่นี่เขียนไว้ตรงๆ ให้เห็นทั้งหมด

VENDORS = {
    "V-1001": {
        "name": "Siam Tech Supply Co., Ltd.",
        # ชื่อไทยอยู่ใน alias — นี่คือเหตุผลที่จับคู่ชื่อไทยกับชื่ออังกฤษได้
        "aliases": ["Siam Tech", "บริษัท สยามเทค ซัพพลาย จำกัด"],
        "tax_id": "0105543000129",
        "bank_account": "1234567890",   # เลขบัญชีที่ "ควรจะเป็น"
        "status": "active",
    },
    "V-1006": {
        "name": "Rungruang Steel Works Co., Ltd.",
        "aliases": ["Rungruang Steel", "บริษัท รุ่งเรือง สตีล เวิร์ค จำกัด"],
        "tax_id": "0105552000110",
        "bank_account": "3120045678",
        "status": "active",
    },
    "V-1012": {
        "name": "Nakhon Chemicals Co., Ltd.",
        "aliases": ["บริษัท นครเคมิคอล จำกัด"],
        "tax_id": "0105539000551",
        "bank_account": "4470098123",
        "status": "on_hold",            # ระงับไว้ — จ่ายอัตโนมัติไม่ได้
    },
}

PURCHASE_ORDERS = {
    # ยอดเก็บแบบ "ไม่รวม VAT" — ใบสั่งซื้อบันทึกสิ่งที่ตกลง ภาษีบวกตอนวางบิล
    "PO-2026-0567": {"vendor": "V-1001", "amount": 135_000.00, "currency": "THB", "status": "open"},
    "PO-2026-0578": {"vendor": "V-1006", "amount":  88_000.00, "currency": "THB", "status": "open"},
    "PO-2026-0545": {"vendor": "V-1001", "amount": 128_000.00, "currency": "THB", "status": "closed"},
}

LEDGER = []          # รายการที่โพสต์แล้ว — ใช้ตรวจใบซ้ำด้วย

print(f"{len(VENDORS)} ผู้ขาย · {len(PURCHASE_ORDERS)} ใบสั่งซื้อ · {len(LEDGER)} รายการในบัญชี\n")
for vid, v in VENDORS.items():
    flag = "" if v["status"] == "active" else f"   ← {v['status']}"
    print(f"  {vid}  {v['name'][:34]:36} บัญชี {v['bank_account']}{flag}")
print()
for po, d in PURCHASE_ORDERS.items():
    flag = "" if d["status"] == "open" else f"   ← {d['status']} (มาวางบิลตอนนี้ = สาย)"
    print(f"  {po}  {d['amount']:>12,.2f} {d['currency']}  {d['vendor']}{flag}")
''')

# ============================================================= documents ====
md(r"""
## ② เอกสารมาถึง

ในระบบจริงคือไฟล์ PDF หรือรูปถ่าย ผ่าน OCR แล้วได้ข้อความออกมา

ที่นี่ข้ามขั้น OCR ไปเลย — เขียนข้อความที่ OCR จะคืนมาไว้ตรงๆ **เพราะสิ่งที่กำลังอธิบายคือสถาปัตยกรรม ไม่ใช่ตัว OCR** ทุกขั้นหลังจากนี้เห็นแค่ข้อความก้อนนี้ ซึ่งเป็นความจริงในระบบจริงเหมือนกัน

สี่ใบ ตั้งใจให้ต่างกันคนละแบบ:

| | เป็นอะไร |
|---|---|
| **A** | ปกติทุกอย่าง ต้องผ่าน |
| **B** | บอก VAT 7% แต่คิด 6% — เลขบนหน้าขัดกันเอง |
| **C** | ภาษาไทย ป้ายกำกับไทย เลขอารบิก — **ใบแจ้งหนี้ไทยจริงหน้าตาแบบนี้** |
| **D** | ใบจริงจากผู้ขายจริง แต่**เลขบัญชีถูกเปลี่ยน** — เอกสารไม่มีอะไรผิดเลย |

ใบ D คือกลโกงที่พบบ่อยที่สุดในโลกจริง และเป็นใบเดียวที่**อ่านให้ดีแค่ไหนก็จับไม่ได้**
""")

code(r'''
# ── เอกสารสี่ใบ ในรูปแบบที่ OCR คืนมา ─────────────────────────────────────
# แต่ละใบมี layout ต่างกันจริง (ป้ายกำกับคนละคำ เรียงคนละแบบ) เพราะนั่นคือ
# สิ่งที่ทำให้ปัญหานี้ยาก — ถ้าทุกใบหน้าตาเหมือนกัน regex ตัวเดียวก็จบ

DOC_A = """TAX INVOICE
Siam Tech Supply Co., Ltd.
199/22 Ratchadaphisek Road, Din Daeng, Bangkok 10400
Tax ID: 0105543000129

Invoice No.: STS-2026-0318
Invoice Date: 24 July 2026
PO Reference: PO-2026-0567

Bill To
Mirtphol Industries PCL

Description                     Qty    Unit Price       Amount
UPS 3kVA                         12      9,500.00    114,000.00
Installation and testing          1     21,000.00     21,000.00

Subtotal                                             135,000.00
VAT 7%                                                 9,450.00
Amount Due                                           144,450.00

Payment Terms: Net 30
Remit to Account No: 1234567890
"""

DOC_B = """TAX INVOICE
Rungruang Steel Works Co., Ltd.
Tax ID: 0105552000110

Invoice No.: RSW-2026-1240
Invoice Date: 27 July 2026
PO Reference: PO-2026-0578

Description                     Qty    Unit Price       Amount
Hot-rolled plate 6mm              4     16,000.00     64,000.00
Cutting and forming               1     24,000.00     24,000.00

Subtotal                                              88,000.00
VAT 7%                                                 5,280.00
Amount Due                                            93,280.00

Remit to Account No: 3120045678
"""

# ป้ายกำกับไทยทั้งหมด เลขอารบิก ปี พ.ศ. — รูปแบบของใบแจ้งหนี้ธุรกิจไทยจริง
DOC_C = """ใบกำกับภาษี / ใบแจ้งหนี้
บริษัท สยามเทค ซัพพลาย จำกัด
เลขประจำตัวผู้เสียภาษี 0105543000129

เลขที่เอกสาร STS-2026-0402
วันที่ออก 29 กรกฎาคม 2569
ใบสั่งซื้อเลขที่ PO-2026-0567

รายละเอียด                    จำนวน   ราคา/หน่วย      รวมเงิน
เครื่องสำรองไฟ UPS 3kVA          12    9,500.00    114,000.00
ติดตั้งและทดสอบระบบ                1   21,000.00     21,000.00

มูลค่าสินค้า/บริการ                                 135,000.00
ภาษีมูลค่าเพิ่ม 7%                                    9,450.00
จำนวนเงินทั้งสิ้น (บาท)                              144,450.00

ชำระเข้าบัญชีเลขที่ 1234567890
"""

DOC_D = """TAX INVOICE
Siam Tech Supply Co., Ltd.
Tax ID: 0105543000129

Invoice No.: STS-2026-0355
Invoice Date: 28 July 2026
PO Reference: PO-2026-0567

Subtotal                                             135,000.00
VAT 7%                                                 9,450.00
Amount Due                                           144,450.00

** PLEASE NOTE: our bank has changed **
Remit to Account No: 9981110022
"""

DOCUMENTS = {
    "A · ปกติ":              DOC_A,
    "B · VAT ผิด":           DOC_B,
    "C · ภาษาไทย":           DOC_C,
    "D · เลขบัญชีถูกเปลี่ยน":  DOC_D,
}

for name, text in DOCUMENTS.items():
    lines = [l for l in text.strip().splitlines() if l.strip()]
    print(f"{name:26} {len(text):>5} ตัวอักษร  {len(lines):>2} บรรทัด   «{lines[0][:40]}»")

print("\n" + "─" * 74)
print("ใบ A เต็มๆ ตามที่ทุกขั้นต่อจากนี้จะเห็น:")
print("─" * 74)
print(DOC_A)
''')

# ============================================================== classify ====
md(r"""
## ③ แยกประเภทเอกสาร

คำตอบตรงนี้เลือก **schema** — รายการช่องที่ต้องหา และช่องไหนสำคัญ — ทุกขั้นหลังจากนี้จึงขึ้นกับการตัดสินตรงนี้

**นับคำสำคัญก่อน เรียกโมเดลเฉพาะตอนคะแนนสูสี** ใบที่เขียน "TAX INVOICE" ตัวใหญ่อยู่บนหัวไม่ต้องใช้โมเดลมาพิสูจน์ คำตอบที่ถูกและถูกที่สุดคือคำตอบที่ดีที่สุด

### กับดักภาษาไทย

เครื่องออกใบเสร็จไทยพิมพ์ว่า **`ใบกำกับภาษีอย่างย่อ`** — แปลว่า "ใบกำกับภาษีแบบย่อ" ซึ่งกรมสรรพากรใช้เรียก*ใบเสร็จ* แต่ข้างในมีคำว่า `ใบกำกับภาษี` ซึ่งเป็นคำสำคัญของ*ใบแจ้งหนี้*ซ้อนอยู่

ถ้าจับคำแบบ substring เฉยๆ **ใบเสร็จไทยทุกใบในประเทศจะถูกจัดเป็นใบแจ้งหนี้** เพราะหัวกระดาษของตัวเอง

กฎที่แก้เรื่องนี้ไม่มีคำว่า "ไทย" อยู่เลย: **เมื่อคำสองคำซ้อนกัน นับเฉพาะคำที่เจาะจงกว่า** — ภาษาอังกฤษก็มีปัญหาเดียวกัน (`No.` ซ้อนใน `Receipt No`)
""")

code(r'''
# ── แยกประเภทด้วยคำสำคัญ ──────────────────────────────────────────────────

KEYWORDS = {
    "invoice": ["tax invoice", "invoice", "amount due", "subtotal",
                "ใบกำกับภาษี", "ใบแจ้งหนี้", "จำนวนเงินทั้งสิ้น"],
    "receipt": ["receipt", "cash", "change",
                "ใบเสร็จ", "ใบเสร็จรับเงิน", "เงินทอน",
                # คำที่เครื่องออกใบเสร็จไทยพิมพ์จริง — เป็นใบเสร็จ ไม่ใช่ใบแจ้งหนี้
                "ใบกำกับภาษีอย่างย่อ"],
    "contract": ["agreement", "contract", "hereby", "party",
                 "สัญญา", "คู่สัญญา"],
}

def shadowed(phrase, matched):
    """จริง เมื่อมีคำที่ยาวกว่าและเจอบนหน้าเหมือนกัน ครอบคำนี้อยู่แล้ว

    กฎทั่วไป ไม่มีความรู้ภาษาใดอยู่ในนี้: เมื่อสองคำที่เจอซ้อนกัน
    ให้เฉพาะคำที่เจาะจงกว่าเป็นคนออกเสียง
    """
    return any(other != phrase and phrase in other for other in matched)

def classify(text):
    low = text.lower()
    found = {kind: {k for k in words if k in low} for kind, words in KEYWORDS.items()}
    everything = set().union(*found.values())

    scores = {}
    for kind, hits in found.items():
        counted = {k for k in hits if not shadowed(k, everything)}
        scores[kind] = (len(counted), sorted(counted), sorted(hits - counted))

    best = max(scores, key=lambda k: scores[k][0])
    runner = sorted((s[0] for k, s in scores.items() if k != best), reverse=True)[0]
    return best, scores[best][0] - runner, scores

# ทดสอบกับทั้งสี่ใบ บวกใบเสร็จไทยที่เป็นกับดัก
THAI_RECEIPT = """บริษัท ปตท. น้ำมันและการค้าปลีก จำกัด
ใบเสร็จรับเงิน / ใบกำกับภาษีอย่างย่อ
เลขที่ PTT-880451
น้ำมันดีเซล B7   50 x 40.00   2,000.00
ภาษีมูลค่าเพิ่ม 7%              140.00
รวมทั้งสิ้น                   2,140.00
"""

print(f"{'เอกสาร':26} {'ตัดสิน':10} {'ห่าง':>5}  คะแนน")
print("─" * 78)
for name, text in list(DOCUMENTS.items()) + [("E · ใบเสร็จไทย (กับดัก)", THAI_RECEIPT)]:
    kind, margin, scores = classify(text)
    tally = "  ".join(f"{k}={v[0]}" for k, v in scores.items())
    print(f"{name:26} {kind:10} {margin:>5}  {tally}")
    for k, (n, counted, ignored) in scores.items():
        if ignored:
            print(f"{'':26} └ {k}: ไม่นับ {ignored} เพราะซ้อนอยู่ในคำที่ยาวกว่า")
''')

# =============================================================== read ×2 ====
md(r"""
## ④ อ่านสองครั้ง ด้วยวิธีที่ต่างกัน

ทั้งสองช่องได้ข้อความเดียวกันและ schema เดียวกัน **ไม่แชร์อะไรกันเลยนอกจากนั้น** — คนละวิธี คนละสมมติฐาน ไม่เห็นคำตอบของกัน

| | อ่านยังไง |
|---|---|
| **ช่อง A — ตามป้ายกำกับ** | เดินตาม schema: "หาเลขที่ใบแจ้งหนี้" แล้วมองหาป้ายที่รู้จัก |
| **ช่อง B — ตามโครงสร้าง** | เดินตามหน้ากระดาษ: เจอตัวเลขรูปแบบเงินตรงไหน แล้วดูว่าคำข้างๆ คืออะไร |

**เข้าหาจากคนละทางโดยตั้งใจ** เพราะสองวิธีที่คิดเหมือนกันจะพลาดเหมือนกัน

ในระบบจริง A คือ Typhoon และ B คือ Claude — คนละบริษัท คนละโมเดล ที่นี่เป็นโค้ดสองก้อนที่เขียนคนละแนวคิด ซึ่ง**หลักการเดียวกัน**: ถ้าทั้งคู่ตอบตรงกันทั้งที่มาคนละทาง คำตอบนั้นน่าเชื่อขึ้นมาก

> สังเกตคอลัมน์ "อ้างอิงจาก" — ทุกค่าต้องบอกได้ว่าอ่านมาจากข้อความไหนบนหน้า ขั้นต่อไปจะเอาข้อความนั้นไปตรวจกับหน้าจริง
""")

code(r'''
import re

# ── schema: ช่องที่ต้องหา และช่องไหนสำคัญ ─────────────────────────────────
# critical = "ถ้าค่านี้มีอยู่แล้วอ่านผิด มีคนได้เงินผิด"
#            ไม่ได้แปลว่า "ต้องมี" — ใบที่ไม่อ้าง PO ก็ผ่านได้
INVOICE_SCHEMA = {
    "invoice_number": {"critical": True},
    "invoice_date":   {"critical": False},
    "vendor_name":    {"critical": True},
    "vendor_tax_id":  {"critical": False},
    "purchase_order": {"critical": True},
    "subtotal":       {"critical": True},
    "tax_rate":       {"critical": False},
    "tax_amount":     {"critical": True},
    "total_amount":   {"critical": True},
    "bank_account":   {"critical": True},
}

def money(s):
    """'144,450.00' → 144450.0 · เลขไทยก็ได้"""
    s = s.translate(str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789"))
    return float(s.replace(",", "").strip())

# ═══ ช่อง A — ตามป้ายกำกับ ════════════════════════════════════════════════
# เดินตาม schema แล้วมองหาป้ายที่รู้จัก ทั้งไทยและอังกฤษ
LABELS = {
    "invoice_number": ["invoice no.:", "invoice no:", "เลขที่เอกสาร", "เลขที่ใบแจ้งหนี้"],
    "invoice_date":   ["invoice date:", "วันที่ออก", "วันที่"],
    "vendor_tax_id":  ["tax id:", "เลขประจำตัวผู้เสียภาษี"],
    "purchase_order": ["po reference:", "ใบสั่งซื้อเลขที่"],
    "subtotal":       ["subtotal", "มูลค่าสินค้า/บริการ"],
    "tax_amount":     ["vat 7%", "ภาษีมูลค่าเพิ่ม 7%"],
    "total_amount":   ["amount due", "จำนวนเงินทั้งสิ้น"],
    "bank_account":   ["remit to account no:", "ชำระเข้าบัญชีเลขที่"],
}

def read_by_label(text):
    out = {}
    for line in text.splitlines():
        low = line.lower()
        for field, labels in LABELS.items():
            if field in out:
                continue
            for label in labels:
                if label in low:
                    tail = line[low.index(label) + len(label):].strip(" :฿")
                    tail = re.sub(r"\(บาท\)", "", tail).strip()
                    if tail:
                        out[field] = (tail, line.strip())   # (ค่า, ข้อความที่อ้างอิง)
                    break
    # ชื่อผู้ขายไม่มีป้าย — เป็นบรรทัดที่ดูเหมือนชื่อบริษัท
    for line in text.splitlines():
        if re.search(r"(Co\.,? Ltd|PCL|บริษัท|จำกัด)", line) and "Bill To" not in line:
            out["vendor_name"] = (line.strip(), line.strip())
            break
    if "tax_amount" in out:
        out["tax_rate"] = ("7", "VAT 7%" if "VAT 7%" in text else "ภาษีมูลค่าเพิ่ม 7%")
    return out

# ═══ ช่อง B — ตามโครงสร้าง ═══════════════════════════════════════════════
# ไม่รู้จักป้ายกำกับเลย: หาตัวเลขที่มีรูปแบบเงิน แล้วเรียงตามขนาด
#   ยอดรวม = ตัวมากสุด · ภาษี = ตัวน้อยสุด · ก่อนภาษี = ตัวกลาง
# วิธีนี้ผิดได้ในแบบที่ช่อง A ไม่ผิด และนั่นคือประโยชน์ของมัน
MONEY_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+\.\d{2}\b")

def read_by_structure(text):
    out = {}
    amounts = []
    for line in text.splitlines():
        hits = MONEY_RE.findall(line)
        if hits and not re.search(r"\d+\s+[\d,]+\.\d{2}\s+[\d,]+\.\d{2}", line):
            amounts.append((money(hits[-1]), line.strip()))
    if amounts:
        tail = amounts[-3:] if len(amounts) >= 3 else amounts
        ordered = sorted(tail, key=lambda a: a[0])
        out["total_amount"] = (f"{ordered[-1][0]:,.2f}", ordered[-1][1])
        out["tax_amount"]   = (f"{ordered[0][0]:,.2f}",  ordered[0][1])
        if len(ordered) >= 3:
            out["subtotal"]  = (f"{ordered[1][0]:,.2f}", ordered[1][1])

    for pattern, field in ((r"\b([A-Z]{2,4}-\d{4}-\d{3,4})\b", "invoice_number"),
                           (r"\b(PO-\d{4}-\d{4})\b",           "purchase_order"),
                           (r"\b(\d{13})\b",                   "vendor_tax_id"),
                           (r"Account No:?\s*(\d{10})|บัญชีเลขที่\s*(\d{10})", "bank_account")):
        for line in text.splitlines():
            m = re.search(pattern, line)
            if m and field not in out:
                out[field] = (next(g for g in m.groups() if g), line.strip())
    for line in text.splitlines():
        if re.search(r"(Co\.,? Ltd|บริษัท.*จำกัด)", line):
            out["vendor_name"] = (line.strip(), line.strip())
            break
    return out

# ── ดูทั้งสองช่องบนใบ A ───────────────────────────────────────────────────
a = read_by_label(DOC_A)
b = read_by_structure(DOC_A)

print(f"{'ช่อง':6} {'FIELD':16} {'ค่าที่อ่านได้':26} อ้างอิงจาก")
print("─" * 96)
for field in INVOICE_SCHEMA:
    for tag, got in (("A", a), ("B", b)):
        if field in got:
            value, evidence = got[field]
            print(f"{tag:6} {field:16} {str(value)[:25]:26} «{evidence[:44]}»")
    print()
''')

# ================================================================= merge ====
md(r"""
## ⑤ รวม · ตรวจว่าอยู่บนหน้าจริง · ให้คะแนนความมั่นใจ

สามเรื่องเกิดที่นี่ และต้องแยกกันให้ชัด เพราะพังคนละแบบ

**รวม** — สองช่องตรงกันไหม เทียบ**หลังจากทำให้เป็นรูปแบบมาตรฐานแล้ว** `144,450.00` กับ `144450.0` คือเลขเดียวกัน ถ้าเทียบสตริงดิบจะรายงานว่าขัดกันทั้งที่ไม่ได้ขัด

**ตรวจว่าอยู่บนหน้า (grounding)** — ข้อความที่อ้างว่าอ่านมา อยู่บนหน้าจริงไหม เช็คกับข้อความจริง ไม่ใช่เชื่อคำอ้าง

**ให้คะแนน** — ความมั่นใจต่อช่อง และมันตอบ**คำถามเดียว**: *เราอ่านค่านี้ถูกไหม* ไม่ใช่ "เอกสารนี้โอเคไหม" ซึ่งเป็นงานของขั้นตัดสิน

> การแยกสองอย่างนี้สำคัญกว่าที่ฟัง — ตอนที่มันเป็นตัวเลขเดียวกัน คำเตือนสามข้อลากเอกสารที่อ่านถูกทุกช่องจาก 95% ลงมา 72% ใต้เกณฑ์ 84% แล้วส่งไปให้คนดูทั้งที่ไม่มีอะไรให้แก้

### สิ่งที่ grounding ทำไม่ได้

มันพิสูจน์ได้แค่ว่าค่านั้นมาจากข้อความบนหน้า — **ถ้า OCR อ่านผิด ค่าที่ผิดก็อยู่บนหน้าจริงในสายตาของ grounding** และข้อความอ้างอิงก็ผิดตามกันไป สองช่องที่อ่านข้อความ OCR ชุดเดียวกันก็จับไม่ได้เหมือนกัน

สิ่งที่กันการอ่านผิดคือ**กฎเลขคณิตในขั้นถัดไป** ไม่ใช่ grounding
""")

code(r'''
# ── รวมสองช่อง ────────────────────────────────────────────────────────────

AGREE, DISAGREE, ONE_CHANNEL = 0.95, 0.55, 0.84   # ความมั่นใจฐาน

def canonical(value):
    """รูปแบบมาตรฐานสำหรับ *เปรียบเทียบ* เท่านั้น
    '144,450.00' และ '144450.0' ต้องเท่ากัน ไม่งั้นจะรายงานว่าขัดกันทั้งที่ไม่ขัด"""
    s = str(value).strip().translate(str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789"))
    try:
        return f"{float(s.replace(',', '')):.2f}"
    except ValueError:
        return " ".join(s.lower().replace(",", " ").split())

def grounded(evidence, page):
    """ข้อความที่อ้างว่าอ่านมา อยู่บนหน้าจริงไหม — เช็ค ไม่ใช่เชื่อ"""
    squash = lambda s: "".join(str(s).lower().split())
    return squash(evidence) in squash(page)

def merge(page, channel_a, channel_b, schema):
    fields = {}
    for name, spec in schema.items():
        a, b = channel_a.get(name), channel_b.get(name)
        if a is None and b is None:
            continue
        value, evidence = a or b

        if a and b:
            agrees = canonical(a[0]) == canonical(b[0])
            confidence, how = (AGREE, "ทั้งสองตรงกัน") if agrees else (DISAGREE, "ขัดกัน")
        else:
            confidence, how = ONE_CHANNEL, "ช่องเดียว"

        on_page = grounded(evidence, page)
        if spec["critical"] and not on_page:
            confidence *= 0.4            # ค่าสำคัญที่ชี้ที่มาไม่ได้ ไว้ใจไม่ได้

        fields[name] = {
            "value": value, "evidence": evidence, "confidence": confidence,
            "grounded": on_page, "how": how, "critical": spec["critical"],
            "raw_a": a[0] if a else None, "raw_b": b[0] if b else None,
        }
    return fields

def document_confidence(fields):
    """ความมั่นใจของทั้งใบ — ช่องสำคัญถ่วงน้ำหนักสามเท่า
    ตอบคำถามเดียว: เราอ่านค่าถูกไหม  ไม่ใช่: ใบนี้จ่ายได้ไหม"""
    if not fields:
        return 0.0
    weight = lambda f: 3.0 if f["critical"] else 1.0
    return sum(f["confidence"] * weight(f) for f in fields.values()) / \
           sum(weight(f) for f in fields.values())

def show_fields(fields, title=""):
    if title:
        print(title)
    print(f"{'':2}{'FIELD':16} {'ค่า':26} {'มั่นใจ':>6} {'บนหน้า':>7}  ช่องทาง")
    print("─" * 82)
    for name, f in fields.items():
        mark = "★" if f["critical"] else " "
        print(f"{mark} {name:16} {str(f['value'])[:25]:26} {f['confidence']:>5.0%} "
              f"{('ใช่' if f['grounded'] else 'ไม่'):>7}  {f['how']}")

# ── ผลบนใบ A ──────────────────────────────────────────────────────────────
fields = merge(DOC_A, read_by_label(DOC_A), read_by_structure(DOC_A), INVOICE_SCHEMA)
show_fields(fields)

print(f"\n★ = ช่องสำคัญ: ถ้าค่านี้ผิด มีคนได้เงินผิด")
print(f"ความมั่นใจทั้งใบ: {document_confidence(fields):.0%}")

# ── ทำไม 'ตรงกัน' ทั้งที่เขียนคนละแบบ ─────────────────────────────────────
differ = [(n, f) for n, f in fields.items()
          if f["raw_a"] and f["raw_b"] and f["raw_a"] != f["raw_b"]]
print("\n" + "─" * 82)
if differ:
    print("สองช่องพิมพ์คนละแบบ แต่นับว่าตรงกัน เพราะเทียบหลังทำเป็นรูปแบบมาตรฐาน:")
    print(f"\n  {'FIELD':16} {'ช่อง A':20} {'ช่อง B':20} เทียบแล้ว")
    for name, f in differ:
        print(f"  {name:16} {f['raw_a'][:19]:20} {f['raw_b'][:19]:20} "
              f"{canonical(f['raw_a'])} == {canonical(f['raw_b'])}")
else:
    print("บนใบนี้ สองช่องพิมพ์ออกมาเหมือนกันทุกช่อง ไม่มีอะไรต้องทำให้เป็นมาตรฐาน")
    print("(ใบ C ภาษาไทยจะเห็นชัดกว่า — ตอนรันทั้งสี่ใบท้าย notebook)")
''')

# ================================================================= rules ====
md(r"""
## ⑥ กฎธุรกิจ

ตอนนี้ค่าถูกอ่านมาแล้ว ถึงเวลาตรวจว่าเอกสารนี้ควรมีอยู่หรือเปล่า สองแบบ:

1. **มันขัดกับตัวเองไหม** — บรรทัดรวมเป็นยอดก่อนภาษี · ยอดก่อนภาษี × อัตรา = ภาษี · ยอดก่อนภาษี + ภาษี = ยอดรวม **คำนวณใหม่ ไม่เชื่อที่พิมพ์มา**
2. **มันตรงกับที่เราบันทึกไว้ไหม** — ผู้ขายอยู่ในทะเบียน · ใบสั่งซื้อมีจริงและยังเปิด · เลขบัญชีตรงกับที่จดไว้ · ไม่เคยโพสต์ไปแล้ว

### สองค่าความคลาดเคลื่อน ห้ามรวมกันเด็ดขาด

| | คือ | ค่า |
|---|---|---|
| `AMOUNT_TOLERANCE` | ใบแจ้งหนี้ต่างจาก**ใบสั่งซื้อ**ได้แค่ไหน — ผ่อนผันทางการค้า | 2% |
| `ARITHMETIC_TOLERANCE` | เอกสารต่างจาก**ตัวเอง**ได้แค่ไหน — แค่การปัดเศษ | 0.01% |

ตอนใช้ตัวเดียวกัน การอ่านผิด 810 บาทผ่านไปได้ในฐานะ "ยอดรวมถูกต้อง" บนใบ 200,090 บาท

### สามระดับความรุนแรง และมันต้องทำงานต่างกันจริง

| | ทำอะไร |
|---|---|
| `blocking` | **หยุดเอกสาร** ไม่มีอะไรผ่านไปได้แม้จะกดอนุมัติ |
| `warning` | บันทึกไว้ ให้คนเห็น แต่**ไม่หยุดอะไรทั้งนั้น** |
| `info` | แค่หมายเหตุ |

> คำเตือนที่หยุดเอกสารได้ คือกฎ blocking ที่ยังไม่ยอมรับตัวเอง — และนั่นคือวิธีที่ระบบ human-in-the-loop กลายเป็นระบบที่มีคนอยู่ในทุก loop
""")

code(r'''
# ── กฎ ────────────────────────────────────────────────────────────────────
AMOUNT_TOLERANCE     = 0.02      # เทียบกับใบสั่งซื้อ — ผ่อนผันทางการค้า
ARITHMETIC_TOLERANCE = 0.0001    # เทียบกับตัวเอง — แค่ปัดเศษ ห้ามรวมกับข้างบน

def num(fields, name):
    if name not in fields:
        return None
    try:
        return money(str(fields[name]["value"]))
    except ValueError:
        return None

def text_of(fields, name):
    return str(fields[name]["value"]).strip() if name in fields else None

def find_vendor(name, tax_id):
    """หาผู้ขาย — เลขภาษีก่อน เพราะเป็นตัวระบุที่แน่นอนที่สุด
    แล้วค่อยชื่อ ซึ่งจับ alias ภาษาไทยด้วย"""
    for vid, v in VENDORS.items():
        if tax_id and v["tax_id"] == str(tax_id).strip():
            return vid, v, "เลขผู้เสียภาษี"
    if name:
        target = canonical(name)
        for vid, v in VENDORS.items():
            for candidate in [v["name"], *v["aliases"]]:
                if canonical(candidate) == target:
                    return vid, v, "ชื่อ"
    return None, None, "ไม่พบ"

def run_rules(fields):
    """คืน list ของ (rule_id, ผ่านไหม, ระดับ, ข้อความ)"""
    out = []
    def check(rule_id, ok, severity, message):
        out.append((rule_id, ok, severity, message))

    subtotal, tax, total = num(fields, "subtotal"), num(fields, "tax_amount"), num(fields, "total_amount")
    rate = num(fields, "tax_rate")

    # ① ขัดกับตัวเองไหม — คำนวณใหม่ ไม่เชื่อที่พิมพ์
    if None not in (subtotal, tax, total):
        slack = max(0.01, abs(total) * ARITHMETIC_TOLERANCE)
        ok = abs(subtotal + tax - total) <= slack
        check("totals_consistent", ok, "blocking",
              f"{subtotal:,.2f} + {tax:,.2f} = {subtotal + tax:,.2f} "
              f"แต่เอกสารเขียน {total:,.2f}" if not ok else "ยอดรวมถูกต้อง")

    if None not in (subtotal, tax, rate):
        expected = round(subtotal * rate / 100, 2)
        ok = abs(expected - tax) <= max(0.01, abs(expected) * ARITHMETIC_TOLERANCE)
        check("tax_rate_consistent", ok, "blocking",
              f"อัตรา {rate:.0f}% ของ {subtotal:,.2f} คือ {expected:,.2f} "
              f"แต่คิดมา {tax:,.2f}" if not ok else f"ภาษีตรงกับอัตรา {rate:.0f}%")

    # ② ตรงกับที่เราบันทึกไว้ไหม
    vid, vendor, how = find_vendor(text_of(fields, "vendor_name"), text_of(fields, "vendor_tax_id"))
    check("vendor_known", vendor is not None, "blocking",
          f"ไม่รู้จักผู้ขาย '{text_of(fields, 'vendor_name')}'" if not vendor
          else f"พบผู้ขาย {vid} (จาก{how})")

    if vendor:
        check("vendor_active", vendor["status"] == "active", "blocking",
              f"ผู้ขายสถานะ {vendor['status']} — จ่ายอัตโนมัติไม่ได้"
              if vendor["status"] != "active" else "ผู้ขายยังใช้งานอยู่")

        printed = text_of(fields, "bank_account")
        if printed:
            ok = printed == vendor["bank_account"]
            check("bank_account_matches", ok, "blocking",
                  f"บนเอกสารเขียน {printed} แต่ทะเบียนจดไว้ {vendor['bank_account']}"
                  if not ok else "เลขบัญชีตรงกับทะเบียน")

    po_ref = text_of(fields, "purchase_order")
    if po_ref:
        po = PURCHASE_ORDERS.get(po_ref)
        check("po_exists", po is not None, "blocking",
              f"ไม่มีใบสั่งซื้อ {po_ref} ในระบบ" if not po else f"พบใบสั่งซื้อ {po_ref}")
        if po:
            check("po_open", po["status"] == "open", "blocking",
                  f"ใบสั่งซื้อ {po_ref} สถานะ {po['status']}"
                  if po["status"] != "open" else "ใบสั่งซื้อยังเปิดอยู่")
            if subtotal is not None:
                # เทียบกับ *ยอดก่อนภาษี* เพราะใบสั่งซื้อเก็บยอดไม่รวม VAT
                ok = subtotal <= po["amount"] * (1 + AMOUNT_TOLERANCE)
                check("po_amount_ok", ok, "blocking",
                      f"เรียกเก็บ {subtotal:,.2f} เกินใบสั่งซื้อ {po['amount']:,.2f}"
                      if not ok else "ยอดอยู่ในวงเงินใบสั่งซื้อ")

    ref = text_of(fields, "invoice_number")
    already = next((r for r in LEDGER if r["invoice_number"] == ref), None)
    check("not_duplicate", already is None, "blocking",
          f"เลขที่ {ref} โพสต์ไปแล้วเป็น {already['record_id']}" if already
          else "ยังไม่เคยโพสต์")

    # ③ คำเตือน — บันทึกไว้ แต่ไม่หยุดอะไร
    if rate is not None:
        check("vat_rate_expected", rate in (0, 7), "warning",
              f"อัตรา VAT {rate:.0f}% ไม่ใช่ค่าที่พบทั่วไปในไทย"
              if rate not in (0, 7) else "อัตรา VAT ปกติ")

    return out

def show_rules(findings):
    failed = [f for f in findings if not f[1]]
    print(f"ตรวจ {len(findings)} ข้อ · ผ่าน {len(findings) - len(failed)} · ไม่ผ่าน {len(failed)}\n")
    for severity in ("blocking", "warning", "info"):
        hits = [f for f in failed if f[2] == severity]
        if not hits:
            continue
        effect = {"blocking": "หยุดเอกสาร", "warning": "แสดงให้เห็น แต่ไม่หยุด",
                  "info": "หมายเหตุ"}[severity]
        print(f"  {severity.upper()}  ({effect})")
        for rule_id, _, _, message in hits:
            print(f"     {rule_id:22} {message}")
    if not failed:
        print("  ผ่านทั้งหมด:")
        for rule_id, _, _, message in findings:
            print(f"     ok  {rule_id:22} {message}")
    return failed

findings = run_rules(fields)
_ = show_rules(findings)
''')

# ================================================================ route =====
md(r"""
## ⑦ ตัดสิน แล้วโพสต์

สองตัวเลขมาเจอกันตรงนี้ และมันไม่ใช่เรื่องเดียวกัน

**ความมั่นใจ** เรื่อง*การอ่าน* — เรามั่นใจแค่ไหนว่าค่าเหล่านี้คือสิ่งที่พิมพ์อยู่บนหน้า

**การตัดสิน** เรื่อง*นโยบาย* — ใบนี้โพสต์ได้ไหม ตัดสินด้วย**โค้ด** จากผลกฎบวกความมั่นใจ

การแยกสองอย่างนี้คือสิ่งที่ทำให้คำเตือนได้เป็นแค่คำเตือนจริงๆ

### `critical` คือเรื่อง *ผลที่ตามมา* ไม่ใช่เรื่อง *ต้องมี*

แปลว่า: ถ้าค่านี้อยู่ตรงนี้แล้วมันผิด มีคนได้เงินผิด — **ไม่ได้แปลว่าเอกสารต้องมีค่านี้** ใบแจ้งหนี้จำนวนมากไม่อ้างใบสั่งซื้อเลย (ใบเล็ก ค่าสมาชิก เบิกค่าใช้จ่าย) และคำถามว่า*ต้องมีไหม*เป็นนโยบายธุรกิจ ตอบด้วยกฎแยกต่างหากที่ปรับได้
""")

code(r'''
import hashlib

AUTO_POST_MIN_CONFIDENCE = 0.84      # ต่ำกว่านี้ ให้คนดู

def decide(fields, findings, threshold=AUTO_POST_MIN_CONFIDENCE):
    """โพสต์ หรือหยุด — ตัดสินด้วยโค้ด ไม่มีโมเดลอยู่ตรงนี้"""
    reasons = []

    for rule_id, ok, severity, message in findings:
        if not ok and severity == "blocking":
            reasons.append(message)

    # ค่าสำคัญที่ *มีอยู่* แต่ชี้ที่มาบนหน้าไม่ได้
    # สังเกต `f["value"]` — ค่าที่ไม่มีเลย ไม่ใช่ปัญหาตรงนี้
    ungrounded = [n for n, f in fields.items()
                  if f["critical"] and f["value"] and not f["grounded"]]
    if ungrounded:
        reasons.append(f"ยืนยันกับหน้าเอกสารไม่ได้: {', '.join(ungrounded)}")

    confidence = document_confidence(fields)
    if confidence < threshold:
        reasons.append(f"ความมั่นใจ {confidence:.0%} ต่ำกว่าเกณฑ์ {threshold:.0%}")

    return ("needs_review" if reasons else "auto_post"), confidence, reasons

def post(fields, confidence):
    """เขียนลงบัญชี — ปลายทางที่ทุกอย่างมุ่งไป"""
    record_id = "REC-" + hashlib.sha1(
        str(fields["invoice_number"]["value"]).encode()).hexdigest()[:8].upper()
    record = {
        "record_id": record_id,
        "invoice_number": str(fields["invoice_number"]["value"]),
        "vendor": str(fields["vendor_name"]["value"]),
        "amount": money(str(fields["total_amount"]["value"])),
        "confidence": confidence,
    }
    LEDGER.append(record)
    return record

# ── ใบ A จนถึงปลายทาง ─────────────────────────────────────────────────────
decision, confidence, reasons = decide(fields, findings)

print(f"ความมั่นใจ : {confidence:>5.0%}   (เกณฑ์ {AUTO_POST_MIN_CONFIDENCE:.0%})")
print(f"ตัดสิน     : {decision.upper()}\n")
if reasons:
    for r in reasons:
        print("  •", r)
else:
    critical_found = [n for n, f in fields.items() if f["critical"] and f["value"]]
    absent = [n for n, s in INVOICE_SCHEMA.items()
              if s["critical"] and n not in critical_found]
    print(f"  • ไม่มีข้อไหน blocking · ค่าสำคัญ {len(critical_found)} ค่า ชี้ที่มาบนหน้าได้ทุกค่า"
          + (f" ({len(absent)} ค่าไม่ได้พิมพ์บนใบนี้: {', '.join(absent)})" if absent else ""))

if decision == "auto_post":
    record = post(fields, confidence)
    print(f"\n{'─' * 60}\nโพสต์ลงบัญชีแล้ว")
    for k, v in record.items():
        print(f"  {k:16} {v if not isinstance(v, float) else f'{v:,.2f}'}")
else:
    print("\nส่งเข้าคิวให้คนตรวจ")

print(f"\nตอนนี้บัญชีมี {len(LEDGER)} รายการ")
''')

# ====================================================== rule from words =====
md(r"""
## ⑧ เขียนกฎด้วยการพิมพ์ประโยค

กฎที่มีมาให้เป็นนโยบายการเงินแบบทั่วไป ของคุณไม่เหมือน และมันเปลี่ยนบ่อย — มักจะเปลี่ยนในวันที่ไม่มีใครว่างเปิดโปรแกรมแก้โค้ด

ระบบจริง**เรียกโมเดลหนึ่งครั้งตรงนี้** เพื่อแปลประโยคเป็นเงื่อนไข แล้ว**ไม่เรียกอีกเลย** โมเดลไม่เคยเห็นเอกสาร กฎที่ได้ถูกอ่านกลับให้คนเขียนดูเป็นภาษาคน เก็บเป็นข้อมูล แล้วจากนั้นประเมินด้วยโค้ดล้วนๆ

เส้นแบ่งตรงนี้คือทั้งหมดของการออกแบบ:

> ถามโมเดลทุกใบว่า *"ใบนี้ VAT ถูกไหม"* → คำตอบขึ้นกับถ้อยคำบนเอกสารเอง ซึ่งเป็นสิ่งที่ผู้ขายมักง่าย หรือคนที่ตั้งใจโกง เป็นคนคุม
>
> ถามครั้งเดียว ต่อหน้าคนที่จะอนุมัติผลลัพธ์ → ไม่เป็นแบบนั้น

ใน notebook นี้ตัวแปลเป็นการจับ pattern ง่ายๆ ให้เห็นว่า *ผลลัพธ์*หน้าตาเป็นยังไง — ซึ่งเป็นส่วนที่สำคัญ เพราะสิ่งที่ถูกเก็บและถูกรันคือเงื่อนไขนี้ ไม่ใช่ประโยค

### ระวังคำว่า `assert`

`assert` คือ**สิ่งที่ต้องเป็นจริงเอกสารถึงจะผ่าน** ประโยคที่พูดว่า "เตือนเมื่อ X" ต้องแปลเป็น `assert: ไม่ใช่ X`

แปลกลับด้านคือได้กฎที่**เด้งกับเอกสารดีทุกใบ และปล่อยเอกสารเสียผ่านทุกใบ**
""")

code(r'''
import re

# ── ตัวแปลประโยค → เงื่อนไข ───────────────────────────────────────────────
# ระบบจริงใช้โมเดลตรงนี้ ครั้งเดียว ตอนเขียนกฎ ที่นี่จับ pattern ให้เห็นว่า
# *ผลลัพธ์* หน้าตายังไง เพราะสิ่งที่ถูกเก็บและถูกรันคือเงื่อนไข ไม่ใช่ประโยค

FIELD_WORDS = {
    "ยอดรวม": "total_amount", "ยอดเงิน": "total_amount", "total": "total_amount",
    "ยอดก่อนภาษี": "subtotal", "subtotal": "subtotal",
    "ใบสั่งซื้อ": "purchase_order", "po": "purchase_order",
    "เลขบัญชี": "bank_account",
    "เลขประจำตัวผู้เสียภาษี": "vendor_tax_id", "เลขที่ผู้เสียภาษี": "vendor_tax_id",
}
# ช่องที่เทียบด้วยตัวเลขได้ — ช่องอื่นเทียบได้แค่ "มี/ไม่มี"
NUMERIC = {"total_amount", "subtotal", "tax_amount"}

def mentioned(sentence):
    """ช่องที่ประโยคพูดถึง เรียงตามตำแหน่งที่ปรากฏ"""
    low = sentence.lower()
    seen = []
    for word, field in FIELD_WORDS.items():
        at = low.find(word)
        if at >= 0 and field not in [f for _, f in seen]:
            seen.append((at, field))
    return [f for _, f in sorted(seen)]

def compile_rule(sentence):
    """ประโยค → กฎ  คืน dict ที่อ่านกลับเป็นภาษาคนได้ และรันได้"""
    low = sentence.lower()
    fields = mentioned(sentence)
    amount = re.search(r"([\d,]{4,})\s*(?:บาท|thb)?", low)
    limit = money(amount.group(1)) if amount else None
    severity = "blocking" if re.search(r"ห้าม|ปฏิเสธ|reject|block", low) else "warning"

    # ช่องที่ประโยคบอกว่า "ต้องมี"
    need = next((f for word, f in FIELD_WORDS.items()
                 if word in low and re.search(rf"ต้องมี[^\.]*{re.escape(word)}", low)), None)

    # เกณฑ์ตัวเลขต้องอยู่บนช่องที่เป็นตัวเลข ไม่งั้นเทียบไม่ได้จริง
    # นี่คือจุดที่เคยพัง: "ใบสั่งซื้อ" ถูกจับเป็นทั้งเกณฑ์และสิ่งที่ต้องมี
    # แล้วไปเทียบ 'PO-2026-0567' <= 100000
    threshold_field = next((f for f in fields if f in NUMERIC and f != need), None)
    if limit and threshold_field is None:
        threshold_field = "total_amount"      # ประโยคพูดถึงจำนวนเงินโดยไม่ระบุช่อง

    # ① "เกิน X ต้องมี Y"
    if limit and need and need != threshold_field:
        return {"id": f"over_{limit:.0f}_needs_{need}", "prompt": sentence,
                "severity": severity,
                "when":   {"field": threshold_field, "op": "gt", "value": limit},
                "assert": {"field": need, "op": "present"},
                "reads":  f"เมื่อ {threshold_field} เกิน {limit:,.0f} ต้องมี {need}"}

    # ② "ห้ามเกิน X"
    if limit and threshold_field:
        return {"id": f"{threshold_field}_max_{limit:.0f}", "prompt": sentence,
                "severity": severity, "when": None,
                "assert": {"field": threshold_field, "op": "lte", "value": limit},
                "reads":  f"ต้องมี {threshold_field} ไม่เกิน {limit:,.0f}"}

    # ③ "ต้องมี X"
    target = need or (fields[0] if fields else None)
    if target:
        return {"id": f"{target}_required", "prompt": sentence, "severity": severity,
                "when": None, "assert": {"field": target, "op": "present"},
                "reads": f"ต้องมี {target}"}
    return None

def evaluate(condition, fields):
    if condition is None:
        return True
    value = fields.get(condition["field"], {}).get("value")
    op = condition["op"]
    if op == "present":
        return value not in (None, "")
    if value is None:
        return True                       # ไม่มีค่าให้เทียบ = ไม่ตัดสิน
    try:
        left = money(str(value))
    except ValueError:
        # เทียบตัวเลขกับค่าที่ไม่ใช่ตัวเลข คือกฎที่เขียนผิด ไม่ใช่เอกสารที่ผิด
        raise ValueError(f"เทียบ {condition['field']}={value!r} กับตัวเลขไม่ได้ "
                         f"— กฎข้อนี้เขียนผิด") from None
    return left > condition["value"] if op == "gt" else left <= condition["value"]

def run_custom(rule, fields):
    """คืน (ผ่านไหม, ข้อความ) หรือ None ถ้ากฎนี้ไม่เกี่ยวกับเอกสารนี้"""
    if not evaluate(rule["when"], fields):
        return None                       # `when` ไม่ตรง — ไม่ใช่ผ่าน ไม่ใช่ตก
    ok = evaluate(rule["assert"], fields)
    return ok, ("เข้าเงื่อนไข" if ok else f"ไม่ผ่าน: {rule['reads']}")

# ── พิมพ์ประโยค แล้วรันกับใบ A ทันที ──────────────────────────────────────
SENTENCES = [
    "ใบแจ้งหนี้เกิน 100,000 บาท ต้องมีใบสั่งซื้ออ้างอิง",
    "ทุกใบต้องมีเลขประจำตัวผู้เสียภาษีของผู้ขาย",
    "ห้ามยอดรวมเกิน 500,000 บาท",
    "ยอดรวมห้ามเกิน 100,000 บาท",
]

print(f"{'ประโยคที่พิมพ์':46} {'ระดับ':9} ผลกับใบ A")
print("─" * 94)
demo_rules = []
for sentence in SENTENCES:
    rule = compile_rule(sentence)
    if not rule:
        print(f"{sentence[:45]:46} {'—':9} แปลไม่ได้")
        continue
    demo_rules.append(rule)
    outcome = run_custom(rule, fields)
    verdict = ("ไม่เข้าเงื่อนไขของใบนี้" if outcome is None
               else "ผ่าน" if outcome[0] else f"ไม่ผ่าน ({rule['severity']})")
    print(f"{sentence[:45]:46} {rule['severity']:9} {verdict}")

print(f"\nแปลได้ {len(demo_rules)} กฎ · อ่านกลับเป็นภาษาคน แล้วเก็บเป็นข้อมูล:\n")
import json as _json
for rule in demo_rules:
    print(f"  {rule['id']}")
    print(f"     อ่านว่า : {rule['reads']}")
    print(f"     เก็บเป็น: when={_json.dumps(rule['when'], ensure_ascii=False)}  "
          f"assert={_json.dumps(rule['assert'], ensure_ascii=False)}")
''')

# ============================================================== all four ====
md(r"""
## ⑨ ทั้งสี่ใบ ตั้งแต่ต้นจนจบ

ทุกขั้นที่เดินมาทีละก้อน มัดรวมเป็นฟังก์ชันเดียว แล้วปล่อยเอกสารทั้งสี่ใบเข้าไป

อ่านคอลัมน์สุดท้ายให้ดี เพราะความผิดสองแบบนี้**ไม่เท่ากัน**:

- ใบที่ควรถูกหยุด แต่ผ่านไป = **จ่ายเงินผิดโดยไม่มีใครรู้**
- ใบที่ควรผ่าน แต่ถูกหยุด = **เสียเวลาคนไปหนึ่งนาที**

ระบบจริงจึงไม่เคยเอาสองอย่างนี้มาหารรวมกัน แต่ละอย่างมีตัวหารของตัวเอง
""")

code(r'''
# ── ทุกขั้น มัดเป็นฟังก์ชันเดียว ──────────────────────────────────────────

def process(page, extra_rules=()):
    """เอกสารเข้า → รายการบัญชี หรือคิวให้คนดู ออก"""
    kind, _, _ = classify(page)                                   # ③
    fields = merge(page, read_by_label(page),                     # ④ + ⑤
                         read_by_structure(page), INVOICE_SCHEMA)
    findings = list(run_rules(fields))                            # ⑥
    for rule in extra_rules:                                      # ⑥ กฎที่เขียนเอง
        outcome = run_custom(rule, fields)
        if outcome is not None:
            findings.append((rule["id"], outcome[0], rule["severity"], outcome[1]))
    decision, confidence, reasons = decide(fields, findings)      # ⑦
    record = post(fields, confidence) if decision == "auto_post" else None
    return {"kind": kind, "fields": fields, "findings": findings, "decision": decision,
            "confidence": confidence, "reasons": reasons, "record": record}

# สิ่งที่ควรเกิดขึ้น — เขียนไว้ก่อนถามระบบ
EXPECTED = {"A · ปกติ": "auto_post",           "B · VAT ผิด": "needs_review",
            "C · ภาษาไทย": "auto_post",        "D · เลขบัญชีถูกเปลี่ยน": "needs_review"}

LEDGER.clear()                       # เริ่มจากบัญชีว่าง ใบ A จะได้ไม่ซ้ำกับตัวเอง

print(f"{'เอกสาร':26} {'ประเภท':9} {'ตัดสิน':13} {'มั่นใจ':>6} {'ควรเป็น':>13}  ผล")
print("─" * 96)
results = {}
for name, page in DOCUMENTS.items():
    r = results[name] = process(page)
    want = EXPECTED[name]
    verdict = ("ถูก" if r["decision"] == want
               else "จ่ายผิดโดยไม่รู้ตัว" if r["decision"] == "auto_post" else "เตือนเกินจริง")
    print(f"{name:26} {r['kind']:9} {r['decision']:13} {r['confidence']:>5.0%} "
          f"{want:>13}  {verdict}")

print("\n" + "═" * 96)
for name, r in results.items():
    if r["decision"] == "auto_post":
        continue
    print(f"\n{name} — ถูกหยุดเพราะ:")
    for rule_id, ok, severity, message in r["findings"]:
        if not ok:
            print(f"   [{severity:8}] {rule_id:22} {message}")

print("\n" + "═" * 96)
print(f"บัญชีปิดท้ายด้วย {len(LEDGER)} รายการ:\n")
for rec in LEDGER:
    print(f"  {rec['record_id']}  {rec['invoice_number']:16} {rec['vendor'][:32]:34} "
          f"{rec['amount']:>12,.2f}  ({rec['confidence']:.0%})")
''')

# =============================================================== closing ====
md(r"""
## สิ่งที่เพิ่งเห็น

| | |
|---|---|
| **อ่าน 2 ครั้ง** | คนละวิธี ไม่คุยกัน — ตรงกันเมื่อไหร่ ค่านั้นน่าเชื่อขึ้น |
| **0 โมเดลในการตัดสิน** | grounding · กฎ · ความมั่นใจ · การตัดสิน เป็นโค้ดทั้งหมด |
| **1 โมเดลตอนเขียนกฎ** | ครั้งเดียว ต่อหน้าคนที่อนุมัติ ไม่ใช่ทุกครั้งที่เอกสารมา |
| **ทุกค่าชี้ที่มาได้** | และข้อความอ้างอิงถูกเอาไปตรวจกับหน้าจริง ไม่ใช่เชื่อ |
| **คำเตือนไม่หยุดอะไร** | ไม่งั้นมันคือกฎ blocking ที่ยังไม่ยอมรับตัวเอง |

### ใบ D คือใบที่ควรจำ

อ่านถูกทุกช่อง เลขบวกลบถูกหมด ผู้ขายมีจริงและยังใช้งานอยู่ ใบสั่งซื้อมีจริงและยังเปิด — **ไม่มีอะไรบนหน้ากระดาษผิดเลยสักอย่าง**

แต่จ่ายไม่ได้ เพราะเลขบัญชีไม่ตรงกับที่ทะเบียนจดไว้

ข้อมูลนั้น**ไม่ได้อยู่บนหน้ากระดาษและไม่มีวันอยู่** OCR ดีขึ้น อ่านสามรอบ หรือตั้งค่าความคลาดเคลื่อนเข้มขึ้น ก็หาไม่เจอ ทางเดียวคือไปเปิดทะเบียนดู

**นี่คือเหตุผลทั้งหมดที่ต้องมี master data** — และเหตุผลที่ระบบวัดผลด้วย *การตัดสินใจ* ไม่ใช่แค่ความแม่นของช่องข้อมูล เอกสารที่อ่านถูก 100% ก็ยังเป็นเอกสารที่ไม่ควรจ่ายได้

---

### ที่ย่อไว้จากของจริง

| ตรงนี้ | ระบบจริง |
|---|---|
| ข้ามขั้น OCR เขียนข้อความไว้เลย | Typhoon OCR 1.5 + ปรับภาพก่อนอ่าน (แก้เอียง แสง คมชัด) โดยวัดผลก่อน-หลังทุกครั้ง แล้วย้อนกลับถ้าไม่ดีขึ้น |
| สองช่องเป็นโค้ดสองแนวคิด | Typhoon กับ Claude — คนละบริษัท คนละโมเดล |
| 10 กฎ | 26 กฎในตัว + กฎที่เขียนเอง รวมถึงกฎที่ต้องเรียก tool ไปค้นข้อมูล |
| ตัวแปลกฎจับ pattern | โมเดลแปลประโยค ครั้งเดียวตอนเขียน |
| 4 ใบ | 30 ใบ วัดผล: อ่านถูกกี่ %, กฎแต่ละข้อยิงถูกไหม, ตัดสินถูกกี่ % |
| `LEDGER` เป็น list | SQLite + คิวรีวิว + API + หน้าเว็บ |

**แต่ลำดับขั้นและหลักการ เหมือนกันทุกอย่าง**

อยากเห็นของจริงทำงาน: `docuflow_explained.ipynb` เดินขั้นเดียวกันนี้ผ่าน pipeline จริง และ `docuflow_walkthrough.ipynb` รันชุดกฎเต็ม — ทั้งคู่ต้องมี repo และ API key สองตัว

---

### ลองเล่นดู

1. แก้ `DOC_A` ให้ยอดรวมผิดสัก 100 บาท → `totals_consistent` จะจับได้ และบอกด้วยว่าต่างเท่าไหร่
2. เปลี่ยน `VENDORS["V-1001"]["status"]` เป็น `"on_hold"` → ใบ A จะถูกหยุด
3. เขียนประโยคใหม่ใน `SENTENCES` แล้วรันใหม่
4. ลบ `PO Reference` ออกจาก `DOC_A` → **ยังผ่านอยู่** เพราะ `critical` แปลว่า "ถ้ามีแล้วผิดจะเสียหาย" ไม่ใช่ "ต้องมี"
5. เปลี่ยน `AUTO_POST_MIN_CONFIDENCE` เป็น `0.99` → ทุกใบไปเข้าคิวคน ซึ่งคือสิ่งที่เกิดขึ้นจริงตอนตั้งเกณฑ์สูงเกิน
""")

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "docuflow_miniature.ipynb", "provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {OUT}  ({len(cells)} cells)")
