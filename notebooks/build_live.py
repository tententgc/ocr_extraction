"""Generate the live-pipeline Colab notebook.

The fourth notebook, and the one to hand somebody who says "show me it working
on my document". You upload a file, and it goes through the real thing: Typhoon
OCR reads the page, two different models read the text independently, their
answers are merged and grounded, three rules and one tool-calling agent check it,
and code — only code — decides `POST = True` or `False`.

What separates it from `explained` and `walkthrough` is that **nothing is
imported from `docuflow`**. Every stage is written out in the notebook against
the vendors' own SDKs. That costs some fidelity (no preprocessing, three rules
instead of twenty-six, the register is a dict) and buys the thing those two
cannot give: no repository, no token, no `pip install -e`, no private clone.
Two API keys from Colab's secret store and a file, and it runs.

What separates it from `miniature` is that it is not a simulation. `miniature`
replaces the models with regexes so it needs no keys at all; this one calls
them, prints both prompts in full before it does, and shows every tool call the
rule agent makes with the payload that came back.

Written in Thai, because the people it is for read Thai.

The outputs below were captured from a real run on Colab — Typhoon OCR on an
uploaded PNG, Typhoon and Claude reading it, Claude calling two tools. They are
spliced in at build time because this notebook cannot be executed locally: it
needs an uploaded document and live keys. Re-running the notebook in Colab
replaces them.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("notebooks/docuflow_live.ipynb")

cells: list[dict] = []
# Captured stdout, keyed by the code cell's position among code cells.
outputs: dict[int, str] = {}
_code_index = 0


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text: str, output: str = "") -> None:
    global _code_index
    cell = {
        "cell_type": "code",
        "execution_count": _code_index + 1 if output else None,
        "metadata": {},
        "outputs": ([{"output_type": "stream", "name": "stdout",
                      "text": output.strip("\n").splitlines(True)}] if output else []),
        "source": text.strip("\n").splitlines(True),
    }
    cells.append(cell)
    _code_index += 1


# ================================================================= title ====
md(r"""
# DocuFlow — เอกสารจริงของคุณ ผ่านทุกขั้นจริง จบที่ POST หรือไม่ POST

**อัปโหลดเอกสารของคุณเอง** แล้วดูมันเดินผ่านทุกขั้น จนได้คำตอบสุดท้าย

โค้ดทั้งหมดเขียนอยู่ใน notebook นี้ — ไม่ clone repo ไม่ import `docuflow` แต่**ทุกขั้นเป็นของจริง**: OCR จริง, โมเดลจริงสองตัว, agent ที่เรียก tool จริง เห็น prompt ทุกอัน เห็น output ทุกขั้น

---

## ทุกขั้นที่จะเกิดขึ้น

| ขั้น | ทำอะไร | ใคร |
|---|---|---|
| ① **OCR** | รูป/PDF → ข้อความ | **Typhoon OCR 1.5** (จริง) |
| ② **แยกประเภท** | ใบแจ้งหนี้? ใบเสร็จ? | คำสำคัญ + โมเดลถ้าสูสี |
| ③ **Agent 1 อ่าน** | หาค่าทุกช่อง + อ้างอิงที่มา | **Typhoon** (จริง) |
| ④ **Agent 2 อ่านซ้ำ** | คำถามเดียวกัน คนละโมเดล คนละ prompt | **Claude** (จริง) |
| ⑤ **รวม + ตรวจ** | สองตัวตรงกันไหม · ค่าอยู่บนหน้าจริงไหม | โค้ด |
| ⑥ **กฎ** | เลขขัดกันเองไหม · ตรงทะเบียนไหม | โค้ด |
| ⑦ **Agent ตรวจกฎ** | เรื่องที่ต้องไปค้นข้อมูลก่อนถึงตอบได้ | **Claude + tool calling** (จริง) |
| ⑧ **ตัดสิน** | **POST True / False** | โค้ดเท่านั้น |

## ผลลัพธ์ที่จะได้ตอนจบ

```
ประเภทเอกสาร     : invoice
ทุก field ที่อ่านได้ : พร้อมค่า ความมั่นใจ และที่มาบนหน้า
ผ่านกฎไหม        : ทีละข้อ พร้อมเหตุผล
Confidence       : xx%
POST             : True / False
```

---

## หลักการเดียวที่ต้องจำ

> ## โมเดล**อ่าน** · โค้ด**ตัดสิน**

โมเดลเก่งเรื่องหาตัวเลขบนหน้าที่ไม่เคยเห็น แต่**ผิดคน**ที่จะถามว่าจ่ายได้ไหม เพราะเอกสารเองพูดโน้มน้าวมันได้ — ขั้น ⑧ จึงไม่มีโมเดลอยู่เลย

**เวลารันทั้งหมด ~40–60 วินาที** (เรียก API 4–6 ครั้ง)
""")

# ============================================================ install/keys ===
code(r'''
#@title ติดตั้ง + คีย์  { display-mode: "form" }
# สามอย่าง: ตัวเรียก Typhoon OCR, OpenAI SDK (ใช้คุยกับ Typhoon), Anthropic SDK
%pip install -q typhoon-ocr openai anthropic

import os, getpass

def read_key(name, required=True):
    """เอาจาก 🔑 panel ถ้ามี — ตั้งครั้งเดียวอยู่ในบัญชี Google ไม่หายตอน disconnect"""
    if os.environ.get(name):
        return os.environ[name], "มีอยู่ใน session แล้ว"
    try:
        from google.colab import userdata
        value = (userdata.get(name) or "").strip()
        if value:
            return value, "จาก 🔑 panel"
    except ImportError:
        pass
    except Exception as exc:
        print(f"   {name}: {type(exc).__name__} — เพิ่มใน 🔑 panel แล้วเปิด Notebook access")
    prompt = f"{name}: " if required else f"{name} (ไม่ใส่ก็ได้ กด Enter ข้าม): "
    return getpass.getpass(prompt).strip(), "พิมพ์เข้ามา (หายเมื่อ runtime ดับ)"

for name, required in (("TYPHOON_API_KEY", True), ("ANTHROPIC_API_KEY", False)):
    os.environ[name], where = read_key(name, required)
    print(f"{name:18} {where if os.environ[name] else 'ไม่มี'}")

TYPHOON_KEY   = os.environ["TYPHOON_API_KEY"]
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

print()
print("Agent 1 (อ่านรอบแรก) : Typhoon  typhoon-v2.5-30b-a3b-instruct")
print("Agent 2 (อ่านรอบสอง) :", "Claude  claude-sonnet-5" if ANTHROPIC_KEY
      else "ไม่มีคีย์ — จะข้ามการอ่านรอบสอง และความมั่นใจจะต่ำลงตามจริง")
print("OCR                  : Typhoon OCR 1.5 (cloud)")
''', """
TYPHOON_API_KEY    จาก 🔑 panel
ANTHROPIC_API_KEY  จาก 🔑 panel

Agent 1 (อ่านรอบแรก) : Typhoon  typhoon-v2.5-30b-a3b-instruct
Agent 2 (อ่านรอบสอง) : Claude  claude-sonnet-5
OCR                  : Typhoon OCR 1.5 (cloud)
""")

# ================================================================ upload ====
md(r"""
## เอกสารของคุณ

รัน cell ข้างล่างแล้ว**เลือกไฟล์จากเครื่อง** — PDF, JPG, PNG ได้หมด

ใบแจ้งหนี้ ใบเสร็จ ใบกำกับภาษี ภาษาไทยหรืออังกฤษก็ได้ ถ่ายด้วยมือถือก็ได้ เอนนิดหน่อยก็ยังอ่านออก

> ถ้ามีไฟล์อยู่ใน `/content/docs` แล้ว cell นี้จะใช้ไฟล์นั้นเลยโดยไม่ถาม — ลบไฟล์ในนั้นออกถ้าอยากอัปโหลดใหม่
""")

code(r'''
from pathlib import Path

DOCS = Path("/content/docs"); DOCS.mkdir(exist_ok=True)
existing = sorted(p for p in DOCS.iterdir() if p.is_file())

if existing:
    DOC_PATH = existing[0]
    print(f"ใช้ไฟล์ที่มีอยู่แล้ว: {DOC_PATH.name}")
    print("(ลบไฟล์ใน /content/docs ออกก่อน ถ้าอยากอัปโหลดใหม่)")
else:
    from google.colab import files
    got = files.upload()
    name = next(iter(got))
    DOC_PATH = DOCS / name
    DOC_PATH.write_bytes(got[name])
    print(f"อัปโหลดแล้ว: {DOC_PATH.name}")

print(f"\nไฟล์   : {DOC_PATH.name}")
print(f"ขนาด   : {DOC_PATH.stat().st_size:,} bytes")
print(f"ชนิด   : {DOC_PATH.suffix.lstrip('.').upper()}")

# แสดงหน้าแรกให้เห็นว่ากำลังอ่านอะไรอยู่
try:
    from IPython.display import Image, display
    if DOC_PATH.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
        display(Image(filename=str(DOC_PATH), width=460))
    else:
        print("\n(PDF — ขั้น OCR จะแปลงเป็นภาพให้เอง)")
except Exception as exc:
    print("แสดงภาพไม่ได้:", exc)
''', """
อัปโหลดแล้ว: invoice_test.png

ไฟล์   : invoice_test.png
ขนาด   : 85,367 bytes
ชนิด   : PNG
""")

# =================================================================== OCR ====
md(r"""
## ① OCR — Typhoon OCR 1.5 (ของจริง)

ภาพหรือ PDF เข้าไป ข้อความออกมา

Typhoon OCR เป็นโมเดลที่ **train มาเพื่ออ่านเอกสารโดยเฉพาะ** ไม่ใช่โมเดลมองภาพทั่วไป — และนั่นสำคัญมาก เคยทดสอบกับ `gpt-5.1` บนใบไทยที่สแกนมาไม่ชัด มันคืน**ชื่อบริษัทและที่อยู่เต็มๆ ที่ไม่มีอยู่บนหน้ากระดาษเลย** ข้อความไทยที่ดูสมเหตุสมผลมาก แต่แต่งขึ้นทั้งหมด

มันคืน**โครงสร้างมาด้วย** ไม่ใช่แค่ตัวอักษร — ตารางกลับมาเป็น HTML เพราะยอดรวมที่ยังอยู่ในช่องของมัน คือยอดรวมที่ยังหาเจอทีหลัง

สองตัวเลขที่ออกมาจากขั้นนี้ และมันคนละเรื่องกัน:

- **คุณภาพข้อความ** — คำนวณจากข้อความที่ได้กลับมา จับพวกอ่านไม่ออกเป็นขยะ
- **degraded** — หน้ากระดาษมันแย่มาตั้งแต่แรก หรือต้องใช้เครื่องมือสำรองอ่าน

หน้าหนึ่งได้คะแนนแรกดีแต่ยังผิดได้ **ข้อความที่กลับมาแบบผิดอย่างมั่นใจ ได้คะแนนเท่าข้อความที่ถูก** — นี่คือเหตุผลที่ตัวที่สองต้องมี และถูกส่งต่อไปจนถึงขั้นตัดสิน
""")

code(r'''
import re, time
from typhoon_ocr import ocr_document

started = time.perf_counter()
OCR_TEXT = ocr_document(
    pdf_or_image_path=str(DOC_PATH),
    task_type="v1.5",          # โหมดที่คืนโครงสร้าง (ตาราง = HTML) ไม่ใช่แค่ตัวอักษร
    target_image_dim=1800,     # ย่อ/ขยายให้ด้านยาวเท่านี้ก่อนส่ง
    page_num=1,
    base_url="https://api.opentyphoon.ai/v1",
    api_key=TYPHOON_KEY,
    model="typhoon-ocr",
    figure_language="English",
).strip()
elapsed = time.perf_counter() - started

# ── คุณภาพข้อความ: คำนวณจากสิ่งที่ได้กลับมา ────────────────────────────────
def score_text(text):
    """คะแนนหยาบๆ ว่าข้อความที่ได้ดูเหมือนเอกสารไหม — จับพวกอ่านออกมาเป็นขยะ"""
    if not text:
        return 0.0, ["ไม่ได้ข้อความอะไรเลย"]
    stripped = re.sub(r"<[^>]+>", " ", text)
    letters = sum(c.isalnum() or c.isspace() for c in stripped)
    ratio = letters / max(1, len(stripped))
    words = [w for w in stripped.split() if len(w) > 1]
    reasons = []
    if len(stripped) < 60:
        reasons.append("ข้อความสั้นผิดปกติ")
    if ratio < 0.75:
        reasons.append(f"สัญลักษณ์แปลกปนเยอะ ({1 - ratio:.0%})")
    if len(words) < 12:
        reasons.append("คำน้อยเกินกว่าจะเป็นเอกสาร")
    return round(min(1.0, ratio * (1 if len(words) >= 12 else 0.5)), 2), reasons

QUALITY, REASONS = score_text(OCR_TEXT)
DEGRADED = QUALITY < 0.7

print(f"เครื่องมือ  : typhoon-ocr v1.5 (cloud)")
print(f"ใช้เวลา     : {elapsed:.1f} วินาที")
print(f"ได้ตัวอักษร : {len(OCR_TEXT):,}")
print(f"คุณภาพ      : {QUALITY:.2f}   degraded={DEGRADED}")
for r in REASONS:
    print(f"              ! {r}")

print("\n" + "═" * 78)
print("ข้อความที่ OCR คืนมา — ทุกขั้นต่อจากนี้เห็นแค่ก้อนนี้")
print("═" * 78)
print(OCR_TEXT[:2000])
if len(OCR_TEXT) > 2000:
    print(f"\n… (อีก {len(OCR_TEXT) - 2000:,} ตัวอักษร)")
''', """
เครื่องมือ  : typhoon-ocr v1.5 (cloud)
ใช้เวลา     : 6.1 วินาที
ได้ตัวอักษร : 599
คุณภาพ      : 0.93   degraded=False

══════════════════════════════════════════════════════════════════════════════
ข้อความที่ OCR คืนมา — ทุกขั้นต่อจากนี้เห็นแค่ก้อนนี้
══════════════════════════════════════════════════════════════════════════════
# TAX INVOICE

Siam Tech Supply Co., Ltd.
199/22 Ratchadaphisek Road, Din Daeng, Bangkok 10400
Tax ID: 0105543000129

Invoice No .: STS-2026-0318
Invoice Date: 24 July 2026
PO Reference: PO-2026-0567

Bill To
Mirtphol Industries PCL

<table><tr><td>Description</td><td>Qty</td><td>Unit Price</td><td>Amount</td></tr><tr><td>UPS 3kVA</td><td>12</td><td>9,500.00</td><td>114,000.00</td></tr><tr><td>Installation and testing</td><td>1</td><td>21,000.00</td><td>21,000.00</td></tr></table>

Subtotal 135,000.00
VAT 7% 9,450.00
Amount Due 144,450.00

Payment Terms: Net 30
Remit to Account No: 1234567890
""")

# ============================================================== classify ====
md(r"""
## ② แยกประเภทเอกสาร

คำตอบตรงนี้เลือก **schema** — รายการช่องที่ต้องหา และช่องไหนสำคัญ — agent ทั้งสองตัวจะได้ schema นี้ไปใช้

**นับคำสำคัญก่อน เรียกโมเดลเฉพาะตอนคะแนนสูสี** ใบที่เขียน TAX INVOICE ตัวใหญ่บนหัวไม่ต้องจ่ายเงินให้โมเดลมาบอก

> **กับดักภาษาไทย:** เครื่องออกใบเสร็จไทยพิมพ์ `ใบกำกับภาษีอย่างย่อ` ซึ่งมีคำว่า `ใบกำกับภาษี` (คำของ*ใบแจ้งหนี้*) ซ้อนอยู่ ถ้าจับ substring เฉยๆ ใบเสร็จไทยทุกใบจะกลายเป็นใบแจ้งหนี้ — **นับเฉพาะคำที่เจาะจงกว่า** ภาษาอังกฤษก็เป็นเหมือนกัน (`No.` ซ้อนใน `Receipt No`)
""")

code(r'''
from openai import OpenAI

typhoon = OpenAI(api_key=TYPHOON_KEY, base_url="https://api.opentyphoon.ai/v1")
TYPHOON_MODEL = "typhoon-v2.5-30b-a3b-instruct"

KEYWORDS = {
    "invoice":  ["tax invoice", "invoice", "amount due", "subtotal",
                 "ใบกำกับภาษี", "ใบแจ้งหนี้", "จำนวนเงินทั้งสิ้น"],
    "receipt":  ["receipt", "cash", "change", "ใบเสร็จ", "ใบเสร็จรับเงิน",
                 "เงินทอน", "ใบกำกับภาษีอย่างย่อ"],   # ← คำของใบเสร็จไทย
    "contract": ["agreement", "contract", "hereby", "สัญญา", "คู่สัญญา"],
}
CONFIDENT_MARGIN = 2      # ห่างขนาดนี้ ไม่ต้องถามโมเดล

def classify(text):
    low = text.lower()
    found = {k: {w for w in words if w in low} for k, words in KEYWORDS.items()}
    every = set().union(*found.values())
    # คำที่ซ้อนอยู่ในคำที่ยาวกว่าและเจอเหมือนกัน ไม่นับ
    scores = {k: len({w for w in hits
                      if not any(o != w and w in o for o in every)})
              for k, hits in found.items()}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, margin = ranked[0][0], ranked[0][1] - ranked[1][1]

    if margin >= CONFIDENT_MARGIN:
        return best, scores, "คำสำคัญชี้ชัด ไม่ต้องเรียกโมเดล"

    prompt = (f"เอกสารนี้เป็นชนิดใด ตอบคำเดียวจาก: invoice, receipt, contract\n\n"
              f"{text[:1500]}")
    print("  คะแนนสูสี — ถามโมเดล ด้วย prompt นี้:")
    print("  " + "─" * 70)
    for line in prompt.splitlines()[:3]:
        print("  " + line)
    print("  …")
    answer = typhoon.chat.completions.create(
        model=TYPHOON_MODEL, max_tokens=10, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content.strip().lower()
    return (answer if answer in KEYWORDS else best), scores, f"โมเดลตอบ: {answer}"

DOC_TYPE, scores, how = classify(OCR_TEXT)

print("คะแนนคำสำคัญ")
for kind, n in sorted(scores.items(), key=lambda kv: -kv[1]):
    print(f"   {kind:10} {n}" + ("   ← ชนะ" if kind == DOC_TYPE else ""))
print(f"\nประเภทเอกสาร : {DOC_TYPE}")
print(f"ตัดสินจาก     : {how}")

# ── schema ที่ผูกกับประเภทนี้ ─────────────────────────────────────────────
# critical = "ถ้าค่านี้มีอยู่แล้วอ่านผิด มีคนได้เงินผิด" ไม่ได้แปลว่า "ต้องมี"
SCHEMAS = {
    "invoice": {
        "invoice_number": {"critical": True,  "desc": "เลขที่ใบแจ้งหนี้ของผู้ขาย"},
        "invoice_date":   {"critical": False, "desc": "วันที่ออกเอกสาร รูปแบบ YYYY-MM-DD"},
        "vendor_name":    {"critical": True,  "desc": "ชื่อผู้ขาย/ผู้ออกเอกสาร ตามที่พิมพ์"},
        "vendor_tax_id":  {"critical": False, "desc": "เลขประจำตัวผู้เสียภาษีของผู้ขาย"},
        "purchase_order": {"critical": True,  "desc": "เลขที่ใบสั่งซื้อที่เอกสารอ้างถึง ถ้ามี"},
        "subtotal":       {"critical": True,  "desc": "ยอดก่อนภาษี เป็นตัวเลข"},
        "tax_rate":       {"critical": False, "desc": "อัตราภาษีเป็นตัวเลข เช่น 7"},
        "tax_amount":     {"critical": True,  "desc": "จำนวนภาษี เป็นตัวเลข"},
        "total_amount":   {"critical": True,  "desc": "ยอดรวมสุทธิ เป็นตัวเลข"},
        "bank_account":   {"critical": True,  "desc": "เลขบัญชีที่ให้โอนเงินไป"},
    },
}
SCHEMAS["receipt"] = SCHEMAS["contract"] = SCHEMAS["invoice"]   # ย่อไว้ให้เดโมสั้น
SCHEMA = SCHEMAS[DOC_TYPE]

print(f"\nschema        : {len(SCHEMA)} ช่อง · สำคัญ {sum(s['critical'] for s in SCHEMA.values())} ช่อง")
print("   " + ", ".join(f"{'★' if s['critical'] else ''}{n}" for n, s in SCHEMA.items()))
''', """
คะแนนคำสำคัญ
   invoice    3   ← ชนะ
   receipt    0
   contract   0

ประเภทเอกสาร : invoice
ตัดสินจาก     : คำสำคัญชี้ชัด ไม่ต้องเรียกโมเดล

schema        : 10 ช่อง · สำคัญ 7 ช่อง
   ★invoice_number, invoice_date, ★vendor_name, vendor_tax_id, ★purchase_order, ★subtotal, tax_rate, ★tax_amount, ★total_amount, ★bank_account
""")

# ============================================================== two reads ===
md(r"""
## ③④ Agent 2 ตัวอ่านเอกสาร — คนละโมเดล คนละ prompt

หัวใจของระบบอยู่ตรงนี้

โมเดลตัวเดียวบอกตัวเลขมา คือ**คำอ้าง** · สองตัวที่ไม่เคยเห็นคำตอบของกันแล้วตอบตรงกัน คือ**หลักฐาน**

วัดจริงบนชุดเอกสาร 24 ใบ: ช่องที่ทั้งสองตรงกัน **ถูก 97%** · ช่องที่ขัดกัน **ถูก 45%** — ช่องว่าง 52 จุดตรงนั้นคือทั้งหมดที่ได้มาจากการเรียกครั้งที่สอง ระบบเลยรู้ว่าคำตอบไหนของตัวเองไม่ควรเชื่อ

**ต้องคนละ backend จริงๆ** — เรียกโมเดลเดิมสองรอบ มันผิดซ้ำแบบเดิม ความเห็นตรงกันเลยแทบไม่มีความหมาย

| | โมเดล | มอง prompt ยังไง |
|---|---|---|
| **Agent 1** | Typhoon | **ไล่ตาม schema** — "หา 10 ช่องนี้" |
| **Agent 2** | Claude | **ไล่ตามหน้ากระดาษ** — "มีป้ายอะไรบ้าง อะไรอยู่ข้างๆ" |

เข้าหาคนละทางโดยตั้งใจ เพราะสอง prompt ที่คิดเหมือนกันจะพลาดเหมือนกัน

### `evidence` — ช่องที่ทำให้ระบบนี้ต่างจากการเรียก LLM เฉยๆ

ทั้งสอง prompt **บังคับให้แนบข้อความที่อ้างว่าอ่านมาจาก** มาด้วยทุกค่า ขั้นถัดไปจะเอาข้อความนั้นไปหาบนหน้าจริง — ค่าที่โมเดลสร้างขึ้นมาเองแต่ชี้ที่มาไม่ได้ ถือว่า**แต่งขึ้น** ไม่ว่ามันจะมั่นใจแค่ไหน

> prompt เต็มๆ พิมพ์ออกมาให้ดูก่อนเรียกทั้งสองตัว
""")

code(r'''
import json

# ═══ AGENT 1 — Typhoon · ไล่ตาม schema ════════════════════════════════════
AGENT1_SYSTEM = """คุณคือตัวสกัดข้อมูลจากเอกสารทางการเงิน
ตอบเป็น JSON เท่านั้น ไม่ต้องอธิบาย

รูปแบบ: {"ชื่อช่อง": {"value": <ค่า>, "evidence": "<ข้อความที่คัดมาจากหน้าเอกสาร>"}}

กฎ:
- evidence ต้องเป็นข้อความที่ปรากฏบนเอกสาร "คัดลอกมาตรงๆ" ห้ามเรียบเรียงใหม่
- ช่องไหนหาไม่เจอ ให้ข้ามไป อย่าเดา อย่าใส่ null
- จำนวนเงินและอัตราภาษี ตอบเป็นตัวเลข ไม่ต้องมีคอมมาหรือสัญลักษณ์เงิน
- วันที่ตอบรูปแบบ YYYY-MM-DD (ปี พ.ศ. ให้ลบ 543 ก่อน)"""

def agent1_prompt(text, schema):
    wanted = "\n".join(f"- {n}: {s['desc']}" for n, s in schema.items())
    return f"""ช่องที่ต้องหา:
{wanted}

เอกสาร:
{text}"""

# ═══ AGENT 2 — Claude · ไล่ตามหน้ากระดาษ ═════════════════════════════════
AGENT2_SYSTEM = """You are reading a financial document that has already been through OCR.

Work through the page as it is laid out, not through a checklist: find every
label the page prints, see what value sits next to it, then map what you found
onto the field names you are given. Some fields will not be on this page at all.

Reply with JSON only:
{"field_name": {"value": <value>, "evidence": "<text copied verbatim from the page>"}}

Rules:
- evidence must be copied from the page exactly. Never paraphrase it.
- Omit a field entirely if the page does not state it. Never guess, never null.
- Amounts and rates as plain numbers: no thousands separators, no currency.
- Dates as YYYY-MM-DD. A Buddhist-era year (25xx) is 543 years ahead: subtract."""

def agent2_prompt(text, schema):
    names = "\n".join(f"- {n}: {s['desc']}" for n, s in schema.items())
    return f"""Field names to map onto:
{names}

The page:
{text}"""

# ── แสดง prompt ให้เห็นก่อนเรียก ──────────────────────────────────────────
def show(title, system, user, limit=520):
    print("┌" + "─" * 76)
    print(f"│ {title}")
    print("├" + "─" * 76)
    print("│ SYSTEM:")
    for line in system.splitlines():
        print("│   " + line[:72])
    print("│ USER:")
    body = user if len(user) <= limit else user[:limit] + f"\n… (+{len(user)-limit:,} ตัวอักษร)"
    for line in body.splitlines():
        print("│   " + line[:72])
    print("└" + "─" * 76)

show("AGENT 1 · Typhoon · ไล่ตาม schema", AGENT1_SYSTEM, agent1_prompt(OCR_TEXT, SCHEMA))
print()
show("AGENT 2 · Claude · ไล่ตามหน้ากระดาษ", AGENT2_SYSTEM, agent2_prompt(OCR_TEXT, SCHEMA))
''', """
┌────────────────────────────────────────────────────────────────────────────
│ AGENT 1 · Typhoon · ไล่ตาม schema
├────────────────────────────────────────────────────────────────────────────
│ SYSTEM:
│   คุณคือตัวสกัดข้อมูลจากเอกสารทางการเงิน
│   ตอบเป็น JSON เท่านั้น ไม่ต้องอธิบาย
│
│   รูปแบบ: {"ชื่อช่อง": {"value": <ค่า>, "evidence": "<ข้อความที่คัดมาจากหน
│
│   กฎ:
│   - evidence ต้องเป็นข้อความที่ปรากฏบนเอกสาร "คัดลอกมาตรงๆ" ห้ามเรียบเรียง
│   - ช่องไหนหาไม่เจอ ให้ข้ามไป อย่าเดา อย่าใส่ null
│   - จำนวนเงินและอัตราภาษี ตอบเป็นตัวเลข ไม่ต้องมีคอมมาหรือสัญลักษณ์เงิน
│   - วันที่ตอบรูปแบบ YYYY-MM-DD (ปี พ.ศ. ให้ลบ 543 ก่อน)
│ USER:
│   ช่องที่ต้องหา:
│   - invoice_number: เลขที่ใบแจ้งหนี้ของผู้ขาย
│   - invoice_date: วันที่ออกเอกสาร รูปแบบ YYYY-MM-DD
│   - vendor_name: ชื่อผู้ขาย/ผู้ออกเอกสาร ตามที่พิมพ์
│   - vendor_tax_id: เลขประจำตัวผู้เสียภาษีของผู้ขาย
│   - purchase_order: เลขที่ใบสั่งซื้อที่เอกสารอ้างถึง ถ้ามี
│   - subtotal: ยอดก่อนภาษี เป็นตัวเลข
│   - tax_rate: อัตราภาษีเป็นตัวเลข เช่น 7
│   - tax_amount: จำนวนภาษี เป็นตัวเลข
│   - total_amount: ยอดรวมสุทธิ เป็นตัวเลข
│   - bank_account: เลขบัญชีที่ให้โอนเงินไป
│
│   เอกสาร:
│   # TAX INVOICE
│
│   Siam Tech Supply Co., Ltd.
│   199/22 Ratchada
│   … (+542 ตัวอักษร)
└────────────────────────────────────────────────────────────────────────────

┌────────────────────────────────────────────────────────────────────────────
│ AGENT 2 · Claude · ไล่ตามหน้ากระดาษ
├────────────────────────────────────────────────────────────────────────────
│ SYSTEM:
│   You are reading a financial document that has already been through OCR.
│
│   Work through the page as it is laid out, not through a checklist: find e
│   label the page prints, see what value sits next to it, then map what you
│   onto the field names you are given. Some fields will not be on this page
│
│   Reply with JSON only:
│   {"field_name": {"value": <value>, "evidence": "<text copied verbatim fro
│
│   Rules:
│   - evidence must be copied from the page exactly. Never paraphrase it.
│   - Omit a field entirely if the page does not state it. Never guess, neve
│   - Amounts and rates as plain numbers: no thousands separators, no curren
│   - Dates as YYYY-MM-DD. A Buddhist-era year (25xx) is 543 years ahead: su
│ USER:
│   Field names to map onto:
│   - invoice_number: เลขที่ใบแจ้งหนี้ของผู้ขาย
│   - invoice_date: วันที่ออกเอกสาร รูปแบบ YYYY-MM-DD
│   - vendor_name: ชื่อผู้ขาย/ผู้ออกเอกสาร ตามที่พิมพ์
│   - vendor_tax_id: เลขประจำตัวผู้เสียภาษีของผู้ขาย
│   - purchase_order: เลขที่ใบสั่งซื้อที่เอกสารอ้างถึง ถ้ามี
│   - subtotal: ยอดก่อนภาษี เป็นตัวเลข
│   - tax_rate: อัตราภาษีเป็นตัวเลข เช่น 7
│   - tax_amount: จำนวนภาษี เป็นตัวเลข
│   - total_amount: ยอดรวมสุทธิ เป็นตัวเลข
│   - bank_account: เลขบัญชีที่ให้โอนเงินไป
│
│   The page:
│   # TAX INVOICE
│
│   Siam Tech Supply Co., Ltd.
│   199
│   … (+554 ตัวอักษร)
└────────────────────────────────────────────────────────────────────────────
""")

code(r'''
import json, re, time

def extract_json(raw):
    """โมเดลชอบห่อ JSON ด้วย ```json — ดึงก้อนในสุดออกมา"""
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    start, end = raw.find("{"), raw.rfind("}")
    return json.loads(raw[start:end + 1]) if start >= 0 else {}

# ═══ เรียก AGENT 1 — Typhoon ══════════════════════════════════════════════
t0 = time.perf_counter()
raw1 = typhoon.chat.completions.create(
    model=TYPHOON_MODEL, max_tokens=2000, temperature=0,
    messages=[{"role": "system", "content": AGENT1_SYSTEM},
              {"role": "user",   "content": agent1_prompt(OCR_TEXT, SCHEMA)}],
).choices[0].message.content
AGENT1 = extract_json(raw1)
t1 = time.perf_counter() - t0
print(f"AGENT 1 · {TYPHOON_MODEL} — {t1:.1f} วินาที · ได้ {len(AGENT1)} ช่อง\n")
print(f"  {'FIELD':16} {'ค่า':24} อ้างอิงจากหน้าเอกสาร")
for name, got in AGENT1.items():
    if isinstance(got, dict):
        print(f"  {name:16} {str(got.get('value'))[:23]:24} «{str(got.get('evidence',''))[:38]}»")

# ═══ เรียก AGENT 2 — Claude ═══════════════════════════════════════════════
AGENT2 = {}
if ANTHROPIC_KEY:
    import anthropic
    claude = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
    t0 = time.perf_counter()
    reply = claude.messages.create(
        model=CLAUDE_MODEL, max_tokens=2000, temperature=0,
        system=AGENT2_SYSTEM,
        messages=[{"role": "user", "content": agent2_prompt(OCR_TEXT, SCHEMA)}],
    )
    AGENT2 = extract_json(reply.content[0].text)
    t2 = time.perf_counter() - t0
    print(f"\nAGENT 2 · {CLAUDE_MODEL} — {t2:.1f} วินาที · ได้ {len(AGENT2)} ช่อง\n")
    print(f"  {'FIELD':16} {'ค่า':24} อ้างอิงจากหน้าเอกสาร")
    for name, got in AGENT2.items():
        if isinstance(got, dict):
            print(f"  {name:16} {str(got.get('value'))[:23]:24} «{str(got.get('evidence',''))[:38]}»")
else:
    print("\nAGENT 2 · ข้าม (ไม่มี ANTHROPIC_API_KEY) — ทุกช่องจะเหลือหลักฐานชั้นเดียว")
''', """
AGENT 1 · typhoon-v2.5-30b-a3b-instruct — 3.2 วินาที · ได้ 10 ช่อง

  FIELD            ค่า                      อ้างอิงจากหน้าเอกสาร
  invoice_number   STS-2026-0318            «Invoice No .: STS-2026-0318»
  invoice_date     2026-07-24               «Invoice Date: 24 July 2026»
  vendor_name      Siam Tech Supply Co., L  «Siam Tech Supply Co., Ltd.»
  vendor_tax_id    0105543000129            «Tax ID: 0105543000129»
  purchase_order   PO-2026-0567             «PO Reference: PO-2026-0567»
  subtotal         135000                   «Subtotal 135,000.00»
  tax_rate         7                        «VAT 7%»
  tax_amount       9450                     «VAT 7% 9,450.00»
  total_amount     144450                   «Amount Due 144,450.00»
  bank_account     1234567890               «Remit to Account No: 1234567890»

AGENT 2 · claude-sonnet-4-5-20250929 — 5.3 วินาที · ได้ 10 ช่อง

  FIELD            ค่า                      อ้างอิงจากหน้าเอกสาร
  invoice_number   STS-2026-0318            «Invoice No .: STS-2026-0318»
  invoice_date     2026-07-24               «Invoice Date: 24 July 2026»
  vendor_name      Siam Tech Supply Co., L  «Siam Tech Supply Co., Ltd.»
  vendor_tax_id    0105543000129            «Tax ID: 0105543000129»
  purchase_order   PO-2026-0567             «PO Reference: PO-2026-0567»
  subtotal         135000.0                 «Subtotal 135,000.00»
  tax_rate         7                        «VAT 7%»
  tax_amount       9450.0                   «VAT 7% 9,450.00»
  total_amount     144450.0                 «Amount Due 144,450.00»
  bank_account     1234567890               «Remit to Account No: 1234567890»
""")

# ================================================================= merge ====
md(r"""
## ⑤ รวม · ตรวจว่าอยู่บนหน้าจริง · ให้คะแนน

สามเรื่อง และต้องแยกกันให้ชัดเพราะพังคนละแบบ

**รวม** — สองตัวตรงกันไหม เทียบ**หลังทำเป็นรูปแบบมาตรฐานแล้ว** `135000` กับ `135000.0` คือเลขเดียวกัน ถ้าเทียบสตริงดิบจะรายงานว่าขัดกันทั้งที่ไม่ได้ขัด

**ตรวจว่าอยู่บนหน้า (grounding)** — เอา `evidence` ที่ agent แนบมา ไปหาบนข้อความ OCR จริง เช็ค ไม่ใช่เชื่อ

**ให้คะแนน** — ความมั่นใจตอบ**คำถามเดียว**: *เราอ่านค่านี้ถูกไหม* ไม่ใช่ "ใบนี้จ่ายได้ไหม" ซึ่งเป็นงานของขั้น ⑧

> ตอนที่มันเป็นตัวเลขเดียวกัน คำเตือนสามข้อลากเอกสารที่อ่านถูกทุกช่องจาก 95% ลงมา 72% ใต้เกณฑ์ 84% แล้วส่งไปให้คนดูทั้งที่ไม่มีอะไรให้แก้ — **คำเตือนที่หยุดเอกสารได้ คือกฎ blocking ที่ยังไม่ยอมรับตัวเอง**

### สิ่งที่ grounding ทำไม่ได้

พิสูจน์ได้แค่ว่าค่ามาจากข้อความบนหน้า — **ถ้า OCR อ่านผิด ค่าที่ผิดก็อยู่บนหน้าจริงในสายตา grounding** และ evidence ก็ผิดตามกัน agent สองตัวที่อ่านข้อความ OCR ชุดเดียวกันก็จับไม่ได้เหมือนกัน

สิ่งที่กันการอ่านผิดคือ**กฎเลขคณิตในขั้น ⑥** ไม่ใช่ grounding
""")

code(r'''
AGREE, DISAGREE, ONE_AGENT = 0.95, 0.55, 0.84

def canonical(v):
    """รูปแบบมาตรฐานสำหรับ *เปรียบเทียบ* เท่านั้น — 135000 กับ 135000.0 ต้องเท่ากัน"""
    s = str(v).strip().translate(str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789"))
    try:
        return f"{float(s.replace(',', '')):.2f}"
    except ValueError:
        return " ".join(s.lower().replace(",", " ").split())

def grounded(evidence, page):
    """evidence ที่ agent อ้าง อยู่บนหน้าจริงไหม — บีบช่องว่างทิ้งก่อนหา"""
    squash = lambda s: "".join(str(s).lower().split())
    return bool(evidence) and squash(evidence) in squash(page)

FIELDS = {}
for name, spec in SCHEMA.items():
    a, b = AGENT1.get(name), AGENT2.get(name)
    a = a if isinstance(a, dict) else None
    b = b if isinstance(b, dict) else None
    if not a and not b:
        continue
    src = a or b
    value, evidence = src.get("value"), src.get("evidence", "")

    if a and b:
        same = canonical(a.get("value")) == canonical(b.get("value"))
        conf, how = (AGREE, "ทั้งสองตรงกัน") if same else (DISAGREE, "ขัดกัน")
    else:
        conf, how = ONE_AGENT, "agent เดียว"

    on_page = grounded(evidence, OCR_TEXT)
    if spec["critical"] and not on_page:
        conf *= 0.4                     # ค่าสำคัญที่ชี้ที่มาไม่ได้ ไว้ใจไม่ได้

    FIELDS[name] = {"value": value, "evidence": evidence, "confidence": conf,
                    "grounded": on_page, "how": how, "critical": spec["critical"],
                    "a": a.get("value") if a else None,
                    "b": b.get("value") if b else None}

def document_confidence(fields):
    """ช่องสำคัญถ่วงน้ำหนัก 3 เท่า — ตอบแค่ว่าอ่านถูกไหม"""
    if not fields:
        return 0.0
    w = lambda f: 3.0 if f["critical"] else 1.0
    return sum(f["confidence"] * w(f) for f in fields.values()) / sum(w(f) for f in fields.values())

CONFIDENCE = document_confidence(FIELDS)

print(f"{'':2}{'FIELD':16} {'ค่า':26} {'มั่นใจ':>6} {'บนหน้า':>7}  ที่มา")
print("─" * 84)
for name, f in FIELDS.items():
    print(f"{'★' if f['critical'] else ' '} {name:16} {str(f['value'])[:25]:26} "
          f"{f['confidence']:>5.0%} {('ใช่' if f['grounded'] else 'ไม่'):>7}  {f['how']}")

print("\n★ = ช่องสำคัญ (ถ้าค่านี้ผิด มีคนได้เงินผิด)")
print(f"ความมั่นใจทั้งใบ: {CONFIDENCE:.0%}")

# ── ตรงกันทั้งที่เขียนคนละแบบ ─────────────────────────────────────────────
differ = [(n, f) for n, f in FIELDS.items()
          if f["a"] is not None and f["b"] is not None and str(f["a"]) != str(f["b"])]
if differ:
    print("\n" + "─" * 84)
    print("สอง agent พิมพ์คนละแบบ แต่นับว่าตรงกัน เพราะเทียบหลังทำเป็นมาตรฐาน:")
    print(f"\n  {'FIELD':16} {'AGENT 1':18} {'AGENT 2':18} เทียบแล้ว")
    for name, f in differ:
        print(f"  {name:16} {str(f['a'])[:17]:18} {str(f['b'])[:17]:18} "
              f"{canonical(f['a'])} == {canonical(f['b'])}")

ungrounded = [n for n, f in FIELDS.items() if not f["grounded"]]
print(f"\nชี้ที่มาบนหน้าไม่ได้: {', '.join(ungrounded) if ungrounded else 'ไม่มี — ทุกค่าตรวจสอบได้'}")
''', """
  FIELD            ค่า                        มั่นใจ  บนหน้า  ที่มา
────────────────────────────────────────────────────────────────────────────────────
★ invoice_number   STS-2026-0318                95%     ใช่  ทั้งสองตรงกัน
  invoice_date     2026-07-24                   95%     ใช่  ทั้งสองตรงกัน
★ vendor_name      Siam Tech Supply Co., Ltd    95%     ใช่  ทั้งสองตรงกัน
  vendor_tax_id    0105543000129                95%     ใช่  ทั้งสองตรงกัน
★ purchase_order   PO-2026-0567                 95%     ใช่  ทั้งสองตรงกัน
★ subtotal         135000                       95%     ใช่  ทั้งสองตรงกัน
  tax_rate         7                            95%     ใช่  ทั้งสองตรงกัน
★ tax_amount       9450                         95%     ใช่  ทั้งสองตรงกัน
★ total_amount     144450                       95%     ใช่  ทั้งสองตรงกัน
★ bank_account     1234567890                   95%     ใช่  ทั้งสองตรงกัน

★ = ช่องสำคัญ (ถ้าค่านี้ผิด มีคนได้เงินผิด)
ความมั่นใจทั้งใบ: 95%

────────────────────────────────────────────────────────────────────────────────────
สอง agent พิมพ์คนละแบบ แต่นับว่าตรงกัน เพราะเทียบหลังทำเป็นมาตรฐาน:

  FIELD            AGENT 1            AGENT 2            เทียบแล้ว
  subtotal         135000             135000.0           135000.00 == 135000.00
  tax_amount       9450               9450.0             9450.00 == 9450.00
  total_amount     144450             144450.0           144450.00 == 144450.00

ชี้ที่มาบนหน้าไม่ได้: ไม่มี — ทุกค่าตรวจสอบได้
""")

# ================================================================= rules ====
md(r"""
## ⑥ กฎ — เอาแค่ 3 ข้อ (เดโม)

ระบบจริงมี 26 กฎ ที่นี่เอาแค่ 3 ข้อให้เห็นรูปแบบ — **หนึ่งข้อต่อหนึ่งประเภทของคำถาม**

| กฎ | ถามอะไร | เห็นอะไรได้ |
|---|---|---|
| `totals_consistent` | ยอดก่อนภาษี + ภาษี = ยอดรวมไหม | เอกสาร**ขัดกับตัวเอง** |
| `tax_rate_consistent` | ยอดก่อนภาษี × อัตรา = ภาษีไหม | เอกสาร**ขัดกับตัวเอง** |
| `bank_account_matches` | เลขบัญชีตรงทะเบียนไหม | **เรื่องนอกหน้ากระดาษ** |

สองข้อแรก**คำนวณใหม่** ไม่เชื่อที่พิมพ์มา · ข้อที่สามคือกลโกงที่พบบ่อยที่สุดในโลกจริง และอ่านเอกสารให้ดีแค่ไหนก็จับไม่ได้

### สองค่าความคลาดเคลื่อน ห้ามรวมกัน

| | คือ | ค่า |
|---|---|---|
| `AMOUNT_TOLERANCE` | ต่างจาก**ใบสั่งซื้อ**ได้แค่ไหน — ผ่อนผันทางการค้า | 2% |
| `ARITHMETIC_TOLERANCE` | ต่างจาก**ตัวเอง**ได้แค่ไหน — แค่ปัดเศษ | 0.01% |

ตอนใช้ตัวเดียวกัน การอ่านผิด 810 บาทผ่านไปได้ในฐานะ "ยอดรวมถูกต้อง" บนใบ 200,090 บาท

### สามระดับ และต้องทำงานต่างกันจริง

`blocking` **หยุดเอกสาร** · `warning` บันทึกไว้ให้เห็นแต่**ไม่หยุดอะไร** · `info` แค่หมายเหตุ
""")

code(r'''
# ── ทะเบียน (ระบบจริงมาจาก CSV ที่ ERP ส่งออก) ─────────────────────────────
VENDORS = {
    "V-1001": {"name": "Siam Tech Supply Co., Ltd.",
               "aliases": ["Siam Tech", "บริษัท สยามเทค ซัพพลาย จำกัด"],
               "tax_id": "0105543000129", "bank_account": "1234567890", "status": "active"},
    "V-1006": {"name": "Rungruang Steel Works Co., Ltd.",
               "aliases": ["บริษัท รุ่งเรือง สตีล เวิร์ค จำกัด"],
               "tax_id": "0105552000110", "bank_account": "3120045678", "status": "active"},
}
PURCHASE_ORDERS = {   # ยอดไม่รวม VAT — ใบสั่งซื้อบันทึกสิ่งที่ตกลง ภาษีบวกตอนวางบิล
    "PO-2026-0567": {"vendor": "V-1001", "amount": 135_000.00, "status": "open",
                     "billed_so_far": 0.00},
    "PO-2026-0578": {"vendor": "V-1006", "amount":  88_000.00, "status": "open",
                     "billed_so_far": 40_000.00},
}

AMOUNT_TOLERANCE, ARITHMETIC_TOLERANCE = 0.02, 0.0001

def val(name):
    f = FIELDS.get(name)
    return f["value"] if f else None

def num(name):
    try:
        return float(str(val(name)).replace(",", ""))
    except (TypeError, ValueError):
        return None

def find_vendor():
    tax, name = val("vendor_tax_id"), val("vendor_name")
    for vid, v in VENDORS.items():
        if tax and v["tax_id"] == str(tax).strip():
            return vid, v, "เลขผู้เสียภาษี"
    if name:
        for vid, v in VENDORS.items():
            if any(canonical(c) == canonical(name) for c in [v["name"], *v["aliases"]]):
                return vid, v, "ชื่อ"
    return None, None, "ไม่พบ"

# ── ที่เก็บผลตรวจ ─────────────────────────────────────────────────────────
FINDINGS = []

def check(rule_id, ok, severity, message):
    """บันทึกผลตรวจหนึ่งข้อ — เขียนทับของเดิมถ้ามี id ซ้ำ

    ทับ ไม่ใช่ต่อท้าย เพราะ notebook เป็นสถานะที่ค้างอยู่: รัน cell agent ซ้ำ
    รอบสองจะได้กฎเดิมโผล่สองครั้ง แล้วรายงานสุดท้ายบอกว่าตรวจ 5 ข้อทั้งที่มี 4
    ซึ่งเจอแน่นอน เพราะท้าย notebook เขียนไว้ให้แก้ค่าแล้วรันซ้ำ
    """
    global FINDINGS
    FINDINGS = [f for f in FINDINGS if f["rule"] != rule_id]
    FINDINGS.append({"rule": rule_id, "ok": ok, "severity": severity, "message": message})

sub, tax, total, rate = num("subtotal"), num("tax_amount"), num("total_amount"), num("tax_rate")

# ── กฎที่ ① เอกสารขัดกับตัวเองไหม — คำนวณใหม่ ไม่เชื่อที่พิมพ์ ─────────────
if None not in (sub, tax, total):
    slack = max(0.01, abs(total) * ARITHMETIC_TOLERANCE)
    ok = abs(sub + tax - total) <= slack
    check("totals_consistent", ok, "blocking",
          f"{sub:,.2f} + {tax:,.2f} = {sub + tax:,.2f} แต่เอกสารเขียน {total:,.2f}"
          if not ok else f"{sub:,.2f} + {tax:,.2f} = {total:,.2f} ถูกต้อง")

if None not in (sub, tax, rate):
    expect = round(sub * rate / 100, 2)
    ok = abs(expect - tax) <= max(0.01, abs(expect) * ARITHMETIC_TOLERANCE)
    check("tax_rate_consistent", ok, "blocking",
          f"อัตรา {rate:.0f}% ของ {sub:,.2f} คือ {expect:,.2f} แต่คิดมา {tax:,.2f}"
          if not ok else f"ภาษี {tax:,.2f} ตรงกับอัตรา {rate:.0f}%")

# ── กฎที่ ② เรื่องที่ไม่ได้อยู่บนหน้ากระดาษ ────────────────────────────────
VENDOR_ID, VENDOR, matched_by = find_vendor()
printed = val("bank_account")
if VENDOR and printed:
    ok = str(printed).strip() == VENDOR["bank_account"]
    check("bank_account_matches", ok, "blocking",
          f"เอกสารเขียน {printed} แต่ทะเบียนจดไว้ {VENDOR['bank_account']}"
          if not ok else f"เลขบัญชี {printed} ตรงกับทะเบียน")
elif not VENDOR:
    check("bank_account_matches", False, "blocking",
          f"ไม่รู้จักผู้ขาย '{val('vendor_name')}' — เทียบเลขบัญชีกับอะไรไม่ได้")

print(f"ผู้ขาย : {VENDOR_ID or '— ไม่พบ'}"
      + (f"  {VENDOR['name']}  (จับคู่จาก{matched_by})" if VENDOR else ""))
print(f"ตรวจ   : {len(FINDINGS)} ข้อ · ผ่าน {sum(f['ok'] for f in FINDINGS)} · "
      f"ไม่ผ่าน {sum(not f['ok'] for f in FINDINGS)}\n")
for f in FINDINGS:
    mark = "ok  " if f["ok"] else f["severity"].upper()
    print(f"  [{mark:8}] {f['rule']:22} {f['message']}")
''', """
ผู้ขาย : V-1001  Siam Tech Supply Co., Ltd.  (จับคู่จากเลขผู้เสียภาษี)
ตรวจ   : 3 ข้อ · ผ่าน 3 · ไม่ผ่าน 0

  [ok      ] totals_consistent      135,000.00 + 9,450.00 = 144,450.00 ถูกต้อง
  [ok      ] tax_rate_consistent    ภาษี 9,450.00 ตรงกับอัตรา 7%
  [ok      ] bank_account_matches   เลขบัญชี 1234567890 ตรงกับทะเบียน
""")

# =========================================================== agent + tools ==
md(r"""
## ⑦ Agent ตรวจกฎ — เรียก tool จริง

3 กฎข้างบนเขียนเป็นเงื่อนไขได้เพราะ**ทุกอย่างที่ต้องรู้อยู่บนหน้ากระดาษหรืออยู่ในทะเบียนแถวเดียว**

แต่กฎแบบนี้เขียนเป็นเงื่อนไขไม่ได้:

> *"ก่อนจ่ายตามใบสั่งซื้อ ให้เช็คว่าใบสั่งซื้อยังเปิดอยู่ และยอดที่วางบิลไปแล้วรวมกับใบนี้ ยังไม่เกินวงเงิน"*

ไม่มีช่องไหนให้เทียบ — ต้อง **ไปค้น → คำนวณจากสิ่งที่ค้นมา → ตัดสิน** ยัดลงเงื่อนไขได้แค่อะไรที่*ใกล้เคียง*และผิด ซึ่งแย่กว่ายอมรับว่ารูปแบบมันไม่เข้ากัน

**ตรงนี้แหละที่ tool calling คุ้มค่า** และเป็นที่เดียวที่ระบบนี้ใช้

### กรงที่ขังมันไว้

agent ตัวนี้ไม่ได้ปล่อยอิสระ:

| | |
|---|---|
| **แผนตายตัว** | เขียนไว้ตอนสร้างกฎ ต่อหน้าคนอนุมัติ ไม่ได้คิดเองตอนรัน |
| **กล่องเครื่องมือปิด** | เรียกได้แค่ 2 ตัวนี้ ไม่มีทางไปแตะอย่างอื่น |
| **คำตอบมีแค่ 3 แบบ** | `pass` / `fail` / `unsure` — **ยกระดับความรุนแรงเองไม่ได้** |
| **ตอบ `unsure` = หยุดเอกสาร** | ไม่ใช่ผ่าน — ไม่รู้ ไม่เท่ากับ ไม่มีปัญหา |

> เห็นทุก tool call ที่มันเรียก พร้อมสิ่งที่ได้กลับไป
""")

code(r'''
import json

# ═══ กล่องเครื่องมือ — agent เรียกได้แค่สองตัวนี้ ════════════════════════
TOOLS = [
    {"name": "lookup_purchase_order",
     "description": "ค้นใบสั่งซื้อจากทะเบียน คืนวงเงิน สถานะ และยอดที่วางบิลไปแล้ว",
     "input_schema": {"type": "object",
                      "properties": {"po_number": {"type": "string"}},
                      "required": ["po_number"]}},
    {"name": "check_arithmetic",
     "description": "บวกเลขให้ถูกต้อง อย่าคำนวณในหัวเอง",
     "input_schema": {"type": "object",
                      "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                      "required": ["a", "b"]}},
]

TOOL_LOG = []
def run_tool(name, args):
    if name == "lookup_purchase_order":
        po = PURCHASE_ORDERS.get(args["po_number"])
        result = po or {"error": "ไม่พบใบสั่งซื้อนี้ในทะเบียน"}
    elif name == "check_arithmetic":
        result = {"sum": args["a"] + args["b"]}
    else:
        result = {"error": "ไม่มีเครื่องมือชื่อนี้"}
    TOOL_LOG.append((name, args, result))
    return result

# ═══ กฎ (ภาษาคน) + แผน — เขียนครั้งเดียวตอนสร้างกฎ ═══════════════════════
AGENT_RULE = {
    "id": "purchase_order_still_has_room",
    "prompt": "ก่อนจ่ายตามใบสั่งซื้อ ให้เช็คว่ายังเปิดอยู่ และยอดที่วางบิลไปแล้ว"
              "รวมกับใบนี้ ยังไม่เกินวงเงิน",
    "severity": "blocking",
    "plan": [
        "ค้นใบสั่งซื้อที่เอกสารอ้างถึง ดูวงเงิน สถานะ และยอดที่วางบิลไปแล้ว",
        "ถ้าสถานะไม่ใช่ open ให้ตอบ fail ทันที และบอกว่าสถานะอะไร",
        "บวกยอดที่วางบิลไปแล้ว กับยอดก่อนภาษีของใบนี้ ด้วย check_arithmetic",
        "เทียบผลบวกกับวงเงิน — เกิน = fail, ไม่เกิน = pass, ค้นไม่เจอ = unsure",
    ],
}

AGENT_RULE_SYSTEM = """คุณตรวจสอบเอกสารตามแผนที่ได้รับ ทีละขั้น ห้ามข้าม

ใช้เครื่องมือที่ให้มาเท่านั้น อย่าคำนวณเลขในหัว ให้เรียก check_arithmetic

เมื่อเดินครบแผนแล้ว ตอบเป็น JSON เท่านั้น:
{"verdict": "pass" | "fail" | "unsure", "reason": "<เหตุผลสั้นๆ ภาษาไทย>",
 "evidence": ["<สิ่งที่ค้นเจอ>", ...]}

- pass  = ตรวจครบแล้วไม่มีปัญหา
- fail  = พบปัญหาชัดเจน
- unsure = ค้นไม่เจอหรือข้อมูลไม่พอตัดสิน (ไม่ใช่ pass)
ห้ามเปลี่ยนระดับความรุนแรง ห้ามเสนอให้ยกเว้น"""

po_ref, subtotal_value = val("purchase_order"), num("subtotal")
AGENT_VERDICT = None

if not ANTHROPIC_KEY:
    print("ข้าม — agent ตรวจกฎต้องใช้ ANTHROPIC_API_KEY")
elif not po_ref:
    print("ข้าม — เอกสารไม่ได้อ้างใบสั่งซื้อ กฎนี้จึงไม่เกี่ยว (ไม่ใช่ผ่าน ไม่ใช่ตก)")
else:
    task = (f"กฎ: {AGENT_RULE['prompt']}\n\nแผน:\n"
            + "\n".join(f"{i}. {s}" for i, s in enumerate(AGENT_RULE["plan"], 1))
            + f"\n\nเอกสารใบนี้:\n  ใบสั่งซื้อที่อ้าง: {po_ref}\n"
              f"  ยอดก่อนภาษี: {subtotal_value:,.2f}")
    print("─" * 78); print(task); print("─" * 78 + "\n")

    messages = [{"role": "user", "content": task}]
    for turn in range(1, 8):
        reply = claude.messages.create(
            model=CLAUDE_MODEL, max_tokens=1200, temperature=0,
            system=AGENT_RULE_SYSTEM, tools=TOOLS, messages=messages,
        )
        messages.append({"role": "assistant", "content": reply.content})
        calls = [b for b in reply.content if b.type == "tool_use"]
        if not calls:
            text = "".join(b.text for b in reply.content if b.type == "text")
            AGENT_VERDICT = extract_json(text)
            break
        results = []
        for call in calls:
            out = run_tool(call.name, call.input)
            print(f"  รอบ {turn} · เรียก {call.name}({json.dumps(call.input, ensure_ascii=False)})")
            print(f"           ได้ → {json.dumps(out, ensure_ascii=False)}")
            results.append({"type": "tool_result", "tool_use_id": call.id,
                            "content": json.dumps(out, ensure_ascii=False)})
        messages.append({"role": "user", "content": results})

    print(f"\n  เดินไป {turn} รอบ · เรียก tool {len(TOOL_LOG)} ครั้ง\n")
    verdict = (AGENT_VERDICT or {}).get("verdict", "unsure")
    print(f"  คำตอบ  : {verdict.upper()}")
    print(f"  เพราะ  : {(AGENT_VERDICT or {}).get('reason', '(ตอบไม่ได้)')}")
    for e in (AGENT_VERDICT or {}).get("evidence", []):
        print(f"  หลักฐาน: {e}")

    # `unsure` ลดจาก blocking เป็น warning — "ตัดสินไม่ได้" ไม่เท่ากับ "พบว่าผิด"
    # แต่ยังไม่ผ่าน จึงยังหยุดเอกสารอยู่ดี
    #
    # กฎที่ผ่าน คงระดับที่ประกาศไว้ ไม่ใช่ลดเป็น warning — ระดับความรุนแรง
    # บอกว่า "ถ้าข้อนี้ตก จะเกิดอะไร" ไม่ได้บอกว่าผลรอบนี้เป็นยังไง
    severity = "warning" if verdict == "unsure" else AGENT_RULE["severity"]
    check(AGENT_RULE["id"], verdict == "pass", severity,
          (AGENT_VERDICT or {}).get("reason", "agent ตัดสินไม่ได้"))
    print(f"\n  บันทึกเป็น: {AGENT_RULE['id']} · "
          f"{'ผ่าน' if verdict == 'pass' else 'ไม่ผ่าน'} · ระดับ {severity}")
''', """
──────────────────────────────────────────────────────────────────────────────
กฎ: ก่อนจ่ายตามใบสั่งซื้อ ให้เช็คว่ายังเปิดอยู่ และยอดที่วางบิลไปแล้วรวมกับใบนี้ ยังไม่เกินวงเงิน

แผน:
1. ค้นใบสั่งซื้อที่เอกสารอ้างถึง ดูวงเงิน สถานะ และยอดที่วางบิลไปแล้ว
2. ถ้าสถานะไม่ใช่ open ให้ตอบ fail ทันที และบอกว่าสถานะอะไร
3. บวกยอดที่วางบิลไปแล้ว กับยอดก่อนภาษีของใบนี้ ด้วย check_arithmetic
4. เทียบผลบวกกับวงเงิน — เกิน = fail, ไม่เกิน = pass, ค้นไม่เจอ = unsure

เอกสารใบนี้:
  ใบสั่งซื้อที่อ้าง: PO-2026-0567
  ยอดก่อนภาษี: 135,000.00
──────────────────────────────────────────────────────────────────────────────

  รอบ 1 · เรียก lookup_purchase_order({"po_number": "PO-2026-0567"})
           ได้ → {"vendor": "V-1001", "amount": 135000.0, "status": "open", "billed_so_far": 0.0}
  รอบ 2 · เรียก check_arithmetic({"a": 0.0, "b": 135000.0})
           ได้ → {"sum": 135000.0}

  เดินไป 3 รอบ · เรียก tool 2 ครั้ง

  คำตอบ  : PASS
  เพราะ  : ใบสั่งซื้อสถานะ open, ยอดวางบิลสะสม 135,000 ไม่เกินวงเงิน 135,000
  หลักฐาน: PO-2026-0567 สถานะ: open
  หลักฐาน: วงเงิน: 135,000.00
  หลักฐาน: วางบิลไปแล้ว: 0.00
  หลักฐาน: ยอดใบนี้: 135,000.00
  หลักฐาน: รวมสะสม: 135,000.00 (ไม่เกินวงเงิน)

  บันทึกเป็น: purchase_order_still_has_room · ผ่าน · ระดับ blocking
""")

# ================================================================ decide ====
md(r"""
## ⑧ ตัดสิน — POST หรือไม่ POST

สองตัวเลขมาเจอกันตรงนี้ และมัน**คนละเรื่อง**

**Confidence** เรื่อง*การอ่าน* — มั่นใจแค่ไหนว่าค่าเหล่านี้คือสิ่งที่พิมพ์อยู่บนหน้า

**การตัดสิน** เรื่อง*นโยบาย* — ใบนี้โพสต์ได้ไหม

**ตรงนี้ไม่มีโมเดลอยู่เลย** โมเดลอ่าน 3 ครั้ง (OCR + agent 2 ตัว) และ agent ตรวจกฎเรียก tool อีก 2 ครั้ง แต่**คำตัดสินสุดท้ายเป็นโค้ดล้วน** — เพราะสิ่งเดียวที่โน้มน้าวโมเดลได้ คือเอกสารที่กำลังถูกตัดสินนั่นเอง

จะ POST ได้ต้องครบทุกข้อ:

1. ไม่มีกฎ `blocking` ข้อไหนไม่ผ่าน
2. ค่าสำคัญที่**มีอยู่** ต้องชี้ที่มาบนหน้าได้ทุกค่า
3. Confidence ≥ เกณฑ์

> ข้อ 2 เขียนว่า "ที่**มีอยู่**" — `critical` คือเรื่อง*ผลที่ตามมา* ไม่ใช่เรื่อง*ต้องมี* ใบที่ไม่อ้างใบสั่งซื้อเลยก็ POST ได้ ส่วนคำถามว่า*ต้องมีไหม* เป็นนโยบายธุรกิจ ตอบด้วยกฎแยกที่ปรับได้
""")

code(r'''
AUTO_POST_MIN_CONFIDENCE = 0.84

reasons = [f["message"] for f in FINDINGS if not f["ok"] and f["severity"] == "blocking"]

ungrounded = [n for n, f in FIELDS.items()
              if f["critical"] and f["value"] not in (None, "") and not f["grounded"]]
if ungrounded:
    reasons.append(f"ค่าสำคัญที่ยืนยันกับหน้าเอกสารไม่ได้: {', '.join(ungrounded)}")

if CONFIDENCE < AUTO_POST_MIN_CONFIDENCE:
    reasons.append(f"ความมั่นใจ {CONFIDENCE:.0%} ต่ำกว่าเกณฑ์ {AUTO_POST_MIN_CONFIDENCE:.0%}")

POST = not reasons

# ══════════════════════════ ผลลัพธ์สุดท้าย ═══════════════════════════════
print("╔" + "═" * 76 + "╗")
print("║" + " ผลลัพธ์ ".center(74) + "  ║")
print("╚" + "═" * 76 + "╝\n")

print(f"ไฟล์            : {DOC_PATH.name}")
print(f"ประเภทเอกสาร     : {DOC_TYPE}")
print(f"ผู้ขาย           : {VENDOR_ID or '— ไม่พบในทะเบียน'}"
      + (f"  {VENDOR['name']}" if VENDOR else ""))

print(f"\n── ทุก field ที่อ่านได้ ({len(FIELDS)} ช่อง) " + "─" * 40)
print(f"{'':2}{'FIELD':17}{'ค่า':28}{'มั่นใจ':>7} {'บนหน้า':>8}")
for name, f in FIELDS.items():
    print(f"{'★' if f['critical'] else ' '} {name:17}{str(f['value'])[:27]:28}"
          f"{f['confidence']:>6.0%} {('ใช่' if f['grounded'] else 'ไม่'):>8}")
missing = [n for n in SCHEMA if n not in FIELDS]
if missing:
    print(f"\n  ไม่ได้พิมพ์บนเอกสารนี้: {', '.join(missing)}")

print(f"\n── ผ่านกฎไหม ({len(FINDINGS)} ข้อ) " + "─" * 48)
for f in FINDINGS:
    status = "ผ่าน  " if f["ok"] else f"ไม่ผ่าน [{f['severity']}]"
    print(f"  {status:22} {f['rule']:30} {f['message'][:60]}")

print(f"\n── สรุป " + "─" * 66)
print(f"  Confidence      : {CONFIDENCE:.0%}   (เกณฑ์ {AUTO_POST_MIN_CONFIDENCE:.0%})")
print(f"  กฎที่ไม่ผ่าน      : {sum(not f['ok'] for f in FINDINGS)} จาก {len(FINDINGS)}")
print(f"  เรียกโมเดล      : OCR 1 + agent อ่าน {1 + bool(ANTHROPIC_KEY)} + agent กฎ (tool {len(TOOL_LOG)} ครั้ง)")
print(f"  โมเดลในการตัดสิน : 0")

print()
print("┏" + "━" * 76 + "┓")
print("┃" + f"  POST = {POST}".ljust(76) + "┃")
print("┗" + "━" * 76 + "┛")
if reasons:
    print("\nไม่ POST เพราะ:")
    for r in reasons:
        print("  •", r)
    print("\n→ เข้าคิวให้คนตรวจ")
else:
    print("\n→ เขียนลงบัญชีอัตโนมัติ ไม่ต้องมีคนดู")

RESULT = {"file": DOC_PATH.name, "doc_type": DOC_TYPE, "vendor": VENDOR_ID,
          "fields": {n: f["value"] for n, f in FIELDS.items()},
          "rules": [{k: f[k] for k in ("rule", "ok", "severity")} for f in FINDINGS],
          "confidence": round(CONFIDENCE, 3), "post": POST, "reasons": reasons}
print("\n── ผลลัพธ์เป็น JSON (พร้อมส่งต่อ) " + "─" * 42)
print(json.dumps(RESULT, ensure_ascii=False, indent=2)[:1200])
''', """
╔════════════════════════════════════════════════════════════════════════════╗
║                                 ผลลัพธ์                                    ║
╚════════════════════════════════════════════════════════════════════════════╝

ไฟล์            : invoice_test.png
ประเภทเอกสาร     : invoice
ผู้ขาย           : V-1001  Siam Tech Supply Co., Ltd.

── ทุก field ที่อ่านได้ (10 ช่อง) ────────────────────────────────────────
  FIELD            ค่า                          มั่นใจ   บนหน้า
★ invoice_number   STS-2026-0318                  95%      ใช่
  invoice_date     2026-07-24                     95%      ใช่
★ vendor_name      Siam Tech Supply Co., Ltd.     95%      ใช่
  vendor_tax_id    0105543000129                  95%      ใช่
★ purchase_order   PO-2026-0567                   95%      ใช่
★ subtotal         135000                         95%      ใช่
  tax_rate         7                              95%      ใช่
★ tax_amount       9450                           95%      ใช่
★ total_amount     144450                         95%      ใช่
★ bank_account     1234567890                     95%      ใช่

── ผ่านกฎไหม (4 ข้อ) ────────────────────────────────────────────────
  ผ่าน                   totals_consistent              135,000.00 + 9,450.00 = 144,450.00 ถูกต้อง
  ผ่าน                   tax_rate_consistent            ภาษี 9,450.00 ตรงกับอัตรา 7%
  ผ่าน                   bank_account_matches           เลขบัญชี 1234567890 ตรงกับทะเบียน
  ผ่าน                   purchase_order_still_has_room  ใบสั่งซื้อสถานะ open, ยอดวางบิลสะสม 135,000 ไม่เกินวงเงิน 13

── สรุป ──────────────────────────────────────────────────────────────────
  Confidence      : 95%   (เกณฑ์ 84%)
  กฎที่ไม่ผ่าน      : 0 จาก 4
  เรียกโมเดล      : OCR 1 + agent อ่าน 2 + agent กฎ (tool 2 ครั้ง)
  โมเดลในการตัดสิน : 0

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  POST = True                                                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

→ เขียนลงบัญชีอัตโนมัติ ไม่ต้องมีคนดู

── ผลลัพธ์เป็น JSON (พร้อมส่งต่อ) ──────────────────────────────────────────
{
  "file": "invoice_test.png",
  "doc_type": "invoice",
  "vendor": "V-1001",
  "fields": {
    "invoice_number": "STS-2026-0318",
    "invoice_date": "2026-07-24",
    "vendor_name": "Siam Tech Supply Co., Ltd.",
    "vendor_tax_id": "0105543000129",
    "purchase_order": "PO-2026-0567",
    "subtotal": 135000,
    "tax_rate": 7,
    "tax_amount": 9450,
    "total_amount": 144450,
    "bank_account": "1234567890"
  },
  "rules": [
    {
      "rule": "totals_consistent",
      "ok": true,
      "severity": "blocking"
    },
    {
      "rule": "tax_rate_consistent",
      "ok": true,
      "severity": "blocking"
    },
    {
      "rule": "bank_account_matches",
      "ok": true,
      "severity": "blocking"
    },
    {
      "rule": "purchase_order_still_has_room",
      "ok": true,
      "severity": "blocking"
    }
  ],
  "confidence": 0.95,
  "post": true,
  "reasons": []
}
""")

# =============================================================== closing ====
md(r"""
## สิ่งที่เพิ่งเกิดขึ้น

| | |
|---|---|
| **OCR 1 ครั้ง** | Typhoon OCR 1.5 — โมเดลที่ train มาอ่านเอกสาร ไม่ใช่โมเดลมองภาพทั่วไป |
| **อ่าน 2 ครั้ง** | Typhoon กับ Claude · คนละ prompt คนละ backend · ไม่เห็นคำตอบของกัน |
| **agent เรียก tool** | เฉพาะกฎที่เขียนเป็นเงื่อนไขไม่ได้ ที่เหลือเป็นโค้ดล้วน |
| **0 โมเดลในการตัดสิน** | grounding · กฎ · confidence · POST เป็นโค้ดทั้งหมด |
| **ทุกค่าชี้ที่มาได้** | และ evidence ถูกเอาไปหาบนหน้าจริง ไม่ใช่เชื่อ |

### ทำไมถึงไม่ให้โมเดลตัดสินตอนจบ

เพราะสิ่งเดียวที่โน้มน้าวโมเดลได้ คือ**เอกสารที่กำลังถูกตัดสินนั่นเอง**

ถามโมเดลทุกใบว่า *"ใบนี้จ่ายได้ไหม"* → คำตอบขึ้นกับถ้อยคำบนเอกสาร ซึ่งผู้ขายมักง่ายหรือคนที่ตั้งใจโกงเป็นคนคุม · เขียนกฎเป็นโค้ดครั้งเดียว ต่อหน้าคนที่อนุมัติ → ไม่เป็นแบบนั้น

---

### ลองต่อ

**1. เปลี่ยนเอกสาร** ลบไฟล์ใน `/content/docs` แล้วรัน cell อัปโหลดใหม่ ลองใบภาษาไทย ใบที่ถ่ายเอียงๆ ใบเสร็จ

**2. ทำให้มันไม่ผ่าน** แก้ `VENDORS["V-1001"]["bank_account"]` เป็นเลขอื่น แล้วรันตั้งแต่ cell กฎ → `bank_account_matches` จะเป็น blocking และ `POST = False`

**3. ให้ agent ตอบ fail** แก้ `PURCHASE_ORDERS["PO-2026-0567"]["status"]` เป็น `"closed"` แล้วรัน cell agent ใหม่ — จะเห็นมันเรียก tool แล้วหยุดที่ขั้น 2 ของแผนทันที

**4. ให้ agent ตอบ unsure** แก้เลข PO ในทะเบียนให้ไม่ตรงกับที่เอกสารอ้าง — จะได้ `unsure` ซึ่ง**หยุดเอกสาร** ไม่ใช่ผ่าน

**5. ดันเกณฑ์ขึ้น** ตั้ง `AUTO_POST_MIN_CONFIDENCE = 0.99` → ทุกใบเข้าคิวคน ซึ่งคือสิ่งที่เกิดขึ้นจริงเวลาตั้งเกณฑ์สูงเกิน

**6. ปิด agent ตัวที่สอง** ลบ `ANTHROPIC_API_KEY` ออกจาก 🔑 panel → ทุกช่องเหลือ 84% แทน 95% เพราะมีหลักฐานชั้นเดียว **ตัวเลขนั้นคือมูลค่าของการอ่านครั้งที่สอง**

---

### ต่างจากระบบจริงตรงไหน

| ตรงนี้ | ระบบจริง |
|---|---|
| 3 กฎ + agent 1 ตัว | 26 กฎในตัว + กฎที่ผู้ใช้เขียนเอง + agent อีกหลายตัว |
| ทะเบียนเขียนไว้ใน notebook | CSV จาก ERP → SQLite |
| ไม่มีการปรับภาพก่อน OCR | แก้เอียง/แสง/คมชัด โดยวัดก่อน-หลัง แล้วย้อนกลับถ้าไม่ดีขึ้น |
| เขียนกฎเป็น dict มือ | พิมพ์ประโยคภาษาคน โมเดลแปลให้ครั้งเดียวตอนสร้าง |
| จบที่ `POST = True/False` | เขียนลง ledger + คิวรีวิว + API + หน้าเว็บ |
| — | วัดผลกับเอกสาร 30 ใบ: อ่านถูกกี่ % กฎยิงถูกไหม ตัดสินถูกกี่ % |

**แต่ลำดับขั้นและหลักการ เหมือนกันทุกอย่าง**
""")

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "docuflow_live.ipynb", "provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
carrying = sum(1 for c in cells if c["cell_type"] == "code" and c["outputs"])
print(f"wrote {OUT}  ({len(cells)} cells, {carrying} carrying captured output)")
