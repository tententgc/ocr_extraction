# DocuFlow — สถาปัตยกรรมระบบและ Workflow แบบละเอียด

> เอกสารฉบับภาษาไทย อธิบายว่าเอกสารหนึ่งใบเดินทางผ่านอะไรบ้าง ตั้งแต่ไฟล์เข้ามาจนกลายเป็น record
> ในระบบปลายทาง — ทีละขั้น พร้อมบอกว่าแต่ละขั้นเรียกอะไร ตัดสินใจอย่างไร และพังแล้วเกิดอะไรขึ้น

---

## 0. ภาพรวมหนึ่งหน้า

```
                                   ┌───────────────────────┐
  upload / URL / batch ──────────► │  ingest (blob store)  │  content-addressed
                                   └───────────┬───────────┘  (sha256)
                                               │
   ┌───────────────────────────────────────────▼───────────────────────────────────────┐
   │                                    PIPELINE                                        │
   │                                                                                    │
   │  1 render ─► 2 ocr ─► 3 classify ─► 4 extract ─► 5 line_items ─► 6 validate ─►     │
   │                                        │                              │            │
   │                                   2 channels                    30 built-in        │
   │                                   (LLM × 2)                     + custom rules     │
   │                                                                       │            │
   │  ─► 7 route ─► 8 reconcile (เฉพาะเมื่อ check ไม่ผ่าน) ─► 9 archive + post          │
   └────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    ▼                          ▼                          ▼
             auto_post                  needs_review                  rejected
          (เขียน record ทันที)        (เข้าคิวให้คนตรวจ)          (อ่านไฟล์ไม่ได้เลย)
```

**หลักการเดียวที่คุมทุกอย่าง:** ค่าที่ผิดหลุดไป downstream แพงกว่าการให้คนตรวจ
ทุกการตัดสินใจในเอกสารนี้จึงเอียงไปทาง "ถ้าไม่มั่นใจ ให้คนดู"

---

## 1. ขั้น `render` — ไฟล์ → ภาพหน้า

| | |
|---|---|
| **Input** | bytes จาก upload / URL / batch queue |
| **ทำอะไร** | `pypdfium2` แปลง PDF เป็นภาพ, Pillow ย่อให้ขอบยาวสุด = `OCR_IMAGE_DIM` (1800px) แล้วเซฟเป็น JPEG q92 |
| **Output** | `list[PageImage]` + `embedded_text` (ถ้า PDF มี text layer) |
| **พังแล้วไง** | `UnsupportedDocumentError` → `decision=rejected` พร้อมข้อความชัดเจน ไม่ throw ออกไปที่ API |

**จุดที่ต้องระวัง:** PDFium **ไม่ thread-safe** — เจอตอนทำ batch queue คือ 2 threads เปิดเอกสารพร้อมกันแล้ว
**segfault ทั้ง process** ตอนนี้ล็อกด้วย `threading.Lock` รอบการ render ซึ่งเสียแค่ ~0.3 วินาทีจากเอกสารที่ใช้เวลารวมหลายสิบวินาที

**ทำไมย่อที่ 1800px:** Typhoon OCR 1.5 เทรนมาที่ขนาดประมาณนี้ ใหญ่กว่านั้นเปลือง vision token โดยไม่ได้ความแม่นเพิ่ม

---

## 2. ขั้น `ocr` — ภาพ → ข้อความ

**Provider chain แบบ escalate ไม่ใช่ retry:**

```
provider_chain(settings) = [engine ที่เลือก] + [fallback เรียงตาม priority]

  typhoon_api (10) → ollama (20) → local/MLX (30) → openai (40) → claude (50)
  → pdftext (60) → tesseract (70)
```

แต่ละหน้าลองทีละ provider — ถ้าผลลัพธ์ **คุณภาพไม่ผ่าน** (`score_page_text`) ก็ข้ามไปตัวถัดไป
ไม่ retry ตัวเดิม เพราะ model เป็น deterministic ลองซ้ำก็ได้ผลเดิม

**การให้คะแนนคุณภาพ** (`ocr/quality.py`) จับ:
- ข้อความสั้นเกินไป / ว่าง
- **repetition loop** — อาการคลาสสิกของ vision model ที่พูดวนซ้ำไม่จบ
- symbol soup — อ่านไม่ออกจนกลายเป็นขยะ

**`purpose_built` — เชื่อ engine ไหนได้แค่ไหน:**

| engine | purpose_built | เหตุผล |
|---|---|---|
| Typhoon OCR (cloud / MLX / Ollama) | ✅ | เป็น model ที่สร้างมาถอดความเอกสารโดยตรง |
| PDF text layer | ✅ | ไม่ใช่การ "อ่าน" เลย เป็นตัวอักษรที่ PDF เก็บไว้เอง |
| GPT / Claude vision | ❌ | vision model ทั่วไป กำลังเดาบนงานที่ไม่ได้เทรนมา |
| Tesseract | ❌ | OCR คลาสสิก อ่อนเรื่อง layout |

ถ้าหน้าไหนอ่านด้วย engine ที่ไม่ `purpose_built` → `degraded=True` → confidence ทุก field โดนคูณ 0.90

> **บั๊กที่เคยมี:** เดิมเช็คด้วยรายชื่อ provider ที่ hardcode ไว้ แต่ Ollama รายงานชื่อเป็น `ollama:<model>`
> ทำให้ **Typhoon OCR ตัวเดียวกัน** ที่รันในเครื่องถูกนับเป็น fallback engine → ทุกเอกสารโดนหัก 0.90
> และ auto-post ไม่ผ่านทั้งที่อ่านถูกหมด ตอนนี้ถามที่ provider แทนการเดาจากชื่อ

**Ollama ใช้ native API ไม่ใช่ OpenAI-compatible** เพราะ compat layer:
- รับ `keep_alive` แล้ว **ทิ้งเงียบๆ** (พิสูจน์แล้ว: ขอ 45 นาที แต่ expiry กลับ *ลดลง*) → model โดน evict ทุก 5 นาที → cold load 168 วินาที
- ตั้ง `num_ctx` ไม่ได้เลย → default 4096 ขณะที่ภาพหน้าเดียวกินไปแล้ว ~3,100 token

---

## 3. ขั้น `classify` — นี่คือเอกสารประเภทอะไร

```
keyword scoring (เร็ว, deterministic)
        │
        ├─ ชนะขาด ────────────────► ใช้เลย
        └─ ก้ำกึ่ง ──► ถาม LLM ──► ใช้ผลนั้น
```

ได้ `doc_type` → เลือก **schema** ที่จะใช้สกัด:

| doc_type | fields | required | critical |
|---|---|---|---|
| invoice | 14 | 4 | 7 |
| receipt | 10 | 4 | 5 |
| purchase_order | 7 | 4 | 3 |
| contract | 13 | 4 | 6 |
| form | 11 | 2 | 3 |
| unknown | 7 | 0 | 0 |

- **required** = ไม่มีค่านี้ห้าม auto-post
- **critical** = ถ้าผิดแล้วเสียเงิน (ยอดเงิน, เลขที่, คู่ค้า) → ต้อง grounded ถึงจะ auto-post

---

## 4. ขั้น `extract` — หัวใจของความแม่นยำ

นี่คือส่วนที่ต่างจากระบบ OCR ทั่วไปมากที่สุด **อ่าน 2 รอบด้วย model คนละตัว**

```
                     OCR text (ทั้งเอกสาร)
                    ┌────────┴────────┐
                    ▼                 ▼
        ┌──────────────────┐  ┌──────────────────┐
        │  Channel 1       │  │  Channel 2       │
        │  primary         │  │  verify          │
        │  field-first     │  │  label-first     │
        │  "หา field นี้"  │  │  "หา label ก่อน  │
        │                  │  │   แล้วอ่านค่าถัดไป"│
        │  Typhoon v2.5    │  │  gpt-5.1         │
        └────────┬─────────┘  └─────────┬────────┘
                 │                      │
                 │   ❗ blind: ไม่เห็นคำตอบของอีกฝั่ง
                 └──────────┬───────────┘
                            ▼
                    ┌───────────────┐
                    │  merge_field  │
                    └───────┬───────┘
                            ▼
                     confidence + grounded
```

### ทำไมต้อง 2 channel และทำไมต้องคนละ model

model ตัวเดียวอ่าน 2 รอบ = ผิดซ้ำแบบเดิม สัญญาณ "ตรงกัน" จึงไม่มีความหมาย
`resolve_verify_llm` จึงเลือก backend ที่ **ต่างจากตัวหลัก** และ **ข้าม backend ที่กำลังรัน OCR อยู่**
(ถ้า OCR อยู่บน Ollama แล้วเอา verify ไปลง Ollama ด้วย → Ollama จะ evict model สลับไปมา
เคสจริงที่วัดได้คือเอกสารใบเดียวใช้เวลาเกิน 10 นาที)

verify มี **time budget 90 วินาที** — ถ้าเกิน ถอยไปใช้ label matcher แบบ deterministic แทนการรอ
(เจอกับ qwen3:4b ซึ่งเป็น reasoning model คิดยาวเป็นนาที)

### `merge_field` ตัดสินอย่างไร

**ขั้นที่ 1 — normalize** ทั้งสองฝั่ง: วันที่ พ.ศ.→ค.ศ., เลขไทย ๐-๙, จำนวนเงินมี comma, tax id

**ขั้นที่ 2 — grounding** เช็ค 2 ชั้นแยกกัน:

| คำถาม | ตอบด้วย |
|---|---|
| ข้อความที่ model ยกมา **มีอยู่บนหน้าจริงไหม** | fuzzy match ≥ 88% |
| ข้อความนั้น **ระบุค่าที่ตอบมาจริงไหม** | เทียบตามชนิดของ field |

การเทียบชนิด (สำคัญมาก):
- **date** → parse ทุก token ที่หน้าตาเหมือนวันที่ในข้อความ แล้วเทียบกันเป็น *วันที่*
  `INVOICE DATE 11/02/2019` **รองรับ** `2019-02-11` ✅
- **money / number / percent** → เทียบเป็น *ตัวเลข* — `Subtotal 145.00` รองรับ `145` ✅
- **text** → เทียบข้อความ

> **บั๊กที่เคยมี (เพิ่งแก้):** เดิมเทียบเป็น substring ตรงๆ ทำให้ `2019-02-11` ไม่อยู่ใน `11/02/2019`
> → ขึ้น "unverified" ทั้งที่อ่านถูก และ confidence ตกจาก 84% เหลือ 40%
> เอกสารที่เขียนวันที่แบบ วัน/เดือน/ปี **ทุกใบ** โดนหมด

**ขั้นที่ 3 — corroboration** ถ้าทั้ง 2 channel ได้ค่าเดียวกัน และต่างก็ยกข้อความที่มีอยู่จริงบนหน้า
→ ถือว่า **grounded** แม้ไม่มี quote ไหนระบุค่านั้นตรงๆ

นี่คือเหตุผลที่รัน 2 channel ตั้งแต่แรก การอ่านอิสระ 2 ครั้งแล้วได้ตรงกันคือหลักฐานที่แข็งแรงกว่า
การที่ quote เดียวมีค่านั้นอยู่ข้างใน — และ literal test ไม่ควรมีสิทธิ์ veto มัน

**ขอบเขตที่ยังคุมไว้:** corroboration ผ่อนแค่ว่า *ค่าผูกกับ quote ยังไง* ไม่เคยผ่อนว่า *หน้ามี quote นั้นไหม*
2 channel เห็นตรงกันแต่ยกข้อความที่ไม่มีบนหน้า = ยัง ungrounded เพราะนั่นคือการแต่งเรื่อง ไม่ใช่การตีความ

**ขั้นที่ 4 — confidence** (`confidence.py`):

```
ฐาน:  2 channel ตรงกัน       0.97
      2 channel ขัดกัน       0.45   ◄── ตัวนี้แหละที่ทำให้ silent error = 0
      channel เดียว grounded 0.86
      channel เดียว ลอยๆ     0.30
      inferred (เดาจากบริบท) 0.70

ตัวคูณ: คุณภาพ OCR  (0.55 + 0.45 × q)
        degraded     × 0.90
        critical แต่ไม่ grounded × 0.60
        มี normalization note   × 0.85
```

---

## 5. ขั้น `line_items` — ตารางรายการ

อ่านจาก HTML table ที่ Typhoon OCR คืนมา และมี fallback อ่านจาก plain text
(สำหรับ engine ที่ไม่คืน markup) ผลรวมของรายการต้องเท่ากับ subtotal ไม่งั้นกฎ
`line_items_sum_to_subtotal` จะจับได้

---

## 6. ขั้น `validate` — กฎธุรกิจ

สร้าง `RuleContext` = ทุกอย่างที่กฎมองเห็นได้ (fields, line items, master data, vendor match,
เอกสารซ้ำ, policy จาก `.env`) แล้วรัน:

### 6.1 กฎ built-in 30 ข้อ

| ไฟล์ | ใช้กับ | จำนวน |
|---|---|---|
| `rules/common.py` | ทุกประเภท | 6 |
| `rules/financial.py` | invoice / receipt / PO | 14 |
| `rules/contract.py` | contract | 5 |
| `rules/form.py` | form | 5 |

3 กลุ่มใหญ่:
- **เลขคณิต** — subtotal + vat = total, vat rate ถูกไหม, รายการรวมกันได้ไหม
- **cross-check master data** — vendor มีจริงไหม, PO มีจริงและยอดตรงไหม, บัญชีธนาคารตรงกับที่ลงทะเบียนไหม
- **ความสมเหตุสมผล** — วันครบกำหนดหลังวันออกบิล, tax id checksum, เอกสารซ้ำ

**severity 3 ระดับ:** `blocking` (ห้าม post) / `warning` (ผ่านแต่ติดธง) / `info`

### 6.2 กฎที่ผู้ใช้เขียนเอง

เขียนเป็นภาษาคนในหน้า Settings → LLM compile เป็น expression **ครั้งเดียวตอนสร้าง**:

```json
{"when":  {"field":"total_amount","op":"gt","value":100000},
 "assert":{"field":"purchase_order_ref","op":"present"}}
```

**LLM ไม่อยู่ใน loop ตอน validate เลย** — เพราะถ้าถาม model ทุกใบว่า "ใบนี้ผ่านไหม"
คำตอบจะถูกกำหนดโดยเนื้อหาในเอกสาร ซึ่งเป็นสิ่งที่ supplier ควบคุมได้

3 ด่านกัน model compile ผิด:
1. **อ่านกลับเป็นภาษาคน** ก่อนเซฟ — จับ logic กลับด้านได้
2. **ลองกับเอกสารที่ประมวลผลไปแล้ว** — ผ่านหมด/fail หมด = สัญญาณว่าเงื่อนไขกลับด้าน
3. **ปฏิเสธ test ที่ไม่มีวัน fail** เช่น `present` บน field true/false

---

## 7. ขั้น `route` — ตัดสินใจ

```
rejected      ← อ่านไฟล์ไม่ได้ / ไม่มี field ไหนสกัดได้เลย

needs_review  ← มี blocking rule ไม่ผ่าน
              │ หรือ required field หาย
              │ หรือ critical field ไม่ grounded
              └ หรือ confidence < AUTO_POST_MIN_CONFIDENCE

auto_post     ← ผ่านทั้งหมดข้างบน
```

**ข้อสังเกต:** เกณฑ์ 3 ข้อแรกเป็นคุณภาพของ *การอ่าน* ไม่ใช่กฎธุรกิจ
ดังนั้นแม้เปิด **free mode** (ปิดกฎธุรกิจทั้งหมด) เอกสารที่อ่านไม่ออกก็ยังเข้าคิวให้คนดูอยู่ดี

---

## 8. ขั้น `reconcile` — Agent สืบสวน (รันเฉพาะเมื่อ check ไม่ผ่าน)

ไม่ได้รันทุกใบ — รันเมื่อมีอะไรผิดปกติแล้วเท่านั้น เป็น **tool-calling loop** จริง:

| tool | ทำอะไร |
|---|---|
| `read_document` | อ่าน OCR text ดิบของหน้าที่ต้องการ |
| `lookup_vendor` | ค้น vendor จากชื่อ / เลขผู้เสียภาษี |
| `lookup_purchase_order` | ดึง PO มาเทียบ |
| `lookup_contract` | ดึงสัญญามาเทียบ |
| `find_posted_records` | หา record ที่เคย post ไปแล้ว (จับซ้ำ) |
| `check_arithmetic` | คำนวณซ้ำแบบ deterministic |
| `validate_tax_id` | ตรวจ checksum เลขผู้เสียภาษีไทย |

Agent คืน: สาเหตุที่แท้จริง, การกระทำที่แนะนำ, และ **correction ที่ต้อง grounded ด้วย**
(ข้อเสนอแก้ค่าที่อ้างข้อความไม่มีบนหน้า จะถูก mark ว่าไม่ grounded)

`MAX_MESSAGES = 40` + `_trim_history()` ตัดที่ขอบ turn เพื่อไม่ให้มี `tool` reply กำพร้า
(เคยเจอ HTTP 422 จาก transcript ที่ยาวไม่จำกัด)

---

## 9. ขั้น `archive` + `post` — เก็บและบันทึก

**archive** — เพิ่งรู้ประเภทเอกสารตอนนี้ จึงเป็นจุดแรกที่ filing ได้:

```
data/archive/invoice/2026-08/DOC-4F2A9B1C3D__acme-inv-88214.pdf
data/archive/contract/2026-08/DOC-1A2B3C4D5E__facility-agreement.pdf
```

ใช้ **hard link** ไม่ใช่ copy — blob store ยังคง de-duplicate อยู่ และ archive ไม่กินพื้นที่เพิ่ม
filing ล้มเหลวไม่ทำให้เอกสารล้มเหลว (log แล้วไปต่อ)

**post** — เขียน record ลง `records` table ผ่าน `post_record()` **ทางเดียวเท่านั้น**
ซึ่งเป็นเหตุผลที่ข้อกำหนด "ค่าผิดแพง" บังคับใช้ได้จริงในที่เดียว

---

## 10. Batch queue — อัปโหลดหลายไฟล์

```
POST /api/documents/batch (20 ไฟล์)
        │
        ├─ validate ทีละไฟล์ (ว่าง / ใหญ่เกิน / นามสกุลไม่รองรับ → reject ทีละใบ)
        ├─ เขียนลง data/inbox/ ด้วยชื่อไม่ซ้ำ
        └─ submit เข้า ThreadPoolExecutor ──► ตอบกลับทันที
                    │
                    │  worker count เลือกตาม engine:
                    │    local (ollama/MLX) = 1   ◄── model เดียวในเครื่อง รันพร้อมกันแล้วสลับไปมาช้ากว่า
                    │    cloud              = 4
                    ▼
        browser poll GET /api/jobs ทุก 2 วินาที
```

ไฟล์ที่พังเป็นของ job ตัวเอง — PDF เสีย 1 ใบ ไม่ทำให้อีก 19 ใบล้ม

---

## 11. Data model

```
documents ──┬──► records      (ledger ปลายทาง — เขียนผ่าน post_record() เท่านั้น)
            ├──► audit_log    (ใครทำอะไรเมื่อไหร่)
            └──► corrections  (คนแก้ field ไหนบ่อย → บอกว่าควรไปปรับปรุงตรงไหนต่อ)

vendors / purchase_orders / contracts   ◄── master data ที่กฎเอาไปเทียบ
app_settings                            ◄── policy ที่ผู้ใช้แก้จากหน้า Settings
custom_rules                            ◄── กฎที่ผู้ใช้เขียนเอง (เก็บเป็น JSON)
```

---

## 12. Configuration ที่สำคัญ

| ตัวแปร | ค่า default | คุมอะไร |
|---|---|---|
| `AUTO_POST_MIN_CONFIDENCE` | 0.90 | ต่ำกว่านี้เข้าคิวคน |
| `FIELD_REVIEW_CONFIDENCE` | 0.75 | field ต่ำกว่านี้ติดธงรายตัว |
| `LLM_VERIFY_TIMEOUT_SECONDS` | 90 | budget ของ channel ที่ 2 |
| `OLLAMA_KEEP_ALIVE` | 30m | กัน cold load 168 วินาที |
| `OLLAMA_NUM_CTX` | 8192 | ภาพหน้าเดียวกิน ~3,100 token |
| `RULES_FREE_MODE` | false | ปิดกฎธุรกิจทั้งหมด (extraction อย่างเดียว) |
| `DISABLED_RULES` | — | ปิดกฎเป็นรายข้อ |
| `PO_REQUIRED_ABOVE` | 50,000 | เกินเท่าไหร่บังคับมี PO |
| `UNKNOWN_VENDOR_BLOCKS` | invoice,purchase_order | ประเภทไหนที่คู่ค้าไม่รู้จัก = block |

---

## 13. ทำไมถึงออกแบบแบบนี้ — สรุป 5 ข้อ

1. **2 channel ที่ผิดคนละแบบ** ดีกว่า model เดียวที่เก่งกว่า — เพราะสิ่งที่ต้องการคือ *สัญญาณว่าไม่มั่นใจ* ไม่ใช่คำตอบที่ดีที่สุด
2. **Grounding บังคับให้ทุกค่ามีที่มา** — model ต้องยกข้อความจากหน้ามา และเราตรวจว่ามีจริง
3. **Validation เป็น deterministic code** — model ที่ตัดสินว่า VAT ถูกไหม คือ model ที่ถูกโน้มน้าวได้
4. **LLM ใช้ตอน compile ไม่ใช่ตอน judge** — สำหรับกฎที่ผู้ใช้เขียนเอง
5. **ทุกการผ่อนเกณฑ์ต้องมองเห็น** — free mode, กฎที่ปิด, threshold ต่ำ ขึ้นเตือนค้างไว้ทุกใบ
