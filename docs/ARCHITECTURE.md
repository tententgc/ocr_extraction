# Architecture

## The problem this solves

A business unit receives documents — invoices, contracts, forms — reads them by
hand, types key fields into a downstream system, and cross-checks those values
against business rules and existing records. It is slow, error-prone and does not
scale.

The hard part is not reading the page. It is **knowing when not to trust what you
read**. An automated pipeline that is right 95% of the time and silently wrong 5%
of the time is worse than useless in accounts payable, because the 5% becomes
wrong payments. Every design decision below follows from that.

## Pipeline

```
 file / URL
     │
     ▼
┌─────────────┐   sha256, magic-byte sniff, SSRF-guarded fetch
│   ingest    │
└─────┬───────┘
      ▼
┌─────────────┐   PDF → page images (pypdfium2), text layer kept as a free cross-check
│   render    │
└─────┬───────┘
      ▼
┌─────────────┐   Typhoon OCR 1.5 → Markdown + HTML tables
│     OCR     │   quality scored; escalates to the next engine if the read is poor
└─────┬───────┘
      ▼
┌─────────────┐   keyword vote, LLM confirmation only when the vote is close
│  classify   │
└─────┬───────┘
      ▼
┌─────────────┐   TWO INDEPENDENT CHANNELS
│   extract   │   ├── rule channel: schema label aliases → value
│             │   └── model channel: schema-driven JSON with a verbatim quote
└─────┬───────┘
      ▼
┌─────────────┐   does the quoted evidence exist on the page?
│   ground    │   does the quote actually contain the reported value?
└─────┬───────┘
      ▼
┌─────────────┐   dates (Buddhist era, Thai numerals), money, tax IDs, enums
│  normalize  │
└─────┬───────┘
      ▼
┌─────────────┐   arithmetic, VAT, duplicates, vendor master, 3-way PO match,
│  validate   │   bank account, contract dates, tax-ID checksum
└─────┬───────┘
      ▼
┌─────────────┐   tool-using agent — only when a check has already failed
│  reconcile  │
└─────┬───────┘
      ▼
┌─────────────┐   auto_post │ needs_review │ rejected
│    route    │
└─────┬───────┘
      ▼
┌─────────────┐   the downstream system of record + audit trail
│    post     │
└─────────────┘
```

Every stage appends a `TraceStep`, so the UI shows exactly what happened, in
order, with timings. `DocumentRun` is frozen — a stage returns a new object
rather than mutating, which keeps the trace a truthful history.

## The four accuracy mechanisms

### 1. Two independent extraction channels

The **rule channel** matches the schema's label aliases (`"Amount Due"`,
`"จำนวนเงินรวมทั้งสิ้น"`) and takes the value that follows. The **model channel**
asks the LLM for the same fields.

When both agree, confidence is 0.97 before penalties. When they disagree, it
drops to 0.45 and the disagreement is written into the field's notes — that
document goes to a human. Agreement between two mechanisms that fail differently
is the cheapest reliable signal available.

The rule channel also means the system still works with no model at all
(`LLM_PROVIDER=rules`), which is what CI runs.

### 2. Evidence grounding

The extraction prompt requires a verbatim quote for every non-null value.
`agents/grounding.py` then checks two things:

- the quote really appears in the OCR text (normalised, then fuzzy at 88%)
- the reported value actually appears **inside that quote**

The second check catches the most dangerous failure mode: a model that quotes a
real line but reports a number that is not in it. An ungrounded critical field
can never auto-post, regardless of confidence.

### 3. Deterministic recomputation

Arithmetic is never taken from the model. `rules/financial.py` recomputes
subtotal + tax and compares it against the printed total. The agent is given a
`check_arithmetic` tool for the same reason — it is told to use the tool rather
than add up itself.

### 4. Conservative routing

Straight-through posting requires **all** of:

- no blocking rule failure
- overall confidence ≥ `AUTO_POST_MIN_CONFIDENCE` (default 90%)
- every required field present
- every critical field grounded in the page text

Anything less goes to a person. A reviewer who approves a document whose blocking
rule still fails gets a 409 — they must correct the value or reject it. That is
the single narrow gate into `store/records.py`.

## Confidence

Confidence is computed from verifiable facts, not from the model's self-report,
which is not calibrated. `confidence.py`:

| Situation | Base |
|---|---|
| Both channels agree | 0.97 |
| Model only, grounded | 0.86 |
| Rule only | 0.80 |
| Channels disagree | 0.45 |
| Model only, ungrounded | 0.30 |

Then multiplied by OCR quality, and reduced for degraded recognition, ungrounded
critical fields, and values that needed reinterpretation during normalisation.

Document confidence weights critical fields ×3 and required fields ×2, then
applies penalties for missing required fields and failed rules — it answers "is
it safe to post this without a human?", not "how did the model do on average?".

## Plugin architecture

OCR engines and agent models are plugins. Adding a backend is one call:

```python
register_ocr(OcrPlugin(
    key="my_engine",
    label="My Engine",
    build=lambda settings, model: MyOcr(model or settings.my_model),
    default_model=lambda settings: settings.my_model,
    list_models=lambda settings: discover(),
    fallback_priority=35,
))
```

The factory, the `/api/models` picker, the CLI and the benchmark all read the
registry, so nothing else changes. Built-in: Ollama, Typhoon Cloud, OpenAI,
Claude, MLX, PDF text layer, Tesseract.

Every engine is tried in priority order, so a missing model or an expired key
degrades the run instead of breaking it — and any run that fell back to a weak
engine is marked `degraded`, which lowers every field's confidence.

## The agent and its tools

The reconciliation agent runs **only after a check has already failed**, so the
happy path costs nothing. It is a real tool-using agent, not a single prompt:

| Tool | Why it exists |
|---|---|
| `read_document` | re-read the page instead of recalling it |
| `lookup_vendor` | master record, registered bank account, status |
| `lookup_purchase_order` | three-way match |
| `lookup_contract` | contract register |
| `find_posted_records` | confirm a suspected duplicate |
| `check_arithmetic` | deterministic sums |
| `validate_tax_id` | Thai check-digit test |

Any correction the agent proposes is **re-grounded before a reviewer sees it**.
If the value is not printed on the page, the UI shows it as "not on page" and
offers no Apply button. The agent assists a human; it never changes a value.

Backends that cannot do tool calling degrade to a single prompted turn
automatically — including models whose advertised capabilities are wrong, which
is detected by the first turn failing.

## Data model

- `documents` — every processed run, full JSON payload, replayable
- `records` — the downstream system of record; written only via `post_record`
- `audit_log` — who did what, when, forced or not
- `corrections` — every reviewer edit, aggregated into accuracy hotspots
- `vendors` / `purchase_orders` / `contracts` — master data

## Measuring it

`docuflow evaluate` runs the ground-truth corpus in an isolated database and
reports field accuracy, defect recall and — the number that matters — the
**silent error rate**: documents that auto-posted while carrying a wrong critical
value. The target is zero; a system with 100% field accuracy and no review queue
would be preferable, but a system with 92% field accuracy and 0% silent errors is
safe to deploy, which the first constraint of the problem demands.

`docuflow benchmark` runs the same corpus across several model combinations so
engine choice is an evidence-based decision.
