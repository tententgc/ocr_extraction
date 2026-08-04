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
┌─────────────┐   skew / blur / contrast / lighting measured, then only the
│ preprocess  │   corrections a measurement asks for — each re-measured and
│             │   reverted if it did not help. Clean pages pass through untouched.
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
│   extract   │   ├── channel 1: field-first LLM, JSON with a verbatim quote
│             │   └── channel 2: label-first LLM on a different backend
└─────┬───────┘
      ▼
┌─────────────┐   does the quoted evidence exist on the page?
│   ground    │   does the quote state the value (compared by field type)?
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
┌─────────────┐   which of the values is the wrong one, and what it should say —
│   diagnose  │   from the relations that still hold. Silent where two would
│             │   explain the failure equally well.
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

Two language models read the same page. **Channel 1** is framed field-first
("find this field"); **channel 2** is framed label-first ("find the label, then
read what follows"). Neither sees the other's answers.

`resolve_verify_llm` picks the second backend, and refuses two of them: the
primary's own backend, because two passes of one model repeat one mistake and
their agreement proves nothing; and the backend hosting local OCR, because a
second local model does not run beside the OCR model, it evicts it. Claude leads
the remaining order and the loop falls through when it is unreachable, so an
outage degrades the cross-check rather than removing it. The pass carries a time
budget (`LLM_VERIFY_TIMEOUT_SECONDS`) because a reasoning model can spend minutes
on a page the first pass read in seconds.

When both agree, confidence is 0.97 before penalties. When they disagree, it
drops to 0.45 and the disagreement is written into the field's notes — that
document goes to a human.

Agreement is compared by type. Numbers compare numerically; a passage agrees with
a longer reading of itself when its words are a subset of the longer one's, so
"laws of Thailand" and "the laws of Thailand" are one answer rather than two.
Numeric dates are resolved before comparison: the page's own convention decides,
detected from any date on it carrying a component above 12, so both channels land
on the same day rather than each guessing at `11/02/2019`.

The deterministic label matcher takes the second slot only when no second model
is reachable, so the system still works with no model at all
(`LLM_PROVIDER=rules`), which is what CI runs.

### 2. Evidence grounding

The extraction prompt requires a verbatim quote for every non-null value.
`agents/grounding.py` then checks two things:

- the quote really appears in the OCR text (normalised, then fuzzy at 88%)
- that quote **states** the reported value, compared by field type: dates as
  dates, money and percentages numerically, and an enum not at all — it is a
  classification the schema constrains, not a transcription

Agreement can also ground a value on its own. When both channels report the same
value and each quotes a line that is really on the page, they corroborate each
other even if no single quote states the value verbatim. The limit is deliberate:
that relaxes how a value ties to its quote, never whether the page contains the
quote.

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

## Batch processing

`POST /api/documents/batch` accepts many files, validates each one, stages them
to `data/inbox/` and queues them, returning immediately. The browser polls
`/api/jobs`. Twenty documents is ten minutes of work and not something to hold a
request open for.

Pool size follows the engine: a local model is one resident model on one machine,
so two documents at once make Ollama swap between them and the pair finishes
slower than running them in sequence — local engines get one worker, cloud
engines four. PDF rendering is behind a process-wide lock because PDFium is not
thread-safe; without it, two workers segfault the interpreter.

## Filing by document type

Ingest storage is content-addressed, because when the bytes arrive nobody knows
what the document is. Once classified, each document is also filed:

```
data/archive/invoice/2026-08/DOC-4F2A9B1C3D__acme-inv-88214.pdf
data/archive/contract/2026-08/DOC-1A2B3C4D5E__facility-agreement.pdf
```

Hard-linked, so the archive costs no extra bytes. Filing never fails a run.

## Runtime policy and user-authored rules

The Settings screen changes the auto-post threshold, switches individual rules
off, and turns every rule off (free mode) — persisted in `app_settings`, with
`.env` remaining the declared default and a reset always available. A rule name
that matches nothing is rejected rather than stored.

Rules can also be written in plain language. The model compiles the sentence into
a declarative expression **once**, at authoring time; from then on it is
evaluated by `rules/expressions.py`, which has no `eval`, no attribute access and
no user-supplied regex. A model consulted per document has its answer shaped by
the document, which is what a supplier controls. Three guards sit on the compile
step: the rule is read back in English, tried against documents already
processed, and refused outright if it can never fail.
