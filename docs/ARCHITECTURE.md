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

> The same three views as diagrams — where a model is called, the pipeline with
> the human loop, and the agents with their tools — are in
> [architecture.drawio.xml](architecture.drawio.xml), openable at
> [diagrams.net](https://app.diagrams.net).

## Pipeline

The order below is the order in `Pipeline._run` (`pipeline.py`). Each box names
what actually does the work.

```
 file / URL
     │
     ▼
┌─────────────┐  ingest/store.py, ingest/fetch.py
│   ingest    │  sha256 content addressing, magic-byte sniff, SSRF-guarded fetch
└─────┬───────┘  a re-upload of the same bytes is the same blob
      ▼
┌─────────────┐  ocr/render.py — pypdfium2, behind a module lock (not thread-safe)
│   render    │  PDF → page images, AND the page's own text layer is pulled out
└─────┬───────┘  capped at OCR_MAX_PAGES; a short read degrades the run and says so
      ▼
┌─────────────┐  ocr/preprocess.py — Pillow + NumPy
│ preprocess  │  measure → correct → re-measure → revert if it did not help
│             │  lighting BEFORE skew (skew is read off an ink mask)
└─────┬───────┘  a clean digital PDF is untouched and writes no second file
      ▼
┌─────────────┐  ocr/engine.py walks ocr/factory.py's provider chain
│    read     │  ① pdf-text-layer — the characters already in the file, ~10ms, free
│             │  ② Typhoon OCR 1.5 → Markdown + HTML tables
└─────┬───────┘  each result scored by ocr/quality.py; poor text escalates
      ▼
┌─────────────┐  agents/classifier.py
│  classify   │  keyword vote first; an LLM confirms only when the vote is close
└─────┬───────┘  only the most specific matched phrase votes
      ▼
┌─────────────┐  TWO INDEPENDENT CHANNELS, different backends
│   extract   │  ① agents/llm_extractor.py — field-first, JSON + verbatim quote
│             │  ② agents/verify_extractor.py — label-first, reports the page as printed
└─────┬───────┘  agents/merge.py reconciles them
      ▼
┌─────────────┐  normalize/ — dates (Buddhist era, Thai numerals), money, tax IDs
│  normalize  │  BEFORE comparison, or the two channels disagree about everything
└─────┬───────┘  and the normalised value is what gets stored
      ▼
┌─────────────┐  does the quoted evidence exist in the OCR text?
│   ground    │  does the quote state the value, compared by field type?
└─────┬───────┘
      ▼
┌─────────────┐  agents/rule_extractor.py — HTML <tr> path, plain-text path
│ line items  │  both drop totals rows; rows must sum to the subtotal
└─────┬───────┘
      ▼
┌─────────────┐  rules/ — 26 built-ins + user rules from rules/custom.py
│  validate   │  arithmetic, VAT, duplicates, vendor master, 3-way PO match,
│             │  bank account, contract parties, tax-ID checksum
└─────┬───────┘  agent rules get a narrowed toolbox (agents/rule_agent.py)
      ▼
┌─────────────┐  rules/repair.py — which value is wrong, from the relations that hold
│   diagnose  │  silent when two values would explain the failure equally well
└─────┬───────┘
      ▼
┌─────────────┐  confidence.py — did we READ it right (not: may it post)
│ confidence  │  only the 7 rules in MISREAD_SIGNALS move the score
└─────┬───────┘
      ▼
┌─────────────┐  routing.py — auto_post │ needs_review │ rejected
│    route    │  no blocking failure · critical values grounded · confidence ≥ threshold
└─────┬───────┘
      ▼
┌─────────────┐  agents/reconciler.py — runs ONLY after a check already failed
│  reconcile  │  so the happy path costs nothing
└─────┬───────┘
      ▼
┌─────────────┐  store/records.py post_record + audit_log · archive.py files by type
│    post     │  export.py takes it back out as CSV or JSONL
└─────────────┘
```

Every stage appends a `TraceStep`, so the UI shows exactly what happened, in
order, with timings. `DocumentRun` is frozen — a stage returns a new object
rather than mutating, which keeps the trace a truthful history.

### Reading: the file first, a model second

`ocr/factory.py` builds an ordered chain and `ocr/engine.py` tries each engine
until one returns text that passes `ocr/quality.py`. The text layer sits at the
front of it:

| | Engine | When it answers |
|---|---|---|
| ① | `pdf-text-layer` | the PDF carries ≥40 characters of its own text |
| ② | configured engine, default Typhoon OCR 1.5 | everything else |
| ③ | Ollama → MLX → OpenAI → Claude → Tesseract | the one above was unavailable |

A digital PDF's characters were written into the file by whatever produced it.
Recognition cannot improve on them and every model call spent on such a page
buys nothing — measured on the demonstration corpus, about **10ms against 54
seconds a page**, and no model call at all.

It is safe at the front because it removes itself. `PdfTextLayerOcr` raises
`OcrUnavailableError` when a page has no embedded text, and the chain skips a
provider that says it cannot help — so a photograph, a JPG or a scan reaches
exactly the engine it would have reached anyway. Text that is present but poor
fails the same quality gate as any model's output and escalates.

`OCR_STRICT=true` disables the whole chain: a benchmark naming an engine must
measure that engine rather than the work of whatever wrote the PDF.
`OCR_PREFER_TEXT_LAYER=false` disables only the fast path.

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

### Confidence and permission are different questions

**Confidence answers one thing: did we read the values correctly.** Whether a
document may post is decided in `routing.decide`, from three conditions that
are checked separately.

Only the seven rules in `MISREAD_SIGNALS` move the score, weighted by how
serious the rule says it is:

| Severity | Penalty |
|---|---|
| `blocking` | −0.20 |
| `warning` | −0.05 |
| `info` | 0 |

Floored at `MISREAD_FLOOR = 0.5`, so no pile of findings can drive a document
to zero.

These were one number once, and every warning quietly became a block: three
advisory findings took a perfectly-read document from 95% to 72% under an 84%
threshold. A rule about *policy* — an unregistered vendor, a closed purchase
order — says nothing about whether the page was read correctly, so it must not
move a number that claims to measure exactly that.

### The three severities behave differently

| | Effect |
|---|---|
| `blocking` | stops the document; nothing posts past one, even on approval |
| `warning` | recorded and shown, **holds nothing** |
| `info` | a note |

A warning that holds documents is a blocking rule that has not admitted it, and
it is how a human-in-the-loop system ends up with a human in every loop.

`po_amount_within_tolerance` is the worked example: billing **over** a purchase
order is a payment risk and blocks; billing **under** it is partial delivery or
progress billing, pays nobody too much, and is a warning.

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
| `read_line_items` | the rows already parsed, rather than hunting the page again |
| `lookup_vendor` | master record, registered bank account, status |
| `lookup_purchase_order` | three-way match |
| `lookup_contract` | contract register |
| `find_posted_records` | confirm a suspected duplicate |
| `check_arithmetic` | deterministic sums |
| `validate_tax_id` | Thai check-digit test |

**A tool's answer has to explain itself**, because it is read by a model — and
increasingly by a model following a rule that another model wrote, neither of
which has seen the schema. `lookup_purchase_order` returned a bare `amount`;
purchase orders are recorded net of tax and invoice totals are not, so an agent
compared the total against it and failed a clean invoice for a 7% overrun that
was just VAT. The answer now carries `amount_basis` and says which invoice
figure to compare it with.

**`select_tools` narrows the box.** A rule declares the tools it needs and gets
those; declaring none gets all of them. Least privilege has a limit, though:
cutting `check_arithmetic` from a rule whose plan says "add these and compare"
does not make the agent safer, it moves the arithmetic back inside the model
where nothing can check it. `review_before_approving` reports that to whoever
approves the rule instead of silently rewriting it.

Any correction the agent proposes is **re-grounded before a reviewer sees it**.
If the value is not printed on the page, the UI shows it as "not on page" and
offers no Apply button. The agent assists a human; it never changes a value.

Backends that cannot do tool calling degrade to a single prompted turn
automatically — including models whose advertised capabilities are wrong, which
is detected by the first turn failing.

## Two documents in one file

A scanner makes one PDF from a stack of paper. Read as one document that file
yields one invoice number and one total, and **every other invoice in the stack
is silently never recorded** — the run looks like an ordinary single invoice
that posted cleanly.

`ingest/splitting.py` finds the boundaries, deterministically, for the same
reason the classifier is: a boundary is a fact about the text. Signals in
priority order:

| | Signal | Meaning |
|---|---|---|
| 1 | `Page 2 of 5` / `หน้า 2 จาก 5` | states outright that it continues something |
| 2 | a document reference that changed | two numbers, two documents |
| 3 | a heading over a reference not seen before | a front page where the last had no number |

The two mistakes cost differently, and the module is weighted accordingly.
Missing a boundary is silent. Splitting where there is none is loud — page two
becomes a fragment with no header and no total, fails half the rules, and is in
the review queue in ten seconds. So it is **willing to split and reluctant to
merge**, and where it cannot tell it reports `uncertain` rather than guessing;
`docuflow split` then exits 2 without writing anything.

## Learning from reviewers

`corrections` recorded every edit and reported hotspots to a person. Nothing in
the pipeline read them back until `unreliable_fields` did.

**What is learned is trust, never values.** A correction says what the right
answer was on *one* document; carrying it forward would put one invoice's total
on another, which is the exact failure the system exists to prevent. What
generalises is that this vendor's layout, in this field, has repeatedly defeated
the extractor — so confidence in that field drops, the document stops
auto-posting, and a person looks at it.

Three guards keep it honest:

- **counted by document, not by edit** — three fixes to one page is one person
  fixing one page; the same field wrong on three documents is a layout
- **scoped to the vendor and type** — one supplier's problem is not everyone's,
  and an unattributable correction teaches nothing
- **aged out after 180 days** — distrust sends documents to review and review
  produces corrections, so without a window that loop only tightens

## Data model

SQLite, and deliberately **two layers in one row** rather than either extreme.

| Table | Shape |
|---|---|
| `documents` | indexed columns + `payload` (the whole run as JSON) |
| `records` | the system of record; written only via `post_record` |
| `audit_log` | who did what, when, forced or not |
| `corrections` | every reviewer edit, aggregated into accuracy hotspots |
| `vendors` / `purchase_orders` / `contracts` | master data, **relational, no JSON** |

`documents` lifts out the seven things the system filters and counts on —
`decision`, `doc_type`, `sha256`, `confidence`, `record_id`, `created_at`,
`filename` — and keeps everything else in `payload`: the OCR text, all fields
with their evidence and grounding, the line items, every finding, and the full
trace.

Neither extreme works. All-JSON makes the review queue slow: it filters on
`decision` and needs an index. All-relational needs a column per field, and the
document types do not share fields — across invoice, receipt and contract,
`currency` is the only extracted field in common, so a fixed table is either
mostly NULL or an `ALTER TABLE` every time a schema is added.

Keeping the whole run in `payload` is also what makes **"re-check under current
policy"** possible: change a rule, and the stored fields are judged again
without re-reading the page. No OCR, no model call.

The trap is that the two layers must move together. Deleting a ledger entry
cleared `record_id` in the column but left `decision = auto_post` in both, so
the queue rendered "Posted automatically" beside an empty record column:

```sql
UPDATE documents
   SET record_id = NULL,
       decision  = ?,                                  -- layer 1
       payload   = json_set(payload, '$.decision', ?)  -- layer 2
 WHERE document_id = ?
```

### Getting it back out

`export.py` streams the ledger as **CSV** for a person or **JSONL** for another
system, and they are not variations on one thing.

CSV flattens the payload into columns, keeps amounts numeric so a spreadsheet
can sum them, and starts with a UTF-8 BOM — without it Excel on Windows renders
every Thai counterparty as mojibake. The columns depend on `doc_type`, because
one header cannot describe an invoice and a contract at once without being
mostly empty for both; a mixed export gets only the columns that mean the same
thing on every type, plus `doc_type` on each row.

JSONL keeps each record whole, payload parsed rather than left as an escaped
string of JSON inside JSON. Both stream and neither is paginated: `per_page`
exists so the browser is not handed ten thousand rows, and an export inheriting
that cap would stop at the first page.

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

## How the guarantees are held

1,021 tests, no API key needed, plus 20 opt-in browser tests.

| Suite | Holds |
|---|---|
| `pytest -q` | the pipeline, rules, normalisation, stores, API |
| `pytest -m ui` | `web/app.js` in chromium — every UI bug that shipped, put back |
| `tests/test_demo_set.py` | each demo filename's claim against its own ground truth |
| `tests/test_demo_notebook_parity.py` | the notebook and the system agree about the same 12 files |
| `tests/test_notebook_prompts.py` | the built `.ipynb` still matches the prompts it was lifted from |

Three habits are worth copying rather than the tests themselves.

**Credentials and business policy are pinned in `conftest.py`.** A green suite
has to be a statement about the code, not about one machine's `.env`. That has
caught six settings the hard way.

**A test written after the fix proves nothing until the bug goes back in.** The
first sticky-header test passed with the bug reintroduced: the viewport was
900px and the content was 900px, so the page could not scroll, sticky
positioning never engaged, and the assertion measured a layout at rest — where
the broken and the fixed version look identical. Every UI test here was checked
by putting its bug back and watching it go red.

**When the evaluation says the system is wrong, suspect the corpus first.** On
this project it has been the corpus more often than the pipeline.
