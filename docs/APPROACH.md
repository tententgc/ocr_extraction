# Approach

Answers to the three questions, grounded in the working system in this repository.
Everything described here is built and measured, not proposed — where something is
a next step rather than shipped, it says so.

---

## 1. Technical Approach & Architecture

### The problem is trust, not reading

Modern OCR reads a page well. The failure that costs money is subtler: the system
returns a value that *looks* right, is wrong, and nobody notices. In accounts
payable that becomes a wrong payment.

So the architecture is organised around a single question — **can this value be
trusted without a human?** — and the headline metric is not field accuracy but
the **silent error rate**: documents posted automatically while carrying a wrong
critical value. A pipeline at 99% field accuracy with silent errors is not
deployable, because you cannot tell which 1% of payments to chase. A pipeline at
91% with a 0% silent error rate is, because everything uncertain is visibly
queued.

Full results, per document, with what was read wrong and why:
[EVALUATION.md](EVALUATION.md). It also records the one silent error this corpus
did produce, and the two gaps that let it through — the more useful half of an
evaluation is the half that fails.

### End-to-end design

```
file / URL → ingest → render → preprocess → OCR → classify → extract → ground
           → normalize → validate → diagnose → route → reconcile → post
```

Each stage appends to an immutable trace, so any decision can be replayed and
audited. `DocumentRun` is frozen: a stage returns a new object rather than
mutating one, which is what makes the trace a truthful history rather than a log.

### Choice of models

**OCR: a document-specific model, not a general vision model.**
Typhoon OCR 1.5 (`typhoon-ocr1.5-3b` / `-2b`) is fine-tuned for document layout
and Thai script. This is not a preference — it is measurable. On a degraded Thai
scan in the corpus:

| Engine | Vendor name read | Address read |
|---|---|---|
| Ground truth | `บริษัท เจ้าพระยา แฟซิลิตี้ จำกัด` | `77/8 ถนนพระราม 3 … ยานนาวา 10120` |
| Typhoon OCR 1.5 (local) | `…แฟชิลิตี้…` (one wrong character) | correct |
| Claude Sonnet 5 | correct | correct |
| **OpenAI gpt-5.1** | **`…แพลติค…`** | **`ถนนพหลโยธิน … พญาไท 10400`** |

`gpt-5.1` did not misread the page — it **invented a plausible Thai company name
and a complete street address that are not on the page at all**. A general vision
model asked to "read this document" will happily produce fluent, wrong text. This
is exactly the failure mode that grounding exists to catch, and exactly why the
default engine is a document OCR model.

Across the whole corpus (`docuflow benchmark`, strict mode, five documents):

| Engine | Field accuracy | Silent errors | Defects caught | s/page |
|---|---|---|---|---|
| Typhoon Cloud `typhoon-ocr-v1.5` | 94.3% | 0% | 100% | 19.1 |
| MLX `typhoon-ocr1.5-2b-8bit` | 92.9% | 0% | 100% | 37.0 |
| Claude `claude-sonnet-5` | 91.4% | 0% | 100% | 30.2 |
| OpenAI `gpt-5.1` | 85.7% | 0% | 100% | 29.0 |
| Tesseract | 78.6% | 0% | 100% | 27.7 |
| PDF text layer | 61.4% | 0% | 50% | 17.2 |

**The silent-error column is the point.** It is 0% for every engine, from the
best to the one that cannot read scans at all. Engine quality moves the
straight-through rate; it does not move whether wrong values escape. That
separation is the whole design, and this table is the evidence for it — a weaker
engine buys you more review work, not more bad payments.

Two counter-intuitive results worth noting: the 2B MLX build beat the 3B Ollama
build (92.9% at 37 s/page versus 82.9% at 75 s/page), and a general-purpose
frontier model came fourth behind a 2B local one. Neither is what a model card
would predict, which is the argument for benchmarking on your own documents.

**Reasoning: a general instruct model with tool calling.** Typhoon 2.5 for Thai
documents, or Claude / OpenAI / a local Ollama model — all interchangeable.

**Why models are plugins.** Model choice ages badly. Registering a backend is one
call, and the factory, the UI picker, the CLI and the benchmark all read the
registry:

```python
register_ocr(OcrPlugin(
    key="my_engine", label="My Engine",
    build=lambda settings, model: MyOcr(model or settings.my_model),
    default_model=lambda settings: settings.my_model,
    fallback_priority=35,
))
```

Seven OCR engines and six agent backends ship today. The operator switches
between them from a dropdown in the UI, with no restart.

### Techniques used, and the ones deliberately not used

**Structured extraction driven by a schema, not a prompt.** One `FieldSpec`
declaration drives four things: the extraction prompt, the deterministic
extractor, the normaliser and the review UI. Adding a document type means adding
a schema — no other file changes.

```python
FieldSpec(
    name="total_amount", label="Total", type="money",
    required=True, critical=True,
    aliases=("Total", "Grand Total", "Amount Due", "ยอดรวมทั้งสิ้น", "รวมทั้งสิ้น"),
)
```

`critical=True` is the important flag: it marks fields where a wrong value costs
money, and those fields get stricter grounding and can never auto-post
unverified.

**Tool calling, used narrowly.** The agent gets seven tools — `read_document`,
`lookup_vendor`, `lookup_purchase_order`, `lookup_contract`,
`find_posted_records`, `check_arithmetic`, `validate_tax_id`. The design rule is
that a tool exists for anything the model must not do from memory: re-read the
page, look up a record, or compute a sum.

**Prompts kept short.** Long prompts cost tokens on every turn and dilute the two
instructions that actually change behaviour — quote your evidence, and abstain
when unsure. The extraction system prompt is seven lines (466 characters); the
reconciler's is nine.

**No fine-tuning.** Typhoon OCR 1.5 is already fine-tuned for this task. Fine-
tuning the extraction model would be premature: the corrections table records
every reviewer edit by document type and vendor, and that dataset is the
prerequisite. Fine-tune when the correction hotspots show a stable, repeating
error the prompt cannot fix — not before.

**No RAG.** RAG retrieves unstructured knowledge. This problem needs exact lookups
against structured master data — a vendor by tax ID, a purchase order by number.
That is a database query, and a database query is right every time. Fuzzy matching
is used only for vendor *names*, where OCR damage is real (`แฟชิลิตี้` still
resolves to the correct vendor at 86% similarity), and the match method and score
are reported so confidence stays honest.

### Handling varied and unstructured documents

The corpus is deliberately inconsistent: seven layouts, two languages, Buddhist-era
dates, Thai numerals, HTML tables, and — in the working corpus — three phone
photographs and two faded photocopies, because that is what an accounts-payable
inbox contains and a clean render measures nothing.

- **Preprocessing** measures skew, sharpness, ink-to-paper separation and
  lighting drift, applies only the corrections a measurement asks for, and
  reverts any that did not improve the number it exists to improve. A clean
  digital PDF passes through untouched. The ordering is load-bearing and was
  found by measuring rather than by reasoning: skew is estimated from a mask of
  where the ink is, so on a page lit from one side that mask marks the shadow,
  and the lighting has to be flattened before the angle can be trusted.
- **Classification** is a cheap keyword vote first; the model is consulted only
  when the vote is close. Most documents never cost a model call.
- **Layout independence** comes from reading labels rather than positions. Where
  a document has no label at all — invoices rarely label their own name — a
  narrowly-scoped letterhead heuristic applies, flagged as positional evidence
  and scored down accordingly.
- **Normalisation** is where a lot of accuracy is won: Buddhist-era years
  (๒๕๖๙ → 2026), Thai numerals (๙๘,๔๔๐.๐๐ → 98440.0), both thousands-separator
  conventions, accounting negatives, and Thai tax-ID check digits.

### Grounding — the core reliability mechanism

Four layers, in order of how much they catch:

**1. Two independent channels.** Two language models read the same page — the
first framed field-first ("find this field"), the second label-first ("find the
label, then read what follows") — on different backends, neither seeing the
other's answers. Agreement scores 0.97; disagreement drops to 0.45 and the
conflict is written into the field's notes.

Different *models* matters as much as different framing: two passes of one model
over one page repeat the same mistake, and agreement between them is not a
signal. The resolver therefore refuses to reuse the primary's backend, and
refuses the backend hosting local OCR as well — a second local model does not run
beside the OCR model, it evicts it.

The deterministic label matcher takes the second slot only when no second model
is reachable, so the system still runs with no credentials at all.

**2. Evidence grounding.** Every non-null value must carry a verbatim quote. Two
checks follow:

- does the quote exist in the OCR text (normalised, then fuzzy at 88%)?
- does that quote **state** the reported value?

The second check compares like with like. Dates are compared as dates, so
`INVOICE DATE 11/02/2019` supports `2019-02-11`; money and percentages are
compared numerically, so `Subtotal 145.00` supports `145`; an enum is a
classification the schema constrains, so a contract that "shall automatically
renew" supports `auto_renew=yes` without the word "yes" appearing anywhere.
A literal substring test failed all three — and reported correct readings as
unverified more often than it caught a wrong one.

The second check catches the worst failure: a model quoting a real line while
reporting a number that is not in it — convincing to a reviewer skimming the page.

**3. Deterministic recomputation.** Arithmetic is never taken from the model.

**4. Calibrated confidence.** Model self-reported confidence is not used; it is
not calibrated. Confidence is computed from verifiable facts:

| Situation | Base score |
|---|---|
| Both channels agree | 0.97 |
| Model only, grounded | 0.86 |
| Rule only | 0.80 |
| Channels disagree | 0.45 |
| Model only, ungrounded | 0.30 |

Then multiplied by OCR quality and reduced for degraded recognition, ungrounded
critical fields, and values that needed reinterpretation.

**Grounding in action.** On the invoice whose stated total is wrong, the agent
correctly determined the total should be 68,480.00. The UI shows that suggestion
marked **"not on page"** with no Apply button — because 68,480.00 is not printed
anywhere; the document says 86,480.00. The agent may not substitute a computed
value for a printed one. A human decides whether the supplier's invoice is wrong.

**The honest limit.** Two channels reading the same OCR text cannot catch an OCR
error — both agree, both are wrong. On the Thai scan, `๑๕` was read as `๑๔` and
the date was confidently wrong. What catches that class of error is the layer
above: business rules and master-data cross-checks. The same document's bank
account was misread as `550001111`, and `bank_account_matches_vendor` blocked it
before it could redirect a payment.

---

## 2. Agentic Workflow & Automation

### Where agents belong, and where they do not

The guiding principle: **an agent is for judgement; deterministic code is for
everything with a right answer.** Using an LLM where a rule suffices adds cost,
latency and a failure mode for nothing.

| Stage | Implementation | Why |
|---|---|---|
| Ingest, render | Deterministic | File handling has a right answer |
| OCR | Model | Genuinely needs one |
| Classify | Keywords first, model only when ambiguous | Most documents are obvious |
| Extract | Deterministic + model, both | Cross-check is the accuracy signal |
| Ground | Deterministic | String matching, not judgement |
| Normalize | Deterministic | Date and money parsing must be repeatable |
| **Validate** | **Deterministic** | **Business rules must be auditable and identical every run** |
| Reconcile | **Agent with tools** | Explaining *why* something failed is judgement |
| Route | Deterministic | Policy, not opinion |
| Post | Deterministic | One narrow gate |

The validation layer is deliberately not agentic. "Does subtotal + tax equal the
total?" has one correct answer, must be identical on every run, and must be
explainable to an auditor. An LLM would make it slower, more expensive and
non-reproducible.

### The agent, precisely scoped

The reconciliation agent runs **only after a check has already failed** — nothing
on the happy path. Its job is to explain the discrepancy in business terms and
tell the reviewer what to check.

It investigates with tools rather than being handed a summary:

```
read_document("Amount Due")     → "Amount Due 86,480.00 THB"
check_arithmetic(64000, 4480, 86480)
                                → computed 68,480.00, off by 18,000.00
lookup_vendor(tax_id="0105561071237")
                                → V-1002 Northwind Logistics, active
```

**It cannot change a value.** It may only propose corrections, and every proposal
is re-grounded against the page before a reviewer sees it. Ungrounded suggestions
are shown as "not on page" with no Apply button.

### Human-in-the-loop placement

Straight-through posting requires **all** of:

- no blocking rule failure
- confidence ≥ threshold (default 90%)
- every required field present
- every critical field grounded in the page text

Anything less queues for a person. The reviewer sees the page image beside
editable fields with confidence bars, the evidence each value came from, the
failed checks, and the agent's assessment.

Crucially, **the reviewer is not a rubber stamp**. Approving a document whose
blocking rule still fails returns a 409:

```
HTTP 409 — Cannot post: Totals do not add up: 64,000.00 + 4,480.00 = 68,480.00,
but the document states 86,480.00 (difference 18,000.00).
Correct these values or reject the document.
```

Correct the value and the rules re-run; only if they now pass does it post. This
keeps the human as a decision-maker rather than a bypass.

Every edit is recorded against document type and vendor, and aggregated into
correction hotspots — which is what tells you where accuracy work pays off next.

### Errors and edge cases

The rule throughout: **degrade, never fail.**

| Situation | Behaviour |
|---|---|
| OCR engine unavailable / model not pulled | Falls back through the engine chain; the run is marked `degraded`, which lowers every field's confidence |
| No LLM at all | Rule channel alone; the system still extracts, validates and routes |
| Model returns invalid JSON | Treated as "no extraction" → human review. Never a guessed structure |
| Model advertises tool support but its template fails | First-turn failure triggers a plain-prompt retry (a real Ollama case) |
| Endpoint rejects a parameter | Adapters negotiate: `max_tokens` → `max_completion_tokens` for gpt-5.1, dropping `temperature` for Claude Opus 5. Read the rejection rather than maintain a model list that goes stale |
| Reasoning model spends its budget thinking | Empty answer with `finish_reason=length` triggers one retry with a larger budget |
| Corrupt or unsupported file | `rejected` with a clear message, never an exception to the caller |
| Duplicate invoice | Blocked, with the earlier record id and posting date |
| Malicious URL (`169.254.169.254`, `localhost`) | Every redirect hop resolved and checked against private ranges before connecting |
| Wrong file extension | Type detected from magic bytes, not the filename |

Two edge cases worth calling out, because both were found by running the system
rather than reasoning about it:

- `parse_date("Net 45")` returned `2045-08-03` — dateutil invented the year from
  "45" and took the day and month from *today*. Now every ambiguous parse runs
  twice with different defaults; any component that changes was invented, and the
  parse is rejected. Silently guessing a date is precisely the behaviour the whole
  design forbids, and it took a ground-truth corpus to notice.
- The alias `PO` matched inside its own value `PO-2026-0412`, yielding
  `2026-0412`. Label matching now requires a real separator after the alias.

---

## 3. Getting It Done & Proving Value

### The smallest version worth shipping

**One document type, one vendor's layout, read-only.** Concretely: invoices from
the top five vendors by volume, extracted and validated, writing to a review
queue only — no posting.

That is deliberately small, and it is enough to answer the only question that
matters at week one: *on real documents from this business, how often is it
right, and how often is it confidently wrong?* Everything else — more document
types, more layouts, straight-through posting — is scaling a known quantity.

Ship in this order:

1. **Week 1 — measure.** Ingest, OCR, extract, and a ground-truth set of ~50 real
   documents with fields keyed in by the staff who do it today. No automation
   yet. Output: a number.
2. **Week 2 — validate.** Business rules and master-data cross-checks against a
   copy of the real vendor and PO tables. Still review-only. Now the queue shows
   the discrepancies the team already knows about, which is what earns trust.
3. **Week 3 — post, narrowly.** Straight-through posting enabled only for
   documents above the confidence threshold with all rules passing. Start the
   threshold high (95%) and lower it as the silent error rate stays at zero.

The system in this repository is roughly steps 1–3 with a synthetic corpus,
because a real corpus was not available.

### Measuring whether it works

**The metric that gates deployment is the silent error rate**, not accuracy.
Everything else is secondary:

| Metric | What it answers | Target |
|---|---|---|
| **Silent error rate** | Did anything wrong reach the ledger? | **0%** |
| Auto-post precision | Of what posted automatically, how much was right? | 100% |
| Defect recall | Do the rules catch known-bad documents? | 100% |
| Straight-through rate | How much work disappeared? | grows over time |
| Field accuracy | Where to spend improvement effort | diagnostic |
| Seconds per page | Cost and throughput | budget-dependent |

Note the asymmetry: silent error rate must be zero from day one, while
straight-through rate is allowed to start low and climb. A system that queues
everything is useless but harmless; a system that posts wrong values is worse
than the manual process it replaced.

**Time saved** is measured directly, not estimated: the corrections table records
which fields a reviewer actually had to touch. A document posted automatically
saved the full handling time; a document where the reviewer changed one field
saved most of it. That is a defensible number to put in front of a business
sponsor, unlike "we estimate 70% automation".

### How the evaluation works here

```bash
docuflow evaluate
```

Every sample carries ground truth **and** a list of the defects planted in it, so
the harness verifies two different things: that values are read correctly, and
that the rules catch what they should. It runs in an isolated database so
evaluation never touches the demo ledger and duplicate detection cannot fire on
its own output.

```bash
docuflow benchmark --combo "ocr=ollama:scb10x/typhoon-ocr1.5-3b,llm=typhoon" \
                   --combo "ocr=claude:claude-sonnet-5,llm=typhoon"
```

Engine choice is then an evidence-based decision on *your* documents, ranked by
fewest silent errors, then accuracy, then speed. This is how the gpt-5.1
hallucination was found — not by reading model cards.

**Measurement itself has to be audited.** The first benchmark run produced three
wrong conclusions, all found by inspecting the results rather than trusting them:

- The PDF-text-layer engine scored 95.7% — because it silently fell through to
  the cloud model on every scanned page and was credited with that model's
  output. A `strict` mode now disables fallback during benchmarking, and its
  honest score is 61.4%.
- Every engine appeared to miss the `duplicate_document` defect. It was not a
  detection failure: a document can only be a duplicate if its predecessor
  posted, and the more cautious engines had held that predecessor for review.
  The benchmark was penalising engines for being careful. Evaluation now seeds
  the prerequisite record.
- A 422 buried in the logs revealed a production bug: the agent's transcript grew
  past the provider's 50-message cap, so the reconciliation agent silently lost
  its investigation on long documents. The transcript is now trimmed at a turn
  boundary, so no tool result is ever orphaned from its call.

A benchmark that is never questioned is just a more confident guess.

### Improving before scaling

The correction hotspots table is the improvement backlog, ordered by how often
reviewers actually fix each field. In priority order:

1. **Schema aliases** — the cheapest fix. A field that is always corrected usually
   has a label the schema does not know. One line, no retraining.
2. **Normalisation** — new date or currency formats.
3. **Rules** — a discrepancy reviewers keep resolving the same way is a rule
   waiting to be written.
4. **Thresholds** — as the silent error rate stays at zero, lower
   `AUTO_POST_MIN_CONFIDENCE` and watch it. This is the main lever for
   straight-through rate, and it is a config change with an immediate measurement.
5. **Engine choice** — re-run the benchmark when a new model appears.
6. **Fine-tuning** — last, and only when the corrections data shows a stable
   repeating error the prompt cannot fix.

Business policy is configuration, not code — `PO_REQUIRED_ABOVE`,
`EXPECTED_VAT_RATES`, `CONTRACT_EXPIRY_WARNING_DAYS` — so finance can change the
rules of engagement without a deployment.

### What I would do differently with a real corpus

The synthetic corpus is honest about layout variety and scan damage, but it is
ten documents authored by the same generator. With real documents I would expect
to find:

- **more document types than anticipated** — credit notes, statements, delivery
  orders arriving in the same inbox
- **long-tail layouts** — the vendor whose invoice is a photograph of a
  handwritten form
- **master data that disagrees with reality** — the most common cause of a
  "wrong" extraction turning out to be right

That last one is why the vendor match reports its method and score rather than a
boolean. When the system says "no such vendor", the answer is often that the
master file is stale — and the reviewer needs to see that distinction to make the
right call.
