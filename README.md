# DocuFlow

**Unstructured documents in, validated records out.**

Drop in an invoice, a contract or a form — a PDF, a phone photo, or a URL to any
document on the internet — and DocuFlow reads it, extracts the fields the business
needs, cross-checks them against business rules and existing records, and either
posts a record downstream or hands it to a human with the discrepancy already
explained.

Built on **Typhoon OCR 1.5** with a pluggable model layer (Ollama, Typhoon Cloud,
OpenAI, Claude, MLX). Runs fully offline with no API key.

---

> **เอกสารภาษาไทย:** [ตอบคำถาม 3 ข้อ](docs/APPROACH.th.md) · [สถาปัตยกรรมและ workflow แบบละเอียด](docs/ARCHITECTURE.th.md) · [กฎธุรกิจทั้ง 26 ข้อ](docs/RULES.th.md)

## The constraint that shaped this

> *Documents are varied and not perfectly structured. Accuracy matters — wrong
> values flowing downstream are costly.*

A pipeline that is right 95% of the time and **silently** wrong 5% of the time is
worse than useless in accounts payable: the 5% becomes wrong payments. So the
headline metric here is not field accuracy, it is the **silent error rate** —
documents that posted automatically while carrying a wrong critical value.

Five mechanisms keep that number down:

1. **A clean page before the model reads it** — skew, blur, contrast and lighting
   are measured, and only the corrections a measurement asks for are applied.
   Each one is then re-measured and thrown away if the number it exists to
   improve did not improve, so a clean digital PDF passes through untouched and
   a phone photo arrives at the OCR model straightened and relit.
2. **Two independent extraction channels** — two language models read the same
   page, one framed field-first and one label-first, neither seeing the other's
   answers. Agreement scores 0.97; disagreement drops to 0.45 and routes to a
   human. The deterministic label matcher takes the second slot only when no
   second model is reachable, so the system still runs with no credentials.
3. **Evidence grounding** — every extracted value must come with a verbatim
   quote, verified to (a) exist on the page and (b) state the reported value.
   That second check is type-aware: a page saying `11/02/2019` does state the
   date `2019-02-11`, and `Subtotal 145.00` does state `145`. Ungrounded
   critical fields can never auto-post.
4. **Deterministic recomputation** — arithmetic is recomputed, never trusted from
   the model, and against a tolerance that is *rounding only*. The 2% allowance
   for an invoice differing from its purchase order is a commercial one and does
   not apply to a document agreeing with itself; lending it there let an
   810-baht misread post as "totals add up" on a 200,000-baht invoice.
5. **Conservative routing** — straight-through posting requires no blocking rule
   failure, confidence above threshold, all required fields present, and all
   critical fields grounded.

And when something does fail, the finding says **which value is wrong and what it
should be**. The amounts on an invoice are tied together by three relations — the
lines sum to the subtotal, the tax is a rate applied to it, the subtotal and tax
make the total — so a single misread value breaks every relation it appears in
and leaves the others standing. Where exactly one field explains the failure it
is named and the correction offered; where two would, nothing is offered, because
a wrong number wearing a suggestion's confidence is worse than no suggestion.

- [docs/APPROACH.md](docs/APPROACH.md) — the reasoning: model and technique
  choices, where agents belong versus deterministic steps, and how value is
  proven and improved before scaling.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the mechanics: pipeline stages,
  confidence model, plugin system, data model.
- [docs/architecture.drawio.xml](docs/architecture.drawio.xml) — three diagrams,
  openable at [diagrams.net](https://app.diagrams.net): where a model is called
  and what it is allowed to decide, the pipeline end to end with the human loop,
  and the agents with their tools.
- [docs/EVALUATION.md](docs/EVALUATION.md) — the numbers, per document, with
  what was read wrong and why.

---

## Quick start

Nothing to install: [`notebooks/docuflow_walkthrough.ipynb`](notebooks/docuflow_walkthrough.ipynb)
runs the whole pipeline in Colab against Typhoon Cloud and Claude, one stage at a
time, printing what each one produced — the page as it arrived and as
preprocessing handed it on, the OCR markdown, both independent readings, which
fields they agreed on, what the rules made of it, which value the arithmetic
blames when something fails, and why it was posted or held. It needs two API keys
and no GPU. It defaults to a phone photograph rather than a clean PDF, because a
clean PDF asks nothing of the system.

To run it locally instead:

```bash
git clone https://github.com/tententgc/mirtphol_doc_extraction.git
cd mirtphol_doc_extraction
uv venv --python 3.12 && uv pip install -e ".[dev]"
cp .env.example .env
```

Pick **one** OCR engine:

```bash
# A. Fully local via Ollama (recommended, no API key)
ollama pull scb10x/typhoon-ocr1.5-3b && ollama pull qwen3:4b

# B. Fully local via MLX (Apple Silicon)
uv pip install -e ".[local]"

# C. Typhoon Cloud (fastest) — put TYPHOON_API_KEY in .env
```

Then start the platform:

```bash
.venv/bin/docuflow serve
```

Open **http://127.0.0.1:8000**. The database, master data and the demo corpus are
created on first run.

To work against the committed corpus and its master data instead:

```bash
.venv/bin/docuflow reset --confirm
.venv/bin/docuflow master load real_samples/master
.venv/bin/docuflow rules load real_samples/rules.json
.venv/bin/docuflow process real_samples/3_invoice_photo_on_a_desk.jpg
```

---

## Using it

### The web platform

| Tab | What it does |
|---|---|
| **Process** | Drag in a file, paste a URL, or click a bundled sample. Watch the pipeline run. |
| **Review queue** | Everything held back, with the page image beside editable fields, confidence bars and the evidence each value came from. |
| **Records** | The downstream system of record — what actually got posted, and by whom. |
| **Insights** | Straight-through rate, average confidence, master data, and which fields reviewers correct most. |

The **OCR** and **Agent** dropdowns in the header switch models live, without a
restart.

### Command line

```bash
docuflow serve                        # the web platform
docuflow process invoice.pdf          # one file
docuflow process https://…/invoice.pdf  # straight from a URL
docuflow demo --reset                 # process the whole demo corpus
docuflow records                      # what has been posted
docuflow status                       # which engines are configured
docuflow models                       # every selectable model and its status

# master data
docuflow master load real_samples/master            # a whole directory, in order
docuflow master import vendors.csv --type vendors   # one file
docuflow master template --type purchase_orders     # the columns it expects
docuflow master show                  # what the register holds
docuflow master clear                 # empty it

# rules you write
docuflow rules load real_samples/rules.json   # load a reviewed rule set
docuflow rules list                   # what is active, and what each one reads as
docuflow rules clear                  # remove them (the 26 built-ins stay)

# starting over, and measuring
docuflow reset --confirm              # empty documents, ledger and master data
docuflow reset --confirm --keep-master  # keep the register, clear the rest
docuflow real-samples --out real_samples  # regenerate the working corpus
docuflow evaluate --out report.json   # accuracy, coverage and time saved
docuflow benchmark                    # compare models on the same corpus
```

---

## Choosing models

`docuflow models` lists everything installed and usable right now:

```
OCR ENGINES
 * typhoon_api:typhoon-ocr-v1.5              ready  [cloud, fastest, document OCR]
 * ollama:scb10x/typhoon-ocr1.5-3b           ready  [local, no key, document OCR]
 * local:mlx-community/typhoon-ocr1.5-2b-8bit ready [local, no key, document OCR]
   ollama:qwen2.5vl:3b                       ready  [local, no key]
   pdftext:-                                 ready  [instant, digital PDFs only]
   openai:gpt-4o                             unavailable (OPENAI_API_KEY is not set)
```

Measured by `docuflow benchmark` on five corpus documents (three digital PDFs,
two degraded scans), each engine run in strict mode so no engine is credited
with a fallback's output:

| Engine | Field accuracy | Silent errors | Defects caught | s/page | Key |
|---|---|---|---|---|---|
| **Typhoon Cloud `typhoon-ocr-v1.5`** | **94.3%** | 0% | 100% | 19.1 | yes |
| MLX `typhoon-ocr1.5-2b-8bit` | 92.9% | 0% | 100% | 37.0 | no |
| Claude `claude-sonnet-5` | 91.4% | 0% | 100% | 30.2 | yes |
| OpenAI `gpt-5.1` | 85.7% | 0% | 100% | 29.0 | yes |
| Tesseract | 78.6% | 0% | 100% | 27.7 | no |
| PDF text layer | 61.4% | 0% | 50% | 17.2 | no |

Read the **silent-error column first**: it is 0% for every engine, including the
weakest. Engine quality determines *how much work gets automated*, not *whether
wrong values escape* — that is what the grounding and rule layers are for.

The PDF text layer reads digital PDFs perfectly and scans not at all, which is
why it scores 61% and misses half the planted defects; in normal (non-strict)
operation it falls through to a real engine on those pages. Ollama's
`typhoon-ocr1.5-3b` measured 82.9% at 75 s/page in an earlier run — slower *and*
less accurate than the 2B MLX build, so prefer MLX when running locally on a Mac.

> **Prefer a document-OCR model over a general vision model.** On a degraded Thai
> scan in the demo corpus, `gpt-5.1` returned a company name and a full street
> address that are **not on the page** — plausible Thai text, entirely invented.
> `claude-sonnet-5` and the Typhoon OCR 1.5 models read the same page correctly.
>
> Evidence grounding catches this — an invented value cannot be quoted from the
> page, so it never auto-posts — but the right defence is to start with an engine
> trained on documents. Run `docuflow benchmark` on your own corpus before
> committing to an engine.

Engines are tried in priority order, so a missing model or expired key degrades
the run rather than breaking it — and any document read by a fallback engine is
flagged `degraded`, which lowers every field's confidence.

### Adding a provider

Providers are plugins. One registration adds an engine to the factory, the UI
picker, the CLI and the benchmark:

```python
register_ocr(OcrPlugin(
    key="my_engine",
    label="My Engine",
    build=lambda settings, model: MyOcr(model or settings.my_model),
    default_model=lambda settings: settings.my_model,
    fallback_priority=35,
))
```

---

## Configuration

Everything lives in `.env` — see [.env.example](.env.example). **Secrets go only
in `.env`, never in source.**

```bash
OCR_PROVIDER=ollama            # ollama | local | typhoon_api | openai | claude | pdftext | tesseract | auto
LLM_PROVIDER=typhoon           # auto | ollama | typhoon | openai | claude | local | rules
LLM_VERIFY_PROVIDER=openai     # the second reading — a *different* backend, or its agreement means nothing
OCR_PREPROCESS=true            # straighten, relight and sharpen before OCR

TYPHOON_API_KEY=               # opentyphoon.ai
OPENAI_API_KEY=                # platform.openai.com
ANTHROPIC_API_KEY=             # console.anthropic.com

AUTO_POST_MIN_CONFIDENCE=0.90  # bar for straight-through posting
AMOUNT_TOLERANCE=0.02          # how far an invoice may differ from its purchase order
ARITHMETIC_TOLERANCE=0.0001    # how far a document may differ from itself: rounding only
```

Those last two answer different questions and must not share a number. An invoice
may legitimately differ from the purchase order it quotes by 2% — freight, a part
shipment, a price that moved between order and delivery. It may not differ from
its own subtotal plus its own tax by anything but rounding.

Business policy is configuration, not code:

```bash
PO_REQUIRED_ABOVE=50000        # invoices above this must quote a PO
EXPECTED_VAT_RATES=0,7         # Thailand is 7%; set yours
CONTRACT_EXPIRY_WARNING_DAYS=90
```

`LLM_PROVIDER=rules` disables the model entirely and runs deterministic
extraction only — no keys, no downloads. That is what CI uses.

---

## Before the model reads it

Documents arrive as photographs taken under one lamp, faxes of faxes, scans fed
in crooked. Four things are measured on every page — how far it is rotated, how
sharp its edges are, how far the ink sits from the paper, and how much the
paper's own brightness drifts across the sheet — and only the corrections a
measurement asks for are applied.

Then each correction is **re-measured and thrown away if it did not help**. That
one rule removes the whole class of failure where a heuristic misfires on an
unusual page and quietly degrades it. Sharpening a crisp digital PDF rings the
glyph edges and makes OCR worse, so all thirteen documents in the demo corpus
pass through untouched and no second file is written for them.

Measured on the working corpus:

| Document | Before | After |
|---|---|---|
| `3_invoice_photo_on_a_desk` | 2.8° skew, lighting drift 0.390 | 0.0°, 0.008 |
| `18_invoice_unlabelled_amounts` | 2.9° skew, lighting drift 0.406 | 0.0°, 0.002 |
| `13_contract_already_expired` | ink-to-paper 0.057 | 0.142 |
| `16_invoice_line_items_do_not_sum` | 1.9° skew, ink-to-paper 0.059 | 0.0°, 0.192 |
| the three clean scans | — | untouched |

Two things are reported rather than corrected: a page tilted further than a
scanner could be, and a page lying on its side. Telling 90° from 270° needs to
know which way the glyphs face, and guessing wrong leaves the model reading
upside down. On the thermal receipt the deskew was attempted, measured as no
improvement and reverted — ten lines of text on 80mm paper is not enough of a
projection profile to trust, and the trace says so rather than rotating it on a
guess.

The prepared image is written **beside** the rendered page, never over it. OCR
reads the clean copy; a reviewer confirming a value still sees what arrived.

---

## What it checks

Twenty-six rules ship with the system, fifteen of them blocking. The ones that
stop a document:

| Rule | Catches |
|---|---|
| `invoice_totals_consistent` | subtotal + tax ≠ stated total |
| `tax_rate_consistent` | tax amount doesn't match the stated rate |
| `vendor_known` | vendor not in the master file |
| `vendor_active` | vendor on hold or blocked |
| `bank_account_matches_vendor` | payment redirected to an unregistered account |
| `po_exists` / `po_vendor_matches` | three-way match failure |
| `po_amount_within_tolerance` | invoice overruns the purchase order |
| `duplicate_document` | invoice already posted |
| `tax_id_checksum` | Thai 13-digit check digit fails |
| `critical_fields_grounded` | a money or identity value not traceable to the page |
| `contract_dates_ordered` | expiry before commencement |

### The three tiers, and what each one actually does

Three severities are only worth having if they behave differently:

| Tier | What happens | Who acts |
|---|---|---|
| **blocking** | The document stops. Nothing posts past a failing blocking rule — not even on approval. | A person corrects the values and re-checks, or rejects it. |
| **warning** | Recorded on the document, shown to the reviewer, carried into the posted record. **Does not stop it.** | Nobody, unless they want to. |
| **info** | Recorded. Notes a mode or a decision rather than a problem. | Nobody. |

That middle row is load-bearing and used not to be true. A warning previously
cut confidence by 8% each, so three advisory findings took a perfectly-read
document from 95% to 72% — under the threshold — and sent it to someone with
nothing to correct. **A warning that holds documents is a blocking rule that has
not admitted it**, and it is how a human-in-the-loop system ends up with a human
in every loop.

So confidence now answers one question — *how sure are we the values were read
correctly* — and routing answers the other. A blocked vendor, an expired
purchase order, a duplicate submission and an unusual VAT rate no longer move
the score at all: every one of them was read perfectly, and calling them
doubtful reading leaves a reviewer unable to tell a document we could not read
from one we read fine and may not pay.

What does move it is a failure saying a *character* was misread — the arithmetic
disagreeing, a check digit failing, a date next decade — because the rest of the
page came off the same engine under the same conditions. Weighted by the rule's
own severity: a blocking one takes a fifth, a warning a twentieth. A page read
at 95% still clears an 84% bar with two warnings against it, and is held once
three of them agree something is off. One advisory finding is noise; three are a
pattern.

### And when one fails, what to do about it

A failing check that only says *something is wrong* leaves the reviewer to work
out which of the numbers to trust, re-read the page and retype it. Where the
document's own arithmetic can answer that, the finding carries the answer:

```
[BLOCKING] invoice_totals_consistent
   Totals do not add up: 42,000.00 + 2,940.00 = 44,940.00, but the document
   states 49,440.00 (difference 4,500.00). the total is the only value that
   explains it — the line items add up to the subtotal and the tax matches the
   stated rate, so those are right. the subtotal plus the tax gives 44,940.00.
   Looking at the page: two digits are transposed: '94' was read where '49'
   balances

   suggested fix   total_amount   49,440.00 → 44,940.00   (92%)   [Apply]
```

The reasoning is worth stating because it is what makes the suggestion safe. The
amounts are tied together by three relations — the lines sum to the subtotal, the
tax is a rate applied to it, the subtotal and tax make the total. A single
misread value breaks every relation it appears in and leaves the others standing,
so the relations that still hold identify the one that broke them. **Where two
fields would explain the failure equally well, nothing is suggested at all** —
that is a page a person has to look at, and naming one at random would put a
wrong number in front of them wearing a suggestion's confidence.

Suggestions decide nothing. A reviewer accepts one, and the full rule set runs
again over what they accepted, exactly as if they had typed it. A changed bank
account is offered at 35% with *"confirm the change with the vendor by a channel
you already had"*, because a box that trains reviewers to click through
payment-redirection fraud is worse than no box.

### Rules you write yourself

Describe a check in plain language and it is compiled once, in front of you, into
one of two things.

**A condition** — the default, and the right answer whenever it fits. `total_amount
gt 100000`, `purchase_order_ref present`. Evaluated after that by plain code: free,
repeatable, and it cannot be talked into a different answer by a document.

**An agent** — for the checks a comparison cannot express, because they need a
lookup and then a judgement about what came back. The rule carries a plan, written
and approved when the rule is saved, and an agent works through it with tools
(`lookup_purchase_order`, `check_arithmetic`, `read_document`, …) on each document
the rule's guard admits.

What keeps the second one from being "ask a model whether the invoice is fine":

- the plan is fixed at authoring time, so every document is checked the same way;
- the guard stays deterministic, so a model only ever sees the documents the rule
  is about — and that is also the cost control;
- a verdict it cannot reach **holds** the document. Unreachable model, unparseable
  reply, a lookup that found nothing: all of them queue it for a person, and none
  of them ever block, because a bad afternoon for a backend should not turn into a
  refused invoice;
- severity, id and message come from the stored rule. The reply contributes a
  verdict and prose, and nothing else.

The whole trace — the plan, which step found what, which tools were called — is on
the finding, so a reviewer can see how the answer was reached rather than being
handed a confident sentence.

A rule set can also be kept in a file and loaded, which is how the seven that
ship with the working corpus are stored. The sentence someone typed sits next to
what it compiled to, so the compilation is reviewable in a diff rather than being
a model asked the same question again on every document:

```bash
docuflow rules load real_samples/rules.json
docuflow rules list
```

---

## The corpora

**[`real_samples/`](real_samples/)** is the working corpus: eighteen documents
that arrive the way documents actually arrive. Ten digital PDFs, three scans,
three phone photographs and two faded photocopies — the arrival is part of the
test, so document 3 is a photograph and only a photograph.

It ships with the master data it reconciles against, as the three CSVs an ERP
would export, and with the seven checks a finance team would add on top of the
built-in twenty-six.

```bash
docuflow reset --confirm                     # a known-empty database
docuflow master load real_samples/master     # vendors, then POs and contracts
docuflow rules load real_samples/rules.json  # 5 expression rules, 2 agent rules
docuflow process real_samples/1_invoice_clean_matches_po.pdf
```

Between them the eighteen cover a transposed total, a miscalculated VAT, a
changed bank account, an unregistered vendor, a blocked one, an overrun against
the purchase order, a currency that disagrees with it, a missing line item,
Buddhist-era dates, a duplicate submission, an expiring contract, an expired one,
an invoice with no labels at all, and a thermal receipt photographed on a desk.
[`real_samples/README.md`](real_samples/README.md) lists what each is for and
which rules it should trip.

Regenerate it after editing `src/docuflow/samples/real_world.py`:

```bash
docuflow real-samples --out real_samples
```

**The demo corpus** is smaller, older and generated at runtime rather than
committed. It exists so `docuflow demo` works on a fresh clone with nothing
loaded, and it is what `docuflow evaluate` and `docuflow benchmark` measure
against.

```bash
docuflow demo --reset
```

---

## Accuracy

```bash
docuflow evaluate            # accuracy, coverage and time saved
docuflow benchmark           # the same corpus across several model combinations
```

Full results, per document, with what was read wrong and why:
**[docs/EVALUATION.md](docs/EVALUATION.md)**.

The headline, measured on the eighteen-document working corpus — Typhoon OCR 1.5
on-device, Typhoon Cloud reading it, GPT-5.1 reading it again:

| Metric | Result |
|---|---|
| **Silent error rate** | **0%** — nothing wrong reached the ledger |
| Planted defects caught | **12/13** (92%) |
| Document type accuracy | **18/18** (100%) |
| Field accuracy (exact) | **191/216** (88.4%) |
| Straight-through rate | 4/18 — the other 14 were held, each for a stated reason |
| Time saved (modelled) | 114 of 135 manual minutes (84%) |

Twenty-two per cent straight-through looks low until you read the corpus:
thirteen of the eighteen documents have a defect planted in them on purpose, and
the four that posted are four that should have.

Read the confidence beside the decision, not instead of it. Six held documents
sit at 94–95%: every one was read correctly and stopped by a closed purchase
order, a changed bank account, a duplicate or a currency mismatch. The two at
60–62% are the two the system genuinely could not read. Those need different
things from a reviewer, and a single number that conflated them told them
neither.

The silent error rate is the number that matters. Field accuracy in the low
nineties with a 0% silent error rate is deployable; 99% accuracy with silent
errors is not, because you cannot tell which payments are wrong.

Because each sample carries ground truth *and* a list of the defects planted in
it, the harness checks both that values are read correctly and that the rules
catch what they should — a corpus that only measures reading would score well on
a system that validates nothing.

---

## Development

```bash
.venv/bin/python -m pytest tests/ -v
.venv/bin/python -m pytest tests/ --cov=docuflow --cov-report=term-missing
```

745 tests, 82% coverage. The suite runs with no API key and no model download: it uses the PDF
text layer for OCR and the deterministic extractor for fields, so the full
pipeline — ingest, render, preprocess, OCR, classify, extract, validate,
diagnose, route, post, review — is exercised in about a minute.

Credentials *and* business policy are pinned in the fixtures. That second half
matters more than it sounds: a developer whose `.env` names a verification
backend, or loosens a tolerance, or turns a rule off while exploring, would
otherwise get a green suite that is a statement about their machine rather than
about the code. That has now caught six settings the hard way, each of which is
named in `tests/conftest.py` with the failure it caused.

```
src/docuflow/
├── agents/       classifier, extractors, grounding, merge, tools, agent loop
├── evaluation/   accuracy harness, benchmark, reporting
├── ingest/       URL fetch (SSRF-guarded), content-addressed blob store
├── llm/          plugin registry + Typhoon / Ollama / OpenAI / Claude / MLX
├── masterdata/   vendors, purchase orders, contracts
├── normalize/    dates, amounts, identifiers, text
├── ocr/          plugin registry, engines, rendering, preprocessing, quality
├── rules/        the business rule engine, and the repair diagnosis
├── samples/      the demo and working corpus generators
├── schemas/      per-document-type field definitions
├── store/        documents, records, corrections
└── web/          the single-page UI
```

## Security

- URL ingestion resolves and checks **every redirect hop** against private,
  loopback and link-local ranges, so a crafted URL cannot reach internal services
  or a cloud metadata endpoint.
- Uploads are typed from their magic bytes, not their filename; path separators
  in remote filenames are stripped.
- Page images are only served from inside the configured page directory.
- Secrets are read from `.env` only. No credential has a default in source.

## Licence

MIT
