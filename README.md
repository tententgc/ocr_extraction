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

## The constraint that shaped this

> *Documents are varied and not perfectly structured. Accuracy matters — wrong
> values flowing downstream are costly.*

A pipeline that is right 95% of the time and **silently** wrong 5% of the time is
worse than useless in accounts payable: the 5% becomes wrong payments. So the
headline metric here is not field accuracy, it is the **silent error rate** —
documents that posted automatically while carrying a wrong critical value.

Four mechanisms keep that number down:

1. **Two independent extraction channels** — a deterministic label matcher and a
   language model read the same page. Agreement scores 0.97; disagreement drops
   to 0.45 and routes to a human.
2. **Evidence grounding** — every extracted value must come with a verbatim quote,
   and that quote is verified to (a) exist on the page and (b) actually contain
   the value. Ungrounded critical fields can never auto-post.
3. **Deterministic recomputation** — arithmetic is recomputed, never trusted from
   the model.
4. **Conservative routing** — straight-through posting requires no blocking rule
   failure, confidence above threshold, all required fields present, and all
   critical fields grounded.

- [docs/APPROACH.md](docs/APPROACH.md) — the reasoning: model and technique
  choices, where agents belong versus deterministic steps, and how value is
  proven and improved before scaling.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the mechanics: pipeline stages,
  confidence model, plugin system, data model.

---

## Quick start

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

Open **http://127.0.0.1:8000**. The database, master data and a corpus of ten
demo documents are created on first run.

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
docuflow models                       # every selectable model and its status
docuflow evaluate --out report.json   # accuracy against ground truth
docuflow benchmark                    # compare models on the same corpus
docuflow status                       # which engines are configured
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
LLM_PROVIDER=auto              # auto | ollama | typhoon | openai | claude | local | rules

TYPHOON_API_KEY=               # opentyphoon.ai
OPENAI_API_KEY=                # platform.openai.com
ANTHROPIC_API_KEY=             # console.anthropic.com

AUTO_POST_MIN_CONFIDENCE=0.90  # bar for straight-through posting
AMOUNT_TOLERANCE=0.02          # PO / arithmetic slack
```

Business policy is configuration, not code:

```bash
PO_REQUIRED_ABOVE=50000        # invoices above this must quote a PO
EXPECTED_VAT_RATES=0,7         # flag unusual rates
CONTRACT_EXPIRY_WARNING_DAYS=90
```

`LLM_PROVIDER=rules` disables the model entirely and runs deterministic
extraction only — no keys, no downloads. That is what CI uses.

---

## What it checks

Roughly twenty rules run against every document. The blocking ones:

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

Warnings (VAT rate unusual, contract expiring, low-confidence fields, degraded
OCR) inform the reviewer without blocking.

---

## The demo corpus

Ten documents with known ground truth, each demonstrating something specific:

| Document | Demonstrates |
|---|---|
| `inv-clean-classic` | Happy path — posts automatically |
| `inv-bad-math` | Stated total 86,480 vs computed 68,480 — blocked |
| `inv-thai-scan` | Thai scan, Buddhist-era dates, Thai numerals |
| `inv-unknown-vendor` | Unregistered vendor + unknown bank account (fraud pattern) |
| `inv-duplicate` | Same invoice sent twice |
| `inv-eur-over-po` | Foreign currency, PO overrun outside tolerance |
| `contract-facility` | Multi-page prose contract |
| `form-vendor-onboarding` | Hand-filled form, invalid tax-ID checksum |
| `receipt-fuel` | Low-contrast thermal receipt |
| `po-orient-paper` | Purchase order against an on-hold vendor |

Four are rendered as degraded scans — rotated, blurred, noisy, JPEG-compressed —
so the numbers are not flattered by clean digital PDFs.

```bash
docuflow demo --reset
```

---

## Accuracy

```bash
docuflow evaluate
```

Measured on the ten-document corpus with Typhoon OCR 1.5 running on-device:

| Metric | Result |
|---|---|
| **Silent error rate** | **0.0%** — nothing wrong reached the ledger |
| Planted defects caught | **8/8 (100%)** |
| Document type accuracy | **100%** |
| Field accuracy (exact) | **91.2%** (114/125) |
| Field accuracy (allowing a one-character OCR slip) | **96.8%** |
| Straight-through rate | 10% — the rest were correctly held |

The silent error rate is the number that matters. Field accuracy of 91% with a
0% silent error rate is deployable; 99% accuracy with silent errors is not,
because you cannot tell which payments are wrong.

Every wrong value in that run was a genuine OCR misread on a degraded scan, and
every one was caught — for example the bank account `5550001111` read as
`550001111`, blocked by `bank_account_matches_vendor` before it could redirect a
payment.

Because each sample carries ground truth *and* a list of the defects planted in
it, the harness verifies both that values are read correctly and that the rules
catch what they should.

```bash
docuflow benchmark   # the same corpus across several model combinations
```

---

## Development

```bash
.venv/bin/python -m pytest tests/ -v
.venv/bin/python -m pytest tests/ --cov=docuflow --cov-report=term-missing
```

250 tests, 82% coverage. The suite runs with no API key and no model download: it
uses the PDF text layer for OCR and the deterministic extractor for fields, so
the full pipeline — ingest, render, OCR, classify, extract, validate, route,
post, review — is exercised in about twenty seconds. Credentials are blanked in
the fixtures, so results do not depend on which keys happen to be in your `.env`.

```
src/docuflow/
├── agents/       classifier, extractors, grounding, merge, tools, agent loop
├── evaluation/   accuracy harness, benchmark, reporting
├── ingest/       URL fetch (SSRF-guarded), content-addressed blob store
├── llm/          plugin registry + Typhoon / Ollama / OpenAI / Claude / MLX
├── masterdata/   vendors, purchase orders, contracts
├── normalize/    dates, amounts, identifiers, text
├── ocr/          plugin registry, engines, rendering, quality scoring
├── rules/        the business rule engine
├── samples/      the demo corpus generator
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
