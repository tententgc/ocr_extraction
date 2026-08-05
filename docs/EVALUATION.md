# Evaluation

What the system does on thirty documents that arrive the way documents actually
arrive — and, more usefully, what it got wrong and why.

**Thai version: [EVALUATION.th.md](EVALUATION.th.md).**

| | |
|---|---|
| Raw report | [`eval/real-samples.json`](eval/real-samples.json) — per document, per field |
| Console output | [`eval/real-samples.txt`](eval/real-samples.txt) — exactly what the run printed |
| Run date | 2026-08-06 |

Reproduce it:

```bash
docuflow evaluate --corpus real_samples --out report.json
```

That command loads the corpus's own master data and rules, runs every document,
and prints the report below. It did not exist until this run: the command the
previous version of this document told you to use scored a **different corpus**,
and the script behind the published numbers was never committed. A figure nobody
can reproduce is a figure nobody should believe.

---

## 1. The headline

| Question | Answer |
|---|---|
| **Did anything wrong reach the ledger?** | **Yes — 2 of 30 (6.7%).** Both Thai, both the same one-stroke character confusion in a vendor name. |
| Did the gate open correctly? | **93.3%** — 28 of 30 went the way the corpus said |
| Were defects caught? | **15 of 15 planted defects (100%)** |
| Were values read correctly? | **90.0%** exact, precision 95%, recall 90% |
| Was the page read? | **86.8%** characters on the 15 pages an engine had to recognise |
| Is the record usable downstream? | **80%** complete, valid and self-consistent |
| What did it cost? | **39.4s** per document mean, 70.4s p90 |

**The silent error rate is the number that matters and it is not zero.** Two
documents posted automatically while carrying a wrong critical value. Section 6
is about them, because a system that is 90% accurate and silently wrong 6.7% of
the time is a system whose output nobody can trust without checking — which is
the work it was supposed to remove.

---

## 2. How the evaluation works

### 2.1 The corpus states its answer before the system is asked

| | |
|---|---|
| Corpus | [`real_samples/`](../real_samples/) — 30 documents |
| Should post (`_True`) | 16 |
| Should be held (`_False`) | 14 |
| Arrival | 16 digital PDFs, 8 scans, 4 phone photographs, 2 faded photocopies |
| Language | 22 English, 8 Thai |
| Types | 23 invoices, 3 receipts, 3 contracts, 1 form |
| Master data | 13 vendors, 12 purchase orders, 6 contracts, loaded from CSV |
| Rules | 26 built-in + 7 written for this corpus (5 expression, 2 agent) |
| OCR | Typhoon OCR 1.5 (3B) on-device via Ollama, after a PDF text-layer fast path |
| Reading 1 | Typhoon `typhoon-v2.5-30b-a3b-instruct`, field-first |
| Reading 2 | Claude `claude-opus-5`, label-first — a different backend on purpose |
| Auto-post threshold | 84% confidence |

The expected outcome is in the filename. `_True` means nothing is wrong and
holding it is a false alarm; `_False` means something is wrong and posting it is
a silent error. **The two are not symmetric and are never summed:** a missed
defect is a wrong payment, a false alarm is a wasted minute.

The two readings are on **different backends** deliberately. Two passes of one
model over one page repeat the same mistake, so their agreement would mean much
less than it appears to.

Documents run in numeric order, because document 15 is a duplicate of document 1
and cannot be detected until document 1 has posted.

### 2.2 What is measured, and why each is needed

| Measurement | Question it answers | Why the others cannot |
|---|---|---|
| **Precision / recall** | Of what was answered, how much was right; of what was there, how much was found | Accuracy alone cannot see a rare class never recognised, nor a value invented for a field the document does not have |
| **OCR accuracy** | Was the page read at all | A field wrong because the characters never arrived is not an extraction problem, and fixing extraction will not help |
| **Data quality** | Is the record fit to hand to another system | The only one that still works in production, where there is no answer sheet |
| **Gatekeeping** | Did the document go the right way, and did each rule fire for the reason it exists | Decision counts cannot tell a working rule from a silent one that another rule happened to cover |
| **Silent error rate** | Did a wrong critical value post with nobody in the way | Everything above can look healthy while this is non-zero |
| **Latency & cost** | What it took | An optimisation is only worth having if the numbers above it did not move |

### 2.3 Three things the harness refuses to do quietly

Each of these moves a headline number in the flattering direction, and each was
found by building this harness rather than by reasoning about it.

**It will not grade against rules that were never asked.** `Settings` fills
anything it was not given from a `.env`. This checkout has seven rules disabled,
and two of them — `bank_account_matches_vendor`, `po_amount_within_tolerance` —
belong to documents whose whole purpose is to be held. Left alone the run reports
**two missed defects**, the metric that means wrong payments, for a reason that
lives in an untracked file. The run switches them back on and says so in the
report.

**It will not score a transcript that was copied rather than read.** The
text-layer fast path returns the PDF's own text, and `reference_text` is that
same text from that same place, so character error rate is zero *by
construction*. Fifteen of thirty documents took that path.

| | Character accuracy | Value recall |
|---|---|---|
| Pages an engine recognised (15) | **86.8%** | **86.2%** |
| Pooled with the text layer (30) | 93.7% | 91.6% |

The recognised figure is the one quoted as OCR accuracy. The pooled one is
printed labelled, not hidden.

**It will not grade against values the page never prints.** Eleven of 298
gradeable values (3.7%) are claimed by ground truth and absent from the text the
renderer was handed:

| Field | N | Documents | Why |
|---|---|---|---|
| `purchase_order_ref` | 3 | 4, 16, 19 | the `dense` and `two_column` layouts draw no purchase-order line |
| `vendor_tax_id` | 2 | 6, 10 | `minimal` labels nothing — that is its stated purpose |
| `payment_terms` | 2 | 6, 10 | same layout |
| `payment_method` | 2 | 11, 24 | receipts do not print one |
| `due_date` | 2 | 7, 19 | derivable from the payment terms, never written down |

These report as missing on **every engine anyone tries**, so field recall below
is understated by roughly this much. They are listed and **not dropped** — a
harness that silently discards the ground truth it cannot verify prints a better
number every time the corpus gets worse.

Two more corpus notes, reported for the same reason:

- `18_invoice_unlabelled_amounts_True.jpg` has a 196-character text layer, so
  nothing can be proved about its ground truth either way.
- `3_invoice_photo_on_a_desk_False.jpg` must be held and names no rule. Its
  defect is real — it quotes `PO-2026-0545`, the only closed purchase order in
  the register — but because it is unstated, whichever rule catches it is scored
  as having fired unexpectedly and defect recall never counts it.

---

## 3. Accuracy, precision and recall

### 3.1 Document type — 100%

| Type | N | Predicted | Precision | Recall | F1 |
|---|---|---|---|---|---|
| invoice | 23 | 23 | 100% | 100% | 100% |
| receipt | 3 | 3 | 100% | 100% | 100% |
| contract | 3 | 3 | 100% | 100% | 100% |
| form | 1 | 1 | 100% | 100% | 100% |
| **Macro** | | | **100%** | **100%** | **100%** |

Macro averaging matters here even at 100%. Twenty-three of thirty documents are
invoices, so answering "invoice" every time scores **77% accuracy** while never
having recognised a receipt — and that happened on this project once, because a
Thai till prints `ใบกำกับภาษีอย่างย่อ`, an *abbreviated* tax invoice, which is a
receipt and contains the invoice keyword whole. Macro recall for `receipt` would
have been 0. Accuracy would have said 77%.

### 3.2 Extracted values

| | Count | |
|---|---|---|
| **Precision** | **95%** | of 341 values answered, 325 were right |
| **Recall** | **90%** | of 361 values present, 325 were found |
| **F1** | **93%** | |
| Wrong | 16 | answered, and not right |
| Missing | 20 | there, and not found — the safe way to be wrong |
| **Spurious** | **0** | not there, and answered anyway |

**Zero spurious values is the most reassuring number in this section**, and it is
invisible to accuracy computed over ground-truth fields — a value invented for a
field the document does not have is never compared against anything. The system
never made one up.

Weakest fields:

| Field | Precision | Recall | F1 | Wrong | Missing |
|---|---|---|---|---|---|
| `contract_value` | 67% | 67% | 67% | 1 | 0 |
| `party_a` | 67% | 67% | 67% | 1 | 0 |
| `contract_number` | 100% | 67% | 80% | 0 | 1 |
| `buyer_name` | 83% | 83% | 83% | 4 | 0 |
| `purchase_order_ref` | 100% | 75% | 86% | 0 | 3 |
| `vendor_tax_id` | 94% | 79% | 86% | 1 | 3 |

Contracts are the weak type — three documents, so one mistake is 33% — and
`purchase_order_ref` and `vendor_tax_id` are dominated by the undrawn values in
§2.3 rather than by reading failures.

---

## 4. Was the page read?

| | Recognised pages (15) | Pooled with text layer (30) |
|---|---|---|
| Character accuracy | **86.8%** | 93.7% |
| Character error rate | 13.20% | 6.3% |
| Value recall | **86.2%** (119/138) | 91.6% |

Worst-read documents:

| Document | Characters | Values | Lost |
|---|---|---|---|
| `11_receipt_photographed_True.jpg` | 55.3% | 86% | `payment_method` |
| `24_receipt_clean_thermal_True.pdf` | 55.4% | 86% | `payment_method` |
| `16_invoice_line_items_do_not_sum_False.jpg` | 64.0% | 40% | invoice number, both dates |
| `27_receipt_thai_fuel_station_True.jpg` | 69.4% | **100%** | — |

**The two receipts at ~55% are a caution about the metric, not the engine.** An
80mm receipt is a few hundred characters, so the denominator is tiny and the
OCR's tidier line breaking dominates the edit distance. Both still recovered 86%
of the values that matter and every amount on them is correct. The Thai fuel
receipt makes the point exactly: 69.4% characters, **100% of the values**.

That divergence is the reason both numbers are reported. CER describes the
transcript; value recall describes the digits that become a payment.

---

## 5. Data quality — usable without an answer sheet

The only section that keeps working in production.

| | | |
|---|---|---|
| Completeness | **88.8%** | 357 of 402 schema fields filled |
| Critical fields | **89.3%** | 176 of 197 |
| **Validity** | **100%** | 0 values in a shape their type does not promise |
| Grounded in the page | **98.9%** | 4 values with no span behind them |
| Self-consistent | **26/30** | documents whose arithmetic closes |
| **Usable as-is** | **80%** | 24 of 30 complete, valid and self-consistent |

**Validity at 100% is worth stating plainly.** Every date is ISO, every amount is
a number, every currency is a three-letter code, every percentage is in range.
Nothing reached the ledger in a shape another system has to guess about — which
is where the quiet failures live: a tax rate stored as `"7%"` reaches
`tax_rate_consistent`, which *skips* rather than fails, and the check stops
existing without a word.

An absent field is counted, not condemned. An invoice with no purchase-order
reference is a legitimate invoice; whether one is *required* is business policy
and configurable.

---

## 6. The gate, and the two silent errors

| | |
|---|---|
| Decision accuracy | **93.3%** — 28 of 30 |
| **Missed defects** | **0 of 14 (0%)** — nothing that should have been held posted |
| False alarms | 2 of 16 (12.5%) |
| Planted defects caught | **15 of 15 (100%)** |
| Rules exercised | 12 tested, 22 **untested** |
| Auto-posted | 14 of 30 (47%) |

False alarms: `18_invoice_unlabelled_amounts_True.jpg` and
`21_invoice_minimal_unlabelled_True.pdf` — both the `minimal` layout, which
labels nothing on purpose. Held for review, nothing wrong with either. Cheap
once; the number to watch if the layout becomes common.

Twenty-two rules are reported **untested**, not passing. Untested is the honest
word, and the list is the specification for the next batch of corpus documents.

### 6.1 The two documents that posted while wrong

Both are Thai, and both are **the same failure**:

| Document | Field | Expected | Read as | Confidence |
|---|---|---|---|---|
| `25_invoice_thai_arabic_digits_True.png` | `vendor_name` | บริษัท สยาม**ซ**ัพพลาย จำกัด | บริษัท สยาม**ช**ัพพลาย จำกัด | 0.843 |
| `30_invoice_thai_no_tax_id_True.png` | `vendor_name` | บริษัท **ซ**ันไรส์ **อิ**เล็คทริคอล จำกัด | บริษัท **ช**ันไรส์ **อี**เล็คทริคอล จำกัด | 0.840 |

**ซ** (U+0E0B) and **ช** (U+0E0A) differ by a single loop. **อิ** and **อี**
differ by one vowel sign. These are the classic Thai OCR confusion pairs.

Three things made this post rather than stop:

1. **No rule fired at all.** Not one of the 33.
2. **Confidence landed on the threshold** — 0.843 and 0.840 against a bar of
   0.840. A margin of 0.003 and 0.000.
3. **`vendor_known` passed.** Vendor resolution is fuzzy on purpose, because
   string equality would block every one-stroke Thai misread. It matched the
   right vendor — so the money would go to the registered bank account, not to
   the misread name.

So the *payment* is right and the *ledger record* is wrong. That is a data-quality
defect rather than a wrong payment, and the metric does not distinguish them —
correctly, because `vendor_name` is declared critical.

**This is a known limitation being measured for the first time.** Grounding
cannot catch an OCR misread: the span exists in the OCR output, and the OCR
output is what was wrong. Nor can two channels reading the same OCR text — both
agents saw `ชัพพลาย`. The documented defences are the arithmetic rules and the
legibility signal, and neither can see a misspelt name on a legible page.

**There is an unused signal sitting right there.** `vendor_known` resolved the
vendor by fuzzy match and recorded the score. The register holds the correct
legal name. A vendor resolved fuzzily whose extracted name is *not* the
register's name is direct evidence of a misread on a critical field — and today
that score is recorded and nothing acts on it. That is the first thing to fix,
and it is a finding this evaluation produced rather than confirmed.

### 6.2 Thai reads better and fails worse

| | Documents | Field accuracy | Silent errors |
|---|---|---|---|
| English | 22 | 88.2% | **0** |
| Thai | 8 | **94.9%** | **2** |

Thai field accuracy is *six points higher* than English, and Thai carries both
silent errors. There is no contradiction: the Thai failure mode is a
single-character confusion inside a name, which fuzzy vendor matching absorbs
and every other check ignores. Average accuracy and worst-case safety are
different questions, and this is what it looks like when they disagree.

---

## 7. Latency and cost

| | |
|---|---|
| Mean per document | **39.4s** |
| Median | 29.6s |
| p90 | 70.4s |
| Slowest | 83.0s |
| Wall clock, 30 documents | 1,186s (19.8 min) |

By stage:

| Stage | Total | Mean | Share |
|---|---|---|---|
| OCR | 532.4s | 35.5s | **45%** |
| Extract | 476.9s | 15.9s | **40%** |
| Reconcile | 108.3s | 7.7s | 9% |
| Validate | 50.5s | 1.9s | 4% |
| Classify | 6.8s | 0.4s | 1% |
| Preprocess | 6.8s | 0.2s | 1% |
| Render | 1.5s | 0.1s | 0% |

By how the document arrived — this is where the text-layer fast path shows:

| Arrival | N | Mean total | Mean OCR |
|---|---|---|---|
| **Digital PDF** | 16 | **24.3s** | **1.6s** |
| Phone photograph | 4 | 44.2s | 26.7s |
| Scan (PNG) | 8 | 57.1s | 36.9s |
| Faded photocopy | 2 | 82.0s | 52.9s |

A digital PDF costs **1.6s of OCR against 36.9s for a scan** — a 23× difference,
and the reason OCR is only 45% of total time rather than the overwhelming
majority. Everything above the fast path is unchanged: the same documents, the
same verdicts.

Agent activity:

| | |
|---|---|
| Agent rules run | 9 (5 pass, 3 fail, 1 unsure) |
| Could not decide | 11% — held for a person, never passed |
| Agent rule tool calls | 56 |
| Reconciler ran on | 14 of 30 documents, 166 tool calls |
| Repeated calls | 11 — asked something already answered |

The 11% undecided rate is a number to watch, not a fault: an agent that cannot
decide holds the document, which is the safe direction. The 11 repeated calls are
waste — each is a turn of a bounded budget spent learning nothing.

---

## 8. Time saved — modelled, not measured

The assumptions print with the number every time, because a modelled figure
presented like a measured one is how a report ends up claiming a saving nobody
can find afterwards.

| | Minutes | Per 1,000 documents |
|---|---|---|
| By hand | 225 | 16.7 days |
| Through here | 114 | 8.4 days |
| — of which rework | **90** | cleaning up the 2 posted errors |
| **Saved** | **111 (49%)** | |
| Machine time | 19.8 | nobody waits for it in a batch |

Assumptions, per document: 4.0 min keying · 3.5 min cross-checking · 1.5 min
reviewing a held document · **45 min cleaning up one silent error**.

**The two silent errors cost 90 modelled minutes — 80% of the saving.** The
arithmetic is deliberately allowed to conclude that the system made things worse,
and this is how close it gets: two documents out of thirty take the saving from
88% down to 49%. That is the whole argument for the silent error rate being the
headline metric rather than field accuracy.

---

## 9. What to do next, in order

1. **Use the vendor-match score.** A vendor resolved by fuzzy match whose
   extracted name differs from the register's is evidence of a misread on a
   critical field. The signal exists and nothing reads it. This is the fix for
   both silent errors.
2. **Fix the eleven undrawn ground-truth values** in `samples/real_world.py`, so
   field recall stops being understated by 3.7% forever.
3. **Declare document 3's rule** so the closed purchase order it quotes is scored
   as the caught defect it is.
4. **Write corpus documents for the 22 untested rules.** Untested is not passing.
5. **Look at the 0.003 confidence margin.** Both silent errors sat on the
   threshold. That is not a coincidence worth ignoring.

---

## Appendix — reproducing this

```bash
docuflow evaluate --corpus real_samples --out report.json
```

The harness switches back on any rule the corpus declares that the local `.env`
has disabled, and records which. To measure this installation exactly as
configured instead:

```bash
docuflow evaluate --corpus real_samples --respect-disabled
```

Tests covering the harness itself, none of which call a model:

| File | What it holds |
|---|---|
| `tests/test_corpus_evaluation.py` | the disabled-rule guard, the text-layer split, report shape |
| `tests/test_corpus_integrity.py` | ground truth against the text the renderer drew |
| `tests/test_classification_scores.py` | precision/recall, and the ignored-class failure |
| `tests/test_data_quality.py` | completeness, validity, consistency |
| `tests/test_thai_pipeline.py` | every Thai path, end to end |
