# Evaluation

What the system does on eighteen documents that arrive the way documents
actually arrive, and — more usefully — what it got wrong and why.

The raw output behind every number here is
[`eval/real-samples.json`](eval/real-samples.json) — per document, per field.

Run it yourself:

```bash
docuflow reset --confirm
docuflow master load real_samples/master
docuflow rules load real_samples/rules.json
docuflow evaluate --out report.json
```

---

## What is being measured, and why these three

The brief asks for accuracy, coverage and time saved. Two of those are
measurements and one is a model, and it matters which is which.

**Accuracy** is field-by-field against ground truth the corpus carries. It is
the number everyone asks for and the least interesting one here, because a
system can be 99% accurate and still be unusable.

**Silent error rate** is the number that matters, and it is not in the brief.
It counts documents that **posted automatically while carrying a wrong critical
value** — a wrong total, a wrong invoice number, a wrong bank account. A
pipeline that is right 95% of the time and silently wrong 5% of the time is
worse than useless in accounts payable, because the 5% becomes wrong payments
and nobody knows which ones. Field accuracy in the low nineties with a zero
silent error rate is deployable. Ninety-nine per cent with silent errors is not.

**Coverage** is the straight-through rate: how many documents needed nobody. A
high number here is only worth having alongside a zero above it; a system that
posts everything scores 100% coverage and is a liability.

**Defect recall** is what stops the corpus flattering the reader. Thirteen of
the eighteen documents have a specific defect planted in them and declare which
rule should catch it. A corpus that only measured reading would score well on a
system that validates nothing.

**Time saved** is modelled, not measured, and the assumptions print with the
number every time. See [the assumptions](#the-time-saved-model) below — they are
deliberately conservative and the arithmetic is allowed to conclude that the
system made things worse.

---

## The setup

| | |
|---|---|
| Corpus | [`real_samples/`](../real_samples/) — 18 documents |
| Arrival | 10 digital PDFs, 3 scans, 3 phone photographs, 2 faded photocopies |
| OCR | Typhoon OCR 1.5 (3B) on-device via Ollama |
| Reading 1 | Typhoon `typhoon-v2.5-30b-a3b-instruct`, field-first |
| Reading 2 | OpenAI `gpt-5.1`, label-first — a different backend on purpose |
| Rules | 26 built-in + 7 written for this corpus (5 expression, 2 agent) |
| Master data | 12 vendors, 10 purchase orders, 5 contracts, loaded from CSV |
| Auto-post threshold | 84% confidence |

The two readings are deliberately on different backends. Two passes of the same
model over the same page repeat the same mistake, so their agreement would mean
much less than it appears to.

Documents run in numeric order because document 15 is a duplicate of document 1
and can only be detected once document 1 has posted.

---

## Results

| Metric | Result |
|---|---|
| **Silent error rate** | **0%** — 0 of 4 auto-posted documents carried a wrong critical value |
| Planted defects caught | **12 / 13** (92%) |
| Document type accuracy | **18 / 18** (100%) |
| Field accuracy (exact) | **191 / 216** (88.4%) |
| Straight-through rate | 4 / 18 (22%) — the other 14 were held, each for a stated reason |
| Wall clock | 523s for 18 documents (29s each) |
| Time saved (modelled) | 114 of 135 manual minutes (84%) |

Twenty-two per cent straight-through looks low until you read the corpus:
thirteen of the eighteen documents have a defect planted in them on purpose. The
four that posted are four that should have. A corpus built to be caught is not a
corpus that measures throughput.

### Per document

`S` marks a silent error — auto-posted with a wrong critical value.

Read the confidence column against the decision column. That pairing is the
useful one, and it only became useful once confidence stopped answering two
questions at once: it now says *how well we read the page*, and the decision says
*whether it may post*. A document at 95% that is held was read perfectly and
stopped by a fact about the world. A document at 60% that is held is one we could
not read. Those need different things from a reviewer, and a single number that
conflated them told them neither.

| # | Document | Arrives as | Decision | Conf. | Fields | S | Defect | Held by |
|---|---|---|---|---|---|---|---|---|
| 1 | `invoice_clean_matches_po` | PDF | **auto-post** | 95% | 14/14 | | — | — |
| 2 | `invoice_total_transposed` | PDF | held | 76% | 14/14 | | 1/1 | the arithmetic |
| 3 | `invoice_photo_on_a_desk` | photo | held | 95% | 14/14 | | — | the PO is closed |
| 4 | `invoice_vat_miscalculated` | PDF | held | 75% | 11/14 | | 2/2 | VAT ≠ stated rate; vendor on hold |
| 5 | `invoice_bank_account_changed` | PDF | held | 95% | 13/13 | | 1/1 | not the vendor's account |
| 6 | `invoice_unknown_vendor` | scan | held | 83% | 10/12 | | 1/1 | vendor not registered |
| 7 | `invoice_from_a_blocked_vendor` | PDF | held | 94% | 11/13 | | 1/1 | vendor blocked |
| 8 | `invoice_over_the_purchase_order` | PDF | held | 95% | 13/14 | | 1/1 | 22% over the order |
| 9 | `invoice_thai_buddhist_dates` | scan | **auto-post** | 84% | 13/13 | | — | — |
| 10 | `invoice_no_po_above_threshold` | PDF | held | 88% | 10/13 | | see below | PO does not exist |
| 11 | `receipt_photographed` | photo | **auto-post** | 88% | 7/9 | | — | — |
| 12 | `contract_expiring_soon` | PDF | held | 82% | 8/8 | | 1/1 | expires in 56 days |
| 13 | `contract_already_expired` | photocopy | held | 68% | 5/8 | | 1/1 | expired |
| 14 | `form_vendor_onboarding` | scan | **auto-post** | 96% | 7/7 | | — | — |
| 15 | `invoice_duplicate_submission` | PDF | held | 95% | 14/14 | | 1/1 | already posted |
| 16 | `invoice_line_items_do_not_sum` | photocopy | held | 62% | 7/13 | | 1/1 | date misread; lines short |
| 17 | `invoice_currency_disagrees_with_po` | PDF | held | 95% | 12/13 | | 1/1 | USD against an SGD order |
| 18 | `invoice_unlabelled_amounts` | photo | held | 60% | 8/10 | | — | required fields missing |

Six documents sit at 94–95% and are still held. Every one of them was read
correctly and stopped by something that has nothing to do with reading: a closed
purchase order, a changed bank account, a duplicate, a currency mismatch. The two
at 60–62% are the two the system genuinely struggled to read — a faded photocopy
and an invoice with no field labels at all — and both are held for that.

### What posted, audited field by field

The straight-through set is where a silent error would live, so it is worth
listing in full rather than summarising:

| Document | Wrong | Missing | Critical wrong |
|---|---|---|---|
| `1_invoice_clean_matches_po` | — | — | none |
| `9_invoice_thai_buddhist_dates` | — | — | none |
| `11_receipt_photographed` | `payment_method` | `tax_rate` | **none** |
| `14_form_vendor_onboarding` | — | — | none |

Three of the four are perfect. The fourth is a thermal receipt photographed on a
desk: every amount, the merchant and the receipt number are right, and what it
got wrong is how the fuel was paid for. That is a real cost of letting warnings
pass — and the right trade, because the alternative was holding all four and
having a person confirm three documents that had nothing wrong with them.

### The one defect not caught as expected

Document 10 has no purchase order and a total above the threshold, so
`purchase_order_required` should have blocked it. It passed — because extraction
read the bank account `8890014455` into the `purchase_order_ref` field. Given a
purchase order reference, the rule correctly concluded one had been quoted.

`po_exists` then blocked the document anyway: *"Purchase order 8890014455 does
not exist in the system."* The outcome was right and the reasoning was right; the
rule that produced it was not the one predicted. Recorded as a miss because
scoring it as a hit would let a corpus grade itself on outcomes rather than on
mechanism — and the underlying fault is real: a bank account landed in a
reference field, and nothing noticed the shape was wrong.

### Which value is wrong

Five documents got a diagnosis. Two are worth reading closely.

**Document 2** — the lines and the VAT both say 44,940 and the total reads
49,440:

> **total_amount** 49,440.00 → **44,940.00** (92%)
> *the total is the only value that explains it — the tax matches the stated rate
> and the line items add up to the subtotal, so those are right. the subtotal plus
> the tax gives 44,940.00. Looking at the page: two adjacent digits are
> transposed: '94' was read where '49' balances*

**Document 4** is the more interesting one, because it disagreed with the corpus
author. The document was written to have a wrong *tax amount* — 4,590 where 7% of
76,500 is 5,355. The system blamed the **rate** instead:

> **tax_rate** 7.0 → **6.0** (92%)
> *the tax rate is the only value that explains it — the subtotal and tax make the
> total, so those are right. the tax as a share of the subtotal gives 6.00*

It is right and the author was wrong. The printed total is consistent with the
tax that was charged, so the value that disagrees with everything else is the
stated rate, not the amount. Both readings fix the document; only one of them is
what the relations actually say.

The other three: a currency taken from the purchase order (70%), and two changed
bank accounts offered at 35% with *"confirm the change with the vendor by a
channel you already had before accepting this"*.

### Preprocessing

Five documents needed work. The three clean scans and ten PDFs passed through
untouched and had no second file written.

| Document | Applied | Skew | Ink-to-paper | Lighting drift |
|---|---|---|---|---|
| `3_invoice_photo_on_a_desk` | relit, deskewed −2.7° | −2.80° → 0.00° | 0.257 → 0.318 | 0.390 → 0.008 |
| `11_receipt_photographed` | relit, deskewed +3.3° | +3.20° → 0.00° | 0.370 → 0.315 | 0.422 → 0.000 |
| `13_contract_already_expired` | deskewed −1.4°, contrast | −1.40° → 0.00° | 0.064 → 0.268 | — |
| `16_invoice_line_items_do_not_sum` | deskewed −1.8°, contrast | −1.80° → 0.00° | 0.057 → 0.248 | — |
| `18_invoice_unlabelled_amounts` | relit, deskewed −2.8° | −2.90° → 0.00° | 0.334 → 0.355 | 0.406 → 0.004 |

### What was read wrong

Twenty-three of 216 fields. Fourteen were **not found** rather than found wrong —
a `tax_rate` on a zero-VAT invoice, a `payment_terms` the page does not state — 
which is the right failure: an absent field is visible, an invented one is not.

The nine genuinely wrong values cluster where you would expect. Document 16, the
faded photocopy, accounts for five of them on its own, including the invoice
number and both dates. That document also produced this evaluation's most useful
finding.

### The silent error this corpus produced, and how it was closed

An earlier run of exactly this corpus posted document 16 automatically at 85%
confidence with its invoice number and its date both wrong: `PRF-SG-20416` read
as `PRF.SG20496`, and `2026-07-17` as `2028-07-17`.

Grounding did not help, and this is the clearest available demonstration of what
grounding is and is not for. It proves a value came from a span of text that
exists on the page — and both wrong values did. The span existed in the OCR
output, and the OCR output was what was wrong. Grounding catches a model
inventing a value. It cannot catch an engine misreading one, and neither can two
channels reading the same OCR text.

Two things should have stopped it and neither did:

- **`dates_not_in_future` fired, as a warning.** It is now graded. Days ahead
  stays a warning — that is a clock, or a supplier billing early. Past four
  months it blocks, because a date two years out is not a document issued early,
  it is a digit misread by the same engine that read the rest of the page. The
  date is not the problem; it is the evidence.
- **Nothing downstream knew the page had been barely legible.** The OCR quality
  score is computed on the text that came back, and text that comes back
  confidently wrong scores as well as text that comes back right — this page
  scored 95%. The one signal available was that preprocessing had to rescue its
  ink from its paper before anything could read it. A page arriving below the
  contrast or sharpness floor now marks the read as degraded.

Document 16 now holds at 62% — the low score is the point, because that page
really was hard to read. Document 1 still posts at 95%.

### The warning tier that was a blocking tier

Running this corpus also exposed something the accuracy numbers could not:
fifteen of eighteen documents were being held, and several of them had nothing
wrong.

Confidence was answering two questions at once. It scored how well the values
were read *and* absorbed every rule failure — a warning cut it by 8% each, so
three advisory findings took a perfectly-read document from 95% to 72%, under an
84% threshold, and sent it to someone with nothing to correct. A warning that
holds documents is a blocking rule that has not admitted it, and it is how a
human-in-the-loop system ends up with a human in every loop.

Confidence now answers one question — did we read this correctly — and routing
answers the other. A blocked vendor, a duplicate, an expired purchase order and
an unusual VAT rate no longer move the score: they were all read perfectly. Only
a failure saying a *character* was misread does, weighted by the rule's own
severity — a fifth for a blocking one, a twentieth for a warning.

The effect on this corpus:

| | Before | After |
|---|---|---|
| Silent errors | 0 | **0** |
| Defects caught | 12/13 | **12/13** |
| Auto-posted | 3/18 | **4/18** |
| Documents held at ≥94% confidence | 0 | 6 |
| Time saved | 83% | **84%** |

The last two rows are the interesting ones. Six documents now sit at 94–95% and
are still held, which is the honest description of what they are: read
correctly, stopped by a fact about the world. Before, they were reported at
38–45%, which told the reviewer the reading was doubtful when it was not.

---

## The time-saved model

Time saved is the one figure here that is not a measurement, and a modelled
number presented like a measured one is how a project reports a saving nobody
can find afterwards. So the model is stated rather than buried:

| Assumption | Minutes |
|---|---|
| Keying one document by hand | 4.0 |
| Cross-checking it against the vendor file, the PO and prior postings | 3.5 |
| Reviewing one the system held, with the values and failing checks on screen | 1.5 |
| Cleaning up one silently posted error | 45.0 |

On this corpus: 135 minutes by hand, 21 minutes through here, **114 saved (84%)**.
Scaled up, that is 16.7 working days per thousand documents by hand against 2.6.
The nine minutes of machine time sit beside those numbers rather than inside them.

Three properties keep it honest.

**A held document is not free.** Counting a review as zero is the single
assumption that inflates these figures most, and it is why automation projects
report savings that never appear in a headcount. The reviewer still opens it,
still reads the values, still decides. What the system saved them is the keying
and the cross-checking.

**A posted error is charged against the saving.** Forty-five minutes each:
somebody finds the wrong record at reconciliation, traces it, raises a
correction. With enough of them the arithmetic concludes the system made things
worse, which it has to be able to conclude or it is not a measurement of
anything.

**Machine time is reported beside the saving, never subtracted from it.** A
model reading a page is not a person waiting for it, and in a batch it is not on
anyone's critical path.

---

## What this evaluation cannot tell you

- **Eighteen documents is a demonstration, not a statistic.** One document
  moving between posted and held shifts the straight-through rate by six points.
  The per-document table is the useful part; the percentages are context.
- **The corpus is synthetic.** The layouts, the damage and the defects were all
  authored here. Real suppliers produce failure modes nobody thought to plant —
  and the two authoring errors found in this corpus (a VAT that was correct when
  it was meant to be wrong, and two contracts with no body text) are a small
  demonstration of how easily a corpus can measure the wrong thing.
- **Ground truth is what the document says, not what is true.** A supplier who
  invoices the wrong amount is extracted correctly and caught by a rule, not by
  accuracy.
- **One run.** Timings in particular move with machine load; the accuracy
  numbers are stable across runs, the per-document seconds are not.
