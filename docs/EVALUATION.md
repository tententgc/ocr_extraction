# Evaluation

What the system does on twenty-four documents that arrive the way documents
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

**Defect recall** is what stops the corpus flattering the reader. Twelve of the
twenty-four documents have a specific defect planted in them and declare which
rule should catch it. A corpus that only measured reading would score well on a
system that validates nothing.

**False alarms** are the number the others cannot see. Twelve documents are
correct and must post; six of those are deliberately laid out unlike anything
the system was tuned on. A pipeline that holds valid documents costs nothing
measurable per document and everything in aggregate, because a queue full of
false alarms is a queue nobody reads — at which point the missed defects stop
being caught either.

**Time saved** is modelled, not measured, and the assumptions print with the
number every time. See [the assumptions](#the-time-saved-model) below — they are
deliberately conservative and the arithmetic is allowed to conclude that the
system made things worse.

---

## The setup

| | |
|---|---|
| Corpus | [`real_samples/`](../real_samples/) — 24 documents, 12 that should post and 12 that should not. Six Thai documents were added afterwards and are measured separately in [section 5](#5-does-it-work-on-thai-documents); every number in sections 1–4 is from the 24-document run, unchanged. |
| Arrival | 14 digital PDFs, 5 scans, 3 phone photographs, 2 faded photocopies |
| OCR | Typhoon OCR 1.5 (3B) on-device via Ollama |
| Reading 1 | Typhoon `typhoon-v2.5-30b-a3b-instruct`, field-first |
| Reading 2 | Claude `claude-opus-5`, label-first — a different backend on purpose |
| Rules | 26 built-in + 7 written for this corpus (5 expression, 2 agent) |
| Master data | 13 vendors, 10 purchase orders, 5 contracts, loaded from CSV |
| Auto-post threshold | 84% confidence |

The two readings are deliberately on different backends. Two passes of the same
model over the same page repeat the same mistake, so their agreement would mean
much less than it appears to.

Documents run in numeric order because document 15 is a duplicate of document 1
and can only be detected once document 1 has posted.

The filename carries the expected outcome — `_True` should post, `_False` should
be held — so the evaluation compares two decisions rather than grading the
system against itself.

---

## The four questions

| | Result |
|---|---|
| **Did anything wrong reach the ledger?** | **No.** 0 missed defects out of 12 documents that should have been held. |
| **Was the page read?** | 92.8% character accuracy; **90.3% recall** on the values that become a payment. |
| **Did the gate open correctly?** | 87.5% — 21 of 24 documents went the way the corpus said they should. |
| **Did each rule earn its place?** | 12 rules exercised, **all 12 at full recall**. 18 untested by this corpus. |

24 documents, 789 seconds (33s each). Field accuracy 252/289 (87.2%). Time saved,
modelled: 158 of 180 manual minutes (88%).

---

## 1. Was the page read correctly?

Every other number is downstream of this one. Two measurements, because they
answer different questions and they disagree.

| | |
|---|---|
| Character accuracy | **92.78%** (CER 7.22%, over 13,587 reference characters) |
| Value recall | **90.3%** — 215 of 238 values that matter survived into the transcript |

The reference is exact rather than estimated: the corpus is drawn here, so the
PDF text layer is the string the renderer was given, captured before the page
was ever photographed.

**Where the characters went.** Two documents drag the corpus figure down hard,
and both are thermal receipts at ~55%. An 80mm receipt is a few hundred
characters, so the denominator is tiny and any reformatting dominates — the OCR
returns a tidier transcript than the raw text layer, and on a page that short
that reads as a third of the characters being wrong. Both receipts nonetheless
score 86% value recall and every amount on them is correct. It is a caution
about the metric on short pages, not about the engine.

Excluding those two, character accuracy across the other 20 documents is above
89% everywhere and above 93% on all but three.

**Where the values went.** The 23 lost values cluster, and the clustering is the
useful part:

| Document | Lost | Why |
|---|---|---|
| `16_line_items_do_not_sum` (photocopy) | 6 of 10 | the worst page in the corpus, 72% characters |
| `13_contract_already_expired` (photocopy) | 3 of 6 | faded prose, 89.8% characters |
| `19_two_column_layout` | 3 of 12 | `buyer_name` misread as "Mirthphol", an inserted h |
| `6`, `10` | `vendor_tax_id`, `payment_terms` | the layout prints neither |

The two faded photocopies account for nine of the twenty-three on their own.
Both were held.

---

## 2. Did each rule fire for the reason it exists?

Every document declares which rules its defect should trip, so each rule gets a
confusion matrix rather than a share of a corpus-wide score.

**Twelve rules exercised, twelve at full recall.** Nothing that should have
fired stayed silent:

| Rule | Recall | Precision | Fired |
|---|---|---|---|
| `invoice_totals_consistent` | 100% | 100% | 1 |
| `tax_rate_consistent` | 100% | 50% | 2 |
| `line_items_sum_to_subtotal` | 100% | 100% | 1 |
| `vendor_known` | 100% | 50% | 2 |
| `vendor_active` | 100% | 100% | 2 |
| `bank_account_matches_vendor` | 100% | 50% | 2 |
| `purchase_order_required` | 100% | 33% | 3 |
| `po_amount_within_tolerance` | 100% | 100% | 1 |
| `po_currency_matches` | 100% | 100% | 1 |
| `duplicate_document` | 100% | 100% | 1 |
| `contract_expiring_soon` | 100% | 100% | 1 |
| `contract_expired` | 100% | 100% | 1 |

**Read the precision column carefully — it is not a fault count.** A rule "fires
unexpectedly" whenever it trips on a document that did not declare it, and most
of those are correct. `purchase_order_required` shows 33% because it fired on
three documents where only one declared it: the other two genuinely quote no
purchase order and are genuinely above the threshold. The corpus declares the
*defect a document was built around*, not every true statement about it. A
precision figure here measures how completely the corpus was annotated, and it
is the honest way round — inflating it would mean annotating after seeing the
output.

**Eighteen rules were never exercised** and are reported as untested rather than
as passing: `tax_id_checksum`, `dates_not_in_future`, `due_date_after_issue`,
`contract_dates_ordered`, `contract_parties_distinct`, `vat_rate_expected` and
the form rules among them. Untested is the honest word, and the list is the
shopping list for the next batch of corpus documents.

---

## 3. Did the document go the right way?

The expectation is in every filename before the system is asked.

| | Documents | Correct | Wrong |
|---|---|---|---|
| Should be **held** (`_False`) | 12 | 12 | **0 missed defects** |
| Should **post** (`_True`) | 12 | 9 | 3 false alarms |

**Zero missed defects.** Nothing that should have been stopped reached the
ledger, and every one of the three posted documents was read perfectly.

**Three false alarms, and all three are the same failure**: a document the
system could not read confidently, held on the confidence threshold with the
rules behaving correctly.

| Document | Confidence | Rules firing |
|---|---|---|
| `9_invoice_thai_buddhist_dates` | 63% | **none** |
| `18_invoice_unlabelled_amounts` | 60% | 1 |
| `21_invoice_minimal_unlabelled` | 72% | 2 |

That is the right shape for the failures to have. None was a rule misfiring on a
valid document; all three are the system declining to post a page it read badly,
which is what the conservatism is for. Two of the three are the invoices with no
field labels at all, which is the hardest case in the corpus by construction.

The structural-variation set is the other half of this number: six correct
invoices in unfamiliar layouts — two-column, dense, minimal, Thai, scanned,
thermal — of which **five posted**. Layout alone does not hold a document.

---

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
| Defects caught | 12/13 | **13/13** |
| Auto-posted | 3/18 | 3/18 |
| Documents held at ≥91% confidence | 0 | **5** |

The last row is the one that matters. Five documents now sit at 91–93% and are
still held, which is the honest description of what they are: read correctly,
stopped by a fact about the world. Before, the same documents reported 38–45%,
telling the reviewer the reading was doubtful when it was not — and a reviewer
who opens a 40% document expecting to fix a misread number, and finds a
perfectly-read invoice from a blocked vendor, learns to distrust the number.

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

On this corpus: 135 minutes by hand, 22.5 minutes through here, **112.5 saved
(83%)**. Scaled up, that is 16.7 working days per thousand documents by hand
against 2.8. The ten minutes of machine time sit beside those numbers rather than
inside them.

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

## 5. Does it work on Thai documents?

Asked because the answer was assumed. Thai had been in the corpus from the start
— two invoices on the `thai_tax` layout — and both posted, so the language looked
covered. It was not, and the shape of the gap is worth stating before the
results.

**Thai numerals and Thai labels are different problems, and only one of them is
loud.** ๙ either survives OCR or it does not, and either way you find out. Arabic
digits under Thai labels come back perfectly and the only question left is
whether anything understood that `จำนวนเงินทั้งสิ้น` is the total — a question
an English-trained extractor can get wrong in silence. The corpus tested only the
first, and the second is what every real Thai business invoice looks like.

So six documents were written to hold the language fixed and vary what is asked
of it: Arabic digits with Buddhist-era dates, the same with the VAT
miscalculated, a receipt, a contract in Thai prose, a supplier who is genuinely
not in the register, and a registered vendor with no tax ID printed — so the
match has to be carried by the Thai name alone.

### What was already fine

Worth reporting, because it is where the search started and it is where the
problem was not:

| | |
|---|---|
| Buddhist-era dates | `24 กรกฎาคม 2569` → 2026-07-24, in either numeral system, slashed or spelled out |
| Amounts | `๑๓๕,๐๐๐.๐๐` and `135,000.00` both parse |
| Thai vendor names | 12 of 12 probes resolve correctly, including spaces lost, legal form dropped, and `ซ` misread as `ช` |
| Unknown Thai suppliers | `หจก. สมชายการช่าง` correctly matches nothing — the register's Thai names are short once `บริษัท` and `จำกัด` are stripped, and a false match here would pay the wrong company |

The vendor matcher already knew about `บริษัท`, `จำกัด` and `มหาชน`, and the
master file already carried Thai aliases. Document 30 — a registered vendor named
only in Thai, no tax ID printed, so there is no exact identifier to fall back on
— read 12 of 12 fields correctly and posted.

### Three faults, none of them where they were expected

**A Thai receipt is classified as an invoice.** The heading a Thai till prints is
`ใบเสร็จรับเงิน / ใบกำกับภาษีอย่างย่อ` — *abbreviated tax invoice*, which is the
Revenue Department's name for the slip you get at a counter. Matched as a bare
substring it also satisfies the invoice keyword `ใบกำกับภาษี`, so every Thai
receipt in the country scores as an invoice on the strength of its own receipt
heading. The same nesting sits one level down between `เลขที่` and
`เลขที่ใบเสร็จ`.

The fix is general and contains no Thai: **when two matched phrases are nested,
only the more specific one votes.** English has the same nesting — `No.` inside
`Receipt No` — and it was doing the same damage there. Keyword-only
classification errors across the corpus went from 3 of 30 to 0 of 30, and two of
the three corrected documents are English receipts. A Thai document found a bug
that was never about Thai.

**A contract auto-posted with both parties misread.** Document 28 posted at 92%
with `party_a` read as `Mitrphol` and `party_b` carrying `แฟชิลิตี้` for
`แฟซิลิตี้` — one stroke apart, the commonest Thai OCR slip there is. Both are
critical fields for a contract. Both were **grounded**, because the misread text
is genuinely in the OCR output and grounding can only prove a value came from the
page, never that the page was read right.

An invoice naming the wrong company is caught downstream by `vendor_known`
resolving the name against the master file. A contract had no equivalent:
`contract_matches_master` compared the dates and the value, and never the
parties, although the contract number is an exact key into a register that stores
both. A page could have named a different counterparty entirely and posted.

The comparison now happens, and how it is made is the load-bearing part. Names
are compared as **organisations** — legal form stripped, fuzzy — so one misread
character is still the same company, and then resolved through the **vendor
register**, because nothing about the two strings
`บริษัท เจ้าพระยา แฟซิลิตี้ จำกัด` and `Chao Phraya Facility Services Co., Ltd.`
says they are one counterparty. The alias column does. String equality here would
have turned a check meant for a substituted company into a false-alarm machine.

**A document took ten minutes instead of thirty seconds.** Both Thai-numeral
documents drew a `408 Request timed out` from Typhoon Cloud, and the engine
treated a timeout the way it treats bad text: escalate to the next provider. The
next provider was the local 3B model, at roughly 600 seconds a page against 3
seconds for the hosted call.

Nothing was wrong with the *result* — the local model read both pages at 96%
quality, undegraded, and both posted correctly. The chain did exactly what it was
built to do. What it did not do was say so: the trace reported `ok` and named a
provider, and the only other evidence was a clock.

Two changes. **Bad text is a verdict on the page and warrants escalation; a
timeout is not and warrants a retry** — `OcrTransientError` now separates the
two, and a 408, 429 or 5xx asks the same engine again before the page moves down
the chain. And when the configured engine is not the one that read the document,
the result says so, separately from `degraded`, because falling back to a slower
engine is not a quality problem and must not be reported as one.

### After the fixes

Six Thai documents, 140 seconds, all three fixes in:

| Document | Type | Decision | Expected | Conf | Fields | OCR |
|---|---|---|---|---|---|---|
| 25 Thai labels, Arabic digits | invoice | post | post | 84% | 13/14 | 95.3% |
| 26 VAT stated 7%, charged 6% | invoice | **hold** | hold | 68% | 14/14 | 98.0% |
| 27 ใบเสร็จรับเงิน, photographed | **receipt** | post | post | 92% | 8/8 | 70.4% |
| 28 สัญญาจ้างบริการ | contract | post | post | 92% | 9/11 | 94.6% |
| 29 หจก. not in the register | invoice | **hold** | hold | 68% | 12/13 | 96.6% |
| 30 Thai name, no tax ID | invoice | post | post | 84% | 12/12 | 95.6% |

**6 of 6 decisions correct. Field accuracy 68 of 72 (94.4%). No document over 31
seconds.** Document 26 is held by `tax_rate_consistent` and document 29 by
`vendor_known` — each the rule it was written to exercise.

Document 27 is now typed `receipt`. Document 28's parties are still misread the
same way, and `contract_matches_master` **passes** — which is the check working,
not the check missing: it compared them, resolved `แฟชิลิตี้` to the same
organisation, and did not raise a false alarm. A substituted company would not
get that treatment.

The four remaining field misses are honest reading errors, not defects:

| Document | Field | What happened |
|---|---|---|
| 25 | `vendor_name` | OCR returned `บริษัท สยามเทคโนโลยี จำกัด` for `บริษัท สยามเทค ซัพพลาย จำกัด` — a plausible invented company name. It posted anyway because the tax ID was read correctly and resolves the vendor exactly, which is why tax ID is tried first. |
| 28 | `party_a`, `party_b` | `Mitrphol`, `แฟชิลิตี้` — one character each |
| 29 | `vendor_tax_id` | misread, which correctly trips `tax_id_checksum` on top of `vendor_known` |

Two further mismatches turned out to be the corpus, and both were the trap this
document already warns about: document 27 declared `tax_rate`, which the receipt
schema has no field for, and `payment_method` as the printed Thai, when the field
is an **enum** — a classification, not a transcription. The system answered
`credit_card` and was marked wrong for being right. Both were removed from the
ground truth rather than from the system.

### What this section does not settle

The retry rides out a *flaky* timeout. On these two pages the timeout is **not
flaky** — it recurred on every attempt across two runs, so the page still falls
back to the local engine and still takes ten minutes. Retrying does not fix that
and was never going to.

Measuring it did change the retry, though. Three attempts made the worst case
three minutes *worse*: each one waits out a full server-side timeout before
failing, and a page that provokes a timeout provokes it every time. One retry
buys what retrying is for — a single unlucky call — and caps the cost of being
wrong at one wasted timeout. `TRANSIENT_ATTEMPTS` is 2 because that was measured,
not chosen.

Whether Typhoon Cloud can process a Thai-numeral page at all is a question about
the hosted model, and two documents cannot answer it. What can be said is
narrower and still useful: **Thai is not the problem, Thai numerals are.** Every
Thai document printed with Arabic digits ran in 11–31 seconds, the same as the
English corpus.

---

## What this evaluation cannot tell you

- **Twenty-four documents is a demonstration, not a statistic.** One document
  moving between posted and held shifts a rate by four points. The per-document
  tables are the useful part; the percentages are context.
- **The corpus is synthetic, and it has been wrong more often than the pipeline
  has.** Every authoring error found so far looked like a system fault until it
  was read closely: a VAT that was correct when it was meant to be wrong; two
  contracts rendered with no body text; four documents declaring ground truth
  their own layout never draws; two real defects labelled as clean cases; and a
  value-recall metric searching for `2026-07-14` on pages that print
  "14 July 2026". Six of those were caught by this evaluation on its first run,
  which is the strongest thing that can be said for it and the weakest thing
  that can be said for the corpus.
- **Precision on a rule is a statement about annotation, not about the rule.**
  The corpus declares the defect each document was built around, not every true
  statement about it, so a rule that fires correctly on an undeclared document
  scores as imprecise. Annotating after seeing the output would fix the number
  and ruin the measurement.
- **Eighteen of the thirty rules were never exercised.** They are reported as
  untested, not as passing.
- **Ground truth is what the document says, not what is true.** A supplier who
  invoices the wrong amount is extracted correctly and caught by a rule, not by
  accuracy.
- **One run.** Timings in particular move with machine load; the accuracy
  numbers are stable across runs, the per-document seconds are not.
