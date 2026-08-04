# Evaluation

What the system does on eighteen documents that arrive the way documents
actually arrive, and — more usefully — what it got wrong and why.

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

<!-- RESULTS -->

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
