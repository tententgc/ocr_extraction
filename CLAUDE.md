# DocuFlow — working notes

Unstructured documents in, validated ledger records out. Typhoon OCR 1.5 plus a
pluggable model layer. The headline metric is the **silent error rate**:
documents that posted automatically while carrying a wrong critical value.

Full docs: [README.md](README.md) · [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
· [docs/EVALUATION.md](docs/EVALUATION.md) · [docs/RULES.th.md](docs/RULES.th.md)

---

## Invariants — break these and the system stops being trustworthy

**Confidence answers one question: did we read the values correctly.** Whether a
document may post is decided in `routing.decide`. These were one number once,
and every warning quietly became a block — three advisory findings took a
perfectly-read document from 95% to 72% under an 84% threshold. Only failures in
`confidence.MISREAD_SIGNALS` move the score, weighted by severity (blocking 0.20,
warning 0.05).

**The three severities must behave differently.** `blocking` stops the document
and nothing posts past one, even on approval. `warning` is recorded and shown but
**does not hold anything**. `info` is a note. A warning that holds documents is a
blocking rule that has not admitted it, and it is how a human-in-the-loop system
ends up with a human in every loop.

**Two tolerances, never merged.** `AMOUNT_TOLERANCE` (2%) is how far an invoice
may differ from the *purchase order* it quotes — a commercial allowance.
`ARITHMETIC_TOLERANCE` (0.01%) is how far a document may differ from *itself* —
rounding only. Sharing one number let an 810-baht misread post as "totals add up"
on a 200,090-baht invoice.

**Grounding proves a value came from text on the page. It cannot catch an OCR
misread** — the span exists in the OCR output, and the OCR output is what was
wrong. Nor can two channels reading the same OCR text. The defences against that
are the arithmetic rules and the legibility signal below.

**A page that arrives below a legibility floor marks the read as degraded.** The
OCR quality score is computed on the returned text, and text that comes back
confidently wrong scores as well as text that comes back right.

**Preprocessing must never make a page worse.** Every correction is gated on a
measurement, applied, re-measured, and thrown away if the metric it exists to
improve did not improve. A clean digital PDF passes through untouched and writes
no second file. The prepared image is written *beside* the original, never over
it — the reviewer must see what arrived.

**Order in `prepare_image` is load-bearing**: flatten the lighting *before*
estimating skew. Skew is measured from an ink mask thresholded against the page's
mean brightness, so on a page lit from one side the mask marks the shadow and the
angle describes the gradient.

**A suggestion that could name two fields names neither.** `rules/repair.py`
blames a value only when exactly one field explains every broken relation. A
wrong number wearing a suggestion's confidence is worse than no suggestion.

---

## Commands

```bash
docuflow reset --confirm                     # empty documents, ledger, master data
docuflow reset --confirm --keep-master       # keep the register
docuflow master load real_samples/master     # a directory, loaded in dependency order
docuflow rules load real_samples/rules.json  # 5 expression + 2 agent rules
docuflow real-samples --out real_samples     # regenerate the corpus
docuflow evaluate --out report.json          # accuracy, coverage, time saved
docuflow serve                               # the web UI on :8000
.venv/bin/python -m pytest -q                # ~753 tests, no API key needed
```

Regenerating the corpus: edit `src/docuflow/samples/real_world.py`, never the
files in `real_samples/`.

Rebuilding the notebook: edit `notebooks/build_walkthrough.py`, then
`python notebooks/build_walkthrough.py`. Executing it needs a scratch harness
that swaps the four Colab-only cells (clone, upload, pip, getpass) for local
equivalents, then restores their source and clears their outputs.

---

## The corpus

`real_samples/` — filenames are `<n>_<slug>_<True|False>.<ext>`. The suffix is
the **expected outcome**, stated before the system is asked:

- `_True` — nothing is wrong; holding it is a false alarm costing a review
- `_False` — something is wrong; posting it is a silent error

The two are not symmetric and the evaluation must not treat them as such: a
missed defect is a wrong payment, a false alarm is a wasted minute.

`REAL_DOCUMENTS` are the defect cases. `VARIED_DOCUMENTS` are correct documents
in unfamiliar layouts — they measure the false-alarm rate, which the accuracy
figures cannot see.

`ground_truth.json` carries `reference_text` per document, taken from the PDF's
text layer *before* it is photographed. That is the reference OCR is scored
against, and it is exact because the corpus is drawn here.

---

## Configuration in this checkout

`.env` is gitignored and holds live keys — **never commit it**. Current setup:
`LLM_PROVIDER=typhoon`, `LLM_VERIFY_PROVIDER=claude`, `OCR_PROVIDER=ollama`,
`EXPECTED_VAT_RATES=0,7,15` (the user's choice; Thailand is 7).

The two readings must be **different backends** — two passes of one model repeat
the same mistake, so their agreement means much less than it looks.

Tests pin credentials *and* business policy in `tests/conftest.py`. That has now
caught six settings the hard way; a green suite must be a statement about the
code, not about one machine's `.env`.

---

## Bugs found the hard way — do not reintroduce

| Symptom | Cause |
|---|---|
| Claude 400 `system: Input should be a valid array` | adapter sent `system=None`; the key must be omitted. Reads like a billing failure. |
| A clean invoice measured as having no contrast | contrast as a percentile spread over all pixels — a page is mostly paper. Measure ink against paper. |
| A densely typeset page reported as unevenly lit | lighting as the mean of each grid cell measures ink density. Read the *paper* level per cell. |
| Photographs left crooked, deskew reverted | skew estimated before the lighting was flattened. |
| Document posted with two wrong values at 85% | `dates_not_in_future` was a flat warning. Now graded: past 120 days it blocks. |
| Fifteen of eighteen documents held | warnings cutting confidence. See the invariants. |
| `OSError: too many open files` | `db.connect()` cached one connection per path forever, 4 FDs each under WAL. LRU-capped now. |
| Two threads rendering PDFs segfaulted | PDFium is not thread-safe; `render.py` holds a module lock. |

---

## Current state

Branch `chore/eval-and-fixes`, based on `main` after PR #1. Pushed.

Three evaluations beyond field accuracy, each answering something the others
cannot:

| Module | Question |
|---|---|
| `evaluation/ocr_accuracy.py` | Was the page read? CER against `reference_text`, plus recall of the values that become a payment. Both, because they disagree — a page can score 95% characters while losing the total. |
| `evaluation/gatekeeping.py` | Did the gate open correctly, and did each rule fire for the reason it exists? Per-rule confusion matrix; a rule no document exercises is **untested**, not passing. |
| `evaluation/effort.py` | Time saved. Modelled, not measured — assumptions print with the answer. |

Scoring gotchas already fixed: strip HTML/markup before comparing characters
(the OCR returns tables on purpose and every `<td>` would score as error), and
judge a value's length on its compact form (`7.0` prints as `"7.00"`, four
characters and none of them distinguishing).

Never sum missed defects and false alarms. Each gets its own rate over its own
population — a missed-defect rate over the whole corpus falls simply by adding
documents that were never going to be held.

**Standing instruction from the user:** commit after every completed feature; run
only small/fast tests — they run the long suites themselves.
