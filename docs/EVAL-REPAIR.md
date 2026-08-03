# Does the repair agent help? — measured

**Answer: not on this corpus, and the reason is the interesting part.**

Ten documents, same rendering, same engines, same rules, one variable:
`REPAIR_DISABLED`. Each arm gets its own database so duplicate detection cannot
see the other arm's postings.

| | repair off | repair on |
|---|---|---|
| Posted automatically | 2 / 10 | **2 / 10** |
| Blocking rule failures | 7 | **7** |
| Fields populated | 122 | **123** |
| Fields recovered by the agent | — | **1** |
| Total wall clock | 384s | 323s |

The latency difference is noise — model and network variance between two runs,
not a saving. The agent only runs on documents that are already failing, and
even then it costs less than the OCR pass it follows.

## Per document

| document | repair off | repair on | recovered |
|---|---|---|---|
| inv-clean-classic | auto_post 93% | auto_post 93% | — |
| inv-thai-scan | auto_post 94% | auto_post 94% | — |
| contract-facility | needs_review 79% | needs_review 78% | — |
| receipt-fuel | needs_review 71% | needs_review 71% | — |
| inv-bad-math | needs_review 45% · 1 blocking | needs_review 45% · 1 blocking | — |
| inv-duplicate | needs_review 45% · 1 blocking | needs_review 45% · 1 blocking | — |
| inv-eur-over-po | needs_review 45% · 1 blocking | needs_review 45% · 1 blocking | `tax_rate` |
| form-vendor-onboarding | needs_review 45% · 1 blocking | needs_review 45% · 1 blocking | — |
| po-orient-paper | needs_review 45% · 1 blocking | needs_review 45% · 1 blocking | — |
| inv-unknown-vendor | needs_review 38% · 2 blocking | needs_review 38% · 2 blocking | — |

## Why it changed nothing

Every held document is held for a reason repair is not allowed to remove, and
the agent identified that correctly in each case. Its own account, per document:

| document | what the agent concluded |
|---|---|
| inv-bad-math | "All totals are consistent with the document's line items and stated values." |
| inv-duplicate | "No missing or incorrect values found; duplicate check failed due to prior posting of identical invoice." |
| po-orient-paper | "Vendor status prevents processing; no extraction errors found." |
| inv-unknown-vendor | "No purchase order reference found on the document, which is required by policy." |
| receipt-fuel | "The currency is not stated on the document, so no value can be proposed." |
| form-vendor-onboarding | "The requested amount is missing from the document and cannot be recovered." |
| contract-facility | "The contract expired 4 days ago, but all required values are correctly stated." |

Those are not extraction failures. An invoice whose subtotal and tax do not add
up to its total is a bad invoice; a second copy of an invoice already posted is
a duplicate; a vendor on hold is on hold. **A business rule failing is not
something to repair, and the distinction is the whole design.**

The two documents held without any blocking rule — contract-facility at 79% and
receipt-fuel at 71% — are held by confidence rather than by a missing field, so
there was nothing for the agent to find. On receipt-fuel it said exactly that:
the currency is genuinely not printed on the page.

## The result that matters most

On **inv-bad-math** the agent proposed a replacement subtotal, and the vetting
step threw it out:

```
rejected subtotal: subtotal was already read and verified; not overwritten
```

That invoice fails because its arithmetic is wrong. The one move that would have
made it pass is to change a number, and the one number that would do it is the
subtotal — which two channels had already read and agreed on. The guard held.
This is the exact failure the design exists to prevent, and it was exercised on
a real document rather than only in a unit test.

## The one recovery

On **inv-eur-over-po** the agent found `tax_rate` printed on the page and missed
by extraction, and it was accepted with the line it came from. The document is
still held, because it is blocked by invoicing more than the purchase order
allows — a real problem that a recovered tax rate does not touch.

## What this says about keeping it

Two-channel extraction on a clean corpus already fills almost everything, so
there is little left to recover. That is a statement about this corpus, not
about the feature: the documents here are generated, well-formed and use labels
the extractor knows. The case repair is built for is the messy one — a scan
with an unusual label, a layout nobody anticipated, a field printed somewhere
the schema did not expect.

On this evidence:

- **Keep it enabled.** It runs only on documents already destined for a human,
  costs less than the OCR that preceded it, and recovered a real field.
- **Do not claim a throughput gain.** There is none here, and the honest number
  is 2/10 either way.
- **Re-measure on real documents.** The corpus that would show a difference is a
  batch of genuine supplier invoices, not this one.

## Reproducing

```bash
REPAIR_DISABLED=true  docuflow process <file>   # baseline
REPAIR_DISABLED=false docuflow process <file>   # with repair
```

The trace carries a `repair` step recording what was proposed, what was
accepted, and the reason anything was rejected.
