"""Generate the Colab walkthrough notebook.

Written as a builder rather than hand-edited JSON: the cells are long, and
escaping them by hand is how a notebook ends up invalid.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("notebooks/docuflow_walkthrough.ipynb")

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text: str) -> None:
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    })


# ---------------------------------------------------------------- title ----
md(r"""
# DocuFlow — an unstructured document, one stage at a time

This notebook runs the whole pipeline **stage by stage**, printing what each one
produced, so you can see where a value came from and why the system did or did
not trust it. At the end the same document goes through in a single call, and the
timings show what each stage cost.

Everything here runs against **Typhoon Cloud** (OCR and the primary reading) and
**Claude** (the second, independent reading). No local model, no GPU.

```
  file ──▶ render ──▶ OCR ──▶ classify ──▶ ┌─ read A: Typhoon ─┐
                                            │                   ├─▶ merge + ground ──▶ rules ──▶ route
                                            └─ read B: Claude  ─┘
```

**Why two readings.** One model reading a page and reporting a number is a claim.
Two models reading it separately and agreeing is evidence. Measured on this
corpus: fields both channels agree on are right **97%** of the time; fields they
disagree on, **45%**. That gap is the whole reason the second call exists — the
system knows which of its own answers to distrust.

**Why grounding.** Every value must be traceable to a span of text on the page.
A value the model produced but the page does not contain is treated as invented
and never posts automatically, however confident the model sounded.

| Stage | What it does | Model? |
|---|---|---|
| render | PDF/photo → page images | no |
| OCR | image → text (Typhoon OCR 1.5) | yes |
| classify | which kind of document | keywords first, model only if unsure |
| extract A | field-first reading | Typhoon |
| extract B | label-first reading | Claude |
| merge | agreement, grounding, confidence | no |
| rules | business checks against master data | no |
| route | post automatically, or hold | no |

Note the "no" column. **Models read; code decides.** A model never chooses whether
a document posts.
""")

# -------------------------------------------------------------- install ----
md(r"""
## 1. Install

The repository is private, so pick whichever of these suits you.
""")

code(r'''
# --- Option A: clone with a GitHub token (private repo) --------------------
# The token is read with getpass, so it is never printed or stored in the
# notebook. Create one at https://github.com/settings/tokens with `repo` scope.
import getpass, os, subprocess, sys

REPO = "tententgc/mirtphol_doc_extraction"

if not os.path.exists("mirtphol_doc_extraction"):
    token = getpass.getpass(f"GitHub token for {REPO} (leave blank to skip): ").strip()
    if token:
        url = f"https://{token}@github.com/{REPO}.git"
        subprocess.run(["git", "clone", "--depth", "1", url], check=True)
        del token, url          # not kept around
    else:
        print("Skipped. Use Option B below.")
''')

code(r'''
# --- Option B: upload a zip of the repository -----------------------------
# Run this instead of Option A if you would rather not make a token.
#   On your machine:  git archive -o docuflow.zip HEAD
# then upload docuflow.zip here.
import os

if not os.path.exists("mirtphol_doc_extraction"):
    from google.colab import files          # noqa: F401 - Colab only
    uploaded = files.upload()
    name = next(iter(uploaded))
    os.makedirs("mirtphol_doc_extraction", exist_ok=True)
    import zipfile
    with zipfile.ZipFile(name) as archive:
        archive.extractall("mirtphol_doc_extraction")
    print("extracted to mirtphol_doc_extraction/")
''')

code(r'''
%pip install -q -e "mirtphol_doc_extraction[claude]"
print("installed")
''')

# ----------------------------------------------------------------- keys ----
md(r"""
## 2. Keys

Two keys, entered with `getpass` so they never appear in the notebook or its
saved output.

| Key | Used for | Get one at |
|---|---|---|
| `TYPHOON_API_KEY` | OCR **and** the primary reading | https://opentyphoon.ai |
| `ANTHROPIC_API_KEY` | the second, independent reading | https://console.anthropic.com/settings/keys |

If you only have the Typhoon key, the notebook still runs: the second channel
falls back to a deterministic label matcher and says so. You will see a lower
agreement rate, which is the honest result.
""")

code(r'''
import getpass, os

os.environ["TYPHOON_API_KEY"] = getpass.getpass("TYPHOON_API_KEY: ").strip()
os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("ANTHROPIC_API_KEY (optional): ").strip()

print("typhoon key set  :", bool(os.environ["TYPHOON_API_KEY"]))
print("anthropic key set:", bool(os.environ["ANTHROPIC_API_KEY"]))
''')

# ---------------------------------------------------------------- setup ----
md(r"""
## 3. Configure the two backends

`Settings` is the whole configuration. Here it is built in code rather than read
from `.env`, so the notebook is self-contained.

The important pair:

* `llm_provider="typhoon"` — the primary reading
* `llm_verify_provider="claude"` — the second one

They are deliberately **different backends**. Two passes of the same model over
the same page tend to repeat the same mistake, so their agreement would mean much
less than it appears to.
""")

code(r'''
import os
from pathlib import Path
from docuflow.config import Settings
from docuflow.bootstrap import bootstrap

WORK = Path("/content/docuflow-data") if Path("/content").exists() else Path("./docuflow-data")

settings = Settings(
    data_dir=WORK,
    # --- OCR: Typhoon Cloud ------------------------------------------------
    ocr_provider="typhoon_api",
    ocr_cloud_model="typhoon-ocr-v1.5",
    ocr_max_pages=4,
    # --- reading 1: Typhoon ------------------------------------------------
    llm_provider="typhoon",
    llm_cloud_model="typhoon-v2.5-30b-a3b-instruct",
    # --- reading 2: Claude, a different backend on purpose -----------------
    llm_verify_provider="claude" if os.environ.get("ANTHROPIC_API_KEY") else "auto",
    llm_anthropic_model="claude-sonnet-5",
    llm_verify_timeout_seconds=120,
    # --- keys --------------------------------------------------------------
    typhoon_api_key=os.environ.get("TYPHOON_API_KEY", ""),
    anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    # A small demo vendor/PO/contract register so the business rules have
    # something to check against. Replace it with your own CSV any time.
    seed_demo_master_data=True,
    # --- business policy ---------------------------------------------------
    # Stated rather than left to default, for two reasons: these are finance
    # decisions and belong in the open, and pinning them means a stray .env in
    # the working directory cannot quietly change what this notebook shows.
    auto_post_min_confidence=0.84,
    po_required_above=50_000,
    expected_vat_rates="0,7",
    amount_tolerance=0.02,
)

info = bootstrap(settings, with_samples=True)
print("data dir :", settings.data_dir)
print("seeded   :", info.get("master_data") or info)
''')

code(r'''
# What each backend thinks of itself right now. `available=False` here is the
# cheapest possible place to find out a key is wrong.
from docuflow.llm.factory import describe_llms, resolve_llm, resolve_verify_llm
from docuflow.ocr.factory import describe_providers

print("OCR")
for row in describe_providers(settings):
    if row["available"]:
        print(f"   {row['key']:22} {row['label']}")

print("\nAGENT MODELS")
for row in describe_llms(settings):
    if row["available"]:
        print(f"   {row['key']:46} {row['label']}")

primary, note = resolve_llm(settings)
verify, verify_note = resolve_verify_llm(settings, primary)
print(f"\nreading 1 : {getattr(primary, 'model', None) or primary} {note}")
print(f"reading 2 : {getattr(verify, 'model', None) or verify} {verify_note}")
if verify is None:
    print("            (no second model — the deterministic label matcher will stand in)")
''')

# ------------------------------------------------------------- document ----
md(r"""
## 4. The document

Either upload your own (PDF, JPG, PNG, TIFF) or use one from the bundled corpus.
The corpus documents carry known ground truth, which is what makes them useful
for a walkthrough — you can check the answers.

| sample | what it demonstrates |
|---|---|
| `inv-clean-classic` | the happy path — everything agrees |
| `inv-bad-math` | stated total ≠ subtotal + tax |
| `inv-thai-scan` | Thai text, Buddhist-era dates, Thai numerals |
| `inv-unknown-vendor` | vendor not in the master file, unfamiliar bank account |
| `inv-eur-over-po` | invoice larger than the purchase order it quotes |
| `contract-facility` | a contract, not an invoice — a different schema |
""")

code(r'''
SAMPLE = "inv-clean-classic"     # ← change this, or upload your own below

document_path = settings.sample_dir / f"{SAMPLE}.pdf"

# --- or upload your own ---------------------------------------------------
UPLOAD_YOUR_OWN = False
if UPLOAD_YOUR_OWN:
    from google.colab import files
    got = files.upload()
    name = next(iter(got))
    document_path = WORK / name
    document_path.write_bytes(got[name])

print("document:", document_path.name, f"({document_path.stat().st_size:,} bytes)")
''')

# --------------------------------------------------------------- render ----
md(r"""
## Stage 1 — Render

A PDF page or a phone photo both become the same thing: an image at a known
size. Nothing is read yet.

`embedded_text_ratio` reports how much of the PDF already carries a text layer.
A born-digital PDF can skip OCR entirely; a scan cannot.
""")

code(r'''
from docuflow.ocr.render import render_document, embedded_text_ratio

pages = render_document(document_path, settings.page_dir / "walkthrough",
                        max_pages=settings.ocr_max_pages, target_dim=settings.ocr_image_dim)

print(f"{len(pages)} page image(s)")
for page in pages:
    print(f"   page {page.page_number}: {page.width}x{page.height}  {page.path.name}")
print(f"\nembedded text layer: {embedded_text_ratio(pages):.0%} of pages")

try:
    from IPython.display import Image, display
    display(Image(filename=str(pages[0].path), width=520))
except ImportError:            # running outside a notebook
    print("page image at", pages[0].path)
''')

# ------------------------------------------------------------------ ocr ----
md(r"""
## Stage 2 — OCR (Typhoon OCR 1.5, cloud)

Typhoon OCR is trained on documents rather than on photographs of the world, so
it returns **structured markdown** — tables stay tables — instead of a flat blob
of words. That structure is what makes a total findable later.

`score_document_text` is a deterministic quality check on the *text*, not on the
model's opinion of itself: gibberish ratio, line structure, expected characters.
A page that scores badly is marked degraded, and every value read from it is
treated with more suspicion downstream.
""")

code(r'''
from docuflow.ocr.engine import run_ocr
from docuflow.ocr.quality import score_document_text

ocr = run_ocr(pages, settings)
quality = score_document_text([p.text for p in ocr.pages])

print(f"provider : {ocr.provider}")
print(f"pages    : {len(ocr.pages)}   characters: {ocr.char_count:,}")
print(f"took     : {ocr.duration_ms / 1000:.1f}s")
print(f"quality  : {quality.score:.2f}  acceptable={quality.acceptable}  degraded={ocr.degraded}")
if quality.reasons:
    print("         ", "; ".join(quality.reasons))
print("\n" + "=" * 78)
print(ocr.full_text[:1800])
print("=" * 78)
''')

# ------------------------------------------------------------- classify ----
md(r"""
## Stage 3 — Classify

Which schema applies? Keyword scoring runs first and is usually decisive — the
words "Tax Invoice" are not ambiguous. The model is only asked when the keywords
are close, because paying for a model call to be told something obvious is waste.

The schema chosen here decides which fields the next stage even looks for.
""")

code(r'''
from docuflow.agents.classifier import classify, classify_by_keywords
from docuflow.schemas.registry import get_schema

doc_type, keyword_confidence, scores = classify_by_keywords(ocr.full_text)
print("keyword scores")
for name, score in sorted(scores.items(), key=lambda kv: -kv[1])[:5]:
    print(f"   {name:16} {score:.2f}" + ("   <- best" if name == doc_type else ""))

result = classify(ocr.full_text, primary)
schema = get_schema(result.doc_type)

print(f"\ndecided  : {result.doc_type}  ({result.confidence:.0%})")
print(f"method   : {result.method}")
print(f"schema   : {len(schema.fields)} fields, "
      f"{len(schema.required_fields)} required, {len(schema.critical_fields)} critical")
print("           " + ", ".join(spec.name for spec in schema.fields))
''')

# ------------------------------------------------------------- extract A ----
md(r"""
## Stage 4 — Reading 1: Typhoon, field-first

The model is given the page text and the field list, and asked for each field
**with the span of text it read the value from**. That second part is the whole
trick: a value without a quote cannot be checked, and a quote that is not on the
page is a fabrication we can detect.
""")

code(r'''
import time
from docuflow.agents.llm_extractor import extract_with_llm

started = time.perf_counter()
primary_read = extract_with_llm(ocr.full_text, schema, primary)
print(f"{getattr(primary, 'model', primary)} — {time.perf_counter() - started:.1f}s, "
      f"ok={primary_read.ok} {primary_read.note}\n")

print(f"{'FIELD':22} {'VALUE':30} EVIDENCE IT QUOTED")
for name, candidate in sorted(primary_read.candidates.items()):
    print(f"{name:22} {str(candidate.raw_value)[:29]:30} "
          f"{(candidate.evidence or '')[:44]!r}")
''')

# ------------------------------------------------------------- extract B ----
md(r"""
## Stage 5 — Reading 2: Claude, label-first

A different model, given a different instruction: walk the labels printed on the
page and report what sits beside each one. It never sees the first reading, so it
cannot be anchored by it.

Two independent readings that agree are worth far more than one confident one —
and where they disagree, the system now knows exactly which fields to distrust.
""")

code(r'''
from docuflow.agents.verify_extractor import verify_with_llm
from docuflow.agents.rule_extractor import extract_with_rules

started = time.perf_counter()
secondary_read = verify_with_llm(ocr.full_text, schema, verify)
elapsed = time.perf_counter() - started

if secondary_read.ok:
    channel = getattr(verify, "model", None) or getattr(verify, "name", "model")
    print(f"{channel} — {elapsed:.1f}s\n")
    print(f"{'FIELD':22} {'VALUE':30} EVIDENCE IT QUOTED")
    for name, candidate in sorted(secondary_read.candidates.items()):
        print(f"{name:22} {str(candidate.raw_value)[:29]:30} "
              f"{(candidate.evidence or '')[:44]!r}")
else:
    print(f"no second model ({secondary_read.note})")
    print("falling back to the deterministic label matcher\n")
    for name, candidate in sorted(extract_with_rules(
            [(p.page_number, p.text) for p in ocr.pages], schema).items()):
        print(f"{name:22} {str(candidate.raw_value)[:29]:30}")
''')

# ----------------------------------------------------------------- merge ----
md(r"""
## Stage 6 — Merge, ground, score

This is where it becomes usable. For every field:

1. **Do the two channels agree?** Compared after normalisation, so `1,090.00` and
   `1090` are the same number and `31/07/2026` and `2026-07-31` are the same day.
2. **Is the value on the page?** The quoted evidence is fuzzy-matched against the
   OCR text (≥88 similarity), and then checked type-aware: does that span
   actually *state* this value, or merely sit near it?
3. **What confidence does that earn?** Agreement and grounding raise it, a
   degraded page lowers it, a single ungrounded channel lowers it a lot.

Read the table below by the `GROUNDED` column. An ungrounded critical field never
auto-posts, no matter how good the number looks.
""")

code(r'''
from docuflow.agents.candidates import from_llm, from_rule
from docuflow.agents.merge import merge_all

pairs = [(p.page_number, p.text) for p in ocr.pages]

channel_a = {n: from_llm(n, c.raw_value, c.evidence, channel="model")
             for n, c in primary_read.candidates.items()}
if secondary_read.ok:
    channel_b = {n: from_llm(n, c.raw_value, c.evidence, channel="verify")
                 for n, c in secondary_read.candidates.items()}
else:
    channel_b = {n: from_rule(c) for n, c in extract_with_rules(pairs, schema).items()}

fields = merge_all(schema, channel_a, channel_b, pairs, quality.score, ocr.degraded)

# The verdict is read off the merged field, not recomputed here: the merge
# compares *normalised* values, so "8 July 2026" and "2026-07-08" are the same
# day and "187,000.00 THB" is the same number as "187000.0". Comparing the raw
# strings would report a disagreement the system correctly did not have.
def channels(field):
    if any("the two readings disagree" in note for note in field.notes):
        return "DIFFERED"
    both = field.name in channel_a and field.name in channel_b
    return "both agreed" if both else "one channel"

print(f"{'FIELD':20} {'VALUE':26} {'CONF':>5} {'GROUNDED':>9}  CHANNELS")
for field in fields:
    if field.is_empty:
        continue
    print(f"{field.name:20} {str(field.value)[:25]:26} {field.confidence:>4.0%} "
          f"{('yes' if field.grounded else 'NO'):>9}  {channels(field)}")

missing = [f.name for f in fields if f.is_empty]
if missing:
    print("\nnot found on the page:", ", ".join(missing))
''')

code(r'''
# The evidence, for anything the page did not support. This is the list a
# reviewer actually needs: not "the model was 91% sure", but "here is the line
# it claims to have read, and it is not on the page".
ungrounded = [f for f in fields if not f.is_empty and not f.grounded]
if not ungrounded:
    print("every populated field is traceable to the page text")
for field in ungrounded:
    print(f"{field.name:20} = {field.value!r}")
    print(f"{'':20}   claimed: {(field.evidence or '')[:70]!r}")
    print(f"{'':20}   notes  : {', '.join(field.notes) or '-'}")
''')

# ----------------------------------------------------------------- rules ----
md(r"""
## Stage 7 — Business rules

No model runs here at all. Twenty-six deterministic checks: arithmetic recomputed,
counterparties looked up in the master file, references checked against what was
already posted, dates ordered.

This is on purpose. A model asked "is this VAT correct?" can be talked out of the
right answer by the document's own wording — which is exactly what a careless
supplier or an attacker controls. Code cannot be talked out of anything.
""")

code(r'''
from datetime import date
from docuflow.rules import RuleContext, run_rules
from docuflow.masterdata.repository import MasterDataRepository
from docuflow.store.records import RecordStore, reference_field_for

master = MasterDataRepository(settings.db_path)
records = RecordStore(settings.db_path)

by_name = {f.name: f for f in fields}
def value(name):
    field = by_name.get(name)
    return None if field is None or field.is_empty else field.value

vendor_name = value("vendor_name") or value("merchant_name") or value("applicant_name")
tax_id = value("vendor_tax_id") or value("merchant_tax_id") or value("tax_id")
vendor_match = master.find_vendor(str(vendor_name) if vendor_name else None,
                                  str(tax_id) if tax_id else None)
reference = value(reference_field_for(result.doc_type))

context = RuleContext(
    document_id="NOTEBOOK-1",
    doc_type=result.doc_type,
    schema=schema,
    fields=fields,
    line_items=(),
    master=master,
    vendor_match=vendor_match,
    ocr_text=ocr.full_text,
    ocr_degraded=ocr.degraded,
    today=date.today(),
    duplicate_of=records.find_duplicate(result.doc_type, str(reference) if reference else None,
                                        vendor_match.vendor_id),
    **RuleContext.policy_from(settings),
)

findings = run_rules(context)
failed = [f for f in findings if not f.passed]

print(f"vendor lookup: {vendor_match.method}"
      + (f" -> {vendor_match.vendor['legal_name']}" if vendor_match.vendor else " -> no match"))
print(f"{len(findings) - len(failed)}/{len(findings)} checks passed\n")

for finding in failed:
    print(f"[{finding.severity.upper():8}] {finding.rule_id}")
    print(f"{'':11} {finding.message}")
if not failed:
    print("nothing failed")
''')

# ----------------------------------------------------------------- route ----
md(r"""
## Stage 8 — Route

Straight-through processing requires **all** of: no blocking failure, confidence
above the threshold, every required field present, and every critical field
grounded. Anything less goes to a person.

The policy is deliberately timid. A document sent for review costs a minute of
someone's attention; a wrong value posted to the ledger costs considerably more,
and nobody finds out until reconciliation.
""")

code(r'''
from docuflow.confidence import document_confidence
from docuflow.routing import decide
from docuflow.models import DocumentRun, SourceInfo

run = DocumentRun(
    document_id="NOTEBOOK-1",
    source=SourceInfo(filename=document_path.name, sha256="0" * 64, byte_size=1),
    doc_type=result.doc_type,
    ocr=ocr,
    fields=fields,
    findings=findings,
)
run = run.evolve(confidence=document_confidence(
    fields, findings,
    frozenset(s.name for s in schema.required_fields),
    frozenset(s.name for s in schema.critical_fields),
))

outcome = decide(run, schema, settings.auto_post_min_confidence)

print(f"confidence : {run.confidence:.0%}   (threshold {settings.auto_post_min_confidence:.0%})")
print(f"decision   : {outcome.decision.upper()}\n")
for reason in outcome.reasons:
    print(" •", reason)
if not outcome.reasons:
    print(" • nothing stood in the way")
''')

# ------------------------------------------------------- rule authoring ----
md(r"""
## Writing a rule in plain language

Type a check in English. A model turns it into an expression **once**, here, in
front of you — and is then never consulted about it again. From that moment the
rule is data, evaluated by plain code on every document.

That boundary is the point. Asking a model on every document lets the document
influence the answer. Asking it once, in front of the person who approves the
result, does not.

Note what comes back: the compiled condition, and the same thing **read back in
English**. Approve the sentence, not the JSON.
""")

code(r'''
from docuflow.rules.authoring import compile_rule

PROMPT = "invoices over 100,000 baht must quote a purchase order"

draft = compile_rule(PROMPT, ("invoice",), primary, mode="expression")

if not draft.ok:
    print("could not compile:", draft.error)
else:
    rule = draft.rule
    print("you typed :", PROMPT)
    print("rule id   :", rule.rule_id, f"({rule.severity})")
    print("\nreads as  :", rule.explanation)
    print("\nreviewers see on failure:")
    print("   ", rule.message)
    print("\ncompiled to:")
    import json as _json
    print(_json.dumps({"when": rule.when, "assert": rule.assert_}, indent=2))
''')

md(r"""
### Some requests are not comparisons

"Invoices over 100,000 need user review" names **no condition to check** —
qualifying *is* the finding. Given only a slot marked `assert`, a model will
invent something to put in it, and what it invents is unrelated to what you
asked. So a rule may carry a guard and no assertion: it flags every document the
guard matches.

And when a check genuinely needs a lookup and a judgement about what comes back —
"only pay this if the purchase order still covers it" — it compiles to an **agent
rule** instead: a fixed plan, a chosen set of tools, and a verdict it cannot
escalate. A verdict it cannot reach holds the document; it is never a pass.
""")

code(r'''
flagging = compile_rule("invoices over 100,000 baht need a person to look at them",
                        ("invoice",), primary, mode="expression")
if flagging.ok:
    print("flag-only rule :", flagging.rule.explanation)
    print("assert         :", flagging.rule.assert_, "(nothing to compare — matching is the finding)")

agentic = compile_rule(
    "check the purchase order this invoice quotes actually covers it; "
    "purchase orders are recorded net of VAT, so compare against the subtotal",
    ("invoice",), primary, mode="agent")
if agentic.ok:
    print("\nagent rule     :", agentic.rule.rule_id)
    print("instruction    :", agentic.rule.instruction)
    print("plan           :")
    for index, step in enumerate(agentic.rule.plan, 1):
        print(f"   {index}. {step}")
    print("tools          :", ", ".join(agentic.rule.tools))
else:
    print("\nagent rule could not be compiled:", agentic.error)
''')

# ------------------------------------------------------- full pipeline ----
md(r"""
## The whole thing in one call

Everything above, as the system actually runs it — plus persistence, the
discrepancy analyst, and the trace.
""")

code(r'''
from docuflow.pipeline import Pipeline
from docuflow.store.custom_rules import CustomRuleStore

if draft.ok:
    CustomRuleStore(settings.db_path).save(draft.rule, replace=True)
    print("saved your rule:", draft.rule.rule_id)

outcome = Pipeline(settings).process_file(document_path, source_kind="path")
processed = outcome.run

print(f"\ndocument   : {processed.document_id}")
print(f"type       : {processed.doc_type}")
print(f"decision   : {processed.decision.upper()}  ({processed.confidence:.0%})")
print(f"record     : {processed.record_id or '- not posted'}")

print(f"\n{'STAGE':12} {'STATUS':>8} {'TIME':>8}  SUMMARY")
for step in processed.trace:
    print(f"{step.step:12} {step.status:>8} {step.duration_ms / 1000:>7.1f}s  {step.summary[:60]}")
''')

code(r'''
# The extracted record — the thing this whole notebook exists to produce.
print(f"{'FIELD':22} {'VALUE':32} {'CONF':>5}  GROUNDED")
for field in processed.fields:
    if field.is_empty:
        continue
    print(f"{field.name:22} {str(field.value)[:31]:32} {field.confidence:>4.0%}  "
          + ("yes" if field.grounded else "NO"))

print("\nwhy it was routed that way:")
for reason in processed.decision_reasons:
    print(" •", reason)
''')

code(r'''
# What the discrepancy analyst made of it, if it ran. It only runs when a check
# failed *and* a person will see the document - it writes a note for a reviewer,
# and an auto-posted document has none.
rec = outcome.reconciliation
if not rec.ran:
    print("analyst did not run:", rec.note)
else:
    print("assessment :", rec.assessment)
    print("root cause :", rec.root_cause)
    print("suggests   :", rec.recommended_action)
    print("tools used :", ", ".join(rec.tools_used) or "-")
    for correction in rec.corrections:
        mark = "verified on page" if correction.grounded else (
            "THE PAGE SAYS OTHERWISE" if correction.contradicts_evidence else "not on page")
        print(f"   [{mark}] {correction.field} -> {correction.suggested_value}")
    if rec.unreliable:
        print("\n   ! nothing it proposed held up against the page — read the note "
              "above as a suggestion to check, not as a finding")
''')

# ------------------------------------------------------------- re-check ----
md(r"""
## Re-checking under a policy that changed

A held document is held by **the policy in force when it was read**. Change a
threshold or switch a rule off, and the queue is still full of decisions made
under the old one.

`reevaluate` judges a stored document again — the page is not read again, no OCR
and no model call. It re-runs the rules and the routing against the current policy
and reports what changed.

For this we need a document that is actually **held**, so here is one whose vendor
is not in the master file.
""")

code(r'''
from docuflow.pipeline import Pipeline

held_outcome = Pipeline(settings).process_file(
    settings.sample_dir / "inv-unknown-vendor.pdf", source_kind="sample")
held = held_outcome.run

print(f"{held.document_id}  {held.decision.upper()}  ({held.confidence:.0%})")
for finding in held.findings:
    if not finding.passed:
        print(f"   [{finding.severity:8}] {finding.rule_id}: {finding.message[:66]}")
''')

md(r"""
Some of those failures say *this record is not in our master file* — a fact about
the demo register, not about the page. A deployment loading its own vendor list
would not see them.

Switch those off, along with the rule you wrote a moment ago, and judge the
document again. Nothing is re-read: no OCR, no model call.

Read the `STILL FAILING` column rather than hoping for a flip. Whether the
decision changes depends on what is left, and what is left is the honest answer —
a missing purchase order is a fact about the page, and no amount of policy
switching makes it go away.
""")

code(r'''
from docuflow.policy import PolicyStore
from docuflow.revalidate import reevaluate

switched_off = ["vendor_known", "po_exists", "contract_matches_master"]
PolicyStore(settings).update(disabled_rules=switched_off)

# A rule you wrote lives in its own store, so it is switched off there. The
# policy only knows about the built-ins - it refuses an id it does not recognise
# rather than silently accepting a typo that would disable nothing.
if draft.ok:
    CustomRuleStore(settings.db_path).set_enabled(draft.rule.rule_id, False)
    switched_off.append(draft.rule.rule_id)

print("switched off:", ", ".join(switched_off), "\n")

again = reevaluate(PolicyStore(settings).settings(), held.document_id)

print(f"{'':8} {'DECISION':>14} {'CONF':>6}  STILL FAILING")
for label, snapshot in (("before", again.before), ("after", again.after)):
    print(f"{label:8} {snapshot['decision']:>14} {snapshot['confidence']:>5.0%}  "
          + (", ".join(snapshot["failing"]) or "-"))
print(f"\nchanged : {again.changed}")
print(f"          {again.message}")
''')

md(r"""
### What re-checking refuses to do

A posted document is never re-judged. The ledger entry points at it, and a
correction downstream is a credit note, not a quiet change of mind about a record
that already left the building.

The other refusal you cannot see here: a document a **person** rejected never
re-posts itself because a rule moved. A policy change can release something the
system held; it cannot overturn somebody's decision.
""")

code(r'''
from docuflow.revalidate import RevalidationError

try:
    reevaluate(PolicyStore(settings).settings(), processed.document_id)
    print("re-checked")
except RevalidationError as refusal:
    print("refused, correctly:")
    print("   ", refusal)
''')

code(r'''
# Put everything back, so re-running the notebook starts from the same place.
PolicyStore(settings).update(disabled_rules=[])
if draft.ok:
    CustomRuleStore(settings.db_path).set_enabled(draft.rule.rule_id, True)
print("policy restored")
''')

# ----------------------------------------------------------------- close ----
md(r"""
## What you just ran

| | |
|---|---|
| **2 model calls per document** | Typhoon reads it, Claude reads it again, independently |
| **0 model calls in the decision** | rules, grounding, confidence and routing are all code |
| **Every value carries a quote** | and the quote is checked against the page, not trusted |
| **The model that writes a rule never runs it** | it compiles English to an expression once, in front of you |

### Where the time goes

On this corpus, per document: OCR ~40%, the two readings ~50%, everything else
~10%. If you need it faster, that is the order to look in — and
`docuflow evaluate` prints both the timings and the accuracy side by side, so you
can tell an optimisation from a regression.

### Running it as a service instead

```bash
docuflow serve                 # the web UI: upload, review, post, write rules
docuflow evaluate              # accuracy + cost against the corpus ground truth
docuflow master template --type vendors   # the CSV columns for your own master data
```

### The honest limits

* Two readings agreeing is strong evidence, not proof. Both can misread the same
  badly-printed digit.
* The rules encode a generic finance policy. Yours is different — switch the
  built-ins off and keep your own, from the Settings screen.
* An agent rule keeps a model in the loop at validation time. It is fenced in by a
  deterministic guard, a fixed plan and a verdict it cannot escalate, but it is
  still the least predictable part of the system. Prefer a plain condition
  wherever one will do.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "docuflow_walkthrough.ipynb", "provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {OUT}  ({len(cells)} cells)")
