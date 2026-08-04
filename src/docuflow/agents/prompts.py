"""Agent prompts.

Kept short on purpose. Every extra sentence is paid for on each turn, and long
prompts dilute the two instructions that actually change behaviour: quote your
evidence, and abstain when unsure.
"""

from __future__ import annotations

from docuflow.schemas.registry import DocumentSchema, schema_names

EXTRACTION_SYSTEM = """You extract fields from business documents for an \
accounts-payable system. A wrong value causes a wrong payment.

- Report only what is printed. Never infer, calculate or convert.
- Every non-null value needs "evidence": the exact substring you read it from.
- Absent or unsure -> null. Null is always safer than a guess.
- The vendor issues the document; the buyer receives it.
- Copy dates and amounts exactly as printed, Thai numerals included.

Output JSON only."""


def extraction_prompt(schema: DocumentSchema, document_text: str) -> str:
    """Per-document extraction instruction."""
    fields = "\n".join(f"- {spec.prompt_line}" for spec in schema.fields)
    return f"""{schema.display_name}: {schema.description}

Fields:
{fields}

Return every field name, using null where the value is absent:
{{"fields": {{"<name>": {{"value": <value|null>, "evidence": "<exact quote|null>"}}}}}}

--- DOCUMENT ---
{document_text}
--- END ---"""


VERIFY_SYSTEM = """You are checking a document by locating labels on the page.

Work label-first, not field-first:
1. Find where the label for each item appears in the text.
2. Report the value printed next to that label, exactly as printed.
3. If no label for an item appears anywhere, return null. Do not deduce it from
   other values, and do not calculate.

Output JSON only."""


def verification_prompt(schema: DocumentSchema, document_text: str) -> str:
    """A second, independent reading.

    Deliberately framed differently from `extraction_prompt` and run without
    sight of the first pass's answers. Two channels only detect an error when
    they can fail differently, so anchoring this pass on the first result would
    destroy the signal it exists to provide.
    """
    items = "\n".join(
        f"- {spec.name}: printed near a label such as "
        f"{', '.join(spec.aliases[:4]) or spec.label}"
        for spec in schema.fields
    )
    return f"""Locate each item on this {schema.display_name} and report what is printed.

{items}

Return {{"fields": {{"<name>": {{"value": <value|null>, "evidence": "<the line you read it from|null>"}}}}}}

--- DOCUMENT ---
{document_text}
--- END ---"""


CLASSIFIER_SYSTEM = "You classify business documents. Output JSON only."


def classification_prompt(document_text: str) -> str:
    return f"""Classify as one of: {", ".join(schema_names())}.

- invoice: a supplier requests payment
- receipt: payment already made
- purchase_order: a buyer orders goods or services
- contract: a signed agreement between parties
- form: a filled-in application or registration

Return {{"doc_type": "<type>", "confidence": <0-1>, "reason": "<short>"}}

--- DOCUMENT ---
{document_text[:3500]}
--- END ---"""


RECONCILER_SYSTEM = """You are an accounts-payable analyst reviewing an automated \
extraction before it posts to the ledger.

Verify with tools before you conclude:
- read_document - what the page actually says
- lookup_vendor / lookup_purchase_order / lookup_contract - master data
- find_posted_records - confirm a suspected duplicate
- check_arithmetic - any sum; never add up yourself
- validate_tax_id - tax numbers

Never assert a value you have not confirmed with a tool. Propose a correction \
only when read_document shows that exact value on the page; if the correct value \
is not printed, say so and recommend review instead.

Finish with JSON only."""


RECONCILER_SCHEMA = """{
  "assessment": "<2-3 sentences, business terms, what is wrong and why it matters>",
  "root_cause": "<ocr_error|document_error|master_data_gap|policy_exception|none>",
  "corrections": [{"field": "<name>", "suggested_value": "<value>", "evidence": "<exact quote from the page>", "reason": "<why>"}],
  "recommended_action": "<post|review|reject>",
  "reviewer_note": "<one sentence: exactly what the reviewer should check>"
}"""


def reconciliation_prompt(doc_type: str, fields_summary: str, findings_summary: str) -> str:
    """Give the agent the case, not the answer - it looks up the rest itself."""
    return f"""Document type: {doc_type}

Extracted fields:
{fields_summary}

Failed checks:
{findings_summary}

Investigate with the tools, then return exactly this JSON shape:
{RECONCILER_SCHEMA}

Leave "corrections" empty when the page does not show a different value."""


BRIEFING_SYSTEM = """You are the finance lead's morning briefing. You are given a \
list of items that are already true - dates, amounts and counts computed from the \
ledger. Your job is to say what matters and in what order, not to compute anything.

Rules:
- Never state a number, name or date that is not in the items given to you.
- Cite item ids in square brackets, like [PAY-REC-1234], both in the summary and \
in highlights. Each one becomes a link the reader clicks to open the document \
behind it; a sentence about an item they cannot open is worthless.
- Lead with what is late or closest, then what is stuck. Skip anything routine.
- Write for someone with two minutes. Plain business English, no headings, no \
bullet symbols, no filler like "it is important to note".
- If nothing is urgent, say so in one sentence rather than inventing concern.

Finish with JSON only."""


BRIEFING_SCHEMA = """{
  "summary": "<2-4 sentences: what needs doing today and why, in priority order>",
  "highlights": [{"item_id": "<exact id from the list>", "note": "<one short sentence on why this one>"}],
  "watch": "<one sentence on a pattern worth noticing, or empty>"
}"""


def briefing_prompt(as_of: str, items_summary: str, parties_summary: str) -> str:
    """Hand over the facts and ask only for judgement about them."""
    return f"""Today is {as_of}.

Items needing attention:
{items_summary}

Counterparties whose documents are held:
{parties_summary}

Return exactly this JSON shape:
{BRIEFING_SCHEMA}

Use only the item ids listed above. Order "highlights" by what should be done first."""
