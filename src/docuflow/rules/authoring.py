"""Turn a sentence into a rule.

Most checks compile to an expression. The model is used exactly once, here, to
translate what someone typed into that expression. It never sees a document and
it is never consulted again: the compiled rule is shown back to the author in
words, saved as data, and from then on evaluated by `expressions.evaluate`.

That boundary is worth keeping wherever it fits. Asking a model on every document
whether the VAT is right makes the answer depend on the document's own wording,
which is precisely what an attacker or a sloppy supplier controls. Asking it
once, in front of the person who will approve the result, does not.

Not every check fits. "Only pay this if the PO still has budget for it" is a
lookup, then arithmetic on what came back, then a judgement - there is no field
to compare. Forcing that into an expression produces something adjacent and
wrong, which is a worse failure than admitting the shape does not fit. Those
compile to an agent rule instead: the plan is written here, once, and approved
here, once, and the agent that works through it on each document is fenced in by
a deterministic guard, a fixed toolbox and a verdict it cannot escalate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from docuflow.llm.base import ChatMessage, LlmProvider, extract_json
from docuflow.rules.custom import (
    DOCUMENT_FACTS,
    CustomRule,
    CustomRuleError,
    build_rule,
    fields_for,
)
from docuflow.schemas.registry import DOCUMENT_SCHEMAS, get_schema

logger = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 1_200
MAX_ATTEMPTS = 2

# What the author asked the compiler to produce.
MODES = frozenset({"auto", "expression", "agent"})

AUTHORING_SYSTEM = """You convert a finance team's plain-English check into one JSON rule.
Reply with JSON only.

{"rule_id":"snake_case","title":"short","message":"what a reviewer reads when it fails",
 "doc_types":["invoice"],"severity":"blocking|warning|info",
 "when":<condition or null>,"assert":<condition>,"fields":["field_used"]}

A condition is one of:
{"all":[c,...]} {"any":[c,...]} {"not":c}
{"field":"name","op":"OP","value":X}
{"field":"name","op":"OP","other_field":"name"}

OP: eq ne gt gte lt lte in not_in contains not_contains starts_with ends_with present absent

Rules:
- "assert" is what must be TRUE for a document to PASS. When the request
  describes when to WARN or REJECT, put the OPPOSITE in "assert":
    "warn when the tax id is missing"  -> "assert":{"field":"tax_id","op":"present"}
    "reject invoices over 1,000,000"   -> "assert":{"field":"total_amount","op":"lte","value":1000000}
- "when" limits which documents the rule applies to; null means all of them.
  "invoices over 50,000 need a PO" is a rule that applies only over 50,000:
    "when":{"field":"total_amount","op":"gt","value":50000},
    "assert":{"field":"purchase_order_ref","op":"present"}
- Some requests name no condition to check at all - they ask for a document to
  be held so a person sees it. Give those a "when" and set "assert" to null:
    "invoices over 100,000 need user review" ->
      "when":{"field":"total_amount","op":"gt","value":100000},"assert":null
  Never invent something to assert. If the request does not say what must be
  true, it is asking to flag, not to check.
- Encode only what was asked. Add no extra condition, exception or escape clause.
- Use only the field names listed. Never invent one.
- For a true/false field, compare it: {"field":"vendor_matched","op":"eq","value":true}.
  Never use present/absent on one - False is still a value, so the test never fails.
- duplicate_found and ocr_degraded report a problem. Never assert one must be
  true: that would make a document have to be broken in order to pass.
- severity "blocking" stops the document posting; "warning" lets it through flagged."""

# Appended in "auto" mode: the way out when the request is not a comparison.
# Without it the model has to force every request into a condition, and what it
# invents when it cannot is unrelated to what was asked.
AGENT_OPTION = """

If - and only if - the check cannot be written as a condition over these fields,
because it needs a lookup, a follow-up on what comes back, or a judgement about
the difference, return an agent rule instead:

{"kind":"agent","rule_id":"snake_case","title":"short",
 "message":"what a reviewer reads when it fails","doc_types":["invoice"],
 "severity":"blocking|warning|info","when":<condition or null>,
 "instruction":"what the agent must decide about each document, in 1-2 sentences",
 "plan":["step","step"],"tools":["tool_name"]}

Tools the agent may use:
{tools}

- The plan is 2 to 5 steps. Each step is one thing to look up or compare, in
  order. Plain sentences, no code.
- List only the tools the plan actually uses.
- "when" is still a plain condition and is still checked without a model. Make it
  as narrow as the request honestly allows: it decides how often the agent runs.
- Prefer a plain condition wherever one will do. It costs nothing, gives the same
  answer every time, and needs no model on the day a document arrives."""

AGENT_SYSTEM = """You convert a finance team's plain-English check into one agent rule:
a short plan that an agent works through against each document, with tools.
Reply with JSON only.

{"kind":"agent","rule_id":"snake_case","title":"short",
 "message":"what a reviewer reads when it fails","doc_types":["invoice"],
 "severity":"blocking|warning|info","when":<condition or null>,
 "instruction":"what the agent must decide about each document, in 1-2 sentences",
 "plan":["step","step"],"tools":["tool_name"]}

Tools the agent may use:
{tools}

"when" decides which documents the agent looks at, and is evaluated without a
model. null means every document of these types. It is one of:
{"all":[c,...]} {"any":[c,...]} {"not":c}
{"field":"name","op":"OP","value":X}
{"field":"name","op":"OP","other_field":"name"}
OP: eq ne gt gte lt lte in not_in contains not_contains starts_with ends_with present absent

Rules:
- The plan is 2 to 5 steps. Each step is one thing to look up or compare, in
  order. Plain sentences, no code.
- List only the tools the plan actually uses. Use only the field names given.
- Encode only what was asked. Add no extra condition or escape clause.
- Make "when" as narrow as the request honestly allows: it decides how often the
  agent runs at all.
- severity "blocking" stops the document posting; "warning" lets it through flagged."""


def _tool_catalogue() -> str:
    from docuflow.agents.tools import TOOL_HELP

    return "\n".join(f"- {name}: {help_text}" for name, help_text in TOOL_HELP.items())


def system_prompt(mode: str) -> str:
    """The instructions for one authoring mode.

    `expression` is the original prompt untouched: someone who asked for a
    deterministic check gets offered nothing else.
    """
    if mode == "agent":
        return AGENT_SYSTEM.replace("{tools}", _tool_catalogue())
    if mode == "auto":
        return AUTHORING_SYSTEM + AGENT_OPTION.replace("{tools}", _tool_catalogue())
    return AUTHORING_SYSTEM


@dataclass(frozen=True)
class AuthoringResult:
    """A compiled rule, or the reason it could not be compiled."""

    rule: CustomRule | None
    error: str = ""
    raw: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.rule is not None


def available_fields(doc_types: tuple[str, ...]) -> list[dict[str, str]]:
    """The variables a rule may use, with a description for the prompt and UI."""
    covered = set(DOCUMENT_SCHEMAS) if "*" in doc_types else set(doc_types)
    shared = frozenset.intersection(*(fields_for(name) for name in covered)) if covered else frozenset()

    described: dict[str, str] = {}
    for doc_type in sorted(covered):
        for spec in get_schema(doc_type).fields:
            if spec.name in shared and spec.name not in described:
                described[spec.name] = f"{spec.label} ({spec.type})"
    for name, description in DOCUMENT_FACTS.items():
        if name in shared:
            described[name] = description
    return [{"name": name, "description": described[name]} for name in sorted(described)]


def _field_catalogue(doc_types: tuple[str, ...]) -> str:
    return "\n".join(f"- {row['name']}: {row['description']}" for row in available_fields(doc_types))


def compile_rule(
    prompt: str, doc_types: tuple[str, ...], llm: LlmProvider | None, mode: str = "auto"
) -> AuthoringResult:
    """Compile `prompt` into a rule for `doc_types`. Never raises.

    `mode` is the author's choice of machinery: `expression` for a deterministic
    condition, `agent` for a plan run with tools, `auto` to let the compiler pick
    the deterministic one wherever it fits.
    """
    if mode not in MODES:
        return AuthoringResult(None, f"'{mode}' is not an authoring mode. Use {', '.join(sorted(MODES))}.")
    if llm is None:
        return AuthoringResult(None, "No agent model is configured, so rules cannot be written from a description.")
    if not prompt.strip():
        return AuthoringResult(None, "Describe the check you want in a sentence.")
    if not doc_types:
        return AuthoringResult(None, "Choose which document types the rule applies to.")

    request = (
        f"Document types: {', '.join(doc_types)}\n\n"
        f"Fields available:\n{_field_catalogue(doc_types)}\n\n"
        f"The check to encode:\n{prompt.strip()}"
    )

    messages = [
        ChatMessage(role="system", content=system_prompt(mode)),
        ChatMessage(role="user", content=request),
    ]

    last_error = ""
    for attempt in range(MAX_ATTEMPTS):
        try:
            reply = llm.complete(messages, max_tokens=MAX_OUTPUT_TOKENS, temperature=0.0)
        except Exception as exc:  # noqa: BLE001 - reported, never raised at the user
            logger.warning("rule authoring call failed: %s", exc)
            return AuthoringResult(None, f"The model could not be reached: {exc}")

        payload = extract_json(reply)
        if payload is None:
            last_error = "The model did not return valid JSON."
        else:
            payload.setdefault("doc_types", list(doc_types))
            # The author's choice of machinery is the author's, not the model's.
            if mode != "auto":
                payload["kind"] = "agent" if mode == "agent" else "expression"
            try:
                return AuthoringResult(build_rule(payload, prompt=prompt.strip()), raw=payload)
            except CustomRuleError as exc:
                last_error = str(exc)

        if attempt + 1 < MAX_ATTEMPTS:
            # Hand the validator's own complaint back: it names the exact field
            # or operator that does not exist, which is what the model needs.
            messages = [
                *messages,
                ChatMessage(role="assistant", content=reply),
                ChatMessage(
                    role="user",
                    content=f"That was rejected: {last_error}\nReturn corrected JSON only.",
                ),
            ]

    return AuthoringResult(None, last_error or "The rule could not be compiled.")
