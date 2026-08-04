"""Agent rules: the checks that cannot be written as a comparison.

An expression rule is a sentence about fields - "the total is at most 100,000",
"a purchase order is quoted". Most finance policy is that shape, and where it is,
that shape is strictly better: it costs nothing, it is repeatable, and the person
who wrote it can read it back in English before it ever runs.

Some policy is not that shape. "Only pay this invoice if the purchase order still
has budget left for it" needs a lookup, then arithmetic against what came back,
then a judgement about what the difference means. There is no field to compare;
there is a small investigation to carry out. Written as an expression it either
cannot be encoded at all, or gets encoded as something adjacent and wrong.

An agent rule keeps the investigation. It carries a plan - written once, when the
rule is authored, and shown to the author like the compiled condition is - and at
validation time an agent works through that plan with tools and returns a verdict
on this one document.

Three things keep that from becoming "ask a model whether the document is fine":

  * the plan is fixed at authoring time, so every document is checked the same
    way, and the author approved the way;
  * the guard (`when`) is still deterministic, so the agent only ever sees the
    documents the rule is about;
  * a verdict it cannot reach holds the document for a person. It is never a pass.
"""

from __future__ import annotations

from typing import Any

MAX_PLAN_STEPS = 6
MAX_STEP_CHARS = 200
MAX_INSTRUCTION_CHARS = 600

RULE_KINDS = frozenset({"expression", "agent"})

# Verdicts an agent may return. `unsure` is not a failure of the document, it is
# a failure to decide - kept separate because they read differently to a
# reviewer and because only one of them should ever be graded as blocking.
VERDICTS = frozenset({"pass", "fail", "unsure"})


class AgentRuleError(ValueError):
    """The agent rule cannot be stored as written."""


def clean_plan(raw: Any) -> tuple[str, ...]:
    """Normalise the plan: short, non-empty steps, bounded in number."""
    if isinstance(raw, str):
        raw = [line for line in raw.splitlines()]
    if not isinstance(raw, (list, tuple)):
        return ()
    steps: list[str] = []
    for entry in raw:
        # Models sometimes return {"step": "...", "tool": "..."} instead of text.
        text = entry.get("step", "") if isinstance(entry, dict) else entry
        text = str(text or "").strip().lstrip("-•0123456789. ").strip()
        if text:
            steps.append(text[:MAX_STEP_CHARS])
    return tuple(steps[:MAX_PLAN_STEPS])


def validate_agent_payload(payload: dict[str, Any], known_tools: frozenset[str]) -> dict[str, Any]:
    """Check the agent-specific half of a rule and return it normalised."""
    instruction = str(payload.get("instruction") or "").strip()
    if not instruction:
        raise AgentRuleError(
            "An agent rule needs an instruction: what the agent has to decide "
            "about each document."
        )

    plan = clean_plan(payload.get("plan"))
    if not plan:
        raise AgentRuleError(
            "An agent rule needs a plan - the steps to work through - so the same "
            "check runs on every document and you can read it before saving."
        )

    wanted = payload.get("tools") or ()
    if isinstance(wanted, str):
        wanted = [name.strip() for name in wanted.split(",")]
    tools = tuple(dict.fromkeys(str(name).strip() for name in wanted if str(name).strip()))
    unknown = [name for name in tools if name not in known_tools]
    if unknown:
        raise AgentRuleError(
            f"Unknown tool(s): {', '.join(unknown)}. Available: {', '.join(sorted(known_tools))}."
        )

    if payload.get("assert") or payload.get("assert_"):
        raise AgentRuleError(
            "An agent rule decides for itself; it must not also carry an `assert` "
            "condition. Use `when` to say which documents it looks at."
        )

    return {
        "instruction": instruction[:MAX_INSTRUCTION_CHARS],
        "plan": plan,
        "tools": tools,
    }


def agent_explanation(instruction: str, plan: tuple[str, ...], described_guard: str) -> str:
    """The rule in words, for the author to approve before it is saved."""
    scope = f"On documents where {described_guard}, an" if described_guard else "An"
    steps = "; ".join(f"{index}. {step}" for index, step in enumerate(plan, start=1))
    return f"{scope} agent decides: {instruction} It works through: {steps}"
