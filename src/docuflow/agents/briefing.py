"""A model writes the covering note. It never supplies the facts.

The agenda is computed from the ledger before the model is called, and the model
receives it as a finished list. It decides what to lead with, what to group and
what to leave out - judgement, which is what it is good at - and it is given no
opportunity to produce a number, because every number is already in the list.

Anything it says is then checked against that list. An item id it did not
receive is dropped rather than shown, on the same reasoning as evidence
grounding one level down: a claim nobody can open is not a claim, it is a
rumour. A reader who clicks through to the invoice is the whole point.

Without a model the page still works. The deterministic summary below is duller
and says the same thing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from docuflow.agents.agent_loop import run_agent
from docuflow.agents.prompts import BRIEFING_SYSTEM, briefing_prompt
from docuflow.llm.base import LlmProvider
from docuflow.store.attention import Agenda, AttentionItem, StuckParty

logger = logging.getLogger(__name__)

MAX_TOKENS = 1200
MAX_HIGHLIGHTS = 6
# More than this and the model is being asked to skim rather than to prioritise.
MAX_ITEMS_IN_PROMPT = 25


@dataclass(frozen=True)
class Highlight:
    item_id: str
    note: str

    def to_json(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "note": self.note}


@dataclass(frozen=True)
class Briefing:
    summary: str
    highlights: tuple[Highlight, ...] = ()
    watch: str = ""
    written_by: str = "rules"
    note: str = ""
    dropped: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "highlights": [highlight.to_json() for highlight in self.highlights],
            "watch": self.watch,
            "written_by": self.written_by,
            "note": self.note,
            # Surfaced rather than hidden: if the model cited something that does
            # not exist, the page should be able to say so.
            "dropped": list(self.dropped),
        }


def _item_line(item: AttentionItem) -> str:
    money = f" {item.amount:,.2f} {item.currency}" if item.amount is not None else ""
    return f"- [{item.item_id}] ({item.urgency}) {item.headline}{money} — {item.detail} [source: {item.source}]"


def _party_line(party: StuckParty) -> str:
    rules = ", ".join(party.rules[:4]) or "no rule failure; held on confidence"
    return (
        f"- {party.counterparty}: {party.documents} document(s) held, "
        f"oldest {party.oldest_days} day(s); {rules}"
    )


def summarise(agenda: Agenda) -> str:
    """The deterministic version. Says less, and cannot be wrong."""
    overdue = [item for item in agenda.items if item.urgency == "overdue"]
    soon = [item for item in agenda.items if item.urgency == "soon"]
    stuck = agenda.parties

    if not agenda.items and not stuck:
        return "Nothing is due and nothing is held. There is no action outstanding."

    parts: list[str] = []
    if overdue:
        parts.append(f"{len(overdue)} item(s) already past their date")
    if soon:
        parts.append(f"{len(soon)} due shortly")
    if stuck:
        worst = stuck[0]
        parts.append(
            f"{sum(party.documents for party in stuck)} document(s) held across "
            f"{len(stuck)} counterparty(ies), most on {worst.counterparty}"
        )
    return ("Outstanding: " + "; ".join(parts) + ".") if parts else "Nothing needs attention today."


def _fallback(agenda: Agenda, note: str) -> Briefing:
    """Lead with whatever is most urgent; no prose beyond the facts."""
    return Briefing(
        summary=summarise(agenda),
        highlights=tuple(
            Highlight(item.item_id, item.detail)
            for item in agenda.items[:MAX_HIGHLIGHTS]
            if item.urgency in {"overdue", "soon"}
        ),
        written_by="rules",
        note=note,
    )


def _cited_id(raw: Any) -> str:
    """The id the model meant.

    It is shown `[CTR-2025-018]` in the prompt and hands back `[CTR-2025-018]`,
    which is the same citation with the brackets it was shown. Rejecting that as
    ungrounded would be the check enforcing punctuation rather than truth - the
    same mistake as demanding a quote match a page character for character.
    """
    return str(raw or "").strip().strip("[]").strip()


def _parse(payload: dict[str, Any], agenda: Agenda, model: str, note: str) -> Briefing:
    """Keep only what the agenda can back up."""
    highlights: list[Highlight] = []
    dropped: list[str] = []

    for entry in (payload.get("highlights") or [])[: MAX_HIGHLIGHTS * 2]:
        if not isinstance(entry, dict):
            continue
        item_id = _cited_id(entry.get("item_id"))
        if agenda.item(item_id) is None:
            # The model referred to something that is not on the agenda. It is
            # not shown, and the fact that it happened is reported.
            dropped.append(item_id or "(no id)")
            continue
        highlights.append(Highlight(item_id, str(entry.get("note", ""))[:240]))
        if len(highlights) >= MAX_HIGHLIGHTS:
            break

    summary = str(payload.get("summary", "")).strip()
    return Briefing(
        summary=summary[:1200] or summarise(agenda),
        highlights=tuple(highlights),
        watch=str(payload.get("watch", "")).strip()[:300],
        written_by=model if summary else "rules",
        note=note,
        dropped=tuple(dropped),
    )


def write_briefing(agenda: Agenda, llm: LlmProvider | None) -> Briefing:
    """Ask the model to prioritise an agenda it cannot add to."""
    if not agenda.items and not agenda.parties:
        return Briefing(summary=summarise(agenda), written_by="rules", note="nothing to brief")
    if llm is None:
        return _fallback(agenda, "no agent model configured")

    items = "\n".join(_item_line(item) for item in agenda.items[:MAX_ITEMS_IN_PROMPT]) or "(none)"
    parties = "\n".join(_party_line(party) for party in agenda.parties) or "(none)"

    result = run_agent(
        llm,
        system=BRIEFING_SYSTEM,
        user=briefing_prompt(agenda.as_of.isoformat(), items, parties),
        tools=[],  # every fact it may use is already in the prompt
        max_tokens=MAX_TOKENS,
    )
    if result.payload is None:
        return _fallback(agenda, result.note or "the model returned nothing usable")

    model = getattr(llm, "model", None) or getattr(llm, "name", "model")
    return _parse(result.payload, agenda, str(model), result.note)
