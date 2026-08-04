"""What needs a person's attention, and which page each item came from.

The rest of the system answers "did we read this correctly". This answers the
question that follows: *and now what*. Two kinds of thing land here.

**Dates that are approaching.** A payable is due on a date somebody read off an
invoice; a contract expires on a date somebody read off an agreement. Both are
already in the ledger and both stop being useful the moment they pass unnoticed.

**Counterparties that keep getting stuck.** One held document is an incident.
The same supplier held four times for the same rule is a problem with the
supplier record, the policy, or the way they issue documents - and it is
invisible in a queue sorted by arrival.

Every item carries the document it came from, so nothing here has to be taken
on trust: the page that produced the number is one click away. That is the same
principle as evidence grounding, applied one level up - a dashboard that cannot
show its source is just a more confident guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from docuflow.db import connect, rows_to_dicts

# Horizons. A payment is worth flagging closer in than a contract renewal:
# paying late costs money next week, renewing late costs a negotiation.
PAYMENT_HORIZON_DAYS = 30
CONTRACT_HORIZON_DAYS = 90
# A document nobody has looked at for this long has stopped being a queue item
# and started being a backlog.
STALE_QUEUE_DAYS = 7

MAX_PER_KIND = 20

# The field that names the other party, whatever the document type calls it.
COUNTERPARTY_FIELDS = ("vendor_name", "merchant_name", "party_b", "applicant_name")


def _urgency(days_left: int | None, soon: int) -> str:
    if days_left is None:
        return "watch"
    if days_left < 0:
        return "overdue"
    return "soon" if days_left <= soon else "watch"


def _days_between(today: date, iso_date: str | None) -> int | None:
    """Days from today to `iso_date`. Negative once the date has passed."""
    if not iso_date:
        return None
    try:
        return (date.fromisoformat(str(iso_date)[:10]) - today).days
    except ValueError:
        return None


def _age_in_days(today: date, iso_date: str | None) -> int:
    """How long ago `iso_date` was. Zero for anything unparseable or in the future."""
    days = _days_between(today, iso_date)
    return max(0, -days) if days is not None else 0


@dataclass(frozen=True)
class AttentionItem:
    """One thing to do, with the document that says so."""

    item_id: str
    kind: str
    headline: str
    detail: str
    urgency: str
    # The reference a person would say out loud - "PO-2026-0488", not the
    # supplier's registered name. Used wherever the item is mentioned in prose.
    label: str = ""
    counterparty: str | None = None
    amount: float | None = None
    currency: str | None = None
    due_date: str | None = None
    days_left: int | None = None
    document_id: str | None = None
    record_id: str | None = None
    source: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "kind": self.kind,
            "headline": self.headline,
            "detail": self.detail,
            "urgency": self.urgency,
            "label": self.label,
            "counterparty": self.counterparty,
            "amount": self.amount,
            "currency": self.currency,
            "due_date": self.due_date,
            "days_left": self.days_left,
            "document_id": self.document_id,
            "record_id": self.record_id,
            "source": self.source,
        }


@dataclass(frozen=True)
class StuckParty:
    """A counterparty whose documents keep being held, and what holds them."""

    counterparty: str
    documents: int
    oldest_days: int
    rules: tuple[str, ...]
    document_ids: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "counterparty": self.counterparty,
            "documents": self.documents,
            "oldest_days": self.oldest_days,
            "rules": list(self.rules),
            "document_ids": list(self.document_ids),
        }


@dataclass(frozen=True)
class Agenda:
    as_of: date
    items: tuple[AttentionItem, ...]
    parties: tuple[StuckParty, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "items": [item.to_json() for item in self.items],
            "parties": [party.to_json() for party in self.parties],
            "counts": {
                "overdue": sum(1 for item in self.items if item.urgency == "overdue"),
                "soon": sum(1 for item in self.items if item.urgency == "soon"),
                "total": len(self.items),
                "stuck_parties": len(self.parties),
            },
        }

    def item(self, item_id: str) -> AttentionItem | None:
        return next((item for item in self.items if item.item_id == item_id), None)


# Sort order for the page: what is already late, then what is close, then the rest.
_URGENCY_RANK = {"overdue": 0, "soon": 1, "watch": 2}


class AttentionStore:
    """Deterministic answers to "what now". No model is involved."""

    def __init__(self, db_path: Path, today: date | None = None) -> None:
        self.db_path = db_path
        self.today = today or date.today()

    # ------------------------------------------------------------- payments

    def payments_due(self, horizon_days: int = PAYMENT_HORIZON_DAYS) -> list[dict[str, Any]]:
        rows = connect(self.db_path).execute(
            """SELECT r.record_id, r.document_id, r.reference, r.counterparty, r.amount,
                      r.currency, r.due_date, r.issue_date, d.doc_type, d.created_at
               FROM records r LEFT JOIN documents d ON d.document_id = r.document_id
               WHERE r.status = 'posted' AND r.record_kind = 'payable'
                 AND r.due_date IS NOT NULL AND r.due_date != ''
               ORDER BY r.due_date"""
        ).fetchall()
        return [
            row for row in rows_to_dicts(rows)
            if (days := _days_between(self.today, row["due_date"])) is not None
            and days <= horizon_days
        ]

    # ------------------------------------------------------------ contracts

    def contracts_expiring(self, horizon_days: int = CONTRACT_HORIZON_DAYS) -> list[dict[str, Any]]:
        """From the contract register, linked to the agreement we read if we read one."""
        rows = connect(self.db_path).execute(
            """SELECT c.contract_number, c.vendor_id, c.party_b, c.expiry_date,
                      c.value, c.currency, r.document_id, r.record_id
               FROM contracts c
               LEFT JOIN records r
                      ON r.record_kind = 'contract' AND r.reference = c.contract_number
               WHERE c.expiry_date IS NOT NULL AND c.expiry_date != ''
               ORDER BY c.expiry_date"""
        ).fetchall()
        return [
            row for row in rows_to_dicts(rows)
            if (days := _days_between(self.today, row["expiry_date"])) is not None
            and days <= horizon_days
        ]

    # ---------------------------------------------------------- commitments

    def unbilled_commitments(self) -> list[dict[str, Any]]:
        """Open purchase orders that no posted invoice has billed against.

        Money committed and then forgotten: the budget is spent as far as the
        business is concerned, and nothing has arrived to reconcile it against.
        """
        billed = {
            str(row["po_ref"])
            for row in connect(self.db_path).execute(
                """SELECT DISTINCT json_extract(f.value, '$.value') AS po_ref
                   FROM records r
                   JOIN documents d ON d.document_id = r.document_id,
                        json_each(d.payload, '$.fields') f
                   WHERE r.record_kind = 'payable'
                     AND json_extract(f.value, '$.name') = 'purchase_order_ref'
                     AND json_extract(f.value, '$.value') IS NOT NULL"""
            ).fetchall()
            if row["po_ref"]
        }
        rows = connect(self.db_path).execute(
            """SELECT p.po_number, p.vendor_id, p.amount, p.currency, p.po_date, v.legal_name
               FROM purchase_orders p LEFT JOIN vendors v ON v.vendor_id = p.vendor_id
               WHERE p.status = 'open' ORDER BY p.po_date"""
        ).fetchall()
        return [row for row in rows_to_dicts(rows) if row["po_number"] not in billed]

    # ------------------------------------------------------- who is stuck

    def stuck_parties(self, limit: int = 8) -> list[StuckParty]:
        """Counterparties with documents held, and the rules holding them."""
        rows = connect(self.db_path).execute(
            f"""SELECT d.document_id, d.created_at,
                       (SELECT json_extract(f.value, '$.value')
                          FROM json_each(d.payload, '$.fields') f
                         WHERE json_extract(f.value, '$.name') IN ({','.join('?' * len(COUNTERPARTY_FIELDS))})
                           AND json_extract(f.value, '$.value') IS NOT NULL
                         LIMIT 1) AS counterparty
                FROM documents d
                WHERE d.decision = 'needs_review'""",
            COUNTERPARTY_FIELDS,
        ).fetchall()

        failing: dict[str, list[str]] = {}
        for row in connect(self.db_path).execute(
            """SELECT d.document_id, json_extract(f.value, '$.rule_id') AS rule_id
               FROM documents d, json_each(d.payload, '$.findings') f
               WHERE d.decision = 'needs_review' AND json_extract(f.value, '$.passed') = 0"""
        ).fetchall():
            failing.setdefault(row["document_id"], []).append(row["rule_id"])

        grouped: dict[str, dict[str, Any]] = {}
        for row in rows_to_dicts(rows):
            name = (row["counterparty"] or "").strip() or "Unidentified counterparty"
            entry = grouped.setdefault(name, {"documents": [], "rules": [], "oldest": 0})
            entry["documents"].append(row["document_id"])
            entry["rules"].extend(failing.get(row["document_id"], []))
            entry["oldest"] = max(entry["oldest"], _age_in_days(self.today, row["created_at"]))

        parties = [
            StuckParty(
                counterparty=name,
                documents=len(entry["documents"]),
                oldest_days=entry["oldest"],
                # Most frequent first: the rule to look at is the repeated one.
                rules=tuple(sorted(set(entry["rules"]), key=lambda r: -entry["rules"].count(r))),
                document_ids=tuple(entry["documents"]),
            )
            for name, entry in grouped.items()
        ]
        parties.sort(key=lambda party: (-party.documents, -party.oldest_days))
        return parties[:limit]

    # -------------------------------------------------------------- agenda

    def agenda(self) -> Agenda:
        """Everything that wants attention, newest deadline first."""
        items: list[AttentionItem] = []

        for row in self.payments_due()[:MAX_PER_KIND]:
            days = _days_between(self.today, row["due_date"])
            items.append(
                AttentionItem(
                    item_id=f"PAY-{row['record_id']}",
                    kind="payment_due",
                    headline=f"{row['counterparty'] or 'Supplier'} — {_money(row['amount'], row['currency'])}",
                    label=str(row["reference"] or row["counterparty"] or row["record_id"]),
                    detail=_deadline_phrase("Payment", days, row["due_date"]),
                    urgency=_urgency(days, soon=7),
                    counterparty=row["counterparty"],
                    amount=row["amount"],
                    currency=row["currency"],
                    due_date=row["due_date"],
                    days_left=days,
                    document_id=row["document_id"],
                    record_id=row["record_id"],
                    source=_source(row.get("doc_type"), row["reference"], row["document_id"]),
                )
            )

        for row in self.contracts_expiring()[:MAX_PER_KIND]:
            days = _days_between(self.today, row["expiry_date"])
            items.append(
                AttentionItem(
                    item_id=f"CTR-{row['contract_number']}",
                    kind="contract_expiring",
                    headline=f"{row['party_b'] or row['vendor_id'] or 'Contract'} — {row['contract_number']}",
                    label=str(row["contract_number"]),
                    detail=_deadline_phrase(
                        "Contract", days, row["expiry_date"], verb="expires", past="expired"
                    ),
                    urgency=_urgency(days, soon=30),
                    counterparty=row["party_b"],
                    amount=row["value"],
                    currency=row["currency"],
                    due_date=row["expiry_date"],
                    days_left=days,
                    document_id=row["document_id"],
                    record_id=row["record_id"],
                    source=(
                        _source("contract", row["contract_number"], row["document_id"])
                        if row["document_id"]
                        else f"contract register entry {row['contract_number']} (no agreement on file)"
                    ),
                )
            )

        for row in self.unbilled_commitments()[:MAX_PER_KIND]:
            items.append(
                AttentionItem(
                    item_id=f"PO-{row['po_number']}",
                    kind="commitment_unbilled",
                    headline=f"{row['legal_name'] or row['vendor_id']} — {row['po_number']}",
                    label=str(row["po_number"]),
                    detail=(
                        f"{_money(row['amount'], row['currency'])} committed on "
                        f"{row['po_date'] or 'an unknown date'}"
                        + (f", {age} day(s) ago" if (age := _age_in_days(self.today, row['po_date'])) else "")
                        + ", and no invoice has been billed against it."
                    ),
                    # No deadline attached: this is money already committed, not a
                    # date approaching, so it never becomes overdue on its own.
                    urgency="watch",
                    counterparty=row["legal_name"],
                    amount=row["amount"],
                    currency=row["currency"],
                    source=f"purchase order register entry {row['po_number']}",
                )
            )

        items.sort(key=lambda item: (_URGENCY_RANK[item.urgency], item.days_left if item.days_left is not None else 9999))
        return Agenda(as_of=self.today, items=tuple(items), parties=tuple(self.stuck_parties()))


def _money(amount: float | None, currency: str | None) -> str:
    if amount is None:
        return "amount unknown"
    return f"{amount:,.2f}{' ' + currency if currency else ''}"


def _deadline_phrase(
    subject: str, days: int | None, on: str, verb: str = "is due", past: str = "was due"
) -> str:
    """Both tenses are passed in: "Contract was expires" is how this reads otherwise."""
    if days is None:
        return f"{subject} {verb} on {on}."
    if days < 0:
        return f"{subject} {past} on {on} — {abs(days)} day(s) ago."
    if days == 0:
        return f"{subject} {verb} today ({on})."
    return f"{subject} {verb} in {days} day(s), on {on}."


def _source(doc_type: str | None, reference: str | None, document_id: str | None) -> str:
    """Where this came from, in the words a person would use."""
    parts = [part for part in ((doc_type or "document").replace("_", " "), reference) if part]
    return " ".join(parts) + (f" · {document_id}" if document_id else "")
