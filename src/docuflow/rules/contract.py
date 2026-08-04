"""Rules for contracts and agreements."""

from __future__ import annotations

from datetime import date

from docuflow.models import RuleFinding
from docuflow.normalize.text import normalize_for_match
from docuflow.rules.context import RuleContext
from docuflow.rules.compare import describe_difference, same_value
from docuflow.rules.engine import failed, passed, rule

# The renewal warning window and maximum plausible term come from configuration.


def _iso(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


@rule("contract", severity="blocking")
def contract_dates_ordered(context: RuleContext) -> RuleFinding | None:
    """Expiry must come after the effective date, within a sane term."""
    start = _iso(context.value("effective_date"))
    end = _iso(context.value("expiry_date"))
    if start is None or end is None:
        return None

    if end <= start:
        return failed(
            "contract_dates_ordered",
            f"Expiry date {end.isoformat()} is not after the effective date {start.isoformat()}.",
            severity="blocking",
            fields=["effective_date", "expiry_date"],
        )
    years = (end - start).days / 365.25
    if years > context.max_contract_years:
        return failed(
            "contract_dates_ordered",
            f"Term of {years:.0f} years is implausible; a year may have been mis-read.",
            severity="warning",
            fields=["effective_date", "expiry_date"],
        )
    return passed("contract_dates_ordered", f"Term runs {start.isoformat()} to {end.isoformat()} ({years:.1f} years).")


@rule("contract", emits=("contract_expiring_soon", "contract_expired"))
def contract_expiring_soon(context: RuleContext) -> RuleFinding | None:
    """Surface contracts that are expiring or already expired."""
    end = _iso(context.value("expiry_date"))
    if end is None:
        return None

    days = (end - context.today).days
    if days < 0:
        # Reported under its own id. An agreement that ended last week is not
        # "expiring soon", and an id that says otherwise gets argued with - one
        # run spent its assessment correcting the label rather than reading the
        # message beside it, which said exactly the right thing.
        # Blocking, where "expiring soon" below is a warning. Now that a warning
        # genuinely does not hold a document, the two cases need different
        # severities or the wrong one of them decides. A contract with two months
        # left is something to put on someone's list; one that ended last week is
        # not a note, it is a supplier being paid under an agreement that no
        # longer exists, and somebody has to decide what to do before it is filed.
        return failed(
            "contract_expired",
            f"This agreement expired {abs(days)} days ago ({end.isoformat()}). It cannot be "
            f"filed as current until it is renewed or superseded.",
            severity="blocking",
            fields=["expiry_date"],
            days_remaining=days,
        )
    if days <= context.contract_expiry_warning_days:
        notice = context.value("notice_period_days")
        notice_note = (
            f" Notice of {notice:g} days is required, so the decision window is "
            f"{'already closed' if notice and days < float(notice) else 'still open'}."
            if notice
            else ""
        )
        return failed(
            "contract_expiring_soon",
            f"Expires in {days} days ({end.isoformat()}).{notice_note}",
            severity="warning",
            fields=["expiry_date", "notice_period_days"],
            days_remaining=days,
        )
    return passed("contract_expiring_soon", f"Expires in {days} days; no action needed yet.")


@rule("contract", severity="blocking")
def contract_parties_distinct(context: RuleContext) -> RuleFinding | None:
    """The two parties must be different organisations."""
    party_a = context.value("party_a")
    party_b = context.value("party_b")
    if not party_a or not party_b:
        return None
    if normalize_for_match(str(party_a)) == normalize_for_match(str(party_b)):
        return failed(
            "contract_parties_distinct",
            f"Both parties were extracted as '{party_a}'; one of them is wrong.",
            severity="blocking",
            fields=["party_a", "party_b"],
        )
    return passed("contract_parties_distinct", "The two contracting parties are distinct.")


@rule("contract", severity="blocking")
def contract_matches_master(context: RuleContext) -> RuleFinding | None:
    """Compare against the contract register when the number is known."""
    number = context.value("contract_number")
    if not number:
        return None

    existing = context.master.get_contract(str(number))
    if existing is None:
        return failed(
            "contract_matches_master",
            f"Contract {number} is not in the contract register; it will be created as a new entry.",
            severity="warning",
            fields=["contract_number"],
        )

    differences: list[str] = []
    for field_name, column in (
        ("effective_date", "effective_date"),
        ("expiry_date", "expiry_date"),
        ("contract_value", "value"),
    ):
        extracted = context.value(field_name)
        recorded = existing.get(column)
        if extracted is None or recorded is None:
            continue
        if not same_value(extracted, recorded, context.amount_tolerance):
            differences.append(describe_difference(field_name, extracted, recorded))

    if differences:
        return failed(
            "contract_matches_master",
            f"Contract {number} differs from the register - " + "; ".join(differences) + ".",
            severity="blocking",
            fields=["effective_date", "expiry_date", "contract_value"],
            differences=differences,
        )
    return passed("contract_matches_master", f"Contract {number} matches the register.")


@rule("contract")
def contract_value_present(context: RuleContext) -> RuleFinding | None:
    """A contract with no value cannot drive commitments."""
    if context.value("contract_value") is not None:
        return passed("contract_value_present", "Contract value was extracted.")
    return failed(
        "contract_value_present",
        "No contract value was found; the commitment cannot be recorded without it.",
        severity="warning",
        fields=["contract_value"],
    )
