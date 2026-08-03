"""Rules for submitted forms (vendor onboarding, claims, registrations)."""

from __future__ import annotations

import re

from docuflow.models import RuleFinding
from docuflow.normalize.identifiers import reference_key
from docuflow.rules.context import RuleContext
from docuflow.rules.engine import failed, passed, rule

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_PHONE_DIGITS = re.compile(r"\d")
MIN_PHONE_DIGITS = 8
MAX_PHONE_DIGITS = 15


@rule("form")
def contact_email_valid(context: RuleContext) -> RuleFinding | None:
    """A malformed e-mail means the applicant cannot be contacted."""
    email = context.value("contact_email")
    if not email:
        return None
    if _EMAIL_RE.match(str(email).strip()):
        return passed("contact_email_valid", f"Contact e-mail {email} is well formed.")
    return failed(
        "contact_email_valid",
        f"'{email}' is not a valid e-mail address; it may have been mis-read.",
        severity="warning",
        fields=["contact_email"],
    )


@rule("form")
def contact_phone_valid(context: RuleContext) -> RuleFinding | None:
    """Phone numbers must have a plausible number of digits."""
    phone = context.value("contact_phone")
    if not phone:
        return None
    digits = _PHONE_DIGITS.findall(str(phone))
    if MIN_PHONE_DIGITS <= len(digits) <= MAX_PHONE_DIGITS:
        return passed("contact_phone_valid", f"Contact number {phone} has a plausible length.")
    return failed(
        "contact_phone_valid",
        f"'{phone}' has {len(digits)} digits, which is not a plausible phone number.",
        severity="warning",
        fields=["contact_phone"],
    )


@rule("form")
def bank_details_complete(context: RuleContext) -> RuleFinding | None:
    """Onboarding without bank details cannot lead to payment."""
    if context.value("bank_account"):
        return passed("bank_details_complete", "Bank account was captured.")
    return failed(
        "bank_details_complete",
        "No bank account was captured; the supplier cannot be paid until this is supplied.",
        severity="warning",
        fields=["bank_account"],
    )


@rule("form")
def bank_account_not_reused(context: RuleContext) -> RuleFinding | None:
    """A bank account already registered to another vendor needs scrutiny."""
    account = context.value("bank_account")
    if not account:
        return None

    owner = context.master.vendor_for_bank_account(str(account))
    if owner is None:
        return passed("bank_account_not_reused", "Bank account is not registered to an existing vendor.")

    applicant = context.value("applicant_name")
    match = context.vendor_match
    if match.matched and match.vendor_id == owner["vendor_id"]:
        return passed("bank_account_not_reused", f"Bank account already belongs to {owner['legal_name']} - consistent.")
    return failed(
        "bank_account_not_reused",
        f"Bank account {account} is already registered to {owner['legal_name']}, "
        f"but this form is submitted by '{applicant}'.",
        severity="blocking",
        fields=["bank_account", "applicant_name"],
        existing_vendor=owner["legal_name"],
    )


@rule("form")
def applicant_already_registered(context: RuleContext) -> RuleFinding | None:
    """Advisory: this applicant may already exist in the master file."""
    match = context.vendor_match
    if not match.matched:
        return None
    vendor = match.vendor
    tax_id = context.value("tax_id")
    same_tax = tax_id and vendor.get("tax_id") and reference_key(str(tax_id)) == reference_key(vendor["tax_id"])
    return failed(
        "applicant_already_registered",
        f"'{context.value('applicant_name')}' resembles existing vendor {vendor['legal_name']} "
        f"({vendor['vendor_id']}){' with the same tax ID' if same_tax else ''} - this may be a duplicate registration.",
        severity="warning",
        fields=["applicant_name"],
        vendor_id=vendor["vendor_id"],
    )
