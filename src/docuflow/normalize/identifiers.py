"""Identifier parsing: tax IDs, bank accounts, document reference numbers."""

from __future__ import annotations

import re

from docuflow.normalize.text import collapse_whitespace, thai_digits_to_arabic

_DIGITS_RE = re.compile(r"\d")
_REF_CLEAN_RE = re.compile(r"[^\w\-/#.]")

THAI_TAX_ID_LENGTH = 13


def digits_only(text: str | None) -> str:
    """All digits in a string, Thai numerals included."""
    if not text:
        return ""
    return "".join(_DIGITS_RE.findall(thai_digits_to_arabic(str(text))))


def thai_tax_id_checksum_ok(tax_id: str) -> bool:
    """Validate the Thai 13-digit tax/national ID check digit.

    Weighted sum of the first 12 digits (weights 13..2), then
    check digit = (11 - sum % 11) % 10.
    """
    if len(tax_id) != THAI_TAX_ID_LENGTH or not tax_id.isdigit():
        return False
    total = sum(int(digit) * (THAI_TAX_ID_LENGTH - index) for index, digit in enumerate(tax_id[:12]))
    return (11 - total % 11) % 10 == int(tax_id[12])


def parse_tax_id(text: str | None) -> str | None:
    """Normalise a tax ID to bare digits.

    Thai IDs are 13 digits; other jurisdictions vary, so anything between 8 and
    20 characters is kept as-is rather than discarded.
    """
    if not text:
        return None
    digits = digits_only(text)
    if len(digits) == THAI_TAX_ID_LENGTH:
        return digits

    # Country-prefixed VAT numbers such as "GB123456789" keep their prefix -
    # dropping it would make two different registrations compare equal.
    cleaned = collapse_whitespace(str(text)).replace(" ", "").replace("-", "").upper()
    if any(char.isalpha() for char in cleaned):
        return cleaned if 8 <= len(cleaned) <= 20 and any(c.isdigit() for c in cleaned) else None

    return digits if 8 <= len(digits) <= 20 else None


def normalize_account_number(text: str | None) -> str | None:
    """Bank account numbers: keep digits, preserve IBAN-style alphanumerics."""
    if not text:
        return None
    raw = collapse_whitespace(str(text)).replace(" ", "").replace("-", "")
    if not raw:
        return None
    if raw.isdigit():
        return raw if 6 <= len(raw) <= 24 else None
    upper = raw.upper()
    if 10 <= len(upper) <= 34 and upper[:2].isalpha() and any(c.isdigit() for c in upper):
        return upper
    digits = digits_only(raw)
    return digits if 6 <= len(digits) <= 24 else None


def normalize_reference(text: str | None) -> str | None:
    """Canonical form for invoice/PO/contract numbers."""
    if not text:
        return None
    cleaned = _REF_CLEAN_RE.sub("", collapse_whitespace(thai_digits_to_arabic(str(text))))
    cleaned = cleaned.strip("-/.#")
    return cleaned.upper() or None


def reference_key(text: str | None) -> str:
    """Comparison key for references: uppercase alphanumerics only.

    `INV-2024/0142`, `inv 20240142` and `INV20240142` all collapse to the same
    key, which is what duplicate detection needs.
    """
    if not text:
        return ""
    return "".join(char for char in thai_digits_to_arabic(str(text)).upper() if char.isalnum())
