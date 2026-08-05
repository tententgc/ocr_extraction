"""Identifier parsing: tax IDs, bank accounts, document reference numbers."""

from __future__ import annotations

import re
import unicodedata

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
    """Bank account numbers: keep digits, preserve IBAN-style alphanumerics.

    Thai numerals are converted first, and the IBAN branch demands ASCII
    letters. Both of those are load-bearing, because Python answers the
    obvious questions about Thai script the unhelpful way.

    `"๑๒๓๔๕๖๗๘๙๐".isdigit()` is **True**, so the early return fired and the
    account reached the ledger in Thai numerals. `bank_account_matches_vendor`
    still passed - it compares through `reference_key`, which converts - so the
    document auto-posted, nobody looked at it, and the export handed another
    system an account number it cannot pay.

    `"เลขที่บัญชี"[:2].isalpha()` is **also True**, so a labelled Thai account
    took the IBAN path and the label was stored as part of the number.
    """
    if not text:
        return None
    raw = collapse_whitespace(thai_digits_to_arabic(str(text))).replace(" ", "").replace("-", "")
    if not raw:
        return None
    if raw.isdigit():
        return raw if 6 <= len(raw) <= 24 else None
    upper = raw.upper()
    if (
        10 <= len(upper) <= 34
        # `isascii()` is the whole guard: an IBAN country code is two Latin
        # letters, and any non-Latin script here means this is not an IBAN.
        and upper[:2].isascii()
        and upper[:2].isalpha()
        and any(char.isdigit() for char in upper)
    ):
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


def _is_key_character(char: str) -> bool:
    """Whether a character carries identity for a reference key.

    Alphanumerics, plus the combining marks that are part of the letter before
    them. `str.isalnum()` alone is False for every Thai vowel sign and tone
    mark - they are category `Mn`, nonspacing marks - so filtering on it
    silently deletes them and collapses words that are not the same word:

        เก่า  and  เก้า   both become  เกา
        ไม่   and  ไม้    both become  ไม
        เสื้อ and  เสือ   both become  เสอ

    This key resolves purchase orders and contracts against the register, so a
    collision does not fail loudly - it matches the *wrong* record, and every
    check downstream then compares a document against a contract that is not
    its own.
    """
    return char.isalnum() or unicodedata.category(char).startswith("M")


def reference_key(text: str | None) -> str:
    """Comparison key for references: uppercase alphanumerics only.

    `INV-2024/0142`, `inv 20240142` and `INV20240142` all collapse to the same
    key, which is what duplicate detection needs.

    NFC first, so a Thai string that arrives decomposed keys the same as the
    composed form of the same word.
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFC", thai_digits_to_arabic(str(text))).upper()
    return "".join(char for char in folded if _is_key_character(char))
