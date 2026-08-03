"""Monetary and numeric parsing.

Handles the formats that actually show up on scanned documents: thousands
separators in either convention, currency symbols and codes, Thai numerals,
parenthesised negatives, and OCR digit confusions around the decimal point.
"""

from __future__ import annotations

import re

from docuflow.normalize.text import thai_digits_to_arabic

CURRENCY_SYMBOLS = {
    "฿": "THB",
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₩": "KRW",
}

CURRENCY_WORDS = {
    "baht": "THB",
    "bath": "THB",
    "thb": "THB",
    "บาท": "THB",
    "usd": "USD",
    "dollar": "USD",
    "dollars": "USD",
    "eur": "EUR",
    "euro": "EUR",
    "jpy": "JPY",
    "yen": "JPY",
    "sgd": "SGD",
    "gbp": "GBP",
    "pound": "GBP",
    "cny": "CNY",
    "rmb": "CNY",
    "yuan": "CNY",
}

_NUMBER_RE = re.compile(r"-?\d[\d,. ']*\d|\d")
_TRAILING_JUNK = re.compile(r"[^\d.,\-]+$")


def _strip_to_number(text: str) -> str | None:
    """Pull the most plausible numeric token out of a noisy string."""
    cleaned = thai_digits_to_arabic(str(text)).strip()
    if not cleaned:
        return None
    negative = "(" in cleaned and ")" in cleaned
    cleaned = cleaned.replace("(", "").replace(")", "")
    matches = _NUMBER_RE.findall(cleaned)
    if not matches:
        return None
    # A money string like "THB 1,234.50 /unit" - take the longest token, which is
    # the amount rather than an incidental digit.
    token = max(matches, key=len)
    token = _TRAILING_JUNK.sub("", token).replace(" ", "").replace("'", "")
    if negative and not token.startswith("-"):
        token = f"-{token}"
    return token or None


def _resolve_separators(token: str) -> str | None:
    """Decide which of `.`/`,` is the decimal point and drop the other."""
    has_dot = "." in token
    has_comma = "," in token

    if has_dot and has_comma:
        # Whichever appears last is the decimal separator.
        decimal = "." if token.rfind(".") > token.rfind(",") else ","
        thousands = "," if decimal == "." else "."
        token = token.replace(thousands, "")
        return token.replace(decimal, ".")

    if has_comma:
        tail = token.rsplit(",", 1)[1]
        # "1,234" -> thousands; "1,50" -> European decimal.
        return token.replace(",", "") if len(tail) == 3 else token.replace(",", ".")

    if has_dot:
        parts = token.split(".")
        if len(parts) > 2:
            # "1.234.567" - all separators are thousands marks.
            return "".join(parts)
        # "1.234" is ambiguous; 3 decimals on money is far rarer than a
        # thousands separator, but we keep it as-is and let rules catch it.
        return token

    return token


def parse_number(text: str | float | int | None) -> float | None:
    """Parse any numeric-looking value into a float, or None."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    token = _strip_to_number(text)
    if token is None:
        return None
    resolved = _resolve_separators(token)
    if resolved is None:
        return None
    try:
        return float(resolved)
    except ValueError:
        return None


def parse_money(text: str | float | int | None) -> float | None:
    """Parse a monetary amount, rounded to 2dp."""
    value = parse_number(text)
    return None if value is None else round(value, 2)


def parse_percent(text: str | float | int | None) -> float | None:
    """Parse a percentage into its numeric value (7% -> 7.0)."""
    value = parse_number(text)
    if value is None:
        return None
    # A model sometimes returns 0.07 for 7%.
    if 0 < value < 1:
        value *= 100
    return round(value, 4)


def detect_currency(text: str) -> str | None:
    """Find an ISO currency code in free text."""
    if not text:
        return None
    for symbol, code in CURRENCY_SYMBOLS.items():
        if symbol in text:
            return code
    lowered = text.casefold()
    for word, code in CURRENCY_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", lowered) or word in lowered:
            return code
    return None


def format_money(value: float | None, currency: str | None = None) -> str:
    """Human-readable amount for summaries and the UI."""
    if value is None:
        return "-"
    formatted = f"{value:,.2f}"
    return f"{formatted} {currency}" if currency else formatted
