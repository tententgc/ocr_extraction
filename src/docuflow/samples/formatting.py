"""Display formatters.

Ground truth is stored in canonical form (ISO dates, float amounts). These
helpers render it the way a real document would print it - Buddhist-era years,
Thai numerals, slashed dates - so the pipeline has to genuinely parse it back.
"""

from __future__ import annotations

from datetime import date

THAI_MONTH_NAMES = (
    "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
    "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม",
)
_ARABIC_TO_THAI = str.maketrans("0123456789", "๐๑๒๓๔๕๖๗๘๙")
BE_OFFSET = 543


def _parse(iso: str | None) -> date | None:
    if not iso:
        return None
    try:
        return date.fromisoformat(iso)
    except ValueError:
        return None


def to_thai_digits(text: str) -> str:
    return text.translate(_ARABIC_TO_THAI)


def date_long_en(iso: str | None) -> str:
    """`2026-07-08` -> `8 July 2026`."""
    parsed = _parse(iso)
    return "" if parsed is None else f"{parsed.day} {parsed.strftime('%B')} {parsed.year}"


def date_us(iso: str | None) -> str:
    """`2026-07-08` -> `Jul 8, 2026`."""
    parsed = _parse(iso)
    return "" if parsed is None else f"{parsed.strftime('%b')} {parsed.day}, {parsed.year}"


def date_slash(iso: str | None) -> str:
    """`2026-07-08` -> `08/07/2026` (day-first)."""
    parsed = _parse(iso)
    return "" if parsed is None else f"{parsed.day:02d}/{parsed.month:02d}/{parsed.year}"


def date_thai(iso: str | None, thai_digits: bool = True) -> str:
    """`2026-07-15` -> `๑๕ กรกฎาคม ๒๕๖๙` (Buddhist era)."""
    parsed = _parse(iso)
    if parsed is None:
        return ""
    rendered = f"{parsed.day} {THAI_MONTH_NAMES[parsed.month - 1]} {parsed.year + BE_OFFSET}"
    return to_thai_digits(rendered) if thai_digits else rendered


def money(value: float | None, decimals: int = 2) -> str:
    """`187000.0` -> `187,000.00`."""
    return "" if value is None else f"{value:,.{decimals}f}"


def money_thai(value: float | None) -> str:
    """Thai-numeral amount, as printed on some Thai tax invoices."""
    return "" if value is None else to_thai_digits(money(value))


def quantity(value: float | None) -> str:
    if value is None:
        return ""
    return str(int(value)) if float(value).is_integer() else f"{value:g}"
