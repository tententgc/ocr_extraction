"""Date parsing, including Thai Buddhist-era years and Thai month names.

Returns ISO `YYYY-MM-DD` strings. Anything genuinely ambiguous returns None so
the field drops to the review queue rather than posting a wrong date.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from dateutil import parser as dateutil_parser

from docuflow.normalize.text import collapse_whitespace, thai_digits_to_arabic

# Buddhist Era is 543 years ahead of the Common Era.
BE_OFFSET = 543
# Any 4-digit year above this is assumed to be Buddhist Era.
BE_THRESHOLD = 2200

THAI_MONTHS = {
    "มกราคม": 1, "ม.ค.": 1, "มค": 1,
    "กุมภาพันธ์": 2, "ก.พ.": 2, "กพ": 2,
    "มีนาคม": 3, "มี.ค.": 3, "มีค": 3,
    "เมษายน": 4, "เม.ย.": 4, "เมย": 4,
    "พฤษภาคม": 5, "พ.ค.": 5, "พค": 5,
    "มิถุนายน": 6, "มิ.ย.": 6, "มิย": 6,
    "กรกฎาคม": 7, "ก.ค.": 7, "กค": 7,
    "สิงหาคม": 8, "ส.ค.": 8, "สค": 8,
    "กันยายน": 9, "ก.ย.": 9, "กย": 9,
    "ตุลาคม": 10, "ต.ค.": 10, "ตค": 10,
    "พฤศจิกายน": 11, "พ.ย.": 11, "พย": 11,
    "ธันวาคม": 12, "ธ.ค.": 12, "ธค": 12,
}

# Two deliberately different defaults, used to detect which date components
# dateutil filled in rather than read from the text.
_PROBE_A = datetime(1999, 1, 3)
_PROBE_B = datetime(2011, 7, 22)

_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_NUMERIC_RE = re.compile(r"\b(\d{1,4})[/.\-](\d{1,2})[/.\-](\d{2,4})\b")
_THAI_RE = re.compile(r"(\d{1,2})\s*([ก-๙.]+)\s*(\d{2,4})")
_YEAR_ONLY_RE = re.compile(r"\b(\d{4})\b")


def _to_ce_year(year: int) -> int:
    """Convert a Buddhist-era year to CE; two-digit years get a century."""
    if year > BE_THRESHOLD:
        return year - BE_OFFSET
    if year < 100:
        # 68 -> 2568 BE -> 2025 CE; 24 -> 2024 CE.
        candidate = 2500 + year if year >= 40 else 2000 + year
        return candidate - BE_OFFSET if candidate > BE_THRESHOLD else candidate
    return year


def _build(year: int, month: int, day: int) -> str | None:
    try:
        return date(_to_ce_year(year), month, day).isoformat()
    except (ValueError, OverflowError):
        return None


def _parse_thai(text: str) -> str | None:
    match = _THAI_RE.search(text)
    if not match:
        return None
    day_str, month_str, year_str = match.groups()
    month_key = month_str.strip().rstrip(".")
    month = THAI_MONTHS.get(month_str.strip()) or THAI_MONTHS.get(month_key)
    if month is None:
        # Try prefix matching: OCR may clip "กุมภาพันธ" from "กุมภาพันธ์".
        month = next(
            (num for name, num in THAI_MONTHS.items() if name.startswith(month_key) and len(month_key) >= 2),
            None,
        )
    if month is None:
        return None
    return _build(int(year_str), month, int(day_str))


def _parse_numeric(text: str, day_first: bool) -> str | None:
    match = _NUMERIC_RE.search(text)
    if not match:
        return None
    first, second, third = (int(part) for part in match.groups())

    # Leading 4-digit group means the string is already year-first.
    if len(match.group(1)) == 4:
        return _build(first, second, third)

    if first > 12 and second <= 12:
        return _build(third, second, first)
    if second > 12 and first <= 12:
        return _build(third, first, second)
    # Both plausible - fall back to the configured convention.
    return _build(third, second, first) if day_first else _build(third, first, second)


def detect_day_first(text: str) -> bool | None:
    """Which way round this document writes numeric dates, from its own evidence.

    `11/02/2019` is genuinely ambiguous, and two models reading it will disagree
    - one returned 2019-11-02 and the other 2019-02-11 for the same page. Asking
    each model to guess produces a disagreement that is not about the document.

    But a document that writes dates one way usually writes them all that way,
    and any date with a component above 12 settles the question for the whole
    page: "26/02/2019" can only be day-first. Returns None when the page never
    says, which leaves the configured default to decide.
    """
    day_first_votes = month_first_votes = 0
    for match in _NUMERIC_RE.finditer(text or ""):
        first, second, _ = (int(part) for part in match.groups())
        if len(match.group(1)) == 4:
            continue  # already year-first, tells us nothing
        if first > 12 >= second:
            day_first_votes += 1
        elif second > 12 >= first:
            month_first_votes += 1
    if day_first_votes == month_first_votes:
        return None
    return day_first_votes > month_first_votes


def parse_date(text: str | date | datetime | None, day_first: bool = True) -> str | None:
    """Parse a date from noisy text into an ISO `YYYY-MM-DD` string.

    `day_first` reflects the dominant convention for Thai/EU documents; it is
    only consulted when the numbers alone cannot disambiguate.
    """
    if text is None:
        return None
    if isinstance(text, datetime):
        return text.date().isoformat()
    if isinstance(text, date):
        return text.isoformat()

    cleaned = collapse_whitespace(thai_digits_to_arabic(str(text)))
    if not cleaned:
        return None

    iso_match = _ISO_RE.search(cleaned)
    if iso_match:
        year, month, day = (int(part) for part in iso_match.groups())
        built = _build(year, month, day)
        if built:
            return built

    for parser in (_parse_thai, lambda value: _parse_numeric(value, day_first)):
        result = parser(cleaned)
        if result:
            return result

    # Last resort: dateutil handles "12 March 2024", "Mar 12, 2024" and friends.
    # It silently fills anything the text does not specify, so parse twice with
    # different defaults and keep only the components that agree. "Net 45" would
    # otherwise become 2045 with today's day and month.
    try:
        first = dateutil_parser.parse(cleaned, dayfirst=day_first, fuzzy=True, default=_PROBE_A)
        second = dateutil_parser.parse(cleaned, dayfirst=day_first, fuzzy=True, default=_PROBE_B)
    except (ValueError, OverflowError, TypeError):
        return None

    # A component that changed with the default was invented, not read.
    if first.year != second.year or first.month != second.month or first.day != second.day:
        return None
    return _build(first.year, first.month, first.day)


def days_between(start: str | None, end: str | None) -> int | None:
    """Whole days from `start` to `end`, both ISO strings."""
    if not start or not end:
        return None
    try:
        return (date.fromisoformat(end) - date.fromisoformat(start)).days
    except ValueError:
        return None
