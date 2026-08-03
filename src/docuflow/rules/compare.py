"""Comparing an extracted value against a recorded one.

Rules keep asking the same question - does what the document says match what the
register says? - and the wrong way to answer it is `str(a) != str(b)`. A date
read as `2025-08-01` and stored as `2025-8-1`, a currency read as `thb` against
`THB`, an amount read as `1500` against `1500.00`: all identical to a person, all
different to string equality, and two of the rules that did it that way were
`blocking`. A false block is not a safe failure - it teaches reviewers that the
blocks are noise.

Deliberately stricter than the agreement test in `agents/merge.py`. Two readings
of a page may agree by one being a fuller version of the other; a contract value
against the register has to be the same value, not a compatible one.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from docuflow.normalize.amounts import parse_number
from docuflow.normalize.dates import parse_date
from docuflow.normalize.text import normalize_for_match


# Three groups around a separator, or a month name beside a year. Strict on
# purpose: "2025-08-01" is a date, "1,500.00" is not, and the number parser
# happily reads 2025 out of the first one - which made two different dates
# compare equal until this test ran ahead of it.
_DATE_SHAPE = re.compile(
    r"^\s*\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}\s*$"
    r"|^\s*[\d๐-๙]{1,2}\s+[^\W\d_]{3,12}\.?\s+[\d๐-๙]{2,4}\s*$"
    r"|^\s*[^\W\d_]{3,12}\.?\s+[\d๐-๙]{1,2},?\s+[\d๐-๙]{2,4}\s*$",
    re.UNICODE,
)


def _as_date(value: Any) -> str | None:
    if isinstance(value, (date, datetime)):
        return value.isoformat()[:10]
    if isinstance(value, str) and _DATE_SHAPE.match(value):
        return parse_date(value)
    return None


def same_value(left: Any, right: Any, tolerance: float = 0.0) -> bool:
    """Whether two values are the same thing written two ways.

    `tolerance` is relative and applies to numbers only - a register rounding an
    amount to the nearest baht should not read as a discrepancy.
    """
    if left is None or right is None:
        return False

    # Dates first: a date parses as a number too, and wrongly.
    left_date, right_date = _as_date(left), _as_date(right)
    if left_date is not None and right_date is not None:
        return left_date == right_date
    if (left_date is None) != (right_date is None):
        return False

    left_number, right_number = parse_number(left), parse_number(right)
    if left_number is not None and right_number is not None:
        allowed = max(abs(right_number) * tolerance, 0.01)
        return abs(left_number - right_number) <= allowed

    return normalize_for_match(str(left)) == normalize_for_match(str(right))


def describe_difference(label: str, extracted: Any, recorded: Any) -> str:
    """One readable line for a mismatch, used in rule messages."""
    return f"{label}: document {extracted}, register {recorded}"
