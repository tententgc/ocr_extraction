"""Value normalisation: raw OCR strings in, canonical typed values out."""

from docuflow.normalize.amounts import parse_money, parse_number, parse_percent
from docuflow.normalize.dates import parse_date
from docuflow.normalize.identifiers import parse_tax_id, normalize_account_number
from docuflow.normalize.text import (
    collapse_whitespace,
    normalize_for_match,
    thai_digits_to_arabic,
)
from docuflow.normalize.apply import normalize_field_value

__all__ = [
    "collapse_whitespace",
    "normalize_account_number",
    "normalize_field_value",
    "normalize_for_match",
    "parse_date",
    "parse_money",
    "parse_number",
    "parse_percent",
    "parse_tax_id",
    "thai_digits_to_arabic",
]
