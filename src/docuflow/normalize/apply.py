"""Apply the right normaliser for a field's declared type."""

from __future__ import annotations

from typing import Any

from docuflow.normalize.amounts import detect_currency, parse_money, parse_number, parse_percent
from docuflow.normalize.dates import parse_date
from docuflow.normalize.identifiers import normalize_account_number, normalize_reference, parse_tax_id
from docuflow.normalize.text import clean_value, collapse_whitespace
from docuflow.schemas.registry import FieldSpec

# Fields whose values are reference numbers rather than free text.
_REFERENCE_FIELDS = frozenset(
    {
        "invoice_number", "receipt_number", "po_number", "purchase_order_ref",
        "contract_number", "reference_number",
    }
)
_ACCOUNT_FIELDS = frozenset({"bank_account"})


def _normalize_enum(raw: str, spec: FieldSpec) -> str | None:
    """Snap a value onto one of the schema's allowed choices."""
    cleaned = clean_value(raw)
    if not cleaned:
        return None
    if not spec.choices:
        return cleaned

    lowered = cleaned.casefold()
    for choice in spec.choices:
        if lowered == choice.casefold():
            return choice

    # Currency fields accept symbols and words ("฿", "baht", "Thai Baht").
    if "THB" in spec.choices:
        detected = detect_currency(cleaned)
        if detected in spec.choices:
            return detected

    for choice in spec.choices:
        if choice.casefold() in lowered or lowered in choice.casefold():
            return choice

    # Yes/no style fields.
    if set(spec.choices) == {"yes", "no"}:
        if lowered in {"true", "y", "yes", "ใช่", "มี"}:
            return "yes"
        if lowered in {"false", "n", "no", "ไม่", "ไม่มี"}:
            return "no"
    return None


def normalize_field_value(raw: Any, spec: FieldSpec) -> tuple[Any, str | None]:
    """Return `(normalized_value, note)` for one raw extracted value.

    `note` is a human-readable explanation when normalisation dropped or
    reinterpreted the value, so the reviewer can see why.
    """
    if raw is None:
        return None, None
    if isinstance(raw, str) and not raw.strip():
        return None, None
    # Models sometimes emit these instead of null.
    if isinstance(raw, str) and raw.strip().casefold() in {"n/a", "na", "none", "null", "-", "unknown", "ไม่ระบุ"}:
        return None, None

    if spec.type in {"money", "number", "percent"}:
        parser = {"money": parse_money, "number": parse_number, "percent": parse_percent}[spec.type]
        value = parser(raw)
        if value is None:
            return None, f"could not parse a {spec.type} from {raw!r}"
        return value, None

    if spec.type == "date":
        value = parse_date(raw)
        if value is None:
            return None, f"could not parse a date from {raw!r}"
        return value, None

    if spec.type == "tax_id":
        value = parse_tax_id(str(raw))
        if value is None:
            return None, f"{raw!r} is not a valid tax identifier"
        return value, None

    if spec.type == "enum":
        value = _normalize_enum(str(raw), spec)
        if value is None:
            return None, f"{raw!r} is not one of {', '.join(spec.choices)}"
        return value, None

    if spec.type == "text":
        return collapse_whitespace(str(raw)) or None, None

    # Plain strings, with per-field special handling.
    text = str(raw)
    if spec.name in _ACCOUNT_FIELDS:
        value = normalize_account_number(text)
        return (value, None) if value else (None, f"{raw!r} is not a usable account number")
    if spec.name in _REFERENCE_FIELDS:
        value = normalize_reference(text)
        return (value, None) if value else (None, f"{raw!r} is not a usable reference")
    return clean_value(text) or None, None
