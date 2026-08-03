"""Deterministic label-based extraction.

This channel needs no model at all. It reads the schema's label aliases, finds
them in the recognised text and takes the value that follows. On well-labelled
documents it is extremely precise, which makes it a useful independent check on
the LLM channel - and a working fallback when no model is available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from docuflow.normalize.amounts import parse_money, parse_number
from docuflow.normalize.dates import parse_date
from docuflow.normalize.text import normalize_for_match, thai_digits_to_arabic, unescape_entities
from docuflow.schemas.registry import DocumentSchema, FieldSpec

# Separators that sit between a label and its value.
_SEPARATOR_RE = re.compile(r"^[\s:：\-–—|=./,]+")
_MONEY_TOKEN_RE = re.compile(r"-?[\d๐-๙][\d๐-๙,. ']*")
_TAG_RE = re.compile(r"<[^>]+>")
_ROW_SPLIT_RE = re.compile(r"</t[dh]>\s*<t[dh][^>]*>|</tr>", re.IGNORECASE)


@dataclass(frozen=True)
class RuleCandidate:
    """One label match, with enough context to judge it against others."""

    field: str
    raw_value: str
    evidence: str
    score: float
    page: int
    # True when the value came from position on the page rather than a label,
    # which is weaker evidence and is scored down accordingly.
    heuristic: bool = False


def _lines_with_pages(ocr_pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Flatten pages into `(page_number, line)`, expanding HTML table rows."""
    rows: list[tuple[int, str]] = []
    for page_number, text in ocr_pages:
        for raw_line in text.splitlines():
            # Typhoon 1.5 returns tables as HTML; split cells into pseudo-lines
            # so a label in one cell and its value in the next still pair up.
            if "<t" in raw_line.lower():
                for cell_group in _ROW_SPLIT_RE.split(raw_line):
                    cleaned = unescape_entities(_TAG_RE.sub(" ", cell_group)).strip()
                    if cleaned:
                        rows.append((page_number, cleaned))
                continue
            cleaned = unescape_entities(raw_line).strip()
            if cleaned:
                rows.append((page_number, cleaned))
    return rows


def _alias_positions(line: str, alias: str) -> list[int]:
    """Word-boundary-safe positions of `alias` within `line`.

    The boundary check is what stops "Subtotal" from matching the alias "Total".
    """
    haystack = line.casefold()
    needle = alias.casefold()
    if not needle:
        return []

    positions = []
    start = 0
    while True:
        index = haystack.find(needle, start)
        if index == -1:
            break
        before = haystack[index - 1] if index > 0 else " "
        after_index = index + len(needle)
        after = haystack[after_index] if after_index < len(haystack) else " "
        # Latin labels must not be embedded inside a longer word. Thai has no
        # spaces between words, so only the preceding character is checked there.
        before_ok = not (before.isalpha() and before.isascii())
        if before_ok and _terminates_label(line, after_index):
            positions.append(index)
        start = index + 1
    return positions


# Characters that can legitimately sit between a label and its value.
_LABEL_TERMINATORS = frozenset({":", "：", "=", "|", ",", "#", "-", "–", "—"})


def _terminates_label(line: str, index: int) -> bool:
    """Whether the text at `index` ends a label rather than continuing a token.

    "PO" in the cell "PO-2026-0412" is part of the identifier, not a label for
    it, so a hyphen or slash pressed straight against the alias disqualifies the
    match. A separator with whitespace around it ("PO - 2026") is still a label.
    """
    if index >= len(line):
        return True
    char = line[index]
    if char.isspace():
        return True
    # "Invoice No." - a full stop is part of the label when a separator follows.
    if char == "." and (index + 1 >= len(line) or not line[index + 1].isalnum()):
        return _terminates_label(line, index + 1)
    if char in {"-", "/", "–", "—"}:
        return False  # identifier continuation such as "PO-2026-0412"
    if char in _LABEL_TERMINATORS:
        return True
    return not (char.isalnum() or char == "_")


def _value_after(line: str, index: int, alias: str) -> str:
    return _SEPARATOR_RE.sub("", line[index + len(alias) :]).strip()


# Longest plausible amount: 11 digits is a hundred billion.
_MAX_AMOUNT_DIGITS = 11


def _looks_like_amount(token: str) -> bool:
    """Reject identifiers that happen to sit next to a money label.

    Tax IDs, account numbers and reference codes are long digit runs with a
    leading zero and no decimal part - never amounts.
    """
    normalized = thai_digits_to_arabic(token).strip()
    digits = re.sub(r"\D", "", normalized)
    if not digits:
        return False
    has_decimals = bool(re.search(r"[.,]\d{1,2}\b", normalized))
    if len(digits) > _MAX_AMOUNT_DIGITS:
        return False
    if not has_decimals and normalized.lstrip("-").startswith("0") and len(digits) > 1:
        return False
    return True


def _extract_typed(remainder: str, spec: FieldSpec, whole_line: str) -> str | None:
    """Pull the part of `remainder` that matches the field's type."""
    if spec.type in {"money", "number"}:
        # Amount lines often carry two numbers ("VAT 7%  6,440.00"); the value is
        # the last money-shaped token on the line.
        tokens = _MONEY_TOKEN_RE.findall(remainder) or _MONEY_TOKEN_RE.findall(whole_line)
        tokens = [token.strip() for token in tokens if any(char.isdigit() for char in token)]
        if spec.type == "money":
            tokens = [token for token in tokens if _looks_like_amount(token)]
        if not tokens:
            return None
        if spec.type == "money":
            # Prefer a token that looks like a formatted amount over a bare "7".
            decimals = [token for token in tokens if re.search(r"[.,]\d{2}\b", thai_digits_to_arabic(token))]
            return (decimals or tokens)[-1]
        return tokens[0]

    if spec.type == "percent":
        match = re.search(r"([\d๐-๙]+(?:[.,][\d๐-๙]+)?)\s*%", remainder) or re.search(
            r"([\d๐-๙]+(?:[.,][\d๐-๙]+)?)\s*%", whole_line
        )
        return match.group(1) if match else (remainder or None)

    if spec.type == "date":
        return remainder if parse_date(remainder) else None

    if spec.type == "tax_id":
        digits = re.sub(r"\D", "", thai_digits_to_arabic(remainder))
        if len(digits) >= 8:
            return digits
        match = re.search(r"\b[A-Z]{2}[\dA-Z]{6,18}\b", remainder.upper())
        return match.group(0) if match else None

    # Plain strings: stop at a second label on the same line.
    value = re.split(r"\s{3,}|\s\|\s", remainder)[0].strip()
    return value or None


def _score_match(line: str, index: int, alias: str, value: str, from_next_line: bool) -> float:
    """Higher is better. Rewards specific labels in label-like positions."""
    score = 1.0 + min(len(alias), 25) / 25.0
    if index == 0:
        score += 1.5  # the label starts the line, so it really is a label
    if line[index + len(alias) : index + len(alias) + 2].lstrip().startswith(":"):
        score += 1.0  # explicit "Label:" form
    if from_next_line:
        score -= 0.3  # a stacked label/value pair is slightly weaker evidence
    if len(value) > 90:
        score -= 1.0  # probably swallowed a sentence
    return score


# Aliases at or below this length are ambiguous ("To", "Tax", "PO") and only
# count when they are clearly acting as a label.
SHORT_ALIAS_MAX = 3


def _alias_is_label_like(line: str, index: int, alias: str) -> bool:
    """Whether a short alias is being used as a label rather than as prose."""
    if len(alias) > SHORT_ALIAS_MAX:
        return True
    if index == 0:
        return True
    return line[index + len(alias) :].lstrip().startswith(":")


@dataclass(frozen=True)
class _Match:
    field: str
    line_index: int
    position: int
    alias: str


def _collect_matches(lines: list[tuple[int, str]], schema: DocumentSchema) -> list[_Match]:
    """Every alias hit across the document, before conflict resolution."""
    matches: list[_Match] = []
    for spec in schema.fields:
        aliases = tuple(dict.fromkeys((*spec.aliases, spec.label, spec.name.replace("_", " "))))
        for line_index, (_, line) in enumerate(lines):
            for alias in aliases:
                if not alias:
                    continue
                for position in _alias_positions(line, alias):
                    if not _alias_is_label_like(line, position, alias):
                        continue
                    matches.append(_Match(spec.name, line_index, position, alias))
    return matches


def _suppress_overlapping(matches: list[_Match]) -> list[_Match]:
    """Within one field, keep only the most specific alias at each position.

    Suppression is deliberately *not* applied across fields: two different fields
    legitimately share a label position, as with "VAT 7%   13,090.00" where the
    rate and the amount are both read off the same line.
    """
    spans_by_field: dict[str, list[tuple[int, int, int]]] = {}
    for match in matches:
        spans_by_field.setdefault(match.field, []).append(
            (match.line_index, match.position, match.position + len(match.alias))
        )

    kept: list[_Match] = []
    for match in matches:
        start, end = match.position, match.position + len(match.alias)
        covered = any(
            line_index == match.line_index
            and other_start <= start
            and other_end >= end
            and (other_end - other_start) > (end - start)
            for line_index, other_start, other_end in spans_by_field[match.field]
        )
        if not covered:
            kept.append(match)
    return kept


def extract_with_rules(
    ocr_pages: list[tuple[int, str]], schema: DocumentSchema
) -> dict[str, RuleCandidate]:
    """Best rule-based candidate per field."""
    lines = _lines_with_pages(ocr_pages)
    specs = {spec.name: spec for spec in schema.fields}
    best: dict[str, RuleCandidate] = {}

    for match in _suppress_overlapping(_collect_matches(lines, schema)):
        spec = specs[match.field]
        page_number, line = lines[match.line_index]
        remainder = _value_after(line, match.position, match.alias)
        from_next_line = False

        # A bilingual header such as "ลูกค้า / ผู้ซื้อ" leaves another label as
        # the remainder; that is not a value either.
        if remainder and _looks_like_label(remainder, schema):
            remainder = ""

        # Stacked layouts put the value on the line below its label
        # ("Bill To" / "Mirtphol Industries PCL").
        if not remainder and match.line_index + 1 < len(lines):
            next_page, next_line = lines[match.line_index + 1]
            if next_page == page_number and not _looks_like_label(next_line, schema):
                remainder = next_line.strip()
                from_next_line = True

        if not remainder and spec.type not in {"money", "number", "percent"}:
            continue

        source_line = lines[match.line_index + 1][1] if from_next_line else line
        value = _extract_typed(remainder, spec, source_line)
        if not value:
            continue

        score = _score_match(line, match.position, match.alias, value, from_next_line)
        current = best.get(spec.name)
        if current is None or score > current.score:
            evidence = f"{line.strip()} {remainder}".strip() if from_next_line else line.strip()
            best[spec.name] = RuleCandidate(
                field=spec.name,
                raw_value=value,
                evidence=evidence,
                score=score,
                page=page_number,
            )

    return _add_letterhead_party(best, lines, schema)


# Document types where the letterhead at the top of page 1 is the issuer.
_LETTERHEAD_ISSUER_TYPES = frozenset({"invoice", "receipt"})
_LETTERHEAD_FIELDS = ("vendor_name", "merchant_name")
_LETTERHEAD_SCAN_LINES = 4
_MIN_ORG_NAME_CHARS = 4
_MAX_ORG_NAME_CHARS = 90


def _add_letterhead_party(
    best: dict[str, RuleCandidate], lines: list[tuple[int, str]], schema: DocumentSchema
) -> dict[str, RuleCandidate]:
    """Fall back to the letterhead for the issuing party.

    Invoices and receipts almost never label their own name - the seller is
    simply printed at the top of the page. Without this, a document with no
    "Vendor:" label loses a required field entirely whenever no language model
    is available.
    """
    if schema.doc_type not in _LETTERHEAD_ISSUER_TYPES:
        return best

    target = next((name for name in _LETTERHEAD_FIELDS if schema.field(name)), None)
    if target is None or target in best:
        return best

    for page_number, line in lines[:_LETTERHEAD_SCAN_LINES]:
        candidate = line.strip()
        if not _MIN_ORG_NAME_CHARS <= len(candidate) <= _MAX_ORG_NAME_CHARS:
            continue
        if _looks_like_label(candidate, schema):
            continue
        # Skip document titles ("TAX INVOICE") and anything numeric-led.
        normalized = normalize_for_match(candidate)
        if normalized in {normalize_for_match(word) for word in schema.keywords}:
            continue
        if not any(char.isalpha() for char in candidate):
            continue
        if candidate[0].isdigit():
            continue
        return {
            **best,
            target: RuleCandidate(
                field=target,
                raw_value=candidate,
                evidence=candidate,
                score=0.8,
                page=page_number,
                heuristic=True,
            ),
        }
    return best


def _looks_like_label(line: str, schema: DocumentSchema) -> bool:
    """Whether a line is itself a label, so it cannot be the value above it."""
    stripped = line.strip()
    if stripped.endswith(":"):
        return True
    normalized = normalize_for_match(stripped)
    return any(
        normalized == normalize_for_match(alias)
        for spec in schema.fields
        for alias in (*spec.aliases, spec.label)
        if alias
    )


def detect_currency_from_text(ocr_pages: list[tuple[int, str]]) -> str | None:
    """Scan the whole document for a currency indicator."""
    found = detect_currency_from_pages(ocr_pages)
    return found[0] if found else None


def detect_currency_from_pages(
    ocr_pages: list[tuple[int, str]],
) -> tuple[str, str, int] | None:
    """Currency, the text that says so, and which page: `(code, evidence, page)`."""
    from docuflow.normalize.amounts import detect_currency_with_evidence

    for number, text in ocr_pages:
        found = detect_currency_with_evidence(text)
        if found:
            return found[0], found[1], number
    return None


# A plain-text table row: a description, then 2-4 numeric columns.
_TEXT_ROW_RE = re.compile(
    r"^(?P<description>\S.*?\S)\s{1,}(?P<numbers>(?:[-\d๐-๙][\d๐-๙,.']*\s+){1,3}[-\d๐-๙][\d๐-๙,.']*)$"
)
_TOTALS_WORDS = frozenset(
    {
        "subtotal", "sub total", "total", "grand total", "amount due", "vat", "tax",
        "balance", "รวมเป็นเงิน", "รวมทั้งสิ้น", "ภาษีมูลค่าเพิ่ม", "จำนวนเงินรวมทั้งสิ้น",
    }
)


def _find_text_line_items(text: str) -> list[dict[str, object]]:
    """Read `description  qty  unit  amount` rows out of plain text.

    Needed for the fallback OCR engines, which return no table markup.
    """
    items: list[dict[str, object]] = []
    for raw_line in text.splitlines():
        match = _TEXT_ROW_RE.match(raw_line.strip())
        if not match:
            continue
        description = match.group("description").strip()
        normalized = normalize_for_match(description)
        # Totals lines look like rows but are not line items.
        if not normalized or any(normalized.startswith(word) for word in _TOTALS_WORDS):
            continue
        if not any(char.isalpha() for char in description):
            continue

        numbers = [parse_money(token) for token in match.group("numbers").split()]
        numbers = [value for value in numbers if value is not None]
        if len(numbers) < 2:
            continue
        items.append(
            {
                "description": description,
                "quantity": numbers[0] if len(numbers) >= 3 else None,
                "unit_price": numbers[-2],
                "amount": numbers[-1],
            }
        )
    return items


def find_line_items(ocr_pages: list[tuple[int, str]]) -> list[dict[str, object]]:
    """Pull table rows that look like `description / qty / unit price / amount`."""
    items: list[dict[str, object]] = []
    for _, text in ocr_pages:
        if "<tr" not in text.lower():
            items.extend(_find_text_line_items(text))
            continue
        for raw_row in re.findall(r"<tr[^>]*>(.*?)</tr>", text, re.DOTALL | re.IGNORECASE):
            cells = [
                _TAG_RE.sub(" ", cell).strip()
                for cell in re.split(r"</t[dh]>", raw_row, flags=re.IGNORECASE)
            ]
            cells = [cell for cell in cells if cell]
            if len(cells) < 3:
                continue
            numbers = [parse_money(cell) for cell in cells[1:]]
            numeric = [value for value in numbers if value is not None]
            if len(numeric) < 2:
                continue  # header row or prose
            description = cells[0]
            if not description or normalize_for_match(description) in {"description", "item", "รายการ"}:
                continue
            items.append(
                {
                    "description": description,
                    "quantity": parse_number(cells[1]) if len(cells) > 1 else None,
                    "unit_price": numeric[-2] if len(numeric) >= 2 else None,
                    "amount": numeric[-1],
                }
            )
    return items
