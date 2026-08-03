"""Text canonicalisation shared by parsing and evidence matching."""

from __future__ import annotations

import html
import re
import unicodedata

_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_WHITESPACE_RE = re.compile(r"\s+")
# Characters OCR commonly emits that carry no meaning for matching.
_NOISE_RE = re.compile(r"[​‌‍﻿­]")
_PUNCT_RE = re.compile(r"[^\w\s฀-๿]", re.UNICODE)
# Typhoon OCR 1.5 returns tables as HTML. Left in place, the tag names survive
# punctuation stripping as the words "table tr td" and wreck every comparison:
# a model quoting "Invoice Number: AA888" would fail to match the page text
# "<td>Invoice Number</td><td>: AA888</td>" and the field would be reported as
# unverified even though it is plainly there.
_HTML_TAG_RE = re.compile(r"<[^>]{1,200}>")


def thai_digits_to_arabic(text: str) -> str:
    """Convert Thai (and Arabic-Indic) numerals to ASCII digits."""
    if not text:
        return ""
    return text.translate(_THAI_DIGITS).translate(_ARABIC_INDIC)


def collapse_whitespace(text: str) -> str:
    """Trim and squeeze runs of whitespace into single spaces."""
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", _NOISE_RE.sub("", text)).strip()


def unescape_entities(text: str) -> str:
    """Decode HTML entities.

    Typhoon OCR 1.5 renders tables as HTML, so a company name like
    "Orient Paper & Print" comes back as "Orient Paper &amp; Print".
    """
    return html.unescape(text) if text and "&" in text else (text or "")


def clean_value(text: str) -> str:
    """Light cleaning applied to every extracted string value."""
    cleaned = collapse_whitespace(unescape_entities(unicodedata.normalize("NFC", text or "")))
    # OCR often leaves a trailing label separator or stray bullet.
    return cleaned.strip(" :;|-•·,")


def normalize_for_match(text: str) -> str:
    """Aggressive normalisation used only to compare two strings.

    Case-folded, punctuation-stripped, digit-unified and whitespace-collapsed so
    that "INV-2024/0142" and "inv 2024 0142" compare equal when we check whether
    a model's quoted evidence really appears in the OCR text.
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", unescape_entities(_HTML_TAG_RE.sub(" ", text))).casefold()
    folded = thai_digits_to_arabic(folded)
    folded = _NOISE_RE.sub("", folded)
    folded = _PUNCT_RE.sub(" ", folded)
    return _WHITESPACE_RE.sub(" ", folded).strip()


def squash(text: str) -> str:
    """Normalised text with all spaces removed - the loosest comparison form."""
    return normalize_for_match(text).replace(" ", "")


def contains_normalized(haystack: str, needle: str) -> bool:
    """Substring test that ignores case, punctuation and spacing differences."""
    if not needle:
        return False
    return squash(needle) in squash(haystack)
