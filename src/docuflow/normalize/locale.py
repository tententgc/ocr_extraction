"""Working out a document's currency when the document never states one.

Plenty of invoices print amounts and no unit at all - a Thai supplier billing a
Thai buyer has no reason to write "THB" any more than a corner shop writes it on
a receipt. The amounts are still in a currency, and a ledger cannot hold "1,070
of something".

The order here is the order of how much a signal is worth:

  1. What the page says. A "฿" or the word "baht" is the document telling you,
     and it is handled before this module is reached.
  2. What the register says. A vendor already onboarded has a billing currency
     on file; that is a fact about the relationship, not a guess about the page.
  3. Where the document is from. Thai script, a Thai tax identifier, a country
     named in an address. This is genuinely an inference and is labelled as one.

Every answer comes back with the reason for it, because a currency nobody can
explain is worse than an empty field - the empty field at least stops.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Country or locale marker -> currency. Only entries where the mapping is
# unambiguous: a country with one national currency for commercial invoices.
COUNTRY_CURRENCY: dict[str, str] = {
    "thailand": "THB", "ประเทศไทย": "THB", "กรุงเทพ": "THB", "bangkok": "THB",
    "singapore": "SGD",
    "japan": "JPY", "tokyo": "JPY",
    "united states": "USD", "u.s.a": "USD", "usa": "USD",
    "united kingdom": "GBP", "london": "GBP",
    "germany": "EUR", "france": "EUR", "netherlands": "EUR", "spain": "EUR",
    "italy": "EUR", "ireland": "EUR", "belgium": "EUR", "austria": "EUR",
    "china": "CNY", "shanghai": "CNY", "shenzhen": "CNY", "beijing": "CNY",
}

# Thai script anywhere on a commercial document is a strong signal on its own.
_THAI_SCRIPT = re.compile(r"[ก-๙]")
# A 13-digit Thai taxpayer identifier, with or without the usual separators.
_THAI_TAX_ID = re.compile(r"\b\d[\d\s-]{11,17}\d\b")
# A document written in Thai is mostly Thai characters. A few percent is an
# address line on an invoice from somewhere else - a signal about where the
# goods are going, not about who is billing or in what.
_THAI_SCRIPT_SHARE = 0.15


@dataclass(frozen=True)
class CurrencyGuess:
    """A currency and the reason it was chosen."""

    currency: str
    reason: str
    # True when a register or a fact settled it rather than an inference.
    from_record: bool = False


def currency_from_vendor(vendor: dict | None) -> CurrencyGuess | None:
    """The billing currency already on file for this counterparty."""
    if not vendor or not vendor.get("currency"):
        return None
    return CurrencyGuess(
        currency=str(vendor["currency"]).upper(),
        reason=f"{vendor.get('legal_name', 'this vendor')} bills in "
               f"{str(vendor['currency']).upper()} in the vendor master file",
        from_record=True,
    )


def currency_from_country(text: str) -> CurrencyGuess | None:
    """Infer the currency from where the document appears to be from."""
    if not text:
        return None

    lowered = text.casefold()

    # Thai first, but only when the document is genuinely written in Thai: the
    # script then outranks any place name, because a Thai invoice shipping goods
    # abroad still bills in baht.
    thai_characters = len(_THAI_SCRIPT.findall(text))
    if thai_characters and thai_characters / max(len(text), 1) >= _THAI_SCRIPT_SHARE:
        return CurrencyGuess("THB", "the document is written in Thai")

    # When a document names more than one country, the earliest one wins. On an
    # invoice the issuer is at the top and the ship-to address is below it, and
    # the currency follows whoever is billing - not where the goods are going.
    matches = [(lowered.find(marker), marker) for marker in COUNTRY_CURRENCY if marker in lowered]
    if matches:
        position, marker = min(matches)
        return CurrencyGuess(
            COUNTRY_CURRENCY[marker], f"the document names {marker.title()}"
        )

    # A Thai taxpayer identifier is 13 digits; other countries do not use that
    # length, so it identifies the jurisdiction on its own.
    for candidate in _THAI_TAX_ID.findall(text):
        if len(re.sub(r"\D", "", candidate)) == 13:
            return CurrencyGuess("THB", "the document carries a 13-digit Thai tax identifier")

    return None


def infer_currency(
    text: str, vendor: dict | None = None, allowed: tuple[str, ...] = ()
) -> CurrencyGuess | None:
    """Best available answer for a document that never printed a currency."""
    for guess in (currency_from_vendor(vendor), currency_from_country(text)):
        if guess is not None and (not allowed or guess.currency in allowed):
            return guess
    return None
