"""Tools the reconciliation agent can call.

Each tool does one thing the model must not do from memory: re-read the page,
look up a master record, or compute arithmetic. Descriptions are deliberately
short - a long description costs context on every turn and rarely helps.

Every tool returns a plain JSON-serialisable dict and never raises.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from rapidfuzz import fuzz

from docuflow.masterdata.repository import MasterDataRepository
from docuflow.normalize.identifiers import thai_tax_id_checksum_ok
from docuflow.normalize.text import normalize_for_match
from docuflow.store.records import RecordStore

logger = logging.getLogger(__name__)

MAX_SEARCH_HITS = 8
SEARCH_CONTEXT_CHARS = 160
THAI_TAX_ID_LENGTH = 13
# Returned when a search finds nothing exactly, so the agent has something to
# read instead of another guess.
NEAREST_LINES = 3
NEAREST_MIN_SIMILARITY = 55
MAX_LINE_ITEMS = 40

# One line per tool, for the screen where someone writes a rule and for the
# authoring prompt. The toolbox itself is built per document, so this is the
# only way to name the tools before a document exists; a test pins the two
# lists together so a tool cannot be added here and forgotten there.
TOOL_HELP: dict[str, str] = {
    "read_document": "search the recognised page text for a word or number",
    "read_line_items": "list the table rows already read from the document, with amounts",
    "lookup_vendor": "find a vendor in the master file by name or tax id",
    "lookup_purchase_order": "fetch a purchase order: vendor, amount, currency, status",
    "lookup_contract": "fetch a contract from the register: parties, dates, value",
    "find_posted_records": "check whether a reference was already posted downstream",
    "check_arithmetic": "recompute subtotal + tax and compare with the stated total",
    "validate_tax_id": "run the Thai 13-digit tax id check-digit test",
}

TOOL_NAMES: frozenset[str] = frozenset(TOOL_HELP)


@dataclass(frozen=True)
class Tool:
    """A callable exposed to the agent."""

    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[dict[str, Any]], dict[str, Any]]

    def schema(self) -> dict[str, Any]:
        """OpenAI-compatible function-tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _string(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def _number(description: str) -> dict[str, Any]:
    return {"type": "number", "description": description}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _read_document(ocr_text: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    lines = ocr_text.splitlines()
    normalized_lines = [normalize_for_match(line) for line in lines]

    def run(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"error": "query is required"}
        needle = normalize_for_match(query)
        if not needle:
            return {"matches": [], "note": "query had no searchable content"}

        matches = [
            {"line": index + 1, "text": lines[index].strip()[:SEARCH_CONTEXT_CHARS]}
            for index, normalized in enumerate(normalized_lines)
            if needle in normalized
        ][:MAX_SEARCH_HITS]

        if matches:
            return {"matches": matches, "found": True}

        # An exact miss used to end here, with nothing to go on. Watched on a
        # real document, an agent answered that by guessing again - five
        # searches for a purchase order under five spellings of the word, each
        # one costing a turn and telling it nothing. The nearest lines cost a
        # fuzzy pass over text already in memory and usually end the hunt.
        return {
            "matches": [],
            "found": False,
            "note": f"'{query}' does not appear in the document text",
            "closest_lines": _nearest_lines(query, lines),
            "hint": "These are the most similar lines, not a match. If none of "
                    "them says what you were looking for, the document does not.",
        }

    return run


def _nearest_lines(query: str, lines: list[str]) -> list[dict[str, Any]]:
    """The lines that look most like the query, with their similarity."""
    scored = [
        (fuzz.partial_ratio(normalize_for_match(query), normalized), index)
        for index, normalized in ((i, normalize_for_match(line)) for i, line in enumerate(lines))
        if normalized
    ]
    scored.sort(reverse=True)
    return [
        {"line": index + 1, "similarity": round(score), "text": lines[index].strip()[:SEARCH_CONTEXT_CHARS]}
        for score, index in scored[:NEAREST_LINES]
        if score >= NEAREST_MIN_SIMILARITY
    ]


def _on_the_page(value: str, ocr_text: str) -> bool:
    return normalize_for_match(str(value)) in normalize_for_match(ocr_text)


def _off_document_note(label: str, value: Any, ocr_text: str) -> dict[str, Any]:
    """Flag an identifier that is not printed on the document being examined.

    Watched on a real run: the agent called find_posted_records with a vendor id
    belonging to a different company, and the tool answered as though the pairing
    made sense. A lookup with an invented argument returns a true answer to a
    question nobody asked, which is the most expensive kind of wrong. Catching it
    here is cheaper than catching it after it reaches a reviewer.
    """
    if value in (None, "") or _on_the_page(value, ocr_text):
        return {}
    return {"argument_warning": f"{label} '{value}' does not appear on this document. "
                                "If you did not read it off the page, say where it came from."}


def _lookup_vendor(master: MasterDataRepository) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        name = args.get("name")
        tax_id = args.get("tax_id")
        if not name and not tax_id:
            return {"error": "supply name or tax_id"}
        match = master.find_vendor(str(name) if name else None, str(tax_id) if tax_id else None)
        if not match.matched:
            # Name-similarity suggestions only. Spelled out because an agent
            # given a bare list of names will otherwise report the nearest one
            # as "the actual vendor" - a confident, invented identification.
            return {
                "found": False,
                "note": (
                    f"No vendor matches '{name or tax_id}'. This vendor is not registered."
                ),
                "best_score": match.score,
                "similar_names_only": [candidate["legal_name"] for candidate in match.candidates[:3]],
                "warning": (
                    "similar_names_only lists names that merely look alike. They are NOT "
                    "this vendor and their identifiers do not belong to this document."
                ),
            }
        vendor = match.vendor
        return {
            "found": True,
            "vendor_id": vendor["vendor_id"],
            "legal_name": vendor["legal_name"],
            "tax_id": vendor.get("tax_id"),
            "bank_account": vendor.get("bank_account"),
            "status": vendor.get("status"),
            "payment_terms": vendor.get("payment_terms"),
            "currency": vendor.get("currency"),
            "matched_by": match.method,
            "match_score": match.score,
        }

    return run


def _lookup_purchase_order(
    master: MasterDataRepository, ocr_text: str
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        number = args.get("po_number")
        if not number:
            return {"error": "po_number is required"}
        warning = _off_document_note("purchase order", number, ocr_text)
        order = master.get_purchase_order(str(number))
        if order is None:
            return {"found": False,
                    "note": f"purchase order {number} is not in the system", **warning}
        vendor = master.get_vendor(order["vendor_id"])
        return {
            "found": True,
            **order,
            "vendor_name": vendor["legal_name"] if vendor else None,
            # A purchase order records what was agreed, before tax. An invoice
            # total includes it. An agent handed a bare `amount` has no way to
            # know that, compares the total against it, and reports every
            # correct invoice as roughly the VAT rate over - which is what
            # happened: a clean 135,000 order was failed for a 144,450 total.
            #
            # Said by the tool rather than left to each rule's wording, because
            # the tool owns the number and a rule can be written by anyone,
            # including a model that has never been told.
            "amount_basis": "net of tax",
            "comparison_note": (
                f"{order['amount']} excludes VAT. Compare it against the invoice "
                f"subtotal, never the total - the total includes tax and will "
                f"look over the order by the tax rate."
            ),
            **warning,
        }

    return run


def _lookup_contract(
    master: MasterDataRepository, ocr_text: str
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        number = args.get("contract_number")
        if not number:
            return {"error": "contract_number is required"}
        warning = _off_document_note("contract", number, ocr_text)
        contract = master.get_contract(str(number))
        if contract is None:
            return {"found": False,
                    "note": f"contract {number} is not in the register", **warning}
        return {"found": True, **contract, **warning}

    return run


def _find_posted_records(
    records: RecordStore,
    master: MasterDataRepository,
    doc_type: str,
    document_id: str,
    ocr_text: str,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        reference = args.get("reference")
        if not reference:
            return {"error": "reference is required"}

        notes = dict(_off_document_note("reference", reference, ocr_text))
        # A vendor id is a system identifier, never printed on a page, so the
        # off-document test does not apply - but it can still be invented, and
        # scoping a duplicate search to the wrong company quietly turns "already
        # paid" into "nothing found".
        vendor_id = args.get("vendor_id")
        if vendor_id:
            vendor = master.get_vendor(str(vendor_id))
            if vendor is None:
                notes["argument_warning"] = (
                    f"vendor_id '{vendor_id}' is not in the vendor master file, so this "
                    f"search was scoped to a company that does not exist."
                )
            else:
                notes["scoped_to_vendor"] = vendor["legal_name"]

        existing = records.find_duplicate(
            doc_type, str(reference), vendor_id, exclude_document_id=document_id
        )
        if existing is None:
            return {"found": False, "note": f"nothing posted under reference {reference}", **notes}
        return {
            "found": True,
            "record_id": existing["record_id"],
            "reference": existing["reference"],
            "amount": existing["amount"],
            "currency": existing["currency"],
            "posted_at": existing["posted_at"],
            "counterparty": existing["counterparty"],
            **notes,
        }

    return run


def _read_line_items(line_items: tuple) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The table rows, already parsed.

    An agent asked about "the largest line" would otherwise search the page text
    and read a total, or a page number, as an amount. These are the rows the
    extractor found, which is also what the deterministic rules were checking.
    """

    def run(args: dict[str, Any]) -> dict[str, Any]:
        rows = [
            {
                "line": index,
                "description": item.description,
                "quantity": item.quantity,
                "unit_price": item.unit_price,
                "amount": item.amount,
            }
            for index, item in enumerate(line_items[:MAX_LINE_ITEMS], start=1)
        ]
        if not rows:
            return {"count": 0, "items": [],
                    "note": "no line items were read from this document"}
        amounts = [row["amount"] for row in rows if row["amount"] is not None]
        return {
            "count": len(rows),
            "items": rows,
            "total_of_amounts": round(sum(amounts), 2) if amounts else None,
            "largest_amount": max(amounts) if amounts else None,
        }

    return run


def _check_arithmetic(args: dict[str, Any]) -> dict[str, Any]:
    """Deterministic arithmetic - the model must never do this itself."""

    def number(key: str) -> float | None:
        value = args.get(key)
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    subtotal, tax_rate = number("subtotal"), number("tax_rate")
    tax_amount, total = number("tax_amount"), number("total")

    result: dict[str, Any] = {"inputs": {"subtotal": subtotal, "tax_rate": tax_rate,
                                         "tax_amount": tax_amount, "total": total}}
    problems: list[str] = []

    if subtotal is not None and tax_rate is not None:
        expected_tax = round(subtotal * tax_rate / 100.0, 2)
        result["expected_tax_amount"] = expected_tax
        if tax_amount is not None and abs(expected_tax - tax_amount) > 0.02:
            problems.append(
                f"tax of {tax_amount:,.2f} does not equal {tax_rate:g}% of {subtotal:,.2f} "
                f"({expected_tax:,.2f})"
            )

    if subtotal is not None:
        computed_total = round(subtotal + (tax_amount or 0.0), 2)
        result["computed_total"] = computed_total
        if total is not None:
            difference = round(computed_total - total, 2)
            result["difference"] = difference
            if abs(difference) > 0.02:
                problems.append(
                    f"subtotal + tax = {computed_total:,.2f} but the document states {total:,.2f} "
                    f"(off by {abs(difference):,.2f})"
                )

    result["consistent"] = not problems
    result["problems"] = problems
    return result


def _validate_tax_id(args: dict[str, Any]) -> dict[str, Any]:
    raw = str(args.get("tax_id", ""))
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return {"valid": False, "note": "no digits supplied"}
    if len(digits) != THAI_TAX_ID_LENGTH:
        return {
            "valid": None,
            "digits": digits,
            "note": f"{len(digits)} digits; the Thai check-digit test only applies to 13-digit ids",
        }
    valid = thai_tax_id_checksum_ok(digits)
    return {
        "valid": valid,
        "digits": digits,
        "note": "check digit is correct" if valid else "check digit is wrong - likely mis-read",
    }


# ---------------------------------------------------------------------------
# Toolbox assembly
# ---------------------------------------------------------------------------


def build_toolbox(
    ocr_text: str,
    master: MasterDataRepository,
    records: RecordStore,
    doc_type: str,
    document_id: str,
    line_items: tuple = (),
) -> list[Tool]:
    """The tools available for one document."""
    return [
        Tool(
            name="read_line_items",
            description="The table rows already read from this document, with their amounts. Use this rather than searching the page for a row.",
            parameters={"type": "object", "properties": {}},
            run=_read_line_items(line_items),
        ),
        Tool(
            name="read_document",
            description="Search the recognised page text. Use this to check what is actually printed before claiming a value.",
            parameters={
                "type": "object",
                "properties": {"query": _string("Text or number to look for, e.g. 'Amount Due' or '86,480'")},
                "required": ["query"],
            },
            run=_read_document(ocr_text),
        ),
        Tool(
            name="lookup_vendor",
            description="Find a vendor in the master file by name or tax ID. Returns registered bank account and status.",
            parameters={
                "type": "object",
                "properties": {
                    "name": _string("Vendor name as printed on the document"),
                    "tax_id": _string("Vendor tax identification number"),
                },
            },
            run=_lookup_vendor(master),
        ),
        Tool(
            name="lookup_purchase_order",
            description="Fetch a purchase order: its vendor, amount, currency and status.",
            parameters={
                "type": "object",
                "properties": {"po_number": _string("Purchase order number, e.g. PO-2026-0412")},
                "required": ["po_number"],
            },
            run=_lookup_purchase_order(master, ocr_text),
        ),
        Tool(
            name="lookup_contract",
            description="Fetch a contract from the register: parties, dates and value.",
            parameters={
                "type": "object",
                "properties": {"contract_number": _string("Contract number, e.g. CT-2025-018")},
                "required": ["contract_number"],
            },
            run=_lookup_contract(master, ocr_text),
        ),
        Tool(
            name="find_posted_records",
            description="Check whether a reference was already posted downstream. Use this to confirm a suspected duplicate.",
            parameters={
                "type": "object",
                "properties": {
                    "reference": _string("Invoice, receipt or contract number"),
                    "vendor_id": _string("Vendor id such as V-1001, to scope the search"),
                },
                "required": ["reference"],
            },
            run=_find_posted_records(records, master, doc_type, document_id, ocr_text),
        ),
        Tool(
            name="check_arithmetic",
            description="Compute subtotal + tax and compare with the stated total. Always use this instead of calculating yourself.",
            parameters={
                "type": "object",
                "properties": {
                    "subtotal": _number("Amount before tax"),
                    "tax_rate": _number("Tax percentage, e.g. 7"),
                    "tax_amount": _number("Tax amount charged"),
                    "total": _number("Total stated on the document"),
                },
            },
            run=_check_arithmetic,
        ),
        Tool(
            name="validate_tax_id",
            description="Run the Thai 13-digit tax ID check-digit test.",
            parameters={
                "type": "object",
                "properties": {"tax_id": _string("Tax identification number")},
                "required": ["tax_id"],
            },
            run=_validate_tax_id,
        ),
    ]


def select_tools(tools: list[Tool], wanted: Iterable[str]) -> list[Tool]:
    """Narrow a toolbox to the named tools; an empty selection means all of them.

    A rule that only ever needs the purchase-order register should not be handed
    the duplicate checker: fewer tools is a shorter prompt, a cheaper turn and
    one less thing an agent can wander into.
    """
    names = frozenset(wanted)
    if not names:
        return tools
    return [tool for tool in tools if tool.name in names]


def to_schemas(tools: list[Tool]) -> list[dict[str, Any]]:
    return [tool.schema() for tool in tools]


def run_tool(tools: list[Tool], name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call, converting any failure into a result the model can read."""
    tool = next((item for item in tools if item.name == name), None)
    if tool is None:
        return {"error": f"unknown tool '{name}'"}
    try:
        return tool.run(args or {})
    except Exception as exc:  # noqa: BLE001 - a tool failure must not end the run
        logger.warning("tool %s failed: %s", name, exc)
        return {"error": f"{name} failed: {exc}"}
