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
from typing import Any, Callable

from docuflow.masterdata.repository import MasterDataRepository
from docuflow.normalize.identifiers import thai_tax_id_checksum_ok
from docuflow.normalize.text import normalize_for_match
from docuflow.store.records import RecordStore

logger = logging.getLogger(__name__)

MAX_SEARCH_HITS = 8
SEARCH_CONTEXT_CHARS = 160
THAI_TAX_ID_LENGTH = 13


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
        return {
            "matches": [],
            "found": False,
            "note": f"'{query}' does not appear in the document text",
        }

    return run


def _lookup_vendor(master: MasterDataRepository) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        name = args.get("name")
        tax_id = args.get("tax_id")
        if not name and not tax_id:
            return {"error": "supply name or tax_id"}
        match = master.find_vendor(str(name) if name else None, str(tax_id) if tax_id else None)
        if not match.matched:
            return {
                "found": False,
                "best_score": match.score,
                "closest": [candidate["legal_name"] for candidate in match.candidates[:3]],
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


def _lookup_purchase_order(master: MasterDataRepository) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        number = args.get("po_number")
        if not number:
            return {"error": "po_number is required"}
        order = master.get_purchase_order(str(number))
        if order is None:
            return {"found": False, "note": f"purchase order {number} is not in the system"}
        vendor = master.get_vendor(order["vendor_id"])
        return {
            "found": True,
            **order,
            "vendor_name": vendor["legal_name"] if vendor else None,
        }

    return run


def _lookup_contract(master: MasterDataRepository) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        number = args.get("contract_number")
        if not number:
            return {"error": "contract_number is required"}
        contract = master.get_contract(str(number))
        if contract is None:
            return {"found": False, "note": f"contract {number} is not in the register"}
        return {"found": True, **contract}

    return run


def _find_posted_records(
    records: RecordStore, doc_type: str, document_id: str
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        reference = args.get("reference")
        if not reference:
            return {"error": "reference is required"}
        existing = records.find_duplicate(
            doc_type, str(reference), args.get("vendor_id"), exclude_document_id=document_id
        )
        if existing is None:
            return {"found": False, "note": f"nothing posted under reference {reference}"}
        return {
            "found": True,
            "record_id": existing["record_id"],
            "reference": existing["reference"],
            "amount": existing["amount"],
            "currency": existing["currency"],
            "posted_at": existing["posted_at"],
            "counterparty": existing["counterparty"],
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
) -> list[Tool]:
    """The tools available for one document."""
    return [
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
            run=_lookup_purchase_order(master),
        ),
        Tool(
            name="lookup_contract",
            description="Fetch a contract from the register: parties, dates and value.",
            parameters={
                "type": "object",
                "properties": {"contract_number": _string("Contract number, e.g. CT-2025-018")},
                "required": ["contract_number"],
            },
            run=_lookup_contract(master),
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
            run=_find_posted_records(records, doc_type, document_id),
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
