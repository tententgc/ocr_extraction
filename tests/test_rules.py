"""Business rules and the routing policy - the guards on the ledger."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from docuflow.masterdata.repository import MasterDataRepository
from docuflow.models import RuleFinding
from docuflow.routing import decide
from docuflow.rules import FREE_MODE_RULE_ID, RuleContext, run_rules
from docuflow.schemas.registry import get_schema

GOOD_INVOICE = {
    "invoice_number": "INV-2026-0731",
    "invoice_date": "2026-07-08",
    "due_date": "2026-08-07",
    "vendor_name": "Siam Tech Supply Co., Ltd.",
    "vendor_tax_id": "0105543000129",
    "purchase_order_ref": "PO-2026-0412",
    "currency": "THB",
    "subtotal": 187_000.00,
    "tax_rate": 7.0,
    "tax_amount": 13_090.00,
    "total_amount": 200_090.00,
    "payment_terms": "Net 30",
    "bank_account": "1234567890",
}


def build_context(
    master: MasterDataRepository,
    fields: dict,
    doc_type: str = "invoice",
    duplicate=None,
    degraded: bool = False,
    make_run=None,
    today: date = date(2026, 8, 3),
) -> RuleContext:
    run = make_run(fields, doc_type=doc_type)
    schema = get_schema(doc_type)
    name = run.value("vendor_name") or run.value("merchant_name") or run.value("applicant_name")
    tax_id = run.value("vendor_tax_id") or run.value("tax_id")
    return RuleContext(
        document_id="DOC-TEST",
        doc_type=doc_type,
        schema=schema,
        fields=run.fields,
        line_items=(),
        master=master,
        vendor_match=master.find_vendor(str(name) if name else None, str(tax_id) if tax_id else None),
        ocr_text="",
        ocr_degraded=degraded,
        amount_tolerance=0.02,
        today=today,
        duplicate_of=duplicate,
        extras={"field_review_confidence": 0.75},
    )


def result_for(findings: tuple[RuleFinding, ...], rule_id: str) -> RuleFinding | None:
    return next((finding for finding in findings if finding.rule_id == rule_id), None)


class TestInvoiceArithmetic:
    def test_consistent_totals_pass(self, master, make_run):
        findings = run_rules(build_context(master, GOOD_INVOICE, make_run=make_run))
        assert result_for(findings, "invoice_totals_consistent").passed is True

    def test_totals_that_do_not_add_up_block(self, master, make_run):
        fields = {**GOOD_INVOICE, "total_amount": 86_480.00, "subtotal": 64_000.00, "tax_amount": 4_480.00}
        finding = result_for(run_rules(build_context(master, fields, make_run=make_run)), "invoice_totals_consistent")
        assert finding.passed is False
        assert finding.severity == "blocking"
        assert finding.detail["difference"] == 18_000.00

    def test_rounding_slack_is_tolerated(self, master, make_run):
        fields = {**GOOD_INVOICE, "total_amount": 200_090.01}
        assert result_for(run_rules(build_context(master, fields, make_run=make_run)), "invoice_totals_consistent").passed

    def test_tax_amount_must_match_the_stated_rate(self, master, make_run):
        fields = {**GOOD_INVOICE, "tax_amount": 9_350.00, "total_amount": 196_350.00}
        finding = result_for(run_rules(build_context(master, fields, make_run=make_run)), "tax_rate_consistent")
        assert finding.passed is False
        assert finding.severity == "blocking"


class TestMasterDataChecks:
    def test_known_vendor_matches_by_tax_id(self, master, make_run):
        finding = result_for(run_rules(build_context(master, GOOD_INVOICE, make_run=make_run)), "vendor_known")
        assert finding.passed is True
        assert finding.detail["vendor_id"] == "V-1001"

    def test_unknown_vendor_blocks(self, master, make_run):
        fields = {**GOOD_INVOICE, "vendor_name": "Summit Solutions Trading", "vendor_tax_id": "0105562099887"}
        finding = result_for(run_rules(build_context(master, fields, make_run=make_run)), "vendor_known")
        assert finding.passed is False
        assert finding.severity == "blocking"

    def test_ocr_damaged_vendor_name_still_matches(self, master):
        # "แฟชิลิตี้" for "แฟซิลิตี้" is a single-character OCR slip.
        match = master.find_vendor("บริษัท เจ้าพระยา แฟชิลิตี้ จำกัด", None)
        assert match.matched is True
        assert match.vendor["vendor_id"] == "V-1004"
        assert match.method == "fuzzy name"

    def test_vendor_on_hold_blocks(self, master, make_run):
        fields = {
            "po_number": "PO-2026-0501",
            "po_date": "2026-07-22",
            "vendor_name": "Orient Paper & Print Co., Ltd.",
            "currency": "THB",
            "total_amount": 48_150.00,
        }
        finding = result_for(
            run_rules(build_context(master, fields, doc_type="purchase_order", make_run=make_run)), "vendor_active"
        )
        assert finding.passed is False
        assert finding.severity == "blocking"

    def test_changed_bank_account_blocks(self, master, make_run):
        fields = {**GOOD_INVOICE, "bank_account": "7778889990"}
        finding = result_for(
            run_rules(build_context(master, fields, make_run=make_run)), "bank_account_matches_vendor"
        )
        assert finding.passed is False
        assert finding.severity == "blocking"
        assert finding.detail["registered"] == "1234567890"


class TestPurchaseOrderMatching:
    def test_matching_po_passes(self, master, make_run):
        findings = run_rules(build_context(master, GOOD_INVOICE, make_run=make_run))
        assert result_for(findings, "po_exists").passed is True
        assert result_for(findings, "po_amount_within_tolerance").passed is True

    def test_missing_po_blocks(self, master, make_run):
        fields = {**GOOD_INVOICE, "purchase_order_ref": "PO-9999-9999"}
        finding = result_for(run_rules(build_context(master, fields, make_run=make_run)), "po_exists")
        assert finding.passed is False
        assert finding.severity == "blocking"

    def test_overrun_beyond_tolerance_blocks(self, master, make_run):
        fields = {**GOOD_INVOICE, "subtotal": 210_000.00, "tax_amount": 14_700.00, "total_amount": 224_700.00}
        finding = result_for(
            run_rules(build_context(master, fields, make_run=make_run)), "po_amount_within_tolerance"
        )
        assert finding.passed is False
        assert finding.severity == "blocking"

    def test_large_invoice_without_a_po_blocks(self, master, make_run):
        fields = {**GOOD_INVOICE, "purchase_order_ref": None}
        finding = result_for(run_rules(build_context(master, fields, make_run=make_run)), "purchase_order_required")
        assert finding.passed is False

    def test_small_invoice_without_a_po_is_allowed(self, master, make_run):
        fields = {
            **GOOD_INVOICE,
            "purchase_order_ref": None,
            "subtotal": 10_000.00,
            "tax_amount": 700.00,
            "total_amount": 10_700.00,
        }
        assert result_for(run_rules(build_context(master, fields, make_run=make_run)), "purchase_order_required").passed


class TestDuplicates:
    def test_no_prior_record_passes(self, master, make_run):
        assert result_for(run_rules(build_context(master, GOOD_INVOICE, make_run=make_run)), "duplicate_document").passed

    def test_prior_record_blocks(self, master, make_run):
        existing = {"record_id": "REC-1", "posted_at": "2026-07-10", "amount": 200090.0, "currency": "THB"}
        finding = result_for(
            run_rules(build_context(master, GOOD_INVOICE, duplicate=existing, make_run=make_run)),
            "duplicate_document",
        )
        assert finding.passed is False
        assert finding.severity == "blocking"


class TestGenericRules:
    def test_invalid_thai_tax_id_blocks(self, master, make_run):
        fields = {**GOOD_INVOICE, "vendor_tax_id": "0105545000124"}
        finding = result_for(run_rules(build_context(master, fields, make_run=make_run)), "tax_id_checksum")
        assert finding.passed is False
        assert finding.severity == "blocking"

    def test_a_slightly_future_date_warns(self, master, make_run):
        """Days ahead is a clock, a timezone, or a supplier billing early."""
        fields = {**GOOD_INVOICE, "invoice_date": "2026-08-20"}   # today is 2026-08-03
        finding = result_for(run_rules(build_context(master, fields, make_run=make_run)),
                             "dates_not_in_future")
        assert finding.passed is False
        assert finding.severity == "warning"

    def test_a_date_years_ahead_blocks(self, master, make_run):
        """Found by an actual silent error, which is why this is graded now.

        A faded photocopy read 2026 as 2028 and its invoice number as
        `PRF.SG20496` where the page said `PRF-SG-20416`. Both values were
        grounded - grounding proves a quote is in the OCR text, and the OCR text
        was what was wrong - and the document posted automatically at 85%.

        This rule was the only one that fired, and as a warning it stopped
        nothing. A date two years out is not a document issued early; it is a
        digit misread by the same engine that read the rest of the page.
        """
        fields = {**GOOD_INVOICE, "invoice_date": "2028-07-17"}
        finding = result_for(run_rules(build_context(master, fields, make_run=make_run)),
                             "dates_not_in_future")
        assert finding.passed is False
        assert finding.severity == "blocking"
        assert "misread year" in finding.message


class TestContractRules:
    CONTRACT = {
        "contract_number": "CT-2025-018",
        "contract_title": "Facility Services Agreement",
        "party_a": "Mirtphol Industries PCL",
        "party_b": "Chao Phraya Facility Services Co., Ltd.",
        "effective_date": "2025-08-01",
        "expiry_date": "2026-07-31",
        "contract_value": 368_000.00,
        "currency": "THB",
        "notice_period_days": 60,
    }

    def test_expired_contract_warns(self, master, make_run):
        """Under its own id: an agreement that ended last week is not "expiring"."""
        findings = run_rules(build_context(master, self.CONTRACT, doc_type="contract",
                                           make_run=make_run))
        finding = result_for(findings, "contract_expired")
        assert finding.passed is False
        assert finding.detail["days_remaining"] < 0
        assert "expired" in finding.message
        assert result_for(findings, "contract_expiring_soon") is None

    def test_a_contract_nearing_its_end_still_reports_as_expiring(self, master, make_run):
        fields = {**self.CONTRACT, "expiry_date": "2026-09-15"}   # today is 2026-08-03
        finding = result_for(
            run_rules(build_context(master, fields, doc_type="contract", make_run=make_run)),
            "contract_expiring_soon",
        )
        assert finding.passed is False
        assert finding.detail["days_remaining"] > 0

    def test_switching_the_check_off_silences_both_of_its_ids(self, master, make_run):
        """One function, two ids; the catalogue entry has to turn off all of it."""
        context = build_context(master, self.CONTRACT, doc_type="contract", make_run=make_run)
        findings = run_rules(context, skip={"contract_expiring_soon"})
        assert result_for(findings, "contract_expired") is None

    def test_reversed_dates_block(self, master, make_run):
        fields = {**self.CONTRACT, "effective_date": "2026-07-31", "expiry_date": "2025-08-01"}
        finding = result_for(
            run_rules(build_context(master, fields, doc_type="contract", make_run=make_run)),
            "contract_dates_ordered",
        )
        assert finding.passed is False
        assert finding.severity == "blocking"

    def test_identical_parties_block(self, master, make_run):
        fields = {**self.CONTRACT, "party_b": "Mirtphol Industries PCL"}
        finding = result_for(
            run_rules(build_context(master, fields, doc_type="contract", make_run=make_run)),
            "contract_parties_distinct",
        )
        assert finding.passed is False

    def test_value_differing_from_the_register_blocks(self, master, make_run):
        fields = {**self.CONTRACT, "contract_value": 500_000.00}
        finding = result_for(
            run_rules(build_context(master, fields, doc_type="contract", make_run=make_run)),
            "contract_matches_master",
        )
        assert finding.passed is False
        assert finding.severity == "blocking"


class TestRouting:
    def _run(self, make_run, fields, findings=(), confidence=0.95, grounded=True):
        run = make_run(fields, grounded=grounded)
        return run.evolve(findings=tuple(findings), confidence=confidence)

    def test_clean_high_confidence_document_posts(self, make_run):
        run = self._run(make_run, GOOD_INVOICE)
        assert decide(run, get_schema("invoice"), 0.90).decision == "auto_post"

    def test_blocking_finding_forces_review(self, make_run):
        blocking = RuleFinding(rule_id="x", passed=False, severity="blocking", message="totals wrong")
        run = self._run(make_run, GOOD_INVOICE, findings=[blocking])
        assert decide(run, get_schema("invoice"), 0.90).decision == "needs_review"

    def test_low_confidence_forces_review(self, make_run):
        run = self._run(make_run, GOOD_INVOICE, confidence=0.80)
        outcome = decide(run, get_schema("invoice"), 0.90)
        assert outcome.decision == "needs_review"
        assert any("below the 90%" in reason for reason in outcome.reasons)

    def test_ungrounded_critical_field_forces_review(self, make_run):
        run = self._run(make_run, GOOD_INVOICE, grounded=False)
        assert decide(run, get_schema("invoice"), 0.90).decision == "needs_review"

    def test_missing_required_field_forces_review(self, make_run):
        fields = {key: value for key, value in GOOD_INVOICE.items() if key != "invoice_number"}
        run = self._run(make_run, fields)
        assert decide(run, get_schema("invoice"), 0.90).decision == "needs_review"

    def test_document_with_no_text_is_rejected(self, make_run):
        from docuflow.models import OcrResult

        run = self._run(make_run, GOOD_INVOICE).evolve(ocr=OcrResult())
        assert decide(run, get_schema("invoice"), 0.90).decision == "rejected"


class TestUnknownVendorSeverity:
    """An unregistered counterparty means different things per document type."""

    UNKNOWN = {**GOOD_INVOICE, "vendor_name": "PTT Station Rama IV", "vendor_tax_id": None}

    def test_invoice_blocks(self, master, make_run):
        # About to pay a party nobody registered: stop.
        finding = result_for(run_rules(build_context(master, self.UNKNOWN, make_run=make_run)), "vendor_known")
        assert finding.passed is False
        assert finding.severity == "blocking"

    def test_receipt_only_warns(self, master, make_run):
        # Money already spent at a petrol station; it was never going to be an
        # AP vendor. Blocking this was a modelling error.
        fields = {
            "receipt_number": "R-770412",
            "receipt_date": "2026-07-17",
            "merchant_name": "PTT Station Rama IV",
            "total_amount": 1500.00,
            "currency": "THB",
        }
        finding = result_for(
            run_rules(build_context(master, fields, doc_type="receipt", make_run=make_run)), "vendor_known"
        )
        assert finding.passed is False
        assert finding.severity == "warning"

    def test_onboarding_form_only_warns(self, master, make_run):
        # The applicant is not registered *because that is what the form is for*.
        fields = {
            "form_type": "Vendor Onboarding",
            "applicant_name": "Andaman Marine Supply Co., Ltd.",
            "reference_number": "VOF-2026-0219",
        }
        finding = result_for(
            run_rules(build_context(master, fields, doc_type="form", make_run=make_run)), "vendor_known"
        )
        assert finding.passed is False
        assert finding.severity == "warning"

    def test_registering_the_vendor_clears_it(self, master, make_run):
        master.add_vendor("PTT Station Rama IV", tax_id="0107544000108")
        finding = result_for(run_rules(build_context(master, self.UNKNOWN, make_run=make_run)), "vendor_known")
        assert finding.passed is True


class TestPolicyIsConfigurable:
    """Finance thresholds come from settings, not from constants in the rules."""

    def test_purchase_order_threshold_is_read_from_policy(self, master, make_run):
        from docuflow.rules.context import RuleContext

        fields = {**GOOD_INVOICE, "purchase_order_ref": None}
        base = build_context(master, fields, make_run=make_run)

        strict = replace(base, po_required_above=1_000.0)
        assert result_for(run_rules(strict), "purchase_order_required").passed is False

        lenient = replace(base, po_required_above=10_000_000.0)
        assert result_for(run_rules(lenient), "purchase_order_required").passed is True

    def test_expected_vat_rates_are_read_from_policy(self, master, make_run):
        fields = {**GOOD_INVOICE, "tax_rate": 10.0, "tax_amount": 18_700.00, "total_amount": 205_700.00}
        base = build_context(master, fields, make_run=make_run)

        assert result_for(run_rules(base), "vat_rate_expected").passed is False
        allowed = replace(base, expected_vat_rates=(0.0, 7.0, 10.0))
        assert result_for(run_rules(allowed), "vat_rate_expected").passed is True

    def test_blocking_document_types_are_read_from_policy(self, master, make_run):
        fields = {**GOOD_INVOICE, "vendor_name": "Nobody Ltd.", "vendor_tax_id": None}
        base = build_context(master, fields, make_run=make_run)

        relaxed = replace(base, unknown_vendor_blocks=frozenset())
        assert result_for(run_rules(relaxed), "vendor_known").severity == "warning"


class TestFreeMode:
    """Extraction-only mode: nothing is checked, and that has to be visible."""

    UNKNOWN = {**GOOD_INVOICE, "vendor_name": "Nobody Ltd.", "vendor_tax_id": None}

    def test_no_rule_is_evaluated(self, master, make_run):
        findings = run_rules(build_context(master, GOOD_INVOICE, make_run=make_run), free_mode=True)
        assert [finding.rule_id for finding in findings] == [FREE_MODE_RULE_ID]

    def test_nothing_can_block(self, master, make_run):
        context = build_context(master, self.UNKNOWN, make_run=make_run)
        assert result_for(run_rules(context), "vendor_known").severity == "blocking"

        findings = run_rules(context, free_mode=True)
        assert not [finding for finding in findings if finding.severity == "blocking"]

    def test_the_notice_does_not_cost_confidence(self, master, make_run):
        """It reports a fact about the run, so it must not read as a failure."""
        finding = run_rules(build_context(master, GOOD_INVOICE, make_run=make_run), free_mode=True)[0]
        assert finding.passed is True
        assert "not checked" in finding.message

    def test_routing_says_the_rules_were_skipped(self, master, make_run):
        run = make_run(GOOD_INVOICE).evolve(confidence=0.95)
        outcome = decide(run, get_schema("invoice"), 0.90, rules_evaluated=False)
        assert outcome.decision == "auto_post"
        assert "disabled" in outcome.reasons[0]


class TestDisabledRules:
    """The selective knob: silence one check, keep the others working."""

    UNKNOWN = {**GOOD_INVOICE, "vendor_name": "Nobody Ltd.", "vendor_tax_id": None}

    def test_skip_by_rule_id(self, master, make_run):
        context = build_context(master, self.UNKNOWN, make_run=make_run)
        findings = run_rules(context, skip={"vendor_known"})
        assert result_for(findings, "vendor_known") is None
        assert result_for(findings, "invoice_totals_consistent").passed is True

    def test_skip_by_function_name_when_the_id_differs(self, master, make_run):
        """`po_exists_and_matches` reports as `po_exists`; both should work."""
        fields = {**GOOD_INVOICE, "purchase_order_ref": "PO-NOT-REAL"}
        context = build_context(master, fields, make_run=make_run)

        assert result_for(run_rules(context), "po_exists").passed is False
        assert result_for(run_rules(context, skip={"po_exists"}), "po_exists") is None
        assert result_for(run_rules(context, skip={"po_exists_and_matches"}), "po_exists") is None

    def test_unknown_names_are_harmless(self, master, make_run):
        context = build_context(master, GOOD_INVOICE, make_run=make_run)
        assert run_rules(context, skip={"no_such_rule"}) == run_rules(context)
