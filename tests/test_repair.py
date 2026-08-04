"""Which value is wrong, and what it should say instead.

Knowing that three numbers do not add up is cheap. Knowing which of the three is
the misread one is the part that costs a reviewer five minutes, and it is
answerable: a single wrong value breaks every relation it appears in and leaves
the others standing, so the relations that still hold name the one that broke
them.

The failure mode this guards against is a confident wrong answer. A suggestion
carries authority - it appears next to a value with a button that accepts it -
so proposing the wrong field is worse than proposing nothing. Most of what is
tested here is the refusal to answer.
"""

from __future__ import annotations

import pytest

from docuflow.rules.repair import (
    Money,
    best_of,
    blame,
    explain_difference,
    money_fixes,
    tax_id_fixes,
    value_fix,
)


class TestFindingTheCulprit:
    def test_the_total_is_wrong_when_everything_else_agrees(self):
        """Lines sum to the subtotal and the tax is 7% of it. Only the total is odd."""
        found = blame(Money(subtotal=68_000.0, tax_amount=4_760.0, tax_rate=7.0,
                            total_amount=27_760.0, line_total=68_000.0))
        assert found is not None
        assert found.field == "total_amount"
        assert found.suggested == 72_760.0
        # The evidence is not that one relation names the total - only one ever
        # can. It is that the two relations which do *not* mention the total
        # still hold, which is what rules out the subtotal and the tax.
        assert found.certain
        assert set(found.corroborated_by) == {"rate", "lines"}

    def test_the_subtotal_is_wrong_when_the_lines_and_the_total_agree(self):
        found = blame(Money(subtotal=60_000.0, tax_amount=4_760.0, tax_rate=7.0,
                            total_amount=72_760.0, line_total=68_000.0))
        assert found is not None and found.field == "subtotal"
        assert found.suggested == 68_000.0

    def test_the_tax_is_wrong_when_the_rate_and_the_total_disagree_with_it(self):
        found = blame(Money(subtotal=68_000.0, tax_amount=4_670.0, tax_rate=7.0,
                            total_amount=72_760.0, line_total=68_000.0))
        assert found is not None and found.field == "tax_amount"
        assert found.suggested == 4_760.0

    def test_a_document_that_adds_up_has_nobody_to_blame(self):
        assert blame(Money(subtotal=68_000.0, tax_amount=4_760.0, tax_rate=7.0,
                           total_amount=72_760.0, line_total=68_000.0)) is None

    def test_rounding_is_not_an_error(self):
        assert blame(Money(subtotal=100.0, tax_amount=7.0, tax_rate=7.0, total_amount=107.01)) is None


class TestRefusingToGuess:
    """Naming the wrong field is worse than naming none."""

    def test_a_relation_missing_a_term_is_not_a_broken_relation(self):
        """A receipt with no tax line is not an arithmetic failure.

        Solving "subtotal + tax = total" for the absent tax happily produces
        899.00 and accuses a field the document never printed. Most receipts
        have no tax line; none of them are broken.
        """
        assert blame(Money(subtotal=100.0, total_amount=999.0)) is None

    def test_a_field_nothing_constrains_is_not_accused(self):
        assert blame(Money(total_amount=500.0)) is None

    def test_one_relation_and_three_unknowns_names_nobody(self):
        """Subtotal, tax and total disagree with nothing else to check them.

        Either the subtotal or the total could be the misread value and the
        arithmetic cannot choose between them.
        """
        assert blame(Money(subtotal=68_000.0, tax_amount=4_760.0, total_amount=27_760.0)) is None

    def test_an_impossible_replacement_is_not_a_candidate(self):
        """Solving a relation backwards produces a tax of minus sixty thousand.

        Ruling those out is what lets an invoice with a single active relation
        still name its culprit: two of the three candidates are arithmetic
        artefacts rather than readings anyone could have made.
        """
        found = blame(Money(subtotal=68_000.0, tax_amount=4_760.0, total_amount=100.0))
        assert found is not None and found.field == "total_amount"
        assert found.certain is False, "nothing corroborates it; there is only one relation"

    def test_an_empty_document_is_not_an_error(self):
        assert blame(Money()) is None

    def test_two_wrong_values_produce_no_suggestion(self):
        """Both the total and the lines are wrong. No single change fixes it,
        so no single change is offered."""
        assert blame(Money(subtotal=68_000.0, tax_amount=4_760.0, tax_rate=7.0,
                           total_amount=1.0, line_total=2.0)) is None


class TestNamingTheSlip:
    def test_transposed_digits(self):
        assert "transposed" in explain_difference(27_760.0, 72_760.0)

    def test_adjacent_transposition_says_so(self):
        explanation = explain_difference(75_200.0, 72_500.0)
        assert explanation and "transposed" in explanation

    def test_a_single_confusable_digit_is_called_out(self):
        explanation = explain_difference(7_070.0, 1_070.0)
        assert "one digit differs" in explanation
        assert "confuse" in explanation, "7 and 1 is the misread OCR makes most"

    def test_a_single_digit_that_is_not_a_known_confusion_is_still_located(self):
        explanation = explain_difference(1_020.0, 1_090.0)
        assert "one digit differs" in explanation
        assert "confuse" not in explanation

    def test_a_misplaced_decimal_point(self):
        assert "one place to the right" in explain_difference(4_500.0, 450.0)
        assert "two places to the left" in explain_difference(4.5, 450.0)

    def test_an_extra_digit_and_a_missing_one(self):
        assert explain_difference(11_070.0, 1_070.0) == "there is one digit too many"
        assert explain_difference(1_070.0, 11_070.0) == "a digit is missing"

    def test_two_numbers_that_are_simply_different_get_no_story(self):
        """Not every wrong number is a misread one, and inventing a reading
        error for a value the supplier got wrong misdirects the reviewer."""
        assert explain_difference(1_234.56, 9_876.54) is None

    def test_the_same_number_has_nothing_to_explain(self):
        assert explain_difference(100.0, 100.0) is None


class TestTheSuggestionItself:
    def test_it_names_the_field_the_value_and_the_reasoning(self):
        fixes = money_fixes(Money(subtotal=68_000.0, tax_amount=4_760.0, tax_rate=7.0,
                                  total_amount=27_760.0, line_total=68_000.0))
        assert len(fixes) == 1
        fix = fixes[0]
        assert fix.field == "total_amount"
        assert fix.current == 27_760.0
        assert fix.suggested == 72_760.0
        assert "transposed" in fix.reason
        assert fix.source == "arithmetic"

    def test_a_corroborated_diagnosis_is_more_confident_than_a_lone_one(self):
        corroborated = money_fixes(Money(subtotal=68_000.0, tax_amount=4_760.0, tax_rate=7.0,
                                         total_amount=27_760.0, line_total=68_000.0))[0]
        alone = money_fixes(Money(subtotal=68_000.0, tax_amount=4_760.0, total_amount=100.0))[0]

        assert corroborated.confidence > alone.confidence
        assert "the only value that explains it" in corroborated.reason
        assert "the only value that explains it" not in alone.reason

    def test_a_ninety_baht_error_on_a_seventy_thousand_baht_invoice_is_not_rounding(self):
        """The first version compared amounts within half a percent and reported
        this document as perfectly consistent."""
        found = blame(Money(subtotal=68_000.0, tax_amount=4_670.0, tax_rate=7.0,
                            total_amount=72_760.0, line_total=68_000.0))
        assert found is not None and found.field == "tax_amount"

    def test_a_correction_to_a_value_that_is_already_right_is_not_offered(self):
        assert value_fix("currency", "THB", "THB", "same") == ()

    def test_an_empty_replacement_is_not_a_correction(self):
        assert value_fix("bank_account", "123", None, "nothing to suggest") == ()


class TestTaxIdentifiers:
    # 0105543000129 is a valid Thai tax ID; damaging one digit gives a number
    # whose check digit locates the damage.
    VALID = "0105543000129"

    def test_a_valid_number_needs_no_repair(self):
        assert tax_id_fixes("vendor_tax_id", self.VALID) == ()

    def test_it_points_at_a_character_and_proposes_a_number_that_checks_out(self):
        from docuflow.normalize.identifiers import thai_tax_id_checksum_ok

        damaged = [
            self.VALID[:position] + replacement + self.VALID[position + 1:]
            for position in range(13)
            for replacement in "0123456789"
            if replacement != self.VALID[position]
        ]
        proposals = [fix for number in damaged for fix in tax_id_fixes("vendor_tax_id", number)]
        assert proposals, "some damaged numbers have exactly one plausible reading"

        for fix in proposals:
            assert thai_tax_id_checksum_ok(fix.suggested)
            assert "digit" in fix.reason

    def test_it_does_not_claim_to_have_recovered_the_original(self):
        """It cannot, and saying otherwise would be the dangerous version.

        A thirteen-digit number has a valid single-digit repair at most
        positions, so the reading this finds is frequently a different valid tax
        ID rather than the one printed. It earns its place by telling a reviewer
        where to look, not by being right.
        """
        proposals = [
            fix
            for position in range(13)
            for replacement in "0123456789"
            for fix in tax_id_fixes(
                "vendor_tax_id", self.VALID[:position] + replacement + self.VALID[position + 1:]
            )
        ]
        assert any(fix.suggested != self.VALID for fix in proposals), (
            "if this ever stops being true the wording below can be strengthened"
        )
        for fix in proposals:
            assert fix.confidence < 0.6
            assert "check this against the page" in fix.reason

    def test_a_number_of_the_wrong_length_is_not_guessed_at(self):
        assert tax_id_fixes("vendor_tax_id", "12345") == ()

    def test_only_readings_a_scanner_could_have_made_are_considered(self):
        """Passing the checksum is not on its own evidence about what was
        misread: most positions have some digit that makes the number valid, and
        the check digit can always be rewritten to match whatever precedes it.
        What narrows it is the shape - a 1 read as a 7 is a reading someone
        could have made, a 1 read as a 4 is an arithmetic coincidence."""
        every = [
            fix
            for position in range(13)
            for replacement in "0123456789"
            for fix in tax_id_fixes(
                "vendor_tax_id",
                self.VALID[:position] + replacement + self.VALID[position + 1:],
            )
        ]
        assert every, "some damaged numbers are diagnosable"
        for fix in every:
            assert "confuse" in fix.reason
            assert fix.confidence < 0.75, "a plausible reading, not a settled fact"

    def test_a_number_with_two_plausible_repairs_is_left_to_a_person(self):
        from docuflow.rules.repair import CONFUSABLE_DIGITS
        from docuflow.normalize.identifiers import thai_tax_id_checksum_ok

        ambiguous = [
            candidate
            for candidate in (
                self.VALID[:position] + replacement + self.VALID[position + 1:]
                for position in range(13)
                for replacement in "0123456789"
            )
            if not thai_tax_id_checksum_ok(candidate)
            and sum(
                1
                for position in range(13)
                for replacement in "0123456789"
                if frozenset({candidate[position], replacement}) in CONFUSABLE_DIGITS
                and thai_tax_id_checksum_ok(
                    candidate[:position] + replacement + candidate[position + 1:]
                )
            )
            > 1
        ]
        for candidate in ambiguous:
            assert tax_id_fixes("vendor_tax_id", candidate) == ()


class TestAgainstARealRule:
    """The diagnosis is only worth anything where the reviewer sees it."""

    def _context(self, master, make_run, fields, rows=()):
        from dataclasses import replace
        from docuflow.models import LineItem
        from tests.test_rules import GOOD_INVOICE, build_context

        context = build_context(master, {**GOOD_INVOICE, **fields}, make_run=make_run)
        return replace(context, line_items=tuple(
            LineItem(description=row[0], amount=row[1]) for row in rows
        ))

    def test_the_totals_rule_names_the_field_and_offers_the_value(self, master, make_run):
        from docuflow.rules.financial import totals_consistent

        # The total reads 200,900 where the subtotal, the tax rate and the line
        # items all say 200,090 - a transposition of the last two digits.
        finding = totals_consistent(self._context(
            master,
            make_run,
            {"total_amount": 200_900.00},
            rows=(("Laptops", 187_000.00),),
        ))

        assert finding.passed is False
        assert finding.fixes, "a failing check with a derivable answer must carry it"
        fix = finding.fixes[0]
        assert fix.field == "total_amount"
        assert fix.suggested == 200_090.00
        assert fix.confidence >= 0.85
        assert "transposed" in finding.message

    def test_the_commercial_allowance_is_not_lent_to_a_documents_own_arithmetic(
        self, master, make_run
    ):
        """Found by the test above, which expected a failure and got a pass.

        The 2% tolerance exists so an invoice may differ from the purchase order
        it quotes - freight, part shipments, a price that moved. Applied to
        "subtotal + tax = total" it allowed 4,001 baht of drift on a 200,090-baht
        invoice, so a total misread as 200,900 was reported as adding up and
        posted. Two different questions, two different numbers.
        """
        from docuflow.rules.financial import po_amount_within_tolerance, totals_consistent

        drifted = self._context(master, make_run, {"total_amount": 200_900.00})
        assert totals_consistent(drifted).passed is False, "0.4% is not rounding"

        # The same 0.4% against the purchase order is ordinary commerce.
        over = self._context(master, make_run, {"subtotal": 187_700.00})
        assert po_amount_within_tolerance(over).passed is True

    def test_a_broken_document_it_cannot_diagnose_still_reports_the_failure(
        self, master, make_run
    ):
        """No suggestion is a fine outcome. A missing failure is not."""
        from docuflow.rules.financial import totals_consistent

        finding = totals_consistent(self._context(
            master, make_run, {"subtotal": 1.0, "tax_amount": 2.0, "total_amount": 999.0,
                               "tax_rate": 50.0},
        ))
        assert finding.passed is False
        assert finding.fixes == ()

    def test_the_currency_rule_takes_its_answer_from_the_purchase_order(
        self, master, make_run
    ):
        from docuflow.rules.financial import po_currency_matches

        finding = po_currency_matches(self._context(master, make_run, {"currency": "EUR"}))
        assert finding.passed is False
        assert finding.fixes[0].field == "currency"
        assert finding.fixes[0].suggested == "THB"
        assert finding.fixes[0].source == "master_data"

    def test_a_changed_bank_account_is_suggested_only_as_something_to_confirm(
        self, master, make_run
    ):
        """A changed account is the signature of payment-redirection fraud.

        Offering the registered one back at high confidence would train a
        reviewer to click through exactly the check that exists to stop it.
        """
        from docuflow.rules.financial import bank_account_matches_vendor

        finding = bank_account_matches_vendor(
            self._context(master, make_run, {"bank_account": "9999999999"})
        )
        assert finding.passed is False
        assert finding.fixes[0].confidence < 0.5
        assert "confirm the change" in finding.fixes[0].reason


class TestGatheringSuggestions:
    def test_one_suggestion_per_field_and_the_strongest_wins(self):
        """Two rules failing on the same misread number is one problem."""
        from docuflow.models import SuggestedFix

        gathered = best_of([
            SuggestedFix(field="total_amount", suggested=1.0, confidence=0.5),
            SuggestedFix(field="total_amount", suggested=2.0, confidence=0.9),
            SuggestedFix(field="currency", suggested="THB", confidence=0.7),
        ])
        assert [fix.field for fix in gathered] == ["total_amount", "currency"]
        assert gathered[0].suggested == 2.0

    def test_a_run_gathers_them_across_every_failing_check(self, make_run):
        from docuflow.models import RuleFinding, SuggestedFix

        run = make_run({"total_amount": 1.0}).evolve(findings=(
            RuleFinding(rule_id="a", passed=False, severity="blocking", message="",
                        fixes=(SuggestedFix(field="total_amount", suggested=72_760.0,
                                            confidence=0.92),)),
            RuleFinding(rule_id="b", passed=False, severity="warning", message="",
                        fixes=(SuggestedFix(field="total_amount", suggested=72_760.0,
                                            confidence=0.7),)),
            RuleFinding(rule_id="c", passed=True, severity="info", message="",
                        fixes=(SuggestedFix(field="currency", suggested="USD", confidence=1.0),)),
        ))

        fixes = run.suggested_fixes
        assert len(fixes) == 1, "the same field, proposed twice, is one suggestion"
        assert fixes[0].confidence == 0.92
        assert all(fix.field != "currency" for fix in fixes), "a passing check proposes nothing"
