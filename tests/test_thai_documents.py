"""Thai documents, at the points where Thai is not just English with new glyphs.

Thai was in the corpus from the start, but only in one shape: invoices printed
with Thai numerals. That shape tests *recognition* - either ๙ comes back or it
does not - and it leaves the harder question untouched. A Thai business invoice
is written in Thai and totalled in 135,000.00, and there the digits are never in
doubt; the only question is whether anything understood that
`จำนวนเงินทั้งสิ้น` is the total. That failure is silent.

Three places Thai behaves differently from English, each with a test here:

* **Terms nest.** Thai builds vocabulary by concatenation, so one document type's
  keyword can sit inside another's. English never does this.
* **Names carry their legal form inline.** `บริษัท ... จำกัด` is the Thai
  "Co., Ltd." and it has to be stripped in the middle of a name, not off the end.
* **Years are Buddhist.** 2569 is 2026, and a page can print that in either
  numeral system.
"""

from __future__ import annotations

import pytest

from docuflow.agents.classifier import classify_by_keywords
from docuflow.normalize.amounts import parse_money
from docuflow.normalize.dates import parse_date
from docuflow.samples.real_world import THAI_DOCUMENTS


class TestThaiTermsNest:
    """`ใบกำกับภาษี` is a substring of the standard Thai receipt heading."""

    RECEIPT = (
        "บริษัท ปตท. น้ำมันและการค้าปลีก จำกัด\n"
        "เลขประจำตัวผู้เสียภาษี 0105554000770\n"
        "ใบเสร็จรับเงิน / ใบกำกับภาษีอย่างย่อ\n"
        "เลขที่ PTT-880451\nวันที่ 28 กรกฎาคม 2569\n"
        "น้ำมันดีเซล B7\n50 x 40.00 2,000.00\n"
        "มูลค่าสินค้า 2,000.00\nภาษีมูลค่าเพิ่ม 7% 140.00\nรวมทั้งสิ้น 2,140.00\n"
    )

    def test_a_thai_receipt_is_a_receipt(self):
        """The heading a Thai till prints is `ใบกำกับภาษีอย่างย่อ` - an
        *abbreviated* tax invoice, which is the Revenue Department's name for a
        receipt. Matched as a plain substring it also satisfies the invoice
        keyword `ใบกำกับภาษี`, and every Thai receipt in the country scores as
        an invoice on the strength of its own receipt heading."""
        doc_type, _, scores = classify_by_keywords(self.RECEIPT)

        assert doc_type == "receipt", scores
        assert scores["receipt"] > scores["invoice"]

    def test_the_nested_invoice_keyword_casts_no_vote(self):
        """Only the more specific of two nested matches counts. This is the
        general rule and it needs no Thai in it: a keyword wholly contained in a
        longer keyword that also matched is not independent evidence."""
        from docuflow.agents.classifier import _shadowed

        matched = frozenset({"ใบกำกับภาษี", "ใบกำกับภาษีอย่างย่อ"})
        assert _shadowed("ใบกำกับภาษี", matched)
        assert not _shadowed("ใบกำกับภาษีอย่างย่อ", matched)

    def test_a_thai_invoice_is_still_an_invoice(self):
        """The fix must not take the keyword away from documents entitled to
        it. A full tax invoice says `ใบกำกับภาษี` and nothing longer."""
        invoice = (
            "บริษัท สยามเทค ซัพพลาย จำกัด\nใบกำกับภาษี / ใบแจ้งหนี้\n"
            "เลขที่เอกสาร STS-2026-0318\nวันที่ออก 24 กรกฎาคม 2569\n"
            "มูลค่าสินค้า/บริการ 135,000.00\nภาษีมูลค่าเพิ่ม 7% 9,450.00\n"
            "จำนวนเงินทั้งสิ้น (บาท) 144,450.00\n"
        )
        doc_type, _, scores = classify_by_keywords(invoice)

        assert doc_type == "invoice", scores


class TestThaiVendorNames:
    """The name on the page is Thai; the master file may hold either form."""

    @pytest.mark.parametrize("printed,expected", [
        ("บริษัท สยามเทค ซัพพลาย จำกัด", "V-1001"),
        ("บริษัท ซันไรส์ อิเล็คทริคอล จำกัด", "V-1011"),
        # Thai is written without spaces between words as often as with them,
        # and OCR reproduces whichever it saw.
        ("บริษัทซันไรส์อิเล็คทริคอลจำกัด", "V-1011"),
        # The legal form dropped, which a letterhead often does.
        ("ซันไรส์ อิเล็คทริคอล", "V-1011"),
        # ซ misread as ช - one stroke apart, and the most common Thai OCR slip.
        ("บริษัท เจ้าพระยา แฟชิลิตี้ จำกัด", "V-1004"),
    ])
    def test_a_thai_name_resolves_without_a_tax_id(self, thai_master, printed, expected):
        """No tax ID, so there is no exact identifier to fall back on and the
        Thai name has to carry the match on its own. This is where a Thai
        vendor is most likely to be lost."""
        match = thai_master.find_vendor(printed, None)

        assert match.vendor is not None, f"{printed} matched nothing"
        assert match.vendor["vendor_id"] == expected

    @pytest.mark.parametrize("printed", [
        "หจก. สมชายการช่าง",
        "บริษัท สมชายการช่าง จำกัด",
    ])
    def test_a_supplier_we_do_not_know_stays_unknown(self, thai_master, printed):
        """The opposite failure, and the expensive one: fuzzy matching a Thai
        name onto a registered vendor it merely reads alike would pay the wrong
        company. `บริษัท` and `จำกัด` are stripped as legal-form noise, so what
        is left to compare is short and the risk is real."""
        assert thai_master.find_vendor(printed, None).vendor is None


class TestContractPartiesAreChecked:
    """The register knows who signed it, and nothing was comparing.

    Found on a Thai contract that auto-posted at 92% with `party_a` and
    `party_b` both misread - `Mirtphol` as `Mitrphol`, and `แฟซิลิตี้` as
    `แฟชิลิตี้`. Both are critical fields for a contract. Both were grounded,
    because the misread text is genuinely in the OCR output and grounding can
    only prove a value came from the page, not that the page was read right.

    An invoice naming the wrong company is caught downstream by `vendor_known`
    resolving the name against the master file. A contract had no equivalent:
    the contract number is an exact key into a register that stores both
    parties, and the comparison was simply never made.
    """

    CONTRACT = {
        "contract_number": "CT-2025-018",
        "contract_title": "Facility Services Agreement",
        "party_a": "Mirtphol Industries PCL",
        "party_b": "Chao Phraya Facility Services Co., Ltd.",
        "effective_date": "2025-08-01",
        "expiry_date": "2026-07-31",
        "contract_value": 368_000.00,
        "currency": "THB",
    }

    def _finding(self, master, make_run, **overrides):
        from tests.test_rules import build_context, result_for
        from docuflow.rules import run_rules

        context = build_context(master, {**self.CONTRACT, **overrides},
                                doc_type="contract", make_run=make_run)
        return result_for(run_rules(context), "contract_matches_master")

    def test_a_contract_naming_a_different_company_is_stopped(self, master, make_run):
        """The failure the check exists for: the page says one counterparty and
        the register says another. Nothing else on the document contradicts it -
        the number, the dates and the value all agree."""
        finding = self._finding(master, make_run, party_b="Northwind Logistics Ltd.")

        assert finding.passed is False
        assert finding.severity == "blocking"
        assert "party_b" in finding.message or "Party B" in finding.message

    @pytest.mark.parametrize("printed", [
        "Chao Phraya Facility Services Co., Ltd.",
        "Chao Phraya Facility Services",              # legal form dropped
        "บริษัท เจ้าพระยา แฟซิลิตี้ จำกัด",             # the Thai legal name
        "บริษัท เจ้าพระยา แฟชิลิตี้ จำกัด",             # ซ misread as ช
    ])
    def test_the_same_company_written_differently_still_passes(self, master, make_run, printed):
        """The check must not fire on spelling. A contract read perfectly except
        for one Thai character is the same agreement with the same company, and
        blocking it turns a check meant for a substituted counterparty into a
        machine for generating false alarms."""
        assert self._finding(master, make_run, party_b=printed).passed is True

    def test_a_party_the_document_never_stated_is_not_a_difference(self, master, make_run):
        """Absent is not wrong. A contract that did not print its counterparty
        has a different problem, and a required-field check is where it belongs."""
        assert self._finding(master, make_run, party_b=None).passed is True


class TestTransientOcrFailures:
    """A timeout is not a verdict on the page.

    Both Thai-numeral documents in the corpus provoked a 408 from Typhoon Cloud,
    and the engine treated it the way it treats bad text: escalate to the next
    provider. The next provider was a local 3B model, and each document took ten
    minutes instead of thirty seconds - with the trace reporting `ok`.
    """

    def _page(self, tmp_path):
        from docuflow.ocr.base import PageImage

        path = tmp_path / "page.png"
        path.write_bytes(b"not really a png")
        return PageImage(page_number=1, path=path, width=100, height=100)

    def _provider(self, name, behaviour):
        from docuflow.ocr.base import OcrProvider

        class Fake(OcrProvider):
            purpose_built = True

            def __init__(self):
                self.name = name
                self.calls = 0

            def available(self):
                return True, ""

            def recognize(self, page):
                self.calls += 1
                return behaviour(self.calls)

        return Fake()

    # Long enough to clear the engine's own quality gate, which is a separate
    # escalation and would otherwise send the page down the chain anyway.
    GOOD_PAGE = (
        "TAX INVOICE\nRungruang Steel Works Co., Ltd.\nTax ID: 0105552000110\n"
        "Invoice No.: RSW-2026-1188\nInvoice Date: 14 July 2026\n"
        "Bill To\nMirtphol Industries PCL\n88/9 Rama IV Road, Bangkok 10110\n"
        "Description Qty Unit Price Amount\nHot-rolled steel plate 20 18,500.00 370,000.00\n"
        "Subtotal 415,000.00 THB\nVAT 7% 29,050.00 THB\nAmount Due 444,050.00 THB\n"
        "Payment Terms: Net 60\nRemit to Account No: 3120045678\n"
    )

    def test_a_timeout_asks_the_same_engine_again(self, tmp_path):
        from docuflow.ocr.base import OcrTransientError
        from docuflow.ocr.engine import _recognize_page

        def flaky(call):
            if call == 1:
                raise OcrTransientError("Error code: 408 - Request timed out")
            return self.GOOD_PAGE

        cloud = self._provider("cloud", flaky)
        local = self._provider("local", lambda _: "should never be reached")

        text, provider, warnings = _recognize_page(self._page(tmp_path), [cloud, local])

        assert provider is cloud, "a timeout sent the page down the chain"
        assert cloud.calls == 2 and local.calls == 0
        assert "444,050.00" in text
        assert any("did not answer" in note for note in warnings)

    def test_an_engine_that_never_answers_still_gives_way(self, tmp_path):
        """Retrying rides out a timeout; it must not sit on an outage."""
        from docuflow.ocr.base import OcrTransientError
        from docuflow.ocr.engine import TRANSIENT_ATTEMPTS, _recognize_page

        def always_times_out(_call):
            raise OcrTransientError("Error code: 503 - upstream unavailable")

        cloud = self._provider("cloud", always_times_out)
        local = self._provider("local", lambda _: self.GOOD_PAGE)

        _text, provider, _warnings = _recognize_page(self._page(tmp_path), [cloud, local])

        assert cloud.calls == TRANSIENT_ATTEMPTS
        assert provider is local

    def test_a_bad_key_is_not_retried(self, tmp_path):
        """Only failures that say nothing about the page are worth repeating.
        An authentication error will be an authentication error next time."""
        from docuflow.ocr.engine import _recognize_page

        def refused(_call):
            raise RuntimeError("Error code: 401 - invalid api key")

        cloud = self._provider("cloud", refused)
        local = self._provider("local", lambda _: self.GOOD_PAGE)

        _text, provider, _warnings = _recognize_page(self._page(tmp_path), [cloud, local])

        assert cloud.calls == 1
        assert provider is local

    @pytest.mark.parametrize("message,transient", [
        ("Error code: 408 - {'detail': 'Request timed out'}", True),
        ("Error code: 429 - rate limit exceeded", True),
        ("Error code: 503 - service unavailable", True),
        ("Error code: 401 - invalid api key", False),
        ("Error code: 400 - malformed request", False),
        ("connection reset by peer", False),
    ])
    def test_which_failures_count_as_transient(self, message, transient):
        from docuflow.ocr.typhoon_cloud import _is_transient

        assert _is_transient(RuntimeError(message)) is transient


class TestBuddhistEraDates:
    @pytest.mark.parametrize("printed,expected", [
        ("24 กรกฎาคม 2569", "2026-07-24"),      # Arabic digits - the common case
        ("๒๔ กรกฎาคม ๒๕๖๙", "2026-07-24"),      # Thai digits
        ("1 มิถุนายน 2569", "2026-06-01"),
        ("31 พฤษภาคม 2570", "2027-05-31"),
        ("24/07/2569", "2026-07-24"),
    ])
    def test_a_buddhist_year_is_converted(self, printed, expected):
        assert str(parse_date(printed)) == expected

    @pytest.mark.parametrize("printed,expected", [
        ("135,000.00", 135_000.00),
        ("๑๓๕,๐๐๐.๐๐", 135_000.00),
        ("44,940.00 บาท", 44_940.00),
    ])
    def test_an_amount_parses_in_either_numeral_system(self, printed, expected):
        assert parse_money(printed) == expected


class TestTheThaiCorpus:
    """What the six Thai documents claim about themselves has to be true."""

    def test_every_thai_invoice_adds_up_the_way_it_says_it_does(self):
        """A document planted with one defect must carry exactly that defect.
        A second, accidental one makes the run untraceable: the rule that fired
        may not be the rule under test."""
        planted = {"invoice_thai_vat_miscalculated": "tax_rate_consistent"}

        for slug, _arrival, _posts, doc in THAI_DOCUMENTS:
            gt = doc.ground_truth
            subtotal, tax, total = gt.get("subtotal"), gt.get("tax_amount"), gt.get("total_amount")
            if subtotal is None or tax is None or total is None:
                continue

            assert abs(subtotal + tax - total) < 0.01, f"{slug}: subtotal + tax != total"
            if doc.line_items:
                lines = sum(item.get("amount") or 0 for item in doc.line_items)
                assert abs(lines - subtotal) < 0.01, f"{slug}: lines do not sum to the subtotal"

            rate = gt.get("tax_rate")
            if rate:
                consistent = abs(round(subtotal * rate / 100, 2) - tax) < 0.01
                if planted.get(doc.doc_id) == "tax_rate_consistent":
                    assert not consistent, f"{slug}: planted VAT defect is not actually wrong"
                else:
                    assert consistent, f"{slug}: unplanted VAT inconsistency"

    def test_the_expected_outcome_matches_what_is_planted(self):
        """`_True` in the filename means nothing is wrong with the document.
        A document with `expected_findings` has something wrong with it by
        definition, so the two cannot both be true."""
        for slug, _arrival, posts, doc in THAI_DOCUMENTS:
            if doc.expected_findings:
                assert not posts, f"{slug} plants {doc.expected_findings} yet is labelled _True"

    def test_the_corpus_covers_arabic_digits_and_more_than_invoices(self):
        """The gap these documents were written to close. Guarding it here
        because it is the kind of coverage that quietly disappears when someone
        prunes the corpus."""
        layouts = {doc.layout for _slug, _arrival, _posts, doc in THAI_DOCUMENTS}
        doc_types = {doc.doc_type for _slug, _arrival, _posts, doc in THAI_DOCUMENTS}

        assert "thai_modern" in layouts, "no Thai document printed with Arabic digits"
        assert {"invoice", "receipt", "contract"} <= doc_types
