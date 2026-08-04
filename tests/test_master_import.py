"""Loading master data from CSV.

Master data is what every rule checks documents against, so a row that loads
wrong mis-judges every document that mentions it, for as long as it sits there.
These tests are mostly about refusals: what the importer must not accept, and
what it must say when it refuses.
"""

from __future__ import annotations

import pytest

from docuflow.masterdata.importer import (
    MasterImportError,
    import_csv,
    read_rows,
)
from docuflow.masterdata.repository import MasterDataRepository

VENDORS = (
    "legal_name,tax_id,bank_account,status,currency\n"
    '"Prosoft CRM Co., Ltd.",0105539000123,1234567890,active,THB\n'
    "Andaman Marine Supply,0107544000108,9876543210,on_hold,THB\n"
)


@pytest.fixture
def master(settings) -> MasterDataRepository:
    """An empty register — these tests load their own, not the demo one."""
    return MasterDataRepository(settings.db_path)


class TestReadingTheFile:
    def test_headers_are_matched_however_the_exporter_named_them(self):
        rows = read_rows("Vendor Name,VAT Number,Account Number\nAcme Ltd,0105543000129,123\n")
        assert rows[0] == {"legal_name": "Acme Ltd", "tax_id": "0105543000129", "bank_account": "123"}

    def test_a_semicolon_separated_export_is_read(self):
        """Excel in a European locale writes semicolons, and users do not know that."""
        rows = read_rows("legal_name;tax_id\nAcme Ltd;0105543000129\n")
        assert rows[0]["legal_name"] == "Acme Ltd"

    def test_a_byte_order_mark_does_not_become_part_of_the_first_header(self):
        rows = read_rows("﻿legal_name,tax_id\nAcme Ltd,0105543000129\n")
        assert rows[0]["legal_name"] == "Acme Ltd"

    def test_an_empty_file_says_so(self):
        with pytest.raises(MasterImportError, match="empty"):
            read_rows("")

    def test_a_header_with_no_rows_says_so(self):
        with pytest.raises(MasterImportError, match="no rows"):
            read_rows("legal_name,tax_id\n")


class TestVendors:
    def test_a_clean_file_loads(self, master):
        report = import_csv(master, "vendors", VENDORS)
        assert report.created == 2
        assert report.errors == ()
        assert {v["legal_name"] for v in master.list_vendors()} == {
            "Prosoft CRM Co., Ltd.", "Andaman Marine Supply"
        }

    def test_vendor_ids_are_generated_when_the_export_has_none(self, master):
        import_csv(master, "vendors", VENDORS)
        assert all(v["vendor_id"].startswith("V-") for v in master.list_vendors())

    def test_re_importing_updates_rather_than_duplicates(self, master):
        """Tomorrow's export of the same file is a correction, not two suppliers."""
        import_csv(master, "vendors", VENDORS)
        changed = VENDORS.replace("1234567890", "5555555555")

        report = import_csv(master, "vendors", changed)

        assert report.updated == 2 and report.created == 0
        assert len(master.list_vendors()) == 2
        assert master.find_vendor("Prosoft CRM Co., Ltd.").vendor["bank_account"] == "5555555555"

    def test_a_row_with_no_name_is_skipped_with_its_row_number(self, master):
        report = import_csv(master, "vendors", "legal_name,tax_id\n,0105543000129\nAcme Ltd,0107544000108\n")
        assert report.created == 1
        assert report.errors[0].row == 2
        assert "legal_name" in report.errors[0].message

    def test_an_unknown_status_is_refused_rather_than_stored(self, master):
        """'inactive' is not a status this system has, and silently keeping it
        would let a blocked vendor look active to `vendor_active`."""
        report = import_csv(master, "vendors", "legal_name,status\nAcme Ltd,inactive\n")
        assert report.created == 0
        assert "must be one of" in report.errors[0].message

    def test_a_foreign_tax_number_is_kept_as_given(self, master):
        """Not every supplier has a Thai 13-digit id."""
        import_csv(master, "vendors", "legal_name,tax_id\nBluepeak GmbH,DE811907980\n")
        assert master.list_vendors()[0]["tax_id"] == "DE811907980"


class TestPurchaseOrders:
    @pytest.fixture(autouse=True)
    def _vendors(self, master):
        import_csv(master, "vendors", VENDORS)

    def test_a_po_is_linked_by_vendor_code(self, master):
        vendor_id = master.list_vendors()[0]["vendor_id"]
        report = import_csv(
            master, "purchase_orders",
            f"po_number,vendor_id,po_date,amount\nPO-1,{vendor_id},2026-07-01,187000.00\n",
        )
        assert report.created == 1
        assert master.get_purchase_order("PO-1")["amount"] == 187000.00

    def test_a_po_can_name_its_vendor_instead_of_coding_it(self, master):
        """A hand-made file has the supplier's name, not the ERP's code."""
        report = import_csv(
            master, "purchase_orders",
            "po_number,vendor_name,amount\nPO-2,Prosoft CRM Co. Ltd,50000\n",
        )
        assert report.created == 1
        assert master.get_purchase_order("PO-2")["vendor_id"]

    def test_a_po_for_an_unknown_vendor_is_refused(self, master):
        """A dangling reference would fail `po_exists` for a reason that has
        nothing to do with the document quoting it."""
        report = import_csv(master, "purchase_orders", "po_number,vendor_id,amount\nPO-3,V-9999,1\n")
        assert report.created == 0
        assert "not in the vendor master file" in report.errors[0].message

    def test_a_date_in_another_format_is_read(self, master):
        vendor_id = master.list_vendors()[0]["vendor_id"]
        import_csv(master, "purchase_orders",
                   f"po_number,vendor_id,po_date,amount\nPO-4,{vendor_id},31/07/2026,1\n")
        assert master.get_purchase_order("PO-4")["po_date"] == "2026-07-31"

    def test_a_date_that_is_not_a_date_is_refused(self, master):
        vendor_id = master.list_vendors()[0]["vendor_id"]
        report = import_csv(master, "purchase_orders",
                            f"po_number,vendor_id,po_date,amount\nPO-5,{vendor_id},soon,1\n")
        assert report.created == 0
        assert "not a date" in report.errors[0].message

    def test_an_amount_that_is_not_a_number_is_refused(self, master):
        vendor_id = master.list_vendors()[0]["vendor_id"]
        report = import_csv(master, "purchase_orders",
                            f"po_number,vendor_id,amount\nPO-6,{vendor_id},about ten thousand\n")
        assert report.created == 0
        assert "not a number" in report.errors[0].message

    def test_a_thousands_separator_is_not_a_reason_to_refuse(self, master):
        vendor_id = master.list_vendors()[0]["vendor_id"]
        import_csv(master, "purchase_orders",
                   f'po_number,vendor_id,amount\nPO-7,{vendor_id},"187,000.00"\n')
        assert master.get_purchase_order("PO-7")["amount"] == 187000.00


class TestContracts:
    def test_a_contract_loads_without_a_vendor(self, master):
        """The contract register often predates the vendor record."""
        report = import_csv(
            master, "contracts",
            "contract_number,party_a,party_b,effective_date,expiry_date,value\n"
            "CT-1,Us PCL,Them Ltd,2026-01-01,2026-12-31,368000\n",
        )
        assert report.created == 1
        assert master.get_contract("CT-1")["vendor_id"] is None

    def test_an_expiry_before_the_start_is_refused(self, master):
        """It would make `contract_expiring_soon` report a contract that already ended."""
        report = import_csv(
            master, "contracts",
            "contract_number,effective_date,expiry_date\nCT-2,2026-12-31,2026-01-01\n",
        )
        assert report.created == 0
        assert "before effective_date" in report.errors[0].message


class TestPartialFailure:
    def test_good_rows_load_even_when_others_do_not(self, master):
        """A hundred-row file with two bad rows imports ninety-eight."""
        report = import_csv(
            master, "vendors",
            "legal_name,status\nGood One,active\n,active\nGood Two,active\nBad Status,inactive\n",
        )
        assert report.created == 2
        assert [error.row for error in report.errors] == [3, 5]
        assert len(master.list_vendors()) == 2

    def test_the_summary_reads_like_a_sentence(self, master):
        report = import_csv(master, "vendors", "legal_name\nAcme\n,\n")
        assert report.summary == "vendors: 1 added, 1 row(s) skipped."


class TestEndpoints:
    def test_a_csv_uploads_through_the_api(self, client, settings):
        response = client.post(
            "/api/master/import?kind=vendors",
            files={"file": ("vendors.csv", VENDORS.encode(), "text/csv")},
        )
        assert response.status_code == 200
        assert response.json()["created"] == 2

    def test_an_unreadable_file_is_a_400_not_a_crash(self, client):
        response = client.post(
            "/api/master/import?kind=vendors", files={"file": ("empty.csv", b"", "text/csv")}
        )
        assert response.status_code == 400

    def test_an_unknown_kind_is_rejected_by_the_route(self, client):
        response = client.post(
            "/api/master/import?kind=widgets", files={"file": ("x.csv", VENDORS.encode(), "text/csv")}
        )
        assert response.status_code == 422

    def test_a_template_can_be_downloaded(self, client):
        body = client.get("/api/master/template?kind=purchase_orders").text
        assert body.startswith("po_number,")

    def test_the_master_file_can_be_emptied(self, client, settings, master):
        import_csv(master, "vendors", VENDORS)
        assert client.delete("/api/master").status_code == 200
        assert master.list_vendors() == []


class TestDemoSeeding:
    def test_the_demo_register_is_not_restored_over_imported_data(self, settings, master):
        """Restarting the server must not put five fictional vendors back."""
        from docuflow.bootstrap import bootstrap

        import_csv(master, "vendors", VENDORS)
        bootstrap(settings, with_samples=False)
        assert len(master.list_vendors()) == 2

    def test_it_is_not_restored_after_the_file_is_cleared(self, settings, master):
        from docuflow.bootstrap import bootstrap

        bootstrap(settings, with_samples=False)      # seeds, because it is empty
        assert master.list_vendors()
        master.clear()

        bootstrap(settings.model_copy(update={"seed_demo_master_data": False}), with_samples=False)
        assert master.list_vendors() == []
