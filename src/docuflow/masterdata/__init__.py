"""Master data: the existing records new documents are checked against."""

from docuflow.masterdata.repository import (
    MasterDataRepository,
    VendorMatch,
)

__all__ = ["MasterDataRepository", "VendorMatch"]
