"""First-run setup: create the database, seed master data, render samples."""

from __future__ import annotations

import logging
from typing import Any

from docuflow.config import Settings
from docuflow.db import init_db
from docuflow.masterdata.repository import MasterDataRepository
from docuflow.samples.catalog import master_data_rows
from docuflow.samples.generator import generate_corpus

logger = logging.getLogger(__name__)


def bootstrap(settings: Settings, with_samples: bool = True) -> dict[str, Any]:
    """Prepare everything the demo needs. Safe to run repeatedly."""
    settings.ensure_directories()
    init_db(settings.db_path)

    master = MasterDataRepository(settings.db_path)
    # Only into an empty register. Once real master data is loaded, restarting
    # the server must not quietly put five fictional vendors back beside it -
    # and clearing the file must not undo itself on the next start.
    seeded: dict[str, int] = {}
    if settings.seed_demo_master_data and master.is_empty():
        seeded = master.seed(master_data_rows())

    result: dict[str, Any] = {
        "database": str(settings.db_path),
        "master_data": seeded,
        "vendors": len(master.list_vendors()),
        "purchase_orders": len(master.list_purchase_orders()),
        "contracts": len(master.list_contracts()),
    }

    if with_samples:
        manifest = generate_corpus(settings.sample_dir)
        result["samples"] = len(manifest)
        result["sample_dir"] = str(settings.sample_dir)

    return result
