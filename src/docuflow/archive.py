"""File a processed document under its type.

Incoming bytes land in content-addressed storage, because at that moment nobody
knows what the document is - the type comes out of classification, several
stages later. Content addressing is right for ingest and wrong for everything
afterwards: a directory of SHA-256 names answers no question a business asks.

So once a document has been classified it is also filed:

    data/archive/invoice/2026-08/DOC-4F2A9B1C3D__acme-inv-88214.pdf
    data/archive/contract/2026-08/DOC-1A2B3C4D5E__facility-agreement.pdf

which is what makes per-type retention, per-type export and "show me every
contract from July" ordinary operations rather than database queries. The blob
is hard-linked, not moved: ingest storage stays intact and de-duplicated, and
the archive costs no extra bytes on the same filesystem.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from docuflow.config import Settings
from docuflow.ingest.store import safe_filename

logger = logging.getLogger(__name__)

# Types that get their own shelf. Anything else files under "other" rather than
# creating a directory per typo or per new schema someone adds.
KNOWN_TYPES = ("invoice", "receipt", "purchase_order", "contract", "form")
OTHER = "other"


def shelf_for(doc_type: str) -> str:
    return doc_type if doc_type in KNOWN_TYPES else OTHER


@dataclass(frozen=True)
class ArchivedDocument:
    path: Path
    doc_type: str
    period: str

    def to_json(self) -> dict[str, Any]:
        return {"path": str(self.path), "doc_type": self.doc_type, "period": self.period}


def archive_dir(settings: Settings) -> Path:
    return settings.data_dir / "archive"


def archive_path(settings: Settings, doc_type: str, period: str, filename: str) -> Path:
    return archive_dir(settings) / shelf_for(doc_type) / period / filename


def _period(created_at: Any) -> str:
    """`YYYY-MM` shelf name, from a datetime or an ISO string."""
    if isinstance(created_at, datetime):
        return created_at.strftime("%Y-%m")
    text = str(created_at or "")
    return text[:7] if len(text) >= 7 else "unknown"


def file_document(
    settings: Settings,
    blob_path: Path,
    document_id: str,
    doc_type: str,
    filename: str,
    created_at: Any,
) -> ArchivedDocument | None:
    """Place a copy of the document on its type's shelf. Never raises.

    Filing is a convenience, not part of the record: if it fails, the document
    is still processed, still stored and still posted. It must not be able to
    fail a run.
    """
    try:
        source = Path(blob_path)
        if not source.exists():
            return None

        period = _period(created_at)
        # The document id leads so the name is unique and sorts by arrival, and
        # the original filename follows so a human can recognise it.
        name = f"{document_id}__{safe_filename(filename or source.name)}"
        if not Path(name).suffix:
            name = f"{name}{source.suffix}"

        target = archive_path(settings, doc_type, period, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return ArchivedDocument(target, shelf_for(doc_type), period)

        try:
            os.link(source, target)
        except OSError:
            # Different filesystem, or a filesystem without hard links.
            shutil.copy2(source, target)
        return ArchivedDocument(target, shelf_for(doc_type), period)
    except Exception as exc:  # noqa: BLE001 - filing must never fail a document
        logger.warning("could not file %s under %s: %s", document_id, doc_type, exc)
        return None


def describe_archive(settings: Settings) -> dict[str, Any]:
    """What is on each shelf, for the operator view."""
    root = archive_dir(settings)
    shelves: list[dict[str, Any]] = []
    if not root.exists():
        return {"root": str(root), "shelves": shelves, "total_documents": 0, "total_bytes": 0}

    total_files = total_bytes = 0
    for shelf in sorted(root.iterdir()):
        if not shelf.is_dir():
            continue
        periods: list[dict[str, Any]] = []
        shelf_files = shelf_bytes = 0
        for period in sorted(shelf.iterdir(), reverse=True):
            if not period.is_dir():
                continue
            files = [item for item in period.iterdir() if item.is_file()]
            size = sum(item.stat().st_size for item in files)
            periods.append({"period": period.name, "documents": len(files), "bytes": size})
            shelf_files += len(files)
            shelf_bytes += size
        shelves.append(
            {
                "doc_type": shelf.name,
                "documents": shelf_files,
                "bytes": shelf_bytes,
                "periods": periods,
            }
        )
        total_files += shelf_files
        total_bytes += shelf_bytes

    return {
        "root": str(root),
        "shelves": shelves,
        "total_documents": total_files,
        "total_bytes": total_bytes,
    }


def rebuild(settings: Settings) -> dict[str, int]:
    """File every document already processed. Safe to run repeatedly.

    For an existing installation, or after changing how shelves are named.
    """
    from docuflow.store.documents import DocumentStore

    documents = DocumentStore(settings.db_path)
    filed = skipped = 0
    for row in documents.list(limit=100_000):
        run = documents.get(row["document_id"])
        if run is None or not run.blob_path:
            skipped += 1
            continue
        result = file_document(
            settings,
            Path(run.blob_path),
            run.document_id,
            run.doc_type,
            run.source.filename,
            run.created_at,
        )
        if result is None:
            skipped += 1
        else:
            filed += 1
    return {"filed": filed, "skipped": skipped}
