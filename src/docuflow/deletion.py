"""Remove a document and everything it left behind.

A document that turns out to be a duplicate scan, a test upload or the wrong
file entirely should be removable, and removing it should not leave three
copies of it on disk. One processed document occupies four places: the stored
run, the content-addressed blob, the hard link on its type's archive shelf, and
the rendered page images.

Two things are deliberately *not* removed:

* **A posted document cannot be deleted.** The ledger row points at it, and a
  record whose source cannot be produced is worse than an untidy database. The
  database enforces this too, through the records foreign key; the check here
  exists to say why rather than to raise a constraint error.
* **The audit log survives.** It is the record that the document existed and
  that someone deleted it, which is exactly the question an audit asks. An
  audit trail that disappears with its subject is not an audit trail.

Reviewer corrections *are* removed with the document. A correction is only
meaningful next to the page it was made against; keeping ones nobody can verify
turns the accuracy signal into noise.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docuflow.archive import archive_dir
from docuflow.config import Settings
from docuflow.db import connect, transaction
from docuflow.store.documents import DocumentStore
from docuflow.store.records import RecordStore

logger = logging.getLogger(__name__)


class DeletionError(RuntimeError):
    """The document could not be deleted."""


@dataclass(frozen=True)
class Deleted:
    document_id: str
    filename: str
    doc_type: str
    corrections: int
    files: tuple[str, ...] = field(default_factory=tuple)
    kept: tuple[str, ...] = field(default_factory=tuple)

    def to_json(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "doc_type": self.doc_type,
            "corrections": self.corrections,
            "files": list(self.files),
            "kept": list(self.kept),
            "message": self.message,
        }

    @property
    def message(self) -> str:
        parts = [f"Deleted {self.filename}"]
        if self.corrections:
            parts.append(f"{self.corrections} correction(s)")
        if self.files:
            parts.append(f"{len(self.files)} file(s)")
        return ", ".join(parts) + "."


def _shared_blob(settings: Settings, sha256: str, document_id: str) -> bool:
    """Does another document hold the same bytes?

    Ingest storage is content-addressed, so two uploads of the same file share
    one blob. Deleting it because one of them is going would take the page out
    from under the other.
    """
    row = connect(settings.db_path).execute(
        "SELECT COUNT(*) AS others FROM documents WHERE sha256 = ? AND document_id != ?",
        (sha256, document_id),
    ).fetchone()
    return bool(row and row["others"])


def _remove(path: Path) -> bool:
    """Delete a file or directory. Never raises: the row is what matters."""
    try:
        if path.is_dir():
            shutil.rmtree(path)
            return True
        if path.exists():
            path.unlink()
            return True
    except OSError as exc:
        logger.warning("could not remove %s: %s", path, exc)
    return False


@dataclass(frozen=True)
class Wiped:
    """What a clear-out removed."""

    documents: int
    records: int
    corrections: int
    files: int
    failed: tuple[str, ...] = field(default_factory=tuple)

    def to_json(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "records": self.records,
            "corrections": self.corrections,
            "files": self.files,
            "failed": list(self.failed),
            "message": self.message,
        }

    @property
    def message(self) -> str:
        if not self.documents and not self.records:
            # The sweep still runs when the database is already empty, and it is
            # usually the run where it finds the most.
            return (
                f"No documents or ledger entries; cleared {self.files} leftover file(s)."
                if self.files else "There was nothing to remove."
            )
        parts = [f"Removed {self.documents} document(s)"]
        if self.records:
            parts.append(f"{self.records} ledger entr{'y' if self.records == 1 else 'ies'}")
        if self.files:
            parts.append(f"{self.files} file(s)")
        message = ", ".join(parts) + ". The audit log and your master data are untouched."
        if self.failed:
            message += f" {len(self.failed)} could not be removed."
        return message


def wipe_documents(settings: Settings, actor: str = "reviewer") -> Wiped:
    """Remove every processed document and every ledger entry.

    Clearing a test run one row at a time is not a workflow, and deleting a
    ledger entry leaves the document that produced it behind - correctly, since
    a page is the evidence for the entry, but with nothing left pointing at it.
    Doing both together is the only way to actually get back to empty.

    Master data is *not* touched. Vendors and purchase orders are a copy of what
    another system holds, not something this run produced; `master clear` exists
    separately for that. Nor is the audit log: it is the record that any of this
    happened.
    """
    documents = DocumentStore(settings.db_path)
    records = RecordStore(settings.db_path)

    posted = records.list(limit=10_000)
    for record in posted:
        records.delete(record["record_id"], actor=actor)

    removed: list[Deleted] = []
    failed: list[str] = []
    # Paged listing would shift under its own deletions, so take the ids first.
    for row in documents.list(limit=10_000):
        try:
            removed.append(delete_document(settings, row["document_id"], actor=actor))
        except DeletionError as exc:  # a document that reappeared, or a locked file
            logger.warning("could not delete %s: %s", row["document_id"], exc)
            failed.append(row["document_id"])

    wiped = Wiped(
        documents=len(removed),
        records=len(posted),
        corrections=sum(item.corrections for item in removed),
        files=sum(len(item.files) for item in removed) + _sweep_orphans(settings),
        failed=tuple(failed),
    )
    records.log("documents_wiped", actor=actor, **{
        k: v for k, v in wiped.to_json().items() if k != "message"
    })
    return wiped


def _files_in(directory: Path) -> list[Path]:
    return [path for path in directory.iterdir() if path.is_file()] if directory.exists() else []


def _sweep_orphans(settings: Settings) -> int:
    """Remove page images and archive shelves nothing refers to any more.

    Deleting a document takes its own files with it, so this only ever finds what
    an earlier interruption left behind - a run that failed after rendering, a
    row removed by hand. Rare, but "start over" that leaves 80K of pages for a
    document id you cannot look up is not starting over.
    """
    # The blob path is part of the stored run rather than a column of its own.
    rows = connect(settings.db_path).execute(
        "SELECT document_id, origin, json_extract(payload, '$.blob_path') AS blob FROM documents"
    ).fetchall()
    live = {row["document_id"] for row in rows}
    referenced = {row["origin"] for row in rows} | {row["blob"] for row in rows}

    removed = 0
    if settings.page_dir.exists():
        for pages in settings.page_dir.iterdir():
            if pages.is_dir() and pages.name not in live and _remove(pages):
                removed += 1
                logger.info("removed orphaned page images for %s", pages.name)

    # An upload is staged in the inbox before it is processed and nothing has
    # ever cleared it, so every file ever uploaded is still sitting there. The
    # staged copy is ours; the file the user chose is still wherever they got it.
    for staged in _files_in(settings.data_dir / "inbox"):
        if str(staged) not in referenced and _remove(staged):
            removed += 1

    # Content-addressed storage keeps one blob per distinct file, so a blob only
    # becomes garbage once every document holding those bytes has gone.
    for blob in _files_in(settings.blob_dir):
        if str(blob) not in referenced and _remove(blob):
            removed += 1

    # The archive shelves documents by type and month. Empty ones are just the
    # shape of a filing cabinet nobody has filed in yet. Deepest first, so a
    # month emptying leaves its type empty and both go.
    archive = archive_dir(settings)
    if archive.exists():
        for shelf in sorted(archive.rglob("*"), reverse=True):
            if shelf.is_dir() and not any(shelf.iterdir()):
                _remove(shelf)
    return removed


def delete_document(settings: Settings, document_id: str, actor: str = "reviewer") -> Deleted:
    """Delete one document, its files and its corrections."""
    documents = DocumentStore(settings.db_path)
    records = RecordStore(settings.db_path)

    run = documents.get(document_id)
    if run is None:
        raise DeletionError(f"Document {document_id} was not found.")
    if run.record_id:
        raise DeletionError(
            f"{document_id} is posted as {run.record_id} and cannot be deleted. The ledger "
            f"entry refers to it; reverse the record downstream if it should not stand."
        )

    removed: list[str] = []
    kept: list[str] = []

    # The rendered pages exist only to be shown next to the fields; nothing else
    # refers to them.
    if _remove(settings.page_dir / document_id):
        removed.append(str(settings.page_dir / document_id))

    # The archive link leads with the document id by design, so it can be found
    # without trusting that the type or period never changed.
    for link in archive_dir(settings).glob(f"*/*/{document_id}__*"):
        if _remove(link):
            removed.append(str(link))

    if run.blob_path:
        if _shared_blob(settings, run.source.sha256, document_id):
            kept.append(f"{run.blob_path} (another document has the same content)")
        elif _remove(Path(run.blob_path)):
            removed.append(run.blob_path)

    with transaction(settings.db_path) as connection:
        corrections = connection.execute(
            "DELETE FROM corrections WHERE document_id = ?", (document_id,)
        ).rowcount
        connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))

    records.log(
        "document_deleted",
        actor=actor,
        document_id=document_id,
        filename=run.source.filename,
        doc_type=run.doc_type,
        decision=run.decision,
        files=removed,
    )
    return Deleted(
        document_id=document_id,
        filename=run.source.filename,
        doc_type=run.doc_type,
        corrections=max(corrections, 0),
        files=tuple(removed),
        kept=tuple(kept),
    )
