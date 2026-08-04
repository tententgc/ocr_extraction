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
