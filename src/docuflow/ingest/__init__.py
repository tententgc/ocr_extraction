"""Ingest: accept a file or a URL and put the bytes somewhere safe."""

from docuflow.ingest.fetch import FetchError, FetchedDocument, fetch_url
from docuflow.ingest.store import StoredBlob, store_bytes

__all__ = ["FetchError", "FetchedDocument", "StoredBlob", "fetch_url", "store_bytes"]
