"""Persistence for processed documents, downstream records and corrections."""

from docuflow.store.corrections import CorrectionStore
from docuflow.store.documents import DocumentStore
from docuflow.store.records import RecordStore, reference_field_for

__all__ = ["CorrectionStore", "DocumentStore", "RecordStore", "reference_field_for"]
