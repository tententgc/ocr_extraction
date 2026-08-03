"""Per-document-type field schemas."""

from docuflow.schemas.registry import (
    DOCUMENT_SCHEMAS,
    FieldSpec,
    DocumentSchema,
    get_schema,
    schema_names,
)

__all__ = [
    "DOCUMENT_SCHEMAS",
    "DocumentSchema",
    "FieldSpec",
    "get_schema",
    "schema_names",
]
