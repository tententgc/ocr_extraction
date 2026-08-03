"""OCR layer: rendering, providers and quality scoring."""

from docuflow.ocr.base import OcrProvider, OcrUnavailableError, PageImage
from docuflow.ocr.engine import run_ocr
from docuflow.ocr.factory import build_provider, describe_providers, provider_chain
from docuflow.ocr.quality import score_document_text, score_page_text
from docuflow.ocr.render import UnsupportedDocumentError, is_supported, render_document

__all__ = [
    "OcrProvider",
    "OcrUnavailableError",
    "PageImage",
    "UnsupportedDocumentError",
    "build_provider",
    "describe_providers",
    "is_supported",
    "provider_chain",
    "render_document",
    "run_ocr",
    "score_document_text",
    "score_page_text",
]
