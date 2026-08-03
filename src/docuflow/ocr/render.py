"""Turn any incoming file into page images the OCR models can read.

Supports PDFs (via pypdfium2) and plain images. PDFs that already carry a text
layer also hand back that text, which later becomes a free accuracy cross-check
against what the vision model reads.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from PIL import Image

from docuflow.ocr.base import PageImage

logger = logging.getLogger(__name__)

# PDFium is not thread-safe, and pypdfium2 does not serialise for us: two
# threads opening documents at once segfaults the interpreter, taking the whole
# server with it. Found by running a batch through the queue with two workers.
#
# Serialising here rather than dropping to one worker, because rendering is
# ~0.3s of a document that spends tens of seconds in OCR and the model calls.
# The concurrency worth having is on those, and they are unaffected.
_PDFIUM_LOCK = threading.Lock()

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"})
PDF_SUFFIXES = frozenset({".pdf"})
# Typhoon OCR 1.5 is trained on pages rendered at roughly this longest edge.
DEFAULT_TARGET_DIM = 1800
PDF_RENDER_DPI_BASE = 72.0


class UnsupportedDocumentError(ValueError):
    """The file is not a document we know how to render."""


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES | PDF_SUFFIXES


def _save_page(image: Image.Image, out_path: Path, target_dim: int) -> tuple[Path, int, int]:
    """Downscale to the target longest edge and write a JPEG."""
    width, height = image.size
    if max(width, height) > target_dim:
        scale = target_dim / float(max(width, height))
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
    image = image.convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="JPEG", quality=92)
    return out_path, image.size[0], image.size[1]


def _render_pdf(path: Path, out_dir: Path, max_pages: int, target_dim: int) -> list[PageImage]:
    with _PDFIUM_LOCK:
        return _render_pdf_unlocked(path, out_dir, max_pages, target_dim)


def _render_pdf_unlocked(path: Path, out_dir: Path, max_pages: int, target_dim: int) -> list[PageImage]:
    import pypdfium2 as pdfium

    pages: list[PageImage] = []
    document = pdfium.PdfDocument(str(path))
    try:
        count = min(len(document), max_pages)
        for index in range(count):
            page = document[index]
            # Render at a scale that lands near the target dimension directly,
            # which is sharper than rendering small and upscaling.
            page_width, page_height = page.get_size()
            scale = max(1.0, target_dim / max(page_width, page_height))
            bitmap = page.render(scale=min(scale, 4.0))
            image = bitmap.to_pil()

            embedded = ""
            try:
                text_page = page.get_textpage()
                embedded = text_page.get_text_range() or ""
                text_page.close()
            except Exception as exc:  # pragma: no cover - malformed PDFs
                logger.debug("no text layer on page %s: %s", index + 1, exc)

            out_path, width, height = _save_page(image, out_dir / f"page-{index + 1:03d}.jpg", target_dim)
            pages.append(
                PageImage(
                    page_number=index + 1,
                    path=out_path,
                    width=width,
                    height=height,
                    embedded_text=embedded.strip(),
                )
            )
            page.close()
    finally:
        document.close()
    return pages


def _render_image(path: Path, out_dir: Path, target_dim: int) -> list[PageImage]:
    with Image.open(path) as image:
        image.load()
        out_path, width, height = _save_page(image, out_dir / "page-001.jpg", target_dim)
    return [PageImage(page_number=1, path=out_path, width=width, height=height)]


def render_document(
    path: Path,
    out_dir: Path,
    max_pages: int = 8,
    target_dim: int = DEFAULT_TARGET_DIM,
) -> list[PageImage]:
    """Render `path` into page images under `out_dir`.

    Raises `UnsupportedDocumentError` for file types we cannot handle, so the
    caller can report a clear message instead of failing deep in a model call.
    """
    suffix = path.suffix.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    if suffix in PDF_SUFFIXES:
        return _render_pdf(path, out_dir, max_pages, target_dim)
    if suffix in IMAGE_SUFFIXES:
        return _render_image(path, out_dir, target_dim)
    raise UnsupportedDocumentError(
        f"Cannot render '{suffix or path.name}'. Supported: PDF, "
        f"{', '.join(sorted(s.lstrip('.') for s in IMAGE_SUFFIXES))}."
    )


def embedded_text_ratio(pages: list[PageImage]) -> float:
    """Share of pages that already carry a usable text layer."""
    if not pages:
        return 0.0
    with_text = sum(1 for page in pages if len(page.embedded_text) > 80)
    return with_text / len(pages)
