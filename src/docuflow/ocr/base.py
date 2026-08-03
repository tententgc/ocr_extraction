"""OCR provider interface.

Every provider takes rendered page images and returns Markdown-ish text. The
pipeline never imports a concrete provider - it asks the factory, so swapping
local for cloud is a config change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PageImage:
    """One rendered page ready for recognition."""

    page_number: int
    path: Path
    width: int
    height: int
    # Text already embedded in the source PDF for this page, when it has a text
    # layer. Used as a cross-check against what the model reads.
    embedded_text: str = ""


class OcrProvider(ABC):
    """Recognises text from page images."""

    name: str = "base"

    @abstractmethod
    def recognize(self, page: PageImage) -> str:
        """Return the recognised text for a single page."""

    def available(self) -> tuple[bool, str]:
        """Whether this provider can run right now, plus a reason if not."""
        return True, ""

    def warm_up(self) -> None:
        """Optionally pre-load models so the first document is not slow."""


class OcrUnavailableError(RuntimeError):
    """Raised when a provider cannot run and no fallback is possible."""
