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

    @property
    def purpose_built(self) -> bool:
        """Whether this engine was built to transcribe documents.

        A purpose-built engine - Typhoon OCR, or a PDF's own text layer - is
        trusted at full confidence. A general vision model or a classical OCR
        binary is doing its best on a task it was not trained for, so the run is
        marked degraded and the values are worth a second look.

        Asked of the provider rather than matched against a list of names, so
        the same model reached through a different host is judged the same way.
        """
        return False

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


class OcrTransientError(RuntimeError):
    """The engine did not answer, and the same engine is worth asking again.

    Separate from a provider returning poor text, because the two want opposite
    responses. Bad text from a deterministic model will be bad again, so the
    engine escalates to the next provider. A timeout or a rate limit says
    nothing about the model at all, and escalating on one steps off a cloud API
    onto whatever is next in the chain - which on this project meant a local 3B
    model and ten minutes per page instead of three seconds.
    """
