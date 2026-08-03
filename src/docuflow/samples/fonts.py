"""Font registration for sample rendering (Thai coverage required)."""

from __future__ import annotations

import logging
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logger = logging.getLogger(__name__)

# Thai-capable TrueType fonts, in preference order. macOS ships all of these.
THAI_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Ayuthaya.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/usr/share/fonts/truetype/tlwg/Garuda.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf"),
)

THAI_FONT_NAME = "DocuFlowThai"
_registered: bool = False


def register_thai_font() -> str | None:
    """Register a Thai-capable font and return its name, or None if unavailable."""
    global _registered
    if _registered:
        return THAI_FONT_NAME

    for candidate in THAI_FONT_CANDIDATES:
        if not candidate.exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont(THAI_FONT_NAME, str(candidate)))
            _registered = True
            logger.debug("registered Thai font from %s", candidate)
            return THAI_FONT_NAME
        except Exception as exc:  # noqa: BLE001 - font files vary by platform
            logger.debug("could not register %s: %s", candidate, exc)

    logger.warning("no Thai-capable font found; Thai samples will render with missing glyphs")
    return None
