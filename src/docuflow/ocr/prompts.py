"""Typhoon OCR 1.5 prompt.

Sourced from the official `typhoon-ocr` package so the locally-hosted model
receives byte-identical instructions to the hosted API. If the package changes
its prompt, we pick that up automatically and only fall back to the pinned copy
when the import is unavailable.
"""

from __future__ import annotations

# Pinned copy of typhoon_ocr.PROMPTS_SYS["v1.5"] (package 0.4.1).
_FALLBACK_V15 = """Extract all text from the image.


Instructions:
- Only return the clean Markdown.
- Do not include any explanation or extra text.
- You must include all information on the page.


Formatting Rules:
- Tables: Render tables using <table>...</table> in clean HTML format.
- Equations: Render equations using LaTeX syntax with inline ($...$) and block ($$...$$).
- Images/Charts/Diagrams: Wrap any clearly defined visual areas (e.g. charts, diagrams, pictures) in:


<figure>
Describe the image's main elements (people, objects, text), note any contextual clues (place, event, culture), mention visible text and its meaning, provide deeper analysis when relevant (especially for financial charts, graphs, or documents), comment on style or architecture if relevant, then give a concise overall summary. Describe in {figure_language}.
</figure>


- Page Numbers: Wrap page numbers in <page_number>...</page_number> (e.g., <page_number>14</page_number>).
- Checkboxes: Use ☐ for unchecked and ☑ for checked boxes.
    """


def v15_prompt(figure_language: str = "English") -> str:
    """The official Typhoon OCR 1.5 extraction prompt."""
    try:
        from typhoon_ocr.ocr_utils import PROMPTS_SYS

        return PROMPTS_SYS["v1.5"](figure_language=figure_language)
    except Exception:  # pragma: no cover - only when the package is absent
        return _FALLBACK_V15.replace("{figure_language}", figure_language)
