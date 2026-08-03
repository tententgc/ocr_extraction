"""OCR execution: pick a provider, recognise every page, escalate on failure.

The old pipeline retried the *same* provider when quality was poor. Retrying a
deterministic model rarely helps, so this engine escalates to the next provider
in the chain instead and reports which one actually produced the text.
"""

from __future__ import annotations

import logging
import time

from docuflow.config import Settings
from docuflow.models import OcrPage, OcrResult
from docuflow.ocr.base import OcrProvider, OcrUnavailableError, PageImage
from docuflow.ocr.factory import provider_chain
from docuflow.ocr.quality import score_document_text, score_page_text

logger = logging.getLogger(__name__)


def _recognize_page(
    page: PageImage, chain: list[OcrProvider]
) -> tuple[str, OcrProvider | None, list[str]]:
    """Try each provider until one returns acceptable text.

    Returns `(text, provider, warnings)`. Falls back to the best text seen if
    every provider produced something questionable. The provider itself comes
    back rather than its name, because whether the text can be trusted is a
    property of the engine, not something to infer from what it is called.
    """
    warnings: list[str] = []
    best_text, best_provider, best_score = "", None, -1.0

    for provider in chain:
        available, reason = provider.available()
        if not available:
            warnings.append(f"{provider.name} unavailable: {reason}")
            continue
        try:
            text = provider.recognize(page)
        except OcrUnavailableError as exc:
            warnings.append(f"{provider.name} skipped: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - any provider failure is recoverable
            logger.warning("OCR provider %s failed on page %s: %s", provider.name, page.page_number, exc)
            warnings.append(f"{provider.name} failed: {exc}")
            continue

        report = score_page_text(text)
        if report.score > best_score:
            best_text, best_provider, best_score = text, provider, report.score
        if report.acceptable:
            return text, provider, warnings
        warnings.append(f"{provider.name} low quality: {'; '.join(report.reasons)}")

    return best_text, best_provider, warnings


def run_ocr(pages: list[PageImage], settings: Settings) -> OcrResult:
    """Recognise every page image and return a consolidated result."""
    if not pages:
        return OcrResult(provider="none", degraded=True, notes=("no pages to recognise",))

    chain = provider_chain(settings)
    if not chain:
        return OcrResult(provider="none", degraded=True, notes=("no OCR provider is available",))

    started = time.perf_counter()
    ocr_pages: list[OcrPage] = []
    providers_used: set[str] = set()
    all_notes: list[str] = []

    improvised: set[str] = set()
    for page in pages:
        page_started = time.perf_counter()
        text, provider, warnings = _recognize_page(page, chain)
        provider_name = provider.name if provider is not None else "none"
        providers_used.add(provider_name)
        if provider is None or not provider.purpose_built:
            improvised.add(provider_name)
        all_notes.extend(warnings)
        ocr_pages.append(
            OcrPage(
                page_number=page.page_number,
                text=text,
                provider=provider_name,
                image_path=str(page.path),
                duration_ms=int((time.perf_counter() - page_started) * 1000),
                warnings=tuple(warnings),
            )
        )

    report = score_document_text([page.text for page in ocr_pages])
    degraded = not report.acceptable or bool(improvised)
    if improvised:
        all_notes.append(
            f"recognised with a general-purpose engine ({', '.join(sorted(improvised))}); "
            "accuracy is reduced"
        )

    return OcrResult(
        pages=tuple(ocr_pages),
        provider="+".join(sorted(providers_used)),
        duration_ms=int((time.perf_counter() - started) * 1000),
        degraded=degraded,
        notes=tuple(dict.fromkeys(all_notes + list(report.reasons))),
    )
