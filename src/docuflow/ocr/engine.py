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
from docuflow.ocr.base import OcrProvider, OcrTransientError, OcrUnavailableError, PageImage
from docuflow.ocr.factory import provider_chain
from docuflow.ocr.quality import score_document_text, score_page_text

logger = logging.getLogger(__name__)

# How many times a provider that did not answer is asked again before the page
# moves down the chain. **One retry, not two**, and the number was measured
# rather than chosen: a page that provokes a timeout usually provokes it every
# time, and each attempt costs a full server-side timeout before it fails. Three
# attempts on a Thai-numeral page added three minutes to a document that was
# going to fall back to the local engine regardless.
#
# One retry buys what retrying is actually for - a single unlucky call - and
# caps what being wrong costs at one wasted timeout.
TRANSIENT_ATTEMPTS = 2
# Seconds before the retry. Short, because the thing being ridden out is one
# slow request rather than a struggling service.
RETRY_BACKOFF = (1.0,)


def _ask(provider: OcrProvider, page: PageImage) -> tuple[str, list[str]]:
    """Recognise one page with one provider, riding out transient failures.

    A provider that returns poor text will return poor text again, so the engine
    escalates. A provider that did not *answer* has said nothing about the page
    at all, and escalating on that is how a two-second cloud call turns into ten
    minutes on a local model - which is exactly what a Typhoon 408 did to every
    Thai-numeral page in the corpus.
    """
    notes: list[str] = []
    for attempt in range(TRANSIENT_ATTEMPTS):
        try:
            return provider.recognize(page), notes
        except OcrTransientError as exc:
            notes.append(f"{provider.name} did not answer (attempt {attempt + 1}): {exc}")
            if attempt == TRANSIENT_ATTEMPTS - 1:
                raise
            logger.info("OCR provider %s timed out on page %s, retrying",
                        provider.name, page.page_number)
            time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)])
    raise OcrTransientError(f"{provider.name} did not answer")  # unreachable, kept total


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
            text, retries = _ask(provider, page)
            warnings.extend(retries)
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

    # Which engine read the page is not a detail when the chain steps from a
    # hosted API onto a local model: the same document goes from three seconds
    # to ten minutes, and without this line the only evidence is a clock. Said
    # separately from `degraded`, because falling back to a *better* engine that
    # happens to be slower is not a quality problem and must not be reported as
    # one.
    configured = settings.resolved_ocr_provider()
    fell_back = {name for name in providers_used if name not in {configured, "none"}}
    if fell_back and chain and chain[0].name not in providers_used:
        all_notes.append(
            f"the configured engine ({chain[0].name}) did not read this document; "
            f"{', '.join(sorted(fell_back))} stood in, which can be far slower"
        )

    return OcrResult(
        pages=tuple(ocr_pages),
        provider="+".join(sorted(providers_used)),
        duration_ms=int((time.perf_counter() - started) * 1000),
        degraded=degraded,
        notes=tuple(dict.fromkeys(all_notes + list(report.reasons))),
    )
