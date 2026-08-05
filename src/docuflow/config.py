"""Environment-driven configuration.

Every tunable lives here so nothing downstream hardcodes a threshold, model name
or path. Settings are read once at import time and treated as immutable.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Provider keys are validated against the plugin registries at build time, so
# they stay plain strings here - registering a new plugin needs no edit to this
# file.
OcrProviderName = str
LlmProviderName = str


class Settings(BaseSettings):
    """Runtime configuration, populated from the environment and `.env`."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OCR ---------------------------------------------------------------
    ocr_provider: OcrProviderName = "local"
    ocr_local_model: str = "mlx-community/typhoon-ocr1.5-2b-8bit"
    # Explicit 1.5 rather than the `typhoon-ocr` alias, so a future alias change
    # cannot silently move us onto a different model version.
    ocr_cloud_model: str = "typhoon-ocr-v1.5"
    ocr_max_pages: int = Field(default=8, ge=1, le=100)
    ocr_image_dim: int = Field(default=1800, ge=512, le=4096)
    ocr_max_tokens: int = Field(default=8192, ge=512, le=32768)
    # Straighten, relight and sharpen each page before the model reads it. Every
    # correction is gated on a measurement and reverted if it does not help, so
    # a clean digital PDF passes through untouched; this is here for the photos
    # and the faxes. Turn it off to measure the OCR engine on raw input.
    ocr_preprocess: bool = True
    # Disable engine fallback. Production wants fallback (a missing model should
    # degrade a run, not break it), but a benchmark must measure the engine it
    # names - otherwise a weak engine silently scores its fallback's accuracy.
    ocr_strict: bool = False

    # --- LLM ---------------------------------------------------------------
    llm_provider: LlmProviderName = "auto"
    # The second, independent extraction pass. `auto` picks a *different*
    # backend from the primary where one is available, because two runs of the
    # same model repeat the same mistake and the agreement signal is worthless.
    # `off` disables verification and leaves the run on a single channel.
    llm_verify_provider: LlmProviderName = "auto"
    llm_verify_model: str = ""
    # Time budget for the second pass. It is a cross-check, not the extraction:
    # if it cannot answer in this long the run falls back to the deterministic
    # label matcher rather than holding the document. Reasoning models are the
    # reason this exists - a local 4B model that "thinks" before answering can
    # spend minutes on a page the primary pass read in seconds.
    llm_verify_timeout_seconds: float = Field(default=90.0, gt=0.0)
    llm_cloud_model: str = "typhoon-v2.5-30b-a3b-instruct"
    llm_local_model: str = "tawankri/typhoon2.5-qwen3-4b-mlx-4Bit"

    # --- Typhoon Cloud -----------------------------------------------------
    # Never give a credential a default here: this file is committed, `.env` is
    # not. The key is read from TYPHOON_API_KEY / .env at startup.
    typhoon_api_key: str = ""
    typhoon_base_url: str = "https://api.opentyphoon.ai/v1"

    # --- Ollama (fully local, no key) --------------------------------------
    ollama_base_url: str = "http://localhost:11434/v1"
    ocr_ollama_model: str = "scb10x/typhoon-ocr1.5-3b"
    # qwen3 rather than scb10x/typhoon2.1-gemma3-4b: the latter's chat template
    # fails to load under Ollama 0.32.x ("unknown test 'tool_calls'"). Switch it
    # back once that model ships a fixed template.
    llm_ollama_model: str = "qwen3:4b"

    # --- OpenAI ------------------------------------------------------------
    openai_api_key: str = ""
    openai_base_url: str = ""  # blank uses the official endpoint
    llm_openai_model: str = "gpt-5.1"
    ocr_openai_model: str = "gpt-5.1"

    # --- Anthropic (Claude) ------------------------------------------------
    anthropic_api_key: str = ""
    anthropic_base_url: str = ""
    # OCR runs on every page, so it takes the balanced model; the agent runs only
    # after a check has failed, so it can afford the strongest one.
    llm_anthropic_model: str = "claude-opus-5"
    ocr_anthropic_model: str = "claude-sonnet-5"

    # --- Routing thresholds ------------------------------------------------
    auto_post_min_confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    field_review_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    # How far an invoice may differ from the purchase order it quotes: a
    # commercial allowance for freight, part shipments and price movement.
    amount_tolerance: float = Field(default=0.02, ge=0.0)
    # How far a document may differ from itself. Rounding only - subtotal plus
    # tax either makes the total or the document is wrong. Kept separate from
    # the commercial allowance above, which is thousands of times looser and
    # would let a misread digit through as "totals add up".
    arithmetic_tolerance: float = Field(default=0.0001, ge=0.0)

    # --- Business policy ---------------------------------------------------
    # These are finance policy, not code. Change them here (or in .env) rather
    # than editing rule modules.
    po_required_above: float = Field(default=50_000.0, ge=0.0)
    expected_vat_rates: str = "0,7"
    # Document types where an unregistered counterparty blocks posting. A
    # receipt records money already spent and a vendor onboarding form exists
    # because the applicant is not registered yet, so neither belongs here.
    unknown_vendor_blocks: str = "invoice,purchase_order"
    contract_expiry_warning_days: int = Field(default=90, ge=0)
    max_contract_years: int = Field(default=25, ge=1)
    future_date_tolerance_days: int = Field(default=2, ge=0)
    # Take the issuing party from the letterhead when no label names it.
    use_letterhead_heuristic: bool = True

    # --- Ollama runtime ----------------------------------------------------
    # How long Ollama holds a model in memory after the last request. Its own
    # default is 5 minutes, which in a demo means almost every document pays a
    # cold model load; that measured 168s against 15s warm.
    ollama_keep_alive: str = "30m"
    # Context window for local OCR. The page image alone costs ~3,100 tokens
    # and Ollama's default window is 4096, which leaves too little room for a
    # dense page - the model then transcribes a page it can no longer see.
    ollama_num_ctx: int = Field(default=8192, ge=2048)
    # Load the local models at server start instead of on the first document.
    ollama_preload: bool = True

    # --- Rule execution ----------------------------------------------------
    # Free mode runs extraction only: no rule is evaluated, nothing is checked
    # against master data, and nothing can block. Meant for exploring arbitrary
    # documents that were never going to match a master file - not for a run
    # whose output anyone intends to trust.
    rules_free_mode: bool = False
    # Comma-separated rule ids to skip while leaving the rest in force, e.g.
    # "vendor_known,po_exists". Ids are the ones shown in the UI and the API.
    disabled_rules: str = ""

    # rule_id -> severity, set from the settings screen. Serialised because
    # Settings is the one object both entry points already receive.
    rule_severity_json: str = ""

    @property
    def rule_severity_overrides(self) -> dict[str, str]:
        """Parsed `rule_severity_json`; empty when nothing is re-graded."""
        import json

        try:
            parsed = json.loads(self.rule_severity_json or "{}")
        except ValueError:
            return {}
        return {str(k): str(v) for k, v in parsed.items()} if isinstance(parsed, dict) else {}

    @property
    def skipped_rules(self) -> frozenset[str]:
        """Parsed `disabled_rules`."""
        return frozenset(
            part.strip() for part in str(self.disabled_rules).split(",") if part.strip()
        )

    @property
    def vat_rates(self) -> tuple[float, ...]:
        """Parsed `expected_vat_rates`, e.g. "0,7" -> (0.0, 7.0)."""
        rates = []
        for part in str(self.expected_vat_rates).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                rates.append(float(part))
            except ValueError:
                continue
        return tuple(rates)

    @property
    def vendor_blocking_types(self) -> frozenset[str]:
        """Parsed `unknown_vendor_blocks`."""
        return frozenset(
            part.strip() for part in str(self.unknown_vendor_blocks).split(",") if part.strip()
        )

    # Switch off the checks that ship with the system while keeping the ones
    # written here. Our rules encode a generic finance policy; a company whose
    # own policy differs should be able to keep their checks without also being
    # told what we think about VAT rates.
    builtin_rules_disabled: bool = False

    # --- Master data -------------------------------------------------------
    # The bundled demo register - five vendors, four purchase orders, two
    # contracts - so a fresh install has something for documents to match
    # against. It is only ever seeded into an *empty* master file, so importing
    # your own replaces it permanently. Set this false to keep it out entirely,
    # including after you clear the master file.
    seed_demo_master_data: bool = True

    # --- Ingest ------------------------------------------------------------
    allow_url_fetch: bool = True
    max_document_bytes: int = Field(default=25 * 1024 * 1024, ge=1024)

    # --- Storage -----------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"

    @field_validator("typhoon_api_key", mode="before")
    @classmethod
    def _fallback_ocr_key(cls, value: str | None) -> str:
        """`TYPHOON_OCR_API_KEY` is the name the typhoon-ocr package uses."""
        return value or os.getenv("TYPHOON_OCR_API_KEY", "")

    # --- Derived paths -----------------------------------------------------
    @property
    def blob_dir(self) -> Path:
        return self.data_dir / "blobs"

    @property
    def page_dir(self) -> Path:
        return self.data_dir / "pages"

    @property
    def master_dir(self) -> Path:
        return self.data_dir / "master"

    @property
    def sample_dir(self) -> Path:
        return self.data_dir / "samples"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "docuflow.db"

    @property
    def has_cloud_key(self) -> bool:
        return bool(self.typhoon_api_key.strip())

    def resolved_ocr_provider(self) -> str:
        """Turn `auto` into a concrete provider name."""
        if self.ocr_provider != "auto":
            return self.ocr_provider
        return "typhoon_api" if self.has_cloud_key else "ollama"

    def ensure_directories(self) -> None:
        for path in (self.blob_dir, self.page_dir, self.master_dir, self.sample_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
