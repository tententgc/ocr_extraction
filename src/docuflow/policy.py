"""Runtime business-policy overrides.

`.env` declares the policy a deployment ships with. This holds the deliberate
departures a user makes from the settings screen - which rules are switched off,
whether they are all switched off, and how confident the pipeline must be before
it posts without a human.

Two properties matter:

  * The base `Settings` is never mutated. Every read is a fresh overlay, so
    clearing an override always returns to the declared configuration.
  * A name that matches no rule is rejected rather than stored. Silently
    accepting a typo would leave someone believing a check was off - or on -
    when it was not, and this file exists to make that belief accurate.

Unlike the model picker, these survive a restart: turning a business rule off is
a decision about the business, not a transient choice of engine.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from docuflow.config import Settings
from docuflow.db import connect, dump_json, load_json, transaction
from docuflow.rules import all_rules

SETTINGS_KEY = "business_policy"
# Below this, straight-through posting is doing very little checking. Allowed -
# it is the user's ledger - but never without saying so.
LOW_THRESHOLD_WARNING = 0.60


class PolicyError(ValueError):
    """The requested policy change is not valid."""


@dataclass(frozen=True)
class Policy:
    """Overrides on top of the configured settings. `None` means "unchanged"."""

    rules_free_mode: bool | None = None
    disabled_rules: tuple[str, ...] | None = None
    auto_post_min_confidence: float | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "rules_free_mode": self.rules_free_mode,
            "disabled_rules": list(self.disabled_rules) if self.disabled_rules is not None else None,
            "auto_post_min_confidence": self.auto_post_min_confidence,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object] | None) -> "Policy":
        if not payload:
            return cls()
        disabled = payload.get("disabled_rules")
        return cls(
            rules_free_mode=payload.get("rules_free_mode"),
            disabled_rules=tuple(disabled) if isinstance(disabled, list) else None,
            auto_post_min_confidence=payload.get("auto_post_min_confidence"),
        )


def resolve_rule_name(name: str) -> str:
    """Canonical id for a rule named by id, alias or function name."""
    wanted = name.strip()
    if not wanted:
        raise PolicyError("A rule name cannot be blank.")
    for info in all_rules():
        if info.matches(wanted):
            return info.rule_id
    raise PolicyError(f"'{wanted}' is not a known rule.")


class PolicyStore:
    """Process-wide policy overlay, persisted and safe to read from threads."""

    def __init__(self, base: Settings, db_path: Path | None = None) -> None:
        self._base = base
        self._db_path = db_path or base.db_path
        self._lock = threading.Lock()
        self._policy = self._load()

    # --- persistence -------------------------------------------------------
    def _load(self) -> Policy:
        try:
            row = connect(self._db_path).execute(
                "SELECT value FROM app_settings WHERE key = ?", (SETTINGS_KEY,)
            ).fetchone()
        except Exception:  # noqa: BLE001 - a missing table must not stop startup
            return Policy()
        return Policy.from_json(load_json(row["value"]) if row else None)

    def _save(self, policy: Policy) -> None:
        with transaction(self._db_path) as connection:
            connection.execute(
                """INSERT INTO app_settings (key, value, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                                  updated_at = excluded.updated_at""",
                (SETTINGS_KEY, dump_json(policy.to_json()), datetime.now(timezone.utc).isoformat()),
            )

    # --- reads -------------------------------------------------------------
    def current(self) -> Policy:
        with self._lock:
            return self._policy

    def settings(self, base: Settings | None = None) -> Settings:
        """`base` (or the configured settings) with the overrides applied."""
        settings = base if base is not None else self._base
        policy = self.current()
        updates: dict[str, object] = {}
        if policy.rules_free_mode is not None:
            updates["rules_free_mode"] = policy.rules_free_mode
        if policy.disabled_rules is not None:
            updates["disabled_rules"] = ",".join(policy.disabled_rules)
        if policy.auto_post_min_confidence is not None:
            updates["auto_post_min_confidence"] = policy.auto_post_min_confidence
        return settings.model_copy(update=updates) if updates else settings

    def describe(self, base: Settings | None = None) -> dict[str, object]:
        """Effective policy plus what the configuration alone would have said."""
        effective = self.settings(base)
        configured = base if base is not None else self._base
        policy = self.current()
        return {
            "rules_free_mode": effective.rules_free_mode,
            "disabled_rules": sorted(effective.skipped_rules),
            "auto_post_min_confidence": effective.auto_post_min_confidence,
            "field_review_confidence": effective.field_review_confidence,
            "overridden": [
                name
                for name, value in (
                    ("rules_free_mode", policy.rules_free_mode),
                    ("disabled_rules", policy.disabled_rules),
                    ("auto_post_min_confidence", policy.auto_post_min_confidence),
                )
                if value is not None
            ],
            "configured": {
                "rules_free_mode": configured.rules_free_mode,
                "disabled_rules": sorted(configured.skipped_rules),
                "auto_post_min_confidence": configured.auto_post_min_confidence,
            },
            "warnings": warnings_for(effective),
        }

    # --- writes ------------------------------------------------------------
    def update(
        self,
        *,
        rules_free_mode: bool | None = None,
        disabled_rules: list[str] | None = None,
        auto_post_min_confidence: float | None = None,
    ) -> Policy:
        """Validate and store a change. Only the fields given are touched."""
        changes: dict[str, object] = {}

        if rules_free_mode is not None:
            changes["rules_free_mode"] = bool(rules_free_mode)

        if disabled_rules is not None:
            # Resolving each name both validates it and collapses aliases, so
            # "po_vendor_matches" and "po_exists_and_matches" store as one id.
            changes["disabled_rules"] = tuple(
                sorted({resolve_rule_name(name) for name in disabled_rules})
            )

        if auto_post_min_confidence is not None:
            threshold = float(auto_post_min_confidence)
            if not 0.0 <= threshold <= 1.0:
                raise PolicyError(
                    f"The auto-post threshold must be between 0 and 1, not {threshold}."
                )
            changes["auto_post_min_confidence"] = threshold

        if not changes:
            return self.current()

        with self._lock:
            self._policy = replace(self._policy, **changes)
            self._save(self._policy)
            return self._policy

    def reset(self) -> Policy:
        """Drop every override and return to the configured policy."""
        with self._lock:
            self._policy = Policy()
            self._save(self._policy)
            return self._policy


def warnings_for(settings: Settings) -> list[str]:
    """Plain-language consequences of the current policy, for the UI to show."""
    notes: list[str] = []
    if settings.rules_free_mode:
        notes.append(
            "Free mode is on: no business rule runs, so nothing is checked against "
            "master data, arithmetic or prior postings."
        )
    disabled = settings.skipped_rules
    if disabled and not settings.rules_free_mode:
        notes.append(
            f"{len(disabled)} rule(s) switched off: {', '.join(sorted(disabled))}."
        )
    if settings.auto_post_min_confidence < LOW_THRESHOLD_WARNING:
        notes.append(
            f"The auto-post threshold is {settings.auto_post_min_confidence:.0%}. "
            "Documents the pipeline is unsure about will post without a human."
        )
    return notes
