"""A single value proposed by one extraction channel.

The pipeline compares two independent channels. Both produce `Candidate`s so
the merge logic does not care which channel is which - only whether they agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Where a value came from. `model` is a language model reading the page,
# `label` is the deterministic label matcher used only when no model is
# available, and `letterhead` is a positional guess.
Channel = Literal["model", "verify", "label", "letterhead"]


@dataclass(frozen=True)
class Candidate:
    """One channel's reading of one field."""

    field: str
    raw_value: str
    evidence: str | None = None
    page: int | None = None
    channel: Channel = "model"

    @property
    def is_positional(self) -> bool:
        """True when the value came from page position rather than a label."""
        return self.channel == "letterhead"


def from_llm(field: str, raw_value: str, evidence: str | None, channel: Channel = "model") -> Candidate:
    return Candidate(field=field, raw_value=raw_value, evidence=evidence, channel=channel)


def from_rule(rule_candidate) -> Candidate:
    """Adapt a `RuleCandidate` from the deterministic fallback extractor."""
    return Candidate(
        field=rule_candidate.field,
        raw_value=rule_candidate.raw_value,
        evidence=rule_candidate.evidence,
        page=rule_candidate.page,
        channel="letterhead" if rule_candidate.heuristic else "label",
    )
