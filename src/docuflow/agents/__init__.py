"""The agent layer: classify, extract, ground, merge, reconcile."""

from docuflow.agents.classifier import Classification, classify, classify_by_keywords
from docuflow.agents.grounding import find_evidence, value_supported_by_evidence
from docuflow.agents.llm_extractor import LlmCandidate, extract_with_llm
from docuflow.agents.merge import merge_all, merge_field
from docuflow.agents.rule_extractor import (
    RuleCandidate,
    detect_currency_from_text,
    extract_with_rules,
    find_line_items,
)

__all__ = [
    "Classification",
    "LlmCandidate",
    "RuleCandidate",
    "classify",
    "classify_by_keywords",
    "detect_currency_from_text",
    "extract_with_llm",
    "extract_with_rules",
    "find_evidence",
    "find_line_items",
    "merge_all",
    "merge_field",
    "value_supported_by_evidence",
]
