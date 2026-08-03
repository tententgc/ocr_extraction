"""Business rules: arithmetic, master-data cross-checks and policy."""

from docuflow.rules.context import RuleContext
from docuflow.rules.engine import (
    FREE_MODE_RULE_ID,
    RuleInfo,
    all_rules,
    describe_rules,
    failed,
    load_all_rules,
    passed,
    rule,
    rules_for,
    run_rules,
)

load_all_rules()

__all__ = [
    "FREE_MODE_RULE_ID",
    "RuleContext",
    "RuleInfo",
    "all_rules",
    "describe_rules",
    "failed",
    "load_all_rules",
    "passed",
    "rule",
    "rules_for",
    "run_rules",
]
