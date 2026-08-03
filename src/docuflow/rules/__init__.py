"""Business rules: arithmetic, master-data cross-checks and policy."""

from docuflow.rules.context import RuleContext
from docuflow.rules.engine import failed, load_all_rules, passed, rule, run_rules, rules_for

load_all_rules()

__all__ = [
    "RuleContext",
    "failed",
    "load_all_rules",
    "passed",
    "rule",
    "rules_for",
    "run_rules",
]
