"""Persistence for user-authored rules.

Stored as JSON rather than as columns: a rule's condition is a tree, and the
shape of that tree is the expression language's business, not the database's.
"""

from __future__ import annotations

from pathlib import Path

from docuflow.db import connect, dump_json, load_json, transaction
from docuflow.rules.custom import CustomRule, CustomRuleError


class CustomRuleStore:
    """Read/write access to the rules a user wrote."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def list_rules(self) -> list[CustomRule]:
        rows = connect(self.db_path).execute(
            "SELECT payload FROM custom_rules ORDER BY created_at, rule_id"
        ).fetchall()
        return [CustomRule.from_json(load_json(row["payload"])) for row in rows]

    def active_for(self, doc_type: str) -> list[CustomRule]:
        """Enabled rules that apply to this document type."""
        return [rule for rule in self.list_rules() if rule.enabled and rule.applies_to(doc_type)]

    def get(self, rule_id: str) -> CustomRule | None:
        row = connect(self.db_path).execute(
            "SELECT payload FROM custom_rules WHERE rule_id = ?", (rule_id,)
        ).fetchone()
        return CustomRule.from_json(load_json(row["payload"])) if row else None

    def save(self, rule: CustomRule, *, replace: bool = False) -> CustomRule:
        """Store a rule. Refuses to overwrite silently."""
        if not replace and self.get(rule.rule_id) is not None:
            raise CustomRuleError(
                f"A rule called '{rule.rule_id}' already exists. Rename it or edit that one."
            )
        with transaction(self.db_path) as connection:
            connection.execute(
                """INSERT INTO custom_rules (rule_id, enabled, created_at, payload)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(rule_id) DO UPDATE SET enabled = excluded.enabled,
                                                      payload = excluded.payload""",
                (rule.rule_id, int(rule.enabled), rule.created_at, dump_json(rule.to_json())),
            )
        return rule

    def set_enabled(self, rule_id: str, enabled: bool) -> CustomRule:
        rule = self.get(rule_id)
        if rule is None:
            raise CustomRuleError(f"There is no rule called '{rule_id}'.")
        from dataclasses import replace as replace_field

        return self.save(replace_field(rule, enabled=enabled), replace=True)

    def delete(self, rule_id: str) -> bool:
        with transaction(self.db_path) as connection:
            cursor = connection.execute("DELETE FROM custom_rules WHERE rule_id = ?", (rule_id,))
        return cursor.rowcount > 0
