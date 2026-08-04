"""What a document costs, and what the model calls bought.

The accuracy harness answers "is it right". This answers "what did it take" -
where the seconds went, how many model calls were made, and what the agents did
with them. The two belong together: an optimisation that shaves a stage is only
worth having if the accuracy report next to it did not move, and an agent is only
worth its call if this shows the call reaching a conclusion.

Nothing here runs anything. Every number is read back off the stored trace, which
the pipeline has been writing all along.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from docuflow.models import DocumentRun

# Stages in the order they run, so a report reads like the pipeline.
STAGE_ORDER = ("render", "ocr", "classify", "extract", "line_items", "validate",
               "route", "reconcile")


def stage_timings(run: DocumentRun) -> dict[str, int]:
    """Milliseconds per stage for one document."""
    return {step.step: step.duration_ms for step in run.trace if step.duration_ms}


def agent_activity(run: DocumentRun) -> dict[str, Any]:
    """What the agents did on one document, from the trace and the findings.

    Deliberately counts *outcomes*, not intentions: a rule that ran and could not
    decide is the interesting number, and it is the one nobody was collecting.
    """
    verdicts: Counter[str] = Counter()
    tools = 0
    repeated = 0
    for finding in run.findings:
        detail = finding.detail or {}
        if detail.get("kind") != "agent":
            continue
        verdicts[str(detail.get("verdict") or "unknown")] += 1
        tools += len(detail.get("tools_used") or ())
        repeated += int(detail.get("repeated_calls") or 0)

    reconcile = next((step for step in run.trace if step.step == "reconcile"), None)
    reconciler_tools = 0
    if reconcile:
        reconciler_tools = len(reconcile.detail.get("tool_calls") or ())
        repeated += int(reconcile.detail.get("repeated_calls") or 0)

    return {
        "agent_rules_run": sum(verdicts.values()),
        "agent_rule_verdicts": dict(verdicts),
        "agent_rule_tool_calls": tools,
        # Calls that asked something already answered in the same run - the shape
        # of an agent going in circles rather than investigating.
        "repeated_tool_calls": repeated,
        "reconciler_ran": reconcile is not None,
        "reconciler_tool_calls": reconciler_tools,
    }


def document_cost(run: DocumentRun) -> dict[str, Any]:
    """Timings and agent activity for one document."""
    stages = stage_timings(run)
    return {
        "total_ms": sum(stages.values()),
        "stage_ms": stages,
        **agent_activity(run),
    }


def _percentile(values: list[int], fraction: float) -> int:
    """The slowest documents are the ones that matter; a mean hides them."""
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def summarize_cost(costs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-document cost into the shape a report prints."""
    if not costs:
        return {"documents": 0}

    totals = [item["total_ms"] for item in costs]
    per_stage: dict[str, list[int]] = {}
    for item in costs:
        for stage, milliseconds in item["stage_ms"].items():
            per_stage.setdefault(stage, []).append(milliseconds)

    verdicts: Counter[str] = Counter()
    for item in costs:
        verdicts.update(item.get("agent_rule_verdicts") or {})

    ordered = [stage for stage in STAGE_ORDER if stage in per_stage]
    ordered += sorted(stage for stage in per_stage if stage not in STAGE_ORDER)

    return {
        "documents": len(costs),
        "total_ms": sum(totals),
        "mean_ms": round(sum(totals) / len(totals)),
        "median_ms": _percentile(totals, 0.5),
        "p90_ms": _percentile(totals, 0.9),
        "slowest_ms": max(totals),
        "stages": {
            stage: {
                "total_ms": sum(per_stage[stage]),
                "mean_ms": round(sum(per_stage[stage]) / len(per_stage[stage])),
                "share": round(sum(per_stage[stage]) / sum(totals), 3) if sum(totals) else 0.0,
                "documents": len(per_stage[stage]),
            }
            for stage in ordered
        },
        "agent_rules_run": sum(item["agent_rules_run"] for item in costs),
        "agent_rule_verdicts": dict(verdicts),
        # Held because the agent could not decide. A rate that climbs is a plan
        # that stopped working, and without this nobody would find out.
        "undecided_rate": round(
            verdicts.get("unsure", 0) / sum(verdicts.values()), 3
        ) if verdicts else 0.0,
        "agent_rule_tool_calls": sum(item["agent_rule_tool_calls"] for item in costs),
        "repeated_tool_calls": sum(item["repeated_tool_calls"] for item in costs),
        "reconciler_runs": sum(1 for item in costs if item["reconciler_ran"]),
        "reconciler_tool_calls": sum(item["reconciler_tool_calls"] for item in costs),
    }
