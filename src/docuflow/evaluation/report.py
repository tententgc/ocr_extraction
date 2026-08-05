"""Human-readable rendering of an evaluation report."""

from __future__ import annotations

from typing import Any

BAR_WIDTH = 22


def _bar(fraction: float, width: int = BAR_WIDTH) -> str:
    filled = max(0, min(width, round(fraction * width)))
    return "█" * filled + "·" * (width - filled)


def _print_cost(cost: dict[str, Any]) -> None:
    """Where the time went, and what the model calls bought.

    Printed beside accuracy on purpose: a stage that got faster while the numbers
    above it fell is not an improvement, and neither is an agent whose calls all
    end in "could not decide".
    """
    if not cost.get("documents"):
        return

    print("\n--- Where the time went " + "-" * 53)
    print(f"Per document         : {cost['mean_ms'] / 1000:.1f}s mean · "
          f"{cost['median_ms'] / 1000:.1f}s median · {cost['p90_ms'] / 1000:.1f}s p90 · "
          f"{cost['slowest_ms'] / 1000:.1f}s slowest")
    print(f"{'STAGE':14} {'TOTAL':>9} {'MEAN':>8} {'SHARE':>7}")
    for stage, data in cost.get("stages", {}).items():
        print(f"{stage:14} {data['total_ms'] / 1000:>8.1f}s {data['mean_ms'] / 1000:>7.1f}s "
              f"{data['share']:>7.0%}")

    if cost.get("agent_rules_run") or cost.get("reconciler_runs"):
        print("\n--- What the agents did " + "-" * 53)
        if cost.get("agent_rules_run"):
            verdicts = ", ".join(f"{count} {name}"
                                 for name, count in sorted(cost["agent_rule_verdicts"].items()))
            print(f"Agent rules run      : {cost['agent_rules_run']}  ({verdicts})")
            print(f"Could not decide     : {cost['undecided_rate']:.0%}  "
                  f"(held for a person, never passed)")
            print(f"Tool calls           : {cost['agent_rule_tool_calls']}")
        print(f"Reconciler ran on    : {cost['reconciler_runs']}/{cost['documents']} documents "
              f"({cost['reconciler_tool_calls']} tool calls)")
        if cost.get("repeated_tool_calls"):
            # A call that asked something already answered. Not free: each one
            # is a turn of a bounded budget spent learning nothing.
            print(f"Repeated calls       : {cost['repeated_tool_calls']}  "
                  f"(asked something already answered)")


def _print_effort(effort: dict[str, Any]) -> None:
    """Time saved, with the assumptions it rests on printed beside it.

    Accuracy and coverage are measured. This is modelled, and a modelled number
    presented like a measured one is how a report ends up claiming a saving
    nobody can find afterwards. So the assumptions are printed every time, not
    kept in a footnote.
    """
    if not effort.get("documents"):
        return

    print("\n--- Time saved " + "-" * 62)
    print(f"By hand              : {effort['manual_minutes']:.0f} min "
          f"({effort['manual_days_per_thousand_documents']} days per 1,000 documents)")
    print(f"Through here         : {effort['assisted_minutes']:.0f} min "
          f"({effort['days_per_thousand_documents']} days per 1,000 documents)")
    if effort["rework_minutes"]:
        print(f"  of which rework    : {effort['rework_minutes']:.0f} min "
              f"cleaning up posted errors")
    print(f"Saved                : {effort['saved_minutes']:.0f} min "
          f"({effort['saved_share']:.0%} of the manual effort)")
    print(f"Machine time         : {effort['machine_minutes']:.1f} min wall clock "
          f"(nobody waits for it in a batch)")

    print("  assuming, per document:")
    for name, value in effort.get("assumptions", {}).items():
        print(f"    {name.replace('_', ' '):48} {value:>5.1f} min")


def print_report(report: dict[str, Any]) -> None:
    """Print the accuracy report to stdout."""
    summary = report.get("summary", {})
    if not summary.get("documents"):
        print("No documents were evaluated.")
        return

    print("\n" + "=" * 78)
    print("ACCURACY REPORT")
    print("=" * 78)
    print(f"Documents evaluated : {summary['documents']}  ({summary['elapsed_seconds']}s)")
    print(f"Document type        : {_bar(summary['type_accuracy'])} {summary['type_accuracy']:.1%}")
    print(f"Field accuracy       : {_bar(summary['field_accuracy'])} {summary['field_accuracy']:.1%} "
          f"({summary['fields_correct']}/{summary['fields_total']} exact)")
    print(f"  allowing near-miss : {_bar(summary['field_accuracy_with_near'])} "
          f"{summary['field_accuracy_with_near']:.1%}")
    print(f"  not found          : {summary['fields_missing']}")

    print("\n--- Straight-through processing " + "-" * 45)
    print(f"Auto-posted          : {summary['auto_posted']}/{summary['documents']} "
          f"({summary['auto_post_rate']:.0%})")
    print(f"Sent to a human      : {summary['documents'] - summary['auto_posted']}")

    print("\n--- Business safety " + "-" * 57)
    silent = summary["silent_errors"]
    verdict = "PASS" if silent == 0 else "FAIL"
    print(f"Silent error rate    : {summary['silent_error_rate']:.1%}  [{verdict}]")
    print("  (auto-posted documents carrying a wrong critical value)")
    if summary["silent_error_documents"]:
        for doc_id in summary["silent_error_documents"]:
            print(f"    ! {doc_id}")
    print(f"Auto-post precision  : {summary['auto_post_precision']:.1%}")
    print(f"Planted defects      : {summary['planted_defects_caught']}/{summary['planted_defects']} caught "
          f"({summary['defect_recall']:.0%} recall)")

    _print_effort(report.get("effort") or {})
    _print_cost(report.get("cost") or {})

    breakdown = report.get("field_breakdown", {})
    weak = [(name, data) for name, data in breakdown.items() if data["accuracy"] < 1.0]
    if weak:
        print("\n--- Fields needing work " + "-" * 53)
        print(f"{'FIELD':24} {'ACCURACY':>9}  {'OK':>3} {'WRONG':>5} {'NEAR':>4} {'MISS':>4}")
        for name, data in weak[:15]:
            print(f"{name:24} {data['accuracy']:>8.0%}  {data['correct']:>3} "
                  f"{data['wrong']:>5} {data['near']:>4} {data['missing']:>4}")

    print("\n--- Per document " + "-" * 60)
    print(f"{'DOCUMENT':26} {'TYPE':>5} {'FIELDS':>8} {'DECISION':>14} {'CONF':>6} DEFECTS")
    for item in report.get("documents", []):
        type_mark = "ok" if item["type_correct"] else "WRONG"
        fields = f"{item['fields_correct']}/{item['fields_total']}"
        defects = ""
        if item["expected_findings"]:
            missed = item["expected_findings_missed"]
            defects = "all caught" if not missed else f"MISSED {', '.join(missed)}"
        if item["silent_error"]:
            defects += f"  SILENT ERROR: {', '.join(item['wrong_critical'])}"
        print(f"{item['doc_id']:26} {type_mark:>5} {fields:>8} {item['decision']:>14} "
              f"{item['confidence']:>6.0%} {defects}")

    print("\n--- Wrong values " + "-" * 60)
    any_wrong = False
    for item in report.get("documents", []):
        for comparison in item["comparisons"]:
            if comparison["status"] in {"wrong", "near", "spurious"}:
                any_wrong = True
                print(f"{item['doc_id']:26} {comparison['field']:20} "
                      f"expected {str(comparison['expected'])[:26]!r} got {str(comparison['actual'])[:26]!r}")
    if not any_wrong:
        print("(none)")
    print()
