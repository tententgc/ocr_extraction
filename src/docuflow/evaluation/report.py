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


def _print_setup(setup: dict[str, Any]) -> None:
    """What was running when these numbers were produced.

    A report that does not name the corpus, the providers and the rules that
    were switched off is a report about a system nobody else has.
    """
    if not setup:
        return
    print("\n--- The run " + "-" * 65)
    print(f"Corpus               : {setup.get('corpus', '?')}")
    master = setup.get("master_loaded") or {}
    if master:
        print("Master data          : "
              + ", ".join(f"{count} {kind.replace('_', ' ')}" for kind, count in master.items()))
    rules = setup.get("rules_loaded") or []
    print(f"Rules written here   : {len(rules)}" + (f"  ({', '.join(rules)})" if rules else ""))
    print(f"Auto-post threshold  : {setup.get('auto_post_threshold', 0):.0%}")
    print(f"OCR / read / verify  : {setup.get('ocr_provider')} / "
          f"{setup.get('llm_provider')} / {setup.get('llm_verify_provider')}")

    off = setup.get("disabled_rules") or []
    if off:
        print(f"Rules switched off   : {', '.join(off)}")
    # The trap this exists for: a rule the corpus expects but the machine has
    # disabled reports as a missed defect, which is the one metric that must
    # never move for a reason outside the code.
    back_on = setup.get("rules_re_enabled") or []
    if back_on:
        print(f"  re-enabled for this run: {', '.join(back_on)}")
        print("  (the corpus declares them; leaving them off would report a")
        print("   missed defect where nothing was missed)")


def _print_ocr(ocr: dict[str, Any], recognised: dict[str, Any], from_text_layer: int = 0) -> None:
    """Was the page read at all - the number every other number sits on.

    The recognised subset leads. A PDF read through its own text layer has a
    transcript identical to the reference by construction, so pooling those in
    reports a character accuracy the OCR never earned.
    """
    if not ocr.get("documents"):
        return
    print("\n--- Was the page read " + "-" * 55)

    if recognised.get("documents"):
        print(f"Character accuracy   : {_bar(recognised['character_accuracy'])} "
              f"{recognised['character_accuracy']:.1%}  "
              f"(CER {recognised['character_error_rate']:.2%})   "
              f"over {recognised['documents']} recognised page(s)")
        print(f"Value recall         : {_bar(recognised['value_recall'])} "
              f"{recognised['value_recall']:.1%}  "
              f"({recognised['values_found']}/{recognised['values_expected']} values survived)")
    if from_text_layer:
        print(f"\nRead without OCR     : {from_text_layer} PDF(s) served from their own text")
        print("  layer. Their transcript IS the reference, so their CER is zero by")
        print("  construction and is excluded from the figure above.")
        print(f"  Corpus-wide, diluted : {ocr['character_accuracy']:.1%} character, "
              f"{ocr['value_recall']:.1%} value recall")
    print("\n  CER describes the transcript; value recall describes the values that")
    print("  become a payment. They disagree, which is why both are here.")
    worst = ocr.get("worst_documents") or []
    if worst:
        print(f"\n{'READ WORST':44} {'CHAR':>6} {'VALUES':>7}  MISSING")
        for score in worst[:5]:
            missing = ", ".join(score["values_missing"][:3]) or "-"
            print(f"{score['document'][:43]:44} {score['character_accuracy']:>6.1%} "
                  f"{score['value_recall']:>7.0%}  {missing}")


def _print_gatekeeping(gate: dict[str, Any]) -> None:
    """Did the gate open for the right documents, and did the rules earn it."""
    if not gate.get("documents"):
        return
    print("\n--- Did the gate open correctly " + "-" * 45)
    print(f"Decision accuracy    : {_bar(gate['decision_accuracy'])} "
          f"{gate['decision_accuracy']:.1%}")
    # Never summed. One is a wrong payment, the other a wasted minute.
    print(f"Missed defects       : {gate['missed_defects']}/{gate['should_hold']} "
          f"({gate['missed_defect_rate']:.0%})   ← a wrong payment each")
    for name in gate.get("missed_defect_documents", []):
        print(f"    ! {name}")
    print(f"False alarms         : {gate['false_alarms']}/{gate['should_post']} "
          f"({gate['false_alarm_rate']:.0%})   ← a wasted review each")
    for name in gate.get("false_alarm_documents", []):
        print(f"    · {name}")

    print(f"\nRules exercised      : {gate['rules_tested']} tested, "
          f"{gate['rules_untested']} untested")
    print("  (a rule no document exercises is untested, not passing)")
    leaky = gate.get("rules_that_missed") or []
    if leaky:
        print(f"{'RULE THAT MISSED':34} {'RECALL':>7}  ON")
        for rule in leaky:
            print(f"{rule['rule_id']:34} {rule['recall']:>7.0%}  "
                  f"{', '.join(rule['missed_on'][:2])}")


def _print_integrity(check: dict[str, Any]) -> None:
    """What the answer sheet is wrong about, printed before what it graded.

    On this project the corpus has been wrong more often than the pipeline, and
    every time it looked like a system fault first.
    """
    if not check.get("documents"):
        return

    print("\n--- Is the corpus gradeable " + "-" * 49)
    if check.get("clean"):
        print("Every gradeable value appears in the text the renderer drew.")
        return

    undrawn = check.get("undrawn_values") or 0
    if undrawn:
        print(f"!! {undrawn} of {check['gradeable_values']} ground-truth values "
              f"({check['undrawn_share']:.1%}) are NOT printed on their page.")
        print("   Extraction is graded against values no reader could find, so they")
        print("   report as missing on every engine anyone tries. Field recall below")
        print("   is understated by this much; nothing has been silently dropped.")
        print(f"\n   {'FIELD':22} {'N':>3}  DOCUMENTS")
        for field, count in (check.get("undrawn_by_field") or {}).items():
            where = sorted({
                item["document"].split("_")[0]
                for item in check.get("undrawn", [])
                if item["field"] == field
            }, key=lambda part: int(part) if part.isdigit() else 0)
            print(f"   {field:22} {count:>3}  documents {', '.join(where)}")

    if check.get("thin_reference"):
        print(f"\n!! {len(check['thin_reference'])} page(s) have almost no text layer, so")
        print("   nothing can be proved about their ground truth either way:")
        for name in check["thin_reference"]:
            print(f"     · {name}")

    if check.get("documents_without_ground_truth"):
        print("\n!! documents with nothing to be graded against:")
        for name in check["documents_without_ground_truth"]:
            print(f"     · {name}")

    if check.get("held_without_a_declared_rule"):
        print("\n · held but naming no rule (the defect is real but unstated, so no")
        print("   rule gets credit for catching it and defect recall cannot see it):")
        for name in check["held_without_a_declared_rule"]:
            print(f"     · {name}")


def _cell(value: float | None) -> str:
    """A score, or a dash where there is nothing to score.

    `None` means the class was never predicted or never occurs. Printing 0%
    there reads as a failure; printing 100% lets an ignored class flatter the
    average. A dash says what is true: nothing was measured.
    """
    return "  -  " if value is None else f"{value:>5.0%}"


def _print_classification(classes: dict[str, Any], extraction: dict[str, Any]) -> None:
    """Precision and recall, for the two things the system predicts.

    Accuracy alone is the wrong summary for both: it cannot see a rare class
    that is never recognised, nor a value invented for a field the document
    does not have.
    """
    if not classes.get("documents"):
        return

    print("\n--- Precision and recall: document type " + "-" * 37)
    print(f"Accuracy             : {_bar(classes['accuracy'])} {classes['accuracy']:.1%}")
    print(f"Macro precision      : {_cell(classes['macro_precision'])}    "
          f"recall {_cell(classes['macro_recall'])}    F1 {_cell(classes['macro_f1'])}")
    print("  macro averages the classes equally, so a rare type that is never")
    print("  recognised cannot be hidden behind the commonest one")
    print(f"\n{'TYPE':16} {'N':>3} {'PRED':>5} {'PREC':>6} {'RECALL':>7} {'F1':>6}")
    for item in classes.get("classes", []):
        print(f"{item['label']:16} {item['support']:>3} {item['predicted']:>5} "
              f"{_cell(item['precision'])} {_cell(item['recall']):>7} {_cell(item['f1']):>6}")

    confused = {
        expected: {got: n for got, n in row.items() if got != expected}
        for expected, row in (classes.get("confusion") or {}).items()
    }
    confused = {k: v for k, v in confused.items() if v}
    if confused:
        print("\nRead as the wrong type:")
        for expected, row in confused.items():
            for got, count in row.items():
                print(f"    {count} x {expected} read as {got}")

    if not extraction.get("values"):
        return
    print("\n--- Precision and recall: extracted values " + "-" * 34)
    print(f"Precision            : {_cell(extraction['precision'])}  "
          f"of {extraction['answered']} values answered, "
          f"{extraction['correct']} were right")
    print(f"Recall               : {_cell(extraction['recall'])}  "
          f"of {extraction['present']} values present, "
          f"{extraction['correct']} were found")
    print(f"F1                   : {_cell(extraction['f1'])}")
    print(f"  wrong    {extraction['wrong']:>3}  answered, and not right")
    print(f"  missing  {extraction['missing']:>3}  there, and not found "
          f"(left for a person - the safe way to be wrong)")
    print(f"  spurious {extraction['spurious']:>3}  not there, and answered anyway "
          f"(invisible to accuracy)")

    weak = [item for item in extraction.get("per_field", []) if (item["f1"] or 1) < 1.0]
    if weak:
        print(f"\n{'WEAKEST FIELDS':24} {'PREC':>6} {'RECALL':>7} {'F1':>6}  "
              f"{'WRONG':>5} {'MISS':>5} {'SPUR':>5}")
        for item in weak[:10]:
            print(f"{item['field']:24} {_cell(item['precision'])} "
                  f"{_cell(item['recall']):>7} {_cell(item['f1']):>6}  "
                  f"{item['wrong']:>5} {item['missing']:>5} {item['spurious']:>5}")


def _print_data_quality(quality: dict[str, Any]) -> None:
    """Is the record fit to hand to another system.

    The one section that needs no answer sheet, which is why it is the one that
    keeps working in production.
    """
    if not quality.get("documents"):
        return

    print("\n--- Data quality: is the record usable " + "-" * 38)
    print(f"Completeness         : {_bar(quality['completeness'])} "
          f"{quality['completeness']:.1%}  "
          f"({quality['fields_filled']}/{quality['fields_defined']} fields filled)")
    print(f"  critical fields    : {_bar(quality['critical_completeness'])} "
          f"{quality['critical_completeness']:.1%}  "
          f"({quality['critical_filled']}/{quality['critical_defined']})")
    print("  an absent field is not automatically a defect - an invoice with no")
    print("  purchase-order reference is a legitimate invoice")
    print(f"Validity             : {_bar(quality['validity'])} {quality['validity']:.1%}  "
          f"({quality['invalid_values']} value(s) not the shape their type promises)")
    for kind, count in (quality.get("invalid_by_kind") or {}).items():
        print(f"    {count:>3} x {kind.replace('_', ' ')}")
    print(f"Grounded in the page : {_bar(quality['grounded_share'])} "
          f"{quality['grounded_share']:.1%}  "
          f"({quality['ungrounded_values']} value(s) with no span behind them)")
    print(f"Self-consistent      : {quality['documents'] - quality['documents_failing_arithmetic']}"
          f"/{quality['documents']} documents whose arithmetic closes")
    print(f"Usable as-is         : {_bar(quality['clean_share'])} {quality['clean_share']:.1%}  "
          f"({quality['clean_documents']}/{quality['documents']} complete, valid and consistent)")

    if quality.get("required_missing_documents"):
        print("\nMissing a required field:")
        for name in quality["required_missing_documents"][:5]:
            print(f"    · {name}")


def print_corpus_report(report: dict[str, Any]) -> None:
    """The four-part report for a `real_samples`-shaped corpus.

    Separate from `print_report` because it has two sections that one cannot
    show - the page was read, and the gate opened correctly - and because the
    corpus report keys documents by filename rather than by catalogue id.
    """
    summary = report.get("summary", {})
    if not summary.get("documents"):
        print("No documents were evaluated.")
        return

    print("\n" + "=" * 78)
    print("CORPUS EVALUATION")
    print("=" * 78)
    _print_setup(report.get("setup") or {})
    _print_integrity(report.get("corpus_integrity") or {})

    print("\n--- Were the values read correctly " + "-" * 42)
    print(f"Documents            : {summary['documents']}  "
          f"({summary['wall_clock_seconds']}s wall clock)")
    print(f"Document type        : {_bar(summary['type_accuracy'])} {summary['type_accuracy']:.1%}")
    print(f"Field accuracy       : {_bar(summary['field_accuracy'])} "
          f"{summary['field_accuracy']:.1%} "
          f"({summary['fields_correct']}/{summary['fields_total']} exact)")
    print(f"  allowing near-miss : {_bar(summary['field_accuracy_with_near'])} "
          f"{summary['field_accuracy_with_near']:.1%}")
    print(f"  not found          : {summary['fields_missing']}")
    print(f"Average confidence   : {summary['average_confidence']:.1%}")

    _print_classification(
        report.get("classification") or {}, report.get("extraction") or {}
    )
    _print_ocr(
        report.get("ocr") or {},
        report.get("ocr_recognised") or {},
        report.get("ocr_from_text_layer") or 0,
    )
    _print_data_quality(report.get("data_quality") or {})
    _print_gatekeeping(report.get("gatekeeping") or {})

    print("\n--- Business safety " + "-" * 57)
    silent = summary["silent_errors"]
    print(f"Silent error rate    : {summary['silent_error_rate']:.1%}  "
          f"[{'PASS' if silent == 0 else 'FAIL'}]")
    print("  (posted automatically while carrying a wrong critical value)")
    for name in summary.get("silent_error_documents", []):
        print(f"    ! {name}")
    print(f"Auto-posted          : {summary['auto_posted']}/{summary['documents']} "
          f"({summary['auto_post_rate']:.0%})")
    print(f"Auto-post precision  : {summary['auto_post_precision']:.1%}")
    print(f"Planted defects      : {summary['planted_defects_caught']}/"
          f"{summary['planted_defects']} caught ({summary['defect_recall']:.0%} recall)")

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
    print(f"{'DOCUMENT':44} {'WANT':>5} {'GOT':>5} {'FIELDS':>7} {'CONF':>6} NOTE")
    for item in report.get("documents", []):
        want = "post" if item["should_post"] else "hold"
        got = "post" if item["posted"] else "hold"
        note = ""
        if want != got:
            note = "MISSED DEFECT" if item["posted"] else "false alarm"
        if item["silent_error"]:
            note += f"  SILENT ERROR: {', '.join(item['wrong_critical'])}"
        if not item["type_correct"]:
            note += f"  type read as {item['classified_as']}"
        print(f"{item['document'][:43]:44} {want:>5} {got:>5} "
              f"{item['fields_correct']}/{item['fields_total']:<5} "
              f"{item['confidence']:>6.0%} {note}")
    print()


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
