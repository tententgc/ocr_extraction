"""What a run costs, and the loop that stops an agent going in circles.

The accuracy harness has always answered "is it right". Nothing answered "what
did it take" - so an agent could spend four identical tool calls reaching a
conclusion, or reach none at all, and no number anywhere would move.
"""

from __future__ import annotations

from docuflow.agents.agent_loop import MAX_REPEATED_CALLS, run_agent
from docuflow.agents.tools import Tool
from docuflow.evaluation.cost import agent_activity, stage_timings, summarize_cost
from docuflow.llm.base import AssistantTurn, LlmProvider, ToolCall
from docuflow.models import RuleFinding, TraceStep


class Looping(LlmProvider):
    """Asks the same question over and over, then answers if allowed to."""

    name = "looping"
    supports_tools = True

    def __init__(self, arguments: dict | None = None) -> None:
        self.arguments = arguments if arguments is not None else {"subtotal": 100, "total": 107}
        self.turns = 0

    def complete(self, messages, max_tokens=2048, temperature=0.0) -> str:
        return '{"verdict":"unsure"}'

    def complete_with_tools(self, messages, tools, max_tokens=2048, temperature=0.0) -> AssistantTurn:
        self.turns += 1
        if not tools:                      # the loop's final "answer now" turn
            return AssistantTurn(content='{"verdict":"fail","reason":"done"}')
        return AssistantTurn(tool_calls=(
            ToolCall(id=str(self.turns), name="counted", arguments=dict(self.arguments)),
        ))


def counting_tool() -> tuple[Tool, list[int]]:
    runs: list[int] = []

    def run(args):
        runs.append(1)
        return {"answer": 42}

    return Tool(name="counted", description="counts how often it really ran",
                parameters={"type": "object", "properties": {}}, run=run), runs


class TestAnAgentGoingInCircles:
    def test_an_identical_call_is_answered_from_what_it_was_already_told(self):
        tool, runs = counting_tool()
        result = run_agent(Looping(), "system", "user", [tool], max_iterations=6)

        assert len(runs) == 1, "the tool itself is only ever run once"
        assert result.repeated_calls >= 1
        repeat = result.calls[1].result
        assert "already made this exact call" in repeat["note"]
        assert repeat["answer"] == 42, "and it is told the same answer, not an error"

    def test_it_is_stopped_rather_than_left_to_spend_its_budget(self):
        tool, _ = counting_tool()
        llm = Looping()
        result = run_agent(llm, "system", "user", [tool], max_iterations=6)

        # Two repeats end the tool loop; without this it burned all six turns.
        assert result.repeated_calls == MAX_REPEATED_CALLS
        assert llm.turns <= MAX_REPEATED_CALLS + 2
        assert result.payload == {"verdict": "fail", "reason": "done"}

    def test_different_arguments_are_not_a_repeat(self):
        """Asking about a second purchase order is research, not a loop."""
        tool, runs = counting_tool()

        class Varying(Looping):
            def complete_with_tools(self, messages, tools, max_tokens=2048, temperature=0.0):
                self.turns += 1
                if not tools or self.turns > 3:
                    return AssistantTurn(content='{"verdict":"pass"}')
                return AssistantTurn(tool_calls=(
                    ToolCall(id=str(self.turns), name="counted",
                             arguments={"po": f"PO-{self.turns}"}),
                ))

        result = run_agent(Varying(), "system", "user", [tool], max_iterations=6)
        assert result.repeated_calls == 0
        assert len(runs) == 3


class TestCostIsReadOffTheTrace:
    def test_stage_timings_come_from_the_run_that_already_recorded_them(self, make_run):
        run = make_run({"total_amount": 1.0}).evolve(trace=(
            TraceStep(step="ocr", duration_ms=4000),
            TraceStep(step="extract", duration_ms=9000),
            TraceStep(step="route", duration_ms=0),
        ))
        assert stage_timings(run) == {"ocr": 4000, "extract": 9000}

    def test_agent_rule_outcomes_are_counted_by_verdict(self, make_run):
        run = make_run({"total_amount": 1.0}).evolve(findings=(
            RuleFinding(rule_id="a", passed=True, severity="info", message="",
                        detail={"kind": "agent", "verdict": "pass",
                                "tools_used": ["lookup_purchase_order"], "repeated_calls": 0}),
            RuleFinding(rule_id="b", passed=False, severity="warning", message="",
                        detail={"kind": "agent", "verdict": "unsure",
                                "tools_used": ["read_document", "read_document"],
                                "repeated_calls": 1}),
            RuleFinding(rule_id="c", passed=False, severity="blocking", message="",
                        detail={"custom": True}),        # an expression rule
        ))
        activity = agent_activity(run)
        assert activity["agent_rules_run"] == 2
        assert activity["agent_rule_verdicts"] == {"pass": 1, "unsure": 1}
        assert activity["agent_rule_tool_calls"] == 3
        assert activity["repeated_tool_calls"] == 1

    def test_the_summary_reports_the_rate_of_undecided_checks(self):
        """A number that climbs means a plan stopped working."""
        costs = [
            {"total_ms": 1000, "stage_ms": {"ocr": 1000}, "agent_rules_run": 1,
             "agent_rule_verdicts": {"pass": 1}, "agent_rule_tool_calls": 1,
             "repeated_tool_calls": 0, "reconciler_ran": False, "reconciler_tool_calls": 0},
            {"total_ms": 3000, "stage_ms": {"ocr": 1000, "extract": 2000}, "agent_rules_run": 1,
             "agent_rule_verdicts": {"unsure": 1}, "agent_rule_tool_calls": 2,
             "repeated_tool_calls": 2, "reconciler_ran": True, "reconciler_tool_calls": 3},
        ]
        summary = summarize_cost(costs)
        assert summary["undecided_rate"] == 0.5
        assert summary["repeated_tool_calls"] == 2
        assert summary["stages"]["ocr"]["share"] == 0.5
        assert summary["slowest_ms"] == 3000
        assert summary["reconciler_runs"] == 1

    def test_an_empty_corpus_is_not_a_division_by_zero(self):
        assert summarize_cost([]) == {"documents": 0}
