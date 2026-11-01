"""DSPy ReAct as a StateGraph, ported mode: dspy.ReAct.forward (v3.4) unrolled into nodes, so the loop is visible and the
MIPROv2-optimised predictors (`program.react`, `program.extract`) are the ones that run. Parity with
ftm/external/dspy_react.py is tested on the mock LLM.

    init -> step* (one react predictor call + one tool call) -> extract -> verdict
Every dspy LM call still goes through OUR BudgetedClient (client_lm). dspy is imported lazily (CI does not install it).

Owner: T3 (Akshat).
"""

from __future__ import annotations

import warnings

from ftm.external.dspy_react import _CTX, build_program, client_lm
from ftm.external.dspy_react import DspyReact as _Legacy
from ftm.external.text_agent import ToolExecutor, case_text, tools_doc
from ftm.harness.interface import ToolBox
from ftm.llm.budget import BudgetedClient, BudgetExceeded, PromptTooLong
from ftm.llm.investigation import Investigation
from ftm.types import AlertContext, Limits, Verdict

from ._base import END, START, new_graph, run_graph


class _Ctx:
    """The dspy settings context of one node: our LM, the legacy warning filter, and the executor of this alert."""

    def __init__(self, s):
        import dspy

        _CTX.executor, _CTX.step = s["ex"], s["scratch"]["tool_step"]
        self.cm = warnings.catch_warnings()
        self.lm = dspy.context(lm=s["scratch"]["lm"])

    def __enter__(self):
        self.cm.__enter__()
        warnings.filterwarnings("ignore", message=".*BaseLM.forward.*deprecated.*")
        self.lm.__enter__()

    def __exit__(self, *a):
        self.lm.__exit__(*a)
        return self.cm.__exit__(*a)


def _swallow(e: Exception) -> bool:
    """The legacy handler: an unparseable model output or a capped / over-long call is a failed alert (no probability);
    anything else is a bug and propagates."""
    from dspy.utils.exceptions import AdapterParseError

    return isinstance(e, AdapterParseError) or isinstance(e.__cause__ or e.__context__, (BudgetExceeded, PromptTooLong))


class DspyReact(_Legacy):
    def graph(self):
        if getattr(self, "_graph", None) is None:
            g = new_graph()
            for name, fn in (("init", self._init), ("step", self._step), ("extract", self._extract), ("verdict", self._verdict)):
                g.add_node(name, fn)
            g.add_edge(START, "init")
            g.add_conditional_edges("init", self._next, ["step", "extract", "verdict"])
            g.add_conditional_edges("step", self._next, ["step", "extract", "verdict"])
            g.add_edge("extract", "verdict")
            g.add_edge("verdict", END)
            self._graph = g.compile()
        return self._graph

    @staticmethod
    def _next(s) -> str:
        sc = s["scratch"]
        if s.get("halt"):
            return "verdict"
        return "step" if sc["idx"] < sc["program"].max_iters and not sc["ended"] else "extract"

    def _init(self, s) -> dict:
        alert, limits = s["alert"], s["limits"]
        program = self.program or build_program(self.max_iters or max(limits.max_llm_calls - 1, 1))
        session = Investigation(alert, s["tools"], limits, fusion=self.fusion)
        budget = BudgetedClient(self.llm, limits, tag=f"dspy_react:{alert.alert_id}")
        banks = list(alert.summary.get("partner_banks", []))
        ex = ToolExecutor(alert, s["tools"], session, self.dsl, banks)
        _CTX.executor, _CTX.step = ex, 0
        inputs = {"case": case_text(alert, str(alert.summary.get("evidence", ""))), "tools_help": tools_doc(self.dsl, banks, session.eps_left)}
        return {"session": session, "budget": budget, "ex": ex, "halt": False,
                "scratch": {"program": program, "lm": client_lm(budget, self.extra), "inputs": inputs, "trajectory": {}, "idx": 0,
                            "ended": False, "p": None, "tool_step": 0}}  # fmt: skip

    def _step(self, s) -> dict:
        from dspy.utils.exceptions import ContextWindowExceededError

        sc = s["scratch"]
        program, trajectory, idx = sc["program"], dict(sc["trajectory"]), sc["idx"]
        try:
            with _Ctx(s):
                try:
                    pred = program._call_with_potential_trajectory_truncation(program.react, trajectory, **sc["inputs"])
                except (ContextWindowExceededError, ValueError):  # dspy: "Ending the trajectory"
                    return {"scratch": {**sc, "ended": True}}
                trajectory[f"thought_{idx}"] = pred.next_thought
                trajectory[f"tool_name_{idx}"] = pred.next_tool_name
                trajectory[f"tool_args_{idx}"] = pred.next_tool_args
                try:
                    trajectory[f"observation_{idx}"] = program.tools[pred.next_tool_name](**pred.next_tool_args)
                except Exception as err:  # noqa: BLE001 - dspy turns a tool error into an observation
                    from dspy.utils.exceptions import format_error_for_lm

                    trajectory[f"observation_{idx}"] = f"Execution error in {pred.next_tool_name}: {format_error_for_lm(err, traceback_frames=5)}"
        except (BudgetExceeded, PromptTooLong, ValueError, TypeError):
            return {"halt": True}
        except Exception as e:  # noqa: BLE001
            if not _swallow(e):
                raise
            return {"halt": True}
        return {"scratch": {**sc, "trajectory": trajectory, "idx": idx + 1, "ended": pred.next_tool_name == "finish",
                            "tool_step": getattr(_CTX, "step", sc["tool_step"])}}  # fmt: skip

    def _extract(self, s) -> dict:
        sc = s["scratch"]
        program = sc["program"]
        try:
            with _Ctx({**s, "scratch": {**sc, "tool_step": getattr(_CTX, "step", sc["tool_step"])}}):
                out = program._call_with_potential_trajectory_truncation(program.extract, sc["trajectory"], **sc["inputs"])
            p = float(out.probability)
            p = p if 0.0 <= p <= 1.0 else None
        except (BudgetExceeded, PromptTooLong, ValueError, TypeError):
            p = None
        except Exception as e:  # noqa: BLE001
            if not _swallow(e):
                raise
            p = None
        return {"scratch": {**sc, "p": p}}

    def _verdict(self, s) -> dict:
        self.last_trace = list(s["ex"].log)
        return {"verdict": s["session"].verdict(s["scratch"]["p"], s["budget"].usage)}

    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        return run_graph(self.graph(), {"alert": alert, "tools": tools, "limits": limits})["verdict"]

    investigate_legacy = _Legacy.investigate
