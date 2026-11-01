"""Shared fixtures for the ported-graph parity tests: a deterministic fuzzing responder per agent family, the alert, and `same`."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random

from ftm.llm.devtools import DevRandomToolBox
from ftm.llm.mock import MockLLM
from ftm.types import AlertContext, Limits

BOOL_QS = ["passed_on(ALERT.receiver, window=after(24), min_frac=0.8)",
           "passed_on(ALERT.receiver, window=after(48), min_frac=0.5)",
           "at_least(n_counterparties(ALERT.receiver, dir=in, window=before(48)), k=5)"]
COUNT_QS = ["n_counterparties(ALERT.receiver, dir=in, window=before(72))",
            "n_counterparties(ALERT.receiver, dir=in, window=before(24))"]
ALL_QS = BOOL_QS + COUNT_QS + ["bogus(ALERT.receiver)", "passed_on(A0000001, window=after(24), min_frac=0.8)"]


def alert(p0: float = 0.4, banks=(220, 1588)) -> AlertContext:
    ev = "\n".join(f"2022-09-0{d} 10:00 | A000000{d} -> A0000002 | {1000 * d:,.2f} US Dollar | Wire | bank 7 -> bank 1588"
                   for d in range(1, 5))  # fmt: skip
    s = {"sender": "A0000001", "receiver": "A0000002", "sender_bank": 7, "receiver_bank": 1588, "amount": 9850.0,
         "currency": "US Dollar", "format": "Wire", "time": "2022-09-05 10:00", "evidence": ev, "partner_banks": list(banks)}  # fmt: skip
    return AlertContext("V0001", "7", "2022-09-05T10:00:00", "val", p0, s)


def rng_for(messages, kw, salt) -> random.Random:
    h = hashlib.sha256(json.dumps([[m["content"] for m in messages], kw.get("seed"), kw.get("temperature"), salt]).encode()).hexdigest()
    return random.Random(int(h[:12], 16))


def action(r: random.Random, banks=(220, 1588)) -> str:
    u = r.random()
    if u < 0.55:
        return f"ask_bank[{r.choice(list(banks) + [999])} | {r.choice([0.25, 0.5, 1.0, 3.0])} | {r.choice(ALL_QS)}]"
    if u < 0.65:
        return "lookup_rules[which questions are permitted]"
    if u < 0.85:
        return f"finish[{r.choice(['0.1', '0.55', '90%', '2', 'x'])}]"
    return "no action here"


def text_responder(salt: int):
    """Responds to every Group D text-loop prompt (ReAct/Reflexion/ReWOO/LATS/LLMCompiler)."""

    def f(messages: list[dict], kw: dict) -> str:
        r = rng_for(messages, kw, salt)
        sys = messages[0]["content"]
        if sys.startswith("You are checking"):
            return r.choice(["CONFIDENT fine", "RETRY more", "retry", "maybe"])
        if sys.startswith("You are an advanced reasoning"):
            return "Lesson: ask about\npassing on. " * r.randint(1, 2)
        if sys.startswith("You are scoring"):
            n = len(messages[-1]["content"].split("Candidate ")) - 1
            return "\n".join(f"Candidate {i + 1}: {r.randint(1, 10)}" for i in range(n))
        if sys.startswith("Resolve the following"):  # ReWOO solver
            return r.choice(["finish[0.7]", "I think 0.2", "0.35", "finish[1.4]", "finish[0.05] ok"])
        if sys.startswith("For the following anti-money"):  # ReWOO planner
            return "\n".join(f"Plan: step {i}\n#E{i} = " + r.choice([f"ask_bank[{r.choice([220, 1588])} | 0.5 | {r.choice(ALL_QS)}]",
                                                                    "lookup_rules[rules]", "ask_bank[#E1 | 0.5 | x]"])
                             for i in range(1, r.randint(1, 5))) if r.random() < 0.9 else "no plan"
        if sys.startswith("Given an anti-money"):  # LLMCompiler planner
            lines = []
            for i in range(1, r.randint(1, 5)):
                dep = " $%d" % r.randint(1, i) if i > 1 and r.random() < 0.3 else ""
                lines.append(f"{i}. " + r.choice([f"ask_bank[{r.choice([220, 1588])} | 0.5 | {r.choice(ALL_QS)}]", f"lookup_rules[r{dep}]"]))
            return "\n".join(lines + [f"{len(lines) + 1}. join()"])
        if sys.startswith("You see an anti-money"):  # LLMCompiler joiner
            return r.choice(["Action: replan\nmissing", "Action: finish[0.45]", "finish[0.2]", "Action: replan"])
        return f"Thought {r.randint(1, 9)}: hmm\nAction: {action(r)}"

    return f


def toolbox(seed: int) -> DevRandomToolBox:
    return DevRandomToolBox(seed=seed, p_decline=0.15, p_timeout=0.05)


def snapshot(v, agent) -> dict:
    d = dataclasses.asdict(v)
    d["_trace"] = json.loads(json.dumps(agent.last_trace, default=str))
    return d


def same(make_old, make_new, responder_factory, *, seeds=range(25), limits=None, extra_attrs=(), alert_fn=alert):
    """Run the legacy loop and the graph on identical mock LLM + toolbox for every seed; they must agree exactly."""
    lim = limits or Limits()
    nonempty = 0
    for seed in seeds:
        out = []
        for new in (False, True):
            seen: list = []
            inner = responder_factory(seed)

            def rec(messages, kw, inner=inner, seen=seen):  # the old loops mutate `messages` in place: copy at call time
                seen.append(json.loads(json.dumps([messages, {k: v for k, v in kw.items() if k != "extra"}], default=str)))
                return inner(messages, kw)

            llm = MockLLM(responder=rec)
            agent = (make_new if new else make_old)(llm)
            run = agent.investigate if new else getattr(agent, "investigate_legacy", agent.investigate)
            v = run(alert_fn(), toolbox(seed), lim)
            snap = snapshot(v, agent)
            for a in extra_attrs:
                snap[a] = json.loads(json.dumps(getattr(agent, a, None), default=str))
            snap["_requests"] = seen
            out.append(snap)
        assert out[0] == out[1], f"seed {seed} differs"
        nonempty += bool(out[0]["questions_log"])
    return nonempty


# -- JSON-schema agents (debate, UoT, BED-LLM, MediQ) ------------------------------------------------------------
def _dist(r, n):
    if r.random() < 0.1:
        n += r.choice([-1, 1])  # wrong length: must be rejected the same way
    return [round(r.random(), 2) for _ in range(max(n, 0))]


def json_responder(salt: int):
    import re

    def f(messages: list[dict], kw: dict) -> str:
        r = rng_for(messages, kw, salt)
        schema = kw.get("json_schema") or {}
        title = schema.get("title")
        user = messages[-1]["content"]
        if r.random() < 0.06:
            return "not json"
        if title == "debate_turn":
            q = None
            alt = schema["properties"]["question"].get("anyOf")
            if alt and r.random() < 0.6:
                p = alt[1]["properties"]
                q = {"bank": r.choice(p["bank"]["enum"]), "eps": r.choice(p["eps"]["enum"]),
                     "question": r.choice(BOOL_QS + COUNT_QS)}  # fmt: skip
            return json.dumps({"probability": round(r.random(), 2), "rationale": f"because {r.randint(0, 3)}", "question": q})
        if title == "uot_step":
            props = schema["properties"]["candidates"]["items"]["properties"]
            cands = []
            for _ in range(r.randint(0, 3)):
                q = r.choice(ALL_QS[:5])
                n = 2 if q in BOOL_QS else 4
                cands.append({"bank": r.choice(props["bank"]["enum"]), "eps": r.choice(props["eps"]["enum"]), "question": q,
                              "p_launder": _dist(r, n), "p_clean": _dist(r, n)})  # fmt: skip
            return json.dumps({"candidates": cands})
        if title == "bed_hypotheses":
            return json.dumps({"hypotheses": [{"label": r.choice(["launder", "clean"]), "description": f"scenario {i}"}
                                              for i in range(r.randint(0, 4))]})  # fmt: skip
        if title == "bed_questions":
            n_h = len(re.findall(r"^H\d+ \[", user, re.MULTILINE))
            n_p = len(re.findall(r"^P\d+:", user, re.MULTILINE))
            props = schema["properties"]["candidates"]["items"]["properties"]
            past = [[_dist(r, r.choice([2, 4])) for _ in range(n_h)] for _ in range(n_p)]
            cands = []
            for _ in range(r.randint(0, 3)):
                q = r.choice(ALL_QS[:5])
                n = 2 if q in BOOL_QS else 4
                cands.append({"bank": r.choice(props["bank"]["enum"]), "eps": r.choice(props["eps"]["enum"]), "question": q,
                              "predictions": [_dist(r, n) for _ in range(n_h)]})  # fmt: skip
            return json.dumps({"past": past, "candidates": cands})
        if title == "mediq_state":
            return json.dumps({"probability": round(r.random(), 2), "confidence": round(r.choice([0.2, 0.5, 0.7, 0.9]), 2)})
        if title == "mediq_question":
            props = schema["properties"]
            return json.dumps({"bank": r.choice(props["bank"]["enum"]), "eps": r.choice(props["eps"]["enum"]), "question": r.choice(ALL_QS)})
        return "{}"

    return f


def dspy_step(tool, args):
    return (f"[[ ## next_thought ## ]]\nt\n\n[[ ## next_tool_name ## ]]\n{tool}\n\n"
            f"[[ ## next_tool_args ## ]]\n{args}\n\n[[ ## completed ## ]]")  # fmt: skip


def dspy_responder(salt: int):
    def f(messages: list[dict], kw: dict) -> str:
        r = rng_for(messages, kw, salt)
        last = str(messages[-1].get("content"))
        if "next_tool_name" in " ".join(str(m.get("content")) for m in messages[:1]) or "next_tool_name" in last:
            u = r.random()
            if u < 0.5:
                return dspy_step("ask_bank", json.dumps({"bank": r.choice([220, 1588, 5]), "eps": r.choice([0.25, 0.5, 3.0]),
                                                         "question": r.choice(ALL_QS)}))  # fmt: skip
            if u < 0.6:
                return dspy_step("lookup_rules", '{"query": "x"}')
            if u < 0.8:
                return dspy_step("finish", "{}")
            if u < 0.9:
                return dspy_step("nonexistent_tool", "{}")
            return "garbage"
        p = r.choice(["0.62", "0.2", "1.7", "high"])
        return f"[[ ## reasoning ## ]]\nr\n\n[[ ## probability ## ]]\n{p}\n\n[[ ## completed ## ]]"

    return f


# -- adapted arms: a minimal fake pipeline driver (the real one lives in ftm.graph.pipeline) -------------------------
def menu_items():
    from ftm.graph.policy import MenuItem

    return [MenuItem("q1", BOOL_QS[0], 0.5, "1588", "passed_on", [("false", 0.4), ("true", 2.5)]),
            MenuItem("q1", BOOL_QS[0], 1.0, "1588", "passed_on", [("false", 0.2), ("true", 4.0)]),
            MenuItem("q2", BOOL_QS[2], 0.5, "1588", "at_least", [("false", 0.7), ("true", 1.6)]),
            MenuItem("q3", COUNT_QS[0], 1.0, "220", "n_counterparties", [("b0", 0.5), ("b1", 1.0), ("b2", 2.2)]),
            MenuItem("q4", COUNT_QS[1], 0.5, "220", "n_counterparties", [("b0", 0.8), ("b1", 1.1), ("b2", 1.3)])]


def drive(policy, llm, *, calls=20, eps=2.0, seed=0, max_steps=12):
    """Run `policy` against a fake pipeline: answers drawn at random, LRs from the menu bins. Checks the contract on every action."""
    import math

    from ftm.graph.policy import Ask, PlannerView, Stop
    from ftm.llm.budget import BudgetedClient

    budget = BudgetedClient(llm, Limits(max_llm_calls=calls), tag="fake")
    rng = random.Random(seed)
    menu, answers, scratch = menu_items(), [], {}
    logit = math.log(0.3 / 0.7)
    stop, failures, step = None, 0, 0
    asked_log = []
    while stop is None and step < max_steps:
        step += 1
        view = PlannerView({"home_bank": "7"}, 0.3, 1 / (1 + math.exp(-logit)), 0.5, "t1 A -> B 100\nt2 B -> C 99", [{"rule_id": "R1", "text": "rule"}], [],
                           menu, list(answers), eps, budget.calls_left, step, False)  # fmt: skip
        actions = policy.propose(view, budget, scratch)
        asks = [a for a in actions if isinstance(a, Ask)]
        stops = [a for a in actions if isinstance(a, Stop)]
        for a in asks:
            item = next((m for m in menu if m.canonical == a.canonical and abs(m.eps - a.eps) < 1e-9), None)
            assert item is not None, f"{policy.name}: ask outside the menu {a}"
            assert a.eps <= eps + 1e-9, f"{policy.name}: ask over the eps left"
            assert a.canonical not in [x["program"] for x in answers], f"{policy.name}: repeated question"
        if not asks:
            if stops:
                stop = stops[0].reason or "policy_stop"
            else:
                failures += 1
                if failures >= 3:
                    stop = "failures"
            continue
        failures = 0
        wave_eps = sum(a.eps for a in asks)
        for a in asks:
            if sum(x["eps"] for x in answers) + a.eps > 2.0 + 1e-9:
                continue
            item = next(m for m in menu if m.canonical == a.canonical and abs(m.eps - a.eps) < 1e-9)
            label, lr = rng.choice(item.lr_bins)
            answers.append({"program": a.canonical, "bank": item.bank, "eps": a.eps, "outcome": "reply" if rng.random() > 0.15 else "decline",
                            "step": step, "value": label, "lr": lr})  # fmt: skip
            logit += math.log(lr)
            asked_log.append(a.canonical)
        eps = max(eps - wave_eps, 0.0)
        if eps < 0.25:
            stop = "eps_used"
        if budget.calls_left <= 0:
            stop = "calls_used"
        if stops:
            stop = stops[0].reason
    return asked_log, stop, scratch, budget


def adapted_responder(salt: int):
    def f(messages: list[dict], kw: dict) -> str:
        r = rng_for(messages, kw, salt)
        schema = kw.get("json_schema") or {}
        t = schema.get("title")
        if r.random() < 0.05:
            return "not json"
        props = schema.get("properties", {})
        if t in ("adapted_choice",):
            return json.dumps({"thought": "hm", "choice": r.randint(0, props["choice"]["maximum"])})
        if t == "adapted_eval":
            return json.dumps({"verdict": r.choice(["CONFIDENT", "RETRY"]), "reason": "r"})
        if t == "adapted_reflect":
            return json.dumps({"reflection": "ask something with more signal"})
        if t == "adapted_plan":
            n = props["plan"]["items"]["properties"]["choice"]["maximum"]
            return json.dumps({"plan": [{"why": "w", "choice": r.randint(1, n)} for _ in range(r.randint(0, 5))]})
        if t == "adapted_scores":
            return json.dumps({"scores": [r.randint(1, 10) for _ in range(r.randint(0, 3))]})
        if t == "adapted_dag":
            n = props["tasks"]["items"]["properties"]["choice"]["maximum"]
            return json.dumps({"tasks": [{"id": i, "choice": r.randint(1, n), "deps": [j for j in range(1, i) if r.random() < 0.3]}
                                         for i in range(1, r.randint(1, 5))]})  # fmt: skip
        if t == "adapted_join":
            return json.dumps({"decision": r.choice(["FINISH", "REPLAN"]), "note": "n"})
        if t == "adapted_uot":
            n = props["candidates"]["items"]["properties"]["choice"]["maximum"]
            return json.dumps({"candidates": [{"choice": r.randint(1, n), "p_launder": _dist(r, 2 + r.randint(0, 1))} for _ in range(r.randint(0, 3))]})
        if t == "adapted_bed_hyp":
            return json.dumps({"hypotheses": [{"label": r.choice(["launder", "clean"]), "description": "d"} for _ in range(r.randint(0, 4))]})
        if t == "adapted_bed_q":
            import re

            user = messages[-1]["content"]
            n_h = len(re.findall(r"^H\d+ \[", user, re.MULTILINE))
            n_p = len(re.findall(r"^P\d+:", user, re.MULTILINE))
            n = props["candidates"]["items"]["properties"]["choice"]["maximum"]
            return json.dumps({"past": [[_dist(r, 2) for _ in range(n_h)] for _ in range(n_p)],
                               "candidates": [{"choice": r.randint(1, n), "predictions": [_dist(r, r.choice([2, 3])) for _ in range(n_h)]}
                                              for _ in range(r.randint(0, 3))]})  # fmt: skip
        if t == "adapted_mediq_state":
            return json.dumps({"probability": r.random(), "confidence": r.choice([0.3, 0.6, 0.9])})
        return "{}"

    return f
