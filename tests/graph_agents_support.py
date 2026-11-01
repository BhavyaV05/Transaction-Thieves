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
                dep = f" $%d" % r.randint(1, i) if i > 1 and r.random() < 0.3 else ""
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
