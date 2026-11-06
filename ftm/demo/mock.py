"""A scripted stand-in for the LLM, for demos and tests (`--llm mock`). It is NOT a model and its "thoughts" are canned.

Owner: T3 (Akshat).

Unlike the engineering dev responder (which composes arbitrary example questions that mostly come back declined), this one
works through the back-tested menu: it picks the unasked question with the largest possible odds shift per unit eps, at the cheapest
back-tested eps level, and stops after `max_questions` (so the posterior visibly moves). Other schemas fall back to the dev responders.
"""

from __future__ import annotations

import json
import math
import re

from ftm.graph.agents.registry import adapted_dev_responder
from ftm.llm.devtools import dev_responder

_OPTION = re.compile(r"^(\d+)\. \[([^\]]*)\] (.+?) \(eps ([\d.]+)\) -> answer bins: (.*)$", re.M)


def _spread(text: str) -> float:
    xs = [float(x) for x in re.findall(r"x(\d+(?:\.\d+)?)", text) if float(x) > 0]
    return max((abs(math.log(x)) for x in xs), default=0.0)


def demo_responder(menu: list[dict], max_questions: int = 2):
    """`menu`: ftm.llm.backtest_table.accepted_menu(table) (id, program, family, eps, max_shift)."""
    by_id = {m["id"]: m for m in menu}

    def planner_step(messages, kw):
        schema = kw["json_schema"]
        props = schema["properties"]
        user = messages[-1]["content"]
        answers = user.split("## Answers so far", 1)[-1].split("## Output", 1)[0]
        asked = answers.count(", eps ")
        qprop = props["question_id"]
        qids = (qprop["anyOf"][0].get("enum", []) if "anyOf" in qprop else qprop.get("enum", []))
        eps_ok = [e for e in props["eps"]["enum"] if e > 0]
        cands = []
        for q in qids:
            m = by_id.get(q)
            if m is None or m["program"] in answers:
                continue
            levels = [e for e in m["eps"] if e in eps_ok]
            if levels:
                cands.append((m["max_shift"] / min(levels), q, min(levels), m))
        banks = props["bank"]["anyOf"][0].get("enum", []) if "anyOf" in props["bank"] else props["bank"].get("enum", [])
        out = {k: None for k in props}
        if "hypothesis" in props:
            out["hypothesis"] = {"typology": "unsure", "rationale": ""}
        if "belief" in props:
            out["belief"] = 0.5
        out["eps"] = 0.0
        if asked < max_questions and cands and banks:
            _, q, eps, m = max(cands, key=lambda c: (c[0], c[1]))
            shift = m["max_shift"]
            out.update(action="select", question_id=q, bank=banks[0], eps=eps)
            why = (f"Scripted mock. Best back-tested question per unit of privacy budget: {m['family']} (an answer can move the odds by up to "
                   f"x{math.exp(shift):.1f}); asking at the cheapest eps level, {eps:g}.")
        else:
            out["action"] = "stop"
            why = "Scripted mock. Enough questions asked for the demo; keep the rest of the privacy budget."
        if "hypothesis" in props:
            out["hypothesis"]["rationale"] = why
        return json.dumps(out)

    def adapted_choice(messages, kw):
        user = messages[-1]["content"]
        asked = 0 if "(no question asked yet)" in user else len(re.findall(r"^\d+\. .*-> (?:reply|decline|timeout|blocked|over_budget)", user, re.M))
        opts = [(_spread(m.group(5)) / max(float(m.group(4)), 1e-9), int(m.group(1)), m.group(2), float(m.group(4))) for m in _OPTION.finditer(user)]
        if asked >= max_questions or not opts:
            return json.dumps({"thought": "Scripted mock. Enough questions asked for the demo; stop.", "choice": 0})
        score, i, fam, eps = max(opts, key=lambda o: (o[0], -o[1]))
        return json.dumps({"thought": f"Scripted mock. Option {i} ({fam}, eps {eps:g}) has the widest likelihood-ratio spread per "
                                      f"unit eps (up to x{math.exp(score * eps):.1f}); ask it.", "choice": i})

    def respond(messages, kw):
        schema = kw.get("json_schema") or {}
        title = schema.get("title")
        if title == "planner_step" and menu:
            return planner_step(messages, kw)
        if title == "adapted_choice":
            return adapted_choice(messages, kw)
        if title and title.startswith("adapted_"):
            return adapted_dev_responder(messages, kw)
        return dev_responder(messages, kw)

    return respond
