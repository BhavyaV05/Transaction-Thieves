"""Shared pieces of the two information-seeking ports, Uncertainty of Thoughts (uot.py) and BED-LLM (bed_llm.py).

Owner: T4 (Shreyas).  Group D, secondary.  Both are reimplementations from the papers (no code is copied).

What both papers share, and what lives here:
  - the answer of a question is reduced to a small set of CELLS so a model can state a predicted answer
    distribution over them: bool -> {false, true}; count and set_count -> bins {0, 1-2, 3-5, 6+};
    k-ary -> its options 0..k-1 (k from the question). `answer_cells`, `cell_of`.
  - the expected information gain (EIG) of a question about a target variable: the expected reduction of the
    target's Shannon entropy once the answer cell is seen, I(target; cell), in bits. `expected_info_gain`.
  - the Bayes update of the weights over hypotheses by an observed cell. `bayes_update`.
  - one JSON call with at most one retry on malformed output (`call_json`), the case text, and the single place
    where a chosen question goes through the shared Investigation layer (`ask_question`).

The likelihoods here come from the LLM's OWN predictions (that is what UoT and BED-LLM do), not from the back-tested
LR table that our planner uses. A model-stated probability is floored (`P_FLOOR`) and renormalised so a confident
but wrong prediction cannot send a weight or a belief to exactly 0 or 1.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

from ftm.dsl.parser import arity as dsl_arity
from ftm.dsl.parser import parse as dsl_parse
from ftm.llm.budget import BudgetedClient, BudgetExceeded, PromptTooLong
from ftm.llm.investigation import DuplicateQuestion, Investigation, describe
from ftm.llm.jsonout import JsonOutputError, parse_and_validate
from ftm.llm.untrusted import quote_untrusted
from ftm.planners.llm_dsl import DslAdapter
from ftm.planners.llm_prompt import EPS_LEVELS
from ftm.types import AlertContext, QuestionLogEntry, Reply

BOOL_CELLS = ("false", "true")
COUNT_CELLS = ("0", "1-2", "3-5", "6+")
COUNT_EDGES = (0.5, 2.5, 5.5)  # a noisy count is rounded first: <=0 -> "0", 1-2, 3-5, >=6
P_FLOOR = 1e-3  # a stated probability is floored here before it is used
MAX_CELLS = 8  # most cells any question has (k-ary over MAX_BANKS = 5 banks has 6)
EVIDENCE_CHARS = 3000  # newest evidence rows kept in a prompt (the shared prompt cap is Limits.max_tokens)
_HIDDEN = {"evidence", "partner_banks", "shape", "top_features"}  # top_features: Sharma only (docs/t1_planner_view.md)


# -- answer cells ---------------------------------------------------------------------------------------------
def answer_cells(kind: str, arity: int | None = None) -> tuple[str, ...]:
    """The cells of an answer of this kind, in order."""
    if kind == "bool":
        return BOOL_CELLS
    if kind in ("count", "set_count"):
        return COUNT_CELLS
    if kind == "kary":
        if not arity or arity < 2:
            raise ValueError("a k-ary question needs its arity (>= 2)")
        return tuple(str(i) for i in range(arity))
    raise ValueError(f"unknown answer kind {kind!r}")


def cell_of(kind: str, value, arity: int | None = None) -> str:
    """Map a (noisy) answer to its cell label. Noisy counts and k-ary values are rounded and clipped into range."""
    cells = answer_cells(kind, arity)
    if kind == "bool":
        return cells[1] if bool(value) else cells[0]
    x = float(value)
    if kind == "kary":
        return cells[min(max(int(round(x)), 0), len(cells) - 1)]
    return cells[sum(1 for e in COUNT_EDGES if x >= e)]


def program_arity(program: str) -> int | None:
    """Number of answer values of a k-ary DSL program (None for other kinds or an unparsable program)."""
    try:
        return dsl_arity(dsl_parse(program))
    except Exception:  # noqa: BLE001 - a candidate the DSL cannot read is simply dropped by the caller
        return None


CELLS_DOC = ("Answer cells: a yes/no question has cells [false, true]; a count or set-count question has cells "
             "[0, 1-2, 3-5, 6+] (the noisy count falls in one of them); a k-ary question has one cell per option, "
             "0 .. k-1 (k is in the function description).")


# -- information theory ---------------------------------------------------------------------------------------
def entropy(p: Sequence[float]) -> float:
    """Shannon entropy in bits of a distribution (zeros contribute nothing; no need to sum to exactly 1)."""
    return -sum(x * math.log2(x) for x in p if x > 0)


def normalize(p: Sequence[float], n: int | None = None, floor: float = P_FLOOR) -> list[float]:
    """A stated distribution -> floored and renormalised. Wrong length, negative or all-zero input -> uniform."""
    n = len(p) if n is None else n
    if n <= 0:
        raise ValueError("a distribution needs at least one cell")
    if len(p) != n or any(x < 0 or not math.isfinite(x) for x in p) or sum(p) <= 0:
        return [1.0 / n] * n
    s = sum(p)
    q = [max(x / s, floor) for x in p]
    t = sum(q)
    return [x / t for x in q]


def expected_info_gain(prior: Sequence[float], p_cell_given_h: Sequence[Sequence[float]]) -> float:
    """EIG in bits of one question about a target with values h = 0..H-1.

    prior[h]            P(target = h), summing to 1 (it is normalised here);
    p_cell_given_h[h]   P(answer cell = c | target = h) over the question's cells.
    EIG = H(target) - E_c[ H(target | c) ] = I(target; cell) = sum_h sum_c P(h) P(c|h) log2( P(c|h) / P(c) ).
    A perfectly informative bool at prior 0.5 gives exactly 1 bit; a question whose answer distribution does not
    depend on h gives 0.
    """
    if len(prior) != len(p_cell_given_h):
        raise ValueError("prior and likelihood rows must have the same length")
    z = sum(prior)
    if z <= 0:
        return 0.0
    pri = [x / z for x in prior]
    n_cells = len(p_cell_given_h[0])
    gain = 0.0
    for c in range(n_cells):
        pc = sum(pri[h] * p_cell_given_h[h][c] for h in range(len(pri)))
        if pc <= 0:
            continue
        post = [pri[h] * p_cell_given_h[h][c] / pc for h in range(len(pri))]
        gain += pc * (entropy(pri) - entropy(post))
    return max(gain, 0.0)


def bayes_update(weights: Sequence[float], likelihoods: Sequence[float]) -> list[float]:
    """w_i' = w_i * L_i / sum_j w_j L_j, where L_i = P(observed cell | hypothesis i). If every product is zero
    (cannot happen with floored likelihoods) the weights are returned unchanged."""
    if len(weights) != len(likelihoods):
        raise ValueError("one likelihood per weight")
    prod = [w * x for w, x in zip(weights, likelihoods, strict=True)]
    z = sum(prod)
    if z <= 0:
        s = sum(weights)
        return [w / s for w in weights] if s > 0 else [1.0 / len(weights)] * len(weights)
    return [x / z for x in prod]


def binary_belief_update(belief: float, p_cell_launder: float, p_cell_clean: float) -> float:
    """P(launder | cell) from P(launder) = belief and the two stated likelihoods of the observed cell."""
    post = bayes_update([belief, 1 - belief], [p_cell_launder, p_cell_clean])
    return min(max(post[0], 1e-6), 1 - 1e-6)


# -- calls, text, asking --------------------------------------------------------------------------------------
@dataclass
class JsonResult:
    obj: dict | None = None
    error: str | None = None  # why there is no object
    out_of_budget: bool = False  # no call left or the prompt is over the cap: the caller should stop, not retry


def call_json(budget: BudgetedClient, messages: list[dict], schema: dict, *, max_tokens: int, temperature: float,
              seed: int, extra: dict | None, check=None) -> JsonResult:
    """One JSON call, retried at most once when the reply is malformed (bad JSON, schema break, or `check` returns a
    message). Both attempts are real, capped calls. After the second failure the step is skipped by the caller."""
    msgs = list(messages)
    error = "no reply"
    for attempt in range(2):
        try:
            resp = budget.chat(msgs, max_tokens=max_tokens, temperature=temperature, json_schema=schema, seed=seed,
                               extra=extra)  # fmt: skip
        except (BudgetExceeded, PromptTooLong) as e:
            return JsonResult(error=str(e), out_of_budget=True)
        try:
            obj = parse_and_validate(resp.text, schema)
            problem = check(obj) if check else None
        except JsonOutputError as e:
            obj, problem = None, str(e)
        if problem is None:
            return JsonResult(obj=obj)
        error = problem
        if attempt == 0:
            msgs = [*msgs, {"role": "assistant", "content": resp.text[:1500]},
                    {"role": "user", "content": f"That reply was not usable ({problem[:300]}). Reply again with ONE "
                     "JSON object in the requested format and nothing else."}]  # fmt: skip
    return JsonResult(error=error)


def case_block(alert: AlertContext, evidence_chars: int = EVIDENCE_CHARS) -> str:
    """Aliased alert summary plus the newest evidence rows within a character budget."""
    summary = {k: v for k, v in alert.summary.items() if k not in _HIDDEN}
    rows = [ln for ln in str(alert.summary.get("evidence", "")).splitlines() if ln.strip()]
    kept, used = [], 0
    for ln in reversed(rows):
        if used + len(ln) > evidence_chars and kept:
            break
        kept.append(ln)
        used += len(ln) + 1
    ev = "\n".join(reversed(kept)) or "(none)"
    return (f"Alert: {summary}\nBank's one-bank model score (probability of laundering): {alert.p0:.3f}\n"
            f"Evidence (home bank's view, time-ordered, aliased):\n{ev}")  # fmt: skip


def history_block(session: Investigation) -> str:
    """One entry per question asked, answers described with the shared safe wrapper for partner notes."""
    if not session.log:
        return "(no question asked yet)"
    return "\n".join(f"{i}. {e.program} (bank {e.bank}, eps {e.eps:g}) -> {describe(e)}"
                     for i, e in enumerate(session.log, start=1))  # fmt: skip


def safe(text: str, max_chars: int = 240) -> str:
    """Model-written text that is shown to the model again (hypotheses): escaped like a partner note."""
    return quote_untrusted(text, max_chars)


@dataclass
class Asked:
    """What `ask_question` learnt: the log entry, the cell of a real answer (else None), the cells of the question."""

    entry: QuestionLogEntry
    kind: str
    cells: tuple[str, ...]
    cell: str | None


def ask_question(session: Investigation, dsl: DslAdapter, banks: list, bank, eps, program: str, step: int) -> tuple[Asked | None, str | None]:
    """Validate and ask one question through the shared Investigation layer (gate, accountant, log).

    Returns (Asked, None) when the question went out, or (None, reason) when it was refused before any bank call
    (invalid program, bank not offered, eps not allowed or over what is left, duplicate)."""
    if str(bank) not in {str(b) for b in banks}:
        return None, f"bank {bank} is not one of the partner banks"
    if eps not in EPS_LEVELS or eps > session.eps_left + 1e-9:
        return None, f"eps {eps} is not allowed now (levels {list(EPS_LEVELS)}, eps left {session.eps_left:g})"
    check = dsl.validate(program)
    if not check.ok:
        return None, f"invalid question: {check.error}"
    if session.already_asked(check.canonical, bank):
        return None, f"already asked bank {bank}: {check.canonical}"
    k = program_arity(check.canonical) if check.kind == "kary" else None
    try:
        cells = answer_cells(check.kind, k)
    except ValueError as e:
        return None, str(e)
    try:
        entry = session.ask(bank, check, float(eps), step)
    except DuplicateQuestion as e:  # pragma: no cover - already_asked above makes this unreachable
        return None, str(e)
    cell = None
    if isinstance(entry.response, Reply):
        try:
            cell = cell_of(check.kind, entry.response.value, k)
        except (TypeError, ValueError):
            cell = None
    return Asked(entry, check.kind, cells, cell), None


def candidate_cells(dsl: DslAdapter, program: str) -> tuple[str | None, tuple[str, ...] | None, str | None]:
    """(canonical, cells, error) of a proposed question, so predictions can be checked before it is asked."""
    check = dsl.validate(program)
    if not check.ok:
        return None, None, f"invalid question: {check.error}"
    k = program_arity(check.canonical) if check.kind == "kary" else None
    try:
        return check.canonical, answer_cells(check.kind, k), None
    except ValueError as e:
        return None, None, str(e)


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def label_marginal(weights: Sequence[float], labels: Sequence[str], preds: Sequence[Sequence[float]],
                   targets: tuple[str, str] = ("launder", "clean")) -> tuple[list[float], list[list[float]]]:
    """Collapse a weighted set of hypotheses onto the label variable: prior[l] = total weight of the hypotheses
    with label l; row[l] = their weight-averaged predicted answer distribution (uniform when the label has no
    weight). The result feeds `expected_info_gain` directly: it is the EIG about the LABEL, not about which
    hypothesis is true (two launder scenarios the question cannot tell apart add nothing)."""
    n_cells = len(preds[0])
    prior, rows = [], []
    for t in targets:
        idx = [i for i, lab in enumerate(labels) if lab == t and weights[i] > 0]
        w = sum(weights[i] for i in idx)
        prior.append(w)
        rows.append([sum(weights[i] * preds[i][c] for i in idx) / w for c in range(n_cells)] if w > 0
                    else [1.0 / n_cells] * n_cells)  # fmt: skip
    z = sum(prior)
    return ([p / z for p in prior] if z > 0 else [0.5, 0.5]), rows
