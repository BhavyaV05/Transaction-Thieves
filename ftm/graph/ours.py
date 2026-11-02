"""Our System 2 planner (C10) as a `PlannerPolicy` of the system graph (ftm.graph.pipeline). Owner: T3 (Akshat).

This is the step logic that used to be the while loop of `LLMPlanner.investigate` (plan.md 6.11, tracks.md 5.4), cut at the
graph's seams so its behaviour on the same LLM answers is unchanged (tests/test_graph_ours_parity.py runs the frozen old loop
beside it):

  propose  (graph node `plan`)  build the prompt (alert summary, aliased evidence, listed questions, strategies, rules, answers so
           far, eps and calls left, the back-tested posterior) and the per-state JSON schema (llm_prompt.py); one LLM call; parse; on a
           bad JSON / bad DSL / repeated or refused question / unroutable subject hand feedback to the next step and propose nothing
           (the pipeline counts it as a failed step); select -> listed question, compose -> new DSL question, stop -> Stop;
  observe  (after node `act`)   the answers: stats, the retry feedback for a refusal or a duplicate, the question export;
  finish   (node `decide`)      the verdict probability: the formula's posterior, except the LLM's own `belief` in the
                                no_backtest / llm_decides ablations (H4 / H7). LLM text never enters fusion.

Batched option (E13, H8): `batch_schedule=(2, 1)` makes step i propose up to schedule[i] questions in one JSON object; they go out
as ONE wave (Investigation.ask_wave: the clock advances by the slowest answer), then the next step. Stop rules are checked before
each wave, no two questions of one family in a wave, the wave's eps must fit the eps left together, and the investigation ends when
the schedule is used up. Off (None) = the sequential loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ftm.graph.pipeline import AskResult, family_of, library_strategies, route
from ftm.graph.policy import Action, Ask, PlannerView, Stop
from ftm.llm.budget import BudgetedClient, BudgetExceeded, PromptTooLong
from ftm.llm.investigation import outcome
from ftm.planners.llm_batches import QuestionBatchWriter
from ftm.planners.llm_dsl import DslAdapter, V0Dsl, retry_hint
from ftm.planners.llm_prompt import (
    Mode,
    PlannerParseError,
    PlannerState,
    build_messages_with_estimate,
    output_token_budget,
    parse_action,
    planner_schema,
)
from ftm.types import Reply

MAX_PROMPT_REFITS = 2  # resends after a server-counted PromptTooLong, per alert
PROMPT_REFIT_MARGIN = 1.02


@dataclass
class PlannerStats:
    """Aggregated over alerts. `usable_output_rate` is the >=95% acceptance metric (stops count as usable)."""

    alerts: int = 0
    outputs: int = 0
    json_ok: int = 0
    stops: int = 0
    questions: int = 0
    dsl_ok: int = 0
    usable: int = 0  # outputs that were a stop or carried a valid question (a batched output counts once, however many questions)
    duplicates: int = 0
    refusals: int = 0  # asks the gate / accountant refused
    refused_repeats: int = 0  # the same refused (question, bank) proposed again, caught before it reached the gate
    seconds_per_alert: list[float] = field(default_factory=list)
    prompts: int = 0  # planner prompts built (a refit rebuilds one)
    rules_offered: int = 0  # summed over prompts: documents the rules mode offered
    rules_in_prompt: int = 0  # summed over prompts: documents that fit under the prompt cap
    questions_offered: int = 0  # summed over prompts: menu items after routing
    questions_in_prompt: int = 0  # summed over prompts: menu items left after the trimmer (the whole-corpus arm displaces some)

    @property
    def json_parse_rate(self) -> float:
        return self.json_ok / self.outputs if self.outputs else float("nan")

    @property
    def valid_dsl_rate(self) -> float:
        return self.dsl_ok / self.questions if self.questions else float("nan")

    @property
    def usable_output_rate(self) -> float:
        return self.usable / self.outputs if self.outputs else float("nan")

    def as_counts(self) -> dict:
        """The integer counters, so a caller can take per-alert differences (E18 per_alert.jsonl)."""
        return {k: getattr(self, k) for k in ("outputs", "json_ok", "stops", "questions", "dsl_ok", "duplicates", "refusals", "refused_repeats")}

    def as_dict(self) -> dict:
        s = sorted(self.seconds_per_alert)
        return {"alerts": self.alerts, "outputs": self.outputs, "json_parse_rate": self.json_parse_rate,
                "valid_dsl_rate": self.valid_dsl_rate, "usable_output_rate": self.usable_output_rate,
                "duplicates": self.duplicates, "refusals": self.refusals, "refused_repeats": self.refused_repeats,
                "mean_rules_offered": self.rules_offered / self.prompts if self.prompts else float("nan"),
                "mean_rules_in_prompt": self.rules_in_prompt / self.prompts if self.prompts else float("nan"),
                "mean_questions_offered": self.questions_offered / self.prompts if self.prompts else float("nan"),
                "mean_questions_in_prompt": self.questions_in_prompt / self.prompts if self.prompts else float("nan"),
                "median_seconds_per_alert": s[len(s) // 2] if s else float("nan")}  # fmt: skip

    @classmethod
    def merged(cls, parts: list[PlannerStats]) -> PlannerStats:
        """One worker's stats per concurrent worker -> the run's. seconds_per_alert is left to the caller (alert order)."""
        out = cls()
        for p in parts:
            for k, v in p.as_counts().items():
                setattr(out, k, getattr(out, k) + v)
            out.alerts += p.alerts
            out.usable += p.usable
            for k in ("prompts", "rules_offered", "rules_in_prompt", "questions_offered", "questions_in_prompt"):
                setattr(out, k, getattr(out, k) + getattr(p, k))
            out.seconds_per_alert += p.seconds_per_alert
        return out


@dataclass
class _Alert:
    """Per-alert private state of the policy (lives in the graph's `scratch`)."""

    banks: list
    strategies: list[dict]
    feedback: str | None = None
    belief: float | None = None
    typology: str | None = None
    slot: int = 0  # position in batch_schedule (a retried step does not use one up)
    token_scale: float = 1.0  # per alert, so results do not depend on alert order
    refits: int = 0
    trace: list[dict] = field(default_factory=list)
    pending: dict | None = None  # the step being asked: rec, action, and what observe needs


class OursPolicy:
    """The planner's step logic. Stateless across alerts except `stats`; per-alert state is in `scratch["ours"]`."""

    def __init__(self, *, mode: Mode, dsl: DslAdapter | None = None, questions: dict[str, dict] | None = None,
                 use_library: bool = False, batch_writer: QuestionBatchWriter | None = None, temperature: float = 0.0,
                 seed: int = 0, extra: dict | None = None, keep_prompts: bool = False, max_output_tokens: int | None = None,
                 batch_schedule: tuple[int, ...] | None = None, stats: PlannerStats | None = None):  # fmt: skip
        self.mode = mode
        self.dsl = dsl or V0Dsl()
        self.questions = questions or {}
        self.use_library = use_library
        self.batch_writer = batch_writer
        self.temperature, self.seed, self.extra = temperature, seed, extra
        self.keep_prompts, self.max_output_tokens = keep_prompts, max_output_tokens
        self.batch_schedule = batch_schedule
        self.stats = stats if stats is not None else PlannerStats()

    @property
    def name(self) -> str:
        return f"ours_{self.mode.value}"

    # -- plan -----------------------------------------------------------------------------------------------------
    def propose(self, view: PlannerView, llm: BudgetedClient, scratch: dict) -> list[Action]:
        st = scratch["_state"]
        alert, limits, session = st["alert"], st["limits"], st["session"]
        o: _Alert = scratch.get("ours")  # type: ignore[assignment]
        if o is None:
            o = scratch["ours"] = _Alert(banks=list(alert.summary.get("partner_banks", [])), strategies=view.strategies)
            scratch["trace"] = o.trace
        schedule = self.batch_schedule
        if schedule is not None and o.slot >= len(schedule):
            return [Stop("schedule_used_up")]
        step = view.step
        state = PlannerState(
            alert_summary=view.alert_summary, evidence_text=view.evidence, partner_banks=o.banks, eps_left=view.eps_left,
            calls_left=view.calls_left, questions=st["menu_raw"], strategies=o.strategies,
            rules=view.rules, answers=session.answers_so_far(), posterior=view.posterior, feedback=o.feedback,
            wave_k=schedule[o.slot] if schedule else 1, p0=view.p0, threshold=view.threshold,
        )  # fmt: skip
        rec: dict = {"step": step}
        o.trace.append(rec)
        o.pending = None
        try:
            fit: dict = {}
            msgs, est = build_messages_with_estimate(self.mode, state, self.dsl, limits.max_tokens, o.token_scale, fit)
            state.rules = state.rules[:fit["rules_kept"]]  # the schema may only cite rules the prompt shows
            self.stats.prompts += 1
            self.stats.rules_in_prompt += fit["rules_kept"]
            self.stats.rules_offered += fit["rules_offered"]
            self.stats.questions_in_prompt += fit["questions_kept"]
            self.stats.questions_offered += fit["questions_offered"]
            resp = llm.chat(msgs, json_schema=planner_schema(self.mode, state),
                            max_tokens=self.max_output_tokens or output_token_budget(self.mode, state.wave_k), temperature=self.temperature,
                            seed=self.seed, extra=self.extra)  # fmt: skip
        except PromptTooLong as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            if e.server_tokens is None or o.refits >= MAX_PROMPT_REFITS:
                return [Stop("prompt_too_long")]
            # the server's tokenizer counts more than the estimate: trim by the measured ratio and resend (the over-long
            # call stays counted in the budget and the cost)
            o.token_scale, o.refits = max(o.token_scale, e.server_tokens / est * PROMPT_REFIT_MARGIN), o.refits + 1
            rec["refit_token_scale"] = o.token_scale
            scratch["_no_fail"] = True
            return []
        except (BudgetExceeded, PlannerParseError) as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            return [Stop(type(e).__name__)]
        self.stats.outputs += 1
        rec.update(text=resp.text, tokens_in=resp.tokens_in, tokens_out=resp.tokens_out, latency_s=resp.latency_s,
                   posterior_before=session.posterior)
        if self.keep_prompts:
            rec["messages"] = msgs
        try:
            action = parse_action(resp.text, self.mode, state)
        except PlannerParseError as e:
            o.feedback = f"{e}. Reply with one JSON object matching the keys listed under Output."
            rec["error"] = f"parse: {e}"
            return []
        self.stats.json_ok += 1
        rec.update(action=action.action, typology=action.typology, rationale=action.rationale, belief=action.belief)
        if action.belief is not None:
            o.belief = action.belief
        if self.use_library and action.typology not in (None, "none", "unsure", o.typology):
            o.typology = action.typology
            o.strategies = library_strategies(alert, st["tools"], o.typology)
        if action.action == "stop":
            self.stats.stops += 1
            self.stats.usable += 1
            return [Stop("model_stop")]
        if state.wave_k > 1:
            return self._propose_wave(action, view, st, rec, o, scratch)
        eps = action.eps
        if action.action == "select":
            item = self.questions[action.question_id]
            program = item["program"]
            if item.get("eps"):  # back-tested at these levels only: any other level would not move the probability
                ok_eps = [e for e in item["eps"] if e <= session.eps_left + 1e-9]
                if not ok_eps:
                    o.feedback = (f"{action.question_id} needs eps {min(item['eps']):g} but only {session.eps_left:g} is "
                                  "left. Choose another question or stop.")
                    rec["error"] = "select: not enough eps"
                    return []
                if eps not in ok_eps:
                    rec["eps_requested"] = eps
                    eps = min(ok_eps, key=lambda e: (abs(e - eps), e))
        else:
            program = action.program
        self.stats.questions += 1
        check = self.dsl.validate(program)
        rec["program"] = program
        if not check.ok:
            o.feedback = f"invalid question {program!r}: {check.error}. {retry_hint(program, check.error, self.dsl.reference_text())}"
            rec["error"] = f"dsl: {check.error}"
            return []
        self.stats.dsl_ok += 1
        bank, why_not = route(check.canonical, alert, action.bank)
        if why_not:
            o.feedback = f"{check.canonical} cannot be asked: {why_not}. Ask a question about the other party or a set slot, or stop."
            rec["error"] = "no_partner_holds_subject"
            return []
        if bank != action.bank:
            rec["bank_requested"] = action.bank
        self.stats.usable += 1
        refused = st["refused"]
        if (check.canonical, str(bank)) in refused:
            self.stats.refused_repeats += 1
            o.feedback = (f"bank {bank} already refused this exact question ({refused[(check.canonical, str(bank))]}). "
                          "Ask a different question, or stop.")
            rec["error"] = "duplicate_refused"
            return []
        o.pending = {"rec": rec, "action": action, "check": check, "bank": bank, "eps": eps, "wave": False}
        scratch.setdefault("_bank_hint", {})[check.canonical] = action.bank  # the model's bank, used only when routing is unknown
        return [Ask(check.canonical, eps, action.rationale or "")]

    def _propose_wave(self, action, view: PlannerView, st: dict, rec: dict, o: _Alert, scratch: dict) -> list[Action]:
        """Validate one batched step's questions (as the old `_wave` did before sending). Returns the Asks of the wave, or []
        with feedback when nothing could be sent."""
        alert, session, refused = st["alert"], st["session"], st["refused"]
        sent: list[tuple] = []  # (action item, check, eps, q, bank) per question that is sent
        notes: list[str] = []
        per_q: list[dict] = []
        reserved, families, keys = 0.0, set(), set()
        for it in action.wave:
            q: dict = {"action": it.action, "bank": it.bank, "eps": it.eps}
            per_q.append(q)
            eps, left = it.eps, session.eps_left - reserved
            if it.action == "select":
                item = self.questions[it.question_id]
                program = item["program"]
                if item.get("eps"):
                    ok_eps = [e for e in item["eps"] if e <= left + 1e-9]
                    if not ok_eps:
                        q["error"] = "select: not enough eps"
                        notes.append(f"{it.question_id} needs eps {min(item['eps']):g} but only {left:g} is left for this wave.")
                        continue
                    if eps not in ok_eps:
                        q["eps_requested"] = eps
                        eps = min(ok_eps, key=lambda e: (abs(e - eps), e))
            else:
                program = it.program
            q["program"], q["eps"] = program, eps
            self.stats.questions += 1  # as in the sequential path, a select that cannot get an eps level is not counted
            check = self.dsl.validate(program)
            if not check.ok:
                q["error"] = f"dsl: {check.error}"
                notes.append(f"invalid question {program!r}: {check.error}. {retry_hint(program, check.error, self.dsl.reference_text())}")
                continue
            self.stats.dsl_ok += 1
            bank, why_not = route(check.canonical, alert, it.bank)
            if why_not:
                q["error"] = "no_partner_holds_subject"
                notes.append(f"{check.canonical} cannot be asked: {why_not}.")
                continue
            if bank != it.bank:
                q["bank_requested"] = it.bank
                q["bank"] = bank
            if (check.canonical, str(bank)) in refused:
                self.stats.refused_repeats += 1
                q["error"] = "duplicate_refused"
                notes.append(f"bank {bank} already refused this exact question ({refused[(check.canonical, str(bank))]}).")
                continue
            if (check.canonical, str(bank)) in keys:
                self.stats.duplicates += 1
                q["error"] = "duplicate_in_wave"
                notes.append(f"{check.canonical} to bank {bank} appears twice in the wave.")
                continue
            fam = family_of(check.canonical)
            if fam is not None and fam in families:
                q["error"] = "same_family_in_wave"
                notes.append(f"{check.canonical} is of the same family ({fam}) as another question in the wave: ask different families together.")
                continue
            if eps > left + 1e-9:
                q["error"] = "wave_over_eps"
                notes.append(f"the wave's questions do not fit the eps left together ({session.eps_left:g}).")
                continue
            keys.add((check.canonical, str(bank)))
            if fam is not None:
                families.add(fam)
            reserved += eps
            q["canonical"] = check.canonical
            sent.append((it, check, eps, q, bank))
        rec["wave"] = per_q
        self.stats.usable += bool(sent)
        if not sent:
            rec["error"] = "wave: no valid question"
            o.feedback = " ".join(notes) + " Propose a different wave, or stop."
            return []
        o.pending = {"rec": rec, "action": action, "sent": sent, "notes": notes, "wave": True}
        scratch["_wave"] = True
        hints = scratch.setdefault("_bank_hint", {})
        for it, check, _eps, _q, _bank in sent:
            hints[check.canonical] = it.bank
        return [Ask(check.canonical, eps, it.rationale or "") for it, check, eps, _, _ in sent]

    # -- after act ------------------------------------------------------------------------------------------------
    def observe(self, results: list[AskResult], scratch: dict) -> None:
        o: _Alert = scratch["ours"]
        p, st = o.pending, scratch["_state"]
        if p is None:
            return
        session, alert, step = st["session"], st["alert"], st["step"]
        rec, action = p["rec"], p["action"]
        if p["wave"]:
            self._observe_wave(results, p, rec, action, o, session, alert, step)
            return
        check, bank, eps = p["check"], p["bank"], p["eps"]
        r = results[0]
        if r.duplicate is not None:
            self.stats.duplicates += 1
            o.feedback = f"{r.duplicate} (the answer is under Answers so far; only bank {bank} holds this subject). Ask a different question, or stop."
            rec["error"] = "duplicate"
            return
        entry = r.entry
        rec.update(canonical=check.canonical, bank=bank, eps=eps, outcome=outcome(entry), posterior_after=entry.posterior_after)
        if entry.gate_decision != "allowed":  # blocked / over budget: not a success, counts toward the stop rule
            self.stats.refusals += 1
            why = session.last_refusal or entry.gate_decision
            o.feedback = f"the {entry.gate_decision} refusal for bank {bank}: {why}. Ask a different question or bank, or stop."
            rec["error"] = f"refused: {why}"
            return  # the pipeline stops the alert when the clock or the per-alert eps budget is spent ("used up")
        o.feedback, o.slot = None, o.slot + 1
        if isinstance(entry.response, Reply):  # the answer, so distillation can re-fuse with the real fusion model
            rec.update(answer_value=entry.response.value, answer_kind=entry.response.kind)
        if action.action == "compose" and self.batch_writer is not None:
            self.batch_writer.add(canonical=check.canonical, kind=check.kind, family=check.family, eps=eps,
                                  bank=bank, alert_id=alert.alert_id, split=alert.split, step=step,
                                  mode=self.mode.value, typology_guess=action.typology)  # fmt: skip

    def _observe_wave(self, results, p, rec, action, o: _Alert, session, alert, step) -> None:
        notes = p["notes"]
        any_allowed = False
        for (it, check, eps, q, bank), r in zip(p["sent"], results, strict=True):
            if r.duplicate is not None:
                self.stats.duplicates += 1
                q["error"] = "duplicate"
                notes.append(f"{r.duplicate} (the answer is under Answers so far).")
                continue
            entry = r.entry
            q.update(canonical=check.canonical, outcome=outcome(entry), posterior_after=entry.posterior_after)
            if entry.gate_decision != "allowed":
                self.stats.refusals += 1
                why = session.last_refusal or entry.gate_decision
                q["error"] = f"refused: {why}"
                notes.append(f"the {entry.gate_decision} refusal for bank {bank}: {why}.")
                continue
            any_allowed = True
            if isinstance(entry.response, Reply):
                q.update(answer_value=entry.response.value, answer_kind=entry.response.kind)
            if it.action == "compose" and self.batch_writer is not None:
                self.batch_writer.add(canonical=check.canonical, kind=check.kind, family=check.family, eps=eps, bank=bank,
                                      alert_id=alert.alert_id, split=alert.split, step=step, mode=self.mode.value,
                                      typology_guess=action.typology)  # fmt: skip
        if not any_allowed:
            rec["error"] = "wave: nothing allowed"
            o.feedback = " ".join(notes) + " Propose a different wave, or stop."
            return
        o.feedback, o.slot = None, o.slot + 1

    # -- decide ---------------------------------------------------------------------------------------------------
    def finish(self, state: dict, scratch: dict) -> float | None:
        """The LLM's belief is the verdict in no_backtest / llm_decides (H4 / H7); else None = the fused posterior."""
        if not self.mode.has_belief:
            return None
        o: _Alert | None = scratch.get("ours")
        belief = o.belief if o is not None else None
        return belief if belief is not None else state["alert"].p0
