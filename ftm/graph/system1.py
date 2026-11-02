"""System 1 in the system graph: the fast model + conformal cascade that answers the small decisions before the planner (C13).

Owner: T3 (Akshat).  Spec: wave 4 (docs/t3_system1.md).  Our open analogue of the paid "System One" decision model.

What it does inside `ftm.graph.pipeline.GraphInvestigator` (switched on with `set_system1`, `run_llm_agent.py --system1 <frozen json>`):

  worth_asking      node `system1`, before each `plan` step: "can a further question still improve the call?"  Settled "no"
                    -> the pipeline stops (reason `system1_not_worth_asking`) and the planner LLM call is skipped.
  injection_present after each answered question that carries a partner-bank note: settled "no" -> the note stays in the log;
                    "yes" or unsure -> the note text is QUARANTINED (removed from the log the planner prompt is built from; the
                    text is kept in the System 1 record only).
  typology          once per alert (prepare): a settled label other than `none` is added to the planner's alert summary as
                    `system1_typology_hint` and keys the strategy-library lookup.
  stop              not a model: the exact cannot_cross rule the pipeline already runs (`stop_cannot_cross`, tuned profile).

Cascade per decision (`ftm.fastmodel.cascade`): fast tier (Qwen3-4B + LoRA on its own server, temperature-scaled one-token label
distribution) settles when its Mondrian conformal set is a singleton; otherwise the LARGE tier (the backbone server, Gemma 4 12B,
the same one-token read of its logprobs) settles on its own conformal set; otherwise the SAFE DEFAULT (ask / quarantine / no hint).
A tier retired on cal (did not beat the majority label) is skipped.  A tier whose server errors counts as unsure.  No stub: both
tiers are real served models; `tiers="fast"` is the ablation arm that never calls the large tier.

Every decision is recorded (decision, tier, label, conformal sets, probabilities, seconds) in `System1.records` -> traces and
`system1.jsonl` of the run, so H5 (speed-up, detection loss vs System 1 off, tier shares, realised error) is measurable.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ftm.fastmodel import context as ctxt
from ftm.fastmodel.cascade import ConformalCascade, ConformalSet
from ftm.fastmodel.decisions import DECISIONS, DecisionExample
from ftm.fastmodel.model import LogprobFastModel

FUNCTIONS = ("worth", "inject", "typology")
TIERS = ("cascade", "fast")
FROZEN_VERSION = 3


@dataclass
class S1Decision:
    decision: str
    label: str | None  # the settled label, or the safe default (None: no default, e.g. typology)
    tier: str  # fast | large | default
    fast_set: list = field(default_factory=list)
    large_set: list = field(default_factory=list)
    seconds: float = 0.0
    error: str | None = None

    @property
    def settled(self) -> bool:
        return self.tier in ("fast", "large")


class System1:
    def __init__(self, frozen: dict, fast_client, large_client=None, *, functions=FUNCTIONS, tiers: str = "cascade",
                 threshold: float = 0.3227):  # fmt: skip
        if frozen.get("version") != FROZEN_VERSION:
            raise ValueError(f"System 1 frozen file must be version {FROZEN_VERSION} (scripts/score_fast_baseline.py report --freeze)")
        for d in frozen["decisions"]:
            if d not in DECISIONS or list(DECISIONS[d].labels) != frozen["labels"][d]:
                raise ValueError(f"frozen file and ftm.fastmodel.decisions disagree on {d}")
        bad = set(functions) - set(FUNCTIONS)
        if bad or tiers not in TIERS:
            raise ValueError(f"functions must be within {FUNCTIONS} and tiers one of {TIERS}")
        if tiers == "cascade" and large_client is None:
            raise ValueError("tiers=cascade needs the large-tier client (the backbone server)")
        self.frozen, self.functions, self.tiers, self.threshold = frozen, tuple(functions), tiers, threshold
        large_off = set(frozen["large_off"]) | (set(frozen["decisions"]) if tiers == "fast" else set())
        self.cascade = ConformalCascade(ConformalSet.from_dict(frozen["fast"]), ConformalSet.from_dict(frozen["large"]),
                                        frozen["fast_off"], large_off)
        self.fast = {d: LogprobFastModel(fast_client, temperature=frozen["temperatures"][d]) for d in frozen["decisions"]}
        self.large = ({d: LogprobFastModel(large_client, temperature=frozen["temperatures_large"][d]) for d in frozen["decisions"]}
                      if large_client is not None else {})
        self.records: list[dict] = []
        self._lock = threading.Lock()

    @classmethod
    def from_file(cls, path: str | Path, fast_client, large_client=None, **kw) -> System1:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), fast_client, large_client, **kw)

    def enabled(self, function: str) -> bool:
        return function in self.functions and {"worth": "worth_asking", "inject": "injection_present",
                                               "typology": "typology"}[function] in self.frozen["decisions"]

    # -- one decision ---------------------------------------------------------------------------------------------
    def decide(self, decision: str, text: str, *, alert_id: str = "", step: int = 0, extra: dict | None = None) -> S1Decision:
        ex = DecisionExample(decision, text)
        t0 = time.perf_counter()
        errors: list[str] = []

        def fast_probs():
            if decision in self.cascade.fast_off:
                return None
            try:
                return self.fast[decision].predict_proba(ex)
            except Exception as e:  # noqa: BLE001 - a server error is "unsure": the next tier / the safe default takes over
                errors.append(f"fast: {e}")
                return None

        def large_probs():
            try:
                return self.large[decision].predict_proba(ex)
            except Exception as e:  # noqa: BLE001
                errors.append(f"large: {e}")
                return {lbl: 1 / len(DECISIONS[decision].labels) for lbl in DECISIONS[decision].labels}  # uniform: full set

        fp = fast_probs()
        if fp is None and decision not in self.cascade.fast_off:
            fp = {lbl: 1 / len(DECISIONS[decision].labels) for lbl in DECISIONS[decision].labels}  # fast server down: full set
        r = self.cascade.decide(decision, fp, large_probs)
        out = S1Decision(decision, r.answer, r.source, r.fast_set, r.large_set, time.perf_counter() - t0, "; ".join(errors) or None)
        rec = {"alert_id": alert_id, "step": step, "decision": decision, "tier": out.tier, "label": out.label,
               "fast_set": out.fast_set, "large_set": out.large_set, "seconds": round(out.seconds, 4), "error": out.error,
               "tiers": self.tiers, **(extra or {})}  # fmt: skip
        with self._lock:
            self.records.append(rec)
        return out

    # -- the three functions --------------------------------------------------------------------------------------
    def worth_asking(self, summary: dict, p0: float, posterior: float, eps_left: float, answers: list[dict], menu: list, *,
                     threshold: float | None = None, alert_id: str = "", step: int = 0) -> S1Decision:
        text = ctxt.step_text(summary, p0, posterior, threshold if threshold is not None else self.threshold, eps_left, answers, menu)
        return self.decide("worth_asking", text, alert_id=alert_id, step=step)

    def screen_note(self, bank, note: str, *, alert_id: str = "", step: int = 0) -> tuple[bool, S1Decision]:
        """(quarantine?, decision). Quarantine unless the cascade settles `no`."""
        d = self.decide("injection_present", ctxt.note_text(bank, note), alert_id=alert_id, step=step)
        quarantine = d.label != "no"
        with self._lock:
            self.records[-1].update(quarantined=quarantine, note_chars=len(note))
        return quarantine, d

    def typology_hint(self, summary: dict, p0: float, *, alert_id: str = "") -> str | None:
        """A settled non-`none` pattern label, else None (no hint)."""
        d = self.decide("typology", ctxt.alert_text(summary, p0), alert_id=alert_id)
        return d.label if d.settled and d.label not in (None, "none") else None

    def alert_records(self, alert_id: str) -> list[dict]:
        with self._lock:
            return [r for r in self.records if r["alert_id"] == alert_id]


def attach(system, s1: System1) -> None:
    """Switch System 1 on for a built investigator: a GraphInvestigator (adapted arms) or an LLMPlanner (`.graph`)."""
    g = getattr(system, "graph", system)
    if not hasattr(g, "set_system1"):
        raise ValueError(f"{type(system).__name__} does not run in the system graph; System 1 needs ftm.graph.pipeline")
    g.set_system1(s1)
