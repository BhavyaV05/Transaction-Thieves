"""C3: a simulated partner bank. It holds its own private ledger and answers typed questions with one noisy value.

Owner: T2 (Bhavya).  Spec: plan.md 6.4, tracks.md 4.4.

`PartnerBank.handle(question, as_of) -> Reply | Decline | Timeout`:
  1. exact answer: the DSL program (C4) runs on this bank's ledger, cut at the answer time (as_of + latency);
  2. noise: the mechanism for the answer kind (C5) with the question's eps and the program's sensitivity;
  3. latency (days): usually under a day, sometimes up to four; declines with p_decline; a reply that never comes
     is a Timeout after `timeout_days` (probability p_timeout);
  4. an optional untrusted text note, independent of the true answer (it carries no information; it is where T4's
     injection set plants text).

Everything a bank draws is seeded from (seed, bank, alert, canonical question, eps, purpose), so a reply does not
depend on the order of questions, shards can run anywhere, and asking the same question again gives the same reply.

No code path returns rows: `handle` returns only Reply/Decline/Timeout whose value is a bool or an int
(tests/test_banks.py checks the types and scans this package for row-returning calls).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from ftm.banks.binding import AlertBinding
from ftm.dsl.executor import ExecError, execute
from ftm.dsl.parser import DslError, arity, canonical, parse, sensitivity
from ftm.privacy.mechanisms import randomize
from ftm.types import BankResponse, Decline, Reply, Timeout

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "banks" / "default.yaml"

NEUTRAL_NOTES = (
    "Reply given under 314(b) procedures.",
    "Figures cover our ledger only.",
    "Counts exclude items still pending.",
    "Please send follow-ups through the usual channel.",
)


@dataclass(frozen=True)
class BankConfig:
    typical_max_days: float = 1.0
    tail_max_days: float = 4.0
    p_tail: float = 0.15  # share of replies slower than typical_max
    min_latency_days: float = 0.02
    timeout_days: float = 5.0
    p_decline: float = 0.05
    p_timeout: float = 0.02
    notes_enabled: bool = True
    p_note: float = 0.3

    @classmethod
    def load(cls, path: Path = CONFIG) -> BankConfig:
        c = yaml.safe_load(Path(path).read_text())
        lat = c.get("latency_days", {})
        d = cls()
        return cls(
            typical_max_days=lat.get("typical_max", d.typical_max_days),
            tail_max_days=lat.get("tail_max", d.tail_max_days),
            p_tail=c.get("p_tail", d.p_tail),
            min_latency_days=c.get("min_latency_days", d.min_latency_days),
            timeout_days=c.get("timeout_days", d.timeout_days),
            p_decline=c.get("p_decline", d.p_decline),
            p_timeout=c.get("p_timeout", d.p_timeout),
            notes_enabled=c.get("notes_enabled", d.notes_enabled),
            p_note=c.get("p_note", d.p_note),
        )  # fmt: skip


@dataclass(frozen=True)
class Question:
    """What an asking bank sends: a DSL program, the eps it pays for, and the alert it is about."""

    program: str
    eps: float
    alert: AlertBinding


NoteFn = Callable[[Question, object, np.random.Generator], "str | None"]


def default_note(question: Question, value, rng: np.random.Generator) -> str | None:
    """A neutral note. It must not depend on `value`."""
    return NEUTRAL_NOTES[int(rng.integers(0, len(NEUTRAL_NOTES)))]


def _seed(*parts) -> int:
    h = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big")


class PartnerBank:
    needs_binding = True  # answers from a real ledger, so it must know the alert's real node ids

    def __init__(self, bank: int, ledger: pl.DataFrame, config: BankConfig | None = None, seed: int = 0,
                 entity_of: dict | None = None, note_fn: NoteFn | None = None):  # fmt: skip
        self.bank = int(bank)
        self._ledger = ledger  # private: nothing here returns it
        self.cfg = config or BankConfig.load()
        self.seed = seed
        self._entity_of = entity_of
        self._note_fn = note_fn or default_note

    def __repr__(self) -> str:
        return f"PartnerBank({self.bank}, {self._ledger.height} tx)"

    def _with_config(self, config: BankConfig) -> PartnerBank:
        """The same bank (ledger, seed, entities, note function) under another config, e.g. more declines (E14). Private: the
        public surface of a bank stays as `test_banks` pins it."""
        return PartnerBank(self.bank, self._ledger, config, self.seed, self._entity_of, self._note_fn)

    def _rng(self, question: Question, canon: str, purpose: str) -> np.random.Generator:
        key = (self.seed, self.bank, question.alert.alert_id, canon, question.eps, purpose)
        return np.random.default_rng(_seed(*key))

    def _latency(self, rng: np.random.Generator) -> float:
        c = self.cfg
        u_tail, u_lat = rng.random(), rng.random()
        if u_tail < c.p_tail:
            return c.typical_max_days + u_lat * (c.tail_max_days - c.typical_max_days)
        return c.min_latency_days + u_lat * (c.typical_max_days - c.min_latency_days)

    def respond(self, question: Question, as_of: datetime, purpose: str = "live") -> tuple[BankResponse, float]:
        """The reply and how long it takes (days). `handle` is this without the wait."""
        try:
            node = parse(question.program)
        except DslError as e:
            return Decline(self.bank, f"invalid question: {e}"), self.cfg.min_latency_days
        canon = canonical(node)
        rng = self._rng(question, canon, purpose)
        # a fixed order of draws, whatever the outcome
        u_decline, u_timeout = rng.random(), rng.random()
        latency = self._latency(rng)
        u_note = rng.random()
        if u_decline < self.cfg.p_decline:
            return Decline(self.bank, "declined"), latency
        if u_timeout < self.cfg.p_timeout:
            return Timeout(self.bank, self.cfg.timeout_days), self.cfg.timeout_days
        cutoff = as_of + timedelta(days=latency)
        ctx = question.alert.exec_context(self.bank, cutoff, self._entity_of)
        try:
            exact = execute(node, self._ledger, ctx)
        except ExecError as e:
            return Decline(self.bank, f"cannot answer: {e}"), latency
        value = randomize(node.prim.kind, exact, question.eps, sensitivity(node), rng, arity(node))
        note = None
        if self.cfg.notes_enabled and u_note < self.cfg.p_note:
            note = self._note_fn(question, value, rng)
        return Reply(self.bank, node.prim.kind, value, question.eps, latency, note), latency

    def handle(self, question: Question, as_of: datetime, purpose: str = "live") -> BankResponse:
        return self.respond(question, as_of, purpose)[0]

    def ask(self, env, question: Question, start: datetime):
        """SimPy process: waits the reply's latency (env time in days since `start`), then returns the reply.

            reply = yield env.process(bank.ask(env, q, start))
        """
        as_of = start + timedelta(days=env.now)
        response, wait = self.respond(question, as_of)
        yield env.timeout(wait)
        return response
