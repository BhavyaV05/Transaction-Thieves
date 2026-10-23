"""The small-decision dataset (hand-off #13, T2 to T3): labelled one-token decisions for the fast model (C13).

Owner: T2 (Bhavya).  Spec: plan.md 6.13/6.14 and 5.x, tracks.md 4.9.  Format: T3's `ftm/fastmodel/decisions.py`
(`decision, context, label, alert_id, split`; the labels below are its `DECISIONS`).

Per alert it writes `per_alert` (default 30) decisions, from the simulator:

  worth_asking      yes iff the alerted transfer belongs to a real laundering attempt (alert label); contexts differ
                    in the evidence window and whether p0 is shown
  typology          the pattern-file typology of the attempt, "none" for a clean alert (laundering alerts with no
                    typology in the pattern file are skipped)
  rule_applies      yes iff the cited rule permits ask_bank in the home bank's jurisdiction for this answer kind
                    (T4's `Gate` rules); permitting and non-permitting citations for the same question
  stop              yes iff no remaining back-tested question could move the posterior across the decision threshold
                    for any answer (`stop_label`); contexts hold the answers so far, produced by the simulated banks
  injection_present yes iff the partner-bank note carries a planted instruction (T4's generator,
                    `ftm.safety.injection`); benign notes are the negatives

Contexts are aliased text: aggregate statistics of what the home bank saw up to the alert time (never past it), public
bank ids, amounts. They contain no account id, alias, name or date, and nothing about the label beyond what the home
bank knew at the alert time. Test-split alerts are never used.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from datetime import timedelta
from pathlib import Path

import polars as pl

from ftm.backtest.engine import BtAlert
from ftm.banks.partner import Question
from ftm.compliance.gate import Gate
from ftm.dsl.expert import expert_programs
from ftm.dsl.parser import canonical, parse
from ftm.fusion.fuse import logit, sigmoid
from ftm.planners.pool import Candidate, QuestionPool
from ftm.safety import injection as inj
from ftm.types import Reply

DECISION_LABELS = {
    "worth_asking": ("yes", "no"),
    "typology": ("fan_in", "fan_out", "cycle", "scatter_gather", "gather_scatter", "bipartite", "stack", "random", "none"),
    "rule_applies": ("yes", "no"),
    "injection_present": ("yes", "no"),
    "stop": ("yes", "no"),
}  # fmt: skip
DEFAULT_MIX = {"worth_asking": 6, "typology": 4, "rule_applies": 8, "stop": 6, "injection_present": 6}
WINDOWS = (24, 48, 72)
MIN_EPS = 0.25


# --- contexts ---------------------------------------------------------------------------------------------------
def _money(x: float) -> str:
    return f"{x:,.0f}"


def evidence_text(a: BtAlert, home: pl.DataFrame, window_h: int) -> str:
    """What the home bank saw about the alerted transfer's counterparty in the `window_h` hours up to the alert: counts
    and amounts only. `home` is the home bank's ledger; rows after the alert time are cut here."""
    b = a.binding
    t0, lo = b.t0, b.t0 - timedelta(hours=window_h)
    cp = b.receiver if b.sender_bank == b.home_bank else b.sender
    w = home.filter((pl.col("timestamp") >= lo) & (pl.col("timestamp") <= t0))
    to_cp = w.filter(pl.col("dst_id") == cp)
    from_cp = w.filter(pl.col("src_id") == cp)
    cash = to_cp.filter(pl.col("payment_format") == "Cash").height if "payment_format" in w.columns else 0
    direction = "sent to" if b.sender_bank == b.home_bank else "received from"
    parts = [
        f"Alert: the home bank {direction} bank {b.partner_bank} a payment of {_money(b.amount)} {b.currency} "
        f"(weekday {t0.weekday()}, hour {t0.hour}).",
        f"Home-bank view of the counterparty account, last {window_h} h: {to_cp.height} payments in from "
        f"{to_cp['src_id'].n_unique()} home-side accounts (total {_money(float(to_cp['amount_received'].sum() or 0))}, "
        f"largest {_money(float(to_cp['amount_received'].max() or 0))}, {cash} in cash); {from_cp.height} payments out "
        f"(total {_money(float(from_cp['amount_paid'].sum() or 0))}).",
    ]
    return " ".join(parts)


def _p0_text(p0: float) -> str:
    return f"One-bank model score: {p0:.2f}."


def _answer_text(prog: str, bank: int, reply) -> str:
    return f"- asked bank {bank}: {prog} -> {reply.value!r} ({reply.kind}, eps {reply.eps_spent:g})"


# --- labels -----------------------------------------------------------------------------------------------------
def stop_label(p: float, remaining: Iterable[Candidate], threshold: float, eps_left: float) -> bool:
    """True iff the investigation may stop: no eps for the smallest question, or no remaining back-tested question has
    an answer that would move the posterior to the other side of the decision threshold."""
    cands = [c for c in remaining if c.eps <= eps_left + 1e-9]
    if eps_left + 1e-9 < MIN_EPS or not cands:
        return True
    side = p >= threshold
    z = float(logit(p))
    for c in cands:
        for cell, lr in c.lr.items():
            reachable = c.p_cell is None or max(c.p_cell.get(cell, (1.0, 1.0))) > 1e-3
            if reachable and lr > 0 and (float(sigmoid(z + math.log(lr))) >= threshold) != side:
                return False
    return True


def rule_label(gate: Gate, rule_id: str, jurisdiction: str, kind: str) -> bool:
    r = gate.rules[rule_id]
    return r.applies_to(jurisdiction) and "ask_bank" in r.permits and kind in r.kinds


# --- the builder ------------------------------------------------------------------------------------------------
class DecisionSetBuilder:
    def __init__(self, ledgers: dict[int, pl.DataFrame], banks: dict, pool: QuestionPool | None = None,
                 threshold: float = 0.3, gate: Gate | None = None, mix: dict | None = None, seed: int = 0,
                 injections: Sequence | None = None):  # fmt: skip
        self.ledgers, self.banks, self.pool, self.threshold = ledgers, banks, pool, threshold
        self.gate = gate or Gate.from_config()
        self.mix = {**DEFAULT_MIX, **(mix or {})}
        self.seed = seed
        self.injections = list(injections) if injections is not None else inj.generate(100, 400, seed)
        self.rules = list(self.gate.rules)
        self.programs = [p["program"] for p in expert_programs("receiver")]

    def per_alert(self) -> int:
        return sum(self.mix.values())

    def rows_for(self, a: BtAlert) -> Iterator[dict]:
        rng = random.Random(f"{self.seed}|{a.alert_id}")
        base = {"alert_id": a.alert_id, "split": a.split}
        home = self.ledgers[a.binding.home_bank]
        for i in range(self.mix["worth_asking"]):
            w = WINDOWS[i % 3]
            ctx = evidence_text(a, home, w) + (" " + _p0_text(a.p0) if i >= 3 else "")
            yield {**base, "decision": "worth_asking", "context": ctx, "label": "yes" if a.label else "no"}
        if a.label is False or a.typology:
            for i in range(self.mix["typology"]):
                yield {**base, "decision": "typology", "context": evidence_text(a, home, WINDOWS[i % 3]),
                       "label": a.typology if a.label else "none"}  # fmt: skip
        yield from self._rule_rows(a, rng, base)
        yield from self._stop_rows(a, rng, base)
        yield from self._injection_rows(a, rng, base)

    def _rule_rows(self, a, rng, base):
        for i in range(self.mix["rule_applies"]):
            prog = rng.choice(self.programs)
            kind = parse(prog).prim.kind
            jur = rng.choice(["US", "US", "SG"])
            permitting = [r for r in self.rules if rule_label(self.gate, r, jur, kind)]
            other = [r for r in self.rules if not rule_label(self.gate, r, jur, kind)]
            rid = rng.choice(permitting if (i % 2 == 0 and permitting) or not other else other)
            rule = self.gate.rules[rid]
            ctx = (f"Home bank jurisdiction: {jur}. Candidate question to a partner bank (answer kind {kind}): "
                   f"{canonical(parse(prog))}. Cited rule {rid} ({rule.jurisdiction}): {' '.join(rule.text.split())}")  # fmt: skip
            yield {**base, "decision": "rule_applies", "context": ctx,
                   "label": "yes" if rule_label(self.gate, rid, jur, kind) else "no"}  # fmt: skip

    def _stop_rows(self, a, rng, base):
        if self.pool is None or not len(self.pool):
            return
        direction = "out" if a.binding.sender_bank == a.binding.home_bank else "in"
        options = self.pool.applicable(direction)
        bank = a.binding.partner_bank
        if not options or bank not in self.banks:
            return
        home = self.ledgers[a.binding.home_bank]
        ev = evidence_text(a, home, 72) + " " + _p0_text(a.p0)
        for i in range(self.mix["stop"]):
            k = i % 3  # answers so far: 0, 1, 2
            asked, families, lines, z, eps_left = set(), set(), [], float(logit(a.p0)), 2.0
            for _ in range(k):
                avail = [c for c in options if c.program not in asked and c.family not in families
                         and c.eps <= eps_left + 1e-9]  # fmt: skip
                if not avail:
                    break
                c = rng.choice(avail)
                asked.add(c.program)
                r = self.banks[bank].handle(Question(c.program, c.eps, a.binding), a.binding.t0, purpose=f"decision{i}")
                if isinstance(r, Reply):
                    eps_left -= r.eps_spent
                    families.add(c.family)
                    lines.append(_answer_text(c.program, bank, r))
                    z += c.log_lr(r.value) or 0.0
            p = float(sigmoid(z))
            remaining = [c for c in options if c.program not in asked and c.family not in families]
            ctx = (ev + f" Answers so far ({len(lines)}):\n" + ("\n".join(lines) if lines else "- none")
                   + f"\nPosterior {p:.2f}; decision threshold {self.threshold:.2f}; eps left {eps_left:g}.")
            yield {**base, "decision": "stop", "context": ctx,
                   "label": "yes" if stop_label(p, remaining, self.threshold, eps_left) else "no"}  # fmt: skip

    def _injection_rows(self, a, rng, base):
        for _ in range(self.mix["injection_present"]):
            item = rng.choice(self.injections)
            bank = a.binding.partner_bank
            ctx = f"Partner bank {bank} replied to a question with an answer and this note: \"{item.note}\""
            yield {**base, "decision": "injection_present", "context": ctx,
                   "label": "yes" if item.injection_present else "no"}  # fmt: skip

    def build(self, alerts: Iterable[BtAlert], max_total: int | None = None) -> Iterator[dict]:
        n = 0
        for a in alerts:
            if a.split == "test":
                raise ValueError("the test split is never used for the decision set")
            for row in self.rows_for(a):
                yield row
                n += 1
                if max_total is not None and n >= max_total:
                    return


# --- writing ----------------------------------------------------------------------------------------------------
def write_set(rows: Iterable[dict], out_dir: str | Path, name: str = "decisions") -> dict:
    """One JSONL file per split plus manifest.json (counts, label balance, sha256). Not committed (README rule 7)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files, counts, labels = {}, Counter(), Counter()
    handles = {}
    try:
        for r in rows:
            f = handles.get(r["split"]) or handles.setdefault(r["split"], (out / f"{name}_{r['split']}.jsonl").open("w", encoding="utf-8"))
            f.write(json.dumps(r) + "\n")
            counts[(r["split"], r["decision"])] += 1
            labels[(r["decision"], r["label"])] += 1
    finally:
        for f in handles.values():
            f.close()
    for split in handles:
        p = out / f"{name}_{split}.jsonl"
        files[split] = {"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    manifest = {"files": files, "n": sum(counts.values()),
                "by_split_decision": {f"{s}/{d}": n for (s, d), n in sorted(counts.items())},
                "labels": {f"{d}/{lab}": n for (d, lab), n in sorted(labels.items())}}  # fmt: skip
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
