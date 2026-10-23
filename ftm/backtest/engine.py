"""The back-testing engine (C7): run a DSL question on past labelled alerts, estimate its LRs, test it with e-values.

Owner: T2 (Bhavya); statistics reviewed by T4.  Spec: plan.md 6.8, tracks.md 4.6.

For one (question, eps) hypothesis:
  1. every training alert whose partner bank can be asked is put to that bank (`PartnerBank.handle(purpose="cal")`),
     so the answer carries exactly the noise a live answer would, and the cost is charged to eps_cal;
  2. the answers (not the labels) fix the answer cells (`choose_bins` for counts);
  3. LR per cell with smoothing and the DP ceiling (`estimate_lr`);
  4. the label-permutation e-value over the alerts in time order (`evalue`).
A batch of hypotheses is then accepted with e-BH at the FDR level; a question that arrives later (an LLM-written
one at test time) is accepted by e-LOND (`OnlineAcceptor`). Both use only the e-values.

Realistic mode (main results): `alerts` are the home banks' own past alerts, labels from those alerts, answers noisy.
Exact mode (ablation, week 6): pass eps = inf and the all-banks labelled set; nothing else changes.

Sharding: a hypothesis is independent of every other one, so `shard_of` splits a batch by question across machines;
every shard writes the same `BacktestCache`, keyed by (namespace, canonical, eps), and `assemble` runs e-BH over the
cache. The namespace names the alert set, seed and configuration, so a stale entry is never read.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ftm.backtest.evalue import ELond, e_bh, log_evalue
from ftm.backtest.lr import cells_for, choose_bins, estimate_lr
from ftm.banks.binding import AlertBinding
from ftm.banks.partner import PartnerBank, Question
from ftm.dsl.parser import Node, arity, canonical, family, parse, sensitivity
from ftm.dsl.spec import ACCOUNT_SLOTS
from ftm.privacy.accountant import PrivacyAccountant, charge_keys
from ftm.privacy.mechanisms import is_exact
from ftm.types import Reply


@dataclass(frozen=True)
class BtAlert:
    """A past alert with its label. The label never goes to a bank or an LLM; only `binding` does."""

    alert_id: str
    label: bool
    binding: AlertBinding
    p0: float = 0.5  # the one-bank score the alert was raised with
    typology: str | None = None  # pattern-file typology of the laundering attempt the alerted transfer belongs to
    split: str = "train"


@dataclass(frozen=True)
class BacktestConfig:
    alpha: float = 0.01  # LR smoothing, on the probability scale (plan.md 6.8)
    fdr: float = 0.10
    min_answers: int = 100  # fewer answers than this and the question cannot be accepted
    min_per_class: int = 15
    max_bins: int = 6
    min_per_bin: int = 20
    shrink: float = 2.0  # e-process bet shrinkage toward the null
    mode: str = "realistic"  # realistic | exact (labels and noise are decided by the caller; this names the cache)

    def namespace(self, alert_set_id: str, seed: int) -> str:
        blob = json.dumps({**asdict(self), "alerts": alert_set_id, "seed": seed}, sort_keys=True)
        return hashlib.blake2b(blob.encode(), digest_size=6).hexdigest()


def asked_bank(node: Node, binding: AlertBinding) -> int | None:
    """Which bank a question is put to: the one that holds its subject. Account slots name the party at that bank,
    set slots go to the partner on the other side of the alerted transfer. None when the subject is a home-bank
    account (a partner would hold nothing about it) or there is no partner."""
    while isinstance(node.subject, Node):
        node = node.subject
    if node.subject in ACCOUNT_SLOTS:
        bank = binding.sender_bank if node.subject == "sender" else binding.receiver_bank
        return None if bank is None or bank == binding.home_bank else bank
    return binding.partner_bank


# --- cache ------------------------------------------------------------------------------------------------------
class BacktestCache:
    """One small JSON file per (namespace, canonical, eps). Safe for several shards writing at once."""

    def __init__(self, root: str | Path, namespace: str):
        self.dir = Path(root) / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, canon: str, eps: float) -> Path:
        key = hashlib.blake2b(f"{canon}|{eps:g}".encode(), digest_size=10).hexdigest()
        return self.dir / f"{key}.json"

    def get(self, canon: str, eps: float) -> dict | None:
        p = self._path(canon, eps)
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, entry: dict) -> None:
        p = self._path(entry["program_canonical"], entry["eps"])
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(entry))
        os.replace(tmp, p)  # atomic: a reader never sees half a file

    def entries(self) -> list[dict]:
        # entries are <20 hex digits>.json; report.json and the other files written next to them are not entries
        keyed = (p for p in self.dir.glob("*.json") if len(p.stem) == 20 and all(c in "0123456789abcdef" for c in p.stem))
        return [json.loads(p.read_text()) for p in sorted(keyed)]


def shard_of(canon: str, n_shards: int) -> int:
    return int(hashlib.blake2b(canon.encode(), digest_size=4).hexdigest(), 16) % n_shards


# --- the engine -------------------------------------------------------------------------------------------------
@dataclass
class Backtester:
    banks: Mapping[int, PartnerBank]
    alerts: Sequence[BtAlert]
    config: BacktestConfig = field(default_factory=BacktestConfig)
    accountant: PrivacyAccountant | None = None
    cache: BacktestCache | None = None

    def __post_init__(self):
        self.alerts = sorted(self.alerts, key=lambda a: (a.binding.t0, a.alert_id))

    def collect(self, node: Node, eps: float) -> tuple[list, list[bool], int]:
        """Put `node` to every eligible alert. Returns (answers, labels) of the replies, in alert-time order, and
        the number of alerts that were eligible."""
        canon = canonical(node)
        answers, labels, eligible = [], [], 0
        for a in self.alerts:
            bank = asked_bank(node, a.binding)
            if bank is None or bank not in self.banks:
                continue
            eligible += 1
            r = self.banks[bank].handle(Question(canon, eps, a.binding), a.binding.t0, purpose="cal")
            if isinstance(r, Reply):
                if self.accountant is not None and not is_exact(eps):
                    self.accountant.spend_cal(bank, eps, charge_keys(node, a.binding, bank))
                answers.append(r.value)
                labels.append(a.label)
        return answers, labels, eligible

    def run_question(self, program: str, eps: float, use_cache: bool = True) -> dict:
        node = parse(program)
        canon = canonical(node)
        if use_cache and self.cache is not None and (hit := self.cache.get(canon, eps)) is not None:
            return hit
        entry = self._evaluate(node, canon, eps)
        if self.cache is not None:
            self.cache.put(entry)
        return entry

    def _evaluate(self, node: Node, canon: str, eps: float) -> dict:
        cfg, kind = self.config, node.prim.kind
        answers, labels, eligible = self.collect(node, eps)
        n, n_l = len(answers), sum(labels)
        entry = {
            "program_canonical": canon, "eps": eps, "family": family(node), "kind": kind, "mode": cfg.mode,
            "n_eligible": eligible, "n_answers": n, "n_launder": n_l, "n_clean": n - n_l,
            "eligible": n >= cfg.min_answers and min(n_l, n - n_l) >= cfg.min_per_class,
            "bins": [], "lr": {}, "ceiling": None, "log_e": 0.0, "e_value": 1.0, "accepted": False,
        }  # fmt: skip
        if not entry["eligible"]:
            return entry
        bins = choose_bins(answers, cfg.max_bins, cfg.min_per_bin) if kind in ("count", "set_count") else ()
        est = estimate_lr(kind, answers, labels, eps, sensitivity(node), arity(node), cfg.alpha, bins)
        log_e = log_evalue(cells_for(kind, answers, bins), labels, cfg.shrink)
        entry.update(bins=list(bins), lr=est.lr, ceiling=est.ceiling, log_e=log_e,
                     e_value=math.exp(log_e) if log_e < 700 else math.inf,
                     cells={k: list(v) for k, v in est.counts.items()})  # fmt: skip
        return entry

    def run_batch(self, programs: Iterable[str], eps_levels: Iterable[float], shard: tuple[int, int] | None = None,
                  ) -> list[dict]:  # fmt: skip
        """Back-test every (program, eps), skipping hypotheses outside this shard (index, n_shards)."""
        out = []
        for p in programs:
            canon = canonical(parse(p))
            if shard is not None and shard_of(canon, shard[1]) != shard[0]:
                continue
            for e in eps_levels:
                out.append(self.run_question(p, e))
        return out


def accept(entries: list[dict], fdr: float = 0.10) -> list[dict]:
    """e-BH over a batch of back-test entries (all of them count towards m, eligible or not). Sets `accepted`."""
    flags = e_bh([e["e_value"] if e["eligible"] else 1.0 for e in entries], fdr)
    for e, f in zip(entries, flags, strict=True):
        e["accepted"] = bool(f) and e["eligible"]
    return entries


def assemble(cache: BacktestCache, fdr: float = 0.10) -> list[dict]:
    """After all shards are done: e-BH over everything in the cache. Returns the entries with `accepted` set."""
    return accept(cache.entries(), fdr)


def lr_table_rows(entries: Iterable[dict]) -> list[dict]:
    """Rows in the format of T3's `LRTable.from_rows` (hand-off #9): program_canonical, eps, accepted, lr, family, bins.

    Extra keys (ignored by `from_rows`): `kind`, `p_cell` = {cell: [P(cell | clean), P(cell | launder)]} measured on the
    noisy back-test answers, which the non-LLM planners' expected-information-gain needs."""
    rows = []
    for e in entries:
        if not e["eligible"]:
            continue
        p_cell = {k: [c[0] / max(e["n_clean"], 1), c[1] / max(e["n_launder"], 1)] for k, c in e["cells"].items()}
        rows.append(
            {"program_canonical": e["program_canonical"], "eps": e["eps"], "accepted": bool(e["accepted"]),
             "lr": e["lr"], "family": e["family"], "bins": e["bins"] or None, "kind": e["kind"],
             "p_cell": p_cell}  # fmt: skip
        )
    return rows


@dataclass
class OnlineAcceptor:
    """Test time: a brand-new question is back-tested on the training alerts and accepted by e-LOND, so the FDR
    bound of the acceptance rule holds as questions keep arriving (plan.md 6.8)."""

    backtester: Backtester
    elond: ELond = field(default_factory=lambda: ELond(0.10))
    decided: dict = field(default_factory=dict)

    def submit(self, program: str, eps: float) -> dict:
        """Decide a question once; asking again returns the same decision and does not use up another test."""
        key = (canonical(parse(program)), float(eps))
        if key not in self.decided:
            entry = dict(self.backtester.run_question(program, eps))
            entry["accepted"] = bool(entry["eligible"] and self.elond.test(entry["e_value"]))
            self.decided[key] = entry
        return self.decided[key]
