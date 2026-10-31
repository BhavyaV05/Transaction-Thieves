"""The raw-sharing baseline arm (H2): what sharing raw records buys, with no noise and no question budget.

Owner: T2 (Bhavya).  Spec: plan.md 7.1 ("Share raw records (3 banks, no noise)"), tracks.md 4.9.

THIS IS THE ONE PLACE WHERE ROWS LEAVE A PARTNER BANK, and only for this arm: `RawShareBank.share_records` hands the home
bank the subject's transactions, which `ftm/banks` never does (its static check stays green because this lives here), and
it asks the compliance gate for the `request_records` action first. The production gate blocks that action
(`allow_raw_share: false`); only a gate built with `allow_raw_share=True` allows it, and the harness toolbox has no such
tool. Nothing else imports this module.

What it does, per alert:
  1. three banks share: the bank that holds the alert's partner-side account (P) hands over that account's transfers up to
     its reply time; the two banks P's account dealt with most (other than the home bank) hand over their own accounts'
     transfers with it (second hop);
  2. the home bank computes, exactly, every expert question's answer on the union of those records, plus second-hop
     features (how the counterparties' own money moved);
  3. a ridge-penalised logistic model on those features and logit(p0), fitted on training alerts, gives the probability.
Unlimited eps, no noise, any feature: an upper limit for what the question route can hope for with the same banks.
H2 asks for the ratio of what questions gain over p0 to what this gains: `gain_ratio`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import polars as pl
from scipy.optimize import minimize

from ftm.backtest.engine import BtAlert
from ftm.banks.binding import AlertBinding
from ftm.banks.partner import PartnerBank, _seed
from ftm.compliance.gate import Gate, GateBlocked
from ftm.dsl.executor import ExecError, execute
from ftm.dsl.expert import expert_programs
from ftm.dsl.parser import parse
from ftm.fusion.fuse import logit, sigmoid
from ftm.types import AlertContext, Limits, QuestionLogEntry, Verdict

N_SHARING_BANKS = 3
LOOKBACK_H = 72 + 24 * 7  # the longest look-back of any expert question


class RawShareBank(PartnerBank):
    """A partner bank that, for the H2 arm only, hands over the transfers of named accounts."""

    def share_latency(self, alert_id: str) -> float:
        return self._latency(np.random.default_rng(_seed(self.seed, self.bank, alert_id, "raw-share")))

    def share_records(self, accounts: set[int], t0, cutoff) -> pl.DataFrame:
        """Transfers (ledger rows) involving `accounts` on this bank's side, from the look-back up to `cutoff`."""
        led = self._ledger
        mine = (pl.col("src_id").is_in(list(accounts)) & (pl.col("from_bank") == self.bank)) | (
            pl.col("dst_id").is_in(list(accounts)) & (pl.col("to_bank") == self.bank)
        )
        return led.filter(mine & (pl.col("timestamp") >= t0 - timedelta(hours=LOOKBACK_H)) & (pl.col("timestamp") <= cutoff))


def collect_records(banks: dict[int, RawShareBank], gate: Gate, b: AlertBinding, seed: int = 0
                    ) -> tuple[pl.DataFrame, list[int], float]:  # fmt: skip
    """The records the three sharing banks hand over for this alert, the bank ids, and the days until the last reply.
    Every hand-over is cleared with the gate first (`GateBlocked` if raw sharing is not enabled)."""
    p = b.partner_bank
    subject = b.receiver if b.sender_bank == b.home_bank else b.sender

    def clear(bank: int) -> None:
        d = gate.check("request_records", alert_id=b.alert_id, home_bank=b.home_bank, bank=bank)
        if not d.allowed:
            raise GateBlocked(d.reason)

    clear(p)
    lat = [banks[p].share_latency(b.alert_id)]
    first = banks[p].share_records({subject}, b.t0, b.t0 + timedelta(days=lat[0]))
    cps = first.select(pl.when(pl.col("src_id") == subject).then(pl.col("to_bank")).otherwise(pl.col("from_bank")).alias("cb"))["cb"]
    ranked = [bk for bk, _ in Counter(int(x) for x in cps.to_list()).most_common() if bk not in (b.home_bank, p) and bk in banks]
    used, frames = [p], [first]
    neighbours = set(first["src_id"].to_list()) | set(first["dst_id"].to_list())
    neighbours.discard(subject)
    for bk in ranked[: N_SHARING_BANKS - 1]:
        clear(bk)
        lat.append(banks[bk].share_latency(b.alert_id))
        frames.append(banks[bk].share_records(neighbours, b.t0, b.t0 + timedelta(days=lat[-1])))
        used.append(bk)
    return pl.concat(frames).unique(maintain_order=True), used, max(lat)


# --- features ---------------------------------------------------------------------------------------------------
def _scalar(v) -> float:
    return float(v)


def feature_programs(slot: str = "receiver") -> list[str]:
    return [p["program"] for p in expert_programs(slot) if not p["program"].startswith("n_same_entity")]


def raw_features(records: pl.DataFrame, b: AlertBinding, bank: int, days: float, programs: list[str]) -> np.ndarray:
    """Exact answers of every expert question on the shared records, plus second-hop features."""
    subject = b.receiver if b.sender_bank == b.home_bank else b.sender
    ctx = b.exec_context(bank, b.t0 + timedelta(days=days))
    ctx = type(ctx)(**{**ctx.__dict__, "receiver": subject, "sender": subject})  # the subject is the partner-side account
    out = []
    for prog in programs:
        try:
            out.append(_scalar(execute(parse(prog), records, ctx)))
        except ExecError:
            out.append(0.0)
    hop = records.filter((pl.col("src_id") != subject) & (pl.col("dst_id") != subject))
    nbr_out = hop.group_by("src_id").agg(pl.col("amount_paid").sum().alias("o"), pl.len().alias("no"))
    nbr_in = hop.group_by("dst_id").agg(pl.col("amount_received").sum().alias("i"))
    j = nbr_out.join(nbr_in, left_on="src_id", right_on="dst_id", how="full", coalesce=True).fill_null(0.0)
    n = j.height
    out += [float(n), float(j["no"].sum() if n else 0), float(j["o"].max() if n else 0.0) / (b.amount + 1.0),
            float(j["i"].max() if n else 0.0) / (b.amount + 1.0), float(hop.height)]  # fmt: skip
    return np.log1p(np.abs(np.array(out))) * np.sign(out)


@dataclass
class RawShareModel:
    ridge: float = 5.0
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    w: np.ndarray | None = None
    c: float = 0.0
    a: float = 1.0

    def fit(self, p0, X, y) -> RawShareModel:
        X, y, z0 = np.asarray(X, float), np.asarray(y, float), logit(np.asarray(p0, float))
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        D = np.column_stack([z0, Z, np.ones(len(y))])
        pen = np.r_[0.0, np.full(Z.shape[1], self.ridge), 0.0]  # a (on logit p0) and the intercept are free
        n = len(y)

        def loss(th):
            z = D @ th
            ll = np.logaddexp(0.0, z) - y * z
            g = D.T @ (sigmoid(z) - y) / n + pen * th / n
            return float(ll.mean() + 0.5 * (pen * th * th).sum() / n), g

        th0 = np.zeros(D.shape[1])
        th0[0] = 1.0
        res = minimize(loss, th0, jac=True, method="L-BFGS-B")
        self.a, self.w, self.c = float(res.x[0]), res.x[1:-1], float(res.x[-1])
        return self

    def predict(self, p0, X) -> np.ndarray:
        Z = (np.asarray(X, float) - self.mu) / self.sd
        return sigmoid(self.a * logit(np.asarray(p0, float)) + Z @ self.w + self.c)


RIDGES = (5.0, 50.0, 500.0, 5000.0)


def select_ridge(p0, X, y, ridges=RIDGES, folds: int = 4) -> float:
    """The ridge with the lowest Brier score under forward-chaining CV, with the rows in time order: cut them into `folds` + 1
    consecutive chunks, fit on the first i chunks and score chunk i (i = 1..folds). Time order, not a random split, because the
    model is applied to later alerts than it is fitted on (a base-rate shift a random split would hide). Ties go to the larger ridge."""
    p0, X, y = np.asarray(p0, float), np.asarray(X, float), np.asarray(y, float)
    edges = [round(i * len(y) / (folds + 1)) for i in range(folds + 2)]
    best, best_score = ridges[-1], np.inf
    for ridge in sorted(ridges, reverse=True):
        score = 0.0
        for i in range(1, folds + 1):
            a, b = slice(0, edges[i]), slice(edges[i], edges[i + 1])
            m = RawShareModel(ridge=ridge).fit(p0[a], X[a], y[a])
            score += float(np.mean((m.predict(p0[b], X[b]) - y[b]) ** 2))
        if score < best_score - 1e-12:
            best, best_score = ridge, score
    return best


@dataclass
class RawShareArm:
    """Fit on training alerts, then investigate: `fit` and `investigate` are the whole arm."""

    banks: dict[int, RawShareBank]
    gate: Gate
    seed: int = 0
    model: RawShareModel = field(default_factory=RawShareModel)
    bindings: dict[str, AlertBinding] = field(default_factory=dict)
    ridges: tuple | None = None  # candidate ridges to choose from by time-ordered CV; None keeps `model.ridge`

    def features(self, alert_binding: AlertBinding):
        slot = "receiver" if alert_binding.sender_bank == alert_binding.home_bank else "sender"
        records, used, days = collect_records(self.banks, self.gate, alert_binding, self.seed)
        return raw_features(records, alert_binding, alert_binding.partner_bank, days, feature_programs(slot)), used

    def fit(self, train: list[BtAlert]) -> RawShareArm:
        train = sorted(train, key=lambda a: a.binding.t0)
        X = np.array([self.features(a.binding)[0] for a in train])
        p0, y = [a.p0 for a in train], [float(a.label) for a in train]
        if self.ridges:
            self.model.ridge = select_ridge(p0, X, y, self.ridges)
        self.model.fit(p0, X, y)
        return self

    def investigate(self, alert: AlertContext, tools, limits: Limits) -> Verdict:
        b = self.bindings[alert.alert_id]
        x, used = self.features(b)
        p = float(self.model.predict([alert.p0], x[None, :])[0])
        log = [QuestionLogEntry(step=i + 1, bank=str(bk), program="request_records", eps=0.0) for i, bk in enumerate(used)]
        return Verdict(alert.alert_id, p, questions_log=log)


def gain_ratio(brier_p0: float, brier_questions: float, brier_raw: float) -> float:
    """H2: the share of the raw-sharing gain over the one-bank score that the question route reaches."""
    denom = brier_p0 - brier_raw
    return float("nan") if denom <= 0 else (brier_p0 - brier_questions) / denom
