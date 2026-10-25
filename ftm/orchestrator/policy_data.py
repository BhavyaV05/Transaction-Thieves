"""Training rows for the learned policy: random rollouts on validation alerts, every affordable question asked at every state.

Owner: T2 (Bhavya).  See `ftm/planners/learned.py` for what the rows are and how they are used. This is the part that talks to
the partner banks directly (as the back-test does, charged to eps_cal), so it lives here and not under `ftm/planners`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import numpy as np

from ftm.backtest.engine import BtAlert
from ftm.banks.partner import Question
from ftm.fusion.fuse import Fusion, posterior
from ftm.planners.learned import FEATURES, Featurizer, PolicyData
from ftm.planners.pool import QuestionPool
from ftm.planners.session import MIN_EPS
from ftm.types import Reply


def collect_policy_data(banks: dict, alerts: Sequence[BtAlert], pool: QuestionPool, fusion: Fusion | None = None,
                        eps_per_alert: float = 2.0, depth: int = 3, max_options: int = 24, seed: int = 0,
                        one_per_family: bool = True, families: Sequence[str] | None = None) -> PolicyData:  # fmt: skip
    """Rows (state, question) -> realised Brier gain, from random rollouts on `alerts` (see the module docstring).
    A declined question or a timeout is a row with gain 0: the policy learns that some questions often go unanswered."""
    fams = list(families) if families is not None else sorted({c.family for c in pool})
    fz = Featurizer(fams)
    X, y, ids = [], [], []
    eps_cal, n_q, n_dec = 0.0, 0, 0
    for a in alerts:
        b = a.binding
        direction = "out" if b.sender_bank == b.home_bank else "in"
        bank = b.partner_bank
        if bank not in banks:
            continue
        options_all = pool.applicable(direction)
        rng = random.Random(f"{seed}|{a.alert_id}")
        lr_by_family: dict[str, float] = {}
        families_asked: set[str] = set()
        asked: set[str] = set()
        eps_left, p = eps_per_alert, float(a.p0)
        label = 1.0 if a.label else 0.0
        for _ in range(depth):
            avail = [c for c in options_all if c.cost <= eps_left + 1e-9 and c.program not in asked
                     and not (one_per_family and c.family in families_asked)]  # fmt: skip
            if eps_left + 1e-12 < MIN_EPS or not avail:
                break
            if len(avail) > max_options:
                avail = rng.sample(avail, max_options)
            answered = []
            for c in avail:
                resp = banks[bank].handle(Question(c.program, c.eps, b), b.t0, purpose="policy")
                n_q += 1
                if isinstance(resp, Reply):
                    eps_cal += resp.eps_spent if math.isfinite(resp.eps_spent) else 0.0
                    llr = c.log_lr(resp.value)
                    nxt = dict(lr_by_family)
                    if llr is not None:
                        nxt[c.family] = nxt.get(c.family, 0.0) + llr
                    p_after = posterior(float(a.p0), nxt, fusion)
                    gain = (p - label) ** 2 - (p_after - label) ** 2
                    answered.append((c, nxt, p_after))
                else:
                    n_dec += 1
                    gain = 0.0
                X.append(fz.row_from(p, float(a.p0), eps_left, len(families_asked), direction, bank, b.amount, c))
                y.append(gain)
                ids.append(a.alert_id)
            if not answered:
                break
            c, lr_by_family, p = rng.choice(answered)  # advance with one random answered question
            asked.add(c.program)
            families_asked.add(c.family)
            eps_left -= c.cost
    return PolicyData(np.array(X, dtype="f8").reshape(-1, len(FEATURES)), np.array(y, dtype="f8"), ids, fams, eps_cal, n_q, n_dec)
