"""Back-testing with the cycle attempts held out (E12): do the accepted questions transfer to a typology they never saw?

Owner: T2 (Bhavya), with T1 (the held-out rows come from `ftm.data.io(heldout_cycles=True)`).  Spec: tracks.md 4.10 week 8.

`scripts/backtest.py --heldout-cycles` back-tests on a train split without any cycle attempt (their rows and alerts are
dropped). `transfer_report` then takes the accepted questions of that run and asks: on validation alerts, split into the
typology that was held out and the rest, do the learned LRs still separate launderers from clean cases? The measure is the
**separation**: the mean log-LR the table assigns to a laundering alert's answer minus the mean it assigns to a clean one
(0 = no information, larger = better; for a well-calibrated table it is the mean log-odds shift the question gives).
`compare_modes` (ftm/backtest/exact.py) compares the accepted sets and strengths of two runs, here with and without cycles.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from ftm.backtest.engine import BacktestConfig, Backtester, BtAlert
from ftm.backtest.lr import answer_key
from ftm.dsl.parser import parse


def separation(entry: dict, answers: Sequence, labels: Sequence[bool]) -> float | None:
    """Mean log-LR of the launderers' answers minus that of the clean ones, using the entry's table. None if either class
    is empty or an answer falls in no cell."""
    bins = entry.get("bins") or None
    vals = {True: [], False: []}
    for a, y in zip(answers, labels, strict=True):
        lr = entry["lr"].get(answer_key(a, bins))
        if lr is None or lr <= 0 or not math.isfinite(lr):
            continue
        vals[bool(y)].append(math.log(lr))
    if not vals[True] or not vals[False]:
        return None
    return sum(vals[True]) / len(vals[True]) - sum(vals[False]) / len(vals[False])


def transfer_report(entries: Sequence[dict], banks, subsets: Mapping[str, Sequence[BtAlert]],
                    config: BacktestConfig | None = None, min_per_class: int = 5) -> dict:  # fmt: skip
    """For every accepted entry and every named subset of alerts: the separation, and summary rows per subset."""
    accepted = [e for e in entries if e.get("accepted")]
    per_question, summary = [], {}
    for name, alerts in subsets.items():
        bt = Backtester(banks, list(alerts), config or BacktestConfig())
        seps = []
        for e in accepted:
            answers, labels, _ = bt.collect(parse(e["program_canonical"]), e["eps"])
            sep = separation(e, answers, labels) if min(sum(labels), len(labels) - sum(labels)) >= min_per_class else None
            per_question.append({"subset": name, "program_canonical": e["program_canonical"], "eps": e["eps"],
                                 "n": len(answers), "n_launder": int(sum(labels)), "separation": sep})  # fmt: skip
            if sep is not None:
                seps.append(sep)
        summary[name] = {"n_alerts": len(alerts), "n_questions_scored": len(seps),
                         "mean_separation": sum(seps) / len(seps) if seps else None,
                         "share_positive": sum(s > 0 for s in seps) / len(seps) if seps else None}  # fmt: skip
    return {"n_accepted": len(accepted), "subsets": summary, "per_question": per_question}
