"""SimPy orchestration (C15): one investigation = one SimPy process, alerts overlapping in simulated time.

Owner: T2 (Bhavya).  Spec: tracks.md 4.8 ("one investigation = one SimPy process; sequential and batched modes").

The harness keeps each alert's own clock (`HarnessToolBox.elapsed_days`, days since the alert): partner replies take the
banks' latencies, a wave costs its slowest answer. This module puts many alerts on one shared clock. Alert i arrives at
`i * arrival_gap_days`; its process runs the investigation (the planner's work is instantaneous in simulated time, the
waiting is the banks') and then holds for the days it took. What that adds to the per-alert latency is the picture of an
analyst team working many alerts at once: how many investigations are open at the same time, when the last one finishes.
Gate and accountant are shared and are touched in arrival order, so the per-account cap sees the alerts in the order they
arrive, as it would live.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import simpy

from ftm.backtest.engine import BtAlert
from ftm.banks.binding import alert_context
from ftm.harness.toolbox import Harness
from ftm.types import Limits, Verdict


@dataclass
class SimResult:
    verdicts: list[Verdict] = field(default_factory=list)
    started: dict = field(default_factory=dict)  # alert_id -> sim time the investigation began
    finished: dict = field(default_factory=dict)  # alert_id -> sim time it ended
    latency_days: dict = field(default_factory=dict)  # alert_id -> days it took
    max_in_flight: int = 0

    @property
    def makespan_days(self) -> float:
        return max(self.finished.values(), default=0.0)


def run_simpy(planner, alerts: Sequence[BtAlert], harness: Harness, limits: Limits, arrival_gap_days: float = 0.05,
              split: str = "val") -> SimResult:  # fmt: skip
    env = simpy.Environment()
    tools = harness.toolbox()
    system = tools.wrap(planner)
    res = SimResult()
    in_flight = [0]

    def investigation(a: BtAlert, delay: float):
        yield env.timeout(delay)  # the alert arrives
        res.started[a.alert_id] = env.now
        in_flight[0] += 1
        res.max_in_flight = max(res.max_in_flight, in_flight[0])
        v = system.investigate(alert_context(a.binding, a.p0, split), tools, limits)  # instantaneous in sim time
        took = tools.last_elapsed_days  # the banks' waiting, summed over sequential asks and maxed over waves
        yield env.timeout(took)
        in_flight[0] -= 1
        res.verdicts.append(v)
        res.finished[a.alert_id] = env.now
        res.latency_days[a.alert_id] = took

    for i, a in enumerate(alerts):
        env.process(investigation(a, i * arrival_gap_days))
    env.run()
    return res
