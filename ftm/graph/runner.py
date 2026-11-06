"""Run many alerts CONCURRENTLY with one toolbox session per worker.

Owner: T3 (Akshat).  Spec: LangGraph rewrite (wave 2).

Mechanism: a thread pool. An alert is the unit of work; each worker owns an investigator and a `SessionToolBox` (begin/end are
per alert, so the stateful session cannot be shared) built over the SAME read-only world: banks, ledgers, gate and accountant
(`share_session_toolbox`). LLM calls are I/O waits on the endpoint, so threads overlap them; the CPU parts are small. The shared
mutable objects are the accountant (per-alert and per-account spend, behind one lock here) and the gate's lazy retriever (idempotent).
Results come back sorted by alert order, so rows are those of a sequential run: bank answers are seeded per (alert, question), the
fusion is pure, and the only cross-alert coupling is the per-account eps cap, which can bind differently under another order
(the cap is 4x the per-alert budget, so it needs four full-budget alerts on one account).

`timing_sample` K: the first K alerts run one by one before the pool starts, for the reference per-alert latency (no contention).
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Worker:
    system: Any  # an Investigator
    tools: Any  # what the system sees (the session, or the library wrapper around it)
    session: Any = None  # the SessionToolBox, None for the dev toolbox


@dataclass
class AlertRun:
    index: int
    result: Any  # whatever `run_one(worker, item)` returned
    seconds: float
    worker: int
    stats: dict = field(default_factory=dict)


def share_session_toolbox(tools):
    """A new session over the same world as `tools` (a HarnessToolBox): shared banks / gate / accountant, own begin/end state."""
    return type(tools)(tools._h)  # noqa: SLF001 - Harness is the shared world


def lock_world(tools) -> None:
    """Serialise the shared accountant and gate (their counters are plain dicts) so concurrent sessions cannot lose an update."""
    h = tools._h  # noqa: SLF001
    lock = threading.RLock()
    for obj, names in ((h.accountant, ("can_spend", "spend", "can_spend_cal", "spend_cal", "remaining")), (h.gate, ("check", "lookup_rules"))):
        for n in names:
            fn = getattr(obj, n, None)
            if fn is not None:
                setattr(obj, n, _locked(fn, lock))


def _locked(fn, lock):
    def wrapped(*a, **kw):
        with lock:
            return fn(*a, **kw)

    return wrapped


def run_alerts(items: list, workers: list[Worker], run_one: Callable[[Worker, Any], Any], *, concurrency: int = 1,
               timing_sample: int = 0, stats_of: Callable[[Worker], dict] | None = None) -> tuple[list[AlertRun], dict]:
    """Run `run_one(worker, item)` on every item. Returns (runs in item order, info). `concurrency` workers (the list must have
    at least that many); the first `timing_sample` items run sequentially on workers[0] first. Per-alert seconds is the wall time of
    that alert's own run; `stats_of(worker)` (integer counters) is differenced around each alert (valid because a worker runs one
    alert at a time)."""
    n_workers = max(1, min(concurrency, len(workers)))
    runs: list[AlertRun | None] = [None] * len(items)

    def one(i: int, w: int) -> None:
        wk = workers[w]
        before = stats_of(wk) if stats_of else {}
        t0 = time.perf_counter()
        res = run_one(wk, items[i])
        sec = time.perf_counter() - t0
        after = stats_of(wk) if stats_of else {}
        runs[i] = AlertRun(i, res, sec, w, {k: after[k] - before[k] for k in after})

    k = min(timing_sample, len(items))
    for i in range(k):
        one(i, 0)
    ref = [runs[i].seconds for i in range(k)]  # type: ignore[union-attr]
    t0 = time.perf_counter()
    rest = list(range(k, len(items)))
    if n_workers == 1:
        for i in rest:
            one(i, 0)
    else:
        free: queue.Queue[int] = queue.Queue()
        for w in range(n_workers):
            free.put(w)

        def task(i: int) -> None:
            w = free.get()
            try:
                one(i, w)
            finally:
                free.put(w)

        with ThreadPoolExecutor(n_workers) as ex:
            for f in [ex.submit(task, i) for i in rest]:
                f.result()  # re-raises a worker's exception
    wall = time.perf_counter() - t0
    ref_med = sorted(ref)[len(ref) // 2] if ref else None
    info = {"concurrency": n_workers, "timing_sample": k, "timing_sample_median_seconds": ref_med,
            "concurrent_wall_seconds": wall, "concurrent_alerts": len(rest)}
    return [r for r in runs if r is not None], info
