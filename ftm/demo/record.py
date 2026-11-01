"""Record one alert through the normal run_llm_agent wiring (harness toolbox, gate, accountant, back-test fusion).

Owner: T3 (Akshat).

`--live` of the demo is: record a one-alert run with scripts/run_llm_agent.py, then replay it step by step. So the live
and the replay views are the same code, and a recorded run dir can be shown later with no GPU.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path

from ftm.demo.alerts import DemoRefusal, refuse_if_test, runner
from ftm.demo.mock import demo_responder


@contextlib.contextmanager
def _patched_mock(r, system: str, backtest: Path | None):
    """Install the demo's scripted responder for `system` in run_llm_agent's DEV_RESPONDERS while a mock run executes."""
    from ftm.llm.backtest_table import accepted_menu, load_backtest

    menu = accepted_menu(load_backtest(backtest, require="realistic").table) if backtest else []
    old = r.DEV_RESPONDERS.get(system)
    r.DEV_RESPONDERS[system] = demo_responder(menu)
    try:
        yield
    finally:
        if old is None:
            r.DEV_RESPONDERS.pop(system, None)
        else:
            r.DEV_RESPONDERS[system] = old


def record_run(system: str, *, data: str, alert_id: str, split: str, part: str | None, run_dir: Path, llm: str = "mock",
               backtest: Path | None = None, mode: str = "compose", k: int = 40, evidence_view: str | None = None,
               seed: int = 0, endpoint: str | None = None, model: str | None = None, eps_per_alert: float | None = None,
               extra: list[str] | None = None) -> Path:
    """Run `system` ("planner" or "<agent>_adapted") on this one alert and write a standard run dir. Returns `run_dir`."""
    refuse_if_test(alert_id, split)
    r = runner()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ids = run_dir / "alert_ids.txt"
    ids.write_text(alert_id + "\n", encoding="utf-8")
    argv = ["--system", system, "--alerts", split, "--data", data, "--n", "1", "--llm", llm, "--toolbox", "harness",
            "--alert-ids", str(ids), "--run-dir", str(run_dir), "--k", str(k), "--seed", str(seed), "--mode", mode]
    if part:
        argv += ["--val-part", part]
    if evidence_view:
        argv += ["--evidence-view", evidence_view]
    if backtest:
        argv += ["--backtest", str(backtest)]
    if endpoint:
        argv += ["--endpoint", endpoint]
    if model:
        argv += ["--model", model]
    if eps_per_alert is not None:
        argv += ["--eps-per-alert", str(eps_per_alert)]
    argv += list(extra or [])
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink):
            if llm == "mock":
                with _patched_mock(r, system, backtest):
                    r.main(argv)
            else:
                r.main(argv)
    except IndexError as e:  # run_llm_agent: no alert left after the askable filter
        raise DemoRefusal(f"{alert_id} cannot be investigated: no partner bank with a ledger holds anything about it "
                          "(the harness world only binds alerts a modelled partner can be asked about)") from e
    if not (run_dir / "runlog.parquet").exists():
        raise DemoRefusal(f"the run for {alert_id} wrote no run log")
    return run_dir


def temp_run_dir(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"demo-{prefix}-"))
