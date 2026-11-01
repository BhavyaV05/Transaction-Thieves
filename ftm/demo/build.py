"""Assemble a `Demo`: load the alert, record (live) or read (replay) one run per system, add the p0-only baseline.

Owner: T3 (Akshat).
"""

from __future__ import annotations

import json
from pathlib import Path

from ftm.demo.alerts import DemoRefusal, find_alert, load_alert, refuse_if_test
from ftm.demo.model import AlertView, ArmRun, Demo
from ftm.demo.replay import arm_from_run, label_from_run, load_table
from ftm.graph.agents.adapted import ADAPTED_SYSTEMS
from ftm.llm.profile import decision_threshold

DECISION = "configs/decision/headroom_v1.yaml"  # the fitted threshold of the production profile (ftm.llm.profile)
HIDDEN_SUMMARY = {"evidence", "partner_banks"}


def adapted_name(name: str) -> str:
    n = name if name.endswith("_adapted") else f"{name}_adapted"
    if n not in ADAPTED_SYSTEMS:
        raise DemoRefusal(f"unknown comparison arm {name!r}; choose from {sorted(ADAPTED_SYSTEMS)}")
    return n


def p0_arm(p0: float) -> ArmRun:
    return ArmRun(name="p0_only", label="p0 only (home bank alone)", p0=p0, final_p=p0, eps_spent=0.0, eps_budget=None)


def alert_view(data: str, alert_id: str, run_summary: dict | None, k: int, evidence_view: str | None, p0_fallback: float | None):
    """(AlertView, summary dict for routing). Degrades to p0-only when the alert data is not on this machine."""
    notes: list[str] = []
    try:
        ctx, label, used = load_alert(data, alert_id, k=k, evidence_view=evidence_view)
        split, _ = find_alert(data, alert_id)
    except DemoRefusal:
        raise
    except Exception as e:  # noqa: BLE001 - data missing on this machine: still show the recorded investigation
        notes.append(f"alert data not available here ({type(e).__name__}); showing the recorded run only")
        return AlertView(alert_id, "?", None, p0_fallback if p0_fallback is not None else float("nan"), notes=notes), {}
    if run_summary and run_summary.get("evidence_view") != used:
        notes.append(f"the run used evidence view {run_summary.get('evidence_view')!r}; this panel shows {used!r}")
    summary = {k_: v for k_, v in ctx.summary.items() if k_ not in HIDDEN_SUMMARY}
    return AlertView(alert_id, ctx.split, ctx.home_bank, ctx.p0, summary={k_: v for k_, v in summary.items() if k_ != "evidence"},
                     evidence=ctx.summary.get("evidence"), evidence_view=used, label=label, notes=notes), ctx.summary


def build_replay(run_dirs: list[Path], alert_id: str, *, data: str | None = None, backtest: Path | None = None, k: int = 40,
                 threshold: float | None = None, clear_below: float | None = None, threshold_source: str | None = None) -> Demo:
    refuse_if_test(alert_id)
    if threshold is None:  # the frozen decision config, not an illustrative number
        threshold, threshold_source = decision_threshold(DECISION), DECISION
    summaries = [json.loads((d / "summary.json").read_text(encoding="utf-8")) if (d / "summary.json").exists() else {} for d in run_dirs]
    data = data or summaries[0].get("data") or "hi_medium"
    ev = summaries[0].get("evidence_view")
    table = load_table(summaries[0], backtest) if backtest else None
    p0_guess = None
    av, routing_summary = alert_view(data, alert_id, summaries[0], k, ev, p0_guess)
    arms = []
    for d, sm in zip(run_dirs, summaries, strict=True):
        arms.append(arm_from_run(d, alert_id, summary_alert=routing_summary, table=table if table is not None else load_table(sm, None)))
    if av.p0 != av.p0:  # nan: no alert data, take the run's
        av.p0 = arms[0].p0
    if av.label is None:
        av.label = label_from_run(run_dirs[0], alert_id)
    return Demo(av, [*arms, p0_arm(av.p0)], threshold, clear_below, threshold_source)


def build_live(alert_id: str, *, data: str, llm: str, backtest: Path | None, compare: list[str], record_dir: Path, mode: str = "compose",
               k: int = 40, evidence_view: str | None = None, seed: int = 0, endpoint: str | None = None, model: str | None = None,
               eps_per_alert: float | None = None, threshold: float | None = None, clear_below: float | None = None,
               threshold_source: str | None = None) -> Demo:
    from ftm.demo.record import record_run

    refuse_if_test(alert_id)
    split, part = find_alert(data, alert_id)
    ctx, label, used = load_alert(data, alert_id, k=k, evidence_view=evidence_view)
    systems = ["planner", *[adapted_name(c) for c in compare]]
    dirs = []
    for s in systems:
        dirs.append(record_run(s, data=data, alert_id=alert_id, split=split, part=part, run_dir=Path(record_dir) / s, llm=llm,
                               backtest=backtest, mode=mode, k=k, evidence_view=used, seed=seed, endpoint=endpoint, model=model,
                               eps_per_alert=eps_per_alert))
    return build_replay(dirs, alert_id, data=data, backtest=backtest, k=k, threshold=threshold, clear_below=clear_below,
                        threshold_source=threshold_source)
