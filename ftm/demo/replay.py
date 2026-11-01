"""Turn a recorded run (outputs/runs/<id>: runlog.parquet, traces.jsonl, per_alert.jsonl, summary.json) into an `ArmRun`.

Owner: T3 (Akshat).

What the replay reads, per field (a missing field is shown as "not recorded", never invented):
  runlog.parquet   step rows: step, bank, program_canonical, eps, gate, response_kind, answer_value, kind, latency_days, posterior,
                   log_lr, rule_id, gate_reason; verdict row: probability, llm_calls, eps_spent, p0, label, stop_reason
  traces.jsonl     per planner step: rationale / thought, action, bank_requested, error (proposals that never went out)
  summary.json     system, model, eps_per_alert, backtest namespace (the LR table, for the answer bin)
  per_alert.jsonl  label (None in blind runs), p0, probability
The compliance rule cited for an allowed question is not logged by the harness: it is re-derived from the gate config (marked).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ftm.demo.alerts import ROOT
from ftm.demo.gloss import bank_why, gloss
from ftm.demo.model import ArmRun, DemoStep

DEFAULT_EPS = 2.0


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()] if path.exists() else []


def _gate():
    from ftm.compliance.gate import Gate

    try:
        return Gate.from_config()
    except Exception:  # noqa: BLE001 - no config: the rule is simply not shown
        return None


def derive_rule(gate, home_bank, bank, program, eps, alert_id):
    """(rule_id, text) the gate would cite for this question, read-only (no accountant is attached)."""
    if gate is None:
        return None, None
    try:
        saved, gate.accountant = gate.accountant, None
        try:
            d = gate.check("ask_bank", alert_id=alert_id, home_bank=home_bank, bank=bank, program=program, eps=float(eps))
        finally:
            gate.accountant = saved
    except Exception:  # noqa: BLE001
        return None, None
    rule = gate.rules.get(d.rule_id) if d.rule_id else None
    return d.rule_id, (rule.text if rule else None)


def load_table(summary: dict, explicit: Path | None):
    """The back-test LR table of the run (or None): `explicit`, else data/backtest/<data>/<namespace> from the run's summary."""
    from ftm.llm.backtest_table import BacktestTableError, load_backtest

    cands = [explicit] if explicit else []
    bt = summary.get("backtest") or {}
    if bt.get("path"):
        cands.append(Path(bt["path"]))
    if bt.get("namespace") and summary.get("data"):
        cands.append(ROOT / "data" / "backtest" / summary["data"] / bt["namespace"])
    for c in cands:
        try:
            return load_backtest(c, require=None).table
        except (BacktestTableError, OSError, FileNotFoundError):
            continue
    return None


def bin_label(entry, value) -> str | None:
    """Readable answer bin of `value` under the table entry (yes / no, or [lo, hi))."""
    if entry is None:
        return None
    if isinstance(value, bool) or set(entry.lr) <= {"true", "false"}:
        return "yes" if value in (True, 1, "true", "True") else "no"
    edges = list(entry.bins or ())
    if not edges or not isinstance(value, (int, float)):
        return None
    i = sum(1 for e in edges if e <= value)
    lo = "-inf" if i == 0 else f"{edges[i - 1]:g}"
    hi = "inf" if i == len(edges) else f"{edges[i]:g}"
    return f"[{lo}, {hi})"


def _value(raw):
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def arm_from_run(run_dir: Path, alert_id: str, *, summary_alert: dict, table=None, label: str | None = None) -> ArmRun:
    """The `ArmRun` of one alert in a recorded run. `summary_alert`: aliased alert summary (banks, for the routing reason)."""
    import polars as pl

    run_dir = Path(run_dir)
    summ = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) if (run_dir / "summary.json").exists() else {}
    if table is None:
        table = load_table(summ, None)
    df = pl.read_parquet(run_dir / "runlog.parquet").filter(pl.col("alert_id") == alert_id)  # time-guard: not transactions
    if not df.height:
        raise SystemExit(f"alert {alert_id} is not in {run_dir} (alerts there: see {run_dir / 'alerts.jsonl'})")
    rows = df.filter(pl.col("event") == "step").sort("step").to_dicts()
    verdict = df.filter(pl.col("event") == "verdict").to_dicts()[0]
    per = next((r for r in _jsonl(run_dir / "per_alert.jsonl") if r.get("alert_id") == alert_id), {})
    trace_rows = next((r["trace"] for r in _jsonl(run_dir / "traces.jsonl") if r.get("alert_id") == alert_id), [])
    by_step = {t.get("step"): t for t in trace_rows if "step" in t}
    notes: list[str] = []
    if not trace_rows:
        notes.append("traces.jsonl has no entry for this alert: planner thoughts are not shown")
    p0 = float(verdict["p0"] if verdict.get("p0") is not None else per.get("p0", float("nan")))
    home = verdict.get("home_bank")
    budget = summ.get("eps_per_alert") if summ else None
    budget = DEFAULT_EPS if budget is None and not summ.get("no_noise") else budget
    if table is None:
        notes.append("back-test table not found: the answer bin comes from the logged LR only")
    gate = _gate()
    if gate is None:
        notes.append("gate config not found: cited rules are not shown")
    steps: list[DemoStep] = []
    left, before = budget, p0
    for r in rows:
        t = by_step.get(r["step"], {})
        prog, bank, eps = r["program_canonical"], r["bank"], r["eps"]
        value = _value(r.get("answer_value"))
        entry = table.get(prog, float(eps)) if (table is not None and prog is not None) else None
        gate_dec = r.get("gate") or "allowed"
        outcome = r.get("response_kind") if gate_dec == "allowed" else gate_dec
        lr = math.exp(r["log_lr"]) if r.get("log_lr") is not None else None
        used = lr is not None
        if lr is None and entry is not None and value is not None and outcome == "reply":
            lr = entry.lr_for(value)
        spent = float(eps) if outcome == "reply" else 0.0
        if left is not None:
            left -= spent
        rid, rtext = r.get("rule_id"), None
        derived = False
        if gate_dec == "allowed" and rid is None:
            rid, rtext = derive_rule(gate, home, bank, prog, eps, alert_id)
            derived = rid is not None
        elif rid and gate is not None and rid in gate.rules:
            rtext = gate.rules[rid].text
        after = r.get("posterior")
        steps.append(DemoStep(
            n=int(r["step"]), kind="ask", thought=t.get("rationale") or t.get("thought"), action=t.get("action"),
            program=prog, gloss=gloss(prog) if prog else None,
            bank=str(bank), bank_why=bank_why(prog, summary_alert, home, bank, t.get("bank_requested")) if summary_alert else None,
            eps=float(eps), gate=gate_dec, rule_id=rid, rule_text=rtext, rule_derived=derived, gate_reason=r.get("gate_reason"),
            eps_spent=spent, eps_left=left, outcome=outcome, answer_value=value, answer_kind=r.get("kind"),
            latency_days=r.get("latency_days"), bin_label=bin_label(entry, value), lr=lr, used=used,
            post_before=before, post_after=after if after is not None else before,
            note=t.get("error")))  # fmt: skip
        if after is not None:
            before = after
    asked = {r["step"] for r in rows}
    for tt in trace_rows:  # planner proposals that never went out (bad JSON, refused, duplicate ...)
        if tt.get("step") is not None and tt["step"] not in asked and tt.get("error"):
            steps.append(DemoStep(n=int(tt["step"]), kind="rejected", thought=tt.get("rationale") or tt.get("thought"),
                                  action=tt.get("action"), program=tt.get("program"), note=str(tt["error"])))
    steps.sort(key=lambda x: x.n)
    last = trace_rows[-1] if trace_rows else {}
    stop = verdict.get("stop_reason") or ("planner stopped" if last.get("action") == "stop" else None)
    if last.get("action") == "stop" or (last.get("choice") == 0 and "choice" in last):
        steps.append(DemoStep(n=int(last.get("step", len(rows) + 1)), kind="stop", thought=last.get("rationale") or last.get("thought"),
                              action="stop", post_before=before, post_after=before, eps_left=left))
    fin = verdict.get("probability")
    return ArmRun(name=str(summ.get("system") or verdict.get("system") or run_dir.name), label=str(summ.get("system") or run_dir.name),
                  p0=p0, final_p=float(fin if fin is not None else (per.get("probability") if per.get("probability") is not None else p0)),
                  steps=steps, eps_budget=budget, eps_spent=verdict.get("eps_spent"), llm_calls=verdict.get("llm_calls"),
                  stop_reason=stop, model=summ.get("model") or verdict.get("model"), mock=(summ.get("model") == "mock"), notes=notes)


def label_from_run(run_dir: Path, alert_id: str):
    per = next((r for r in _jsonl(Path(run_dir) / "per_alert.jsonl") if r.get("alert_id") == alert_id), {})
    return per.get("label")
