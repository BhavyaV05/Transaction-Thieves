"""The end-to-end demo (ftm/demo, scripts/demo.py): rendering, HTML export, test-split refusal, replay from a synthetic trace."""

import io
import json
import sys
from pathlib import Path

import polars as pl
import pytest
from rich.console import Console

from ftm.demo.alerts import DemoRefusal, find_alert, refuse_if_test
from ftm.demo.build import adapted_name, p0_arm
from ftm.demo.gloss import bank_why, gloss
from ftm.demo.mock import demo_responder
from ftm.demo.model import AlertView, Demo, zone
from ftm.demo.render import export_html, render
from ftm.demo.replay import arm_from_run, bin_label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

PROG = "n_counterparties(ALERT.receiver, dir=in, window=before(72))"
SUMMARY = {"sender": "A0000001", "receiver": "A0000002", "sender_bank": 70, "receiver_bank": 220, "amount": 9500.0,
           "currency": "US Dollar", "format": "Wire", "time": "2022-09-05 12:00", "direction": "out"}


def make_run(tmp_path: Path, *, traces: bool = True, backtest_missing: bool = True) -> Path:
    """A synthetic one-alert run dir with the files run_llm_agent writes (one reply, one blocked ask, a stop)."""
    d = tmp_path / "run"
    d.mkdir()
    rows = [
        {"alert_id": "V000001", "event": "step", "step": 1, "bank": "220", "program_canonical": PROG, "eps": 1.0, "gate": "allowed",
         "response_kind": "reply", "answer_value": "7", "kind": "count", "latency_days": 0.5, "posterior": 0.6, "log_lr": 0.4054651},
        {"alert_id": "V000001", "event": "step", "step": 2, "bank": "220", "program_canonical": "passed_on(ALERT.receiver, window=after(24), min_frac=0.8)",
         "eps": 1.0, "gate": "blocked", "response_kind": None, "answer_value": None, "kind": None, "latency_days": None, "posterior": 0.6,
         "log_lr": None, "gate_reason": "no US rule permits it"},
        {"alert_id": "V000001", "event": "verdict", "probability": 0.6, "p0": 0.5, "home_bank": "70", "llm_calls": 3, "eps_spent": 1.0, "label": 1},
    ]
    pl.DataFrame(rows, strict=False).write_parquet(d / "runlog.parquet")
    if traces:
        tr = [{"step": 1, "rationale": "fan-in looks likely", "action": "compose", "bank_requested": 70},
              {"step": 2, "rationale": "check pass-through", "action": "compose"},
              {"step": 3, "action": "stop", "rationale": "enough"}]
        (d / "traces.jsonl").write_text(json.dumps({"alert_id": "V000001", "p0": 0.5, "trace": tr}) + "\n")
    (d / "per_alert.jsonl").write_text(json.dumps({"alert_id": "V000001", "label": 1, "p0": 0.5, "probability": 0.6}) + "\n")
    (d / "summary.json").write_text(json.dumps({"system": "ours_compose", "model": "mock", "eps_per_alert": 2.0, "data": "hi_medium"}))
    return d


def demo_of(arm) -> Demo:
    av = AlertView("V000001", "val", "70", 0.5, summary=SUMMARY, evidence="2022-09-05 11:00 | A1 -> A2 | 5 US Dollar | ACH | bank 70 -> bank 220\n"
                   "2022-09-05 12:00 | A0000001 -> A0000002 | 9,500.00 US Dollar | Wire | bank 70 -> bank 220 (outgoing) <- ALERT", label=1)
    return Demo(av, [arm, p0_arm(0.5)])


def text_of(demo: Demo, **kw) -> str:
    c = Console(file=io.StringIO(), width=130, force_terminal=False, color_system=None)
    render(demo, c, **kw)
    return c.file.getvalue()


def test_replay_from_synthetic_trace(tmp_path):
    arm = arm_from_run(make_run(tmp_path), "V000001", summary_alert=SUMMARY)
    assert [s.kind for s in arm.steps] == ["ask", "ask", "stop"]
    s1, s2, _ = arm.steps
    assert s1.thought == "fan-in looks likely" and s1.bank == "220" and "bank 220" in s1.bank_why and "re-routed" in s1.bank_why
    assert s1.lr == pytest.approx(1.5) and s1.post_before == 0.5 and s1.post_after == 0.6
    assert s1.eps_left == pytest.approx(1.0) and s1.eps_spent == 1.0 and s1.answer_value == 7
    assert s2.gate == "blocked" and s2.eps_left == pytest.approx(1.0) and s2.gate_reason == "no US rule permits it"
    assert arm.final_p == 0.6 and arm.p0 == 0.5 and arm.n_asked == 1


def test_render_text_has_every_panel(tmp_path):
    arm = arm_from_run(make_run(tmp_path), "V000001", summary_alert=SUMMARY)
    out = text_of(demo_of(arm), reveal_label=True)
    for needle in ("(a) The alert as the home bank sees it", "A0000002", "Step 1", "In English", "Routed to", "Compliance gate", "Privacy accountant",
                   "Likelihood ratio", "Posterior", "0.500 -> 0.600", "(c) The decision", "ESCALATE", "True label", "p0 only"):
        assert needle in out, needle
    assert "BLOCKED" in out and "no US rule permits it" in out


def test_label_hidden_by_default(tmp_path):
    arm = arm_from_run(make_run(tmp_path), "V000001", summary_alert=SUMMARY)
    assert "True label" not in text_of(demo_of(arm))


def test_replay_degrades_without_traces(tmp_path):
    arm = arm_from_run(make_run(tmp_path, traces=False), "V000001", summary_alert=SUMMARY)
    assert arm.steps[0].thought is None and any("traces.jsonl" in n for n in arm.notes)
    out = text_of(demo_of(arm))
    assert "not recorded" in out and "Step 1" in out


def test_compare_mode_and_pause(tmp_path):
    arm = arm_from_run(make_run(tmp_path), "V000001", summary_alert=SUMMARY)
    other = arm_from_run(make_run_copy(tmp_path), "V000001", summary_alert=SUMMARY)
    other.name, other.label = "react_adapted", "react_adapted"
    d = demo_of(arm)
    d.arms.insert(1, other)
    slept = []
    out = text_of(d, pause=0.5, sleep=slept.append)
    assert "side by side" in out and "Comparison on this alert" in out and "react_adapted" in out
    assert slept and all(s == 0.5 for s in slept)


def make_run_copy(tmp_path: Path) -> Path:
    import shutil

    dst = tmp_path / "run2"
    shutil.copytree(tmp_path / "run", dst)
    return dst


def test_html_export_is_self_contained(tmp_path):
    arm = arm_from_run(make_run(tmp_path), "V000001", summary_alert=SUMMARY)
    out = tmp_path / "demo.html"
    export_html(demo_of(arm), str(out))
    page = out.read_text(encoding="utf-8")
    assert page.startswith("<!DOCTYPE html>") and "V000001" in page and "Step 1" in page and "<style>" in page
    for ext in ("http://", "https://", "<link", "<script", "src="):
        assert ext not in page, ext


def test_test_ids_are_refused(capsys):
    with pytest.raises(DemoRefusal):
        refuse_if_test("F000123")
    with pytest.raises(DemoRefusal):
        refuse_if_test("V000001", "test")
    with pytest.raises(DemoRefusal):
        find_alert("hi_medium", "F000123")
    import demo as cli

    assert cli.main(["--alert", "F000123", "--llm", "mock"]) == 2
    assert cli.main(["--alert", "F000123", "--replay", "nowhere"]) == 2
    assert "never the test split" in capsys.readouterr().err


def test_zone_and_names():
    assert zone(0.05, 0.5, 0.1) == "auto-clear" and zone(0.3, 0.5, 0.1) == "human review" and zone(0.5, 0.5, 0.1) == "escalate"
    assert adapted_name("react") == "react_adapted" == adapted_name("react_adapted")
    with pytest.raises(DemoRefusal):
        adapted_name("nope")


def test_gloss_and_routing():
    assert "receiver account" in gloss(PROG) and "72 h up to the alert" in gloss(PROG)
    assert gloss("at_least(n_tx(ALERT.receiver, dir=in, window=before(72)), k=5)").startswith("Is the answer to this at least 5")
    assert gloss("junk(") == "(not a valid question)"
    assert "bank 220" in bank_why(PROG, SUMMARY, 70, 220)
    assert "home bank" in bank_why("n_counterparties(ALERT.sender, dir=in, window=before(72))", SUMMARY, 70, 70)


def test_bin_label_without_table():
    assert bin_label(None, 3) is None


def test_mock_responder_picks_by_info_per_eps():
    menu = [{"id": "B1", "program": "p1", "family": "f1", "eps": [2.0], "max_shift": 1.0},
            {"id": "B2", "program": "p2", "family": "f2", "eps": [1.0, 2.0], "max_shift": 0.8}]
    schema = {"title": "planner_step", "properties": {"hypothesis": {}, "action": {}, "question_id": {"anyOf": [{"enum": ["B1", "B2"]}, {"type": "null"}]},
                                                      "program": {"type": "null"}, "bank": {"anyOf": [{"enum": [5]}, {"type": "null"}]},
                                                      "eps": {"enum": [0.0, 1.0, 2.0]}, "rule_id": {"type": "null"}}}
    out = json.loads(demo_responder(menu)([{"role": "user", "content": "## Answers so far\n(none)\n## Output"}], {"json_schema": schema}))
    assert out["action"] == "select" and out["question_id"] == "B2" and out["eps"] == 1.0
    done = json.loads(demo_responder(menu)([{"role": "user", "content": "## Answers so far\np2, eps 1; p1, eps 2\n## Output"}], {"json_schema": schema}))
    assert done["action"] == "stop"


BT = ROOT / "data" / "backtest" / "hi_medium" / "7df6bad5337d"


@pytest.mark.skipif(not BT.exists() or not (ROOT / "data" / "alerts" / "hi_medium" / "val.parquet").exists(), reason="needs the val data and a back-test")
def test_live_mock_on_real_val_alert_then_replay(tmp_path):
    import demo as cli

    rec = tmp_path / "rec"
    c1 = Console(file=io.StringIO(), width=140, color_system=None)
    assert cli.main(["--alert", "V001549", "--llm", "mock", "--backtest", str(BT), "--record-dir", str(rec), "--pause", "0",
                     "--evidence-view", "", "--html", str(tmp_path / "a.html")], console=c1) == 0
    live = c1.file.getvalue()
    assert "Step 1" in live and "(c) The decision" in live
    c2 = Console(file=io.StringIO(), width=140, color_system=None)
    assert cli.main(["--alert", "V001549", "--replay", str(rec / "planner"), "--evidence-view", "", "--pause", "0"], console=c2) == 0
    assert "Step 1" in c2.file.getvalue()
    assert (tmp_path / "a.html").exists()


def test_mock_banner_and_production_threshold(tmp_path):
    from ftm.demo.build import DECISION, build_replay

    d = make_run(tmp_path)  # summary model "mock"
    demo = build_replay([d], "V000001", data="hi_medium")
    assert demo.threshold == pytest.approx(0.322699) and demo.clear_below is None and demo.threshold_source == DECISION
    out = text_of(demo)
    assert "SCRIPTED MOCK LLM" in out and "illustrative" not in out and DECISION in out
    assert zone(0.2, 0.3227) == "auto-clear" and zone(0.4, 0.3227) == "escalate"
