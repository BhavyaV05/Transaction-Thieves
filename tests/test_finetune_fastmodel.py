"""T3 weeks 7-10: distillation filter, SFT script dry-run, fast model (logprobs, JSON baseline), temperature
scaling, conformal cascade."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from ftm.fastmodel.calibration import fit_temperature, softmax_t, top_label_ece
from ftm.fastmodel.cascade import ConformalCascade, ConformalSet, evaluate
from ftm.fastmodel.decisions import DECISIONS, DecisionExample, messages, to_sft_example
from ftm.fastmodel.model import JsonDecisionModel, LogprobFastModel, label_logps
from ftm.fusion.fuse import Fusion
from ftm.harness.interface import ScriptedToolBox
from ftm.llm.client import ChatResponse
from ftm.llm.fusion_adapter import FusedLR
from ftm.llm.investigation import LREntry, LRTable, NaiveLRFusion
from ftm.llm.mock import MockLLM
from ftm.planners.llm import LLMPlanner
from ftm.planners.llm_distill import Teacher, distill_examples
from ftm.types import AlertContext, Limits, Reply

ROOT = Path(__file__).resolve().parents[1]
Q = "passed_on(ALERT.receiver, window=after(24), min_frac=0.8)"


# -- distillation -------------------------------------------------------------------------------------------
def trace_rec(step, action="compose", answer=True, outcome="reply", canonical=Q, eps=0.5, error=None):
    text = json.dumps({"action": action, "program": canonical if action == "compose" else None})
    r = {"step": step, "messages": [{"role": "user", "content": f"s{step}"}], "text": text, "action": action,
         "outcome": outcome, "canonical": canonical, "eps": eps, "bank": 220, "answer_value": answer,
         "answer_kind": "bool"}  # fmt: skip
    if error:
        r["error"] = error
    return r


TEACHER = Teacher("qwen3-27b", "a" * 40)
TABLE = LRTable([LREntry(Q, 0.5, True, {"true": 4.0, "false": 0.25}, family="f")])


def test_distill_keeps_accepted_questions_that_move_toward_label():
    # p0 = 0.2; "true" has LR 4 (posterior 0.5), "false" has LR 0.25
    traces = [
        {"alert_id": "T1", "p0": 0.2, "trace": [trace_rec(1), trace_rec(2, answer=False), trace_rec(3, "stop")]},
        {"alert_id": "T2", "p0": 0.2, "trace": [trace_rec(1, answer=False), trace_rec(2, "stop")]},  # label 0: kept
        {"alert_id": "T3", "p0": 0.2, "trace": [trace_rec(1, canonical="fan_in(x)"), trace_rec(2, "stop")]},  # rejected q
        {"alert_id": "V1", "p0": 0.2, "trace": [trace_rec(1)]},  # val: not in splits
    ]
    alerts = {"T1": {"split": "train", "label": 1}, "T2": {"split": "train", "label": 0},
              "T3": {"split": "train", "label": 1}, "V1": {"split": "val", "label": 1}}  # fmt: skip
    fusion = FusedLR(Fusion("naive"), TABLE)
    kw = {"fusion": fusion, "teacher": TEACHER, "filter": "realised"}  # the E16 v1 filter
    ex, counts = distill_examples(traces, alerts, accepted=lambda c, e: c == Q, **kw)
    kept = [(e["alert_id"], e["step"]) for e in ex]
    assert kept == [("T1", 1), ("T1", 3), ("T2", 1), ("T2", 2)]  # T1 step 2 moved away; T3 has no kept question
    assert ex[0]["messages"][-1] == {"role": "assistant", "content": json.dumps(json.loads(traces[0]["trace"][0]["text"]), separators=(",", ":"))}
    assert counts["kept_questions"] == 2 and counts["kept_stops"] == 2
    assert all(e["teacher"] == {"model": "qwen3-27b", "revision": "a" * 40} for e in ex) and counts["teacher"] == ex[0]["teacher"]
    with pytest.raises(ValueError):
        distill_examples(traces, alerts, accepted=lambda c, e: True, splits=("train", "test"), **kw)


def test_distill_posteriors_follow_the_given_fusion_not_the_trace():
    # the same answer moves the posterior toward label 1 under naive fusion, away from it under a flipped fitted one
    traces = [{"alert_id": "T1", "p0": 0.2, "trace": [trace_rec(1)]}]
    alerts = {"T1": {"split": "train", "label": 1}}
    kw = {"accepted": lambda c, e: True, "teacher": TEACHER, "filter": "realised"}
    flipped = FusedLR(Fusion("shared", ["f"], a=1.0, w=np.array([-1.0]), c=0.0, fitted=True), TABLE)
    assert len(distill_examples(traces, alerts, fusion=FusedLR(Fusion("naive"), TABLE), **kw)[0]) == 1
    assert distill_examples(traces, alerts, fusion=flipped, **kw)[0] == []


def test_distill_refuses_missing_acceptance_placeholder_fusion_and_old_traces():
    traces = [{"alert_id": "T1", "p0": 0.2, "trace": [trace_rec(1)]}]
    alerts = {"T1": {"split": "train", "label": 1}}
    good = FusedLR(Fusion("naive"), TABLE)
    with pytest.raises(ValueError, match="realistic"):
        distill_examples(traces, alerts, fusion=good, teacher=TEACHER, filter="realised")
    with pytest.raises(ValueError, match="placeholder"):
        distill_examples(traces, alerts, lambda c, e: True, fusion=NaiveLRFusion(TABLE), teacher=TEACHER, filter="realised")
    with pytest.raises(ValueError, match="no p0"):
        distill_examples([{"alert_id": "T1", "trace": traces[0]["trace"]}], alerts, lambda c, e: True, fusion=good, teacher=TEACHER, filter="realised")
    with pytest.raises(ValueError):
        Teacher("m", "")


def _alert_msgs(i, eps_left=2.0):
    summ = {"home_bank": "70", "sender_bank": 220, "receiver_bank": 70, "counterparty_bank": 220}
    return [{"role": "user", "content": f"## Alert\n{json.dumps(summ)}\n\n## Budget\neps left: {eps_left:g}; LLM calls left: 9\n"}]


def test_distill_expected_filter_is_cross_fitted_right_bank_and_adds_stops():
    # sender is at bank 220, receiver at home 70: ALERT.sender questions go to 220, ALERT.receiver ones to nobody
    q_s = "new_account(ALERT.sender, days=2)"
    table = LRTable([LREntry(q_s, 0.5, True, {"true": 4.0, "false": 0.25}, family="new_account")])
    fusion = FusedLR(Fusion("naive"), table)

    def tr(aid, label_helps, bank=220):
        r = trace_rec(1, canonical=q_s, answer=label_helps, eps=0.5)
        r["messages"], r["bank"] = _alert_msgs(1), bank
        r["text"] = json.dumps({"hypothesis": {"typology": "x", "rationale": "r"}, "action": "compose", "question_id": None,
                                "program": q_s, "bank": bank, "eps": 0.5, "rule_id": None})
        return {"alert_id": aid, "p0": 0.4, "trace": [r]}

    # 40 alerts over time; "true" answers on label-1 alerts (gain > 0), so the family has positive held-out gain
    traces = [tr(f"T{i}", i % 2 == 0) for i in range(40)]
    alerts = {f"T{i}": {"split": "train", "label": int(i % 2 == 0), "bank": 70, "timestamp": f"2022-09-01T{i // 2:02d}:00"} for i in range(40)}
    wrong = tr("TW", True, bank=1588)
    alerts["TW"] = {"split": "train", "label": 1, "bank": 70, "timestamp": "2022-09-02T00:00"}
    kw = {"fusion": fusion, "teacher": TEACHER, "table": table, "folds": 4, "min_family_support": 5}
    ex, counts = distill_examples([*traces, wrong], alerts, accepted=lambda c, e: True, **kw)
    assert counts["kept_questions"] == 40 and counts["wrong_bank"] == 1 and counts["filter"] == "expected"
    assert not any(e["alert_id"] == "TW" and e["kind"] != "stop" for e in ex)  # the wrong-bank step is never a question example
    # flipped labels: the family now hurts on held-out folds -> nothing kept, whatever each alert's own outcome
    bad = {k: {**v, "label": 1 - v["label"]} for k, v in alerts.items()}
    assert distill_examples(traces, bad, accepted=lambda c, e: True, **kw)[1]["kept_questions"] == 0
    # stop examples: p0 far from the threshold and the best accepted question cannot cross it
    far = [{**t, "p0": 0.001} for t in traces[:3]]
    ex2, c2 = distill_examples(far, alerts, accepted=lambda c, e: True, stop_threshold=0.5, **kw)
    assert c2["stop_examples"] == 3 and all(json.loads(e["messages"][-1]["content"])["action"] == "stop" for e in ex2 if e["kind"] == "stop")
    s = json.loads(next(e for e in ex2 if e["kind"] == "stop")["messages"][-1]["content"])
    assert s["program"] is None and s["bank"] is None and s["eps"] == 0.0


def test_planner_keep_prompts_records_what_distillation_needs():
    step = json.dumps({"hypothesis": {"typology": "fan_in", "rationale": "r"}, "action": "compose", "question_id": None,
                       "program": Q, "bank": 220, "eps": 0.5, "rule_id": None})  # fmt: skip
    p = LLMPlanner(MockLLM(script=[step]), keep_prompts=True)
    alert = AlertContext("T1", "70", "2022-09-05T12:00:00", "train", 0.2, {"partner_banks": [220], "evidence": "e"})
    p.investigate(alert, ScriptedToolBox([Reply("220", "bool", True, 0.5, 0.2)]), Limits(eps_per_alert=0.5))
    rec = p.last_trace[0]
    assert {"messages", "text", "posterior_before", "posterior_after", "canonical", "eps", "outcome", "answer_value", "bank"} <= set(rec)
    assert rec["answer_value"] == Reply("220", "bool", True, 0.5, 0.2).value


# -- SFT script -----------------------------------------------------------------------------------------------
def _script():
    spec = importlib.util.spec_from_file_location("train_planner", ROOT / "scripts" / "train_planner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("cfg", ["sft_planner.yaml", "sft_fast.yaml"])
def test_sft_configs_and_dry_run(tmp_path, cfg, capsys):
    mod = _script()
    c = mod.load_config(ROOT / "configs" / "llm" / cfg)
    assert len(c["seeds"]) >= 3 and len(c["revision"]) == 40 and c["output_dir"].startswith("models/")
    data = tmp_path / "d.jsonl"
    ex = to_sft_example(DecisionExample("stop", "ctx", "no", "A1"))
    data.write_text(json.dumps(ex) + "\n")
    assert mod.main(["train", "--config", str(ROOT / "configs" / "llm" / cfg), "--dataset", str(data), "--dry-run"]) == 0
    assert '"examples": 1' in capsys.readouterr().out
    data.write_text(json.dumps({"messages": [{"role": "user", "content": "x"}]}) + "\n")
    with pytest.raises(SystemExit):
        mod.main(["train", "--config", str(ROOT / "configs" / "llm" / cfg), "--dataset", str(data), "--dry-run"])


# -- decisions + fast models ----------------------------------------------------------------------------------
def test_decisions_tokens_and_sft_example():
    for d in DECISIONS.values():
        assert len(d.labels) == len(d.tokens) == len(set(d.tokens))
        assert d.safe_default is None or d.safe_default in d.labels
    ex = to_sft_example(DecisionExample("typology", "ctx", "cycle"))
    assert ex["messages"][-1]["content"] == DECISIONS["typology"].tokens[DECISIONS["typology"].labels.index("cycle")]
    assert "C = cycle" in messages(DecisionExample("typology", "ctx"))[1]["content"]
    with pytest.raises(ValueError):
        to_sft_example(DecisionExample("stop", "ctx", "maybe"))


class LogprobClient:
    def __init__(self, top):
        self.top, self.requests = top, []

    def chat(self, messages, **kw):
        self.requests.append(kw)
        raw = {"choices": [{"message": {"content": "yes"}, "logprobs": {"content": [{"top_logprobs": self.top}]}}]}
        return ChatResponse(text="yes", tokens_in=10, tokens_out=1, latency_s=0.01, model="qwen3-4b", raw=raw)


def test_logprob_fast_model_reads_label_tokens_and_temperature():
    top = [{"token": "yes", "logprob": np.log(0.6)}, {"token": " Yes", "logprob": np.log(0.2)},
           {"token": "no", "logprob": np.log(0.1)}, {"token": "maybe", "logprob": np.log(0.1)}]  # fmt: skip
    assert label_logps(top, "stop") == pytest.approx(np.log([0.8, 0.1]))  # " Yes" and "yes" add (logsumexp)
    client = LogprobClient(top)
    p = LogprobFastModel(client).predict_proba(DecisionExample("stop", "ctx"))
    assert p["yes"] == pytest.approx(8 / 9) and client.requests[0]["max_tokens"] == 1
    assert client.requests[0]["extra"]["logprobs"] is True
    hot = LogprobFastModel(client, temperature=5.0).predict_proba(DecisionExample("stop", "ctx"))
    assert 0.5 < hot["yes"] < p["yes"]


def test_json_baseline_spreads_rest_and_falls_back_to_uniform():
    m = JsonDecisionModel(MockLLM(script=['{"label": "cycle", "confidence": 0.6}', "garbage"]))
    p = m.predict_proba(DecisionExample("typology", "ctx"))
    assert p["cycle"] == pytest.approx(0.6) and p["fan_in"] == pytest.approx(0.05) and sum(p.values()) == pytest.approx(1)
    u = m.predict_proba(DecisionExample("typology", "ctx"))
    assert len(set(u.values())) == 1


# -- calibration ----------------------------------------------------------------------------------------------
def test_temperature_scaling_recovers_overconfidence():
    rng = np.random.default_rng(0)
    n, k, t_true = 4000, 3, 2.5
    true_logits = rng.normal(0, 1.5, (n, k))
    y = np.array([rng.choice(k, p=softmax_t(z[None], 1.0)[0]) for z in true_logits])
    over = true_logits * t_true  # model is overconfident by a factor t_true
    t = fit_temperature(over, y)
    assert t == pytest.approx(t_true, rel=0.15)
    assert top_label_ece(softmax_t(over, t), y) < top_label_ece(softmax_t(over, 1.0), y)


# -- cascade --------------------------------------------------------------------------------------------------
def synthetic(n, sharp, rng, decision="stop"):
    probs, labels = [], []
    for _ in range(n):
        y = "yes" if rng.random() < 0.5 else "no"
        conf = rng.uniform(0.5, 1.0) if rng.random() < sharp else rng.uniform(0.5, 0.6)
        right = rng.random() < conf
        p_yes = conf if (y == "yes") == right else 1 - conf
        probs.append({"yes": p_yes, "no": 1 - p_yes})
        labels.append(y)
    return probs, labels


def test_conformal_sets_cover_at_1_minus_alpha():
    rng = np.random.default_rng(1)
    cal_p, cal_y = synthetic(2000, 0.7, rng)
    cs = ConformalSet(alpha=0.1).fit(["stop"] * 2000, cal_p, cal_y)
    test_p, test_y = synthetic(4000, 0.7, rng)
    cover = np.mean([y in cs.predict_set("stop", p) for p, y in zip(test_p, test_y, strict=True)])
    assert cover >= 0.88
    assert ConformalSet(alpha=0.1).fit(["stop"] * 3, cal_p[:3], cal_y[:3]).q["stop"] == 1.0  # too few points


def test_cascade_routing_and_report():
    rng = np.random.default_rng(2)
    fp, fy = synthetic(1500, 0.6, rng)
    lp, ly = synthetic(1500, 0.95, rng)
    alpha = 0.1
    casc = ConformalCascade(ConformalSet(alpha).fit(["stop"] * 1500, fp, fy), ConformalSet(alpha).fit(["stop"] * 1500, lp, ly))
    sure = casc.decide("stop", {"yes": 0.99, "no": 0.01}, lambda: pytest.fail("8B must not be called"))
    assert sure.source == "fast" and sure.answer == "yes"
    calls = []
    unsure = casc.decide("stop", {"yes": 0.5, "no": 0.5}, lambda: calls.append(1) or {"yes": 0.01, "no": 0.99})
    assert unsure.source == "large" and unsure.answer == "no" and calls == [1]
    dflt = casc.decide("stop", {"yes": 0.5, "no": 0.5}, lambda: {"yes": 0.5, "no": 0.5})
    assert dflt.source == "default" and dflt.answer == DECISIONS["stop"].safe_default
    tp, ty = synthetic(2000, 0.6, rng)
    big = [casc.decide("stop", p, lambda: {"yes": 0.5, "no": 0.5}) for p in tp]
    rep = evaluate(big, ty, alpha)["stop"]
    assert 0 < rep["kept_fast"] < 1 and rep["kept_fast"] + rep["to_large"] + rep["to_default"] == pytest.approx(1)
    assert rep["within_bound"]
