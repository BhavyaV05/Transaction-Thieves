"""Small-decision dataset (hand-off #13): format, labels, no leakage, determinism."""

import importlib.util
import json
import re
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from ftm.backtest.engine import BacktestConfig
from ftm.backtest.synth import make_world
from ftm.compliance.gate import ALIAS_RE, DATE_RE, RAW_ACCOUNT_RE, Gate
from ftm.dsl.parser import parse
from ftm.fusion.fuse import logit
from ftm.orchestrator.decision_set import (
    DECISION_LABELS,
    DEFAULT_MIX,
    DecisionSetBuilder,
    evidence_text,
    rule_label,
    stop_label,
    write_set,
)
from ftm.orchestrator.runner import pool_from_backtest
from ftm.planners.pool import Candidate
from ftm.safety import injection as inj


@pytest.fixture(scope="module")
def env():
    w = make_world(150, seed=4)
    alerts = sorted(w.alerts, key=lambda a: a.binding.t0)
    cut = int(len(alerts) * 0.6)
    alerts = [replace(a, split="train" if i < cut else "val") for i, a in enumerate(alerts)]
    cfg = BacktestConfig(min_answers=40, min_per_class=6, min_per_bin=10)
    pool, _ = pool_from_backtest(w.banks, alerts[:cut], ["n_counterparties(ALERT.receiver, dir=in, window=before(72))"], (1.0, 2.0), cfg)
    assert len(pool) >= 1
    rows = list(DecisionSetBuilder(w.ledgers, w.banks, pool, 0.3, seed=1).build(alerts))
    return w, alerts, pool, rows


def test_every_alert_gets_the_configured_mix_and_labels_are_valid(env):
    w, alerts, pool, rows = env
    per = {}
    for r in rows:
        assert set(r) == {"decision", "context", "label", "alert_id", "split"}
        assert r["label"] in DECISION_LABELS[r["decision"]], r
        per.setdefault(r["alert_id"], []).append(r["decision"])
    for a in alerts:
        got = per[a.alert_id]
        assert got.count("rule_applies") == DEFAULT_MIX["rule_applies"] and got.count("worth_asking") == 6
        assert got.count("injection_present") == 6 and got.count("stop") == 6
        assert 20 <= len(got) <= 50
    assert {r["split"] for r in rows} == {"train", "val"}


def test_labels_come_from_the_right_sources(env):
    w, alerts, pool, rows = env
    by = {a.alert_id: a for a in alerts}
    gate = Gate.from_config()
    inj_items = {i.note: i.injection_present for i in inj.generate(100, 400, 1)}
    for r in rows:
        a = by[r["alert_id"]]
        if r["decision"] == "worth_asking":
            assert r["label"] == ("yes" if a.label else "no")
        elif r["decision"] == "typology":
            assert r["label"] == (a.typology if a.label else "none")
        elif r["decision"] == "injection_present":
            note = re.search(r'this note: "(.*)"$', r["context"], re.S).group(1)
            assert r["label"] == ("yes" if inj_items[note] else "no")
        elif r["decision"] == "rule_applies":
            m = re.search(r"jurisdiction: (\w+)\. .*\(answer kind (\w+)\).*Cited rule ([A-Z0-9-]+)", r["context"], re.S)
            jur, kind, rid = m.groups()
            assert r["label"] == ("yes" if rule_label(gate, rid, jur, kind) else "no")
    rl = [r["label"] for r in rows if r["decision"] == "rule_applies"]
    assert 0.3 < rl.count("yes") / len(rl) < 0.7  # balanced by construction
    assert {"yes", "no"} <= {r["label"] for r in rows if r["decision"] == "stop"}


def test_a_laundering_alert_without_a_typology_is_skipped_for_typology():
    w = make_world(40, seed=6)
    a = next(x for x in w.alerts if x.label)
    rows = list(DecisionSetBuilder(w.ledgers, w.banks, None, seed=0).rows_for(replace(a, typology=None, split="train")))
    assert "typology" not in {r["decision"] for r in rows} and "stop" not in {r["decision"] for r in rows}


def test_contexts_are_aliased_and_leak_nothing_about_the_label(env):
    w, alerts, pool, rows = env
    for r in rows:
        c = r["context"]
        assert not ALIAS_RE.search(c) and not RAW_ACCOUNT_RE.search(c) and not DATE_RE.search(c), c
    # the same alert with the label flipped gives the same worth_asking / typology contexts
    a = alerts[0]
    b = DecisionSetBuilder(w.ledgers, w.banks, None, seed=1)
    ctx = lambda al: [r["context"] for r in b.rows_for(al) if r["decision"] in ("worth_asking", "typology")]  # noqa: E731
    assert ctx(replace(a, label=True, typology="cycle")) == ctx(replace(a, label=False, typology=None))


def test_evidence_only_uses_the_past(env):
    w, alerts, _, _ = env
    a = alerts[len(alerts) // 2]
    full = w.ledgers[a.binding.home_bank]
    past = full.filter(pl.col("timestamp") <= a.binding.t0)
    assert evidence_text(a, full, 72) == evidence_text(a, past, 72)


def test_stop_label_oracle():
    def c(lr, p_cell=None, eps=1.0):
        return Candidate("p(ALERT.receiver)", eps, "f", "bool", "receiver", lr, (), p_cell, 1)

    flip = c({"true": 8.0, "false": 0.5})
    assert not stop_label(0.2, [flip], 0.5, 2.0)  # a 'true' answer lifts 0.2 to 0.67: crosses 0.5
    assert stop_label(0.2, [c({"true": 1.5, "false": 0.7})], 0.5, 2.0)  # nothing can reach 0.5
    assert stop_label(0.2, [flip], 0.5, 0.1)  # no eps for even the smallest question
    assert stop_label(0.2, [], 0.5, 2.0)  # nothing left to ask
    assert stop_label(0.2, [c({"true": 8.0, "false": 0.5}, {"true": (0.0, 0.0), "false": (1.0, 1.0)})], 0.5, 2.0)  # unreachable cell
    assert not stop_label(float(1 / (1 + 2.718281828 ** -logit(0.9))), [c({"true": 1.0, "false": 0.01})], 0.5, 2.0)  # a 'false' answer drops 0.9 below 0.5


def test_the_rule_labels_follow_the_gate():
    g = Gate.from_config()
    assert rule_label(g, "US-314B-SHARE", "US", "count") and not rule_label(g, "US-314B-SHARE", "SG", "count")
    assert rule_label(g, "SG-COSMIC-REQUEST", "SG", "bool") and not rule_label(g, "INTERNAL-SAR-DRAFT", "US", "bool")
    assert parse("n_banks(ALERT.receiver, dir=in, window=before(24))").prim.kind == "count"


def test_deterministic_and_the_test_split_is_refused(env, tmp_path):
    w, alerts, pool, rows = env
    again = list(DecisionSetBuilder(w.ledgers, w.banks, pool, 0.3, seed=1).build(alerts))
    assert again == rows
    assert len(list(DecisionSetBuilder(w.ledgers, w.banks, pool, 0.3, seed=1).build(alerts, max_total=50))) == 50
    with pytest.raises(ValueError, match="test split"):
        list(DecisionSetBuilder(w.ledgers, w.banks, pool).build([replace(alerts[0], split="test")]))
    m = write_set(rows, tmp_path)
    assert m["n"] == len(rows) and set(m["files"]) == {"train", "val"}
    lines = (tmp_path / "decisions_train.jsonl").read_text().splitlines()
    assert json.loads(lines[0]).keys() == rows[0].keys() and m["by_split_decision"]["train/stop"] > 0


def test_cli_synthetic(tmp_path):
    spec = importlib.util.spec_from_file_location("mk", Path(__file__).resolve().parents[1] / "scripts" / "make_decision_set.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    m = cli.main(["--synthetic", "100", "--out", str(tmp_path), "--max-total", "1000"])
    assert m["n"] == 1000 and (tmp_path / "manifest.json").exists()
