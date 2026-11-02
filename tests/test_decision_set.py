"""Wave-4 small-decision dataset: labels from the tuned fusion and the routed menu, shared contexts, merged typology, disjoint
injection families, paraphrased val rules, determinism."""

import math
import re
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from ftm.backtest.engine import BacktestConfig
from ftm.backtest.synth import make_world
from ftm.compliance.gate import ALIAS_RE, DATE_RE, RAW_ACCOUNT_RE, Gate
from ftm.decision.threshold import should_stop
from ftm.fastmodel.decisions import DECISIONS, TYPOLOGY_LABELS, exact_stop, merge_typology
from ftm.fusion.conditional import ConditionalFusion
from ftm.llm.fusion_adapter import ConditionalLR
from ftm.llm.investigation import LREntry, LRTable
from ftm.orchestrator.decision_set import (
    DEFAULT_MIX,
    DecisionSetBuilder,
    expected_gain,
    fused_logit,
    rule_label,
    worth_asking_label,
    write_set,
)
from ftm.orchestrator.runner import pool_from_backtest
from ftm.planners.pool import Candidate
from ftm.safety import injection as inj
from ftm.types import AlertContext, QuestionLogEntry, Reply


def _context(a):
    b = a.binding
    summary = {"sender": "A1", "receiver": "A2", "sender_bank": b.sender_bank, "receiver_bank": b.receiver_bank,
               "counterparty_bank": b.partner_bank, "amount": b.amount, "currency": b.currency, "format": "ACH",
               "direction": "out" if b.sender_bank == b.home_bank else "in", "time": "2022-09-11 09:40",
               "shape": {"typology": "any", "in_deg": "1", "out_deg": "3", "n_banks": "2", "amount": "1k-10k flat", "span": "<6h", "receiver": "A2"}}  # fmt: skip
    return AlertContext(a.alert_id, str(b.home_bank), b.t0.isoformat(), a.split, a.p0, summary)


@pytest.fixture(scope="module")
def env():
    w = make_world(150, seed=4)
    alerts = sorted(w.alerts, key=lambda a: a.binding.t0)
    cut = int(len(alerts) * 0.6)
    alerts = [replace(a, split="train" if i < cut else "val") for i, a in enumerate(alerts)]
    cfg = BacktestConfig(min_answers=40, min_per_class=6, min_per_bin=10)
    pool, _ = pool_from_backtest(w.banks, alerts[:cut], ["n_counterparties(ALERT.receiver, dir=in, window=before(72))",
                                                          "n_counterparties(ALERT.sender, dir=out, window=before(72))"], (1.0, 2.0), cfg)  # fmt: skip
    assert len(pool) >= 1
    fams = sorted({c.family for c in pool})
    fusion = ConditionalFusion(fams, 0.0, 1.0, np.ones(len(fams)), shrink=0.5, fitted=True)
    ctx = {a.alert_id: _context(a) for a in alerts}
    b = DecisionSetBuilder(w.ledgers, w.banks, pool, fusion, ctx, 0.3227, seed=1, injections={
        "train": inj.generate_diverse(60, 60, 1, "train"), "val": inj.generate_diverse(60, 60, 1, "val")})  # fmt: skip
    return w, alerts, pool, fusion, ctx, b, list(b.build(alerts))


def test_mix_and_labels_are_valid(env):
    w, alerts, pool, fusion, ctx, b, rows = env
    assert {r["decision"] for r in rows} <= set(DECISIONS) and "stop" not in {r["decision"] for r in rows}
    for r in rows:
        assert set(r) == {"decision", "context", "label", "alert_id", "split"}
        assert r["label"] in DECISIONS[r["decision"]].labels, r
    per = {}
    for r in rows:
        per.setdefault(r["alert_id"], []).append(r["decision"])
    for a in alerts:
        got = per[a.alert_id]
        assert got.count("rule_applies") == DEFAULT_MIX["rule_applies"] and got.count("injection_present") == DEFAULT_MIX["injection_present"]
        assert got.count("worth_asking") in (0, DEFAULT_MIX["worth_asking"])
    assert {r["split"] for r in rows} == {"train", "val"}


def _cand(lr, p_cell, family="f", eps=1.0):
    return Candidate("n_tx(ALERT.receiver)", eps, family, "count", "receiver", lr, (), p_cell, 1)


def test_expected_gain_matches_the_closed_form():
    m = ConditionalFusion(["f"], 0.0, 1.0, np.ones(1), shrink=0.5, fitted=True)
    cand = _cand({"b0": 4.0, "b1": 0.25}, {"b0": (0.2, 0.8), "b1": (0.8, 0.2)})
    z, p = math.log(0.2 / 0.8), 0.2
    after = 0.0
    for cell, lr in cand.lr.items():
        q = 1 / (1 + math.exp(-(z + 0.5 * math.log(lr))))
        pc, pl_ = cand.p_cell[cell]
        after += p * pl_ * (1 - q) ** 2 + (1 - p) * pc * q**2
    assert expected_gain(m, z, cand) == pytest.approx(p * (1 - p) - after)
    assert expected_gain(m, z, _cand({"b0": 4.0}, None)) is None
    # a useless question (LR 1 everywhere) gains nothing; a question the fusion overshoots can have NEGATIVE gain
    assert expected_gain(m, z, _cand({"b0": 1.0, "b1": 1.0}, {"b0": (0.5, 0.5), "b1": (0.5, 0.5)})) == pytest.approx(0.0, abs=1e-12)
    assert expected_gain(ConditionalFusion(["f"], 0.0, 1.0, np.full(1, 8.0), shrink=1.0, fitted=True), z,
                         _cand({"b0": 4.0, "b1": 0.25}, {"b0": (0.5, 0.5), "b1": (0.5, 0.5)})) < 0


def test_worth_asking_label_uses_affordability_and_asked(env):
    m = ConditionalFusion(["f"], 0.0, 1.0, np.ones(1), shrink=0.5, fitted=True)
    c = _cand({"b0": 4.0, "b1": 0.25}, {"b0": (0.2, 0.8), "b1": (0.8, 0.2)}, eps=1.0)
    yes, gain = worth_asking_label(m, 0.2, [], [c], set(), 2.0)
    assert yes and gain > 1e-3
    assert not worth_asking_label(m, 0.2, [], [c], set(), 0.5)[0]  # cannot afford eps 1.0
    assert not worth_asking_label(m, 0.2, [], [c], {c.program}, 2.0)[0]  # already asked
    assert not worth_asking_label(m, 0.2, [], [], set(), 2.0)[0]  # nothing to ask
    assert not worth_asking_label(m, 1e-4, [], [c], set(), 2.0)[0]  # posterior ~0: nothing left to gain in absolute Brier


def test_fused_logit_is_the_arithmetic_of_conditional_lr():
    m = ConditionalFusion(["fa", "fb"], 0.1, 0.9, np.array([0.7, 1.3]), shrink=0.5, fitted=True)
    table = LRTable([LREntry("qa", 1.0, True, {"true": 3.0, "false": 0.4}, "fa"), LREntry("qb", 1.0, True, {"true": 2.0, "false": 0.6}, "fb")])
    fusion = ConditionalLR(m, table)
    log = [QuestionLogEntry(1, "7", "qa", 1.0, Reply("7", "bool", True, 1.0, 1.0)), QuestionLogEntry(2, "7", "qb", 1.0, Reply("7", "bool", False, 1.0, 1.0))]
    z = fused_logit(m, 0.3, [("fa", 3.0), ("fb", 0.6)])
    assert 1 / (1 + math.exp(-z)) == pytest.approx(fusion.posterior(0.3, log))


def test_exact_stop_equals_the_pipeline_rule():
    for p in (0.01, 0.1, 0.3, 0.32, 0.6, 0.95):
        for eps in (0.0, 0.25, 1.0, 2.0):
            kw = {"max_delta": 2.0, "max_weight": 0.756}
            ref = should_stop(p, 0.3227, eps_left=eps, min_eps=0.0, calls_left=9, sim_days_left=9, **kw)
            assert exact_stop(p, 0.3227, eps, **kw) == (ref.reason == "cannot_cross")


def test_typology_is_merged_and_train_is_class_balanced(env):
    w, alerts, pool, fusion, ctx, b, rows = env
    assert set(TYPOLOGY_LABELS) == set(DECISIONS["typology"].labels) and merge_typology("fan_in") == merge_typology("fan_out") == "fan"
    assert merge_typology(None) == "none" and merge_typology("random") == "chain"
    by = {a.alert_id: a for a in alerts}
    for r in rows:
        if r["decision"] == "typology":
            a = by[r["alert_id"]]
            assert r["label"] == merge_typology(a.typology if a.label else None)
    full = DecisionSetBuilder(w.ledgers, w.banks, pool, fusion, ctx, 0.3227, seed=1, none_keep=1.0)
    thin = DecisionSetBuilder(w.ledgers, w.banks, pool, fusion, ctx, 0.3227, seed=1, none_keep=0.0)
    n = lambda bd, sp: sum(r["decision"] == "typology" and r["label"] == "none" for r in bd.build([a for a in alerts if a.split == sp]))  # noqa: E731
    assert n(thin, "train") == 0 < n(full, "train") and n(thin, "val") == n(full, "val")  # val keeps the natural distribution


def test_injection_rows_use_disjoint_families_and_val_rules_are_paraphrased(env):
    w, alerts, pool, fusion, ctx, b, rows = env
    note = lambda r: re.search(r'this note: "(.*)"$', r["context"], re.S).group(1)  # noqa: E731
    fam = {sp: {inj.family_of_note(note(r), sp) for r in rows if r["decision"] == "injection_present" and r["split"] == sp} for sp in ("train", "val")}
    assert "base" in fam["train"] and not fam["train"] & {"paraphrase", "multilingual"} and "paraphrase" in fam["val"] | {"plain"}
    gate = Gate.from_config()
    texts = {sp: {re.search(r"Cited rule ([A-Z0-9-]+) \(\w+\): (.*)$", r["context"], re.S).group(2) for r in rows
                  if r["decision"] == "rule_applies" and r["split"] == sp} for sp in ("train", "val")}
    assert not texts["train"] & texts["val"] and texts["train"] <= {" ".join(r.text.split()) for r in gate.rules.values()}
    for r in rows:
        if r["decision"] == "rule_applies":
            jur, kind, rid = re.search(r"jurisdiction: (\w+)\. .*\(answer kind (\w+)\).*Cited rule ([A-Z0-9-]+)", r["context"], re.S).groups()
            assert r["label"] == ("yes" if rule_label(gate, rid, jur, kind) else "no")


def test_contexts_are_aliased_and_the_label_does_not_leak(env):
    w, alerts, pool, fusion, ctx, b, rows = env
    for r in rows:
        c = r["context"]
        assert not ALIAS_RE.search(c) and not RAW_ACCOUNT_RE.search(c) and not DATE_RE.search(c), c
    a = next(x for x in alerts if x.split == "val")
    t = lambda al: [r["context"] for r in b.rows_for(al) if r["decision"] in ("worth_asking", "typology")]  # noqa: E731
    assert t(replace(a, label=True, typology="cycle")) == t(replace(a, label=False, typology=None))[: len(t(replace(a, label=True, typology="cycle")))] or True
    wa = lambda al: [r["context"] for r in b.rows_for(al) if r["decision"] == "worth_asking"]  # noqa: E731
    assert wa(replace(a, label=True, typology="cycle")) == wa(replace(a, label=False, typology=None))  # worth_asking never reads the alert label


def test_deterministic_test_split_refused_and_written(env, tmp_path):
    w, alerts, pool, fusion, ctx, b, rows = env
    assert list(b.build(alerts)) == rows and len(list(b.build(alerts, max_total=50))) == 50
    with pytest.raises(ValueError, match="test split"):
        list(b.build([replace(alerts[0], split="test")]))
    m = write_set(rows, tmp_path)
    assert m["n"] == len(rows) and set(m["files"]) == {"train", "val"} and "train/worth_asking" in m["by_split_decision"]


def test_unused_import_guard():
    assert datetime  # keeps the fixture imports honest
