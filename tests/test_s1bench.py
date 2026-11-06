"""S1BENCH core: metrics, permutations, typed decisions over arbitrary option sets, collapse checks and the selection rule."""

import math
import re

import numpy as np
import pytest

from ftm.fastmodel.decisions import DECISIONS
from ftm.fastmodel.model import LogprobFastModel
from ftm.fastmodel.s1bench import (
    Candidate,
    ListedTwoStage,
    TaskSpec,
    auroc,
    brier_multiclass,
    collapse_flags,
    core_metrics,
    ece_equal_mass,
    ece_from,
    log_loss,
    macro_f1,
    make_decision,
    order_bias,
    paired_cluster_bootstrap,
    permutations,
    realign,
    registered,
    select_candidate,
)
from ftm.fastmodel.typed import LETTERS, TypedDecider


class Resp:
    def __init__(self, top):
        self.raw = {"choices": [{"message": {"content": top[0]["token"]}, "logprobs": {"content": [{"top_logprobs": top}]}}]}
        self.latency_s = 0.001


class PromptClient:
    """Favours the option whose text contains `favour` in the prompt's option list; letters or yes/no words."""

    def __init__(self, favour: str | None = None, hot: float = 5.0):
        self.favour, self.hot, self.prompts = favour, hot, []

    def chat(self, messages, **kw):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        m = re.search(r"Answer with exactly one letter: (.*?)\.(?:\n|$)", prompt, re.S)
        if m:
            opts = re.findall(r"(?:^|, |; )([A-Z]) = ([^,;]*)", m.group(1))
            toks = [t for t, _ in opts]
            fav = [i for i, (_, text) in enumerate(opts) if self.favour and self.favour in text]
        else:
            toks, fav = ["yes", "no"], ([0] if self.favour == "yes" else [1] if self.favour == "no" else [])
        w = np.ones(len(toks))
        for i in fav:
            w[i] = math.exp(self.hot)
        w = w / w.sum()
        return Resp([{"token": t, "logprob": float(math.log(x))} for t, x in zip(toks, w, strict=True)])


def decider(client, cls=TypedDecider):
    return cls(LogprobFastModel(client, temperature=1.0))


# -- metrics ------------------------------------------------------------------------------------------------
def test_brier_logloss_ece_hand_values():
    P = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    y = np.array([0, 1, 1])
    assert brier_multiclass(P, y) == pytest.approx(np.mean([0.02, 0.08, 0.72]))
    assert log_loss(P, y) == pytest.approx(-np.mean(np.log([0.9, 0.8, 0.4])))
    # one bin = |mean accuracy - mean confidence|
    assert ece_equal_mass(P, y, bins=1) == pytest.approx(abs(2 / 3 - np.mean([0.9, 0.8, 0.6])))
    assert ece_from([0.5, 0.5], [1.0, 0.0], bins=1) == 0.0
    assert math.isnan(ece_from([], []))


def test_ece_equal_mass_bins_have_equal_count():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1, 100)
    hit = (rng.uniform(size=100) < conf).astype(float)
    P = np.stack([conf, 1 - conf], 1)
    y = np.where(hit == 1, 0, 1)
    assert 0 <= ece_equal_mass(P, y, bins=10) < 0.25


def test_macro_f1_and_auroc_match_reference():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 200)
    P = rng.dirichlet(np.ones(4), 200)
    assert macro_f1(P, y) == pytest.approx(sk.f1_score(y, P.argmax(1), average="macro", labels=sorted(set(y))))
    yb = rng.integers(0, 2, 300)
    s = np.round(rng.uniform(size=300), 1)  # ties
    assert auroc(s, yb) == pytest.approx(sk.roc_auc_score(yb, s))
    assert auroc([0.1, 0.2], [1, 1]) is None


def test_core_metrics_flags_majority_baseline():
    y = np.array([0] * 8 + [1] * 2)
    P = np.tile([0.9, 0.1], (10, 1))
    m = core_metrics(P, y, ["a", "b"])
    assert m["majority_accuracy"] == 0.8 and m["accuracy"] == 0.8 and m["pred_distinct"] == 1 and m["majority_label"] == "a"
    assert m["pred_majority_share"] == 1.0 and m["brier_prior"] == pytest.approx(0.32)


# -- permutations ------------------------------------------------------------------------------------------
def test_permutations_are_fixed_non_identity_and_realign_roundtrips():
    ps = permutations(5)
    assert ps[0] == list(range(5)) and len(ps) == 4 and all(p != ps[0] for p in ps[1:]) and ps == permutations(5)
    assert permutations(2)[1:] == [[1, 0]] * 3
    probs = np.array([0.5, 0.2, 0.2, 0.05, 0.05])
    p = ps[1]
    shown = [probs[c] for c in p]  # what a model that is perfectly order-invariant would say in the shown order
    assert realign(shown, p) == pytest.approx(probs)


def test_order_bias_zero_for_invariant_model_and_positive_otherwise():
    P = np.array([[0.7, 0.3], [0.2, 0.8]])
    y = np.array([0, 1])
    assert order_bias(P, [P, P, P], y)["mean_tv"] == 0.0
    flipped = P[:, ::-1]
    r = order_bias(P, [flipped] * 3, y)
    assert r["accuracy_range"] == 1.0 and r["mean_tv"] == pytest.approx(0.5) and r["argmax_flip_rate"] == 1.0


# -- typed decisions ---------------------------------------------------------------------------------------
def test_make_decision_formats_and_registry_cleanup():
    yn = make_decision("t1", "q?", ["yes", "no"])
    assert yn.tokens == ("yes", "no")
    lettered = make_decision("t2", "q?", ["yes", "no"], lettered=True)
    assert lettered.tokens == ("A", "B")
    many = make_decision("t3", "q?", [f"l{i}" for i in range(30)])
    assert len(many.tokens) == 30 and many.tokens[26].startswith("#")
    dup = make_decision("t4", "q?", ["x", "x", "y"])
    assert dup.labels == ("x", "x (2)", "y")
    with registered(yn):
        assert "t1" in DECISIONS
    assert "t1" not in DECISIONS


def test_decide_through_registered_decision_follows_the_favoured_option():
    d = make_decision("s1b::test", "Which one?", ["alpha", "beta", "gamma"])
    dec = decider(PromptClient(favour="beta"))
    with registered(d):
        p = dec.decide(d.name, "some text")
    assert list(p) == ["alpha", "beta", "gamma"] and max(p, key=p.get) == "beta" and sum(p.values()) == pytest.approx(1)


def test_listed_two_stage_lists_every_label_in_stage_one_and_sums_to_one():
    labels = [f"label{i}" for i in range(40)]
    d = make_decision("s1b::big", "Pick", labels)
    client = PromptClient(favour="label31")
    with registered(d):
        p = decider(client, ListedTwoStage).decide(d.name, "text")
    assert sum(p.values()) == pytest.approx(1) and len(p) == 40
    stage1 = client.prompts[0]
    assert "label0" in stage1 and "label25" in stage1 and "A = one of:" in stage1  # whole group listed
    # production stage 1 shows only first .. last
    prod_client = PromptClient(favour="label31")
    with registered(d):
        decider(prod_client).decide(d.name, "text")
    assert "label5" not in prod_client.prompts[0] and "labels label0 .. label25" in prod_client.prompts[0]


# -- bootstrap / collapse / selection ----------------------------------------------------------------------
def test_paired_bootstrap_detects_a_shift_and_not_noise():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.2, 0.4, 400)
    clusters = np.repeat(np.arange(100), 4)
    better = base - 0.05
    r = paired_cluster_bootstrap(better, base, clusters, n_boot=500)
    assert r["ci_hi"] < 0 and r["diff"] == pytest.approx(-0.05) and r["n_clusters"] == 100
    noise = base + rng.normal(0, 0.01, 400)
    r2 = paired_cluster_bootstrap(noise, base, clusters, n_boot=500)
    assert r2["ci_lo"] < 0 < r2["ci_hi"]
    with pytest.raises(ValueError):
        paired_cluster_bootstrap([1.0], [1.0, 2.0], [0])


def test_collapse_flags():
    y = np.array([0] * 70 + [1] * 30)
    const = np.tile([0.9, 0.1], (100, 1))
    f = collapse_flags(const, y, [0.7, 0.3], 0)
    assert f["constant"] and f["collapsed"] and f["majority_only"]
    P = np.where(y[:, None] == 1, [0.1, 0.9], [0.9, 0.1])  # a perfect model
    assert not collapse_flags(P, y, [0.7, 0.3], 0)["collapsed"]
    noskill = np.tile([0.7, 0.3], (100, 1)).astype(float)
    noskill[:50] = [0.3, 0.7]  # random-ish, accuracy ~0.5, Brier above prior
    assert collapse_flags(noskill, y, [0.7, 0.3], 0)["no_skill"]


def _cand(name, brier, ece, lat, params, collapsed=False, n=200):
    rng = np.random.default_rng(sum(map(ord, name)))
    return Candidate(name, params, lat, {"d1": np.full(n, brier) + rng.normal(0, 0.02, n)}, {"d1": ece}, {"d1": collapsed})


def test_selection_rule_chain_brier_then_ece_then_latency_then_size():
    clusters = {"d1": list(range(200))}
    # clearly best Brier wins outright
    a, b = _cand("a", 0.10, 0.05, 1.0, 100), _cand("b", 0.20, 0.01, 0.1, 10)
    r = select_candidate([a, b], clusters, n_boot=200)
    assert r["selected"] == "a" and r["tied"] == ["a"]
    # tied Brier -> lower ECE
    c, d = _cand("c", 0.1000, 0.05, 1.0, 100), _cand("d", 0.1002, 0.01, 1.0, 100)
    assert select_candidate([c, d], clusters, n_boot=200)["selected"] == "d"
    # tied Brier and ECE within the margin -> lower latency
    e, f = _cand("e", 0.1000, 0.010, 2.0, 100), _cand("f", 0.1001, 0.012, 0.5, 400)
    assert select_candidate([e, f], clusters, n_boot=200)["selected"] == "f"
    # all tied -> the smaller model
    g, h = _cand("g", 0.1000, 0.010, 1.0, 400), _cand("h", 0.1001, 0.010, 1.0, 40)
    assert select_candidate([g, h], clusters, n_boot=200)["selected"] == "h"
    # a collapsed candidate is ineligible however good its Brier
    i, j = _cand("i", 0.01, 0.0, 0.1, 1, collapsed=True), _cand("j", 0.2, 0.1, 1.0, 100)
    r = select_candidate([i, j], clusters, n_boot=200)
    assert r["selected"] == "j" and r["ineligible"] == ["i"]
    assert select_candidate([i], clusters, n_boot=50)["selected"] is None


def test_taskspec_binary_flag():
    assert TaskSpec("t", "q", ("yes", "no")).binary_yes_no and not TaskSpec("t", "q", ("a", "b")).binary_yes_no
    assert LETTERS[0] == "A"
