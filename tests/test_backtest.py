"""C7: e-values, e-BH, e-LOND, LR estimation, the back-testing engine (synthetic world), cache and sharding."""

import math
import random

import numpy as np
import pytest

from ftm.backtest.engine import (
    BacktestCache,
    BacktestConfig,
    Backtester,
    OnlineAcceptor,
    accept,
    asked_bank,
    assemble,
    lr_table_rows,
    shard_of,
)
from ftm.backtest.evalue import (
    ELond,
    e_bh,
    evalue,
    log_evalue,
    log_evalue_path,
    null_false_acceptance,
)
from ftm.backtest.lr import answer_key, choose_bins, estimate_lr
from ftm.backtest.synth import make_world
from ftm.dsl.parser import canonical, parse
from ftm.privacy.accountant import PrivacyAccountant

CFG = BacktestConfig(min_answers=60, min_per_class=10, min_per_bin=15)


def random_null_run(rng, n=150, n_cells=3, prevalence=0.2):
    labels = [rng.random() < prevalence for _ in range(n)]
    cells = [rng.randrange(n_cells) for _ in range(n)]  # independent of the labels
    return cells, labels


# --- e-values ---------------------------------------------------------------------------------------------------
def test_evalue_is_valid_under_the_null():
    rng = random.Random(0)
    runs = [random_null_run(rng) for _ in range(1500)]
    e = np.array([evalue(c, y) for c, y in runs])
    assert e.mean() < 1.15  # E[E] <= 1 under the null (Monte Carlo slack)
    assert (e >= 10).mean() <= 0.10 + 0.02  # Markov: P(E >= 1/alpha) <= alpha


def test_evalue_is_valid_at_any_stopping_time():
    # Ville: P(sup_t E_t >= 1/alpha) <= alpha, so peeking after every alert does not inflate errors
    rng = random.Random(1)
    hits = 0
    for _ in range(1500):
        c, y = random_null_run(rng, n=120)
        hits += max(log_evalue_path(c, y)) >= math.log(10)
    assert hits / 1500 <= 0.10 + 0.02


def test_evalue_grows_when_the_answer_separates_the_classes():
    rng = random.Random(2)
    labels = [rng.random() < 0.2 for _ in range(200)]
    cells = [int(y) if rng.random() < 0.85 else int(not y) for y in labels]
    assert evalue(cells, labels) > 1e6
    weak = [int(y) if rng.random() < 0.25 else rng.randrange(2) for y in labels]
    assert 1e6 > evalue(weak, labels)


def test_evalue_ignores_the_label_it_is_about_to_see():
    # a cell is scored only from earlier alerts: reordering the future cannot change the first factors
    cells, labels = [0, 1, 0, 1, 0, 0, 1, 1, 0, 0], [0, 1, 0, 1, 0, 0, 1, 0, 0, 1]
    p = log_evalue_path(cells, labels)
    p2 = log_evalue_path(cells[:6] + cells[6:][::-1], labels[:6] + labels[6:])
    assert p[:6] == p2[:6]


def test_evalue_edge_cases():
    assert log_evalue([], []) == 0.0
    assert evalue([0, 0, 0], [False, False, False]) == 1.0  # no launderers: nothing to learn
    assert evalue([0, 1, 0, 1], [True, True, True, True]) == 1.0
    with pytest.raises(ValueError):
        evalue([0], [True, False])


# --- e-BH and e-LOND --------------------------------------------------------------------------------------------
def test_e_bh_rejects_the_big_ones_and_only_them():
    assert e_bh([1000.0, 1.0, 0.5, 400.0, 2.0], 0.1) == [True, False, False, True, False]
    assert e_bh([5.0, 5.0, 5.0], 0.1) == [False, False, False]
    assert e_bh([], 0.1) == []
    assert e_bh([math.inf, 1.0], 0.1) == [True, False]


def test_false_acceptance_on_null_questions_is_at_most_ten_percent():
    # plan.md 6.8 "done when": random predicates (answers independent of the labels), 20 questions per batch
    rng = random.Random(3)
    labels = [rng.random() < 0.2 for _ in range(150)]
    hyps = [[rng.randrange(3) for _ in labels] for _ in range(20)]
    res = null_false_acceptance(hyps, labels, n_rep=1200, alpha=0.10, seed=4)
    assert res["false_acceptance_rate"] <= 0.10
    assert res["mean_accepted"] <= 0.12


def test_e_bh_keeps_real_signal_among_nulls():
    rng = random.Random(5)
    n_real_accepted, n_null_accepted = 0, 0
    for _ in range(40):
        labels = [rng.random() < 0.2 for _ in range(200)]
        real = [int(y) if rng.random() < 0.8 else int(not y) for y in labels]
        nulls = [[rng.randrange(3) for _ in labels] for _ in range(19)]
        flags = e_bh([evalue(c, labels) for c in [real, *nulls]], 0.1)
        n_real_accepted += flags[0]
        n_null_accepted += sum(flags[1:])
    assert n_real_accepted == 40 and n_null_accepted <= 4


def test_e_lond_controls_fdr_on_a_stream_of_nulls_and_finds_signal():
    rng = random.Random(6)
    false_runs = 0
    for _ in range(600):
        el = ELond(0.1)
        false_runs += any(el.test(evalue(*random_null_run(rng, n=100))) for _ in range(15))
    assert false_runs / 600 <= 0.10 + 0.02
    el = ELond(0.1)
    assert el.test(1e9) and el.rejections == 1
    assert el.level() == pytest.approx(0.1 * (1 / 6) * 2)  # earlier discoveries raise the next level
    assert ELond.from_dict(el.to_dict()).level() == el.level()
    assert sum(ELond.gamma(t) for t in range(1, 5000)) == pytest.approx(1.0, abs=1e-3)


# --- LR estimation ----------------------------------------------------------------------------------------------
def test_answer_keys_follow_t3s_contract():
    assert (answer_key(True), answer_key(False)) == ("true", "false")
    assert answer_key(3, [2.0, 5.0]) == "b1" and answer_key(1, [2.0, 5.0]) == "b0" and answer_key(5, [2.0, 5.0]) == "b2"
    assert answer_key(2) == "2"  # k-ary: the index itself


def test_bool_lr_matches_the_plan_formula_and_is_clipped():
    # 100 clean (20 yes), 20 launder (14 yes): P(yes|L)=.7, P(yes|C)=.2
    values = [True] * 20 + [False] * 80 + [True] * 14 + [False] * 6
    labels = [False] * 100 + [True] * 20
    est = estimate_lr("bool", values, labels, math.inf, 1, alpha=0.01)
    assert est.lr["true"] == pytest.approx((0.7 + 0.01) / (0.2 + 0.01))
    assert est.lr["false"] == pytest.approx((0.3 + 0.01) / (0.8 + 0.01))
    clipped = estimate_lr("bool", values, labels, 0.5, 1, alpha=0.01)
    assert clipped.lr["true"] == pytest.approx(math.exp(0.5)) and clipped.ceiling == pytest.approx(math.exp(0.5))
    assert all(1 / clipped.ceiling <= v <= clipped.ceiling for v in clipped.lr.values())


def test_count_cells_and_kary_cells():
    rng = np.random.default_rng(0)
    values = list(rng.poisson(3, 400))
    bins = choose_bins(values, 6, 20)
    assert list(bins) == sorted(bins) and 1 <= len(bins) <= 5
    labels = [bool(v > 4) for v in values]
    est = estimate_lr("count", values, labels, 2.0, 1, bins=bins)
    assert set(est.lr) == {f"b{i}" for i in range(len(bins) + 1)}
    assert est.lr[f"b{len(bins)}"] > 1 > est.lr["b0"]
    k = estimate_lr("kary", [0, 1, 2, 3] * 25, [False, True] * 50, 1.0, 1, arity=4)
    assert set(k.lr) == {"0", "1", "2", "3"}
    assert choose_bins([0] * 200 + [1] * 5) == ()  # a bin of 5 is too small: no split


# --- the engine on a synthetic world ---------------------------------------------------------------------------
INFORMATIVE = "n_counterparties(ALERT.receiver, dir=in, window=before(72))"
NULLISH = "dormant(ALERT.receiver, days=7)"  # receivers are fresh accounts for every alert: independent of the label


@pytest.fixture(scope="module")
def world():
    return make_world(320, seed=11, signal=1.0)


def test_informative_question_is_accepted_and_a_null_one_is_not(world):
    bt = Backtester(world.banks, world.alerts, CFG)
    es = accept(bt.run_batch([INFORMATIVE, NULLISH], [1.0, 2.0]), 0.1)
    by = {(e["program_canonical"], e["eps"]): e for e in es}
    assert by[(canonical(parse(INFORMATIVE)), 2.0)]["accepted"]
    assert not by[(canonical(parse(NULLISH)), 1.0)]["accepted"] and not by[(canonical(parse(NULLISH)), 2.0)]["accepted"]
    e = by[(canonical(parse(INFORMATIVE)), 2.0)]
    assert e["kind"] == "count" and e["family"] == "n_counterparties:in" and e["n_answers"] > 200
    assert max(e["lr"].values()) > 1.5 > 0.9 > min(e["lr"].values())
    assert max(e["lr"].values()) <= e["ceiling"] + 1e-9


def test_no_signal_world_accepts_nothing(world):
    flat = make_world(320, seed=12, signal=0.0)
    bt = Backtester(flat.banks, flat.alerts, CFG)
    progs = [INFORMATIVE, NULLISH, "passed_on(ALERT.receiver, window=after(24), min_frac=0.5)",
             "n_tx(ALERT.receiver, dir=in, window=around(72), max_amount=2000)",
             "at_least(n_counterparties(ALERT.receiver, dir=in, window=before(72)), k=3)",
             "returns_home(ALERT.receiver, window=after(72))"]  # fmt: skip
    es = accept(bt.run_batch(progs, [2.0]), 0.1)
    assert not any(e["accepted"] for e in es)


def test_permuted_labels_make_every_question_null(world):
    # the real answers of real questions, with each question's own labels shuffled: nothing may be accepted
    bt = Backtester(world.banks, world.alerts, CFG)
    progs = [INFORMATIVE, "passed_on(ALERT.receiver, window=after(24), min_frac=0.5)", NULLISH,
             "returns_home(ALERT.receiver, window=after(72))"]  # fmt: skip
    hyps = []
    for p in progs:
        ans, lab, _ = bt.collect(parse(p), 2.0)
        numeric = isinstance(ans[0], int) and not isinstance(ans[0], bool)
        bins = choose_bins(ans, 6, 15) if numeric else None
        hyps.append(([answer_key(a, bins) for a in ans], lab))
    rng = random.Random(7)
    any_accept = 0
    for _ in range(300):
        es = []
        for cells, lab in hyps:
            shuffled = list(lab)
            rng.shuffle(shuffled)
            es.append(evalue(cells, shuffled))
        any_accept += any(e_bh(es, 0.1))
    assert any_accept / 300 <= 0.10
    # and without the shuffle the informative one is clearly accepted
    assert evalue(*hyps[0]) > 1e6


def test_eps_cal_is_charged_per_answer_and_exact_mode_is_free(world):
    acc = PrivacyAccountant(eps_cal=None)
    bt = Backtester(world.banks, world.alerts, CFG, accountant=acc)
    e = bt.run_question(INFORMATIVE, 1.0)
    assert acc.report()["eps_cal_spent"] == pytest.approx(1.0 * e["n_answers"])
    assert sum(acc.cal_by_bank.values()) == pytest.approx(acc.cal_spent) and len(acc.cal_by_bank) == 3
    before = acc.cal_spent
    ex = bt.run_question(INFORMATIVE, math.inf)
    assert acc.cal_spent == before and ex["eps"] == math.inf and ex["ceiling"] == math.inf
    assert ex["e_value"] >= e["e_value"] or ex["e_value"] > 1e6  # no noise: at least as clear


def test_cache_hits_do_not_recompute_or_recharge(world, tmp_path):
    ns = CFG.namespace("synth-320", 11)
    acc = PrivacyAccountant()
    bt = Backtester(world.banks, world.alerts, CFG, accountant=acc, cache=BacktestCache(tmp_path, ns))
    a = bt.run_question(INFORMATIVE, 2.0)
    spent = acc.cal_spent
    b = bt.run_question(INFORMATIVE, 2.0)
    assert a == b and acc.cal_spent == spent
    assert BacktestCache(tmp_path, ns).get(canonical(parse(INFORMATIVE)), 2.0) == a
    assert BacktestCache(tmp_path, CFG.namespace("synth-320", 12)).get(canonical(parse(INFORMATIVE)), 2.0) is None


def test_shards_write_one_cache_and_assemble_matches_an_unsharded_run(world, tmp_path):
    progs = [INFORMATIVE, NULLISH, "returns_home(ALERT.receiver, window=after(72))",
             "n_tx(ALERT.receiver, dir=in, window=around(72), max_amount=2000)"]  # fmt: skip
    ns = CFG.namespace("synth-320", 11)
    cache = BacktestCache(tmp_path / "sharded", ns)
    for i in range(3):
        Backtester(world.banks, world.alerts, CFG, cache=cache).run_batch(progs, [2.0], shard=(i, 3))
    assert len(cache.entries()) == len(progs)
    assert {shard_of(canonical(parse(p)), 3) for p in progs} <= {0, 1, 2}
    whole = accept(Backtester(world.banks, world.alerts, CFG).run_batch(progs, [2.0]), 0.1)
    merged = assemble(cache, 0.1)
    key = lambda e: e["program_canonical"]  # noqa: E731
    assert sorted(merged, key=key) == sorted(whole, key=key)


def test_too_little_data_is_never_accepted(world):
    bt = Backtester(world.banks, world.alerts[:40], CFG)
    e = bt.run_question(INFORMATIVE, 2.0)
    assert not e["eligible"] and e["e_value"] == 1.0 and e["lr"] == {}
    assert accept([e])[0]["accepted"] is False
    assert lr_table_rows([e]) == []


def test_lr_table_rows_have_the_format_t3s_lrtable_reads(world):
    bt = Backtester(world.banks, world.alerts, CFG)
    rows = lr_table_rows(accept(bt.run_batch([INFORMATIVE], [2.0]), 0.1))
    assert set(rows[0]) == {"program_canonical", "eps", "accepted", "lr", "family", "bins", "kind", "p_cell"}
    assert rows[0]["accepted"] is True and rows[0]["bins"] and all(isinstance(v, float) for v in rows[0]["lr"].values())
    assert set(rows[0]["lr"]) == {f"b{i}" for i in range(len(rows[0]["bins"]) + 1)}
    pc = rows[0]["p_cell"]  # P(cell | clean), P(cell | launder): each class sums to 1
    assert set(pc) == set(rows[0]["lr"]) and all(abs(sum(v[i] for v in pc.values()) - 1) < 1e-9 for i in (0, 1))


def test_online_acceptor_decides_each_new_question_once(world):
    bt = Backtester(world.banks, world.alerts, CFG)
    oa = OnlineAcceptor(bt)
    a = oa.submit(INFORMATIVE, 2.0)
    assert a["accepted"] and oa.elond.t == 1
    assert oa.submit(INFORMATIVE, 2.0) is a and oa.elond.t == 1  # a repeat is not a new test
    assert not oa.submit(NULLISH, 2.0)["accepted"] and oa.elond.t == 2


def test_asked_bank_is_the_bank_that_holds_the_subject(world):
    b = world.alerts[0].binding
    assert asked_bank(parse("n_tx(ALERT.receiver, dir=in, window=before(72))"), b) == b.receiver_bank
    assert asked_bank(parse("n_tx(ALERT.sender, dir=in, window=before(72))"), b) is None  # home-bank account
    assert asked_bank(parse("set_contacts(ALERT.payers, dir=in, window=before(72))"), b) == b.partner_bank
    assert asked_bank(parse("at_least(n_tx(ALERT.receiver, dir=in, window=before(72)), k=2)"), b) == b.receiver_bank
