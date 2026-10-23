"""Week 6: exact-mode back-testing, the realistic-vs-exact comparison, the no-noise arm of the harness, the eps sweep."""

import importlib.util
import json
import math
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
import yaml

from ftm.backtest.engine import BacktestConfig, Backtester, accept
from ftm.backtest.exact import compare_modes, exact_alerts
from ftm.backtest.synth import HOME, PARTNERS, make_world
from ftm.banks.binding import alert_context
from ftm.banks.partner import Question
from ftm.compliance.gate import GateBlocked
from ftm.dsl.executor import execute
from ftm.dsl.parser import canonical, parse
from ftm.orchestrator.eps_sweep import BUDGETS, ceiling, eps_sweep, fixed_level, synthetic_eps_sweep
from ftm.orchestrator.runner import make_harness
from ftm.planners.pool import Candidate
from ftm.types import Limits

PROGS = [
    "n_counterparties(ALERT.receiver, dir=in, window=before(72))",
    "n_tx(ALERT.receiver, dir=in, window=around(72), max_amount=2000)",
    "passed_on(ALERT.receiver, window=after(24), min_frac=0.5)",
    "dormant(ALERT.receiver, days=7)",
    "returns_home(ALERT.receiver, window=after(72))",
]
PROG = PROGS[0]
CFG = BacktestConfig(min_answers=50, min_per_class=8, min_per_bin=12)


# --- the no-noise arm of the harness -----------------------------------------------------------------------------------
def test_the_no_noise_arm_is_an_explicit_opt_in_and_is_exact_and_free():
    w = make_world(60, seed=7)
    a = w.alerts[0]
    ctx = alert_context(a.binding, 0.2)
    bank = a.binding.receiver_bank
    for allow in (False, True):
        h = make_harness(w.banks, {a.alert_id: a.binding}, eps_levels=(1.0, 2.0), allow_exact=allow)
        t = h.toolbox()
        t.begin(ctx, Limits(max_sim_time_days=1000.0))
        if not allow:
            with pytest.raises(GateBlocked):
                t.ask_bank(bank, PROG, math.inf)
            continue
        first = t.ask_bank(bank, PROG, math.inf)
        for _ in range(11):  # far beyond the 2.0 budget: nothing is priced
            t.ask_bank(bank, PROG, math.inf)
        r = first
        assert t.eps_spent == 0 and not h.accountant.alert_spent
        direct = w.banks[bank].handle(Question(canonical(parse(PROG)), math.inf, a.binding), a.binding.t0)
        assert r == direct
        exact = execute(parse(PROG), w.ledgers[bank], a.binding.exec_context(bank, a.binding.t0 + pl_days(r.latency_days)))
        assert r.value == exact  # no noise at all
        with pytest.raises(GateBlocked):
            t.ask_bank(1, PROG, math.inf)  # the home bank still cannot ask itself, and the gate's other checks stay on
        with pytest.raises(GateBlocked):
            t.ask_bank(bank, "nope(", math.inf)


def pl_days(d):
    from datetime import timedelta

    return timedelta(days=d)


def test_candidate_cost_treats_infinite_eps_as_free_but_ranks_it_as_one_unit():
    c = Candidate("p", math.inf, "f", "bool", "receiver", {}, (), None, 1)
    assert c.cost == 0.0 and c.rank_cost == 1.0
    assert Candidate("p", 0.5, "f", "bool", "receiver", {}, (), None, 1).cost == 0.5


# --- the sweep -------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rows():
    return synthetic_eps_sweep("eig", 300, 3, programs=PROGS)


def test_sweep_has_one_row_per_budget_with_the_theoretical_ceiling(rows):
    assert [r["budget"] for r in rows] == list(BUDGETS)
    for r in rows:
        assert r["ceiling_log_odds"] == r["budget"] and r["alerts"] > 80
        assert r["ceiling_lr"] == (math.exp(r["budget"]) if math.isfinite(r["budget"]) else math.inf)
    assert ceiling(2.0)["ceiling_lr"] == pytest.approx(math.exp(2.0))


def test_every_finite_budget_is_respected_and_the_exact_arm_spends_nothing_priced(rows):
    for r in rows:
        if math.isfinite(r["budget"]):
            assert r["mean_eps"] <= r["budget"] + 1e-9, r
    assert rows[-1]["mean_questions"] > 0 and rows[-1]["budget"] == math.inf


def test_more_budget_does_not_hurt_and_questions_help(rows):
    gain = {r["budget"]: r["brier_gain"] for r in rows}
    assert gain[2.0] > 0.015 and gain[4.0] >= gain[1.0] - 0.005 and gain[math.inf] > gain[1.0] - 0.005
    assert gain[1.0] >= gain[0.5] - 0.005
    q = {r["budget"]: r["mean_questions"] for r in rows}
    assert q[0.5] <= q[1.0] <= q[2.0] <= q[4.0] + 1e-9


def test_fixed_level_leaves_room_for_two_questions():
    assert fixed_level(2.0, (0.25, 0.5, 1.0, 2.0)) == 1.0 and fixed_level(4.0, (0.25, 0.5, 1.0, 2.0)) == 2.0
    assert fixed_level(0.25, (0.25, 0.5, 1.0, 2.0)) == 0.25  # nothing fits twice: the smallest level


def test_sweep_is_deterministic():
    a = synthetic_eps_sweep("random", 240, 5, programs=PROGS[:2], budgets=(1.0, 2.0))
    b = synthetic_eps_sweep("random", 240, 5, programs=PROGS[:2], budgets=(1.0, 2.0))
    strip = lambda rs: [{k: v for k, v in r.items() if k != "mean_seconds_per_alert"} for r in rs]  # noqa: E731
    assert strip(a) == strip(b)


def test_eps_sweep_skips_the_exact_point_without_an_exact_pool():
    w = make_world(200, seed=2)
    alerts = sorted(w.alerts, key=lambda a: a.binding.t0)
    train, val = alerts[:120], [replace(a, p0=0.2) for a in alerts[120:]]
    from ftm.orchestrator.runner import pool_from_backtest

    pool, _ = pool_from_backtest(w.banks, train, PROGS[:1], (1.0, 2.0), CFG)
    out = eps_sweep("eig", w.banks, val, pool, None, budgets=(2.0, math.inf))
    assert [r["budget"] for r in out] == [2.0]


# --- exact mode: every bank's labels ------------------------------------------------------------------------------------
def alert_transactions(w):
    full = w.full.with_row_index("tx_id").with_columns(pl.col("tx_id").cast(pl.Int64))
    labels = {(a.binding.sender, a.binding.receiver, a.binding.t0): a.label for a in w.alerts}
    rows = [r for r in full.iter_rows(named=True) if (r["src_id"], r["dst_id"], r["timestamp"]) in labels]
    tx = pl.DataFrame(rows, schema=full.schema)
    y = pl.Series([labels[(r["src_id"], r["dst_id"], r["timestamp"])] for r in tx.iter_rows(named=True)])
    return tx.select("tx_id", "timestamp", "from_bank", "to_bank", "src_id", "dst_id", "amount_received", "receiving_currency"), y


def test_exact_alerts_are_a_label_stratified_cross_bank_sample():
    w = make_world(300, seed=8)
    tx, y = alert_transactions(w)
    out = exact_alerts(tx, y, w.ledgers, n_pos=30, neg_ratio=3, seed=1)
    assert sum(a.label for a in out) == 30 and len(out) == 120
    assert all(a.binding.partner_bank is not None and a.binding.partner_bank != a.binding.home_bank for a in out)
    assert {a.binding.home_bank for a in out} > {HOME}  # both ends of a transfer are used as the home bank
    assert [a.binding.t0 for a in out] == sorted(a.binding.t0 for a in out)
    assert out == exact_alerts(tx, y, w.ledgers, n_pos=30, neg_ratio=3, seed=1)
    assert out != exact_alerts(tx, y, w.ledgers, n_pos=30, neg_ratio=3, seed=2)
    with pytest.raises(ValueError, match="locked"):
        exact_alerts(tx, y, w.ledgers, split="test")
    few = exact_alerts(tx, y, w.ledgers, n_pos=10_000, neg_ratio=1, seed=0)  # asks for more than exists: takes what is there
    assert sum(a.label for a in few) == sum(y.to_list())


def test_exact_mode_has_no_noise_and_compares_with_realistic():
    w = make_world(320, seed=8)
    tx, y = alert_transactions(w)
    ex_alerts = exact_alerts(tx, y, w.ledgers, n_pos=60, neg_ratio=3, seed=1)
    both = [*PROGS[:2], PROGS[0].replace("receiver", "sender"), PROGS[3]]
    exact = accept(Backtester(w.banks, ex_alerts, replace(CFG, mode="exact")).run_batch(both, [math.inf]), 0.1)
    real = accept(Backtester(w.banks, w.alerts, CFG).run_batch(both, [2.0]), 0.1)
    assert all(e["ceiling"] == math.inf for e in exact if e["eligible"])
    cmp = compare_modes(real, exact)
    assert cmp["n_common"] >= 3 and cmp["accepted_both"] >= 1
    assert cmp["accepted_realistic"] == cmp["accepted_both"] + cmp["accepted_only_realistic"]
    assert 0 < cmp["median_strength_ratio_realistic_to_exact"] < 3


def test_compare_modes_on_hand_made_entries():
    def e(p, strength, acc, elig=True, eps=2.0):
        return {"program_canonical": p, "eps": eps, "eligible": elig, "accepted": acc,
                "lr": {"true": math.exp(strength), "false": math.exp(-strength / 2)}}  # fmt: skip

    real = [e("a", 1.0, True), e("b", 0.5, False), e("c", 2.0, True), e("d", 0.2, False), e("z", 9, True, elig=False)]
    ex = [e("a", 2.0, True, eps=math.inf), e("b", 1.0, True, eps=math.inf), e("c", 4.0, True, eps=math.inf),
          e("d", 0.4, False, eps=math.inf)]  # fmt: skip
    c = compare_modes(real, ex)
    assert (c["n_common"], c["accepted_realistic"], c["accepted_exact"], c["accepted_both"]) == (4, 2, 3, 2)
    assert (c["accepted_only_realistic"], c["accepted_only_exact"]) == (0, 1)
    assert c["strength_rank_correlation"] == pytest.approx(1.0)
    assert c["median_strength_ratio_realistic_to_exact"] == pytest.approx(0.5)
    assert compare_modes([], ex) == {"n_common": 0}


# --- CLI: exact labels end to end on a fabricated dataset ----------------------------------------------------------------
@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("exact")
    w = make_world(420, seed=21)
    tx, y = alert_transactions(w)
    full = w.full.with_row_index("tx_id").with_columns(pl.col("tx_id").cast(pl.Int64), pl.lit("train").alias("split"))
    lab = {(r["src_id"], r["dst_id"], r["timestamp"]): v for r, v in zip(tx.iter_rows(named=True), y.to_list(), strict=True)}
    full = full.with_columns(pl.Series("is_laundering", [lab.get((r["src_id"], r["dst_id"], r["timestamp"]), False)
                                                          for r in full.iter_rows(named=True)]))  # fmt: skip
    for d in ("pq", "views", "alerts"):
        (root / d).mkdir()
    full.write_parquet(root / "pq" / "transactions.parquet")
    bank_ids = (HOME, *PARTNERS, 5)
    for b in bank_ids:
        full.filter((pl.col("from_bank") == b) | (pl.col("to_bank") == b)).write_parquet(root / "views" / f"bank={b}.parquet")
    pl.DataFrame({"rank": list(range(1, len(bank_ids) + 1)), "bank": list(bank_ids), "n_tx": [1] * len(bank_ids),
                  "n_laundering": [0] * len(bank_ids)}).write_parquet(root / "views" / "banks.parquet")  # fmt: skip
    cfg = root / "mini.yaml"
    cfg.write_text(yaml.safe_dump({"name": "mini", "parquet_dir": "pq", "views_dir": "views", "alerts_dir": "alerts"}))
    return root, cfg


def load_cli(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / "backtest.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_cli_exact_labels_then_compare(dataset, monkeypatch, capsys):
    root, cfg = dataset
    monkeypatch.chdir(root)
    q = root / "q.jsonl"
    q.write_text("\n".join(PROGS[:2] + [PROGS[0].replace("receiver", "sender")]))
    cli = load_cli("bt_exact")
    base = ["--data", str(cfg), "--questions", "none", "--questions-file", str(q), "--out", "out", "--n-pos", "75", "--neg-ratio", "3"]
    cli.main([*base, "--exact-labels"])
    cli.main([*base, "--exact-labels", "--assemble"])
    capsys.readouterr()
    (ns_dir,) = (root / "out" / "mini").iterdir()
    entries = [json.loads(x) for x in (ns_dir / "lr_table.jsonl").read_text().splitlines()]
    assert entries and all(e["eps"] in ("Infinity", math.inf) or e["eps"] == math.inf for e in entries)
    out = cli.main(["--compare", str(ns_dir), str(ns_dir)])
    assert out["n_common"] >= 1 and out["accepted_only_realistic"] == 0 == out["accepted_only_exact"]
    from ftm.backtest.exact import load_exact_world

    with pytest.raises(ValueError, match="locked"):
        load_exact_world(str(cfg), "test", root=str(root))


def test_eps_sweep_cli_synthetic(tmp_path):
    spec = importlib.util.spec_from_file_location("sw", Path(__file__).resolve().parents[1] / "scripts" / "eps_sweep.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    out = tmp_path / "rows.json"
    rows = cli.main(["--system", "random", "--synthetic", "200", "--budgets", "1", "2", "--out", str(out)])
    assert [r["budget"] for r in rows] == [1.0, 2.0, math.inf] and len(json.loads(out.read_text())) == 3
