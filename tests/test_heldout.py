"""E12: back-testing with the cycle attempts held out, and the transfer report."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
import yaml

from ftm.backtest.engine import BacktestConfig, Backtester, accept
from ftm.backtest.heldout import separation, transfer_report
from ftm.backtest.synth import HOME, PARTNERS, make_world
from ftm.backtest.world import load_world

INFORMATIVE = "n_counterparties(ALERT.receiver, dir=in, window=before(72))"
CFG = BacktestConfig(min_answers=50, min_per_class=8, min_per_bin=12)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("held")
    w = make_world(520, seed=31, signal=1.0)
    full = w.full.with_row_index("tx_id").with_columns(pl.col("tx_id").cast(pl.Int64))
    cut = sorted(a.binding.t0 for a in w.alerts)[int(0.6 * len(w.alerts))]
    full = full.with_columns(pl.when(pl.col("timestamp") < cut).then(pl.lit("train")).otherwise(pl.lit("val")).alias("split"))
    for d in ("pq", "views", "alerts"):
        (root / d).mkdir()
    full.write_parquet(root / "pq" / "transactions.parquet")
    for b in (HOME, *PARTNERS, 5):
        full.filter((pl.col("from_bank") == b) | (pl.col("to_bank") == b)).write_parquet(root / "views" / f"bank={b}.parquet")
    key = {(r["src_id"], r["dst_id"], r["timestamp"]): r for r in full.iter_rows(named=True)}
    alert_rows, pattern_rows = [], []
    for i, a in enumerate(w.alerts):
        if a.label and i % 5 < 2:
            a = replace(a, typology="cycle")  # two in five launderers are cycles, so the held-out typology has enough alerts
        b = a.binding
        r = key[(b.sender, b.receiver, b.t0)]
        alert_rows.append(dict(alert_id=a.alert_id, tx_id=r["tx_id"], bank=HOME, timestamp=b.t0, label=a.label, p0=0.2,
                               typology=a.typology, split=r["split"]))  # fmt: skip
        if a.typology:
            pattern_rows.append(dict(tx_id=r["tx_id"], attempt_id=len(pattern_rows), typology=a.typology, split=r["split"]))
    al = pl.DataFrame(alert_rows)
    for sp in ("train", "val"):
        al.filter(pl.col("split") == sp).drop("split").write_parquet(root / "alerts" / f"{sp}.parquet")
    pl.DataFrame(pattern_rows).write_parquet(root / "pq" / "patterns.parquet")
    cfg = root / "mini.yaml"
    cfg.write_text(yaml.safe_dump({"name": "mini", "parquet_dir": "pq", "views_dir": "views", "alerts_dir": "alerts"}))
    n_cycle_train = al.filter((pl.col("typology") == "cycle") & (pl.col("split") == "train")).height
    return root, cfg, n_cycle_train


def test_heldout_cycles_drop_the_cycle_alerts_and_rows_from_train_only(dataset):
    root, cfg, n_cycle = dataset
    assert n_cycle >= 3
    full = load_world(str(cfg), "train", root=root)
    held = load_world(str(cfg), "train", root=root, heldout_cycles=True)
    assert held.n_heldout_dropped == n_cycle and len(held.alerts) == len(full.alerts) - n_cycle
    assert all(a.typology != "cycle" for a in held.alerts) and any(a.typology == "cycle" for a in full.alerts)
    rows = lambda w: sum(led.height for led in w.ledgers.values())  # noqa: E731
    assert rows(full) - rows(held) == 2 * n_cycle  # each dropped transfer sits in the views of its two banks
    assert held.alert_set_id != full.alert_set_id and "heldout" in held.alert_set_id
    # validation is untouched: cycle alerts stay so they can be evaluated
    vfull, vheld = load_world(str(cfg), "val", root=root), load_world(str(cfg), "val", root=root, heldout_cycles=True)
    assert [a.alert_id for a in vfull.alerts] == [a.alert_id for a in vheld.alerts] and vheld.n_heldout_dropped == 0
    assert rows(vfull) == rows(vheld) and any(a.typology == "cycle" for a in vheld.alerts)
    with pytest.raises(ValueError, match="locked"):
        load_world(str(cfg), "test", root=root, heldout_cycles=True)


def test_separation_is_the_mean_log_lr_gap():
    entry = {"lr": {"true": 4.0, "false": 0.5}, "bins": None}
    sep = separation(entry, [True, True, False, False, True], [True, True, False, False, False])
    import math

    assert sep == pytest.approx((math.log(4) * 1.0) - (math.log(0.5) * 1 / 3 + math.log(4) * 1 / 3 + math.log(0.5) * 1 / 3))
    assert separation(entry, [True, False], [True, True]) is None  # no clean alert
    assert separation(entry, ["zzz", "zzz"], [True, False]) is None  # no cell


def test_accepted_questions_transfer_to_the_held_out_typology(dataset):
    root, cfg, _ = dataset
    train = load_world(str(cfg), "train", root=root, heldout_cycles=True)
    entries = accept(Backtester(train.banks, train.alerts, CFG).run_batch([INFORMATIVE, "dormant(ALERT.receiver, days=7)"], [2.0]), 0.1)
    assert any(e["accepted"] for e in entries)
    val = load_world(str(cfg), "val", root=root, heldout_cycles=True)
    cyc = [a for a in val.alerts if a.typology == "cycle"]
    assert sum(a.label for a in cyc) >= 8
    rest = [a for a in val.alerts if a.typology != "cycle"]
    # cycle launderers against all clean alerts: the held-out typology versus the clean background
    clean = [a for a in val.alerts if not a.label]
    rep = transfer_report(entries, val.banks, {"cycle_vs_clean": cyc + clean, "other_vs_clean": rest}, CFG, min_per_class=3)
    assert rep["n_accepted"] >= 1 and set(rep["subsets"]) == {"cycle_vs_clean", "other_vs_clean"}
    other = rep["subsets"]["other_vs_clean"]
    assert other["n_questions_scored"] >= 1 and other["mean_separation"] > 0.1 and other["share_positive"] == 1.0
    cy = rep["subsets"]["cycle_vs_clean"]
    assert cy["n_questions_scored"] >= 1 and cy["mean_separation"] > 0  # the signal here does not depend on the typology
    assert all(q["subset"] in rep["subsets"] for q in rep["per_question"])


def test_cli_heldout_run_and_comparison_with_the_full_run(dataset, monkeypatch, capsys):
    root, cfg, _ = dataset
    monkeypatch.chdir(root)
    spec = importlib.util.spec_from_file_location("bt_held", Path(__file__).resolve().parents[1] / "scripts" / "backtest.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    q = root / "q.jsonl"
    q.write_text(INFORMATIVE)
    base = ["--data", str(cfg), "--eps", "2", "--questions", "none", "--questions-file", str(q), "--out", "out"]
    for extra in ([], ["--heldout-cycles"]):
        cli.main([*base, *extra])
        cli.main([*base, *extra, "--assemble"])
    capsys.readouterr()
    dirs = sorted((root / "out" / "mini").iterdir())
    assert len(dirs) == 2 and all((d / "lr_table.jsonl").exists() for d in dirs)
    out = cli.main(["--compare", str(dirs[0]), str(dirs[1])])
    assert out["n_common"] == 1
    assert json.loads((dirs[0] / "report.json").read_text())["alert_set"] != json.loads((dirs[1] / "report.json").read_text())["alert_set"]
