"""scripts/backtest.py and ftm.backtest.world on a tiny fabricated dataset in the real file layout."""

import importlib.util
import json
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest
import yaml

from ftm.backtest.synth import HOME, PARTNERS, make_world
from ftm.backtest.world import load_world

INFORMATIVE = "n_counterparties(ALERT.receiver, dir=in, window=before(72))"


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("ds")
    w = make_world(260, seed=21, signal=1.0)
    train = w.full.with_row_index("tx_id").with_columns(pl.col("tx_id").cast(pl.Int64), pl.lit("train").alias("split"))
    # three transfers after the last alert, in the locked test split: they must never become visible
    test_rows = train.head(3).with_columns(
        (pl.col("tx_id") + 1_000_000).alias("tx_id"),
        pl.lit(train["timestamp"].max() + timedelta(days=2)).alias("timestamp"),
        pl.lit("test").alias("split"),
    )
    full = pl.concat([train, test_rows])
    pq, views, alerts = root / "pq", root / "views", root / "alerts"
    for d in (pq, views, alerts):
        d.mkdir()
    full.write_parquet(pq / "transactions.parquet")
    for b in (HOME, *PARTNERS, 5):
        full.filter((pl.col("from_bank") == b) | (pl.col("to_bank") == b)).write_parquet(views / f"bank={b}.parquet")
    rows = []
    for a in w.alerts:
        b = a.binding
        m = train.filter((pl.col("src_id") == b.sender) & (pl.col("dst_id") == b.receiver) & (pl.col("timestamp") == b.t0))
        rows.append(dict(alert_id=a.alert_id, tx_id=int(m["tx_id"][0]), bank=HOME, timestamp=b.t0, label=a.label, p0=0.3))
    pl.DataFrame(rows).write_parquet(alerts / "train.parquet")
    cfg = root / "mini.yaml"
    cfg.write_text(yaml.safe_dump({"name": "mini", "parquet_dir": "pq", "views_dir": "views", "alerts_dir": "alerts"}))
    return root, cfg, w, test_rows["timestamp"][0]


def test_load_world_reads_alerts_banks_and_binds_slots(dataset):
    root, cfg, w, _ = dataset
    rw = load_world(str(cfg), "train", seed=0, root=root)
    assert len(rw.alerts) == len(w.alerts) and rw.n_skipped == 0
    assert set(rw.banks) == set(PARTNERS)
    a = rw.alerts[0]
    b = {x.alert_id: x for x in w.alerts}[a.alert_id]
    assert [x.binding.t0 for x in rw.alerts] == sorted(x.binding.t0 for x in rw.alerts)
    assert (a.alert_id, a.label) == (b.alert_id, b.label)
    assert (a.binding.sender, a.binding.receiver, a.binding.payers) == (b.binding.sender, b.binding.receiver, b.binding.payers)
    assert a.binding.partner_bank == b.binding.partner_bank


def test_test_split_rows_and_the_test_split_itself_are_unreachable(dataset):
    root, cfg, _, test_time = dataset
    with pytest.raises(ValueError, match="locked"):
        load_world(str(cfg), "test", root=root)
    rw = load_world(str(cfg), "train", root=root)
    assert max(b._ledger["timestamp"].max() for b in rw.banks.values()) < test_time - timedelta(days=1)


def test_cli_shards_write_one_cache_and_assemble_builds_the_lr_table(dataset, monkeypatch, capsys):
    root, cfg, _, _ = dataset
    monkeypatch.chdir(root)
    spec = importlib.util.spec_from_file_location("backtest_cli", Path(__file__).resolve().parents[1] / "scripts" / "backtest.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    qfile = root / "q.jsonl"
    qfile.write_text("\n".join([
        json.dumps({"program": INFORMATIVE}),
        "dormant(ALERT.receiver, days=7)",
        "not a program",
    ]))  # fmt: skip
    base = ["--data", str(cfg), "--eps", "2", "--questions", "none", "--questions-file", str(qfile), "--out", "out"]
    for i in range(2):
        cli.main([*base, "--shard", f"{i}/2"])
    capsys.readouterr()
    cli.main([*base, "--assemble"])
    out = capsys.readouterr().out
    report = json.loads(out[out.index("{"):])
    assert report["n_hypotheses"] == 2 and report["n_accepted"] == 1
    (ns_dir,) = (root / "out" / "mini").iterdir()
    table = [json.loads(line) for line in (ns_dir / "lr_table.jsonl").read_text().splitlines()]
    assert {r["program_canonical"]: r["accepted"] for r in table}[INFORMATIVE] is True
    assert (ns_dir / "report.json").exists() and list(ns_dir.glob("eps_cal_shard*"))
