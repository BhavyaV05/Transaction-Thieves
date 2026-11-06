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


def test_cli_can_enumerate_the_search_space():
    spec = importlib.util.spec_from_file_location("backtest_cli2", Path(__file__).resolve().parents[1] / "scripts" / "backtest.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    ns = type("A", (), {"questions": ["enumerate"], "max_size": 3, "questions_file": None})()
    progs = cli.read_programs(ns)
    assert len(progs) == len(set(progs)) > 400
    assert any("ALERT.sender" in p for p in progs) and any("ALERT.receiver" in p for p in progs)


# --- ledger history across splits (rewrite finding 21) ----------------------------------------------------------------
@pytest.fixture(scope="module")
def two_split_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("ds2")
    w = make_world(260, seed=21, signal=1.0)
    full = w.full.with_row_index("tx_id").with_columns(pl.col("tx_id").cast(pl.Int64))
    cut = full["timestamp"].quantile(0.6, "nearest")
    full = full.with_columns(pl.when(pl.col("timestamp") <= cut).then(pl.lit("train")).otherwise(pl.lit("val")).alias("split"))
    pq, views, alerts = root / "pq", root / "views", root / "alerts"
    for d in (pq, views, alerts):
        d.mkdir()
    full.write_parquet(pq / "transactions.parquet")
    for b in (HOME, *PARTNERS, 5):
        full.filter((pl.col("from_bank") == b) | (pl.col("to_bank") == b)).write_parquet(views / f"bank={b}.parquet")
    for split in ("train", "val"):
        rows = []
        for a in w.alerts:
            b = a.binding
            m = full.filter((pl.col("src_id") == b.sender) & (pl.col("dst_id") == b.receiver) & (pl.col("timestamp") == b.t0))
            if m["split"][0] == split:
                rows.append(dict(alert_id=a.alert_id, tx_id=int(m["tx_id"][0]), bank=HOME, timestamp=b.t0, label=a.label, p0=0.3))
        pl.DataFrame(rows).write_parquet(alerts / f"{split}.parquet")
    cfg = root / "mini.yaml"
    cfg.write_text(yaml.safe_dump({"name": "mini", "parquet_dir": "pq", "views_dir": "views", "alerts_dir": "alerts"}))
    return root, cfg, full, cut


def test_the_val_ledger_holds_the_train_rows_too_but_an_answer_never_sees_past_its_cut(two_split_dataset):
    from ftm.dsl.executor import execute
    from ftm.dsl.parser import parse

    root, cfg, full, cut = two_split_dataset
    tr = load_world(str(cfg), "train", root=root)
    va = load_world(str(cfg), "val", root=root)
    assert va.alerts and tr.alerts
    for b, bank in va.banks.items():
        led = bank._ledger
        assert led["timestamp"].min() <= cut  # train history is there
        n_val = full.filter(((pl.col("from_bank") == b) | (pl.col("to_bank") == b)) & (pl.col("split") != "test")).height
        assert led.height == n_val
        assert max(t.banks[b]._ledger["timestamp"].max() for t in (tr,) if b in t.banks) <= cut
    # history question at the first val alert: it sees the train rows before the alert
    a = va.alerts[0]
    bank = va.banks[a.binding.partner_bank]
    node = parse("n_counterparties(ALERT.receiver, dir=in, window=before(72))")
    ctx = a.binding.exec_context(bank.bank, a.binding.t0)
    led = bank._ledger
    assert execute(node, led, ctx) == execute(node, led.filter(pl.col("timestamp") <= ctx.cutoff), ctx)
    later = led.head(5).with_columns(pl.lit(a.binding.t0 + timedelta(days=1)).alias("timestamp"),
                                     pl.lit(a.binding.receiver, dtype=led.schema["dst_id"]).alias("dst_id"), pl.lit(2**30, dtype=led.schema["src_id"]).alias("src_id"))
    assert execute(node, pl.concat([led, later]), ctx) == execute(node, led, ctx)  # a row after the cut is invisible
