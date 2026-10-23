"""T1 data pipeline tests on a tiny synthetic AMLworld-format fixture (built in tmp_path).

The real-HI-Small checks at the bottom are marked slow and skip when the output is absent.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ftm import guards
from ftm.data import alias, io, patterns, splits
from ftm.data.build import build_dataset

TRANS = """Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
2022/09/01 00:10,001,A1,002,B1,100.00,US Dollar,100.00,US Dollar,ACH,0
2022/09/01 01:00,0001,A1,002,B1,50.00,US Dollar,50.00,US Dollar,ACH,0
2022/09/01 02:00,002,B1,003,C1,10.00,Euro,10.00,Euro,Wire,1
2022/09/01 03:00,003,C1,001,A1,10.00,Euro,10.00,Euro,Wire,1
2022/09/02 05:00,001,A1,003,C1,5.50,Euro,5.50,Euro,ACH,0
2022/09/02 05:00,001,A1,003,C1,5.50,Euro,5.50,Euro,ACH,1
2022/09/03 06:00,004,D1,001,A1,7.00,Euro,7.00,Euro,ACH,1
2022/09/04 07:00,004,D1,002,B1,9.00,Euro,9.00,Euro,ACH,1
2022/09/04 08:00,001,A1,005,Z9,3.00,Euro,3.00,Euro,Cash,0
"""
ACCOUNTS = """Bank Name,Bank ID,Account Number,Entity ID,Entity Name
Bank #1,1,A1,E1,Corporation #1
Bank #2,2,B1,E2,Corporation #2
Bank #3,3,C1,E3,Corporation #3
Bank #4,4,D1,E4,Corporation #4
"""
PATTERNS = """BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 2 hops
2022/09/01 02:00,002,B1,003,C1,10.00,Euro,10.00,Euro,Wire,1
2022/09/01 03:00,003,C1,001,A1,10.00,Euro,10.00,Euro,Wire,1
END LAUNDERING ATTEMPT - CYCLE

BEGIN LAUNDERING ATTEMPT - GATHER-SCATTER:  Max 2-degree Fan-In
2022/09/02 05:00,001,A1,003,C1,5.50,Euro,5.50,Euro,ACH,1
2022/09/03 06:00,004,D1,001,A1,7.00,Euro,7.00,Euro,ACH,1
2022/09/04 07:00,004,D1,002,B1,9.00,Euro,9.00,Euro,ACH,1
2022/09/04 09:00,004,D1,002,B1,99.00,Euro,99.00,Euro,ACH,1
END LAUNDERING ATTEMPT - GATHER-SCATTER

BEGIN LAUNDERING ATTEMPT - STACK
2022/09/01 03:00,003,C1,001,A1,10.00,Euro,10.00,Euro,Wire,1
END LAUNDERING ATTEMPT - STACK
"""
SPLIT_DAYS = {"train": [1, 2], "val": [3, 3], "test": [4, 4]}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("ds")
    raw = root / "raw"
    raw.mkdir()
    (raw / "TEST_Trans.csv").write_text(TRANS)
    (raw / "TEST_accounts.csv").write_text(ACCOUNTS)
    (raw / "TEST_Patterns.txt").write_text(PATTERNS)
    cfg = {
        "name": "TEST", "raw_dir": "raw", "parquet_dir": "pq", "views_dir": "views",
        "n_banks": 2, "split_days": SPLIT_DAYS, "alias_seeds": [0, 1],
    }  # fmt: skip
    manifest = build_dataset(cfg, root=root, log=lambda *_: None)
    return {"root": root, "pq": root / "pq", "views": root / "views", "manifest": manifest, "cfg": cfg}


def tx(built, **kw):
    return io.scan_transactions(built["pq"], **kw).collect()


def test_tx_id_is_csv_row_index_and_dtypes(built):
    df = pl.read_parquet(built["pq"] / "transactions.parquet")
    assert df["tx_id"].to_list() == list(range(9))
    assert df.schema["tx_id"] == pl.Int64 and df.schema["day"] == pl.Int32
    assert df.schema["timestamp"] == pl.Datetime("us")
    assert df.schema["is_laundering"] == pl.Boolean
    assert df.schema["from_bank"] == pl.Int64 and df.schema["src_id"] == pl.Int64
    assert df.schema["amount_received"] == pl.Float64
    assert df["day"].to_list() == [1, 1, 1, 1, 2, 2, 3, 4, 4]


def test_bank_leading_zeros_are_one_bank(built):
    df = pl.read_parquet(built["pq"] / "transactions.parquet")
    assert df["from_bank"].to_list()[:2] == [1, 1]
    assert df["src_id"][0] == df["src_id"][1]


def test_accounts_include_unlisted_accounts(built):
    acc = io.read_accounts(built["pq"])
    assert acc.columns == ["node_id", "bank", "account", "entity_id", "entity_name"]
    assert acc["node_id"].to_list() == list(range(acc.height))
    assert built["manifest"]["n_accounts_not_in_accounts_csv"] == 1
    z9 = acc.filter(pl.col("account") == "Z9")
    assert z9["bank"].item() == 5 and z9["entity_id"].item() is None
    df = pl.read_parquet(built["pq"] / "transactions.parquet")
    assert df["dst_id"].null_count() == 0


def test_build_is_deterministic_and_idempotent(built, tmp_path):
    again = build_dataset(built["cfg"], root=built["root"], log=lambda *_: None)
    assert again["build_key"] == built["manifest"]["build_key"]
    a = pl.read_parquet(built["pq"] / "transactions.parquet")
    build_dataset(built["cfg"], root=built["root"], force=True, log=lambda *_: None)
    assert a.equals(pl.read_parquet(built["pq"] / "transactions.parquet"))


def test_pattern_parsing_and_matching(built):
    pat = pl.read_parquet(built["pq"] / "patterns.parquet")
    st = built["manifest"]["patterns"]
    assert st["n_pattern_lines"] == 7 and st["n_attempts"] == 3
    assert st["n_unmatched_lines"] == 2  # the 99.00 line, and the 2nd use of an already-used tx
    # multi-match: the duplicated 5.50 transaction maps to the laundering one (tx 5), not tx 4
    assert st["n_lines_matching_multiple_tx"] == 1
    by_attempt = {a: sorted(g["tx_id"]) for (a,), g in pat.group_by("attempt_id")}
    assert by_attempt[0] == [2, 3] and by_attempt[1] == [5, 6, 7]
    assert pat.filter(pl.col("tx_id") == 5)["typology"].item() == "gather_scatter"
    assert set(pat["typology"]) <= set(patterns.TYPOLOGIES)
    # tx 6, 7 laundering but tx 4 is not; every laundering tx is in an attempt here
    assert st["n_laundering_tx_not_in_any_attempt"] == 0


def test_attempts_table(built):
    att = pl.read_parquet(built["pq"] / "attempts.parquet").sort("attempt_id")
    assert att["typology"].to_list() == ["cycle", "gather_scatter", "stack"]
    assert att["spans_split"].to_list() == [False, True, False]
    assert att["splits"][1].to_list() == ["test", "train", "val"]
    assert att["n_tx"].to_list() == [2, 3, 0]
    assert att["first_timestamp"][1] == datetime(2022, 9, 2, 5, 0)


def test_split_assignment(built):
    df = pl.read_parquet(built["pq"] / "transactions.parquet")
    assert df["split"].to_list() == ["train"] * 6 + ["val", "test", "test"]
    assert built["manifest"]["per_split"]["test"] == {"n_tx": 2, "n_laundering": 1}
    with pytest.raises(ValueError):
        splits.validate_split_days({"train": [1, 2], "val": [4, 5], "test": [6, 7]})
    with pytest.raises(ValueError):
        splits.validate_split_days(None)


def test_test_part_marks_bulk_and_tail():
    cfg = {"split_days": {"train": [1, 3], "val": [4, 4], "test": [5, 9]}, "test_bulk_days": [5, 6], "test_tail_days": [7, 9]}
    df = pl.DataFrame({"day": list(range(1, 10)), "tx_id": list(range(9))})
    out = df.with_columns(splits.test_part(cfg).alias("p"))
    assert out["p"].to_list() == [None] * 4 + ["bulk"] * 2 + ["tail"] * 3
    assert out["tx_id"].to_list() == df["tx_id"].to_list() and df.columns == ["day", "tx_id"]  # a pure expression: no schema change
    with pytest.raises(ValueError):
        splits.test_part({**cfg, "test_tail_days": [8, 9]})  # gap between bulk and tail
    with pytest.raises(ValueError):
        splits.test_part({**cfg, "test_bulk_days": [5, 6], "test_tail_days": [7, 8]})  # does not reach the end of test


@pytest.mark.parametrize("name,bulk,tail", [("hi_small", [9, 10], [11, 18]), ("hi_medium", [13, 16], [17, 28])])
def test_real_configs_tile_the_test_split(name, bulk, tail):
    cfg = io.load_config(name)
    assert cfg["test_bulk_days"] == bulk and cfg["test_tail_days"] == tail
    days = pl.DataFrame({"day": list(range(1, cfg["split_days"]["test"][1] + 1))})
    p = days.with_columns(splits.test_part(cfg).alias("p"))
    assert p.filter(pl.col("day") <= cfg["split_days"]["val"][1])["p"].is_null().all()
    assert p["p"].drop_nulls().n_unique() == 2


def test_test_split_is_locked(built, monkeypatch):
    monkeypatch.delenv(guards.FINAL_RUN_ENV, raising=False)
    with pytest.raises(guards.TestSplitLocked):
        tx(built, split="test")
    with pytest.raises(guards.TestSplitLocked):
        tx(built, split=["train", "test"])
    with pytest.raises(guards.TestSplitLocked):
        io.scan_view(built["views"], 1, split="test")
    with pytest.raises(guards.TestSplitLocked):
        io.scan_patterns(built["pq"], split="test")
    with pytest.raises(guards.TestSplitLocked):
        io.scan_attempts(built["pq"], split="test")
    # a late as_of alone must not leak the test days
    late = tx(built, as_of=datetime(2030, 1, 1))
    assert "test" not in set(late["split"]) and late.height == 7
    monkeypatch.setenv(guards.FINAL_RUN_ENV, "1")
    assert tx(built, split="test").height == 2
    assert tx(built, as_of=datetime(2030, 1, 1)).height == 9


def test_readers_need_split_or_as_of(built):
    with pytest.raises(ValueError):
        tx(built)
    with pytest.raises(ValueError):
        io.scan_view(built["views"], 1)
    with pytest.raises(ValueError):
        tx(built, split="dev")


def test_time_guard_on_readers(built):
    df = tx(built, split="train", as_of="2022-09-01T02:00:00")
    assert df["timestamp"].max() == datetime(2022, 9, 1, 2, 0)
    assert df.height == 3
    v = io.scan_view(built["views"], 1, as_of=datetime(2022, 9, 1, 2, 0)).collect()
    assert v["timestamp"].max() <= datetime(2022, 9, 1, 2, 0)


def test_view_membership(built):
    banks = io.read_banks(built["views"])
    assert banks.columns == ["rank", "bank", "n_tx", "n_laundering"]
    assert banks["bank"].to_list() == [1, 2]  # 1: 7 tx; 2 and 3 tie at 4 tx, the lower bank id wins
    all_tx = pl.read_parquet(built["pq"] / "transactions.parquet")
    for b in banks["bank"]:
        v = pl.read_parquet(built["views"] / f"bank={b}.parquet")
        want = all_tx.filter((pl.col("from_bank") == b) | (pl.col("to_bank") == b))
        assert v["tx_id"].to_list() == want["tx_id"].to_list()
    assert banks["n_tx"].to_list() == [7, 4]
    assert banks["n_laundering"].to_list() == [3, 2]
    assert built["manifest"]["banks"]["n_tx_in_any_view"] == 9  # every tx touches bank 1 or 2


def test_alias_is_deterministic_and_unordered():
    ids = np.arange(5000)
    a = alias.make_alias_table(ids, seed=7)
    assert a.equals(alias.make_alias_table(ids[::-1], seed=7))  # input order does not matter
    assert not a.equals(alias.make_alias_table(ids, seed=8))
    assert a["alias"].n_unique() == 5000
    n = a["alias"].str.slice(1).map_elements(lambda s: int(s, 16), return_dtype=pl.Int64).to_numpy()
    assert abs(np.corrcoef(ids, n)[0, 1]) < 0.1  # no order leak
    assert (n == ids).mean() < 0.01


def test_apply_aliases_removes_raw_accounts(built):
    al = alias.read_aliases(built["pq"], 0)
    assert al.height == io.read_accounts(built["pq"]).height
    df = tx(built, split="train")
    out = alias.apply_aliases(df, al)
    assert {"src_alias", "dst_alias"} <= set(out.columns)
    assert not {"src_id", "dst_id", "from_account", "to_account"} & set(out.columns)
    assert out["tx_id"].to_list() == df["tx_id"].to_list()
    assert out["src_alias"].null_count() == 0
    lazy = alias.apply_aliases(df.lazy(), al).collect()
    assert lazy.equals(out)
    # the same account gets the same alias on both sides
    assert out["src_alias"][0] == out["src_alias"][1]


def test_heldout_cycles_switch(built):
    base = tx(built, split="train")
    held = tx(built, split="train", heldout_cycles=True)
    assert set(base["tx_id"]) - set(held["tx_id"]) == {2, 3}  # the cycle attempt
    assert tx(built, split="val", heldout_cycles=True).height == tx(built, split="val").height
    v = io.scan_view(
        built["views"], 1, split="train", heldout_cycles=True, parquet_dir=built["pq"]
    ).collect()
    assert not {2, 3} & set(v["tx_id"])
    # direct unit test of the function on a frame with a mixed split column
    frame = pl.DataFrame({"tx_id": [0, 1, 2], "split": ["train", "val", "train"]})
    pat = pl.DataFrame({"tx_id": [0, 1], "typology": ["cycle", "cycle"]})
    assert splits.drop_cycle_attempts(frame, pat)["tx_id"].to_list() == [1, 2]


# ---- real HI-Small output (slow; skipped when it has not been built) ----

REAL = Path(__file__).resolve().parents[1] / "data" / "parquet" / "hi_small"
needs_real = pytest.mark.skipif(not (REAL / "manifest.json").exists(), reason="HI-Small not built")


@pytest.mark.slow
@needs_real
def test_real_hi_small_manifest_and_tables():
    import json

    m = json.loads((REAL / "manifest.json").read_text())
    assert 5_000_000 < m["n_tx"] < 5_200_000 and 5_000 < m["n_laundering"] < 5_400
    assert sum(s["n_tx"] for s in m["per_split"].values()) == m["n_tx"]
    assert m["patterns"]["n_unmatched_lines"] == 0
    lf = pl.scan_parquet(REAL / "transactions.parquet")
    s = lf.select(
        pl.len(),
        (pl.col("tx_id") == pl.int_range(pl.len())).all().alias("ids"),
        pl.col("split").null_count().alias("nulls"),
        pl.col("src_id").null_count().alias("n1"),
    ).collect()
    assert s["len"].item() == m["n_tx"] and s["ids"].item() and s["nulls"].item() == 0
    assert s["n1"].item() == 0


@pytest.mark.slow
@needs_real
def test_real_hi_small_views_and_splits(monkeypatch):
    monkeypatch.delenv(guards.FINAL_RUN_ENV, raising=False)
    views_dir = REAL.parents[1] / "views" / "hi_small"
    banks = io.read_banks(views_dir)
    assert banks.height == 30 and banks["n_tx"].is_sorted(descending=True)
    top = banks["bank"][0]
    v = io.scan_view(views_dir, top, split="train").collect()
    assert ((v["from_bank"] == top) | (v["to_bank"] == top)).all()
    with pytest.raises(guards.TestSplitLocked):
        io.scan_transactions(REAL, split="test")
    train, val = (io.scan_transactions(REAL, split=s).select("day").collect() for s in ("train", "val"))
    assert train["day"].max() < val["day"].min()
