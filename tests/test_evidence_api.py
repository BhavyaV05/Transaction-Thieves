"""T1 `ftm.retrieval.evidence_views.view`: the four evidence views behind one stable call, on a toy dataset in the real layout."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ftm.data import alias, io
from ftm.retrieval import evidence_views as ev
from ftm.retrieval import views as stub_views

HOME, T0 = 70, datetime(2022, 9, 5, 12, 0)


def _toy(tmp_path, monkeypatch, n=300):
    rng = np.random.default_rng(0)
    ts = sorted(T0 - timedelta(minutes=int(m)) for m in rng.integers(1, 3 * 24 * 60 - 1, n))
    rows = [(i, t, int(rng.integers(0, 30)), int(rng.integers(0, 30)), HOME, int(rng.choice([HOME, 220])), float(rng.uniform(1, 9000)))
            for i, t in enumerate(ts)]  # fmt: skip
    rows.append((n, T0, 1, 2, HOME, 220, 777.0))  # the alert, last
    df = pl.DataFrame(rows, schema=["tx_id", "timestamp", "src_id", "dst_id", "from_bank", "to_bank", "amount_paid"], orient="row")
    m = df.height
    df = df.with_columns(
        pl.lit("US Dollar").alias("payment_currency"), pl.lit("ACH").alias("payment_format"), pl.lit("x").alias("from_account"),
        pl.lit("y").alias("to_account"), pl.lit([False] * m).alias("is_laundering"), pl.lit("val").alias("split"),
        pl.col("timestamp").dt.day().cast(pl.Int32).alias("day"),
    )  # fmt: skip
    views, pq, al = tmp_path / "views", tmp_path / "pq", tmp_path / "alerts"
    for d in (views, pq / "aliases", al, tmp_path / "data" / "retrieval" / "toy"):
        d.mkdir(parents=True)
    df.write_parquet(views / f"bank={HOME}.parquet")
    alias.make_alias_table(list(range(40)), seed=0).write_parquet(alias.alias_path(pq, 0))
    pl.DataFrame({"alert_id": ["V000001"], "tx_id": [n], "bank": [HOME], "timestamp": [T0]}).write_parquet(al / "val.parquet")
    pl.DataFrame({"alert_id": ["T000001"], "tx_id": [0], "bank": [HOME], "timestamp": [df["timestamp"][0]]}).write_parquet(al / "train.parquet")
    pl.DataFrame({"tx_id": np.arange(m), "score": rng.random(m), "split": ["val"] * m}).write_parquet(tmp_path / "data" / "retrieval" / "toy" / "edge_scores.parquet")
    cfg = {"views_dir": str(views), "parquet_dir": str(pq), "alerts_dir": str(al)}
    monkeypatch.setattr(io, "load_config", lambda name: cfg)
    monkeypatch.chdir(tmp_path)
    for f in (ev._scores, ev._alert_rows, ev._aliases):
        f.cache_clear()
    return df


def test_line_chars_is_exact():
    cand = pl.DataFrame({"amount_paid": [1234567.891, 5.0], "payment_currency": ["US Dollar", "Euro"], "payment_format": ["ACH", "Wire"],
                         "from_bank": [70, 70], "to_bank": [220, 70]})  # fmt: skip
    got = ev.line_chars(cand, 1, 70)
    real = []
    for i, r in enumerate(cand.iter_rows(named=True)):
        line = stub_views.format_trail(
            pl.DataFrame({**{k: [v] for k, v in r.items()}, "tx_id": [i], "timestamp": [T0], "src_alias": ["A0000001"], "dst_alias": ["A0000002"]}),
            alert_tx_id=1, home_bank=70)  # fmt: skip
        real.append(len(line) + 1)
    assert got.tolist() == real


@pytest.mark.parametrize("name", ["amount", "scorer", "scorer_pcst", "long_prompt", "amount_1hop", "scorer_greedy"])
def test_view_is_aliased_time_ordered_and_guarded(tmp_path, monkeypatch, name):
    if name == "scorer_pcst":
        pytest.importorskip("pcst_fast")
        from ftm.retrieval import pcst

        try:
            pcst.check_binding()
        except RuntimeError as e:
            pytest.skip(str(e))
    df = _toy(tmp_path, monkeypatch)
    v = ev.view("V000001", name, k=40, data="toy")
    lines = v.text.split("\n")
    assert len(lines) == len(v.tx_ids) and (len(v.tx_ids) <= 40 or name == "long_prompt")
    assert v.tx_ids[-1] == 300 and "<- ALERT" in lines[-1]  # the alerted edge is in and is the newest
    when = dict(zip(df["tx_id"], df["timestamp"], strict=True))
    ts = [when[t] for t in v.tx_ids]
    assert ts == sorted(ts) and max(ts) <= T0  # time-ordered, nothing after the alert
    assert "src_id" not in v.text and "is_laundering" not in v.text and v.text.count(" | ") >= 4 * len(lines)
    assert v.n_candidates > 40


def test_long_prompt_is_capped_at_the_budget(tmp_path, monkeypatch):
    _toy(tmp_path, monkeypatch, n=2000)
    v = ev.view("V000001", "long_prompt", data="toy")
    assert v.tokens_est <= ev.EVIDENCE_TOKENS + 1 and v.n_candidates > len(v.tx_ids) > 40
    small = ev.view("V000001", "long_prompt", data="toy", max_evidence_tokens=500)
    assert small.tokens_est <= 501 and set(small.tx_ids) <= set(v.tx_ids)
    assert small.tx_ids == v.tx_ids[-len(small.tx_ids):]  # the newest edges are kept


def test_view_resolves_alert_context_and_refuses_a_wrong_time(tmp_path, monkeypatch):
    from ftm.types import AlertContext

    _toy(tmp_path, monkeypatch)
    ok = AlertContext("V000001", str(HOME), T0.isoformat(), "val", 0.3, {})
    assert ev.view(ok, "amount", data="toy").tx_ids[-1] == 300
    bad = AlertContext("V000001", str(HOME), (T0 + timedelta(hours=1)).isoformat(), "val", 0.3, {})
    with pytest.raises(ValueError):
        ev.view(bad, "amount", data="toy")
    with pytest.raises(KeyError):
        ev.view("V999999", "amount", data="toy")
    with pytest.raises(ValueError):
        ev.view(ok, "nope", data="toy")


def test_h6_report_paired_difference_and_margin():
    rows = []
    for a in range(30):  # 30 alerts with 4 pattern edges; scorer+pcst finds 3, the stub finds 1, in a few attempts
        for view, hits in (("scorer_pcst", 3), ("amount_1hop", 1), ("amount", 0), ("scorer", 3), ("scorer_greedy", 3), ("long_prompt", 0)):
            rows.append({"alert_id": f"V{a}", "attempt_id": a // 3, "view": view, "hits": hits, "n_relevant": 4, "n_sel": 40 if a < 10 else 12})
    rep = ev.h6_report(pl.DataFrame(rows), n_boot=200)
    c = rep["all"]["comparisons"]["scorer_pcst_minus_amount_1hop_recall"]
    assert rep["all"]["alerts_with_pattern_edges"] == 30 and rep["all"]["attempts"] == 10
    assert rep["pool_1hop_at_least_k"]["alerts_with_pattern_edges"] == 10
    assert c["diff"] == pytest.approx(0.5) and c["meets_margin"] and c["ci95_attempts"][0] > 0.15
    assert rep["all"]["comparisons"]["scorer_pcst_minus_scorer_greedy_recall"]["diff"] == 0
