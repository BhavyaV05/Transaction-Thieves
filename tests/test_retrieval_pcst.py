"""T1 PCST evidence views: time-respecting constraint, no edge after as_of, aliasing. Synthetic data only."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ftm.data import alias
from ftm.retrieval import evidence_views as ev
from ftm.retrieval import pcst


def test_reachable_only_along_time_respecting_walks():
    #        A->B t1,  B->C t2,  C->D t3 (alert),  Z->B t2.5 (too late for B->C),  Y->A t0.5
    src = np.array([0, 1, 2, 9, 8])
    dst = np.array([1, 2, 3, 1, 0])
    t = np.array([1.0, 2.0, 3.0, 2.5, 0.5])
    r = pcst.reachable_to_alert(src, dst, t, alert=2)
    assert r.tolist() == [True, True, True, False, True]
    assert pcst.is_one_trail(src, dst, t, 2, np.array([True, True, True, False, True]))
    assert not pcst.is_one_trail(src, dst, t, 2, np.array([True, True, True, True, False]))  # Z->B never reaches the alert in time
    assert pcst.is_connected(src, dst, np.array([True, True, True, False, True]))
    assert not pcst.is_connected(src, dst, np.array([True, False, True, False, False]))  # A->B and C->D are apart


def test_ties_in_time_can_chain():
    src, dst, t = np.array([0, 1, 2]), np.array([1, 2, 3]), np.array([5.0, 5.0, 5.0])
    assert pcst.reachable_to_alert(src, dst, t, alert=2).all()


def test_greedy_trail_is_always_a_trail_and_respects_k():
    rng = np.random.default_rng(0)
    n = 3000
    src, dst, t = rng.integers(0, 300, n), rng.integers(0, 300, n), rng.integers(0, 1000, n).astype(float)
    prize = rng.random(n)
    a = int(np.argmax(t))
    sel = pcst.greedy_trail(src, dst, t, prize, a, k=40)
    assert sel.sum() == 40 and pcst.is_one_trail(src, dst, t, a, sel)
    assert (t[sel] <= t[a]).all()


def test_pcst_picks_the_high_prize_path_through_cheap_connectors():
    pytest.importorskip("pcst_fast")
    try:
        pcst.check_binding()
    except RuntimeError as e:
        pytest.skip(str(e))  # CI's gnn extra installs the unpatched wheel
    # alert 0->1, then 1->2 (low prize connector), 2->3 (high), 3->4 (high); far edge 7->8 high prize but unreachable
    src = np.array([0, 1, 2, 3, 7])
    dst = np.array([1, 2, 3, 4, 8])
    prize = np.array([0.0, 0.01, 0.9, 0.9, 0.9])
    m = pcst.pcst_edges(src, dst, prize, alert=0, edge_cost=0.05)
    assert m[:4].all() and not m[4]
    t = np.array([4.0, 3.0, 2.0, 1.0, 1.0])
    sel, pc = pcst.select_pcst_trail(src, dst, t, prize, 0, k=4)
    assert sel[:4].all() and not sel[4] and pc[:4].all()


def test_binding_guard_fires_on_garbage(monkeypatch):
    pytest.importorskip("pcst_fast")
    import pcst_fast

    pcst.check_binding.cache_clear()
    monkeypatch.setattr(pcst_fast, "pcst_fast", lambda *a: (np.zeros(4, np.int64), np.zeros(3, np.int64)))
    with pytest.raises(RuntimeError, match="pcst_fast_binding.patch"):
        pcst.check_binding()
    pcst.check_binding.cache_clear()


def _view_dir(tmp_path):
    t0 = datetime(2022, 9, 1)
    rows = []
    for i, (h, split) in enumerate([(0, "train"), (30, "train"), (60, "val"), (70, "val"), (80, "val"), (200, "test")]):
        rows.append({"tx_id": i, "timestamp": t0 + timedelta(hours=h), "src_id": i, "dst_id": i + 1, "from_bank": 1, "to_bank": 2,
                     "amount_paid": 10.0 + i, "payment_currency": "USD", "payment_format": "ACH", "split": split, "is_laundering": False})  # fmt: skip
    pl.DataFrame(rows).write_parquet(tmp_path / "bank=1.parquet")
    return {"views_dir": str(tmp_path)}


def test_candidates_stop_at_as_of_and_window_and_never_return_test(tmp_path, monkeypatch):
    monkeypatch.delenv("FTM_FINAL_RUN", raising=False)
    cfg = _view_dir(tmp_path)
    as_of = datetime(2022, 9, 1) + timedelta(hours=75)
    c = ev.candidates(cfg, 1, as_of, window_hours=72)
    assert c["tx_id"].to_list() == [1, 2, 3]  # 0 is older than 72 h, 4 is after as_of, 5 is test
    assert c["timestamp"].max() <= as_of
    late = ev.candidates(cfg, 1, datetime(2022, 9, 20), window_hours=24 * 30)
    assert 5 not in late["tx_id"].to_list()  # a late as_of cannot reach the test split


def test_select_views_k_alert_included_and_aliased(tmp_path):
    pytest.importorskip("pcst_fast")
    try:
        pcst.check_binding()
    except RuntimeError as e:
        pytest.skip(str(e))
    rng = np.random.default_rng(1)
    n, t0 = 400, datetime(2022, 9, 1)
    cand = pl.DataFrame({
        "tx_id": np.arange(n), "timestamp": [t0 + timedelta(minutes=int(m)) for m in np.sort(rng.integers(0, 4000, n))],
        "src_id": rng.integers(0, 40, n), "dst_id": rng.integers(0, 40, n), "from_bank": 1, "to_bank": 2,
        "amount_paid": rng.uniform(1, 1000, n), "payment_currency": "USD", "payment_format": "ACH",
    })  # fmt: skip
    i = n - 1
    masks = ev.select_views(cand, i, rng.random(n) ** 3, k=40)
    assert set(masks) == set(ev.VIEWS) - set(ev.EXPLAINER_VIEWS)  # the explainer views need their scores (`explain=`)
    t = cand["timestamp"].dt.epoch("s").to_numpy().astype("f8")
    for name, m in masks.items():
        assert m[i] and (m.sum() <= 40 or name == "long_prompt") and (t[m] <= t[i]).all(), name
    src, dst = cand["src_id"].to_numpy(), cand["dst_id"].to_numpy()
    assert pcst.is_one_trail(src, dst, t, i, masks["scorer_pcst"]) and pcst.is_one_trail(src, dst, t, i, masks["scorer_greedy"])
    al = alias.make_alias_table(np.arange(40), seed=0)
    tr = ev.format_trail(cand, masks["scorer_pcst"], al)
    assert tr.columns == ev.TRAIL_COLS and tr["timestamp"].is_sorted()
    assert not {"src_id", "dst_id", "is_laundering"} & set(tr.columns)
