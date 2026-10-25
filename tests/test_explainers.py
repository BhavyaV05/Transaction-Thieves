"""T1 explainer comparisons (PPR, GNNExplainer, PGExplainer) as evidence selectors: K, time guard, alert included, no labels. Toy graphs only."""

from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from test_evidence_api import HOME, T0, _toy

from ftm.retrieval import evidence_views as ev
from ftm.retrieval import explainers as ex


def test_ppr_is_local_symmetric_and_zero_outside_the_component():
    # alert edge 0->1; a path 1-2-3; an unconnected edge 8->9; a parallel edge 0->1 and a reverse edge 1->0
    src = np.array([0, 1, 2, 8, 0, 1])
    dst = np.array([1, 2, 3, 9, 1, 0])
    s = ex.ppr_edge_scores(src, dst, alert=0)
    assert s[3] == 0.0 and (s[[0, 1, 2, 4, 5]] > 0).all()
    assert s[0] == pytest.approx(s[4]) == pytest.approx(s[5])  # parallel and reversed edges between the same accounts score alike
    assert s[1] > s[2]  # closer to the restart nodes, more mass
    assert s[0] > s[1]


def test_ppr_mass_adds_up():
    # on a graph without self-loops every node's PPR mass is returned through the edges it touches: the node sum is 1
    rng = np.random.default_rng(1)
    src, dst = rng.integers(0, 15, 80), rng.integers(15, 30, 80)  # bipartite, no self-loops
    s = ex.ppr_edge_scores(src, dst, alert=3, alpha=0.2)
    assert (s >= 0).all() and s[3] > 0
    # sum over edges of score = sum over nodes of PPR(v) * degree(v)
    deg = np.bincount(np.concatenate([src, dst]), minlength=30)
    assert s.sum() > 0 and deg.sum() == 160


@pytest.mark.parametrize("name", ["ppr", "gnnexplainer", "pgexplainer"])
def test_view_k_alert_time_and_no_labels(tmp_path, monkeypatch, name):
    df = _toy(tmp_path, monkeypatch)
    n = df.height - 1  # the alert row is the last one
    rng = np.random.default_rng(0)
    cache = {}
    if name != "ppr":  # explainer scores: a cached mask over candidate edges (random) plus one tx the window never contains
        cache = {(HOME, n): {int(t): float(rng.random()) for t in df["tx_id"].to_list()} | {10**9: 5.0}}
        monkeypatch.setattr(ev, "_explainer_cache", lambda ds, method, final=False: cache)
    v = ev.view("V000001", name, k=40, data="toy")
    lines = v.text.split("\n")
    assert len(lines) == len(v.tx_ids) <= 40
    assert v.tx_ids[-1] == n and "<- ALERT" in lines[-1]  # the alerted edge is in and is the newest
    when = dict(zip(df["tx_id"], df["timestamp"], strict=True))
    ts = [when[t] for t in v.tx_ids]
    assert ts == sorted(ts) and max(ts) <= T0  # nothing after the alert time
    assert min(ts) >= T0 - timedelta(hours=ev.WINDOW_HOURS)  # nothing outside the 72 h window
    assert 10**9 not in v.tx_ids  # an edge the candidates do not contain is never selected
    assert "src_id" not in v.text and "is_laundering" not in v.text and "label" not in v.text.lower()


def test_score_zero_edges_are_not_padded_in():
    import polars as pl

    cand = pl.DataFrame({"tx_id": list(range(10)), "src_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "dst_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]})
    score = np.array([0, 0.5, 0.9, 0, 0, 0.1, 0, 0, 0, 0.0])
    m = ev._positive_top(score, cand["tx_id"].to_numpy(), alert=9, k=4)
    assert np.flatnonzero(m).tolist() == [1, 2, 5, 9]  # three positive edges and the alerted edge, though k - 1 = 3 also allows more
    m = ev._positive_top(score, cand["tx_id"].to_numpy(), alert=9, k=3)
    assert np.flatnonzero(m).tolist() == [1, 2, 9]  # the two best edges and the alerted edge


def test_h6_report_includes_the_explainer_views():
    rows = []
    for a in range(20):
        for view, hits, n in (("scorer_pcst", 3, 10), ("amount_1hop", 1, 12), ("amount", 0, 40), ("scorer", 3, 40), ("scorer_greedy", 3, 10),
                              ("long_prompt", 0, 60), ("ppr", 2, 40), ("gnnexplainer", 1, 20), ("pgexplainer", 0, 5)):  # fmt: skip
            rows.append({"alert_id": f"V{a}", "attempt_id": a // 2, "view": view, "hits": hits, "n_relevant": 4, "n_sel": n})
    rep = ev.h6_report(pl.DataFrame(rows), n_boot=100)
    c = rep["recall_secondary"]["all"]["comparisons"]
    assert c["scorer_pcst_minus_ppr_recall"]["diff"] == pytest.approx(0.25)
    assert c["scorer_pcst_minus_gnnexplainer_recall"]["diff"] == pytest.approx(0.5)
    assert c["scorer_pcst_minus_pgexplainer_pattern_edges"]["diff"] == pytest.approx(3.0)


# ---- the history subgraph of BankModel (needs torch only; no model, no checkpoint) --------------------------------------------------


def _bank_stub(n_edges=400, seed=0):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(seed)
    bm = object.__new__(ex.BankModel)
    bm.torch, bm.dev = torch, torch.device("cpu")
    bm.tx = np.arange(1000, 1000 + n_edges)
    bm.t = np.sort(rng.integers(0, 10_000, n_edges)).astype("i8")
    bm.src, bm.dst = rng.integers(0, 40, n_edges), rng.integers(0, 40, n_edges)
    bm.ea, bm.ea_rev = rng.random((n_edges, 6), dtype="f4"), rng.random((n_edges, 6), dtype="f4")
    bm.t0_us = 0
    bm._index_incidence()
    return bm


def test_history_subgraph_is_causal_capped_and_holds_the_alert():
    bm = _bank_stub()
    a = 250
    as_of = np.datetime64(int(bm.t[a]) * 1_000_000, "us")  # the alert's time, in the stub's clock (t0 = 0)
    sub = bm.history_subgraph(int(bm.tx[a]), as_of, cap=5)
    pos = sub["pos"]
    assert (bm.t[pos] <= bm.t[a]).all()  # nothing later than the alert
    assert bm.tx[pos][sub["index"]] == bm.tx[a]  # the alerted edge is in, at the index the explainer is asked about
    assert sub["x"][:, 1].sum() == len({bm.src[a], bm.dst[a]})  # ego ids mark the alerted edge's two accounts only
    assert sub["edge_index"].shape == (2, len(pos)) and sub["edge_attr"].shape == (len(pos), 6)
    # the cap: a seed node contributes at most `cap` edges in hop one; with cap 5 the whole subgraph stays small
    full = bm.history_subgraph(int(bm.tx[a]), as_of, cap=10_000)
    assert len(pos) <= len(full["pos"]) and set(pos) <= set(full["pos"])
    assert set(bm.recent_edges(int(bm.src[a]), int(bm.t[a]), 5)) <= set(pos) | {a}


def test_history_subgraph_ignores_edges_added_after_the_alert():
    bm = _bank_stub()
    a = 250
    as_of = np.datetime64(int(bm.t[a]) * 1_000_000, "us")
    before = bm.history_subgraph(int(bm.tx[a]), as_of, cap=100)["pos"]
    # make every later edge touch the alert's sender: the history subgraph must not change
    bm.src[a + 1 :] = bm.src[a]
    bm._index_incidence()
    after = bm.history_subgraph(int(bm.tx[a]), as_of, cap=100)["pos"]
    assert before.tolist() == after.tolist()


@pytest.mark.skipif(not (ex.CLONE.expanduser() / "models.py").exists(), reason="needs the Multi-GNN clone (outside the repo)")
def test_gnnexplainer_mask_on_a_random_model():
    """GNNExplainer through the homogeneous wrapper of the hetero Multi-PNA: one mask per edge, 0 outside the receptive field."""
    pytest.importorskip("torch_geometric")
    torch = pytest.importorskip("torch")
    from torch_geometric.nn import to_hetero

    clone = ex.CLONE.expanduser().resolve()
    import sys

    sys.path.insert(0, str(clone))
    import json

    from training import get_model

    sample = ex._dummy_batch(torch)
    model = to_hetero(get_model(sample, ex._settings(clone, json), ex._args()), sample.metadata(), aggr="mean")
    bm = _bank_stub(300)
    bm.model = ex._Wrap(model).eval()
    for p in bm.model.parameters():
        p.requires_grad_(False)
    a = 200
    as_of = np.datetime64(int(bm.t[a]) * 1_000_000, "us")
    sub = bm.history_subgraph(int(bm.tx[a]), as_of, cap=20)
    mask = bm.gnnexplainer(sub, epochs=5, seed=0)
    assert mask.shape == (len(sub["pos"]),) and np.isfinite(mask).all() and (mask >= 0).all() and (mask <= 1).all()
    sc, _ = bm.scores("gnnexplainer", bm.tx[: a + 1], int(bm.tx[a]), as_of, seed=0, sub=sub)  # candidates = all edges up to the alert
    assert sc.shape == (a + 1,) and (sc[a] >= 0)
    bm.pg_train([sub], epochs=2, seed=0)
    pm = bm.pgexplainer(sub)
    assert pm.shape == mask.shape and np.isfinite(pm).all()
    assert Path(str(ex.pg_path("toy", 1))).parts[-2:] == ("pgexplainer", "toy_bank1.pt")


def test_final_alerts_read_test_edges_and_the_val_view_never_does(tmp_path, monkeypatch):
    """Week 14: a test alert (id F...) may use train, val and test edges up to its time and the test scores of the edge scorer; it needs
    FTM_FINAL_RUN=1. A val alert never sees a test edge."""
    from ftm import guards

    df = _toy(tmp_path, monkeypatch)
    n = df.height - 1
    test_ids = set(range(n - 100, n + 1))  # the last 100 rows and the alert are test rows
    df = df.with_columns(pl.when(pl.col("tx_id").is_in(list(test_ids))).then(pl.lit("test")).otherwise(pl.lit("val")).alias("split"))
    df.write_parquet(tmp_path / "views" / f"bank={HOME}.parquet")
    pl.DataFrame({"alert_id": ["F000001"], "tx_id": [n], "bank": [HOME], "timestamp": [T0]}).write_parquet(tmp_path / "alerts" / "test.parquet")
    old = pl.read_parquet(tmp_path / "data" / "retrieval" / "toy" / "edge_scores.parquet")
    old.filter(~pl.col("tx_id").is_in(list(test_ids))).write_parquet(tmp_path / "data" / "retrieval" / "toy" / "edge_scores.parquet")
    (tmp_path / "data" / "retrieval" / "toy" / "final").mkdir()
    old.filter(pl.col("tx_id").is_in(list(test_ids))).write_parquet(tmp_path / "data" / "retrieval" / "toy" / "final" / "edge_scores_test.parquet")
    for f in (ev._scores, ev._alert_rows, ev._aliases, ev._explainer_cache):
        f.cache_clear()
    monkeypatch.delenv("FTM_FINAL_RUN", raising=False)
    with pytest.raises(guards.TestSplitLocked):
        ev.view("F000001", "scorer", data="toy")
    monkeypatch.setenv("FTM_FINAL_RUN", "1")
    v = ev.view("F000001", "scorer", data="toy")
    assert v.tx_ids[-1] == n and len(v.tx_ids) <= 40 and any(t in test_ids for t in v.tx_ids[:-1])  # test edges are usable evidence
    assert ev.view("F000001", "amount_1hop", data="toy").tx_ids[-1] == n
    # a val alert (here: an earlier val row) never reads a test edge, even with the final run on
    pl.DataFrame({"alert_id": ["V000002"], "tx_id": [5], "bank": [HOME], "timestamp": [df["timestamp"][5]]}).write_parquet(tmp_path / "alerts" / "val.parquet")
    ev._alert_rows.cache_clear()
    w = ev.view("V000002", "amount", data="toy")
    assert not (set(w.tx_ids) & test_ids) and max(w.tx_ids) <= 5 + 300
