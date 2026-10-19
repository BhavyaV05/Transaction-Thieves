"""T4: eval library (metrics, cluster bootstrap, Holm). Synthetic data, plus one slow real-data check."""

from pathlib import Path

import numpy as np
import pytest

from ftm.eval import bootstrap as bs
from ftm.eval import metrics as mt
from ftm.eval import stats as st


def test_brier_and_skill():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.5, 0.5, 1.0])
    assert mt.brier(y, p) == pytest.approx(0.125)
    assert mt.brier_skill(y, np.full(4, 0.5)) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        mt.brier(y, np.array([0.1, np.nan, 0.2, 0.3]))


def test_gap_closed_and_tiny_denominator():
    assert mt.gap_closed(0.010, 0.007, 0.004) == pytest.approx(0.5)
    assert mt.gap_closed(0.010, 0.010, 0.004) == pytest.approx(0.0)
    assert np.isnan(mt.gap_closed(0.010, 0.009, 0.010 - 1e-7))


def test_pr_auc_matches_hand_computed_average_precision():
    # sorted by score: y = 1, 0, 1, 0 -> AP = 1/2 * (1/1) + 1/2 * (2/3)
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1])
    assert mt.pr_auc(y, p) == pytest.approx(0.5 + 1 / 3)
    # ties are one threshold: both rows of the tie enter together
    assert mt.pr_auc(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(0.5)
    assert np.isnan(mt.pr_auc(np.zeros(3), np.array([0.1, 0.2, 0.3])))


def test_pr_auc_random_scores_near_prevalence():
    rng = np.random.default_rng(0)
    y = (rng.random(200_000) < 0.01).astype(int)
    assert mt.pr_auc(y, rng.random(len(y))) == pytest.approx(0.01, abs=0.002)


def test_balanced_accuracy_caught_and_flips():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1])
    assert mt.balanced_accuracy(y, p, 0.5) == pytest.approx(0.5)
    c = mt.caught_at_capacity(y, p, day=np.array([1, 1, 2, 2]), per_day=1)
    assert c == {"reviewed": 2, "caught": 1, "positives": 2, "recall": 0.5}
    f = mt.flips(y, p, np.array([0.9, 0.9, 0.1, 0.1]), 0.5)
    assert f["wrong_to_right"] == 2 and f["right_to_wrong"] == 0


def test_ece_zero_when_calibrated_and_large_when_not():
    rng = np.random.default_rng(1)
    p = rng.random(100_000)
    y = (rng.random(len(p)) < p).astype(int)
    assert mt.ece(y, p) < 0.01
    assert mt.ece(y, np.clip(p + 0.3, 0, 1)) > 0.2


def test_cluster_ids_union_attempts_and_shared_accounts():
    att = [7, 7, None, None, None, 9]
    src = [1, 2, 3, 4, 5, 6]
    dst = [10, 11, 2, 12, 13, 5]  # row 2 pays account 2 (row 1's sender); row 5 pays account 5 (row 4)
    cl = bs.cluster_ids(att, [src, dst])
    assert cl[0] == cl[1] == cl[2]  # attempt 7, then account 2
    assert cl[4] == cl[5]  # account 5
    assert len({cl[0], cl[3], cl[4]}) == 3
    rep = bs.cluster_report(cl)
    assert rep["n_clusters"] == 3 and rep["largest"] == 3 and rep["singletons"] == 1


def test_hub_cap_breaks_giant_component_but_keeps_attempts():
    # account 99 is a hub that every row touches; rows 0 and 1 also share attempt 3
    att = [3, 3, None, None, None]
    src = [1, 2, 4, 5, 6]
    dst = [99, 99, 99, 99, 99]
    assert len(set(bs.cluster_ids(att, [src, dst]))) == 1
    cl = bs.cluster_ids(att, [src, dst], hub_cap=3)
    assert cl[0] == cl[1] and len(set(cl)) == 4


def test_cluster_ids_attempt_and_account_keys_do_not_collide():
    cl = bs.cluster_ids([5, None], [[1, 5]])  # attempt 5 and account 5 are different things
    assert cl[0] != cl[1]


def test_bootstrap_means_match_plain_resampling():
    rng = np.random.default_rng(2)
    v = rng.random(50)
    cl = np.repeat(np.arange(10), 5)
    fast = bs.bootstrap_means(v, cl, n_boot=4000, seed=3)[:, 0]
    slow = bs.cluster_bootstrap(lambda idx: v[idx].mean(), cl, n_boot=4000, seed=4)
    assert fast.mean() == pytest.approx(v.mean(), abs=0.01)
    assert fast.std() == pytest.approx(slow.std(), rel=0.15)


def test_cluster_ci_wider_than_row_ci_when_rows_are_correlated():
    rng = np.random.default_rng(5)
    k, m = 200, 20  # 200 clusters of 20 near-identical rows
    base = np.repeat(rng.normal(size=k), m)
    v = base + 0.05 * rng.normal(size=k * m)
    cl = np.repeat(np.arange(k), m)
    w_cluster = np.diff(bs.percentile_ci(bs.bootstrap_means(v, cl, 2000)[:, 0]))[0]
    w_row = np.diff(bs.percentile_ci(bs.bootstrap_means(v, np.arange(k * m), 2000)[:, 0]))[0]
    assert w_cluster > 3 * w_row


def test_brier_diff_and_gap_closed_ci():
    rng = np.random.default_rng(6)
    n = 5000
    y = (rng.random(n) < 0.2).astype(int)
    p_upper = np.clip(0.2 + 0.6 * (y - 0.2) + 0.05 * rng.normal(size=n), 0.01, 0.99)
    p_one = np.full(n, 0.2)
    p_half = (p_one + p_upper) / 2
    cl = np.arange(n) // 5
    d = bs.brier_diff(y, p_one, p_upper, cl, n_boot=2000)
    assert d["diff"] > 0 and d["ci"][0] > 0
    g = bs.gap_closed_ci(y, p_one, p_half, p_upper, cl, n_boot=2000)
    lo, hi = g["ci"]
    assert lo < g["gap_closed"] < hi
    assert 0.6 < g["gap_closed"] < 0.85  # squared error: half-way in p closes ~75% of the Brier gap
    assert g["nan_share"] == 0


def test_bootstrap_pvalue():
    assert bs.bootstrap_pvalue(np.full(999, 1.0)) == pytest.approx(1 / 1000)
    assert bs.bootstrap_pvalue(np.full(999, -1.0)) == pytest.approx(1.0)
    assert bs.bootstrap_pvalue(np.full(999, -1.0), alternative="less") == pytest.approx(1 / 1000)


def test_holm_known_example():
    r = st.holm({"H1": 0.01, "H2": 0.04, "H3": 0.03, "H4": 0.005, "H9": 0.2})
    # sorted: H4 .005 (x5), H1 .01 (x4), H3 .03 (x3) -> .09 > .05 stops here
    assert [r[h]["reject"] for h in ("H4", "H1", "H3", "H2", "H9")] == [True, True, False, False, False]
    assert r["H4"]["p_adj"] == pytest.approx(0.025)
    assert r["H1"]["p_adj"] == pytest.approx(0.04)
    assert r["H3"]["p_adj"] == pytest.approx(0.09)
    assert r["H2"]["p_adj"] == pytest.approx(0.09)  # monotone: max(0.09, 2 * 0.04)
    assert list(r) == ["H1", "H2", "H3", "H4", "H9"]


def test_fixed_sequence_stops_at_first_failure():
    r = st.fixed_sequence({"H1": 0.01, "H2": 0.2, "H3": 0.001})
    assert r["H1"]["reject"] and not r["H2"]["reject"] and not r["H3"]["reject"]
    assert r["H2"]["tested"] and not r["H3"]["tested"]


def test_equivalence_and_non_inferiority():
    m = st.equivalence_margin(brier_best_simple=0.010, brier_random=0.014)
    assert m == pytest.approx(0.002)
    assert st.equivalent((-0.001, 0.0015), m)
    assert not st.equivalent((-0.001, 0.0025), m)
    assert st.non_inferior((-0.003, 0.0019), m)


P0_FILE = Path("data/scores/hi_small/p0.parquet")


@pytest.mark.slow
@pytest.mark.skipif(not P0_FILE.exists(), reason="needs T1's HI-Small p0 (data/scores/hi_small/p0.parquet)")
def test_real_hi_small_gap_matches_t1_report():
    """Same rows as ftm/baselines/README.md 'Our split': Brier 0.001167 one-bank vs 0.000906 see-everything."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("eval_report", "scripts/eval_report.py")
    er = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(er)
    out = er.report(er.load_frame("hi_small", "rows"), n_boot=500, seed=0)
    assert out["one_bank"]["brier"] == pytest.approx(0.001167, abs=2e-6)
    assert out["see_everything"]["brier"] == pytest.approx(0.000906, abs=2e-6)
    row = out["brier_diff_ob_minus_se"]["row"]["ci95"]
    clus = out["brier_diff_ob_minus_se"]["attempt+accounts"]["ci95"]
    assert clus[1] - clus[0] >= row[1] - row[0]  # clustering can only widen the interval here


def test_eval_report_on_synthetic_frame():
    import importlib.util

    import polars as pl

    spec = importlib.util.spec_from_file_location("eval_report", "scripts/eval_report.py")
    er = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(er)
    rng = np.random.default_rng(7)
    n = 400
    y = (rng.random(n) < 0.1).astype(int)
    f = pl.DataFrame({
        "tx_id": np.arange(n), "y": y, "p_ob": np.full(n, 0.1),
        "p_se": np.clip(0.1 + 0.5 * (y - 0.1), 0.01, 0.99),
        "attempt_id": [int(i // 4) if yi else None for i, yi in enumerate(y)],
        "src_id": rng.integers(0, 300, n), "dst_id": rng.integers(300, 600, n),
    })  # fmt: skip
    out = er.report(f, n_boot=200, seed=0)
    assert out["n"] == n and out["one_bank"]["brier"] > out["see_everything"]["brier"]
    assert set(out["brier_diff_ob_minus_se"]) == {"attempt+accounts", "attempt+tx", "row"}
    assert out["brier_diff_ob_minus_se"]["row"]["clusters"]["n_clusters"] == n
