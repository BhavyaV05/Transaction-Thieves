"""T1 DP-FedAvg: accounting is monotone, the edge-level features are per-transaction, and a tiny federated run learns / reports eps. Synthetic."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ftm.baselines import federated as fed


def test_epsilon_monotone_in_noise_and_steps():
    pytest.importorskip("opacus")
    e = [fed.rdp_epsilon(s, 0.01, 200) for s in (0.7, 1.0, 2.0, 4.0)]
    assert all(a > b for a, b in zip(e, e[1:], strict=False))
    assert fed.rdp_epsilon(1.0, 0.01, 400) > fed.rdp_epsilon(1.0, 0.01, 200)
    assert fed.rdp_epsilon(1.0, 0.02, 200) > fed.rdp_epsilon(1.0, 0.01, 200)
    assert fed.rdp_epsilon(1.0, 0.01, 200, copies=2) > fed.rdp_epsilon(1.0, 0.01, 200)  # a record in two views
    assert fed.rdp_epsilon(0.0, 0.01, 200) == float("inf")


def test_noise_for_epsilon_hits_the_budget_and_is_monotone():
    pytest.importorskip("opacus")
    sig = [fed.noise_for_epsilon(eps, 0.01, 200, copies=2) for eps in (1, 2, 4, 8)]
    assert all(a > b for a, b in zip(sig, sig[1:], strict=False))
    for eps, s in zip((1, 2, 4, 8), sig, strict=True):
        got = fed.rdp_epsilon(s, 0.01, 200, copies=2)
        assert got <= eps and got > 0.9 * eps
    # client level: Gaussian mechanism over R rounds, q = 1
    assert fed.noise_for_epsilon(2, 1.0, 20) > fed.noise_for_epsilon(8, 1.0, 20)


def _frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    t0 = datetime(2022, 9, 1)
    return pl.DataFrame({
        "tx_id": np.arange(n), "timestamp": [t0 + timedelta(minutes=int(m)) for m in np.sort(rng.integers(0, 3000, n))],
        "from_bank": 1, "to_bank": rng.integers(1, 3, n), "src_id": rng.integers(0, 8, n), "dst_id": rng.integers(0, 8, n),
        "amount_received": rng.uniform(1, 1e4, n), "amount_paid": rng.uniform(1, 1e4, n),
        "receiving_currency": rng.choice(["USD", "EUR"], n), "payment_currency": rng.choice(["USD", "EUR"], n),
        "payment_format": rng.choice(["ACH", "Cheque", "Wire"], n),
    })  # fmt: skip


def test_tx_features_depend_on_the_transaction_only_ctx_features_do_not():
    f = _frame()
    codes = fed.category_codes([f])
    full, part = fed.tx_features(f, codes), fed.tx_features(f.filter(pl.col("tx_id") % 2 == 0), codes)
    assert np.array_equal(full[::2], part)  # dropping other transactions changes nothing: the edge-level premise
    assert full.shape[1] == 5 + sum(len(v) for v in codes.values())
    cfull, cpart = fed.ctx_features(f, codes), fed.ctx_features(f.filter(pl.col("tx_id") % 2 == 0), codes)
    assert not np.array_equal(cfull[::2], cpart)  # neighbourhood aggregates do depend on other rows


def _clients(k=4, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for b in range(k):
        X = rng.normal(size=(n, 6)).astype("f4")
        y = (rng.random(n) < 1 / (1 + np.exp(-(3 * X[:, 0] - 6)))).astype("f4")  # feature 0 carries the signal
        m = n // 3
        out.append({"bank": b, "Xtr": X[:m], "ytr": y[:m], "Xva": X[m:], "yva": y[m:], "tx_va": np.arange(n - m),
                    "part_va": np.where(np.arange(n - m) < (n - m) // 2, "fit", "eval")})  # fmt: skip
    return out


@pytest.mark.parametrize("unit", ["edge", "client"])
def test_run_federated_learns_and_reports_epsilon(unit):
    pytest.importorskip("opacus")
    pytest.importorskip("flwr")
    c = _clients()
    r = fed.run_federated(c, unit, None, rounds=6, local_steps=40, q=0.1, lr=0.5, pos_weight=5.0)
    assert r["epsilon"] == float("inf") and r["sigma"] == 0
    assert fed.evaluate(c, r["raw"])["brier_skill"] > 0
    p = fed.run_federated(c, unit, 4.0, rounds=6, local_steps=40, q=0.1, lr=0.5, pos_weight=5.0)
    assert 0 < p["epsilon"] <= 4.0 and p["sigma"] > 0 and p["delta"] == fed.DELTA


def test_final_run_scores_test_rows_without_changing_the_val_model():
    """Week 14: the test rows ride along (`Xte`); training uses train rows only, so the val raw scores are bit-identical, and the Platt
    calibrator is fit on the val `fit` rows only (changing a test label or score cannot change it)."""
    pytest.importorskip("opacus")
    pytest.importorskip("flwr")
    c = _clients()
    rng = np.random.default_rng(1)
    ct = [{**x, "Xte": rng.normal(size=(500, 6)).astype("f4"), "yte": (rng.random(500) < 0.1).astype("f4"), "tx_te": np.arange(500)} for x in c]
    a = fed.run_federated(c, "client", 4.0, rounds=4, local_steps=20, q=0.1, lr=0.5, pos_weight=5.0, seed=3)
    b = fed.run_federated(ct, "client", 4.0, rounds=4, local_steps=20, q=0.1, lr=0.5, pos_weight=5.0, seed=3)
    assert a["raw_test"] is None and len(b["raw_test"]) == len(c)
    for x, y in zip(a["raw"], b["raw"], strict=True):
        np.testing.assert_array_equal(x, y)
    p = fed.test_p0(ct, b["raw"], b["raw_test"])
    assert p.shape == (500 * len(c),) and ((p >= 0) & (p <= 1)).all()
    # the calibrator sees val `fit` rows only: perturbing the val `eval` labels leaves the test p0 unchanged
    ct2 = [{**x, "yva": np.where(x["part_va"] == "eval", 1 - x["yva"], x["yva"])} for x in ct]
    np.testing.assert_array_equal(p, fed.test_p0(ct2, b["raw"], b["raw_test"]))
