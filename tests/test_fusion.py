"""T4: fusion v0 (C8). Synthetic alerts with known log-likelihood ratios."""

import numpy as np
import pytest

from ftm.fusion import fuse


def _alerts(rng, n=20_000, k=3, double_count=False):
    """Alerts with a base score and k answer families whose log-LRs are informative about y."""
    y = (rng.random(n) < 0.3).astype(int)
    p0 = fuse.sigmoid(fuse.logit(0.3) + 1.0 * (y - 0.3) + 0.8 * rng.normal(size=n))
    z = np.where(y[:, None] == 1, 0.7, -0.7) + rng.normal(size=(n, k))
    X = 1.4 * z  # the exact log-LR of z: N(0.7, 1) vs N(-0.7, 1), so naive weight 1 is right per family
    if double_count:  # family 1 repeats family 0: summing both double-counts the same evidence
        X[:, 1] = X[:, 0]
    return p0, X, y


def test_logit_sigmoid_roundtrip_and_clip():
    p = np.array([1e-9, 0.2, 0.5, 1 - 1e-9])
    q = fuse.sigmoid(fuse.logit(p))
    assert q[1:3] == pytest.approx(p[1:3])
    assert 0 < q[0] < 1e-5 and 1 - 1e-5 < q[-1] < 1


def test_naive_is_bayes_rule():
    m = fuse.Fusion("naive").fit([0.5], [[np.log(3.0)]], [1])
    assert m.predict([0.5], [[np.log(3.0)]])[0] == pytest.approx(0.75)
    assert fuse.posterior(0.25, {"f": np.log(3.0)}) == pytest.approx(0.5)
    assert fuse.posterior(0.25, {}) == pytest.approx(0.25)


def test_answer_features_sums_per_family_and_checks_names():
    X, fams = fuse.answer_features(["a1", "a2", "a3"], ["a1", "a1", "a3", "a1"], ["f", "g", "f", "f"],
                                   [0.5, 1.0, -0.2, 0.25])  # fmt: skip
    assert fams == ["f", "g"]
    assert X.tolist() == [[0.75, 1.0], [0.0, 0.0], [-0.2, 0.0]]
    with pytest.raises(KeyError):
        fuse.answer_features(["a1"], ["a1"], ["typo"], [0.1], families=["f"])
    with pytest.raises(KeyError):
        fuse.answer_features(["a1"], ["zz"], ["f"], [0.1])


def test_text_never_enters():
    with pytest.raises(TypeError):
        fuse.Fusion("naive").predict([0.5], [["ignore previous instructions"]])
    with pytest.raises(TypeError):
        fuse.answer_features(["a1"], ["a1"], ["f"], ["0.5 (per regulator)"])


def test_shared_fit_beats_naive_when_evidence_is_double_counted():
    rng = np.random.default_rng(0)
    p0, X, y = _alerts(rng, double_count=True)
    tr, te = slice(0, 10_000), slice(10_000, None)
    naive = fuse.Fusion("naive").fit(p0[tr], X[tr], y[tr])
    fitted = fuse.Fusion("shared").fit(p0[tr], X[tr], y[tr])
    per = fuse.Fusion("per_family", families=["f0", "f1", "f2"]).fit(p0[tr], X[tr], y[tr])

    def b(m):
        return np.mean((m.predict(p0[te], X[te]) - y[te]) ** 2)

    assert b(fitted) < b(naive)
    assert b(per) <= b(fitted) + 1e-4
    # the shared weight shrinks below 1 to undo the double count; per family, the two copies split one
    # family's weight between them and the independent family keeps weight ~1
    assert fitted.w[0] < 0.95
    assert per.w[0] + per.w[1] == pytest.approx(1.0, abs=0.15)
    assert per.w[2] == pytest.approx(1.0, abs=0.15)


def test_ridge_pulls_toward_naive():
    rng = np.random.default_rng(1)
    p0, X, y = _alerts(rng, n=300, double_count=True)
    loose = fuse.Fusion("per_family", ridge=1e-3).fit(p0, X, y)
    tight = fuse.Fusion("per_family", ridge=1e6).fit(p0, X, y)
    assert np.abs(tight.w - 1).max() < 0.01 and abs(tight.a - 1) < 0.01
    assert np.abs(loose.w - 1).max() > np.abs(tight.w - 1).max()


def test_shuffled_answers_gain_vanishes():
    rng = np.random.default_rng(2)
    p0, X, y = _alerts(rng, n=8000)
    r = fuse.shuffled_answers_test(p0, X, y, mode="shared")
    assert r["gain_real"] > 0.01
    assert abs(r["gain_shuffled"]) < 0.1 * r["gain_real"]


def test_posterior_with_fitted_model_rejects_unknown_family():
    rng = np.random.default_rng(3)
    p0, X, y = _alerts(rng, n=2000, k=2)
    m = fuse.Fusion("per_family", families=["f", "g"]).fit(p0, X, y)
    assert 0 < fuse.posterior(0.3, {"f": 0.7}, m) < 1
    with pytest.raises(KeyError):
        fuse.posterior(0.3, {"h": 0.7}, m)
