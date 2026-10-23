"""T3: adapter from our Fusion protocol to T4's ftm.fusion (PR #7 review)."""

import numpy as np
import pytest

from ftm.fusion.fuse import Fusion, answer_features
from ftm.llm.fusion_adapter import FusedLR
from ftm.llm.investigation import LREntry, LRTable, NaiveLRFusion
from ftm.types import Decline, QuestionLogEntry, Reply

FAMS = ["f", "g", "h"]
PROGS = [f"prog_{f}" for f in FAMS]
TABLE = LRTable([LREntry(p, 0.5, p != "prog_h", {"true": 3.0, "false": 0.4}, family=f) for p, f in zip(PROGS, FAMS, strict=True)])


def entry(i, prog, value=True, gate="allowed", reply=True):
    e = QuestionLogEntry(i, "220", prog, 0.5, gate_decision=gate)
    e.response = Reply("220", "bool", value, 0.5, 1.0) if reply else Decline("220", 1.0)
    return e


def random_logs(rng, n=200):
    logs = []
    for _ in range(n):
        log = [entry(i, PROGS[rng.integers(3)], bool(rng.integers(2)), gate=str(rng.choice(["allowed", "blocked"])), reply=rng.random() < 0.8)
               for i in range(rng.integers(0, 6))]  # fmt: skip
        logs.append((float(rng.uniform(0.001, 0.999)), log))
    return logs


def test_fusion_naive_equals_naive_lr_fusion():
    # same selection (accepted LR, allowed reply), same clip, same logit/sigmoid: posteriors agree to rounding
    rng = np.random.default_rng(0)
    old, new = NaiveLRFusion(TABLE), FusedLR(Fusion("naive"), TABLE)
    for p0, log in random_logs(rng):
        assert new.posterior(p0, log) == pytest.approx(old.posterior(p0, log), abs=1e-12)
        assert [new.uses(e) for e in log] == [old.uses(e) for e in log]


def _fit_on_synthetic(mode, rng, n=4000):
    logs = random_logs(rng, n)
    p0 = np.array([p for p, _ in logs])
    ids = [str(i) for i in range(n)]
    uses = NaiveLRFusion(TABLE).uses
    rows = [(ids[i], TABLE.get(e.program, e.eps).family, np.log(TABLE.get(e.program, e.eps).lr_for(e.response.value)))
            for i, (_, log) in enumerate(logs) for e in log if uses(e)]  # fmt: skip
    X, fams = answer_features(ids, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], families=FAMS[:2])
    truth = 1.0 / (1 + np.exp(-(0.7 * np.log(p0 / (1 - p0)) + 0.5 * X.sum(axis=1) - 0.3)))  # known miscalibrated mixture
    y = (rng.random(n) < truth).astype(float)
    m = Fusion(mode).fit(p0, X, y)
    m.families = fams  # fit() does not set them
    return m


@pytest.mark.parametrize("mode", ["shared", "per_family"])
def test_fitted_model_is_applied_through_the_adapter(mode):
    m = _fit_on_synthetic(mode, np.random.default_rng(1))
    assert m.a < 0.95 and m.c < -0.1  # the fit found the synthetic shrinkage, so it differs from naive
    fused, naive = FusedLR(m, TABLE), NaiveLRFusion(TABLE)
    log = [entry(1, "prog_f"), entry(2, "prog_g", False), entry(3, "prog_h")]  # h: not accepted, no evidence
    X = np.array([[np.log(3.0), np.log(0.4)]])
    assert fused.posterior(0.3, log) == pytest.approx(m.predict([0.3], X)[0])
    assert fused.posterior(0.3, log) != pytest.approx(naive.posterior(0.3, log), abs=1e-3)
    assert fused.posterior(0.3, []) == pytest.approx(m.predict([0.3], np.zeros((1, 2)))[0])  # no answers: a, c still apply
    assert FusedLR.from_params(m.params(), TABLE).posterior(0.3, log) == pytest.approx(fused.posterior(0.3, log))


def test_fitted_model_refuses_unknown_family_and_unnamed_per_family():
    m = Fusion("per_family", ["f"], 1.0, np.array([1.0]), 0.0, 1.0, True)
    with pytest.raises(KeyError):
        FusedLR(m, TABLE).posterior(0.3, [entry(1, "prog_g")])  # family g is not in the model
    with pytest.raises(ValueError):
        FusedLR(Fusion("per_family", [], 1.0, np.array([1.0]), 0.0, 1.0, True), TABLE)
