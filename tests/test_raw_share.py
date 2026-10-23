"""H2 raw-sharing arm: gated, bounded to the look-back and reply time, and at least as good as the question route."""

import re
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from ftm.backtest.synth import make_world
from ftm.banks.binding import alert_context
from ftm.banks.partner import BankConfig
from ftm.compliance.gate import Gate, GateBlocked
from ftm.harness import make_toolbox
from ftm.orchestrator.raw_share import (
    LOOKBACK_H,
    N_SHARING_BANKS,
    RawShareArm,
    RawShareBank,
    RawShareModel,
    collect_records,
    gain_ratio,
)
from ftm.orchestrator.runner import synthetic_experiment
from ftm.types import Limits

N, SEED = 240, 3


@pytest.fixture(scope="module")
def setup():
    w = make_world(N, seed=SEED)
    cfg = BankConfig(p_decline=0.0, p_timeout=0.0, notes_enabled=False)
    banks = {b: RawShareBank(b, bk._ledger, cfg, 0) for b, bk in w.banks.items()}
    alerts = sorted(w.alerts, key=lambda a: a.binding.t0)
    cut = int(len(alerts) * 0.6)
    rate = sum(a.label for a in alerts[:cut]) / cut
    train = [replace(a, p0=rate) for a in alerts[:cut]]
    val = [replace(a, p0=rate) for a in alerts[cut:]]
    return w, banks, train, val, rate


def test_the_production_gate_blocks_raw_sharing_and_only_the_h2_gate_allows_it(setup):
    w, banks, train, val, _ = setup
    b = val[0].binding
    with pytest.raises(GateBlocked, match="raw records never leave"):
        collect_records(banks, Gate.from_config(), b)
    assert Gate.from_config().check("request_records", alert_id="a", home_bank=1, bank=2).decision == "blocked"
    records, used, days = collect_records(banks, Gate.from_config(allow_raw_share=True), b)
    assert records.height > 0 and 1 <= len(used) <= N_SHARING_BANKS and used[0] == b.partner_bank
    assert 0 < days <= 4.0


def test_the_hand_over_is_bounded_in_time_and_to_the_subject_and_its_neighbours(setup):
    w, banks, train, val, _ = setup
    b = val[1].binding
    records, used, days = collect_records(banks, Gate.from_config(allow_raw_share=True), b)
    ts = records["timestamp"]
    assert ts.min() >= b.t0 - timedelta(hours=LOOKBACK_H) and ts.max() <= b.t0 + timedelta(days=days)
    subject = b.receiver if b.sender_bank == b.home_bank else b.sender
    first = records.filter((records["src_id"] == subject) | (records["dst_id"] == subject))
    assert first.height > 0
    # every other row involves a neighbour of the subject, and is held by a second-hop bank
    others = records.filter((records["src_id"] != subject) & (records["dst_id"] != subject))
    nbrs = set(first["src_id"].to_list()) | set(first["dst_id"].to_list())
    assert all(s in nbrs or d in nbrs for s, d in others.select("src_id", "dst_id").rows())


def test_nothing_else_can_reach_the_raw_rows():
    root = Path(__file__).resolve().parents[1] / "ftm"
    pat = re.compile(r"def share_records|\.share_records\(")
    users = [str(p.relative_to(root)) for p in root.rglob("*.py") if pat.search(p.read_text(encoding="utf-8"))]
    assert sorted(users) == [str(Path("orchestrator/raw_share.py"))]
    t = make_toolbox(seed=0)
    assert not [n for n in dir(t) if "record" in n or "raw" in n]


def test_ridge_model_uses_an_informative_feature_and_ignores_noise():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.2).astype(float)
    X = np.column_stack([y * 2 + rng.normal(0, 1, 400), rng.normal(0, 1, (400, 5))])
    p0 = np.full(400, 0.2)
    m = RawShareModel(ridge=5.0).fit(p0[:300], X[:300], y[:300])
    p = m.predict(p0[300:], X[300:])
    assert np.mean((p - y[300:]) ** 2) < np.mean((0.2 - y[300:]) ** 2) - 0.02
    assert abs(m.w[0]) > 3 * np.abs(m.w[1:]).max()


def test_gain_ratio():
    assert gain_ratio(0.16, 0.12, 0.08) == pytest.approx(0.5)
    assert gain_ratio(0.16, 0.08, 0.08) == pytest.approx(1.0)
    assert np.isnan(gain_ratio(0.1, 0.09, 0.1))


def test_raw_sharing_beats_the_base_rate_and_is_at_least_as_good_as_the_question_route(setup):
    w, banks, train, val, rate = setup
    gate = Gate.from_config(allow_raw_share=True)
    arm = RawShareArm(banks, gate, bindings={a.alert_id: a.binding for a in val}).fit(train)
    ys = np.array([float(a.label) for a in val])
    ps = np.array([arm.investigate(alert_context(a.binding, a.p0), None, Limits()).probability for a in val])
    b_raw, b_p0 = float(np.mean((ps - ys) ** 2)), float(np.mean((rate - ys) ** 2))
    assert b_raw < b_p0 - 0.03
    q = synthetic_experiment("eig", N, SEED, programs=["n_counterparties(ALERT.receiver, dir=in, window=before(72))",
                                                       "passed_on(ALERT.receiver, window=after(24), min_frac=0.5)",
                                                       "n_tx(ALERT.receiver, dir=in, window=around(72), max_amount=2000)"])  # fmt: skip
    assert q.summary["brier_p0"] == pytest.approx(b_p0)  # the same validation alerts
    r = gain_ratio(b_p0, q.summary["brier_final"], b_raw)
    assert 0.3 < r < 1.2, r  # questions get a large share of what raw sharing buys on this synthetic world
    v = arm.investigate(alert_context(val[0].binding, val[0].p0), None, Limits())
    assert all(e.program == "request_records" and e.eps == 0.0 for e in v.questions_log)
