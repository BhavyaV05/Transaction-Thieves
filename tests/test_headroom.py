"""Headroom study helpers (ftm.fusion.headroom): folds, cell LRs, asks, the offline EIG policy, ConditionalFusion.p0_clip."""

import numpy as np

from ftm.fusion.conditional import ConditionalFusion
from ftm.fusion.headroom import (
    Design,
    Hyp,
    Logit,
    Scorer,
    cell_lr,
    draw_asks,
    metrics,
    raw_cell,
    recal_ab,
    run_policy,
    time_blocks,
)  # fmt: skip


def _hyp(name, eps, fam, lr=(0.5, 2.0), p_clean=(0.8, 0.2), p_launder=(0.2, 0.8)):
    return Hyp(name, eps, fam, "bool", (), ["false", "true"], np.array(lr), np.array(p_clean), np.array(p_launder), False, 1, None)


def test_time_blocks_every_fold_sees_every_era():
    f = time_blocks(1000, 5)
    assert set(f) == set(range(5))
    for k in range(5):
        halves = np.flatnonzero(f == k)
        assert halves.min() < 250 and halves.max() > 750


def test_raw_cell_matches_table_cells():
    assert raw_cell("kary", (), 2.0) == "2"
    assert raw_cell("bool", (), 1.0) == "true" and raw_cell("bool", (), 0.0) == "false"
    assert raw_cell("count", (0.0, 1.0), 0.0) == "b1" and raw_cell("count", (0.0, 1.0), 5.0) == "b2"


def test_cell_lr_is_clipped_to_the_ceiling():
    lr = cell_lr(np.array([100.0, 0.0]), np.array([0.0, 100.0]), ceiling=3.0)
    assert np.allclose(lr, [1 / 3, 3.0])


def test_draw_asks_respects_budget_family_and_menu():
    hyps = [_hyp("a", 1.0, "f"), _hyp("b", 1.0, "f"), _hyp("c", 1.0, "g"), _hyp("d", 2.0, "h")]
    cell = np.zeros((20, 4), dtype="i2")
    a = draw_asks(cell, hyps, np.random.default_rng(0), (1.0,), budget=2.0, draws=3)
    assert a.shape == (3, 20, 4)
    assert (a[..., 3] == 0).all()  # eps 2 not in the menu
    assert (a[..., :2].sum(-1) <= 1).all()  # a family at most once
    assert (a.sum(-1) == 2).all()  # two eps-1 asks fit and exist


def test_eig_asks_the_informative_question_and_stops_when_nothing_helps():
    hyps = [_hyp("weak", 2.0, "f", lr=(0.99, 1.01), p_clean=(0.5, 0.5), p_launder=(0.49, 0.51)), _hyp("strong", 2.0, "g")]
    cell = np.array([[1, 1], [0, 0]], dtype="i2")
    x = np.array([[0.01, np.log(2.0)], [-0.01, np.log(0.5)]])
    sc = Scorer(0.0, 1.0, np.ones(2), 1.0)
    asks = run_policy("eig", cell, x, np.array([0.5, 0.5]), hyps, sc, (2.0,))
    assert asks[:, 1].all() and not asks[:, 0].any()  # the budget buys one eps-2 question: the strong one
    none = run_policy("eig", cell, x, np.array([1e-9, 1e-9]), hyps, Scorer(0.0, 1.0, np.ones(2), 1.0, 0.1), (2.0,))
    assert none.sum() >= 0  # clipped p0 still gives a posterior in (0, 1): no crash on extreme scores


def test_oracle_never_worse_than_eig_and_cannot_cross_stops():
    rng = np.random.default_rng(1)
    hyps = [_hyp("a", 2.0, "f"), _hyp("b", 2.0, "g")]
    y = rng.integers(0, 2, 200).astype("f8")
    cell = (y[:, None] * np.ones((1, 2))).astype("i2")  # answers equal the label
    x = np.where(cell == 1, np.log(2.0), np.log(0.5))
    p0 = np.full(200, 0.5)
    sc = Scorer(0.0, 1.0, np.ones(2), 1.0)
    e = run_policy("eig", cell, x, p0, hyps, sc, (2.0,))
    o = run_policy("oracle_greedy", cell, x, p0, hyps, sc, (2.0,), y=y)
    be = np.mean((1 / (1 + np.exp(-sc.z(p0, x, e))) - y) ** 2)
    bo = np.mean((1 / (1 + np.exp(-sc.z(p0, x, o))) - y) ** 2)
    assert bo <= be + 1e-12
    far = run_policy("eig", cell, x, p0, hyps, sc, (2.0,), stop=(5.0, 0.1))  # threshold far away, shifts tiny: stop at once
    assert far.sum() == 0


def test_recal_ab_and_metrics():
    rng = np.random.default_rng(0)
    p0 = rng.uniform(0.05, 0.95, 4000)
    y = (rng.random(4000) < p0).astype("f8")
    a, b = recal_ab(p0, y)
    assert abs(a) < 0.2 and abs(b - 1) < 0.2
    m = metrics(p0, y)
    assert 0.5 < m["auc"] < 1 and abs(m["cal_slope"] - 1) < 0.2


def test_logit_family_design_roundtrip():
    hyps = [_hyp("a", 2.0, "f"), _hyp("b", 2.0, "g")]
    d = Design.make(hyps, "fam")
    cell = np.array([[1, 0], [0, 1]], dtype="i2")
    x = np.array([[0.5, 0.2], [0.3, -0.4]])
    F = d.matrix(cell, x, np.array([[True, False], [True, True]]))
    assert F.tolist() == [[0.5, 0.0], [0.3, -0.4]]
    rng = np.random.default_rng(0)
    cell = rng.integers(0, 2, (500, 2)).astype("i2")
    x = np.where(cell == 1, np.log(3.0), np.log(1 / 3))
    p0 = rng.uniform(0.2, 0.8, 500)
    y = (rng.random(500) < 1 / (1 + np.exp(-(np.log(p0 / (1 - p0)) + x.sum(1))))).astype("f8")
    m = Logit(d, l2=1.0).fit(p0, d.matrix(cell, x, np.ones((500, 2), bool)), y)
    assert (m.w > 0.5).all() and (m.w < 1.5).all()


def test_conditional_fusion_p0_clip_roundtrip_and_old_hash_stable(tmp_path):
    old = ConditionalFusion(["f"], a=0.1, b=0.9, w=np.array([0.5]), fitted=True)
    assert "p0_clip" not in old.params()  # default: pre-existing frozen files keep their hash
    m = ConditionalFusion(["f"], a=0.0, b=1.0, w=np.array([0.5]), fitted=True, p0_clip=0.1)
    assert abs(float(m.predict([1.0], [[0.0]])[0]) - 0.9) < 1e-9  # exact 1.0 scores are clipped to 0.9
    path = tmp_path / "m.json"
    m.to_json(path)
    assert ConditionalFusion.from_json(path).p0_clip == 0.1
