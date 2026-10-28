"""The four additions to the final analysis (issues 85-88): power, eps = infinity, review capacity (S1), p0-adjusted differences (S2).

Owner: T4 (Shreyas).  Synthetic world only (ftm/eval/final_synth.py): no data, no LLM, seconds on a laptop.
"""

from __future__ import annotations

import importlib.util
import json
import math
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from ftm.eval import final as fn
from ftm.eval import final_additions as fadd
from ftm.eval import final_tables as ft
from ftm.eval.final_synth import build_run

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("final_analysis", ROOT / "scripts" / "final_analysis.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("add")
    env = build_run(root)
    out = root / "out"
    assert script.main(["--split", "val", "--run", str(env["run_dir"]), "--inputs", str(env["inputs"]), "--outdir", str(out), "--n-boot", "500", "--no-figures"]) == 0
    env["out"] = out
    env["status"] = json.loads((out / "status.json").read_text())
    env["numbers"] = json.loads((out / "numbers.json").read_text())
    return env


# ---- 1. power (issue 86) ---------------------------------------------------------------------------------------------
def test_power_details_arithmetic():
    d = fn.power_details(900, 0.0118)  # mde = 2.8 * SE
    assert d["se_paired_difference"] == pytest.approx(0.0118 / 2.8)
    assert d["alerts_needed_for_target"] == math.ceil(900 * (0.0118 / 0.01) ** 2) == 1254
    assert d["mde_at_projected"] == pytest.approx(0.0118 * math.sqrt(900 / 3475))
    assert d["projected_test_alerts"] == 3475 and d["target_brier_difference"] == 0.01
    d2 = fn.power_details(900, 0.0118, target=0.02, n_projected=1000)
    assert d2["alerts_needed_for_target"] == math.ceil(900 * (0.0118 / 0.02) ** 2) and d2["projected_test_alerts"] == 1000
    none = fn.power_details(900, None)
    assert none["alerts_needed_for_target"] is None and fn.power_sentence(none) == ""


def test_power_txt_and_status_json(run):
    txt = (run["out"] / "power.txt").read_text()
    assert "0.0232" in txt and "80% power" in txt  # the old statement is intact
    assert "needs about" in txt and "projected test size of 3475 alerts" in txt
    p = run["status"]["power"]
    assert p["status"] == "ok" and p["file"] == "power.txt"
    assert p["alerts_needed_for_target"] == math.ceil(p["n_alerts"] * (p["mde_at_n"] / 0.01) ** 2)
    assert p["mde_at_projected"] == pytest.approx(p["mde_at_n"] * math.sqrt(p["n_alerts"] / 3475))
    assert run["numbers"]["power_statement"]["status"] == "ok" and run["numbers"]["power_statement"]["alerts_needed_for_target"] == p["alerts_needed_for_target"]


# ---- 2. eps = infinity (issue 85) ------------------------------------------------------------------------------------
def _rows(**gains):
    return [{"budget": float(k[1:]) if k != "inf" else math.inf, "gain": v} for k, v in gains.items()]


def test_eps_reading_is_neutral_and_follows_the_rule():
    assert fadd.eps_reading([{"budget": 2.0, "gain": -0.0058}, {"budget": math.inf, "gain": -0.0287}]).startswith("Noise is not the limiter")
    assert fadd.eps_reading([{"budget": 2.0, "gain": 0.002}, {"budget": math.inf, "gain": 0.020}]).startswith("Noise is the limiter")
    assert fadd.eps_reading([{"budget": 2.0, "gain": 0.0035}, {"budget": math.inf, "gain": 0.0050}]).startswith("Noise is not the limiter")
    assert "No no-noise" in fadd.eps_reading([{"budget": 2.0, "gain": 0.0}])
    assert "No eps = 2" in fadd.eps_reading([{"budget": math.inf, "gain": 0.0}])


def test_eps_infinity_table_from_the_sweep_json(run):
    import csv

    rows = list(csv.DictReader((run["out"] / "eps_infinity.csv").open()))
    assert [r["eps"] for r in rows] == ["0.5", "1", "2", "4", "infinity (no noise)"]
    assert float(rows[-1]["brier gain"]) == pytest.approx(-0.029) and float(rows[2]["gain minus gain at eps 2"]) == 0.0
    md = (run["out"] / "eps_infinity.md").read_text()
    assert "Noise is not the limiter" in md and "validation sweep" in md
    assert run["status"]["eps_infinity"]["status"] == "ok" and run["status"]["eps_infinity"]["reading"].startswith("Noise is not the limiter")
    assert "Noise is not the limiter" in run["numbers"]["eps_infinity"]["text"]


def test_eps_infinity_not_run_without_a_path_and_reads_string_inf(tmp_path):
    w = ft.Writer(tmp_path)
    with pytest.raises(fn.Missing, match="eps_sweep.path"):
        fadd.eps_infinity({}, "val", w)
    p = tmp_path / "sweep.json"
    p.write_text(json.dumps([{"system": "eig", "budget": 2.0, "alerts": 5, "brier_p0": 0.2, "brier_final": 0.19, "mean_questions": 1.0},
                             {"system": "eig", "budget": "inf", "alerts": 5, "brier_p0": 0.2, "brier_final": 0.15, "mean_questions": 6.0}]))
    rows = fadd.eps_infinity({"eps_sweep": {"path": str(p)}}, "val", w)  # gain is computed from the Brier columns when brier_gain is absent
    assert rows[-1]["eps"] == "infinity (no noise)" and rows[-1]["brier gain"] == pytest.approx(0.05)
    assert w.status["eps_infinity"]["reading"].startswith("Noise is the limiter")
    p.write_text("not json")
    with pytest.raises(fn.AnalysisInputError):
        fadd.eps_infinity({"eps_sweep": {"path": str(p)}}, "val", w)


def test_eps_infinity_not_run_in_status_when_unset(tmp_path):
    env = build_run(tmp_path)
    inp = yaml.safe_load(Path(env["inputs"]).read_text())
    inp["eps_sweep"] = {"path": None}
    Path(env["inputs"]).write_text(yaml.safe_dump(inp))
    out = tmp_path / "o"
    assert script.main(["--split", "val", "--run", str(env["run_dir"]), "--inputs", str(env["inputs"]), "--outdir", str(out), "--n-boot", "50", "--no-figures"]) == 0
    st = json.loads((out / "status.json").read_text())["eps_infinity"]
    assert st["status"] == "not_run" and "eps_sweep.path" in st["reason"]
    assert json.loads((out / "numbers.json").read_text())["eps_infinity"]["status"] == "not_run"


# ---- 3. review capacity (S1, issue 87) -------------------------------------------------------------------------------
def test_topk_point_matches_brute_force_and_resamples_are_deterministic():
    rng = np.random.default_rng(1)
    n = 200
    y = (rng.random(n) < 0.3).astype(float)
    s = rng.random(n) + 0.5 * y
    cl = np.arange(n) // 2
    r1 = fadd._topk_resamples({"a": s}, y, cl, 0.1, 300, 0)
    r2 = fadd._topk_resamples({"a": s}, y, cl, 0.1, 300, 0)
    top = np.argsort(-s, kind="stable")[:20]
    assert r1["a"]["k"] == 20 and r1["a"]["precision"] == pytest.approx(y[top].mean()) and r1["a"]["recall"] == pytest.approx(y[top].sum() / y.sum())
    np.testing.assert_array_equal(r1["a"]["prec_s"], r2["a"]["prec_s"])
    assert abs(np.nanmean(r1["a"]["prec_s"]) - r1["a"]["precision"]) < 0.08  # resamples centre near the point


def test_topk_resample_equals_explicit_cluster_resample():
    """The cumulative-weight shortcut gives the same numbers as literally stacking the drawn clusters and taking the top ceil(frac * rows)."""
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0], float)
    s = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.6, 0.3, 0.4])
    cl = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    wc = next(iter(fadd.bs._weights(4, 1, 5, 1)))[0]
    rows = np.concatenate([np.flatnonzero(cl == c).tolist() * int(k) for c, k in enumerate(wc)]).astype(int)
    ys, ss = y[rows], s[rows]
    kk = math.ceil(0.25 * len(rows))
    top = np.argsort(-ss, kind="stable")[:kk]
    r = fadd._topk_resamples({"a": s}, y, cl, 0.25, 1, 5)["a"]
    # ties inside a resample (duplicated rows) cannot change the counts: equal scores carry the same label
    assert r["prec_s"][0] == pytest.approx(ys[top].mean()) and r["rec_s"][0] == pytest.approx(ys[top].sum() / ys.sum())


def test_capacity_is_read_from_fixed_and_refused_on_test_without_it():
    fake = types.SimpleNamespace(fixed={}, fixed_derived={}, split="val")
    cap, how = fadd.resolve_capacity(fake)
    assert cap["fraction"] == 0.10 and "placeholder" in cap["status"] and fake.fixed_derived["review_capacity"] == cap and "default" in how
    fake = types.SimpleNamespace(fixed={"review_capacity": {"kind": "top_fraction", "fraction": 0.2}}, fixed_derived={}, split="test")
    assert fadd.resolve_capacity(fake)[0]["fraction"] == 0.2 and not fake.fixed_derived
    with pytest.raises(fn.Missing, match="review_capacity is not fixed"):
        fadd.resolve_capacity(types.SimpleNamespace(fixed={}, fixed_derived={}, split="test"))
    with pytest.raises(fn.AnalysisInputError):
        fadd.resolve_capacity(types.SimpleNamespace(fixed={"review_capacity": {"fraction": 1.5}}, fixed_derived={}, split="val"))


def test_review_capacity_table_and_fixed_on_val(run):
    import csv

    rows = list(csv.DictReader((run["out"] / "review_capacity.csv").open()))
    ob = rows[0]
    assert ob["system"] == "one bank (p0)" and int(ob["reviewed (K)"]) == 90  # 10% of 900
    ours = next(r for r in rows if r["system"].startswith("ours_compose/harness/ported/full_test"))
    assert float(ours["precision"]) > float(ob["precision"]) and float(ours["recall minus one bank"]) > 0  # planted skill 1.6
    assert float(ours["recall"]) < 0.4  # 90 reviewed of 251 positives: recall cannot pass K / positives
    assert run["status"]["review_capacity"]["status"] == "ok" and run["status"]["review_capacity"]["capacity"]["fraction"] == 0.10
    assert "PLACEHOLDER" not in (run["out"] / "review_capacity.md").read_text() and "placeholder" in (run["out"] / "review_capacity.md").read_text()
    fixed = json.loads((run["out"] / "fixed_on_val.json").read_text())
    assert fixed["review_capacity"]["fraction"] == 0.10 and "placeholder" in fixed["review_capacity"]["status"]
    assert run["numbers"]["review_capacity"]["status"] == "ok"
    # the secondary analysis is not a hypothesis: it does not enter hypotheses.json or any Holm family
    assert "S1" not in json.loads((run["out"] / "hypotheses.json").read_text())


def test_test_run_reads_the_capacity_from_the_fixed_file(run, tmp_path, monkeypatch):
    fixed = json.loads((run["out"] / "fixed_on_val.json").read_text())
    fixed["review_capacity"] = {"kind": "top_fraction", "fraction": 0.2, "status": "pre-registered"}
    fp = tmp_path / "fixed.json"
    fp.write_text(json.dumps(fixed))
    out = tmp_path / "o2"
    assert script.main(["--split", "val", "--run", str(run["run_dir"]), "--inputs", str(run["inputs"]), "--outdir", str(out), "--n-boot", "50", "--no-figures",
                        "--fixed", str(fp)]) == 0
    import csv

    assert int(list(csv.DictReader((out / "review_capacity.csv").open()))[0]["reviewed (K)"]) == 180
    assert json.loads((out / "fixed_on_val.json").read_text())["review_capacity"]["fraction"] == 0.2  # carried forward, not overwritten by the default


# ---- 4. p0-adjusted estimator (S2, issue 88) ------------------------------------------------------------------------
def _planted(n=600, seed=3, slope_p0=-0.4, slope_q=0.0, noise=0.03):
    rng = np.random.default_rng(seed)
    p0 = rng.random(n)
    q = rng.integers(0, 4, n).astype(float)
    cl = np.arange(n) // 3
    d = -0.05 + slope_p0 * (p0 - 0.5) + slope_q * q + rng.normal(0, noise, n)
    return d, p0, q, cl


def test_adjustment_keeps_the_point_estimate_and_narrows_the_ci():
    d, p0, q, cl = _planted()
    r = fadd.adjusted_difference(d, {"p0": p0, "answered questions": q}, cl, 1000, 0)
    assert r["adjusted"] == pytest.approx(r["unadjusted"]) == pytest.approx(d.mean())  # same estimand, same point estimate
    assert r["width_adjusted"] < 0.85 * r["width_unadjusted"]  # p0 explains most of the variance
    assert r["slopes"]["p0"]["estimate"] == pytest.approx(-0.4, abs=0.05) and r["slopes"]["p0"]["ci"][0] < -0.4 < r["slopes"]["p0"]["ci"][1] + 0.05
    assert 0.5 < r["r2"] < 1.0
    assert r["covariates"] == ["p0", "answered questions"] and not r["dropped_constant"]


def test_adjustment_is_deterministic_and_matches_an_explicit_refit():
    d, p0, q, cl = _planted(n=120)
    a = fadd.adjusted_difference(d, {"p0": p0, "answered questions": q}, cl, 200, 7)
    b = fadd.adjusted_difference(d, {"p0": p0, "answered questions": q}, cl, 200, 7)
    np.testing.assert_array_equal(a["samples_adjusted"], b["samples_adjusted"])
    # first resample refitted by hand: stack the drawn clusters, OLS on centred covariates, intercept
    _, cli = np.unique(cl, return_inverse=True)
    w0 = next(iter(fadd.bs._weights(int(cli.max()) + 1, 200, 7, 500)))[0]
    rows = np.repeat(np.arange(len(d)), w0[cli].astype(int))
    X = np.column_stack([np.ones(len(rows)), (p0 - p0.mean())[rows], (q - q.mean())[rows]])
    beta = np.linalg.lstsq(X, d[rows], rcond=None)[0]
    assert a["samples_adjusted"][0] == pytest.approx(beta[0], abs=1e-9)
    assert a["samples_unadjusted"][0] == pytest.approx(d[rows].mean(), abs=1e-9)


def test_constant_covariate_is_dropped():
    d, p0, _, cl = _planted(n=90)
    r = fadd.adjusted_difference(d, {"p0": p0, "answered questions": np.ones(90)}, cl, 100, 0)
    assert r["dropped_constant"] == ["answered questions"] and list(r["slopes"]) == ["p0"] and math.isfinite(r["width_adjusted"])
    r0 = fadd.adjusted_difference(d, {"answered questions": np.ones(90)}, cl, 100, 0)  # no usable covariate: identical to the unadjusted estimator
    assert r0["width_adjusted"] == pytest.approx(r0["width_unadjusted"]) and r0["adjusted"] == pytest.approx(r0["unadjusted"])


def test_adjusted_estimates_table_on_the_synthetic_world(run):
    import csv

    rows = list(csv.DictReader((run["out"] / "adjusted_estimates.csv").open()))
    assert [r["hypothesis"] for r in rows] == ["H1", "H3", "H3", "H3"]
    h = json.loads((run["out"] / "hypotheses.json").read_text())
    h1 = rows[0]
    assert float(h1["unadjusted"]) == pytest.approx(h["H1"]["brier_diff_vs_one_bank"]["point"], abs=1e-4)  # the same primary difference
    assert float(h1["adjusted"]) == pytest.approx(float(h1["unadjusted"]), abs=1e-4)
    assert float(h1["adjusted CI width"]) <= float(h1["unadjusted CI width"]) * 1.02 and float(h1["slope on p0"]) < 0
    assert all("sensitivity, unadjusted is primary" in r["note"] for r in rows)
    assert "sensitivity, unadjusted is primary" in (run["out"] / "adjusted_estimates.md").read_text()
    assert run["status"]["adjusted_estimates"]["status"] == "ok" and run["numbers"]["adjusted_estimates"]["status"] == "ok"
    assert "S2" not in h  # not a hypothesis, not in a Holm family
    # the primary verdicts did not move
    assert h["H1"]["verdict"] == "supported"


def test_answered_questions_come_from_the_step_rows(run):
    panel_log = None
    import polars as pl

    env_files = run["files"]["ext_react"]
    panel_log = pl.read_parquet(env_files)
    assert (panel_log["event"] == "step").any()  # the external agent has step rows; the helper counts replies per alert
    key = fn.Key("react", "harness", "ported", "subset")
    ids = [f"V{i:05d}" for i in range(5)]
    fake_run = types.SimpleNamespace(log=panel_log.with_columns(pl.lit("subset").alias("_alert_set")))
    fake = types.SimpleNamespace(run=fake_run, alerts=pl.DataFrame({"alert_id": ids}))
    na = fadd.answered_per_alert(fake, key)
    assert na.shape == (5,) and (na >= 0).all() and na.max() <= 1.0  # one reply per alert per seed, averaged over seeds
    assert fadd.answered_per_alert(fake, None) is None


# ---- prereg and docs wiring ------------------------------------------------------------------------------------------
def test_prereg_lists_s1_and_s2_outside_the_holm_families():
    h = yaml.safe_load((ROOT / "prereg" / "hypotheses.yaml").read_text())
    assert {"S1", "S2"} <= set(h["sensitivity"]) and all(v["tier"] == "sensitivity" for v in h["sensitivity"].values())
    assert "S1" not in h["hypotheses"] and "S2" not in h["hypotheses"]
    fam = {x for t in h["statistics"]["correction"].values() for x in t["family"]}
    assert not fam & {"S1", "S2"}
    assert h["sensitivity"]["S1"]["capacity"]["fraction"] == fadd.DEFAULT_CAPACITY["fraction"]


def test_new_outline_ids_are_in_the_numbers_table():
    ids = ft.outline_ids(ROOT / "docs" / "report" / "00_outline.md")
    assert {"power_statement", "eps_infinity", "review_capacity", "adjusted_estimates"} <= set(ids)
