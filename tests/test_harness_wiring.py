"""T3: harness toolbox wiring in run_llm_agent.py, the back-test table adapter, run-log v1 rows. Synthetic, no data files."""

import importlib.util
import json
import math
from pathlib import Path

import polars as pl
import pytest

from ftm.eval import runlog, runs
from ftm.llm.backtest_table import BacktestTableError, TableAcceptance, load_backtest
from ftm.llm.fusion_adapter import FusedLR
from ftm.llm.investigation import LRTable

PROG = "n_counterparties(ALERT.receiver, dir=in, window=before(72))"


def _script():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("run_llm_agent", root / "scripts" / "run_llm_agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _table_dir(tmp_path, rows, mode="realistic", entries=True):
    """A directory like `scripts/backtest.py --assemble` leaves: lr_table.jsonl, report.json, cache entries."""
    d = tmp_path / "bt" / "abc123"
    d.mkdir(parents=True)
    (d / "lr_table.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "report.json").write_text(json.dumps({"namespace": "abc123"}))
    if entries:
        for i, r in enumerate(rows):
            (d / f"{i:02d}.json").write_text(json.dumps({**r, "mode": mode, "eligible": True, "e_value": 1.0}))
    return d


def _row(program=PROG, eps=1.0, accepted=True, lr=None, bins=(1.0, 2.0)):
    lr = lr or {"b0": 0.5, "b1": 1.0, "b2": 3.0}
    return {"program_canonical": program, "eps": eps, "accepted": accepted, "lr": lr, "family": "n_counterparties:in",
            "bins": list(bins)}  # fmt: skip


# -- adapter ------------------------------------------------------------------------------------------------------
def test_loads_dir_or_file_and_reads_mode_from_cache(tmp_path):
    d = _table_dir(tmp_path, [_row(eps=0.5), _row(eps=1.0), _row(eps=2.0, accepted=False)])
    for p in (d, d / "lr_table.jsonl"):
        bt = load_backtest(p)
        assert bt.mode == "realistic" and bt.namespace == "abc123" and bt.rows == 3
        assert bt.accepted_by_eps == {0.5: 1, 1.0: 1, 2.0: 0}
        assert bt.table.get(PROG, 1.0).accepted and not bt.table.get(PROG, 2.0).accepted
    assert bt.missing_eps() == [0.25]  # a planner level T2 did not back-test: never accepted
    assert bt.summary()["eps_levels_without_rows"] == [0.25]


def test_refuses_exact_mode_and_unknown_mode_and_mismatch(tmp_path):
    d = _table_dir(tmp_path, [_row()], mode="exact")
    with pytest.raises(BacktestTableError, match="exact"):
        load_backtest(d)
    assert load_backtest(d, require=None).mode == "exact"  # reports may read it
    with pytest.raises(BacktestTableError, match="not 'realistic'"):
        load_backtest(d, mode="realistic")  # the stated mode contradicts the cache
    bare = _table_dir(tmp_path / "x", [_row()], entries=False)
    with pytest.raises(BacktestTableError, match="cannot establish"):
        load_backtest(bare)
    assert load_backtest(bare, mode="realistic").mode == "realistic"  # table copied away from its cache
    with pytest.raises(BacktestTableError, match="exact"):
        load_backtest(bare, mode="exact")
    with pytest.raises(BacktestTableError, match="not found"):
        load_backtest(tmp_path / "nope")


def test_table_acceptance_is_realistic_only(tmp_path):
    bt = load_backtest(_table_dir(tmp_path, [_row(eps=1.0), _row(eps=2.0, accepted=False)]))
    acc = TableAcceptance(bt)
    assert acc(PROG, 1.0) and not acc(PROG, 2.0)
    assert not acc(PROG, 0.25) and not acc("n_tx(ALERT.sender, dir=any, window=before(24))", 1.0)
    exact = load_backtest(_table_dir(tmp_path / "e", [_row()], mode="exact"), require=None)
    with pytest.raises(BacktestTableError):
        TableAcceptance(exact)


def test_t2_row_format_feeds_lrtable(tmp_path):
    t = LRTable.from_rows(json.loads(x) for x in (_table_dir(tmp_path, [_row()]) / "lr_table.jsonl").read_text().splitlines())
    e = t.get(PROG, 1.0)
    assert e.lr_for(0) == 0.5 and e.lr_for(1) == 1.0 and e.lr_for(7) == 3.0  # bins [1, 2]: b0 b1 b2


# -- run script on the harness toolbox ------------------------------------------------------------------------------
def _run(mod, tmp_path, *extra, system="planner", n="6"):
    out = tmp_path / f"out_{len(list(tmp_path.glob('out_*')))}"
    assert mod.main(["--system", system, "--n", n, "--toolbox", "harness", "--out", str(out), "--synthetic-labels",
                     *extra]) == 0  # fmt: skip
    return next(out.iterdir())


def test_harness_run_writes_v1_rows_and_is_labelled_mock(tmp_path):
    mod = _script()
    run = _run(mod, tmp_path)
    log = pl.read_parquet(run / "runlog.parquet")
    assert (log["variant"] == "harness_mock").all() and (log["log_version"] == 1).all()
    assert runlog.validate(log.columns, log["event"].to_list()) == []
    v, s = log.filter(pl.col("event") == "verdict"), log.filter(pl.col("event") == "step")
    assert v["p0"].null_count() == 0 and v["home_bank"].null_count() == 0 and (v["trials"] == 1).all()
    assert v["eps_spent"].null_count() == 0 and v["sim_days"].null_count() == 0
    assert s["gate"].is_in(["allowed", "blocked", "over_budget"]).all() and s["kind"].null_count() < s.height
    summary = json.loads((run / "summary.json").read_text())
    assert summary["questions"]["asked"] == s.height and summary["questions"]["eps_spent_total"] > 0
    assert abs(summary["questions"]["eps_spent_total"] - v["eps_spent"].sum()) < 1e-9
    # eps_spent is what the toolbox charged: replies only
    spent = s.filter(pl.col("response_kind") == "reply")["eps"].sum()
    assert abs(spent - v["eps_spent"].sum()) < 1e-9


def test_backtest_flag_gives_log_lr_and_scoring_runs(tmp_path):
    mod = _script()
    first = _run(mod, tmp_path)
    steps = pl.read_parquet(first / "runlog.parquet").filter(pl.col("event") == "step")
    asked = steps.filter(pl.col("response_kind") == "reply").select("program_canonical", "eps", "kind").unique()
    assert asked.height
    full = {"true": 2.0, "false": 0.5, **{f"b{i}": 1.0 + i for i in range(8)}, **{str(i): 1.5 for i in range(40)}}
    rows = [{"program_canonical": r["program_canonical"], "eps": r["eps"], "accepted": True, "lr": full,
             "family": "fam", "bins": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] if r["kind"] in ("count", "set_count") else None}
            for r in asked.iter_rows(named=True)]  # fmt: skip
    d = _table_dir(tmp_path, rows)
    run = _run(mod, tmp_path, "--backtest", str(d))
    log = pl.read_parquet(run / "runlog.parquet")
    s = log.filter(pl.col("event") == "step")
    used = s.filter(pl.col("log_lr").is_not_null())
    assert used.height > 0 and used["family"].null_count() == 0
    assert all(math.isfinite(x) for x in used["log_lr"].to_list())
    summary = json.loads((run / "summary.json").read_text())
    assert summary["fusion"] == "NaiveLRFusion" and summary["backtest"]["mode"] == "realistic"
    assert summary["questions"]["used_by_fusion"] == used.height
    v = log.filter(pl.col("event") == "verdict")
    labels = dict(zip(v["alert_id"].to_list(), v["label"].to_list(), strict=True))
    [r] = runs.score(log, labels, n_boot=50)  # T4's scoring runs on our run dir
    assert r["n"] == 6 and r["system"] == "ours_compose" and r["variant"] == "harness_mock"
    assert r["mean_eps_spent"] > 0


def test_exact_table_refused_by_run_script(tmp_path):
    mod = _script()
    d = _table_dir(tmp_path, [_row()], mode="exact")
    with pytest.raises(SystemExit, match="exact"):
        mod.main(["--toolbox", "harness", "--backtest", str(d), "--n", "1", "--out", str(tmp_path / "o")])


def test_fusion_params_need_a_table(tmp_path):
    mod = _script()
    p = tmp_path / "f.json"
    p.write_text("{}")
    with pytest.raises(SystemExit):
        mod.main(["--fusion-params", str(p), "--n", "1", "--out", str(tmp_path / "o")])


def test_fused_lr_over_loaded_table(tmp_path):
    from ftm.fusion import fuse

    bt = load_backtest(_table_dir(tmp_path, [_row()]))
    f = FusedLR(fuse.Fusion("naive"), bt.table)
    assert f.table is bt.table


def test_group_d_and_reflexion_trials_through_harness(tmp_path):
    mod = _script()
    for system in ("react", "reflexion"):
        run = _run(mod, tmp_path, system=system, n="3")
        log = pl.read_parquet(run / "runlog.parquet")
        v = log.filter(pl.col("event") == "verdict")
        assert (log["system"] == system).all() and v["trials"].min() >= 1
        if system == "reflexion":  # the per-alert map stays in summary.json, the count is in the v1 `trials` column
            per = json.loads((run / "summary.json").read_text())["trials_per_alert"]
            assert sorted(per.values()) == sorted(v["trials"].to_list())


def test_mock_harness_runs_never_write_test_rows(tmp_path):
    mod = _script()
    with pytest.raises(SystemExit, match="harness_mock"):
        mod.refuse_devtools_test("harness_mock", "test")
    mod.refuse_devtools_test("harness", "test")
    mod.refuse_devtools_test("harness_mock", "val")
