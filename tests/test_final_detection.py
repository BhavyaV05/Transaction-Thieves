"""T1: the detection run-log writer of the final run (scripts/final_detection.py). Synthetic final scores under tmp_path, FTM_FINAL_RUN monkeypatched."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import final_detection as fd  # noqa: E402

from ftm.eval import runlog  # noqa: E402
from ftm.final import plan, runner  # noqa: E402

N, DATA = 60, "hi_medium"


@pytest.fixture()
def root(tmp_path):
    """data/ with a test alert set (carrying a label column that must never be read) and a score file of every supported system and variant."""
    rng = np.random.default_rng(0)
    d = tmp_path / "data"
    (d / "alerts" / DATA).mkdir(parents=True)
    tx, bank = np.arange(1000, 1000 + N), rng.integers(1, 6, N)
    pl.DataFrame({"alert_id": [f"F{i:06d}" for i in range(N)], "tx_id": tx, "bank": bank, "p0": rng.random(N), "label": rng.random(N) < 0.3, "test_tail": [False] * N}).write_parquet(d / "alerts" / DATA / "test.parquet")
    fin = d / "scores" / DATA / "final"
    fin.mkdir(parents=True)
    for name, kind, model, datasets in fd.SOURCES.values():
        if DATA not in datasets:
            continue
        # every transaction in two bank views (the alert's home bank and another), except 5 alerts that have no score
        rows = [(int(t), int(b)) for t, b in zip(tx[5:], bank[5:], strict=True)] + [(int(t), 99) for t in tx[5:]]
        df = pl.DataFrame({"tx_id": [r[0] for r in rows], "bank": [r[1] if kind == "view" else None for r in rows], "split": "test", "p0": rng.random(len(rows)), "label": True})
        if kind == "tx":
            df = df.unique("tx_id")
        df = df.with_columns(model=pl.lit(model or "other"))
        for seed in (0,):
            df.write_parquet(fin / name.format(seed=seed, eps="8"))
    return d


def run(monkeypatch, root, tmp_path, system, variant, ids=None, **kw):
    f = tmp_path / "ids.txt"
    f.write_text("\n".join(ids or [f"F{i:06d}" for i in range(0, N, 2)]))
    out = tmp_path / f"out_{system}_{variant}"
    argv = ["final_detection.py", "--system", system, "--variant", variant, "--data", kw.get("data", DATA), "--seed", "0", "--alert-ids", str(f), "--out", str(out), "--root", str(root), "--run-id", "g5", "--commit", "abc"]
    monkeypatch.setattr(sys, "argv", argv)
    fd.main()
    return out / "runlog.parquet"


def test_every_supported_system_writes_a_valid_label_free_run_log(root, tmp_path, monkeypatch):
    monkeypatch.setenv("FTM_FINAL_RUN", "1")
    ids = {f"F{i:06d}" for i in range(0, N, 2)}
    for system, variant in [k for k, v in fd.SOURCES.items() if DATA in v[3]]:
        p = run(monkeypatch, root, tmp_path, system, variant)
        df = pl.read_parquet(p)
        assert "label" not in df.columns and not runlog.validate(df.columns, df["event"].unique().to_list())
        assert set(df["alert_id"]) == ids and df.height == len(ids) and set(df["event"]) == {"verdict"} and set(df["split"]) == {"test"}
        assert set(df["system"]) == {system} and set(df["variant"]) == {variant} and set(df["run_id"]) == {"g5"}
        assert runner.validate_runlog(p, ids)["n_alerts_with_verdict"] == len(ids)  # the runner's own check (no labels, ids in the set, no duplicates)
        assert df["probability"].null_count() == 3  # alerts 0, 2, 4 have no score; the rest do
        assert df["p0"].null_count() == 0 and df["probability"].drop_nulls().is_between(0, 1).all()


def test_one_bank_scores_are_matched_on_the_home_bank(root, tmp_path, monkeypatch):
    monkeypatch.setenv("FTM_FINAL_RUN", "1")
    df = pl.read_parquet(run(monkeypatch, root, tmp_path, "multi_gnn", "one_bank")).sort("alert_id")
    al = pl.read_parquet(root / "alerts" / DATA / "test.parquet", columns=["alert_id", "tx_id", "bank"])
    sc = pl.read_parquet(root / "scores" / DATA / "final" / "p0_multi_gnn__one_bank.parquet")
    want = al.join(sc, on=["tx_id", "bank"]).select("alert_id", "p0").rename({"p0": "want"})
    j = df.join(want, on="alert_id")
    assert j.height == df.height - 3 and np.allclose(j["probability"].to_numpy(), j["want"].to_numpy())


def test_refuses_without_the_final_run_lock(root, tmp_path, monkeypatch):
    monkeypatch.delenv("FTM_FINAL_RUN", raising=False)
    with pytest.raises(Exception, match="locked|FTM_FINAL_RUN"):
        run(monkeypatch, root, tmp_path, "multi_gnn", "one_bank")
    assert not (tmp_path / "out_multi_gnn_one_bank").exists()


def test_unknown_unavailable_missing_and_stray_inputs_exit_with_a_reason(root, tmp_path, monkeypatch):
    monkeypatch.setenv("FTM_FINAL_RUN", "1")
    for system, variant, why in [("nope", "x", "unknown"), ("fraudgt", "one_bank", "no one-bank FraudGT"), ("exstraqt", "see_everything", "not implemented"), ("fraudgt", "see_everything", "only on")]:
        with pytest.raises(SystemExit, match=why):
            run(monkeypatch, root, tmp_path, system, variant)
    (root / "scores" / DATA / "final" / "p0_mega_gnn__see_everything.parquet").unlink()
    with pytest.raises(SystemExit, match="does not exist"):
        run(monkeypatch, root, tmp_path, "mega_gnn", "see_everything")
    with pytest.raises(SystemExit, match="not in"):
        run(monkeypatch, root, tmp_path, "multi_gnn", "one_bank", ids=["F000000", "F999999"])


def test_every_group_a_and_b_system_the_plan_can_emit_has_a_score_file_or_a_clear_error():
    ext = yaml.safe_load((ROOT / "prereg" / "external_list.yaml").read_text())
    cfg = plan.load_jobs_config()
    jobs = [j for j in plan.build_plan(ext, cfg, None, None) if j.group in ("A", "B")]
    assert jobs and {j.system for j in jobs} >= {"multi_gnn", "mega_gnn", "fraudgt", "dp_fedavg_gnn", "tian_2025"}
    for j in jobs:
        try:
            name, kind, _ = fd.source(j.system, j.variant, DATA, 0)
            assert name.endswith(".parquet") and kind in ("view", "tx")
        except SystemExit as e:
            assert "no T1 scores" in str(e) and j.system in str(e), (j.system, j.variant, e)


def test_the_detection_template_runs_under_the_interpreter_of_run_final(tmp_path):
    """jobs.yaml's template starts with `python`; the runner replaces it with sys.executable (`python` is 3.10 without polars on the laptop)."""
    from types import SimpleNamespace

    cfg = plan.load_jobs_config()
    tpl = cfg["external_cmd"]["templates"]["detection"]
    job = SimpleNamespace(cmd=tpl, alerts="subset", system="multi_gnn", variant="one_bank", job_id="multi_gnn.one_bank.subset")
    ctx = SimpleNamespace(run_dir=tmp_path, data="hi_medium", machine_name="t1", run_id="g5", commit="abc")
    argv = runner.command_for(job, ctx, tmp_path / "job", tmp_path / "ids.txt", 0)
    assert argv[0] == sys.executable and argv[1] == "scripts/final_detection.py" and argv[argv.index("--system") + 1] == "multi_gnn"
    assert argv[argv.index("--alert-ids") + 1] == str(tmp_path / "ids.txt") and argv[argv.index("--out") + 1] == str(tmp_path / "job")
    assert (ROOT / argv[1]).exists()


def test_test_alerts_cmd_is_the_final_intake_for_the_run_data():
    """jobs.yaml: the fallback that builds data/alerts/{data}/test.parquet is `make_alerts.py --final` on the run's dataset (run_final formats {data} and {out})."""
    from ftm.final import t1

    cfg = plan.load_jobs_config()
    argv = t1._cmd_argv(cfg["test_alerts"]["cmd"].format(data="hi_medium", out="unused"))
    assert argv[0] == sys.executable and argv[1:] == ["scripts/make_alerts.py", "--data", "hi_medium", "--final"]
    assert "--final" in (ROOT / argv[1]).read_text()
