"""T2's final-run runner for the non-LLM arms (ftm/orchestrator/final_nonllm.py, scripts/final_non_llm.py), synthetic world only."""

import importlib.util
import json
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from ftm.backtest.synth import make_world
from ftm.eval.runlog import validate
from ftm.final import plan as fp
from ftm.final import runner as rn
from ftm.harness.runlog_rows import to_frame
from ftm.orchestrator import final_nonllm as fn
from ftm.orchestrator.ablations import load_spec, run_ablations, synthetic_env
from ftm.orchestrator.runner import run_alerts
from ftm.planners.learned import split_val

ROOT = Path(__file__).resolve().parents[1]
spec_ = importlib.util.spec_from_file_location("final_non_llm", ROOT / "scripts" / "final_non_llm.py")
cli = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(cli)

SEED = 0
ARMS = ["random", "eig", "eig_eps_0.5", "eig_cap_2", "eig_batched_2_1", "eig_decline_20", "eig_fusion_per_family"]


@pytest.fixture(scope="module")
def env():
    spec = load_spec()
    return spec, synthetic_env(320, SEED, spec.eps_levels[1:], enumerator=False)


def brier(rows, alerts):
    y = {a.alert_id: float(a.label) for a in alerts}
    p0 = {a.alert_id: a.p0 for a in alerts}
    v = [r for r in rows if r["event"] == "verdict" and r["alert_id"] in y]
    p = [p0[r["alert_id"]] if r["probability"] is None else r["probability"] for r in v]
    return float(np.mean([(pi - y[r["alert_id"]]) ** 2 for pi, r in zip(p, v, strict=True)]))


def test_the_final_runner_reproduces_the_g4_arms_on_the_same_alerts(env):
    """G5 builds an arm with the code G4 builds it with: on the same alerts, banks, tables and seed the probabilities are the
    G4 ones (here compared through the Brier score, computed outside the runner from the labels the synthetic world has)."""
    spec, e = env
    fit, ev = split_val(e.val, spec.fit_frac)
    g4 = {r["arm"]: r for r in run_ablations(e.banks, e.val, e.pools, spec, seed=SEED, n_boot=20, only=ARMS)}
    for name in ARMS:
        res = fn.run_arm(fn.find_arm(spec, name), spec, banks=e.banks, alerts=ev, fit=fit, fit_banks=e.banks, pools=e.pools,
                         policy=None, seed=SEED, base={"run_id": "t", "commit": "c", "machine": "m"}, split="val")  # fmt: skip
        assert brier(res.rows, ev) == pytest.approx(g4[name]["brier_final"], abs=1e-12), name
        assert res.counts["mean_questions"] == pytest.approx(g4[name]["mean_questions"])


def test_a_blind_run_writes_no_label_and_no_score(env, tmp_path):
    spec, e = env
    fit, ev = split_val(e.val, spec.fit_frac)
    res = fn.run_arm(fn.find_arm(spec, "eig"), spec, banks=e.banks, alerts=ev, fit=fit, fit_banks=e.banks, pools=e.pools,
                     policy=None, seed=SEED, base={"run_id": "t", "commit": "c", "machine": "m"}, split="test")  # fmt: skip
    assert all(r.get("label") is None for r in res.rows)
    assert set(res.counts) == {"alerts", "with_partner_bank", "no_partner_bank", "mean_questions", "mean_eps", "refused_share",
                               "mean_sim_days", "p95_sim_days"}  # fmt: skip
    path = fn.write_run(tmp_path, res, {"arm": "eig"})
    log = pl.read_parquet(path)
    assert validate(log.columns, log["event"].to_list()) == [] and log["label"].null_count() == log.height
    assert set(log["system"]) == {"eig"} and set(log["variant"]) == {"eig"} and set(log["split"]) == {"test"}
    text = (tmp_path / "summary.json").read_text()
    assert "brier" not in text and "positives" not in text
    # run_alerts itself: blind drops the labels and the scores, not the rest
    s = run_alerts.__doc__
    assert "blind" in s
    bad = to_frame([{**r, "label": 1} if r["event"] == "verdict" else r for r in res.rows])
    assert bad["label"].null_count() < bad.height
    with pytest.raises(RuntimeError, match="labels"):
        fn.write_run(tmp_path / "bad", fn.ArmResult([{**r, "label": 1} if r["event"] == "verdict" else r for r in res.rows]), {})


def test_run_alerts_blind_has_no_brier_and_unblind_is_unchanged(env):
    spec, e = env
    fit, ev = split_val(e.val, spec.fit_frac)
    arm = fn.find_arm(spec, "eig")
    from ftm.orchestrator.ablations import prepare_arm

    def go(blind):
        prep = prepare_arm(arm, spec, banks=e.banks, bindings={a.alert_id: a.binding for a in ev}, fit=fit, fit_banks=e.banks,
                           pools=e.pools, policy=None, seed=SEED, cache={})  # fmt: skip
        base = {"run_id": "t", "commit": "", "machine": "", "system": "eig", "variant": "eig", "seed": SEED}
        return run_alerts(prep.planner, ev, prep.harness, prep.limits, base, "val", blind=blind)

    open_, blind = go(False), go(True)
    assert open_.summary["brier_final"] is not None and blind.summary["brier_final"] is None
    assert blind.summary["brier_p0"] is None and blind.summary["brier_gain"] is None and blind.summary["positives"] is None
    assert [v.probability for v in open_.verdicts] == [v.probability for v in blind.verdicts]  # same investigations
    assert any(r["label"] is not None for r in open_.rows if r["event"] == "verdict")
    assert all(r.get("label") is None for r in blind.rows)


def test_alerts_without_a_partner_bank_still_get_a_verdict_row_scored_as_p0(env):
    spec, e = env
    fit, ev = split_val(e.val, spec.fit_frac)
    res = fn.run_arm(fn.find_arm(spec, "eig"), spec, banks=e.banks, alerts=ev[:5], fit=fit, fit_banks=e.banks, pools=e.pools,
                     policy=None, seed=SEED, base={"run_id": "t", "commit": "c", "machine": "m"}, split="test",
                     unbound=[("U1", "3", 0.02), ("U2", "4", 0.5)])  # fmt: skip
    un = [r for r in res.rows if r["alert_id"] in ("U1", "U2")]
    assert [(r["event"], r["probability"], r["p0"], r["stop_reason"], r["home_bank"]) for r in un] == [
        ("verdict", None, 0.02, "no_partner_bank", "3"), ("verdict", None, 0.5, "no_partner_bank", "4")]  # fmt: skip
    assert res.counts["alerts"] == 7 and res.counts["no_partner_bank"] == 2
    assert fn.write_run  # the rows pass the frame's schema
    to_frame(res.rows)


def test_an_arm_with_no_table_says_so_instead_of_running(env, tmp_path):
    spec, e = env
    cfg = fn.FinalConfig("hi_medium", {"realistic": None, "enumerator": None, "exact_labels": None})
    with pytest.raises(FileNotFoundError, match="tables.enumerator"):
        fn.load_pools(cfg, fn.find_arm(spec, "enumerator"))
    with pytest.raises(RuntimeError, match="cannot run"):
        fit, ev = split_val(e.val, spec.fit_frac)
        fn.run_arm(fn.find_arm(spec, "enumerator"), spec, banks=e.banks, alerts=ev, fit=fit, fit_banks=e.banks, pools=e.pools,
                   policy=None, seed=SEED, base={}, split="test")  # fmt: skip
    problems = fn.check_config(cfg, spec, tmp_path)
    assert len(problems) == 3 and all("is not set" in p for p in problems)
    t = tmp_path / "t.jsonl"
    t.write_text("")
    cfg2 = fn.FinalConfig("hi_medium", {"realistic": str(t), "enumerator": "nope.jsonl", "exact_labels": str(t)})
    assert [("does not exist" in p) for p in fn.check_config(cfg2, spec, tmp_path)] == [True]


def test_load_config_rejects_unknown_tables_and_hashes_the_file(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump({"data": "hi_small", "tables": {"realistic": "a.jsonl"}}))
    c = fn.load_config(f)
    assert c.data == "hi_small" and c.tables["realistic"] == "a.jsonl" and c.tables["enumerator"] is None and len(c.sha256) == 64
    f.write_text(yaml.safe_dump({"data": "x", "tables": {"bogus": "a"}}))
    with pytest.raises(ValueError, match="unknown question tables"):
        fn.load_config(f)
    assert fn.load_config().data == "hi_medium"  # the checked-in config parses


def test_the_checked_in_config_covers_every_pool_the_arms_use():
    spec = load_spec()
    cfg = fn.load_config()
    assert {a.settings["pool"] for a in spec.arms} <= set(fn.POOL_NAMES) == set(cfg.tables)
    assert fn.arm_names(spec)[-1] == fn.RAW_SHARE and len(set(fn.arm_names(spec))) == len(spec.arms) + 1


def test_the_raw_share_arm_runs_blind_and_asks_for_records_only():
    w = make_world(160, seed=3)
    alerts = sorted(w.alerts, key=lambda a: a.binding.t0)
    cut = int(len(alerts) * 0.6)
    rate = sum(a.label for a in alerts[:cut]) / cut
    train, ev = [replace(a, p0=rate) for a in alerts[:cut]], [replace(a, p0=rate) for a in alerts[cut:]]
    res = fn.run_raw_share(train_banks=w.banks, train=train, banks=w.banks, alerts=ev, seed=0,
                           base={"run_id": "t", "commit": "c", "machine": "m"}, split="test", unbound=[("U1", "1", 0.1)])  # fmt: skip
    v = [r for r in res.rows if r["event"] == "verdict"]
    assert len(v) == len(ev) + 1 and all(r["label"] is None for r in v)
    assert {r["system"] for r in res.rows} == {"raw_share"} and {r["variant"] for r in res.rows} == {"raw_share"}
    steps = [r for r in res.rows if r["event"] == "step"]
    assert steps and {r["program_canonical"] for r in steps} == {"request_records"} and {r["eps"] for r in steps} == {0.0}
    assert all(0 < r["probability"] < 1 for r in v if r["alert_id"] != "U1")


# -- the command ---------------------------------------------------------------------------------------------------
def test_cli_check_and_list(capsys, tmp_path):
    assert cli.main(["--list"]) == 0
    assert "eig_fusion_per_family" in capsys.readouterr().out
    assert cli.main(["--check"]) == 1  # the checked-in config has no table paths yet
    assert "is not set" in capsys.readouterr().out
    t = tmp_path / "t.jsonl"
    t.write_text("")
    good = tmp_path / "ok.yaml"
    good.write_text(yaml.safe_dump({"data": "hi_medium", "tables": {k: str(t) for k in fn.POOL_NAMES}}))
    assert cli.main(["--check", "--config", str(good)]) == 0
    with pytest.raises(SystemExit):
        cli.main(["--arm", "nope", "--out", str(tmp_path), "--data", "synthetic"])
    with pytest.raises(SystemExit):
        cli.main(["--arm", "eig", "--out", str(tmp_path / "x")])  # real data needs --alert-ids


def test_cli_dry_run_writes_a_log_the_final_runner_accepts(tmp_path):
    ids = [f"E{i:05d}" for i in range(40)]
    f = tmp_path / "ids.txt"
    f.write_text("\n".join(ids) + "\n")
    out = tmp_path / "job"
    argv = ["--arm", "eig_eps_1", "--alert-ids", str(f), "--out", str(out), "--data", "synthetic", "--synthetic-n", "200",
            "--run-id", "r", "--commit", "c0", "--machine", "m"]  # fmt: skip
    assert cli.main(argv) == 0
    counts = rn.validate_runlog(out / "runlog.parquet", set(ids))
    assert counts["n_alerts_with_verdict"] == 40 and counts["seeds"] == [0]
    meta = json.loads((out / "summary.json").read_text())
    assert meta["blind"] is True and meta["ids_sha256"] == rn.ids_sha256(ids) and meta["arm"] == "eig_eps_1"
    log = pl.read_parquet(out / "runlog.parquet")
    assert set(log["run_id"]) == {"r"} and set(log["machine"]) == {"m"} and set(log["commit"]) == {"c0"}
    # the same command twice gives the same log
    out2 = tmp_path / "job2"
    cli.main([*argv[:5], str(out2), *argv[6:]])
    assert pl.read_parquet(out2 / "runlog.parquet").equals(log)


def test_real_data_without_the_test_lock_is_refused(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"data": "hi_small", "tables": {k: str(cfg) for k in fn.POOL_NAMES}}))
    ids = tmp_path / "ids.txt"
    ids.write_text("T1\n")
    with pytest.raises(ValueError, match="locked until G5"):
        cli.main(["--arm", "eig", "--alert-ids", str(ids), "--out", str(tmp_path / "o"), "--config", str(cfg)])


# -- the job plan --------------------------------------------------------------------------------------------------
def test_every_nonllm_job_of_the_plan_has_this_command():
    ext = yaml.safe_load((ROOT / "prereg" / "external_list.yaml").read_text())
    abl = yaml.safe_load(fp.ABLATIONS.read_text())
    jobs = fp.build_plan(ext, fp.load_jobs_config(), abl, None)
    by = {j.job_id: j for j in jobs}
    ctx = types.SimpleNamespace(run_dir=Path("run"), data="hi_medium", machine_name="m", run_id="g5", commit="abc")
    for arm in abl["arms"] + [{"name": fn.RAW_SHARE}]:
        for alerts in ("subset", "full_test"):
            job = by[f"nonllm.{arm['name']}.{alerts}" if arm["name"] != fn.RAW_SHARE else f"raw_share.raw_share.{alerts}"]
            assert job.cmd and "scripts/final_non_llm.py" in job.cmd, job.job_id
            argv = rn.command_for(job, ctx, Path("out"), Path(f"{alerts}_ids.txt"), 0)
            assert argv[argv.index("--arm") + 1] == arm["name"]
            assert argv[argv.index("--alert-ids") + 1].endswith(f"{alerts}_ids.txt") and "--data" in argv
            assert job.machine_class == "cpu" and job.est_hours > 0
    cmd = by["raw_share.raw_share.subset"]
    assert cmd.owner == "T2" and cmd.group == "extra"


def test_command_templates_keep_windows_paths_intact():
    """The final-run command is split with shlex (no shell); a Windows path must survive it with its backslashes."""
    job = fp.Job("nonllm.eig.subset", "nonllm", "eig", "external_cmd", "subset", [0], "cpu", 0.5,
                 cmd="python scripts/final_non_llm.py --arm {variant} --alert-ids {alert_ids} --out {out}")  # fmt: skip
    ctx = types.SimpleNamespace(run_dir=Path("run"), data="hi_medium", machine_name="m", run_id="g5", commit="abc")
    out = Path(r"C:\Users\x y\run\nonllm.eig.subset")
    ids = Path(r"C:\Users\x y\run\subset_ids.txt")
    argv = rn.command_for(job, ctx, out, ids, 0)
    assert argv[-1] == str(out) and argv[-3] == str(ids)
    assert argv[:3] == ["python", "scripts/final_non_llm.py", "--arm"]
