"""T1: FraudGT dump into p0 (calibration, fill of the unscored rows, subset check, stored-calibrator final mode) and the final-run wrapper. Synthetic."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

pytest.importorskip("sklearn")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fraudgt_final as ff  # noqa: E402
import fraudgt_scores as fs  # noqa: E402

N_VAL, N_TEST = 4000, 1500


def _rows(split: str, n: int, first_tx: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = datetime(2022, 9, 8 if split == "val" else 12)
    ts = np.sort(rng.integers(0, 2 * 1440, n))
    y = (rng.random(n) < 0.08).astype(int)
    return pl.DataFrame({"tx_id": np.arange(first_tx, first_tx + n), "timestamp": [t0 + timedelta(minutes=int(m)) for m in ts], "is_laundering": y.astype(bool)}), y, rng


@pytest.fixture()
def world(tmp_path, monkeypatch):
    """Val and test rows, the dump directory (scored 99 %), the scratch outputs; `io` replaced by in-memory frames."""
    val, yv, rng = _rows("val", N_VAL, 0, 1)
    test, yt, _ = _rows("test", N_TEST, N_VAL, 2)
    dump = tmp_path / "dump"
    dump.mkdir()

    def write(df, y, name, drop):
        keep = np.ones(df.height, bool)
        keep[drop] = False
        raw = np.clip(0.02 + 0.5 * y * rng.random(df.height) + 0.3 * rng.random(df.height) * (rng.random(df.height) < 0.3), 0, 1)
        pl.DataFrame({"tx_id": df["tx_id"].to_numpy()[keep], "label": 1 - y[keep], "score": raw[keep]}).write_csv(dump / name)  # label deliberately wrong: must be ignored

    write(val, yv, "pred_val.csv", [3, 17, 900])
    write(test, yt, "pred_test.csv", [5, 600])
    (dump / "pred_meta.json").write_text(json.dumps({"best_epoch": 7, "best_val_f1": 0.5}))
    frames = {"val": val, "test": test}
    monkeypatch.setattr(fs.io, "load_config", lambda ds: {"parquet_dir": "unused"})
    monkeypatch.setattr(fs.io, "scan_transactions", lambda pq, split, columns: frames[split].lazy().select(columns))
    monkeypatch.setattr(fs, "SCORES", tmp_path / "scores")
    monkeypatch.setattr(fs, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(fs, "commit", lambda: "abc1234")
    return dump, val, test, tmp_path


def _args(dump):
    return type("A", (), {"dump": str(dump)})()


def test_val_run_calibrates_on_fit_fills_unscored_and_ignores_the_dump_label(world):
    dump, val, _, tmp = world
    fs.val_run(_args(dump), "hi_small")
    t = pl.read_parquet(tmp / "scores" / "hi_small" / "p0_fraudgt.parquet")
    assert t.height == N_VAL and t["tx_id"].n_unique() == N_VAL
    assert {"tx_id", "bank", "mode", "model", "split", "val_part", "p0", "score_raw", "label", "seed", "commit", "scored"} <= set(t.columns)
    assert t["mode"].unique().to_list() == ["see_everything"] and t["bank"].null_count() == N_VAL
    assert t["label"].cast(pl.Int8).to_list() == val.sort(["timestamp", "tx_id"])["is_laundering"].cast(pl.Int8).to_list()  # our labels, not the dump's (flipped)
    un = t.filter(~pl.col("scored"))
    assert un.height == 3 and un["score_raw"].null_count() == 3
    fill = t.filter(pl.col("val_part") == "fit")["label"].cast(pl.Float64).mean()
    assert np.allclose(un["p0"].to_numpy(), fill)
    assert t.filter(pl.col("scored"))["p0"].is_between(0, 1).all() and set(t["val_part"].unique()) == {"fit", "cal", "eval"}
    meta = json.loads((tmp / "runs" / "p0_hi_small_fraudgt_see_everything.json").read_text())
    assert meta["unscored"] == 3 and meta["scored"] == N_VAL - 3 and meta["calibration"] in ("isotonic", "platt")
    assert meta["pred_meta"]["best_epoch"] == 7 and meta["commit"] == "abc1234" and meta["eval"]["brier"] > 0


def test_subset_check_rejects_unknown_and_duplicate_tx_ids(world):
    dump, val, _, _ = world
    d = pl.read_csv(dump / "pred_val.csv")
    pl.concat([d, pl.DataFrame({"tx_id": [10**9], "label": [0], "score": [0.5]})]).write_csv(dump / "pred_val.csv")
    with pytest.raises(SystemExit, match="not rows of ours"):
        fs.read_dump(dump / "pred_val.csv", val)
    pl.concat([d, d.head(1)]).write_csv(dump / "pred_val.csv")
    with pytest.raises(SystemExit, match="duplicate"):
        fs.read_dump(dump / "pred_val.csv", val)
    d.with_columns(score=pl.col("score") + 2).write_csv(dump / "pred_val.csv")
    with pytest.raises(SystemExit, match="outside"):
        fs.read_dump(dump / "pred_val.csv", val)


def test_final_refuses_without_the_guard_then_reproduces_val_and_writes_test(world, monkeypatch):
    dump, _, test, tmp = world
    fs.val_run(_args(dump), "hi_small")
    monkeypatch.delenv("FTM_FINAL_RUN", raising=False)
    with pytest.raises(Exception, match="locked|FTM_FINAL_RUN"):
        fs.final_run(_args(dump), "hi_small")
    assert not (tmp / "scores" / "hi_small" / "final").exists()
    monkeypatch.setenv("FTM_FINAL_RUN", "1")
    fs.final_run(_args(dump), "hi_small")
    t = pl.read_parquet(tmp / "scores" / "hi_small" / "final" / "p0_fraudgt.parquet")
    assert t.height == N_TEST and set(t["split"]) == {"test"} and t["val_part"].null_count() == N_TEST
    assert t.filter(~pl.col("scored")).height == 2 and t["p0"].is_not_null().all()
    rec = json.loads((tmp / "runs" / "final_p0_hi_small_fraudgt.json").read_text())
    assert rec["val_p0_max_abs_diff"] < 1e-6 and rec["unscored"] == 2
    # the same stored calibrator: a test row with the same raw score as a val row gets the same p0
    v = pl.read_parquet(tmp / "scores" / "hi_small" / "p0_fraudgt.parquet").filter(pl.col("scored"))
    cal, _, _ = fs.calibrate(v["score_raw"].to_numpy(), v["label"].cast(pl.Int8).to_numpy(), (v["val_part"] == "fit").to_numpy())
    s = t.filter(pl.col("scored")).head(50)
    assert np.allclose(cal(s["score_raw"].to_numpy()), s["p0"].to_numpy(), atol=1e-12)


def test_final_exits_when_the_val_dump_no_longer_reproduces(world, monkeypatch):
    dump, _, _, _ = world
    fs.val_run(_args(dump), "hi_small")
    d = pl.read_csv(dump / "pred_val.csv")
    d.with_columns(score=(pl.col("score") * 0.5)).write_csv(dump / "pred_val.csv")  # a different val dump than the one the calibrator saw
    monkeypatch.setenv("FTM_FINAL_RUN", "1")
    with pytest.raises(SystemExit, match="do not use"):
        fs.final_run(_args(dump), "hi_small")


def test_wrapper_builds_the_rescoring_and_calibration_commands_without_running_them(tmp_path):
    cfg = {"checkout": "~/co/FraudGT", "config": "configs/AML-Small-HI/x.yaml", "dump_dir": "data/fraudgt/val_run", "data_dir": "/data/fg"}
    a, b = ff.build(cfg, dump=str(tmp_path / "d"))
    assert a["cwd"] == str(Path("~/co/FraudGT").expanduser()) and a["argv"][1:3] == ["-m", "fraudGT.main"]
    assert a["env"] == {"T1_DUMP_ONLY": "1", "T1_OUT": str(tmp_path / "d")}  # T1_EVAL_TEST comes from run_final's step env, never from here
    assert "T1_EVAL_TEST" not in a["env"] and "out_dir" in a["argv"] and a["argv"][a["argv"].index("out_dir") + 1] == f"{tmp_path / 'd'}/final"
    assert a["argv"][a["argv"].index("dataset.dir") + 1] == "/data/fg"
    assert b["argv"][1:] == ["scripts/fraudgt_scores.py", "--final", "--data", "hi_small", "--dump", str(tmp_path / "d")] and b["cwd"] == str(ff.ROOT)
    assert ff.build(cfg)[0]["env"]["T1_OUT"] == str(ff.ROOT / "data/fraudgt/val_run")


def test_wrapper_refuses_without_the_lock_and_without_t1_eval_test(monkeypatch, capsys):
    monkeypatch.delenv("FTM_FINAL_RUN", raising=False)
    monkeypatch.setattr(sys, "argv", ["fraudgt_final.py"])
    with pytest.raises(Exception, match="locked|FTM_FINAL_RUN"):
        ff.main()
    monkeypatch.setenv("FTM_FINAL_RUN", "1")
    monkeypatch.delenv("T1_EVAL_TEST", raising=False)
    with pytest.raises(SystemExit, match="T1_EVAL_TEST"):
        ff.main()
    monkeypatch.setattr(sys, "argv", ["fraudgt_final.py", "--print"])
    ff.main()
    assert "T1_DUMP_ONLY=1" in capsys.readouterr().out
