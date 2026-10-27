"""T1: `scripts/train_layerwise.py` on a sampled dataset (HI-Medium): val run, then `--final` at the val-chosen step. Toy views, FTM_FINAL_RUN monkeypatched, outputs in tmp_path."""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("torch_scatter")
pytest.importorskip("sklearn")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_layerwise as tl  # noqa: E402
from test_layerwise import _toy_views  # noqa: E402

ARGS = ["--data", "hi_medium", "--steps", "8", "--eval-every", "4", "--lr", "5e-3", "--batch", "40", "--fanout", "4", "--eval-batch", "16", "--device", "cpu",
        "--arms", "sync_sgd:bwd", "--seeds", "0"]  # fmt: skip


@pytest.fixture()
def toy(monkeypatch, tmp_path):
    frames, x = _toy_views(False)
    frames_te, x_te = _toy_views(True)
    parts = {b: np.where(np.arange((f["split"] == "val").sum()) % 2 == 0, "fit", "eval") for b, f in frames.items()}
    cut = (datetime(2022, 9, 7), datetime(2022, 9, 8))

    def load(ds, banks, min_train_pos, final=False):
        base = (frames, x, parts, {}, cut)
        return base + (frames_te, x_te) if final else base

    monkeypatch.setattr(tl, "load", load)
    monkeypatch.setattr(tl, "commit", lambda: "abc1234")
    monkeypatch.setattr(sys, "argv", ["train_layerwise.py", *ARGS, "--scratch", str(tmp_path)])
    monkeypatch.setattr(tl, "SCORES", tmp_path / "scores")
    monkeypatch.setattr(tl, "RUNS", tmp_path / "runs")
    return tmp_path, frames_te


def test_hi_medium_is_sampled_and_the_final_mode_reproduces_the_val_scores(toy, monkeypatch):
    tmp, frames_te = toy
    monkeypatch.delenv("FTM_FINAL_RUN", raising=False)
    tl.main()
    row = json.loads((tmp / "runs" / "layerwise_hi_medium.json").read_text())["rows"][0]
    assert row["sampled"] is True and row["fanout"] == 4 and row["eval_batch"] == 16 and (tmp / "scores" / "hi_medium" / "ceydeli" / "sync_sgd_bwd_seed0.parquet").exists()
    # the final mode: locked without the flag, then reproduces the stored val scores and writes the test rows
    monkeypatch.setattr(sys, "argv", ["train_layerwise.py", *ARGS, "--scratch", str(tmp), "--final"])
    with pytest.raises(Exception, match="locked|FTM_FINAL_RUN"):
        tl.main()
    assert not (tmp / "scores" / "hi_medium" / "final").exists()
    monkeypatch.setenv("FTM_FINAL_RUN", "1")
    tl.main()
    rec = json.loads((tmp / "runs" / "final_layerwise_hi_medium.json").read_text())
    assert rec["sampled"] is True and rec["records"][0]["val_spearman_vs_stored"] > 0.999 and rec["records"][0]["val_max_abs_diff_vs_stored"] < 1e-6
    t = pl.read_parquet(tmp / "scores" / "hi_medium" / "final" / "sync_sgd_bwd_seed0.parquet")
    assert t.height == sum(int((f["split"] == "test").sum()) for f in frames_te.values()) and set(t["split"]) == {"test"} and t["p0"].is_between(0, 1).all()
