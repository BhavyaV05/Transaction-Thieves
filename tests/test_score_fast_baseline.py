"""T3: scripts/score_fast_baseline.py row selection (val only, joined to val_part, deterministic cap)."""

import importlib.util
import json
import sys
from pathlib import Path

import polars as pl
import pytest

SPEC = importlib.util.spec_from_file_location("score_fast_baseline", Path(__file__).resolve().parents[1] / "scripts" / "score_fast_baseline.py")
sfb = importlib.util.module_from_spec(SPEC)
sys.modules["score_fast_baseline"] = sfb
SPEC.loader.exec_module(sfb)


def _write(tmp_path, n_alerts=30):
    parts = ["fit", "cal", "eval"]
    alerts = pl.DataFrame({"alert_id": [f"V{i}" for i in range(n_alerts)], "val_part": [parts[i % 3] for i in range(n_alerts)]})
    alerts.write_parquet(tmp_path / "val.parquet")
    with (tmp_path / "d.jsonl").open("w") as f:
        for i in range(n_alerts):
            for dec in ("worth_asking", "stop"):
                f.write(json.dumps({"alert_id": f"V{i}", "split": "val", "decision": dec, "context": "c",
                                    "label": "yes" if i % 2 else "no"}) + "\n")
    return tmp_path / "d.jsonl", tmp_path / "val.parquet"


def test_load_rows_joins_parts_and_cap_is_per_decision_part_and_deterministic(tmp_path):
    d, a = _write(tmp_path)
    full = sfb.load_rows(d, a, None, 0)
    assert len(full) == 60 and {r["val_part"] for r in full} == {"fit", "cal", "eval"}
    c1, c2 = sfb.load_rows(d, a, 4, 0), sfb.load_rows(d, a, 4, 0)
    assert [r["i"] for r in c1] == [r["i"] for r in c2]
    assert len(c1) == 2 * 3 * 4  # 2 decisions x 3 parts x cap
    assert [r["i"] for r in c1] != [r["i"] for r in sfb.load_rows(d, a, 4, 1)]


def test_load_rows_refuses_non_val_rows(tmp_path):
    d, a = _write(tmp_path, 3)
    rows = [json.loads(x) for x in d.read_text().splitlines()]
    rows[0]["split"] = "train"
    d.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit, match="val rows only"):
        sfb.load_rows(d, a, None, 0)
