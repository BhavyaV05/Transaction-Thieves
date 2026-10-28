import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("val_runs_table", Path(__file__).resolve().parents[1] / "scripts" / "val_runs_table.py")
vrt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vrt)


def test_systems_with_low_verdict_rate_are_not_ranked():
    best = {"rewoo": {"system": "rewoo", "brier": 0.1985, "verdict_rate": 0.0},
            "react": {"system": "react", "brier": 0.19, "verdict_rate": 0.96},
            "uot": {"system": "uot", "brier": 0.18, "verdict_rate": 0.9}}
    ok, bad = vrt.split_eligible(best)
    assert [r["system"] for r in ok] == ["react"] and [r["system"] for r in bad] == ["uot", "rewoo"]
