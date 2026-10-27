"""scripts/e8_report.py on toy run dirs: parity checks and the H6 difference."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e8_report", ROOT / "scripts" / "e8_report.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _arm(root: Path, view: str, probs: list[float | None], **summary) -> None:
    d = root / view / "c000"
    d.mkdir(parents=True)
    base = {"run_id": f"r_{view}", "evidence_view": view, "model": "m", "mode": "compose", "seed": 0, "alerts": "val",
            "val_part": "eval", "data": "toy", "fusion": "NaiveLRFusion", "limits": {}, "alerts_hash": "h",
            "backtest": {"path": "t.jsonl"}}  # fmt: skip
    (d / "summary.json").write_text(json.dumps({**base, **summary}))
    rows = [{"alert_id": f"V{i}", "label": i % 2, "p0": 0.3, "probability": p, "seconds": 1.0, "outputs": 2, "json_ok": 2,
             "questions": 1, "dsl_ok": 1} for i, p in enumerate(probs)]  # fmt: skip
    (d / "per_alert.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_h6_difference_and_no_verdict_is_p0(tmp_path):
    good = [0.1 if i % 2 == 0 else 0.9 for i in range(20)]
    for v in mod.VIEWS:
        _arm(tmp_path, v, good if v == "scorer_pcst" else [None] * 20)
    clusters = tmp_path / "clusters.json"
    clusters.write_text(json.dumps({f"V{i}": i // 2 for i in range(20)}))
    mod.main(["--runs", str(tmp_path), "--clusters", str(clusters), "--n-boot", "200"])
    rep = json.loads((tmp_path / "e8_report.json").read_text())
    assert rep["arms"]["amount"]["brier"] == pytest.approx(rep["p0_brier"])  # no verdict -> p0
    assert rep["h6"]["diff"] < 0 and rep["h6"]["rejects_null"] and rep["h6"]["alerts_with_different_p"] == 20


def test_arms_must_differ_only_in_the_view(tmp_path):
    for v in mod.VIEWS:
        _arm(tmp_path, v, [0.2] * 4, **({"seed": 1} if v == "long_prompt" else {}))
    clusters = tmp_path / "c.json"
    clusters.write_text(json.dumps({f"V{i}": i for i in range(4)}))
    with pytest.raises(SystemExit, match="more than the evidence view"):
        mod.main(["--runs", str(tmp_path), "--clusters", str(clusters), "--n-boot", "50"])
