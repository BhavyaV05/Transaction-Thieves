"""T1 final-run configs: they parse, name scripts and flags that exist, and the checker never runs under FTM_FINAL_RUN."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import t1_final_check as chk  # noqa: E402


def test_parse_command():
    assert chk.parse_command("python scripts/train_gnn.py --our-split --model mega_gnn --max-epochs 100") == (
        "scripts/train_gnn.py", ["--max-epochs", "--model", "--our-split"])
    assert chk.parse_command("python3.12 -m ftm.retrieval.evidence_views --data hi_small --parts cal eval") == (
        "-m ftm.retrieval.evidence_views", ["--data", "--parts"])
    assert chk.parse_command("ls -l") == (None, [])


def test_committed_configs_are_complete_and_name_real_scripts():
    problems, final = chk.check(artefacts=False, run_help=False)
    assert problems == []
    assert len(final) >= 7 and all("to_build" in line or "needs_val_run" in line or "ready" in line or "done" in line for line in final)
    for name in chk.REQUIRED:  # every flag named in a val_command is in the script source (CI has no torch for --help)
        cfg = chk.yaml.safe_load((chk.CONFIGS / f"{name}.yaml").read_text())
        for key, val in chk.walk(cfg):
            if key in ("val_command", "val_commands") and isinstance(val, str):
                target, flags = chk.parse_command(val)
                src = chk.ROOT / (target[3:].replace(".", "/") + ".py" if target.startswith("-m ") else target)
                assert src.exists() and all(f in src.read_text() for f in flags), (name, val)


def test_check_flags_runs_help_and_finds_a_missing_flag():
    ok, _ = chk.check_flags("scripts/make_alerts.py", ["--data"])
    assert ok
    ok, msg = chk.check_flags("scripts/make_alerts.py", ["--no-such-flag"])
    assert not ok and "--no-such-flag" in msg
    ok, msg = chk.check_flags("scripts/nope.py", [])
    assert not ok


def test_refuses_the_final_run(monkeypatch):
    monkeypatch.setenv("FTM_FINAL_RUN", "1")
    with pytest.raises(SystemExit):
        chk.check()


def test_items_waiting_for_a_val_run_are_listed_as_expected_states_not_problems():
    w = chk.waiting()
    assert any(line.startswith("fraudgt.final_entry") and line.endswith("needs_val_run") for line in w)
    problems, _ = chk.check(artefacts=False, run_help=False)
    assert problems == []
