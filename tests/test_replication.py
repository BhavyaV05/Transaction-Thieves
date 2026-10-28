"""T4: the replication check (time-block sign agreement, Mantel-Haenszel) and the G2 filter, on planted series only."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from ftm.backtest.lr import estimate_lr
from ftm.eval import g2
from ftm.eval import replication as rep

CFG = rep.ReplicationConfig()
ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- pure pieces --------------------------------------------------------------------------------------------------
def test_log_lr_is_the_alpha_smoothed_ratio_of_estimate_lr():
    answers, labels = rep.planted_series(600, 1, "genuine")
    est = estimate_lr("bool", answers, labels, None, 1.0)  # eps None: exact, no ceiling clip
    cells = rep.cells_for("bool", answers, None)
    for cell in ("true", "false"):
        assert rep.cell_log_lr(cells, labels, cell) == pytest.approx(math.log(est.lr[cell]))


def test_blocks_are_equal_count_and_in_time_order():
    b = rep.blocks_of(10, 4)
    assert [len(x) for x in b] == [3, 3, 2, 2] and np.concatenate(b).tolist() == list(range(10))


def test_best_cell_takes_the_largest_abs_log_lr_and_the_first_on_a_tie():
    cells = ["a", "a", "b", "b"] * 25
    labels = [True, False, True, False] * 25  # both cells have the same rate in both classes: log LR 0 everywhere
    assert rep.best_cell(cells, labels, ["b", "a"]) == "b"
    cells = ["a" if y else "b" for y in labels]
    assert rep.best_cell(cells, labels, ["a", "b"]) in ("a", "b")


def test_mh_matches_a_hand_computation():
    # two blocks; cell "x" is 2 of 4 launder and 1 of 4 clean in block 1, 0 of 2 launder and 1 of 6 clean in block 2
    cells = ["x", "x", "o", "o", "x", "o", "o", "o"] + ["o", "o", "x", "o", "o", "o", "o", "o"]
    labels = [True] * 4 + [False] * 4 + [True] * 2 + [False] * 6
    # block 1 is not ordered by class here, so use explicit arrays in time order
    cells = ["x", "x", "o", "o", "x", "o", "o", "o", "o", "o", "x", "o", "o", "o", "o", "o"]
    labels = [True, True, True, True, False, False, False, False, True, True, False, False, False, False, False, False]
    num = (2 + 0.5) * 4 / 8 + (0 + 0.5) * 6 / 8
    den = (1 + 0.5) * 4 / 8 + (1 + 0.5) * 2 / 8
    assert rep.mh_log_lr(cells, labels, "x", 2) == pytest.approx(math.log(num / den))


def test_mh_is_nan_without_both_classes_in_any_block():
    assert math.isnan(rep.mh_log_lr(["a"] * 6, [True] * 6, "a", 2))


@pytest.mark.parametrize(
    ("pooled", "block_lrs", "thin", "flag", "reason"),
    [
        (-0.9, [-0.2, -0.3], False, True, "ok"),
        (-0.9, [-0.2, 0.1], False, False, "sign_flip"),
        (-0.9, [0.0, -0.3], False, False, "zero_block"),  # exactly zero has no sign: not a confirmation
        (0.0, [0.1, 0.2], False, False, "zero_pooled"),
        (math.nan, [0.1, 0.2], False, False, "zero_pooled"),
        (-0.9, [-0.2, math.nan], False, False, "zero_block"),
        (-0.9, [-0.2, -0.3], True, False, "thin_block"),  # a block that cannot be judged cannot confirm
    ],
)
def test_sign_rule_states_zero_nan_and_thin_blocks_explicitly(pooled, block_lrs, thin, flag, reason):
    blocks = [{"log_lr": v, "thin": thin and i == 1} for i, v in enumerate(block_lrs)]
    assert rep.sign_replicates(pooled, blocks) == (flag, reason)


def test_few_answers_never_replicate():
    answers, labels = rep.planted_series(80, 0, "genuine")  # halves of 40, ~10 launder: ok; blocks of 4 x 20 < 30
    assert rep.check_question(answers, labels, "bool", None, rep.ReplicationConfig(sign_blocks=4))["reason"] == "thin_block"


# --- the planted confound and the genuine effect ------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["bool", "count"])
def test_a_planted_confound_has_a_strong_pooled_lr_but_does_not_replicate(kind):
    answers, labels = rep.planted_series(4000, 3, "confound", kind)
    bins = (3.0, 6.0) if kind == "count" else None
    r = rep.check_question(answers, labels, kind, bins)
    assert abs(r["pooled_log_lr"]) > 0.5  # the pooled evidence the back-test would claim
    assert not r["replicates"] and r["reason"] == "sign_flip"
    assert all(abs(b["log_lr"]) < 0.35 for b in r["blocks"])  # inside each half there is almost nothing
    assert abs(r["mh_log_lr"]) < 0.25  # the time-stratified estimate recovers ~0


@pytest.mark.parametrize("kind", ["bool", "count"])
@pytest.mark.parametrize("blocks", [2, 4])
def test_a_genuine_effect_replicates_and_mh_recovers_the_true_lr(kind, blocks):
    answers, labels = rep.planted_series(4000, 5, "genuine", kind)
    bins = (3.0, 6.0) if kind == "count" else None
    r = rep.check_question(answers, labels, kind, bins, rep.ReplicationConfig(sign_blocks=blocks))
    assert r["replicates"] and r["reason"] == "ok" and len(r["blocks"]) == blocks
    true = math.log(0.7 / 0.3) if kind == "bool" else None
    sign = np.sign(r["pooled_log_lr"])
    assert all(np.sign(b["log_lr"]) == sign for b in r["blocks"])
    assert np.sign(r["mh_log_lr"]) == sign and abs(r["mh_log_lr"] - r["pooled_log_lr"]) < 0.15
    if true is not None:  # bool: the best cell is "true" (+0.85) or "false" (-0.85), both |log(7/3)|
        assert abs(abs(r["mh_log_lr"]) - true) < 0.15


def test_a_null_question_does_not_replicate_by_sign_alone_only_when_a_block_flips():
    answers, labels = rep.planted_series(4000, 9, "null")
    r = rep.check_question(answers, labels, "bool", None)
    assert abs(r["mh_log_lr"]) < 0.2 and abs(r["pooled_log_lr"]) < 0.2


# --- the report over entries --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def planted():
    return rep.planted_world(800, 0, 400)


def test_report_rows_flag_exactly_the_confounded_questions(planted):
    entries, collect, val_collect = planted
    rows = rep.replication_report(None, None, entries, CFG, collector=collect)
    assert [r["replicates"] for r in rows] == [False, True, False, True]  # the order of rep.PLANTED
    assert all(r["matches_cache"] for r in rows)
    assert {"question", "eps", "cell", "claimed_log_lr", "pooled_log_lr", "block_log_lrs", "mh_log_lr", "replicates"} <= set(rows[0])
    # the claimed LR is the entry's (clipped to the ceiling), the pooled one is not: same sign, never larger
    assert all(r["claimed_log_lr"] * r["pooled_log_lr"] > 0 and abs(r["claimed_log_lr"]) <= abs(r["pooled_log_lr"]) + 1e-9
               for r in rows)  # fmt: skip
    rep.add_validation(rows, entries, val_collect)
    assert [r["agrees_on_val"] for r in rows if r["replicates"]] == [True, True]  # the genuine ones hold on val
    s = rep.summarize(rows)
    assert (s["n_questions"], s["n_replicating"], s["n_non_replicating"]) == (4, 2, 2)
    assert s["val_agreement_replicating"] == {"n": 2, "agree": 2, "rate": 1.0}
    assert s["mean_abs_mh"] < s["mean_abs_pooled"]
    assert "2 of 4" in rep.summary_line(s, "t")
    assert rep.to_markdown(rows, s, "t").count("\n| `") == 4


def test_report_skips_questions_that_are_not_accepted(planted):
    entries, collect, _ = planted
    off = [dict(entries[0], accepted=False), *entries[1:]]
    assert len(rep.replication_report(None, None, off, CFG, collector=collect)) == 3
    assert len(rep.replication_report(None, None, off, CFG, collector=collect, only_accepted=False)) == 4


# --- the G2 filter ------------------------------------------------------------------------------------------------
def test_filter_drops_exactly_the_non_replicating_entries_and_keeps_m(planted):
    entries, collect, _ = planted
    rows = rep.replication_report(None, None, entries, CFG, collector=collect)
    extra = dict(entries[0], program_canonical="not_judged(ALERT.receiver)", accepted=True)  # not in the map: left alone
    batch = [*entries, extra]
    snapshot = json.dumps(batch, sort_keys=True)
    out, info = g2.apply_replication_filter(batch, rep.replicates_map(rows), 0.10, {"replication_blocks": 2})
    assert json.dumps(batch, sort_keys=True) == snapshot  # the inputs are not modified
    assert len(out) == len(batch) and info["m"] == len(batch)
    flags = rep.replicates_map(rows)
    for e, o in zip(batch, out, strict=True):
        key = (e["program_canonical"], e["eps"])
        if key in flags and not flags[key]:
            assert o["e_value"] == 1.0 and o["accepted"] is False
        else:
            assert o["e_value"] == e["e_value"] and o["accepted"] is True
    assert (info["n_judged"], info["n_non_replicating"], info["n_dropped"]) == (4, 2, 2)
    assert (info["accepted_before"], info["accepted_after"]) == (5, 3)
    assert info["replication_blocks"] == 2  # the output states the filter used


def test_an_all_replicating_map_changes_nothing(planted):
    entries, _, _ = planted
    out, info = g2.apply_replication_filter(entries, {(e["program_canonical"], e["eps"]): True for e in entries})
    assert out == entries and info["n_dropped"] == 0 and info["accepted_after"] == info["accepted_before"]


@pytest.fixture(scope="module")
def synth():
    return g2.synthetic_inputs(360, 3, eps_levels=(1.0, 2.0))


def test_g2_with_the_filter_uses_exactly_the_kept_questions(synth):
    banks, entries, val = synth
    accepted = [e for e in entries if e["accepted"]]
    assert len(accepted) >= 2
    drop = {(accepted[0]["program_canonical"], accepted[0]["eps"])}
    rep_map = {(e["program_canonical"], e["eps"]): (e["program_canonical"], e["eps"]) not in drop for e in accepted}
    filtered, info = g2.apply_replication_filter(entries, rep_map)
    assert info["n_dropped"] == 1 and info["m"] == len(entries)
    base = g2.run_g2(banks, entries, val, ask_eps=1.0, eps_levels=(1.0, 2.0), n_boot=100)
    r = g2.run_g2(banks, filtered, val, ask_eps=1.0, eps_levels=(1.0, 2.0), n_boot=100)
    assert base["pool"] == info["accepted_before"] and r["pool"] == info["accepted_after"] <= base["pool"] - 1


def test_the_scripts_leave_the_default_output_unchanged_and_state_the_filter_when_used(tmp_path, synth):
    mod = load_script("g2_check")
    args = ["--world", "synth", "--eps-levels", "1", "2", "--ask-eps", "1.0", "--n-boot", "100", "--seed", "3"]
    out0 = tmp_path / "base.json"
    mod.main([*args, "--out", str(out0)])
    base = json.loads(out0.read_text())
    assert "replication_filter" not in base
    a = mod.argparse.Namespace(replication_filter=None, replication_blocks=None)
    entries = [dict(e) for e in synth[1]]
    assert rep.filter_from_args(a, entries, lambda: pytest.fail("train must not be loaded"))[1] is None  # off by default

    accepted = [e for e in synth[1] if e["accepted"]]
    flt = tmp_path / "flt.json"
    rows = [{"question": e["program_canonical"], "eps": e["eps"], "replicates": i > 0} for i, e in enumerate(accepted)]
    rep.write_report(flt, rows, {}, CFG)
    out1 = tmp_path / "filtered.json"
    mod.main([*args, "--replication-filter", str(flt), "--out", str(out1)])
    r = json.loads(out1.read_text())
    assert r["replication_filter"]["n_dropped"] == 1 and r["replication_filter"]["replication_filter"] == str(flt)
    assert r["pool"] == r["replication_filter"]["accepted_after"] <= base["pool"] - 1


def test_fusion_diagnostics_script_takes_the_filter_too(tmp_path, synth):
    mod = load_script("fusion_diagnostics")
    accepted = [e for e in synth[1] if e["accepted"]]
    rows = [{"question": e["program_canonical"], "eps": e["eps"], "replicates": i > 0} for i, e in enumerate(accepted)]
    flt = tmp_path / "flt.json"
    rep.write_report(flt, rows, {}, CFG)
    args = ["--world", "synth", "--eps-levels", "1", "2", "--ask-eps", "1.0", "--n-boot", "30", "--n-boot-share", "10", "--seed", "3"]
    assert mod.main([*args, "--replication-filter", str(flt), "--out", str(tmp_path / "d.json")]) == mod.DONE
    r = json.loads((tmp_path / "d.json").read_text())
    assert r["replication_filter"]["n_dropped"] == 1 and r["pool"] == r["replication_filter"]["accepted_after"]
    assert mod.main([*args, "--out", str(tmp_path / "d0.json")]) == mod.DONE
    assert "replication_filter" not in json.loads((tmp_path / "d0.json").read_text())


def test_both_filter_options_at_once_is_an_error():
    a = importlib.import_module("argparse").Namespace(replication_filter="x.json", replication_blocks=2)
    with pytest.raises(ValueError, match="not both"):
        rep.filter_from_args(a, [], lambda: None)


def test_replication_check_script_runs_on_the_planted_world(tmp_path):
    mod = load_script("replication_check")
    out = tmp_path / "r.json"
    assert mod.main(["--synthetic", "--val-part", "cal", "--out", str(out)]) == mod.DONE
    doc = json.loads(out.read_text())
    assert doc["summary"]["n_replicating"] == 2 and len(doc["rows"]) == 4 and doc["config"]["sign_blocks"] == 2
    assert out.with_suffix(".md").exists()
    flt, cfg = rep.load_filter(out)
    assert sorted(flt.values()) == [False, False, True, True] and cfg["mh_blocks"] == 4
    assert mod.main(["--data", "no_such_dataset", "--backtest-dir", str(tmp_path / "x" / "y")]) == mod.NOT_RUN
