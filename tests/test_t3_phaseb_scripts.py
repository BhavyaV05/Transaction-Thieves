"""T3 phase B scripts on synthetic inputs: E10 table + frozen cascade JSON, the inputs for T4's analysis, the val-run table."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


REPORT = {"alpha": 0.05, "temperatures": {"stop": 1.3}, "conformal_q": {"fast": {"stop": 0.4}, "large": {"stop": 0.6}},
          "eval_argmax_accuracy": {"stop": {"n": 100, "fast": 0.9, "large": 0.95}},
          "decisions": {"stop": {"n_fit": 50, "n_cal": 50, "n_eval": 100, "temperature": 1.3, "fast_full_set": False, "large_full_set": False,
                                 "always_defer": False, "top_label_ece_eval": 0.03, "kept_fast": 0.7, "to_large": 0.2, "to_default": 0.1,
                                 "error_marginal": 0.05, "bound_2alpha": 0.1, "within_bound": True}}}


def test_e10_report_writes_table_and_the_json_t4_reads(tmp_path):
    e10, fin = load("e10_report"), load("t3_final_inputs")
    rep = tmp_path / "r.json"
    rep.write_text(json.dumps(REPORT))
    lat = tmp_path / "lat.json"
    lat.write_text(json.dumps({"tuned_s0": {"ms_per_decision": 20.0}, "large": {"ms_per_decision": 400.0}}))
    md, js = tmp_path / "e10.md", tmp_path / "fast.json"
    assert e10.main(["--arm", f"tuned_s0={rep}", "--latency", str(lat), "--frozen", "tuned_s0", "--md", str(md), "--json", str(js)]) == 0
    d = json.loads(js.read_text())
    t = d["decision_types"][0]
    assert t["name"] == "stop" and t["error_marginal"] == 0.05 and t["kept"] == 0.7
    assert abs(t["speedup"] - 400 / (20 + 0.3 * 400)) < 1e-9 and d["conformal_q"]["fast"]["stop"] == 0.4
    assert "tuned_s0" in md.read_text() and fin.check_fast(js)["frozen_arm"] == "tuned_s0"


def test_t3_final_inputs_e18_csv(tmp_path):
    fin = load("t3_final_inputs")
    for st in ("stage1", "stage2"):
        (tmp_path / "e18" / st).mkdir(parents=True)
        (tmp_path / "e18" / st / "summary.csv").write_text("candidate,brier\nm,0.2\n")
    assert fin.e18_csv(tmp_path / "e18", tmp_path / "o.csv") == 2
    assert (tmp_path / "o.csv").read_text().splitlines()[0] == "stage,candidate,brier"


def test_e10_final_three_phases_on_the_synthetic_path_are_blind(tmp_path):
    e10f = load("e10_final")
    ids = tmp_path / "ids.txt"
    ids.write_text("F000001\nF000002\n")
    out = tmp_path / "final_synthetic" / "j"
    for ph in ("fast", "large", "cascade"):
        assert e10f.main(["--phase", ph, "--alert-ids", str(ids), "--out", str(out)]) == 0
    dec = [json.loads(x) for x in (out / "decisions.jsonl").read_text().splitlines()]
    assert len(dec) == 2 * 5 * 2 and all("label" not in d for d in dec)
    import polars as pl

    df = pl.read_parquet(out / "runlog.parquet")
    assert sorted(df["alert_id"].to_list()) == ["F000001", "F000002"] and "label" not in df.columns or df["label"].null_count() == df.height


def test_e10_report_embeds_served_model_trivial_baseline_and_selection(tmp_path):
    e10 = load("e10_report")
    assert e10.served_model("tuned_s2") == "fast-s2" and e10.served_model("untuned_4b") is None
    rep, sel, js = tmp_path / "r.json", tmp_path / "s.json", tmp_path / "fast.json"
    rep.write_text(json.dumps(REPORT))
    sel.write_text(json.dumps({"chosen": "tuned_s2", "rule": "r", "scores": {}, "balance": {"stop": {"trivial_majority_eval_acc": 0.77}}}))
    assert e10.main(["--arm", f"tuned_s2={rep}", "--select", str(sel), "--frozen", "tuned_s2", "--md", str(tmp_path / "e.md"), "--json", str(js)]) == 0
    d = json.loads(js.read_text())
    assert d["served_model"] == "fast-s2" and d["selection"]["chosen"] == "tuned_s2"
    assert d["decision_types"][0]["trivial_majority"] == 0.77 and d["decision_types"][0]["accuracy_fast"] == 0.9


def test_e10_select_rule_uses_cal_only_ties_to_lowest_seed_and_flags_bounds():
    sel = load("e10_select")
    assert sel.at_bound(20.0) and sel.at_bound(0.05) and not sel.at_bound(1.0)
    cal = {a: {"stop": {"cal_error_marginal": e, "cal_kept_fast": k}} for a, e, k in
           [("tuned_s0", 0.02, 0.8), ("tuned_s1", 0.02, 0.8), ("tuned_s2", 0.2, 0.99)]}  # fmt: skip
    c = sel.choose(cal, ["tuned_s0", "tuned_s1", "tuned_s2"], 0.05)
    assert c["chosen"] == "tuned_s0" and not c["scores"]["tuned_s2"]["within_2alpha_on_cal"]  # s2 is over the bound, s0/s1 tie
    cal["tuned_s2"]["stop"]["cal_error_marginal"] = 0.1
    assert sel.choose(cal, ["tuned_s0", "tuned_s1", "tuned_s2"], 0.05)["chosen"] == "tuned_s2"
    rows = [{"decision": "stop", "val_part": p, "label": y} for p, y in [("fit", "yes"), ("fit", "yes"), ("fit", "no"), ("cal", "no"), ("eval", "yes"), ("eval", "no")]]
    b = sel.balance(rows)["stop"]
    assert b["majority_label_fit"] == "yes" and b["trivial_majority_eval_acc"] == 0.5
