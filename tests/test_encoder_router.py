"""Week 10: the classifier baselines (TF-IDF floor, ModernBERT/DeBERTa), the RouteLLM-style router, the async study."""

import importlib.util
import json
import math
import random
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ftm.fastmodel.decisions import DECISIONS, DecisionExample
from ftm.fastmodel.encoder import ENCODERS, EncoderConfig, config_dict
from ftm.fastmodel.router import ConfidenceRouter, _auc, confidence_features, route_report
from ftm.orchestrator.ablations import synthetic_env
from ftm.orchestrator.async_study import batching_grid, batching_study, h8_check, load_study
from ftm.planners.learned import split_val

ROOT = Path(__file__).resolve().parents[1]
WORDS = "bank wire cash payment account transfer amount home partner ring layer pass".split()


def rows(n, seed=0, flip=0.08, decision="worth_asking"):
    """Separable toy decisions: `alpha` in the text means yes, `beta` means no, with a share of flipped labels."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        y = rng.random() < 0.5
        words = rng.sample(WORDS, 6) + ["alpha" if y else "beta"]
        rng.shuffle(words)
        label = "yes" if (y != (rng.random() < flip)) else "no"
        out.append(DecisionExample(decision, " ".join(words), label, f"A{i}", "train"))
    return out


# --- pins --------------------------------------------------------------------------------------------------------------
def test_the_encoders_are_pinned_to_a_revision():
    assert {"modernbert-base", "deberta-v3-base", "deberta-v3-small"} <= set(ENCODERS)
    for hub, rev in ENCODERS.values():
        assert "/" in hub and len(rev) == 40 and set(rev) <= set("0123456789abcdef")
    c = config_dict(EncoderConfig(model="deberta-v3-small"))
    assert c["hub_id"] == "microsoft/deberta-v3-small" and c["revision"] == ENCODERS["deberta-v3-small"][1]


# --- the router --------------------------------------------------------------------------------------------------------
def test_confidence_features_are_top_margin_and_normalised_entropy():
    f = confidence_features(np.array([0.7, 0.2, 0.1]))
    assert f[0] == pytest.approx(0.7) and f[1] == pytest.approx(0.5) and 0 < f[2] < 1
    assert confidence_features(np.array([0.5, 0.5]))[2] == pytest.approx(1.0)
    assert confidence_features(np.array([1.0, 0.0]))[2] == pytest.approx(0.0, abs=1e-9)


def test_the_learned_router_scores_confident_answers_higher_and_the_threshold_sets_the_share():
    rng = np.random.default_rng(0)
    probs, correct = [], []
    for _ in range(600):
        conf = rng.uniform(0.5, 1.0)
        probs.append(np.array([conf, 1 - conf]))
        correct.append(rng.random() < conf)  # a calibrated fast tier
    r = ConfidenceRouter("learned").fit(["worth_asking"] * 600, probs, correct)
    assert r.score("worth_asking", np.array([0.95, 0.05])) > r.score("worth_asking", np.array([0.55, 0.45]))
    scores = [r.score("worth_asking", p) for p in probs]
    for share in (0.1, 0.3, 0.5):
        r.set_share(scores, share)
        got = np.mean([r.defers("worth_asking", p) for p in probs])
        assert abs(got - share) < 0.02
    r.set_share(scores, 0.0)
    assert not any(r.defers("worth_asking", p) for p in probs)
    r.set_share(scores, 1.0)
    assert all(r.defers("worth_asking", p) for p in probs)
    assert ConfidenceRouter("maxprob").fit(["worth_asking"], [probs[0]], [True]).score("worth_asking", np.array([0.8, 0.2])) == 0.8
    with pytest.raises(ValueError):
        ConfidenceRouter("magic").fit(["a"], [probs[0]], [True])


def test_the_router_never_returns_the_safe_default_and_calls_the_large_tier_only_when_it_defers():
    r = ConfidenceRouter("maxprob").fit(["stop"], [np.array([0.5, 0.5])], [True])
    r.threshold = 0.9
    calls = []

    def large():
        calls.append(1)
        return {"yes": 0.1, "no": 0.9}

    sure = r.decide("stop", {"yes": 0.97, "no": 0.03}, large)
    assert (sure.answer, sure.source, calls) == ("yes", "fast", [])
    unsure = r.decide("stop", {"yes": 0.6, "no": 0.4}, large)
    assert (unsure.answer, unsure.source, len(calls)) == ("no", "large", 1)


def test_auc_matches_the_definition():
    assert _auc(np.array([0.1, 0.2, 0.8, 0.9]), [False, False, True, True]) == 1.0
    assert _auc(np.array([0.9, 0.8, 0.2, 0.1]), [False, False, True, True]) == 0.0
    assert _auc(np.array([0.5, 0.5, 0.5, 0.5]), [False, True, False, True]) == 0.5
    assert _auc(np.array([0.1, 0.2]), [True, True]) is None


def toy_report(n=3200, seed=0, alpha=0.1, large_right=1.0):
    rng = random.Random(seed)
    exs, parts, fast, large = [], [], [], []
    for i in range(n):
        y = rng.random() < 0.5
        label = "yes" if y else "no"
        conf = rng.uniform(0.5, 0.99)
        right = rng.random() < conf  # the fast tier is right as often as it is sure
        pick = label if right else ("no" if y else "yes")
        p = {pick: conf, ("no" if pick == "yes" else "yes"): 1 - conf}
        fast.append(np.log([p["yes"], p["no"]]))
        part = "fit" if i < n // 2 else "cal" if i < 3 * n // 4 else "eval"
        parts.append(part)
        exs.append(DecisionExample("worth_asking", f"row {i}", label, f"A{i}", "val"))
        sure = rng.random() < large_right
        lp = label if sure else ("no" if y else "yes")
        large.append(None if part == "fit" else {lp: 0.97, ("no" if lp == "yes" else "yes"): 0.03})
    return route_report(exs, parts, fast, large, alpha)


def test_route_report_curve_runs_from_fast_only_to_large_only():
    rep = toy_report()
    assert rep["rows"] == {"fit": 1600, "cal": 800, "eval": 800}
    for kind in ("learned", "maxprob"):
        curve = rep["routers"][kind]["curve"]
        assert curve[0]["error"] == pytest.approx(rep["fast_only_error"]) and curve[0]["large_share"] == 0.0
        assert curve[-1]["error"] == pytest.approx(rep["large_only_error"]) == 0.0 and curve[-1]["large_share"] == 1.0
        errs = [c["error"] for c in curve]
        assert errs[0] > 0.1 and errs[-1] == 0.0
        assert errs[5] < errs[0]  # half of the rows to a perfect 8B, picked by confidence: fewer errors than none
        assert rep["routers"][kind]["auc_fast_correct_eval"] > 0.6  # confidence does say something about being right
        m = rep["routers"][kind]["matched_to_conformal"]
        assert abs(m["large_share"] - rep["conformal"]["unsettled_share_cal"]) < 0.08
    c = rep["conformal"]
    assert c["kept_fast"] + c["to_large"] + c["to_default"] == pytest.approx(1.0)


def test_route_report_refuses_rows_it_may_not_use():
    ex = [DecisionExample("worth_asking", "x", "yes", "A", "val")] * 3
    lp = [np.log([0.6, 0.4])] * 3
    with pytest.raises(ValueError, match="val part"):
        route_report(ex, ["fit", "cal", "train"], lp, [None, {"yes": 1, "no": 0}, {"yes": 1, "no": 0}])
    with pytest.raises(ValueError, match="8B must be scored"):
        route_report(ex, ["fit", "cal", "eval"], lp, [None, None, {"yes": 1, "no": 0}])


# --- TF-IDF floor --------------------------------------------------------------------------------------------------------
def test_tfidf_floor_learns_the_toy_rule_and_keeps_unseen_labels_at_the_floor():
    pytest.importorskip("sklearn")
    from ftm.fastmodel.encoder import FLOOR_LOGP, TfidfDecisionModel

    m = TfidfDecisionModel().fit(rows(400, 0))
    test = rows(200, 1)
    acc = np.mean([DECISIONS["worth_asking"].labels[int(np.argmax(m.raw_logps(e)))] == e.label for e in test])
    assert acc > 0.85
    p = m.predict_proba(test[0])
    assert set(p) == {"yes", "no"} and sum(p.values()) == pytest.approx(1.0)
    typ = [DecisionExample("typology", f"fan {i % 3}", "fan_in" if i % 2 else "cycle", "A", "train") for i in range(40)]
    lp = TfidfDecisionModel().fit(typ).logps_batch("typology", ["fan 1"])[0]
    assert lp.shape == (9,) and (lp == FLOOR_LOGP).sum() == 7  # seven of nine typologies never seen
    with pytest.raises(KeyError):
        m.raw_logps(DecisionExample("stop", "x", "yes"))


# --- ModernBERT / DeBERTa code path on a tiny random encoder -----------------------------------------------------------
@pytest.fixture(scope="module")
def tiny():
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {w: i for i, w in enumerate(["[PAD]", "[UNK]", "alpha", "beta", *WORDS])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]")
    cfg = transformers.BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                  intermediate_size=32, max_position_embeddings=32, num_labels=2)  # fmt: skip
    return tok, cfg, transformers


def test_the_encoder_trains_scores_saves_and_loads(tiny, tmp_path):
    import torch

    from ftm.fastmodel.encoder import EncoderDecisionModel, train_decision_encoder

    tok, cfg, tf = tiny
    torch.manual_seed(0)
    model = tf.BertForSequenceClassification(cfg)
    ecfg = EncoderConfig(epochs=12, batch_size=32, lr=3e-3, max_len=16, seed=0)
    tok, model, hist = train_decision_encoder(rows(320, 0, flip=0.0), "worth_asking", ecfg, tok, model, device="cpu")
    assert len(hist) == 12 and hist[-1]["loss"] < hist[0]["loss"] and hist[-1]["train_accuracy"] > 0.95
    m = EncoderDecisionModel({"worth_asking": (tok, model)}, max_len=16, device="cpu")
    test = rows(100, 5, flip=0.0)
    lp = m.logps_batch("worth_asking", [e.context for e in test])
    assert lp.shape == (100, 2) and np.allclose(np.exp(lp).sum(1), 1.0, atol=1e-5)
    assert np.mean([DECISIONS["worth_asking"].labels[int(np.argmax(x))] == e.label for x, e in zip(lp, test, strict=True)]) > 0.9
    assert np.allclose(m.raw_logps(test[3]), lp[3], atol=1e-5)  # batch order does not matter
    assert sum(m.predict_proba(test[0]).values()) == pytest.approx(1.0)
    m.save(tmp_path / "enc", {"config": "toy"})
    back = EncoderDecisionModel.load(tmp_path / "enc", device="cpu")
    assert np.allclose(back.logps_batch("worth_asking", [e.context for e in test]), lp, atol=1e-5)
    with pytest.raises(KeyError, match="no classifier"):
        m.logps_batch("stop", ["x"])


def test_training_rejects_labels_outside_the_decision_and_empty_sets(tiny):
    from ftm.fastmodel.encoder import train_decision_encoder

    tok, cfg, tf = tiny
    model = tf.BertForSequenceClassification(cfg)
    with pytest.raises(ValueError, match="not in"):
        train_decision_encoder([DecisionExample("worth_asking", "x", "maybe")], "worth_asking", EncoderConfig(), tok, model, "cpu")
    with pytest.raises(ValueError, match="no training rows"):
        train_decision_encoder([DecisionExample("stop", "x", "yes")], "worth_asking", EncoderConfig(), tok, model, "cpu")


# --- the script ------------------------------------------------------------------------------------------------------------
def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_set(tmp_path):
    def dump(path, exs, split):
        path.write_text("\n".join(json.dumps({"decision": e.decision, "context": e.context, "label": e.label,
                                              "alert_id": e.alert_id, "split": split}) for e in exs))  # fmt: skip

    train = rows(400, 0)
    val = rows(240, 9)
    dump(tmp_path / "train.jsonl", train, "train")
    dump(tmp_path / "val.jsonl", val, "val")
    ids = [e.alert_id for e in val]
    parts = ["fit" if i < 120 else "cal" if i < 180 else "eval" for i in range(len(ids))]
    pl.DataFrame({"alert_id": ids, "val_part": parts}).write_parquet(tmp_path / "val.parquet")
    rng = random.Random(1)
    with (tmp_path / "large.jsonl").open("w") as f:
        for i, (e, p) in enumerate(zip(val, parts, strict=True)):
            if p != "fit":
                ans = e.label if rng.random() < 0.9 else ("no" if e.label == "yes" else "yes")
                f.write(json.dumps({"i": i, "probs": {ans: 0.9, ("no" if ans == "yes" else "yes"): 0.1}}) + "\n")


def test_script_tfidf_then_route_with_the_format_of_t3s_scorer(tmp_path):
    pytest.importorskip("sklearn")
    write_set(tmp_path)
    cli = load_script("encoder_baseline")
    common = ["--decisions", str(tmp_path / "val.jsonl"), "--alerts", str(tmp_path / "val.parquet")]
    assert cli.main(["tfidf", "--decisions-train", str(tmp_path / "train.jsonl"), *common, "--out", str(tmp_path / "fast.jsonl")]) == 0
    scored = [json.loads(x) for x in (tmp_path / "fast.jsonl").read_text().splitlines()]
    assert sorted(r["i"] for r in scored) == list(range(240)) and all(len(r["logps"]) == 2 for r in scored)
    assert cli.main(["route", *common, "--fast", str(tmp_path / "fast.jsonl"), "--large", str(tmp_path / "large.jsonl"),
                     "--out", str(tmp_path / "route.json")]) == 0  # fmt: skip
    rep = json.loads((tmp_path / "route.json").read_text())
    assert rep["rows"] == {"fit": 120, "cal": 60, "eval": 60} and set(rep["routers"]) == {"learned", "maxprob"}
    assert rep["selection"]["large_kind"] == "json" and rep["fast_only_error"] < 0.3
    # T3's own report reads the same file
    t3 = load_script("score_fast_baseline")
    out = tmp_path / "t3.json"
    assert t3.main(["--decisions", str(tmp_path / "val.jsonl"), "--alerts", str(tmp_path / "val.parquet"), "report",
                    "--fast", str(tmp_path / "fast.jsonl"), "--large", str(tmp_path / "large.jsonl"), "--out", str(out)]) == 0  # fmt: skip
    assert "worth_asking" in json.loads(out.read_text())["decisions"]


def test_script_trains_on_train_rows_only(tmp_path):
    write_set(tmp_path)
    cli = load_script("encoder_baseline")
    with pytest.raises(SystemExit, match="train split only"):
        cli.read_train(tmp_path / "val.jsonl")


# --- async runs ----------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def env():
    return synthetic_env(300, 6)


def test_a_budget_of_two_leaves_the_second_wave_slot_empty_and_four_fills_it(env):
    ev = split_val(env.val)[1]
    pool = env.pools["realistic"]
    low = batching_study("eig", env.banks, ev, pool, budget=2.0)
    high = batching_study("eig", env.banks, ev, pool, budget=4.0)
    assert {r["arm"] for r in low} == {"sequential", "batched"} and low[0]["budget"] == 2.0 == low[1]["budget"]
    assert low[1]["share_2plus"] < 0.3  # T3's finding: the (2, 1) schedule never reaches its second slot at eps 2
    assert high[1]["share_2plus"] > 0.9 and high[1]["mean_sim_days"] < high[0]["mean_sim_days"]  # a wave costs its slowest answer
    assert high[0]["mean_eps"] > low[0]["mean_eps"]


def test_h8_check_reads_the_two_arms():
    seq = {"arm": "sequential", "brier_gain": 0.05, "mean_sim_days": 2.0, "share_2plus": 1.0}
    bat = {"arm": "batched", "brier_gain": 0.045, "mean_sim_days": 1.1, "share_2plus": 1.0}
    h = h8_check([seq, bat])
    assert h["gain_kept"] == pytest.approx(0.9) and h["time_ratio"] == pytest.approx(0.55) and h["claim_holds"] is True
    assert h8_check([seq, {**bat, "mean_sim_days": 1.8}])["claim_holds"] is False
    assert h8_check([seq, {**bat, "brier_gain": 0.02}])["claim_holds"] is False
    assert h8_check([{**seq, "brier_gain": 0.0}, bat])["claim_holds"] is None  # nothing to keep a share of
    assert math.isnan(float("nan")) and h8_check([seq, {**bat, "share_2plus": 0.0}])["claim_holds"] is None


def test_grid_and_load_study(env):
    ev = split_val(env.val)[1][:60]
    pool = env.pools["realistic"]
    grid = batching_grid(["random"], env.banks, ev, pool, budgets=(2.0, 4.0))
    assert [r["arm"] for r in grid].count("h8") == 2 and sum(r["arm"] in ("sequential", "batched") for r in grid) == 4
    load = load_study("random", env.banks, ev, pool, gaps=(1.0, 0.001))
    by = {(r["arrival_gap_days"], r["arm"]): r for r in load}
    assert by[(0.001, "sequential")]["max_in_flight"] > by[(1.0, "sequential")]["max_in_flight"]  # a busier queue overlaps more
    assert by[(0.001, "sequential")]["makespan_days"] < by[(1.0, "sequential")]["makespan_days"]
    assert by[(1.0, "batched")]["mean_latency_days"] <= by[(1.0, "sequential")]["mean_latency_days"] + 1e-9


def test_async_cli(tmp_path):
    cli = load_script("async_study")
    out = cli.main(["--synthetic", "240", "--systems", "random", "--budgets", "4", "--gaps", "0.5", "--out", str(tmp_path), "--seed", "2"])
    assert (tmp_path / "async.json").exists() and len(out["h8"]) == 1 and len(out["rows"]) == 2 and len(out["load"]) == 2
    with pytest.raises(SystemExit):
        cli.main(["--synthetic", "100", "--systems", "learned"])  # needs --policy
