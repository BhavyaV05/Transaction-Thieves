"""The RAG-bench switches of scripts/run_llm_agent.py: --evidence-view none, --rules {none,fixed,alert,whole}, --no-exemplars
(ftm/compliance/rules_mode.py, the pipeline's rules_mode, the prompt trimmer, FinFRE-RAG zero-shot, summary.json fields)."""

import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from graph_agents_support import alert as g_alert
from test_graph_ours_parity import PLAIN, STOP, Box
from test_groupc_ports import alert, rows

from ftm.compliance.query import FIXED_RULES_QUERY, rules_query
from ftm.compliance.rules_mode import RULES_MODES, RulesModeToolBox, fetch_rules
from ftm.external import finfre_rag as ff
from ftm.graph.agents import finfre_rag as graph_ff
from ftm.llm.devtools import DevRandomToolBox
from ftm.llm.mock import MockLLM
from ftm.planners.llm import LLMPlanner
from ftm.planners.llm_dsl import V0Dsl
from ftm.planners.llm_prompt import RULES_KEEP, Mode, PlannerState, build_messages_with_estimate
from ftm.types import NO_EVIDENCE_TEXT, Limits

ROOT = Path(__file__).resolve().parents[1]
CORPUS = [{"rule_id": f"D{i:02d}", "text": f"document {i} " + "x" * 280} for i in range(31)]


class RulesBox(Box):
    def __init__(self):
        super().__init__()
        self.queries = []

    def lookup_rules(self, q):
        self.queries.append(q)
        return [{"rule_id": f"T{i}", "text": f"top {i} for {q[:12]}"} for i in range(8)]

    def all_rules(self):
        return list(CORPUS)


def prompt_of(llm):
    return llm.requests[0]["messages"][-1]["content"]


def section(text, title):
    m = re.search(rf"## {re.escape(title)}[^\n]*\n(.*?)(?=\n## |\Z)", text, re.S)
    return m.group(1) if m else None


# -- the rules modes ------------------------------------------------------------------------------------------------------
def test_fetch_rules_per_mode():
    box, a = RulesBox(), PLAIN
    assert fetch_rules(box, "none", a) == [] and box.queries == []
    fixed = fetch_rules(box, "fixed", a)
    assert box.queries == [FIXED_RULES_QUERY] and len(fixed) == 5
    alert_mode = fetch_rules(box, "alert", a)
    assert box.queries[-1] == rules_query(a) != FIXED_RULES_QUERY and len(alert_mode) == 5
    assert fetch_rules(box, "whole", a) == CORPUS
    with pytest.raises(ValueError):
        fetch_rules(box, "all", a)
    with pytest.raises(RuntimeError, match="all_rules"):
        fetch_rules(
            Box(), "whole", a
        )  # a toolbox without the whole corpus fails loudly, it never falls back to the top 5


def test_rules_mode_toolbox_wraps_a_systems_own_calls():
    box = RulesBox()
    assert RulesModeToolBox(box, "none").lookup_rules("my query") == []
    RulesModeToolBox(box, "fixed").lookup_rules("my query")
    RulesModeToolBox(box, "alert").lookup_rules("my query")
    assert box.queries == [FIXED_RULES_QUERY, "my query"]
    assert RulesModeToolBox(box, "whole").lookup_rules("my query") == CORPUS
    assert RulesModeToolBox(box, "none").search_library({}) == box.search_library(
        {}
    )  # everything else passes through


def test_dev_toolbox_has_a_whole_corpus():
    assert DevRandomToolBox(seed=0).all_rules()


@pytest.mark.parametrize("mode", RULES_MODES)
def test_planner_prompt_per_rules_mode(mode):
    box, llm = RulesBox(), MockLLM(script=[STOP])
    LLMPlanner(llm, rules_mode=mode).investigate(PLAIN, box, Limits())
    rules = section(prompt_of(llm), "Rules")
    if mode == "none":
        assert rules is None and "Rules" not in prompt_of(llm)
    elif mode == "whole":
        assert rules is not None and len(rules.splitlines()) == len(
            CORPUS
        )  # the 31 documents all fit under the cap here
    else:
        assert len(rules.splitlines()) == 5
        assert box.queries == [FIXED_RULES_QUERY if mode == "fixed" else rules_query(PLAIN)]


def test_use_rules_false_is_still_no_rules():
    llm = MockLLM(script=[STOP])
    LLMPlanner(llm, use_rules=False, rules_mode="whole").investigate(PLAIN, RulesBox(), Limits())
    assert section(prompt_of(llm), "Rules") is None


def test_whole_corpus_is_trimmed_to_the_cap_and_the_count_is_recorded():
    st = PlannerState(
        alert_summary={},
        evidence_text="e1\ne2",
        partner_banks=[1],
        eps_left=1.0,
        calls_left=5,
        rules=list(CORPUS),
    )
    rep = {}
    msgs, est = build_messages_with_estimate(Mode.COMPOSE, st, V0Dsl(), 100_000, 1.0, rep)
    assert rep["rules_kept"] == rep["rules_offered"] == 31
    full = est
    cap = full - 1200  # about 1.2k tokens too many: surplus rules go, the first RULES_KEEP never
    rep2 = {}
    _, est2 = build_messages_with_estimate(Mode.COMPOSE, st, V0Dsl(), cap, 1.0, rep2)
    assert est2 <= cap and RULES_KEEP <= rep2["rules_kept"] < 31 and rep2["rules_offered"] == 31
    rules_text = section(msgs[-1]["content"], "Rules")
    assert rules_text.splitlines()[0].startswith("D00")  # the head of the list is what stays


def test_planner_records_rules_in_prompt():
    llm = MockLLM(script=[STOP])
    p = LLMPlanner(llm, rules_mode="whole")
    p.investigate(PLAIN, RulesBox(), Limits())
    d = p.stats.as_dict()
    assert d["mean_rules_offered"] == 31 and d["mean_rules_in_prompt"] == 31
    q = LLMPlanner(MockLLM(script=[STOP]), rules_mode="none")
    q.investigate(PLAIN, RulesBox(), Limits())
    assert q.stats.as_dict()["mean_rules_in_prompt"] == 0


def test_rules_mode_on_the_adapted_arms_and_validation():
    from ftm.graph.agents.registry import adapted_system

    seen = []

    class B(RulesBox):
        def lookup_rules(self, q):
            seen.append(q)
            return super().lookup_rules(q)

    sys_ = adapted_system(
        "react_adapted",
        MockLLM(responder=lambda m, kw: '{"thought": "t", "choice": 0}'),
        rules_mode="fixed",
    )
    sys_.investigate(g_alert(), B(), Limits())
    assert seen == [FIXED_RULES_QUERY]
    with pytest.raises(ValueError):
        LLMPlanner(MockLLM(), rules_mode="most")


# -- evidence view none ---------------------------------------------------------------------------------------------------
def test_evidence_none_view_reads_nothing_and_prints_one_line():
    from ftm.retrieval.evidence_views import view

    v = view(object(), "none")  # never touches the alert or the data
    assert (
        v.text == NO_EVIDENCE_TEXT and v.tx_ids == [] and v.n_candidates == 0 and v.tokens_est == 0
    )


def test_planner_and_torres_see_no_trail_with_view_none():
    none = replace(PLAIN, summary={**PLAIN.summary, "evidence": NO_EVIDENCE_TEXT})
    llm = MockLLM(script=[STOP])
    LLMPlanner(llm).investigate(none, RulesBox(), Limits())
    assert section(prompt_of(llm), "Evidence").strip() == NO_EVIDENCE_TEXT
    llm2 = MockLLM(script=[STOP])
    LLMPlanner(llm2).investigate(PLAIN, RulesBox(), Limits())
    assert "row1" in section(prompt_of(llm2), "Evidence")
    from ftm.external.torres import Torres

    items = Torres(MockLLM()).bundle(none, RulesBox())
    assert not any(
        i.kind == "transaction" for i in items
    )  # the placeholder line is not an evidence row


# -- FinFRE-RAG zero-shot -------------------------------------------------------------------------------------------------
def test_finfre_zero_shot_prompt_has_no_exemplar_block():
    pool = ff.ExemplarPool(rows(100), k_features=5)
    for cls in (ff.FinfreRag, graph_ff.FinfreRag):
        few, ids = cls(MockLLM(), pool).prompt(alert(), ["- q -> yes"])
        zero, zids = cls(MockLLM(), pool, use_exemplars=False).prompt(alert(), ["- q -> yes"])
        body = zero[1]["content"]
        assert few[0] == zero[0] and len(ids) == 20 and zids == []
        assert (
            "Example 1:" not in body
            and "historical cases" not in body
            and "It is a fraud" not in body
        )
        assert (
            body.startswith(ff.ZERO_SHOT_INSTRUCTION)
            and "Current case: The alerted payment:" in body
        )
        assert "Partner-bank answers" in body
        cur = few[1]["content"].split("Current case: ")[1]
        assert body.endswith(cur)  # the current case itself is byte-identical


def test_finfre_zero_shot_runs_end_to_end_and_traces_no_exemplars():
    pool = ff.ExemplarPool(rows(100), k_features=5)
    for cls in (ff.FinfreRag, graph_ff.FinfreRag):
        f = cls(MockLLM(responder=ff.dev_responder), pool, use_exemplars=False)
        v = f.investigate(alert(), DevRandomToolBox(seed=0), Limits())
        assert v.probability is not None and f.last_trace["exemplars"] == []


# -- the driver: flags and summary.json -----------------------------------------------------------------------------------
def run_driver(tmp_path, *flags, system="planner"):
    out = tmp_path / "run"
    cmd = [
        sys.executable,
        str(ROOT / "scripts/run_llm_agent.py"),
        "--system",
        system,
        "--alerts",
        "synthetic",
        "--n",
        "2",
        "--llm",
        "mock",
        "--profile",
        "none",
        "--run-dir",
        str(out),
        "--run-id",
        "t",
        *flags,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=False)
    return r, out


@pytest.mark.parametrize(
    "flags,expect", [((), "alert"), (("--rules", "none"), "none"), (("--rules", "fixed"), "fixed")]
)
def test_summary_records_rules_mode(tmp_path, flags, expect):
    r, out = run_driver(tmp_path, *flags)
    assert r.returncode == 0, r.stderr[-800:]
    s = json.loads((out / "summary.json").read_text())
    assert s["rules_mode"] == expect and s["exemplars"] is None and "evidence_view" in s
    assert "mean_rules_in_prompt" in s["planner"]


def test_summary_records_whole_corpus_with_the_dev_toolbox(tmp_path):
    r, out = run_driver(tmp_path, "--rules", "whole", "--toolbox", "dev")
    assert r.returncode == 0, r.stderr[-800:]
    s = json.loads((out / "summary.json").read_text())
    assert s["rules_mode"] == "whole" and s["planner"]["mean_rules_offered"] == 2


def test_summary_records_exemplars_and_flag_is_finfre_only(tmp_path):
    r, out = run_driver(
        tmp_path, "--no-exemplars", "--evidence-view", "none", system="finfre_rag_native"
    )
    assert r.returncode == 0, r.stderr[-800:]
    s = json.loads((out / "summary.json").read_text())
    assert s["exemplars"] == "none" and s["evidence_view"] == "none"
    r2, _ = run_driver(tmp_path / "b", "--no-exemplars")
    assert r2.returncode != 0 and "finfre_rag only" in r2.stderr


def test_the_bench_report_unblocks_the_arms_and_verifies_settings(monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import t3_rag_bench_report as rb

    monkeypatch.setattr(rb, "LIB_PATHS", dict.fromkeys(rb.LIB_PATHS))  # about the flags: the library files are gitignored outputs
    arms = {a.name: a for a in rb.build_arms()}
    for n in (
        "ev_none",
        "rules_none",
        "rules_fixed",
        "rules_whole",
        "finfre_rag_native_zero_shot",
        "finfre_rag_ported_zero_shot",
    ):
        assert rb.blocked(arms[n]) is None
    assert rb.expected_settings(arms["rules_fixed"])["rules_mode"] == "fixed"
    assert rb.expected_settings(arms["ev_none"])["evidence_view"] == "none"
    assert rb.expected_settings(arms["finfre_rag_native_zero_shot"])["exemplars"] == "none"
    assert rb.expected_settings(arms["finfre_rag_native_pool_nat"])["exemplars"] == "retrieved"
    good = rb.Loaded(
        "x",
        (),
        None,
        None,
        None,
        {
            "settings": {
                "evidence_view": ["none"],
                "edge_scorer": ["v2"],
                "rules_mode": ["alert"],
                "exemplars": [None],
            }
        },
    )
    assert rb.settings_mismatch(arms["ev_none"], good) == []
    assert rb.settings_mismatch(
        arms["rules_none"], good
    )  # the chunks say rules_mode alert, the arm demands none
